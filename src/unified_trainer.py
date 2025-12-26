# unified_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from configs import *
from common import (
    check_group_leakage, load_data, setup_logging, set_seed, get_image_data_transforms_v2,
    calculate_global_weighted_r2, enforce_physical_constraints,
    plot_training_history, apply_tta, upsample_minority_classes, EWC
)
from dataset import BiomassDataset, MosaicDataset
from models import BiomassUnifiedModel, initialize_weights

def train_one_epoch(model, loader, optimizer, criterion_biomass, criterion_aux, 
                    criterion_species, criterion_month, device, ewc=None):
    model.train()
    running = {'loss': 0, 'bio': 0, 'aux': 0, 'sp': 0, 'mo': 0, 'phy': 0}
    weights = COL_WEIGHTS_TENSOR.view(1, -1)
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device) # Linear KG
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        # LOG-SPACE TRANSFORMATION
        # We train the model to predict Log(1 + Mass_KG)
        log_targets = torch.log1p(targets)
        
        optimizer.zero_grad()
        
        # Model outputs Log-Space predictions
        biomass_log_pred, aux_pred, species_logits, month_logits = model(images)
        
        # 1. Biomass Loss (Log Space)
        # Compare Log Preds vs Log Targets
        raw_bio_loss = criterion_biomass(biomass_log_pred, log_targets)
        loss_bio = (raw_bio_loss * weights).sum() / images.size(0)
        
        # 2. Physics Consistency Loss (Linear Space Consistency)
        # Ensure that exp(LogTotal) == exp(LogC) + exp(LogD) + exp(LogG)
        log_c = biomass_log_pred[:, 0]
        log_d = biomass_log_pred[:, 1]
        log_g = biomass_log_pred[:, 2]
        log_t = biomass_log_pred[:, 3]
        
        # Reconstruct Linear Components
        lin_c = torch.expm1(log_c)
        lin_d = torch.expm1(log_d)
        lin_g = torch.expm1(log_g)
        lin_t_direct = torch.expm1(log_t)
        lin_t_sum = lin_c + lin_d + lin_g
        
        # Penalty: Difference between Predict_Total and Sum_Components
        loss_phy = nn.functional.smooth_l1_loss(lin_t_direct, lin_t_sum.detach()) 
        # Note: Detached sum prevents gradient fighting, or allow flow? 
        # Let's flow gradients to both to encourage mutual agreement.
        loss_phy = nn.functional.smooth_l1_loss(lin_t_direct, lin_t_sum)
        
        # 3. Aux Losses
        loss_aux = criterion_aux(aux_pred, aux_feats)
        loss_sp = criterion_species(species_logits, species_id)
        loss_mo = criterion_month(month_logits, month_target)
        
        # 4. Total
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + 
                 loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + 
                 loss_mo * MONTH_FEAT_WEIGHT + 
                 loss_phy * PHYSICS_FEAT_WEIGHT)
        
        if ewc is not None:
            total += ewc.penalty(model)
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running['loss'] += total.item()
        running['bio'] += loss_bio.item()
        running['aux'] += loss_aux.item()
        running['sp'] += loss_sp.item()
        running['mo'] += loss_mo.item()
        running['phy'] += loss_phy.item()
        
        pbar.set_postfix({'L': f"{total.item():.2f}", 'Bio': f"{loss_bio.item():.4f}"})
    
    n = len(loader)
    return {k: v/n for k, v in running.items()}

@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, criterion_species, criterion_month, device):
    model.eval()
    running = {'loss': 0, 'bio': 0, 'aux': 0, 'sp': 0, 'mo': 0, 'phy': 0}
    all_targets, all_preds = [], []
    weights = COL_WEIGHTS_TENSOR.view(1, -1)
    
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device) # Linear KG
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        # Log Targets for Loss Calculation
        log_targets = torch.log1p(targets)
        
        if USE_TTA:
            biomass_log_pred, aux_pred, species_logits, month_logits = apply_tta(model, images, device)
        else:
            biomass_log_pred, aux_pred, species_logits, month_logits = model(images)
        
        # Loss in LOG SPACE
        raw_bio_loss = criterion_biomass(biomass_log_pred, log_targets)
        loss_bio = (raw_bio_loss * weights).sum() / images.size(0)
        
        # Physics Consistency (Validation visibility)
        log_c, log_d, log_g, log_t = biomass_log_pred[:, 0], biomass_log_pred[:, 1], biomass_log_pred[:, 2], biomass_log_pred[:, 3]
        loss_phy = nn.functional.smooth_l1_loss(torch.expm1(log_t), torch.expm1(log_c) + torch.expm1(log_d) + torch.expm1(log_g))

        loss_aux = criterion_aux(aux_pred, aux_feats)
        loss_sp = criterion_species(species_logits, species_id)
        loss_mo = criterion_month(month_logits, month_target)
        
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + 
                 loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + 
                 loss_mo * MONTH_FEAT_WEIGHT + 
                 loss_phy * PHYSICS_FEAT_WEIGHT)
        
        running['loss'] += total.item()
        running['bio'] += loss_bio.item()
        running['aux'] += loss_aux.item()
        running['sp'] += loss_sp.item()
        running['mo'] += loss_mo.item()
        running['phy'] += loss_phy.item()
        
        # Inverse to Linear for R2 Calculation
        pred_linear = torch.expm1(biomass_log_pred)
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(pred_linear.cpu().numpy())
    
    targets_real = np.concatenate(all_targets)
    preds_real = np.concatenate(all_preds)
    
    # Enforce constraints (Linear Space)
    preds_real = enforce_physical_constraints(preds_real)
    
    # R2 on Linear Data
    r2 = calculate_global_weighted_r2(targets_real, preds_real, OFFICIAL_WEIGHTS)
    
    n = len(loader)
    return {
        'loss': running['loss']/n, 
        'bio': running['bio']/n, 
        'aux': running['aux']/n, 'sp': running['sp']/n, 'mo': running['mo']/n,
        'phy': running['phy']/n,
        'r2': r2, 'r2_display': max(r2, -2.0)
    }

def run_training():
    session_dir = setup_logging(file_name_part="Unified_LogSpace_Mosaic")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())

    logger.info("Loading Data...")
    df = load_data(logger)
    
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    df['week_period'] = df['Sampling_Date'].dt.to_period('W')
    unique_weeks = sorted(df['week_period'].unique())
    logger.info(f"Total Weeks Found: {len(unique_weeks)}")
    
    # Create Global Species Map to ensure consistency across folds
    species_list = sorted(df['Species'].unique().tolist())
    global_species_map = {s: i for i, s in enumerate(species_list)}
    logger.info(f"Global Species Map: {len(global_species_map)} species")
    
    model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=len(species_list)).to(DEVICE)
    initialize_weights(model) 
    if FREEZE_BACKBONE: model.freeze_backbone(BACKBONE_FREEZE_FRACTION)
    
    # Losses (Log Space -> Huber is good)
    criterion_bio = nn.HuberLoss(delta=0.5, reduction='none') 
    criterion_aux = nn.HuberLoss(delta=1.0, reduction='mean')
    criterion_sp = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
    criterion_mo = nn.HuberLoss(delta=1.0, reduction='mean')

    train_tf, val_tf = get_image_data_transforms_v2()
    ewc = None
    
    start_idx = max(INITIAL_HISTORY_WEEKS, 1)
    stride = HOLDOUT_WEEKS
    total_weeks = len(unique_weeks)
    
    for fold_idx in range(start_idx, total_weeks - HOLDOUT_WEEKS + 1, stride):
        current_holdout_weeks = unique_weeks[fold_idx : fold_idx + stride]
        history_weeks = unique_weeks[:fold_idx]
        
        logger.info(f"\n{'='*20} WALK-FORWARD STEP {fold_idx}/{total_weeks} {'='*20}")
        
        df_history = df[df['week_period'].isin(history_weeks)].copy()
        df_holdout = df[df['week_period'].isin(current_holdout_weeks)].copy()
        
        train_dfs, val_dfs = [], []
        
        for species_id in df_history['Species'].unique():
            species_df = df_history[df_history['Species'] == species_id]
            n = len(species_df)
            if n < 2:
                train_dfs.append(species_df)
                continue
            split_idx = int(n * 0.8)
            train_dfs.append(species_df.iloc[:split_idx])
            val_dfs.append(species_df.iloc[split_idx:])
            
        df_train = pd.concat(train_dfs).sort_values('Sampling_Date') if train_dfs else pd.DataFrame(columns=df.columns)
        df_val = pd.concat(val_dfs).sort_values('Sampling_Date') if val_dfs else pd.DataFrame(columns=df.columns)
        
        if len(df_train) < 2 or len(df_val) < 1: continue

        # Upsampling with detailed logging
        counts_before = df_train['Species'].value_counts()
        df_train = upsample_minority_classes(df_train, 'Species')
        counts_after = df_train['Species'].value_counts()
        
        upsampled_info = []
        for sp in counts_before.index:
            diff = counts_after.get(sp, 0) - counts_before[sp]
            if diff > 0:
                upsampled_info.append(f"{sp}:+{diff}")
        
        logger.info(f"Upsampling Report for Fold {fold_idx}:")
        logger.info(f"  Total Original: {counts_before.sum()} | Total New: {len(df_train)} (Added {len(df_train) - counts_before.sum()})")
        logger.info(f"  Species Upsampled: {', '.join(upsampled_info)}")
        logger.info(f"  Final Distribution: {counts_after.to_dict()}")

        # DATASETS
        # Train: Mosaic Dataset Wrapping BiomassDataset
        train_ds_base = BiomassDataset(df_train, transform=train_tf, species_to_id=global_species_map)
        train_ds = MosaicDataset(train_ds_base, prob=0.6, image_size=IMAGE_SIZE) # 60% Mosaic
        
        val_ds = BiomassDataset(df_val, transform=val_tf, species_to_id=global_species_map)
        holdout_ds = BiomassDataset(df_holdout, transform=val_tf, species_to_id=global_species_map)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=PATIENCE)
        
        best_score = -float('inf')
        early_stop_counter = 0

        history = {
            'train_loss': [], 'val_loss': [], 'ind_loss': [],
            'loss_biomass': [], 'ind_loss_biomass': [], 'val_loss_biomass': [],
            'loss_aux': [], 'ind_loss_aux': [], 'val_loss_aux': [],
            'loss_species': [], 'ind_loss_species': [], 'val_loss_species': [],
            'loss_month': [], 'ind_loss_month': [], 'val_loss_month': [],
            'loss_physics': [], 'ind_loss_physics': [], 'val_loss_physics': [],
            'val_r2': [], 'holdout_r2': []
        }

        for epoch in range(EPOCHS):
            t_m = train_one_epoch(model, train_loader, optimizer, criterion_bio, 
                                  criterion_aux, criterion_sp, criterion_mo, DEVICE, ewc)
            v_m = validate(model, val_loader, criterion_bio, criterion_aux, criterion_sp, criterion_mo, DEVICE)
            h_m = validate(model, holdout_loader, criterion_bio, criterion_aux, criterion_sp, criterion_mo, DEVICE)
            
            # Penalized Scoring Formula
            avg_r2 = (v_m['r2'] + h_m['r2']) / 2
            consistency_penalty = 0.5 * abs(v_m['r2'] - h_m['r2'])
            current_score = avg_r2 - consistency_penalty
            
            scheduler.step(current_score) # Step on Custom Score
            
            # --- Store Metrics ---
            # Total Losses
            history['train_loss'].append(t_m['loss'])
            history['val_loss'].append(v_m['loss'])
            history['ind_loss'].append(h_m['loss'])
            
            # Component Losses - Train
            history['loss_biomass'].append(t_m['bio'])
            history['loss_aux'].append(t_m['aux'])
            history['loss_species'].append(t_m['sp'])
            history['loss_month'].append(t_m['mo'])
            history['loss_physics'].append(t_m['phy'])
            
            # Component Losses - Validation
            history['val_loss_biomass'].append(v_m['bio'])
            history['val_loss_aux'].append(v_m['aux'])
            history['val_loss_species'].append(v_m['sp'])
            history['val_loss_month'].append(v_m['mo'])
            history['val_loss_physics'].append(v_m['phy'])
            
            # Component Losses - Holdout
            history['ind_loss_biomass'].append(h_m['bio'])
            history['ind_loss_aux'].append(h_m['aux'])
            history['ind_loss_species'].append(h_m['sp'])
            history['ind_loss_month'].append(h_m['mo'])
            history['ind_loss_physics'].append(h_m['phy'])
            
            # R2
            history['val_r2'].append(v_m['r2_display'])
            history['holdout_r2'].append(h_m['r2_display'])

            logger.info(f"[Epoch {epoch+1}]")
            logger.info(f"  Train: L={t_m['loss']:.3f} (Bio={t_m['bio']:.4f}, Aux={t_m['aux']:.4f}, Sp={t_m['sp']:.3f}, Mo={t_m['mo']:.3f}, Phy={t_m['phy']:.4f})")
            logger.info(f"  Valid: L={v_m['loss']:.3f} (Bio={v_m['bio']:.4f}, Aux={v_m['aux']:.4f}, Sp={v_m['sp']:.3f}, Mo={v_m['mo']:.3f}, Phy={v_m['phy']:.4f}), R2={v_m['r2']:.4f}")
            logger.info(f"  Hold:  L={h_m['loss']:.3f} (Bio={h_m['bio']:.4f}, Aux={h_m['aux']:.4f}, Sp={h_m['sp']:.3f}, Mo={h_m['mo']:.3f}, Phy={h_m['phy']:.4f}), R2={h_m['r2']:.4f}")
            logger.info(f"  Score: {current_score:.4f} (Avg: {avg_r2:.3f}, Pen: {consistency_penalty:.3f})")

            if current_score > best_score:
                best_score = current_score
                early_stop_counter = 0
                logger.info(f"   >> New Best Score! Saving for fold {fold_idx} with score {current_score:.4f}")
                logger.info(f" validation score: {v_m['r2']:.4f}, holdout score: {h_m['r2']:.4f}")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_fold_{fold_idx}.pth"))
            else:
                early_stop_counter += 1
            
            plot_training_history(history, fold_idx, session_dir)
            
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping.")
                break
            
        # EWC Update
        if os.path.exists(os.path.join(session_dir, f"best_fold_{fold_idx}.pth")):
            model.load_state_dict(torch.load(os.path.join(session_dir, f"best_fold_{fold_idx}.pth"), weights_only=True))
        ewc = EWC(model, train_loader, DEVICE, importance=EWC_IMPORTANCE)

if __name__ == "__main__":
    run_training()
