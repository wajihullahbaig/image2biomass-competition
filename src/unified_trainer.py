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
    check_group_leakage, load_data, setup_logging, set_seed, get_image_data_transforms_v1,
    calculate_global_weighted_r2, enforce_physical_constraints,
    plot_training_history, apply_tta, upsample_minority_classes, EWC
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel, initialize_weights

def train_one_epoch(model, loader, optimizer, criterion_biomass, criterion_aux, 
                    criterion_species, criterion_month, device, ewc=None):
    model.train()
    running = {'loss': 0, 'bio': 0, 'aux': 0, 'sp': 0, 'mo': 0}
    weights = COL_WEIGHTS_TENSOR.view(1, -1)
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        optimizer.zero_grad()
        
        biomass_pred, aux_pred, species_logits, month_logits = model(images)
        
        # 1. Biomass Loss
        raw_bio_loss = criterion_biomass(biomass_pred, targets)
        loss_bio = (raw_bio_loss * weights).sum() / images.size(0)
        
        # 2. Aux Losses
        loss_aux = criterion_aux(aux_pred, aux_feats)
        loss_sp = criterion_species(species_logits, species_id)
        loss_mo = criterion_month(month_logits, month_target)
        
        # 3. Total
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + 
                 loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + 
                 loss_mo * MONTH_FEAT_WEIGHT)
        
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
        
        pbar.set_postfix({'L': f"{total.item():.2f}", 'Bio': f"{loss_bio.item():.4f}"})
    
    n = len(loader)
    return {k: v/n for k, v in running.items()}

@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, criterion_species, criterion_month, device):
    model.eval()
    running = {'loss': 0, 'bio': 0, 'aux': 0, 'sp': 0, 'mo': 0}
    all_targets, all_preds = [], []
    weights = COL_WEIGHTS_TENSOR.view(1, -1)
    
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        if USE_TTA:
            biomass_pred, aux_pred, species_logits, month_logits = apply_tta(model, images, device, n_passes=5)
        else:
            biomass_pred, aux_pred, species_logits, month_logits = model(images)
        
        raw_bio_loss = criterion_biomass(biomass_pred, targets)
        loss_bio = (raw_bio_loss * weights).sum() / images.size(0)
        
        loss_aux = criterion_aux(aux_pred, aux_feats)
        loss_sp = criterion_species(species_logits, species_id)
        loss_mo = criterion_month(month_logits, month_target)
        
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + 
                 loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + 
                 loss_mo * MONTH_FEAT_WEIGHT)
        
        running['loss'] += total.item()
        running['bio'] += loss_bio.item()
        running['aux'] += loss_aux.item()
        running['sp'] += loss_sp.item()
        running['mo'] += loss_mo.item()
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(biomass_pred.cpu().numpy())
    
    targets_real = np.concatenate(all_targets)
    preds_real = enforce_physical_constraints(np.concatenate(all_preds))
    r2 = calculate_global_weighted_r2(targets_real, preds_real, OFFICIAL_WEIGHTS)
    
    n = len(loader)
    return {
        'loss': running['loss']/n, 
        'bio': running['bio']/n, 
        'aux': running['aux']/n, 'sp': running['sp']/n, 'mo': running['mo']/n,
        'r2': r2, 'r2_display': max(r2, -2.0)
    }

def run_training():
    session_dir = setup_logging(file_name_part="Unified_WalkForward")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())

    logger.info("Loading Data...")
    df = load_data(logger)
    
    # 1. Strictly Sort Data by Date
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    # 2. Assign Weeks (Periods)
    df['week_period'] = df['Sampling_Date'].dt.to_period('W')
    unique_weeks = sorted(df['week_period'].unique())
    logger.info(f"Total Weeks Found: {len(unique_weeks)}")
    
    # 3. Initialize Model (Continuous Learning)
    # The model persists across folds!
    model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=df['Species'].nunique()).to(DEVICE)
    initialize_weights(model) 
    if FREEZE_BACKBONE: model.freeze_backbone(BACKBONE_FREEZE_FRACTION)
    
    criterion_bio = nn.HuberLoss(delta=0.2, reduction='none') 
    criterion_aux = nn.HuberLoss(delta=1.0)
    criterion_sp = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_mo = nn.HuberLoss(delta=1.0)

    train_tf, val_tf = get_image_data_transforms_v1()
    ewc = None
    
    # 4. WALK FORWARD LOOP
    # Start from index 1 (Week 1) so we have Week 0 as history
    for fold_idx in range(1, len(unique_weeks)):
        current_holdout_week = unique_weeks[fold_idx]
        history_weeks = unique_weeks[:fold_idx] # All weeks BEFORE holdout
        
        logger.info(f"\n{'='*20} WALK-FORWARD STEP {fold_idx}/{len(unique_weeks)-1} {'='*20}")
        
        # 5. Create History and Holdout Dataframes
        df_history = df[df['week_period'].isin(history_weeks)].copy()
        df_holdout = df[df['week_period'] == current_holdout_week].copy()
        
        # 6. Temporal Split of History (Train/Val)
        # We take the LAST 20% of history as the Validation set (most recent past)
        # The first 80% is the Training set (distant past)
        split_point = int(len(df_history) * 0.8)
        
        # If history is tiny (e.g. week 0 is small), ensure at least some train/val
        if split_point < 2: split_point = len(df_history) - 1
            
        df_train = df_history.iloc[:split_point].copy()
        df_val = df_history.iloc[split_point:].copy()
        
        # Logging to confirm logic
        logger.info(f"  Holdout Week: {current_holdout_week} ({len(df_holdout)} samples)")
        logger.info(f"  History Range: {history_weeks[0]} -> {history_weeks[-1]}")
        logger.info(f"  Train Split (Earlier): {len(df_train)} samples")
        logger.info(f"  Val Split   (Recent):  {len(df_val)} samples")
        #
        
        if len(df_train) < 2 or len(df_val) < 1 or len(df_holdout) < 1:
            logger.warning("  Skipping step due to insufficient samples.")
            continue

        # Upsample only the Training portion
        
        logger.info(f"Before upsampling: {df_train['Species'].value_counts()}")        
        df_train = upsample_minority_classes(df_train, 'Species', logger)
        
        logger.info(f"After upsampling: {df_train['Species'].value_counts()}")
        logger.info(f"Heldout Species: {df_holdout['Species'].value_counts()}")

        # Loaders
        train_loader = DataLoader(BiomassDataset(df_train, transform=train_tf), 
                                  batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        val_loader = DataLoader(BiomassDataset(df_val, transform=val_tf), 
                                batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(BiomassDataset(df_holdout, transform=val_tf), 
                                    batch_size=BATCH_SIZE, shuffle=False)

        # Reset Optimizer for new step (but Model weights persist)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=PATIENCE)
        
        best_r2 = -float('inf')
        
        # Metrics History
        history = {
            'train_loss': [], 'val_loss': [], 'ind_loss': [],
            'loss_biomass': [], 'val_loss_biomass': [], 'ind_loss_biomass': [],
            'loss_aux': [], 'val_loss_aux': [], 'ind_loss_aux': [],
            'loss_species': [], 'val_loss_species': [], 'ind_loss_species': [],
            'loss_month': [], 'val_loss_month': [], 'ind_loss_month': [],
            'val_r2': [], 'holdout_r2': []
        }

        for epoch in range(EPOCHS):
            # Train
            t_m = train_one_epoch(model, train_loader, optimizer, criterion_bio, 
                                  criterion_aux, criterion_sp, criterion_mo, DEVICE, ewc)
            # Val (Recent History)
            v_m = validate(model, val_loader, criterion_bio, criterion_aux, criterion_sp, criterion_mo, DEVICE)
            # Test (Future Holdout)
            h_m = validate(model, holdout_loader, criterion_bio, criterion_aux, criterion_sp, criterion_mo, DEVICE)
            
            scheduler.step(v_m['r2'])

            # Store Logs
            history['train_loss'].append(t_m['loss'])
            history['loss_biomass'].append(t_m['bio'])
            history['loss_aux'].append(t_m['aux'])
            history['loss_species'].append(t_m['sp'])
            history['loss_month'].append(t_m['mo'])

            history['val_loss'].append(v_m['loss'])
            history['val_loss_biomass'].append(v_m['bio'])
            history['val_loss_aux'].append(v_m['aux'])
            history['val_loss_species'].append(v_m['sp'])
            history['val_loss_month'].append(v_m['mo'])
            
            history['ind_loss'].append(h_m['loss'])
            history['ind_loss_biomass'].append(h_m['bio'])
            history['ind_loss_aux'].append(h_m['aux'])
            history['ind_loss_species'].append(h_m['sp'])
            history['ind_loss_month'].append(h_m['mo'])
            
            history['val_r2'].append(v_m['r2_display'])
            history['holdout_r2'].append(h_m['r2_display'])

            # Detailed componentwise logging of losses
            logger.info(
                f"  [Epoch {epoch+1}/{EPOCHS}] "
                f"Train Loss: {t_m['loss']:.4f} (Bio: {t_m['bio']:.4f}, Aux: {t_m['aux']:.4f}, Sp: {t_m['sp']:.4f}, Mo: {t_m['mo']:.4f}) | "
                f"Val Loss: {v_m['loss']:.4f} (Bio: {v_m['bio']:.4f}, Aux: {v_m['aux']:.4f}, Sp: {v_m['sp']:.4f}, Mo: {v_m['mo']:.4f}, R2: {v_m['r2_display']:.4f}) | "
                f"Holdout Loss: {h_m['loss']:.4f} (Bio: {h_m['bio']:.4f}, Aux: {h_m['aux']:.4f}, Sp: {h_m['sp']:.4f}, Mo: {h_m['mo']:.4f}, R2: {h_m['r2_display']:.4f})"
            )


            if v_m['r2'] > best_r2:
                best_r2 = v_m['r2']
                logger.info(f">> New Best R2: {best_r2:.4f} -vs- Holdout R2: {h_m['r2_display']:.4f}")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_fold_{fold_idx}.pth"))
            
            plot_training_history(history, fold_idx, session_dir)
            
        # --- EWC UPDATE ---
        # Load best weights from this step to calculate Fisher Information for the NEXT step
        model.load_state_dict(torch.load(os.path.join(session_dir, f"best_fold_{fold_idx}.pth"), weights_only=True))
        # Use a subset of history for EWC to save time, or full train_loader
        ewc = EWC(model, train_loader, DEVICE, importance=EWC_IMPORTANCE)
        logger.info("EWC Constraints updated for next fold.")

if __name__ == "__main__":
    run_training()