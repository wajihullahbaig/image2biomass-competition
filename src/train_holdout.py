# train_holdout.py
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json

# Local Imports
import configs
from configs import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, TAXONOMY_FEAT_WEIGHT, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, N_FOLDS,
    BIOMASS_FEAT_WEIGHT, AUX_FEAT_WEIGHT, SPECIES_FEAT_WEIGHT, PHYSICS_FEAT_WEIGHT,
    OFFICIAL_WEIGHTS, config_str,
    CORE_SPECIES
)
from common import (
    load_data, get_image_data_transforms, save_batch_images, 
    set_seed, calculate_global_weighted_r2,
    upsample_minority_classes, get_taxonomy_targets
)
from log_and_plots import (
    setup_logging, plot_training_history, 
    log_fold_details
)
from dataset import BiomassDataset, MixupDataset
from models import BiomassUnifiedModel

# -----------------------------------------------------------------------------
# TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_ce, device, epoch, session_dir=None):
    model.train()
    metrics = defaultdict(float)
    scaler = torch.amp.GradScaler('cuda')
    
    all_preds_log = []
    all_targets_g = []
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets_g = batch['targets'].to(device)
        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(device)
        species_vec = batch['species_id'].to(device)
        
        if epoch == 0 and batch_idx < 5 and session_dir:
            save_batch_images(images, "train", batch_idx, session_dir, max_batches_to_save=5)
        
        taxonomy_targets = get_taxonomy_targets(species_vec)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            biomass_out, aux_out, species_logits, taxonomy_logits = model(images)
            
            # Loss Components
            loss_bio = criterion_reg(biomass_out, targets_log) * BIOMASS_FEAT_WEIGHT
            loss_aux = criterion_reg(aux_out, aux_feats) * AUX_FEAT_WEIGHT
            loss_sp = criterion_ce(species_logits, species_vec) * SPECIES_FEAT_WEIGHT
            loss_tax = criterion_ce(taxonomy_logits, taxonomy_targets) * TAXONOMY_FEAT_WEIGHT
            
            # Physics Loss
            pred_c = torch.expm1(biomass_out[:, 0])
            pred_d = torch.expm1(biomass_out[:, 1])
            pred_g = torch.expm1(biomass_out[:, 2])
            derived_total = torch.log1p(pred_c + pred_d + pred_g + 1e-8)
            loss_phy = criterion_reg(biomass_out[:, 3], derived_total) * PHYSICS_FEAT_WEIGHT

            total_loss = loss_bio + loss_aux + loss_sp + loss_tax + loss_phy
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio'] += loss_bio.item() * B
        metrics['train_aux'] += loss_aux.item() * B
        metrics['train_sp']  += loss_sp.item() * B
        metrics['train_tax'] += loss_tax.item() * B  
        metrics['train_phy'] += loss_phy.item() * B

        # Component Losses
        with torch.no_grad():
            metrics['train_loss_c'] += nn.functional.mse_loss(biomass_out[:, 0], targets_log[:, 0]).item() * B
            metrics['train_loss_d'] += nn.functional.mse_loss(biomass_out[:, 1], targets_log[:, 1]).item() * B
            metrics['train_loss_g'] += nn.functional.mse_loss(biomass_out[:, 2], targets_log[:, 2]).item() * B
            metrics['train_loss_t'] += nn.functional.mse_loss(biomass_out[:, 3], targets_log[:, 3]).item() * B
            metrics['train_loss_gdm'] += nn.functional.mse_loss(biomass_out[:, 4], targets_log[:, 4]).item() * B
            
            metrics['train_loss_ndvi'] += nn.functional.mse_loss(aux_out[:, 0], aux_feats[:, 0]).item() * B
            metrics['train_loss_h']    += nn.functional.mse_loss(aux_out[:, 1], aux_feats[:, 1]).item() * B
            metrics['train_loss_int']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B
        
        all_preds_log.append(biomass_out.detach().cpu())
        all_targets_g.append(targets_g.detach().cpu())
        
        pbar.set_postfix({'L': total_loss.item()})
        
    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_g).numpy()
    final_metrics['train_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, OFFICIAL_WEIGHTS)
    
    return final_metrics

@torch.no_grad()
def validate(model, loader, criterion_reg, criterion_ce, device, prefix='val'):
    model.eval()
    metrics = defaultdict(float)
    all_preds_log, all_targets_g = [], []
    
    for batch in loader:
        images = batch['image'].to(device)
        targets_g = batch['targets'].to(device)
        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(device)
        species_vec = batch['species_id'].to(device)
        taxonomy_targets = get_taxonomy_targets(species_vec)
        
        biomass_out, aux_out, species_logits, taxonomy_logits = model(images)
        
        loss_bio = criterion_reg(biomass_out, targets_log) * BIOMASS_FEAT_WEIGHT
        loss_aux = criterion_reg(aux_out, aux_feats) * AUX_FEAT_WEIGHT
        loss_sp = criterion_ce(species_logits, species_vec) * SPECIES_FEAT_WEIGHT
        loss_tax = criterion_ce(taxonomy_logits, taxonomy_targets) * TAXONOMY_FEAT_WEIGHT
        
        pred_c = torch.expm1(biomass_out[:, 0])
        pred_d = torch.expm1(biomass_out[:, 1])
        pred_g = torch.expm1(biomass_out[:, 2])
        derived_total = torch.log1p(pred_c + pred_d + pred_g + 1e-8)
        loss_phy = criterion_reg(biomass_out[:, 3], derived_total) * PHYSICS_FEAT_WEIGHT

        total_loss = loss_bio + loss_aux + loss_sp + loss_tax + loss_phy
        
        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_sp']  += loss_sp.item() * B
        metrics[f'{prefix}_tax'] += loss_tax.item() * B 
        metrics[f'{prefix}_phy'] += loss_phy.item() * B
        
        # Component Losses
        metrics[f'{prefix}_loss_c'] += nn.functional.mse_loss(biomass_out[:, 0], targets_log[:, 0]).item() * B
        metrics[f'{prefix}_loss_d'] += nn.functional.mse_loss(biomass_out[:, 1], targets_log[:, 1]).item() * B
        metrics[f'{prefix}_loss_g'] += nn.functional.mse_loss(biomass_out[:, 2], targets_log[:, 2]).item() * B
        metrics[f'{prefix}_loss_t'] += nn.functional.mse_loss(biomass_out[:, 3], targets_log[:, 3]).item() * B
        metrics[f'{prefix}_loss_gdm'] += nn.functional.mse_loss(biomass_out[:, 4], targets_log[:, 4]).item() * B
        
        metrics[f'{prefix}_loss_ndvi'] += nn.functional.mse_loss(aux_out[:, 0], aux_feats[:, 0]).item() * B
        metrics[f'{prefix}_loss_h']    += nn.functional.mse_loss(aux_out[:, 1], aux_feats[:, 1]).item() * B
        metrics[f'{prefix}_loss_int']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B

        all_preds_log.append(biomass_out.cpu())
        all_targets_g.append(targets_g.cpu())
        
    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_g).numpy()
    final_metrics[f'{prefix}_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, OFFICIAL_WEIGHTS)
    
    return final_metrics

def save_metadata(session_dir, species_list, target_cols):
    metadata = {
        'species_list': species_list,
        'target_cols': target_cols,
        'backbone': configs.BACKBONE,
        'image_height': configs.IMAGE_HEIGHT,
        'image_width': configs.IMAGE_WIDTH,
        'num_species': len(species_list),
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    session_dir = setup_logging(file_name_part="stratified_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())
    
    # 1. Load Data
    df = load_data(logger)
    
    # CRITICAL: Sort by date for strict temporal splitting
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    logger.info("Data sorted by Sampling_Date.")
    
    species_list = CORE_SPECIES
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    save_metadata(session_dir, species_list, target_cols)

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # DATA SPLIT Strategy: 
    # 1. Global Holdout (Last 15% of Data) - STRICT FUTURE
    # 2. Development Set (First 85% of Data) - CV (Stratified Group?) 
    #    User requested StratifiedKFold on Dev Set.
    # -------------------------------------------------------------------------
    total_len = len(df)
    holdout_split_idx = int(total_len * 0.85)
    
    dev_df = df.iloc[:holdout_split_idx].copy()
    global_holdout_df = df.iloc[holdout_split_idx:].copy()
    
    logger.info(f"\n{'='*40}")
    logger.info(f"STRATIFIED HOLDOUT CONFIGURATION")
    logger.info(f"{'='*40}")
    logger.info(f"Total Samples: {total_len}")
    logger.info(f"Development Set (85%): {len(dev_df)} ({dev_df['Sampling_Date'].min().date()} -> {dev_df['Sampling_Date'].max().date()})")
    logger.info(f"Global Holdout (15%): {len(global_holdout_df)} ({global_holdout_df['Sampling_Date'].min().date()} -> {global_holdout_df['Sampling_Date'].max().date()})")
    
    global_holdout_df.to_csv(os.path.join(splits_dir, "global_holdout.csv"), index=False)
    
    # Prepare Data Transforms
    train_transform, val_transform = get_image_data_transforms()
    
    # Holdout Dataset (Constant across folds)
    holdout_ds = BiomassDataset(global_holdout_df, transform=val_transform)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # STRATIFIED SPLIT on Development Set
    # Ensures every fold sees every State + FunctionalGroup combination
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    split_key = dev_df['StratifyKey'] 
    
    best_overall_score = -float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, split_key)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        train_df = dev_df.iloc[train_idx].copy()
        val_df = dev_df.iloc[val_idx].copy()
        
        # Log Temporal Ranges
        logger.info(f"Train: {train_df['Sampling_Date'].min().date()} -> {train_df['Sampling_Date'].max().date()} (n={len(train_df)})")
        logger.info(f"Val:   {val_df['Sampling_Date'].min().date()} -> {val_df['Sampling_Date'].max().date()} (n={len(val_df)})")
        
        log_fold_details(logger, train_df, val_df) 
        
        # Upsampling (Train Only)
        train_df = upsample_minority_classes(train_df, target_col='FunctionalGroup')
        
        # Save Fold Splits
        train_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_val.csv"), index=False)
        
        # Datasets
        train_ds_base = BiomassDataset(train_df, transform=train_transform)        
        train_ds = MixupDataset(train_ds_base, prob=0.10, alpha=0.4)        
        val_ds = BiomassDataset(val_df, transform=val_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        dummy_ds = BiomassDataset(train_df[:1], transform=train_transform)
        n_aux = dummy_ds[0]['aux_feats'].shape[0]        
        model = BiomassUnifiedModel(num_species=len(species_list), num_aux=n_aux).to(DEVICE)       
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-3, min_lr=1e-6)
        
        criterion_reg = nn.MSELoss() 
        criterion_ce = nn.CrossEntropyLoss()
        
        history = defaultdict(list)        
        best_fold_score = -float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # 1. Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_ce, 
                DEVICE, epoch, session_dir=session_dir
            )
            
            # 2. Validate (Temporal Slice)
            val_metrics = validate(model, val_loader, criterion_reg, criterion_ce, DEVICE, prefix='val')
            
            # 3. Holdout (Global Future)
            hol_metrics = validate(model, holdout_loader, criterion_reg, criterion_ce, DEVICE, prefix='holdout')
            
            # 4. Custom Score
            v_r2 = val_metrics['val_r2']
            h_r2 = hol_metrics['holdout_r2']
            
            avg_r2 = (v_r2 + h_r2) / 2
            consistency_penalty = 0.5 * abs(v_r2 - h_r2)
            current_score = avg_r2 - consistency_penalty
            
            # Scheduler Step (Maximize Score)
            scheduler.step(current_score)
            
            log_msg = (f"Ep {epoch} | T_Loss: {train_metrics['train_loss']:.3f} | V_Loss: {val_metrics['val_loss']:.3f} | H_Loss: {hol_metrics['holdout_loss']:.3f} | "
                       f"T_R2: {train_metrics['train_r2']:.4f} | V_R2: {v_r2:.4f} | H_R2: {h_r2:.4f} | "
                       f"Score: {current_score:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
            logger.info(log_msg)
            
            # Store History
            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            for k, v in hol_metrics.items(): history[k].append(v)
            history['score'].append(current_score)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Save Best
            if current_score > best_fold_score:
                best_fold_score = current_score
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold+1}.pth"))
                logger.info(f"*** Fold {fold+1} Best Score: {best_fold_score:.4f} (V:{v_r2:.3f}, H:{h_r2:.3f}) ***")
                patience_counter = 0
                
                if best_fold_score > best_overall_score:
                    best_overall_score = best_fold_score
                    torch.save(model.state_dict(), os.path.join(session_dir, "best_model_overall.pth"))
            else:
                patience_counter += 1
                
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early Stopping Triggered")
                break
                
            plot_training_history(history, fold+1, session_dir)

if __name__ == '__main__':    
    main()
