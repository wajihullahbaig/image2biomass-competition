
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json
import math

# Local Imports
import configs
from configs import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, N_FOLDS,
    BIOMASS_FEAT_WEIGHT, AUX_FEAT_WEIGHT, SPECIES_FEAT_WEIGHT, MONTH_FEAT_WEIGHT, PHYSICS_FEAT_WEIGHT,
    OFFICIAL_WEIGHTS, config_str
)
from common import (
    load_data, get_image_data_transforms_v2, 
    set_seed, calculate_global_weighted_r2,
    upsample_minority_classes
)
from log_and_plots import (
    setup_logging, plot_training_history, 
    log_fold_details, log_upsample_stats
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel

def train_one_epoch(model, loader, optimizer, criterion_huber, criterion_ce, device, epoch):
    model.train()
    
    # Trackers
    total_loss_sum = 0
    bio_loss_sum = 0
    aux_loss_sum = 0
    sp_loss_sum = 0
    mo_loss_sum = 0
    phy_loss_sum = 0
    
    scaler = torch.amp.GradScaler('cuda')
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    for batch in pbar:
        # Move to device
        images = batch['image'].to(device)
        # targets are in KG scale. We need Log1p(target) for training
        targets_kg = batch['targets'].to(device)
        targets_log = torch.log1p(targets_kg) 
        
        aux_feats = batch['aux_feats'].to(device)
        species_ids = batch['species_id'].to(device)
        month_sincos = batch['month_sin_cos'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Forward
            # biomass_out: [Log_C, Log_D, Log_G, Log_T, Log_GDM]
            biomass_out, aux_out, species_logits, month_logits = model(images)
            
            # --- 1. Biomass Loss (Huber in Log Space) ---
            # Preds are already Log Space (Softplus output)
            # Match columns: [Clover, Dead, Green, Total, GDM]
            loss_bio = criterion_huber(biomass_out, targets_log) * BIOMASS_FEAT_WEIGHT
            
            # --- 2. Aux Loss (NDVI, LogHeight, Interaction) ---
            loss_aux = criterion_huber(aux_out, aux_feats) * AUX_FEAT_WEIGHT
            
            # --- 3. Species Loss (CrossEntropy) ---
            loss_sp = criterion_ce(species_logits, species_ids) * SPECIES_FEAT_WEIGHT
            
            # --- 4. Month Loss (Huber on Sin/Cos) ---
            loss_mo = criterion_huber(month_logits, month_sincos) * MONTH_FEAT_WEIGHT
            
            # --- 5. Physics Consistency Loss ---
            # Reconstruct Total/GDM from components in Linear Space and compare
            # biomass_out indices: 0:C, 1:D, 2:G, 3:Total, 4:GDM
            pred_c = torch.expm1(biomass_out[:, 0])
            pred_d = torch.expm1(biomass_out[:, 1])
            pred_g = torch.expm1(biomass_out[:, 2])
            
            # Derived
            derived_total = pred_c + pred_d + pred_g
            derived_gdm = pred_c + pred_g
            
            # Model's direct predictions
            pred_total = torch.expm1(biomass_out[:, 3])
            pred_gdm = torch.expm1(biomass_out[:, 4])
            
            # Loss: Consistency between Derived and Predicted
            loss_phy_total = nn.functional.huber_loss(pred_total, derived_total)
            loss_phy_gdm = nn.functional.huber_loss(pred_gdm, derived_gdm)
            
            loss_phy = (loss_phy_total + loss_phy_gdm) * PHYSICS_FEAT_WEIGHT

            # --- Total Loss ---
            total_loss = loss_bio + loss_aux + loss_sp + loss_mo + loss_phy
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update trackers
        B = images.size(0)
        total_loss_sum += total_loss.item() * B
        bio_loss_sum += loss_bio.item() * B
        aux_loss_sum += loss_aux.item() * B
        sp_loss_sum += loss_sp.item() * B
        mo_loss_sum += loss_mo.item() * B
        phy_loss_sum += loss_phy.item() * B
        
        pbar.set_postfix({'L': total_loss.item()})
        
    N = len(loader.dataset)
    return {
        'train_loss': total_loss_sum / N,
        'train_bio': bio_loss_sum / N,
        'train_aux': aux_loss_sum / N,
        'train_sp': sp_loss_sum / N,
        'train_mo': mo_loss_sum / N,
        'train_phy': phy_loss_sum / N
    }

@torch.no_grad()
def validate(model, loader, criterion_huber, criterion_ce, device):
    model.eval()
    
    total_loss_sum = 0
    bio_loss_sum = 0
    aux_loss_sum = 0
    sp_loss_sum = 0
    mo_loss_sum = 0
    phy_loss_sum = 0
    
    all_preds_log = []
    all_targets_kg = []
    
    for batch in loader:
        images = batch['image'].to(device)
        targets_kg = batch['targets'].to(device) # Raw KG
        targets_log = torch.log1p(targets_kg) 
        
        aux_feats = batch['aux_feats'].to(device)
        species_ids = batch['species_id'].to(device)
        month_sincos = batch['month_sin_cos'].to(device)
        
        # Forward
        biomass_out, aux_out, species_logits, month_logits = model(images)
        
        # Losses
        loss_bio = criterion_huber(biomass_out, targets_log) * BIOMASS_FEAT_WEIGHT
        loss_aux = criterion_huber(aux_out, aux_feats) * AUX_FEAT_WEIGHT
        loss_sp = criterion_ce(species_logits, species_ids) * SPECIES_FEAT_WEIGHT
        loss_mo = criterion_huber(month_logits, month_sincos) * MONTH_FEAT_WEIGHT
        
        # Physics Loss
        pred_c = torch.expm1(biomass_out[:, 0])
        pred_d = torch.expm1(biomass_out[:, 1])
        pred_g = torch.expm1(biomass_out[:, 2])
        derived_total = pred_c + pred_d + pred_g
        derived_gdm = pred_c + pred_g
        pred_total = torch.expm1(biomass_out[:, 3])
        pred_gdm = torch.expm1(biomass_out[:, 4])
        loss_phy = (nn.functional.huber_loss(pred_total, derived_total) + \
                    nn.functional.huber_loss(pred_gdm, derived_gdm)) * PHYSICS_FEAT_WEIGHT

        total_loss = loss_bio + loss_aux + loss_sp + loss_mo + loss_phy
        
        # Track
        B = images.size(0)
        total_loss_sum += total_loss.item() * B
        bio_loss_sum += loss_bio.item() * B
        aux_loss_sum += loss_aux.item() * B
        sp_loss_sum += loss_sp.item() * B
        mo_loss_sum += loss_mo.item() * B
        phy_loss_sum += loss_phy.item() * B
        
        all_preds_log.append(biomass_out.cpu())
        all_targets_kg.append(targets_kg.cpu())
        
    N = len(loader.dataset)
    metrics = {
        'val_loss': total_loss_sum / N,
        'val_bio': bio_loss_sum / N,
        'val_aux': aux_loss_sum / N,
        'val_sp': sp_loss_sum / N,
        'val_mo': mo_loss_sum / N,
        'val_phy': phy_loss_sum / N
    }
    
    # Calculate R2
    preds_log = torch.cat(all_preds_log).numpy()
    preds_kg = np.expm1(preds_log) # Convert back to linear KG for R2 calculation
    targets_kg = torch.cat(all_targets_kg).numpy()
    
    # R2 on KG scale
    r2_score = calculate_global_weighted_r2(targets_kg, preds_kg, OFFICIAL_WEIGHTS)
    metrics['val_r2'] = r2_score
    
    return metrics



def save_metadata(session_dir, species_mapping, target_cols):
    metadata = {
        'species_list': list(species_mapping.keys()),
        'target_cols': target_cols,
        'backbone': configs.BACKBONE,
        'image_height': configs.IMAGE_HEIGHT,
        'image_width': configs.IMAGE_WIDTH,
        'num_species': len(species_mapping),
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

def main(args):
    # Setup
    session_dir = setup_logging(file_name_part="ts_split_train")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())
    # Load Data
    df = load_data(logger)
    
    # Sort by Date for TimeSeriesSplit
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    
    # Global Species Mapping (Ensures consistency across folds)
    species_list = sorted(df['Species'].unique().tolist())
    species_to_id = {s: i for i, s in enumerate(species_list)}
    logger.info(f"Global Species Mapping Created: {len(species_list)} species found.")
    
    # Save Metadata
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    metadata = save_metadata(session_dir, species_to_id, target_cols)
    logger.info(f"Metadata saved to {session_dir}/metadata.json")

    # K-Folder
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    
    best_overall_r2 = -float('inf')
    train_transform, val_transform = get_image_data_transforms_v2()
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        # Split
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Temporal Check
        max_train_date = train_df['Sampling_Date'].max()
        
        # STRICT TEMPORAL SEPARATION: Drop Val samples <= Max Train Date
        original_val_len = len(val_df)
        val_df = val_df[val_df['Sampling_Date'] > max_train_date].reset_index(drop=True)
        dropped_count = original_val_len - len(val_df)
        
        # Log Logic
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} validation samples to enforce strict temporal order (Val > {max_train_date})")
            
        if len(val_df) == 0:
            logger.warning("Validation set is empty after strict temporal filtering! Skipping fold.")
            continue
            
        min_val_date = val_df['Sampling_Date'].min()
        
        # Log Details
        log_fold_details(logger, train_df, val_df)

        # Upsampling (Train Only)

        # Upsampling (Train Only)
        logger.info(f"Train size before upsample: {len(train_df)}")
        train_df_before = train_df.copy() # Capture for logging
        
        train_df = upsample_minority_classes(train_df, 'Species')
        
        # CRITICAL: Re-sort by Date to honor temporal order after upsampling
        train_df = train_df.sort_values('Sampling_Date').reset_index(drop=True)
        logger.info(f"Train size after upsample: {len(train_df)}")
        
        log_upsample_stats(logger, train_df_before, train_df)

        # Log Details - updated one
        log_fold_details(logger, train_df, val_df)
        
        # Datasets
        train_ds = BiomassDataset(train_df, transform=train_transform, species_to_id=species_to_id)
        val_ds = BiomassDataset(val_df, transform=val_transform, species_to_id=species_to_id)
        
        # Shuffle=False to respect temporal order (Curriculum Learning / Streaming)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        model = BiomassUnifiedModel(num_species=len(species_list)).to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # ReduceLROnPlateau: More aggressive now (patience 2, threshold 1e-2)
        # mode='min' monitors val_loss. factor=0.25 slashes LR.
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, threshold=1e-2)
        
        criterion_huber = nn.HuberLoss() # Default delta=1.0 is fine for log-space
        criterion_ce = nn.CrossEntropyLoss()
        
        # History
        history = defaultdict(list)
        best_fold_r2 = -float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Train
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion_huber, criterion_ce, DEVICE, epoch)
            
            # Val
            val_metrics = validate(model, val_loader, criterion_huber, criterion_ce, DEVICE)
            
            # Step Scheduler (based on val_loss)
            scheduler.step(val_metrics['val_loss'])
            
            # Logging
            log_msg = f"Ep {epoch} | T_Loss: {train_metrics['train_loss']:.4f} | V_Loss: {val_metrics['val_loss']:.4f} | V_R2: {val_metrics['val_r2']:.4f}"
            logger.info(log_msg)
            
            # Save History
            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Best Model Logic
            if val_metrics['val_r2'] > best_fold_r2:
                best_fold_r2 = val_metrics['val_r2']
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold+1}.pth"))
                logger.info(f"*** New Best Fold R2: {best_fold_r2:.4f} ***")
                patience_counter = 0
                
                # Global Best
                if best_fold_r2 > best_overall_r2:
                    best_overall_r2 = best_fold_r2
                    torch.save(model.state_dict(), os.path.join(session_dir, "best_model_overall.pth"))
            else:
                patience_counter += 1
                
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early Stopping Triggered")
                break
                
            if args.dry_run and epoch >= 1:
                break
                
            plot_training_history(history, fold+1, session_dir)
            
        if args.dry_run:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Fast run for debugging')
    args = parser.parse_args()
    
    main(args)
