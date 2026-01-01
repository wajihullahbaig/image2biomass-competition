# train3.py
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    CORE_SPECIES, TAXONOMY_IDXS
)
from common import (
    load_data, get_image_data_transforms, save_batch_images, 
    set_seed, calculate_global_weighted_r2, get_taxonomy_targets,
    upsample_minority_classes
)
from log_and_plots import (
    setup_logging, plot_training_history, 
    log_fold_details, log_upsample_stats
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel

# -----------------------------------------------------------------------------
# TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_ce, device, epoch, fold=None, session_dir=None):
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
        
        if epoch == 0 and fold is not None and session_dir is not None:
            save_batch_images(images, fold, batch_idx, session_dir, max_batches_to_save=10)
        
        taxonomy_targets = get_taxonomy_targets(species_vec)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            biomass_out, aux_out, species_logits, taxonomy_logits = model(images)
            
            loss_bio = criterion_reg(biomass_out, targets_log) * BIOMASS_FEAT_WEIGHT
            loss_aux = criterion_reg(aux_out, aux_feats) * AUX_FEAT_WEIGHT
            loss_sp = criterion_ce(species_logits, species_vec) * SPECIES_FEAT_WEIGHT
            loss_tax = criterion_ce(taxonomy_logits, taxonomy_targets) * TAXONOMY_FEAT_WEIGHT
            
            # Physics Loss (Self-Consistency)
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
        
        all_preds_log.append(biomass_out.detach().cpu())
        all_targets_g.append(targets_g.detach().cpu())
        
        pbar.set_postfix({'L': total_loss.item()})
        
    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    
    # Calculate Train R2
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_g).numpy()
    final_metrics['train_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, OFFICIAL_WEIGHTS)
    
    return final_metrics

@torch.no_grad()
def validate(model, loader, criterion_reg, criterion_ce, device):
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
        metrics['val_loss'] += total_loss.item() * B
        metrics['val_bio'] += loss_bio.item() * B
        metrics['val_aux'] += loss_aux.item() * B
        metrics['val_sp']  += loss_sp.item() * B
        metrics['val_tax'] += loss_tax.item() * B 
        metrics['val_phy'] += loss_phy.item() * B
        
        all_preds_log.append(biomass_out.cpu())
        all_targets_g.append(targets_g.cpu())
        
    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_g).numpy()
    final_metrics['val_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, OFFICIAL_WEIGHTS)
    
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
    # File name part reflects the Month Group strategy
    session_dir = setup_logging(file_name_part="month_group_kfold")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())
    
    # 1. Load Data
    df = load_data(logger)
    
    # 2. Prepare Month and Stratification Columns
    # month extraction (1-12)
    df['Month'] = df['Sampling_Date'].dt.month
    
    # Stratify by Species, but group rare ones to "Rare" for splitter safety
    species_counts = df['Species'].value_counts()
    rare_species = species_counts[species_counts < N_FOLDS].index
    df['StratifySpecies'] = df['Species'].apply(lambda x: 'Rare' if x in rare_species else x)
    
    # Groups: Month (Temporal Robustness)
    # This forces the model to generalize to unseen months/seasons.
    df['Groups'] = df['Month'].astype(str)
    
    species_list = CORE_SPECIES
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    save_metadata(session_dir, species_list, target_cols)

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # STRATIFIED GROUP K-FOLD
    # Ensures each fold has a distinct set of MONTHS.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    best_overall_r2 = -float('inf')
    train_transform, val_transform = get_image_data_transforms()
    
    # Iterate Folds
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['StratifySpecies'], groups=df['Groups'])):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Log which months are in validation for this fold
        val_months = sorted(val_df['Month'].unique())
        logger.info(f"Validation Months for Fold {fold+1}: {val_months}")
        
        log_fold_details(logger, train_df, val_df)

        # Optional Upsampling (Functional Group balance)
        # Note: Upsampling is done AFTER splitting to avoid leakage
        train_df = upsample_minority_classes(train_df, target_col='FunctionalGroup')
        
        # Save Splits
        train_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_val.csv"), index=False)
        
        # Datasets
        train_ds = BiomassDataset(train_df, transform=train_transform)        
        val_ds = BiomassDataset(val_df, transform=val_transform)
        
        # Shuffle=True in DataLoader since temporal sequence is handled by Month splits
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        dummy_ds = BiomassDataset(train_df[:1], transform=train_transform)
        n_aux = dummy_ds[0]['aux_feats'].shape[0]        
        model = BiomassUnifiedModel(num_species=len(species_list), num_aux=n_aux).to(DEVICE)       
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=5, threshold=1e-3, min_lr=1e-5)
        
        criterion_reg = nn.MSELoss() 
        criterion_ce = nn.CrossEntropyLoss()
        
        history = defaultdict(list)        
        best_fold_r2 = -float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_ce, 
                DEVICE, epoch, fold=fold+1, session_dir=session_dir
            )
            val_metrics = validate(model, val_loader, criterion_reg, criterion_ce, DEVICE)

            scheduler.step(val_metrics['val_loss'])
            
            log_msg = (f"Ep {epoch} | T_Loss: {train_metrics['train_loss']:.2f} | T_R2: {train_metrics['train_r2']:.4f} | "
                       f"V_Loss: {val_metrics['val_loss']:.2f} | V_R2: {val_metrics['val_r2']:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.1e}")
            logger.info(log_msg)
            
            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            if val_metrics['val_r2'] > best_fold_r2:
                best_fold_r2 = val_metrics['val_r2']
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold+1}.pth"))
                logger.info(f"*** New Best Fold R2: {best_fold_r2:.4f} ***")
                patience_counter = 0
                
                if best_fold_r2 > best_overall_r2:
                    best_overall_r2 = best_fold_r2
                    torch.save(model.state_dict(), os.path.join(session_dir, "best_model_overall.pth"))
            else:
                patience_counter += 1
                
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early Stopping Triggered")
                break
                
            plot_training_history(history, fold+1, session_dir)

if __name__ == '__main__':    
    main()
