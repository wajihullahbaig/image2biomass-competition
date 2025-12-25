# unified_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
import json
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
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        biomass_pred, aux_pred, species_logits, month_logits = model(images)
        
        weights = COL_WEIGHTS_TENSOR.view(1, -1)
        w_sum = weights.sum()
        # Per-sample weighted loss
        weighted_loss = nn.functional.huber_loss(biomass_pred, targets, reduction='none') * weights
        loss_bio = (weighted_loss.sum(dim=1) / w_sum).mean()
        
        loss_aux = criterion_aux(aux_pred, aux_feats).mean()
        loss_sp = criterion_species(species_logits, species_id) / 10.0
        loss_mo = criterion_month(month_logits, month_target).mean()
        
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + loss_mo * MONTH_FEAT_WEIGHT)
        
        if ewc is not None:
            total += ewc.penalty(model)
        
        total = total / ACCUMULATION_STEPS
        total.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running['loss'] += total.item() * ACCUMULATION_STEPS
        running['bio'] += loss_bio.item()
        running['aux'] += loss_aux.item()
        running['sp'] += loss_sp.item()
        running['mo'] += loss_mo.item()
        
        pbar.set_postfix({'Bio': f"{loss_bio.item():.4f}", 'Sp': f"{loss_sp.item():.4f}"})
    
    n = len(loader)
    return {k: v/n for k, v in running.items()}

def log_dataset_stats(df, logger, title="Dataset Stats"):
    logger.info(f"\n--- {title} ---")
    logger.info(f"Samples: {len(df)}, Species: {df['Species'].nunique()}, States: {df['State'].nunique()}")
    if 'Sampling_Date' in df.columns:
        months = pd.to_datetime(df['Sampling_Date']).dt.month
        logger.info(f"Months: {sorted(months.unique())}")
    for s, c in df['Species'].value_counts().items():
        logger.info(f"  {s}: {c}")

@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, criterion_species, criterion_month, device):
    model.eval()
    running = {'loss': 0, 'bio': 0, 'aux': 0, 'sp': 0, 'mo': 0}
    all_targets, all_preds = [], []
    
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
        
        weights = COL_WEIGHTS_TENSOR.view(1, -1)
        w_sum = weights.sum()
        
        weighted_loss = nn.functional.huber_loss(biomass_pred, targets, reduction='none') * weights
        loss_bio = (weighted_loss.sum(dim=1) / w_sum).mean()
        
        loss_aux = criterion_aux(aux_pred, aux_feats).mean()
        loss_sp = criterion_species(species_logits, species_id) / 10.0
        loss_mo = criterion_month(month_logits, month_target).mean()
        
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + loss_mo * MONTH_FEAT_WEIGHT)
        
        running['loss'] += total.item()
        running['bio'] += loss_bio.item()
        running['aux'] += loss_aux.item()
        running['sp'] += loss_sp.item()
        running['mo'] += loss_mo.item()
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(biomass_pred.cpu().numpy())
    
    # Already in KG space, no need for expm1
    targets_real = np.concatenate(all_targets)
    preds_real = enforce_physical_constraints(np.concatenate(all_preds))
    
    # Official metric expects GRAMS? 
    # If calculate_global_weighted_r2 handles scale (it's R2, so it should be fine), but weights are relative.
    # However, enforce_physical_constraints works on what it's given. It sums components.
    # If we trained in kg, preds are kg.
    
    r2 = calculate_global_weighted_r2(targets_real, preds_real, OFFICIAL_WEIGHTS)
    
    n = len(loader)
    result = {k: v/n for k, v in running.items()}
    result['r2'] = r2
    return result

def species_stratified_split(df, val_ratio=0.2):
    train_dfs, val_dfs = [], []
    for species in df['Species'].unique():
        species_df = df[df['Species'] == species].copy()
        species_df = species_df.sort_values('Sampling_Date')
        split_idx = max(1, int(len(species_df) * (1 - val_ratio)))
        train_dfs.append(species_df.iloc[:split_idx])
        val_dfs.append(species_df.iloc[split_idx:])
    train_df = pd.concat(train_dfs).sort_values('Sampling_Date').reset_index(drop=True)
    val_df = pd.concat(val_dfs).sort_values('Sampling_Date').reset_index(drop=True)
    return train_df, val_df

def run_training():
    session_dir = setup_logging(file_name_part="Unified_Trainer")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())

    logger.info("Loading Data...")
    df = load_data(logger)
    
    species_list = sorted(df['Species'].unique().tolist())
    metadata = {
        'species_list': species_list,
        'species_to_id': {s: i for i, s in enumerate(species_list)},
        'target_cols': TARGET_COLS,
        'aux_cols': ['Pre_GSHH_NDVI', 'Height_Ave_cm_log'],
        'official_weights': OFFICIAL_WEIGHTS,
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    df['year_week'] = df['Sampling_Date'].dt.to_period('W')
    unique_weeks = sorted(df['year_week'].unique())
    
    logger.info(f"Weeks: {len(unique_weeks)}, Total Samples: {len(df)}")
    
    train_transform, val_transform = get_image_data_transforms_v1()
    
    model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=len(species_list)).to(DEVICE)
    initialize_weights(model)
    if FREEZE_BACKBONE:
        model.freeze_backbone(freeze_fraction=BACKBONE_FREEZE_FRACTION)
    
    criterion_biomass = nn.HuberLoss(delta=1.0)
    criterion_aux = nn.HuberLoss(delta=1.0)
    criterion_species = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_month = nn.HuberLoss(delta=1.0)

    fold_results = []
    holdout_results = []
    ewc = None

    for fold in range(1, len(unique_weeks)):
        history_weeks = unique_weeks[:fold]
        holdout_week = unique_weeks[fold]
        
        logger.info(f"\n{'='*20} Fold {fold}/{len(unique_weeks)-1} {'='*20}")
        logger.info(f"Train: {history_weeks[0]} to {history_weeks[-1]}, Holdout: {holdout_week}")

        df_history = df[df['year_week'].isin(history_weeks)].copy()
        df_holdout = df[df['year_week'] == holdout_week].copy()
        
        if len(df_holdout) == 0:
            logger.warning(f"Skipping - no holdout samples")
            continue

        train_df, val_df = species_stratified_split(df_history, val_ratio=0.2)
        
        if len(train_df) < 5 or len(val_df) < 2:
            logger.warning(f"Skipping - insufficient data (train={len(train_df)}, val={len(val_df)})")
            continue

        log_dataset_stats(train_df, logger, f"Fold {fold} Train (pre-upsample)")
        train_df = upsample_minority_classes(train_df, 'Species', logger)
        log_dataset_stats(train_df, logger, f"Fold {fold} Train (post-upsample)")
        log_dataset_stats(val_df, logger, f"Fold {fold} Val")
        log_dataset_stats(df_holdout, logger, f"Fold {fold} Holdout")

        train_ds = BiomassDataset(train_df, transform=train_transform, species_to_id=metadata['species_to_id'])
        val_ds = BiomassDataset(val_df, transform=val_transform, species_to_id=metadata['species_to_id'])
        holdout_ds = BiomassDataset(df_holdout, transform=val_transform, species_to_id=metadata['species_to_id'])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=len(train_ds) > BATCH_SIZE)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

        best_val_r2 = -float('inf')
        best_holdout_r2 = -float('inf')
        epochs_no_improve = 0
        
        history = {
            'train_loss': [], 'val_loss': [], 'ind_loss': [], 'val_r2': [], 'holdout_r2': [],
            'loss_biomass': [], 'loss_aux': [], 'loss_species': [], 'loss_month': [],
            'val_loss_biomass': [], 'val_loss_aux': [], 'val_loss_species': [], 'val_loss_month': [],
            'ind_loss_biomass': [], 'ind_loss_aux': [], 'ind_loss_species': [], 'ind_loss_month': []
        }

        for epoch in range(EPOCHS):
            train_m = train_one_epoch(model, train_loader, optimizer, criterion_biomass, 
                                       criterion_aux, criterion_species, criterion_month, DEVICE, ewc)
            scheduler.step()
            
            val_m = validate(model, val_loader, criterion_biomass, criterion_aux, 
                             criterion_species, criterion_month, DEVICE)
            hold_m = validate(model, holdout_loader, criterion_biomass, criterion_aux, 
                              criterion_species, criterion_month, DEVICE)

            history['train_loss'].append(train_m['loss'])
            history['loss_biomass'].append(train_m['bio'])
            history['loss_aux'].append(train_m['aux'])
            history['loss_species'].append(train_m['sp'])
            history['loss_month'].append(train_m['mo'])
            history['val_loss'].append(val_m['loss'])
            history['val_loss_biomass'].append(val_m['bio'])
            history['val_loss_aux'].append(val_m['aux'])
            history['val_loss_species'].append(val_m['sp'])
            history['val_loss_month'].append(val_m['mo'])
            history['ind_loss'].append(hold_m['loss'])
            history['ind_loss_biomass'].append(hold_m['bio'])
            history['ind_loss_aux'].append(hold_m['aux'])
            history['ind_loss_species'].append(hold_m['sp'])
            history['ind_loss_month'].append(hold_m['mo'])
            history['val_r2'].append(val_m['r2'])
            history['holdout_r2'].append(hold_m['r2'])

            logger.info(
                f"E{epoch+1:02d} | TrL:{train_m['loss']:.3f} Bio:{train_m['bio']:.3f} | "
                f"VL:{val_m['loss']:.3f} VR2:{val_m['r2']:.3f} | HR2:{hold_m['r2']:.3f}"
            )

            if val_m['r2'] > best_val_r2:
                best_val_r2 = val_m['r2']
                best_holdout_r2 = hold_m['r2']
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_fold{fold}.pth"))
                logger.info(f"  → Best VR2:{best_val_r2:.4f} HR2:{best_holdout_r2:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(f"Early stop at epoch {epoch+1}")
                    break

            plot_training_history(history, fold, session_dir)

        model.load_state_dict(torch.load(os.path.join(session_dir, f"best_fold{fold}.pth"), weights_only=True))
        ewc = EWC(model, train_loader, DEVICE, importance=EWC_IMPORTANCE)
        logger.info(f"EWC updated after fold {fold}")

        fold_results.append(best_val_r2)
        holdout_results.append(best_holdout_r2)

    logger.info(f"\n{'='*40}")
    logger.info(f"Val R2: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    logger.info(f"Holdout R2: {np.mean(holdout_results):.4f} +/- {np.std(holdout_results):.4f}")

if __name__ == "__main__":
    run_training()
