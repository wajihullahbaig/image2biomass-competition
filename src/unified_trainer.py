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
    
    # Weights for the 5 biomass components [Clover, Dead, Green, Total, GDM]
    weights = COL_WEIGHTS_TENSOR.view(1, -1)
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        biomass_pred, aux_pred, species_logits, month_logits = model(images)
        
        # --- 1. Biomass Loss (Weighted) ---
        # Calculate per-sample, per-component loss
        raw_bio_loss = criterion_biomass(biomass_pred, targets) # Shape: (B, 5)
        # Apply component importance weights
        weighted_bio_loss = raw_bio_loss * weights 
        # Mean over batch
        loss_bio = weighted_bio_loss.sum() / images.size(0)
        
        # --- 2. Auxiliary Losses ---
        loss_aux = criterion_aux(aux_pred, aux_feats)
        loss_sp = criterion_species(species_logits, species_id)
        loss_mo = criterion_month(month_logits, month_target)
        
        # --- 3. Total Loss Combination ---
        total = (loss_bio * BIOMASS_FEAT_WEIGHT + 
                 loss_aux * AUX_FEAT_WEIGHT + 
                 loss_sp * SPECIES_FEAT_WEIGHT + 
                 loss_mo * MONTH_FEAT_WEIGHT)
        
        # EWC Regularization
        if ewc is not None:
            total += ewc.penalty(model)
        
        total.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        # Update Stats
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
        
        # --- Calculate Losses ---
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
    
    # --- Calculate Metrics ---
    targets_real = np.concatenate(all_targets)
    preds_real = enforce_physical_constraints(np.concatenate(all_preds))
    
    r2 = calculate_global_weighted_r2(targets_real, preds_real, OFFICIAL_WEIGHTS)
    r2_display = max(r2, -2.0) # Clamp for readable logs
    
    n = len(loader)
    return {
        'loss': running['loss']/n, 
        'bio': running['bio']/n, 
        'aux': running['aux']/n,
        'sp': running['sp']/n,
        'mo': running['mo']/n,
        'r2': r2, 
        'r2_display': r2_display
    }

def run_training():
    session_dir = setup_logging(file_name_part="Unified_Trainer")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    logger.info(config_str())

    logger.info("Loading Data...")
    df = load_data(logger)
    
    # Time-Series Split
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    df['year_week'] = df['Sampling_Date'].dt.to_period('W')
    unique_weeks = sorted(df['year_week'].unique())
    
    species_list = sorted(df['Species'].unique().tolist())
    num_species = len(species_list)
    
    train_transform, val_transform = get_image_data_transforms_v1()
    
    # Model
    model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=num_species).to(DEVICE)
    initialize_weights(model) 
    
    if FREEZE_BACKBONE:
        model.freeze_backbone(freeze_fraction=BACKBONE_FREEZE_FRACTION)
    
    # Loss
    criterion_biomass = nn.HuberLoss(delta=0.2, reduction='none') 
    criterion_aux = nn.HuberLoss(delta=1.0)
    criterion_species = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_month = nn.HuberLoss(delta=1.0)

    # Training Loop
    ewc = None
    
    for fold in range(1, len(unique_weeks)):
        history_weeks = unique_weeks[:fold]
        holdout_week = unique_weeks[fold]
        
        logger.info(f"\n{'='*20} Fold {fold}/{len(unique_weeks)-1} {'='*20}")
        logger.info(f"Train: {history_weeks[0]} -> {history_weeks[-1]}")
        logger.info(f"Holdout: {holdout_week}")

        df_history = df[df['year_week'].isin(history_weeks)].copy()
        df_holdout = df[df['year_week'] == holdout_week].copy()
        
        if len(df_holdout) == 0: continue

        # Train/Val Split
        split_idx = int(len(df_history) * 0.8)
        train_df = df_history.iloc[:split_idx].copy()
        val_df = df_history.iloc[split_idx:].copy()
        
        if len(train_df) < 5: continue
        
        train_df = upsample_minority_classes(train_df, 'Species', logger)

        train_ds = BiomassDataset(train_df, transform=train_transform)
        val_ds = BiomassDataset(val_df, transform=val_transform)
        holdout_ds = BiomassDataset(df_holdout, transform=val_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_val_r2 = -float('inf')
        
        # --- UPDATED HISTORY: Added Independent Fold Component Losses ---
        history = {
            'train_loss': [], 'val_loss': [], 'ind_loss': [],
            'loss_biomass': [], 'val_loss_biomass': [], 'ind_loss_biomass': [],
            
            # Aux
            'loss_aux': [], 'val_loss_aux': [], 'ind_loss_aux': [],
            # Species
            'loss_species': [], 'val_loss_species': [], 'ind_loss_species': [],
            # Month
            'loss_month': [], 'val_loss_month': [], 'ind_loss_month': [],
            
            'val_r2': [], 'holdout_r2': []
        }

        for epoch in range(EPOCHS):
            train_m = train_one_epoch(model, train_loader, optimizer, criterion_biomass, 
                                       criterion_aux, criterion_species, criterion_month, DEVICE, ewc)
            
            val_m = validate(model, val_loader, criterion_biomass, criterion_aux, 
                             criterion_species, criterion_month, DEVICE)
            
            hold_m = validate(model, holdout_loader, criterion_biomass, criterion_aux, 
                              criterion_species, criterion_month, DEVICE)
            
            scheduler.step(val_m['r2'])

            # --- Append Metrics ---
            # Train
            history['train_loss'].append(train_m['loss'])
            history['loss_biomass'].append(train_m['bio'])
            history['loss_aux'].append(train_m['aux'])
            history['loss_species'].append(train_m['sp'])
            history['loss_month'].append(train_m['mo'])

            # Val
            history['val_loss'].append(val_m['loss'])
            history['val_loss_biomass'].append(val_m['bio'])
            history['val_loss_aux'].append(val_m['aux'])
            history['val_loss_species'].append(val_m['sp'])
            history['val_loss_month'].append(val_m['mo'])
            
            # Independent (Holdout) - NOW CAPTURED
            history['ind_loss'].append(hold_m['loss'])
            history['ind_loss_biomass'].append(hold_m['bio'])
            history['ind_loss_aux'].append(hold_m['aux'])
            history['ind_loss_species'].append(hold_m['sp'])
            history['ind_loss_month'].append(hold_m['mo'])
            
            # R2
            history['val_r2'].append(val_m['r2_display'])
            history['holdout_r2'].append(hold_m['r2_display'])

            logger.info(
                f"E{epoch+1:02d} | "
                f"TrBio:{train_m['bio']:.4f} | "
                f"VBio:{val_m['bio']:.4f} VR2:{val_m['r2']:.4f} | "
                f"HR2:{hold_m['r2']:.4f}"
            )

            if val_m['r2'] > best_val_r2:
                best_val_r2 = val_m['r2']
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_fold{fold}.pth"))
            
            plot_training_history(history, fold, session_dir)

        model.load_state_dict(torch.load(os.path.join(session_dir, f"best_fold{fold}.pth")))
        ewc = EWC(model, train_loader, DEVICE, importance=EWC_IMPORTANCE)
        logger.info(f"Updated EWC for Fold {fold}")

if __name__ == "__main__":
    run_training()