# unified_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

from configs import *
from common import (
    load_data, setup_logging, set_seed, get_image_data_transforms,
    calculate_global_weighted_r2, enforce_physical_constraints,
    plot_training_history
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel, initialize_weights

def train_one_epoch(model, loader, optimizer, criterion_biomass, criterion_aux, device, scheduler=None):
    model.train()
    running_loss = 0.0
    running_loss_biomass = 0.0
    running_loss_aux = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        biomass_pred, aux_pred = model(images)
        
        # Loss calculation
        # 1. Biomass Loss (Weighted MSE)
        loss_biomass = criterion_biomass(biomass_pred, targets)
        # Apply official weights
        weighted_loss_biomass = (loss_biomass * COL_WEIGHTS_TENSOR).mean()
        
        # 2. Aux Loss (MSE)
        loss_aux = criterion_aux(aux_pred, aux_feats)
        
        # Total Loss
        total_loss = weighted_loss_biomass + 0.5 * loss_aux # 0.5 is a hyperparameter for aux importance
        
        total_loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        running_loss += total_loss.item()
        running_loss_biomass += weighted_loss_biomass.item()
        running_loss_aux += loss_aux.item()
        
        pbar.set_postfix({
            'L': f"{total_loss.item():.4f}", 
            'LB': f"{weighted_loss_biomass.item():.4f}",
            'LA': f"{loss_aux.item():.4f}"
        })
        
    return {
        'loss': running_loss / len(loader),
        'loss_biomass': running_loss_biomass / len(loader),
        'loss_aux': running_loss_aux / len(loader)
    }

@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        
        biomass_pred, aux_pred = model(images)
        
        # We only care about biomass for the main validation metric
        loss_biomass = criterion_biomass(biomass_pred, targets)
        weighted_loss_biomass = (loss_biomass * COL_WEIGHTS_TENSOR).mean()
        
        running_loss += weighted_loss_biomass.item()
        
        all_preds.append(biomass_pred.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Enforce physics for the metric calculation
    all_preds_phys = enforce_physical_constraints(all_preds)
    
    # Calculate Special R2
    r2_score = calculate_global_weighted_r2(all_targets, all_preds_phys, OFFICIAL_WEIGHTS)
    
    return {
        'loss': running_loss / len(loader),
        'r2': r2_score
    }

def run_training():
    # 1. Setup
    session_dir = setup_logging(file_name_part="Unified_Trainer")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    
    df = load_data(logger)
    
    # 2. Save Metadata for Inference
    species_list = sorted(df['Species'].unique().tolist())
    metadata = {
        'species_list': species_list,
        'species_to_id': {s: i for i, s in enumerate(species_list)},
        'target_cols': TARGET_COLS,
        'aux_cols': ['Pre_GSHH_NDVI', 'Height_Ave_cm_log'],
        'official_weights': OFFICIAL_WEIGHTS,
        'image_size': IMAGE_SIZE,
        'backbone': BACKBONE_S1
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'wb') as f:
        f.write(json.dumps(metadata, indent=4).encode('utf-8'))
    logger.info(f"Metadata saved to {os.path.join(session_dir, 'metadata.json')}")

    # 3. Prepare Groups for GroupKFold
    # Combine State and Sampling_Date for a robust group
    df['group'] = df['State'] + "_" + df['Sampling_Date'].astype(str)
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    train_transform, val_transform = get_image_data_transforms()
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['group'])):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ds = BiomassDataset(train_df, transform=train_transform)
        val_ds = BiomassDataset(val_df, transform=val_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = BiomassUnifiedModel(backbone_name=BACKBONE_S1).to(DEVICE)
        initialize_weights(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion_biomass = nn.MSELoss(reduction='none') # We apply weights manually
        criterion_aux = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)
        
        best_r2 = -float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'loss_biomass': [], 'loss_aux': []}
        
        for epoch in range(STAGE1_EPOCHS):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion_biomass, criterion_aux, DEVICE, scheduler)
            val_metrics = validate(model, val_loader, criterion_biomass, criterion_aux, DEVICE)
            
            history['train_loss'].append(train_metrics['loss'])
            history['loss_biomass'].append(train_metrics['loss_biomass'])
            history['loss_aux'].append(train_metrics['loss_aux'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_r2'].append(val_metrics['r2'])
            
            logger.info(f"Epoch {epoch+1}/{STAGE1_EPOCHS} - "
                        f"Train Loss: {train_metrics['loss']:.4f} (B: {train_metrics['loss_biomass']:.4f}, A: {train_metrics['loss_aux']:.4f}) | "
                        f"Val Loss: {val_metrics['loss']:.4f} | R2: {val_metrics['r2']:.4f}")
            
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                logger.info(f"New Best R2: {best_r2:.4f} (Fold {fold+1})")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold}.pth"))
            
            # Plotting after EACH epoch via common utility
            plot_training_history(history, fold, session_dir)
                
        fold_results.append(best_r2)
        
    logger.info(f"\nFinal Resume: Mean R2 across folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

if __name__ == "__main__":
    run_training()
