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
    plot_training_history, apply_tta
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel, initialize_weights

def train_one_epoch(model, loader, optimizer, criterion_biomass, criterion_aux, criterion_species, device, scheduler=None):
    model.train()
    running_loss = 0.0
    running_loss_biomass = 0.0
    running_loss_aux = 0.0
    running_loss_species = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        biomass_pred, aux_pred, species_logits = model(images)
        
        # Loss calculation
        # 1. Biomass Loss (Weighted MSE)
        loss_biomass = criterion_biomass(biomass_pred, targets)
        weighted_loss_biomass = (loss_biomass * COL_WEIGHTS_TENSOR).mean()
        
        # 2. Aux Loss (Huber)
        loss_aux = criterion_aux(aux_pred, aux_feats)
        
        # 3. Species Loss (Cross Entropy)
        loss_species = criterion_species(species_logits, species_id)
        
        # 4. Total Val Loss (Balanced weights)
        total_loss = BIOMASS_FEAT_WEIGHT * weighted_loss_biomass + AUX_FEAT_WEIGHT * loss_aux + SPECIES_FEAT_WEIGHT * loss_species
        
        total_loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        running_loss += total_loss.item()
        running_loss_biomass += weighted_loss_biomass.item()
        running_loss_aux += loss_aux.item()
        running_loss_species += loss_species.item()
        
        pbar.set_postfix({
            'L': f"{total_loss.item():.4f}", 
            'LB': f"{weighted_loss_biomass.item():.4f}",
            'LA': f"{loss_aux.item():.4f}",
            'LS': f"{loss_species.item():.4f}"
        })
        
    return {
        'loss': running_loss / len(loader),
        'loss_biomass': running_loss_biomass / len(loader),
        'loss_aux': running_loss_aux / len(loader),
        'loss_species': running_loss_species / len(loader)
    }

@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, criterion_species, device):
    model.eval()
    
    # Explicit initialization
    running_loss = 0.0
    running_aux_loss = 0.0
    running_species_loss = 0.0
    running_biomass_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['targets'].to(device) # Raw gram scale
            aux_feats = batch['aux_feats'].to(device)
            species_id = batch['species_id'].to(device)
            
            # Forward pass (now in Raw Space)
            if USE_TTA:
                biomass_pred, aux_pred, species_logits = apply_tta(model, images, device)
            else:
                biomass_pred, aux_pred, species_logits = model(images)
            
            # 1. Prediction Clamping (Max 256.0 grams)
            # Physical limit and biomass cannot be negative
            biomass_pred_clamped = torch.clamp(biomass_pred, 0.0, 256.0)
            
            # 3. Component Losses (Calculated on RAW targets)
            loss_biomass = criterion_biomass(biomass_pred, targets)
            loss_aux = criterion_aux(aux_pred, aux_feats)
            loss_species = criterion_species(species_logits, species_id)
            
            weighted_loss_biomass = (loss_biomass * COL_WEIGHTS_TENSOR).mean()
            
            # 4. Total Val Loss (Balanced weights)
            total_val_loss = (BIOMASS_FEAT_WEIGHT * weighted_loss_biomass + AUX_FEAT_WEIGHT * loss_aux + SPECIES_FEAT_WEIGHT * loss_species).item()
            
            # 5. Accumulate Metrics
            running_loss += total_val_loss
            running_biomass_loss += weighted_loss_biomass.item()
            running_aux_loss += loss_aux.item()
            running_species_loss += loss_species.item()
            
            all_preds.append(biomass_pred_clamped.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
    # Aggregate results
    all_preds_concat = np.concatenate(all_preds, axis=0)
    all_targets_concat = np.concatenate(all_targets, axis=0)
    
    # Enforce physics for the metric calculation
    all_preds_phys = enforce_physical_constraints(all_preds_concat)
    r2_score = calculate_global_weighted_r2(all_targets_concat, all_preds_phys, OFFICIAL_WEIGHTS)
    
    num_batches = len(loader)
    return {
        'loss': running_loss / num_batches,
        'loss_biomass': running_biomass_loss / num_batches,
        'loss_aux': running_aux_loss / num_batches,
        'loss_species': running_species_loss / num_batches,
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
    df['group'] = df['State'] + "_" + df['season'] + "_" + df['Sampling_Date'].astype(str)
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    train_transform, val_transform = get_image_data_transforms()
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['group'])):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ds = BiomassDataset(train_df, transform=train_transform, species_to_id=metadata['species_to_id'])
        val_ds = BiomassDataset(val_df, transform=val_transform, species_to_id=metadata['species_to_id'])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = BiomassUnifiedModel(backbone_name=BACKBONE_S1, num_species=len(species_list)).to(DEVICE)
        initialize_weights(model)
        
        # Freezing logic for small dataset optimization
        if FREEZE_BACKBONE:
            logger.info(f"Freezing backbone (Fraction: {BACKBONE_FREEZE_FRACTION})")
            model.freeze_backbone(freeze_fraction=BACKBONE_FREEZE_FRACTION)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        # Using HuberLoss for raw-space: robust to high-grams outliers
        criterion_biomass = nn.HuberLoss(reduction='none', delta=1.0) 
        criterion_aux = nn.HuberLoss(delta=1.0) 
        criterion_species = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LEARNING_RATE,
            epochs=STAGE1_EPOCHS,
            steps_per_epoch=steps_per_epoch
        )
        
        best_r2 = -float('inf')
        history = {
            'train_loss': [], 'val_loss': [], 'val_r2': [], 
            'loss_biomass': [], 'loss_aux': [], 'loss_species': [],
            'val_loss_biomass': [], 'val_loss_aux': [], 'val_loss_species': []
        }
        
        for epoch in range(STAGE1_EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, 
                criterion_biomass, criterion_aux, criterion_species, 
                DEVICE, scheduler
            )
            val_metrics = validate(
                model, val_loader, 
                criterion_biomass, criterion_aux, criterion_species, 
                DEVICE
            )
            
            history['train_loss'].append(train_metrics['loss'])
            history['loss_biomass'].append(train_metrics['loss_biomass'])
            history['loss_aux'].append(train_metrics['loss_aux'])
            history['loss_species'].append(train_metrics['loss_species'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_r2'].append(val_metrics['r2'])
            history['val_loss_biomass'].append(val_metrics['loss_biomass'])
            history['val_loss_aux'].append(val_metrics['loss_aux'])
            history['val_loss_species'].append(val_metrics['loss_species'])
            
            logger.info(f"Epoch {epoch+1}/{STAGE1_EPOCHS} - "
                        f"Train Loss: {train_metrics['loss']:.4f} ("
                        f"Biomass: {train_metrics['loss_biomass']:.4f}, "
                        f"Aux: {train_metrics['loss_aux']:.4f}, "
                        f"Species: {train_metrics['loss_species']:.4f}) | "
                        f"Val Loss: {val_metrics['loss']:.4f} ("
                        f"Biomass: {val_metrics['loss_biomass']:.4f}, "
                        f"Aux: {val_metrics['loss_aux']:.4f}, "
                        f"Species: {val_metrics['loss_species']:.4f}) | "
                        f"R2: {val_metrics['r2']:.4f}")
            
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                logger.info(f"New Best R2: {best_r2:.4f} (Fold {fold+1})")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold}.pth"))
            

            plot_training_history(history, fold, session_dir)
                
        fold_results.append(best_r2)
        
    logger.info(f"\nFinal Resume: Mean R2 across folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

if __name__ == "__main__":
    run_training()
