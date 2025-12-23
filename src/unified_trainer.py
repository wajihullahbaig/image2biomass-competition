# unified_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

from configs import *
from common import (
    check_group_leakage, load_data, setup_logging, set_seed, get_image_data_transforms,
    calculate_global_weighted_r2, enforce_physical_constraints,
    plot_training_history, apply_tta, upsample_minority_classes
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel, initialize_weights

def train_one_epoch(model, loader, optimizer, criterion_biomass, criterion_aux, criterion_species, criterion_month, device, scheduler=None):
    model.train()
    running_loss = 0.0
    running_loss_biomass = 0.0
    running_loss_aux = 0.0
    running_loss_species = 0.0
    running_loss_month = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        aux_feats = batch['aux_feats'].to(device)
        species_id = batch['species_id'].to(device)
        month_target = batch['month_sin_cos'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        biomass_pred, aux_pred, species_logits, month_logits = model(images)
        
        # Loss calculation
        # 1. Biomass Loss (Weighted MSE)
        weights = COL_WEIGHTS_TENSOR.view(1, -1)
        loss_biomass = criterion_biomass(biomass_pred, targets)
        loss_biomass = (loss_biomass * weights).sum() / weights.sum()
        
        # 2. Auxiliary Loss (Huber on NDVI/Height)
        loss_aux = criterion_aux(aux_pred, aux_feats).mean()
        
        # 3. Species Loss (Cross Entropy)
        loss_species = criterion_species(species_logits, species_id)
        
        # 4. Month Loss (Huber on Sin/Cos) - Phenology Regularizer
        loss_month = criterion_month(month_logits, month_target).mean()
        
        # Combined Loss
        total_loss = (loss_biomass * BIOMASS_FEAT_WEIGHT) + \
                     (loss_aux * AUX_FEAT_WEIGHT) + \
                     (loss_species * SPECIES_FEAT_WEIGHT) + \
                     (loss_month * MONTH_FEAT_WEIGHT) 
        
        total_loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        running_loss += total_loss.item()
        running_loss_biomass += loss_biomass.item()
        running_loss_aux += loss_aux.item()
        running_loss_species += loss_species.item()
        running_loss_month += loss_month.item()
        
        pbar.set_postfix({
            'L_Bio': f"{loss_biomass.item()/1000:.4f}", 
            'L_Aux': f"{loss_aux.item():.4f}",
            'L_Sp': f"{loss_species.item():.4f}",
            'L_Mo': f"{loss_month.item():.4f}"
        })
        
    return {
        'loss': running_loss / len(loader),
        'loss_biomass': running_loss_biomass / len(loader),
        'loss_aux': running_loss_aux / len(loader),
        'loss_species': running_loss_species / len(loader),
        'loss_month': running_loss_month / len(loader)
    }

def log_dataset_stats(df, logger, title="Dataset Stats"):
    logger.info(f"\n--- {title} ---")
    logger.info(f"Total Samples: {len(df)}")
    
    # 1. Target Stats
    logger.info("Target Distributions (g):")
    for col in TARGET_COLS:
        stats = df[col].describe()
        logger.info(f"  - {col:15}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Max={stats['max']:.2f}")
    
    # 2. Categorical Counts
    logger.info(f"Unique Species: {df['Species'].nunique()}")
    counts = df['Species'].value_counts()
    logger.info("  Species Breakdown:")
    for s, count in counts.items():
        logger.info(f"    - {s:25}: {count}")
        
    logger.info(f"Unique States : {df['State'].nunique()} ({', '.join(df['State'].unique())})")
    
    # Seasonality
    if 'Sampling_Date' in df.columns:
        months = pd.to_datetime(df['Sampling_Date']).dt.month
        logger.info(f"Unique Months : {months.nunique()}")


@torch.no_grad()
def validate(model, loader, criterion_biomass, criterion_aux, criterion_species, criterion_month, device):
    model.eval()
    running_loss = 0.0
    running_loss_biomass = 0.0
    running_loss_species = 0.0
    running_loss_month = 0.0
    running_loss_aux = 0.0
    
    all_targets = []
    all_preds_biomass = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['targets'].to(device) # Raw gram scale
            aux_feats = batch['aux_feats'].to(device)
            species_id = batch['species_id'].to(device)
            month_target = batch['month_sin_cos'].to(device)
            
            # Forward pass (now in Raw Space)
            if USE_TTA:
                biomass_pred, aux_pred, species_logits, month_logits = apply_tta(model, images, device, n_passes=5)
            else:
                biomass_pred, aux_pred, species_logits, month_logits = model(images)
            
            # 1. Prediction Clamping (Max 256.0 grams)
            # Physical limit and biomass cannot be negative
            biomass_pred = torch.clamp(biomass_pred, 0.0, 256.0)
            
            # Loss Calculation
            weights = COL_WEIGHTS_TENSOR.view(1, -1)
            loss_biomass = criterion_biomass(biomass_pred, targets)
            loss_biomass = (loss_biomass * weights).sum() / weights.sum()
            
            loss_aux = criterion_aux(aux_pred, aux_feats).mean()
            loss_species = criterion_species(species_logits, species_id)
            loss_month = criterion_month(month_logits, month_target).mean()
            
            total_loss = (loss_biomass * BIOMASS_FEAT_WEIGHT) + \
                         (loss_aux * AUX_FEAT_WEIGHT) + \
                         (loss_species * SPECIES_FEAT_WEIGHT) + \
                         (loss_month * MONTH_FEAT_WEIGHT)
            
            running_loss += total_loss.item()
            running_loss_biomass += loss_biomass.item()
            running_loss_aux += loss_aux.item() # Accumulate aux loss
            running_loss_species += loss_species.item()
            running_loss_month += loss_month.item()
            
            all_targets.append(targets.cpu().numpy())
            all_preds_biomass.append(biomass_pred.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_preds_biomass = np.concatenate(all_preds_biomass)
       
    # Calculate R2 Score (Official Weighted Global Metric)
    # We use the OFFICIAL_WEIGHTS from configs.py
    # Order: [Clover, Dead, Green, Total, GDM]
    official_avg_r2 = calculate_global_weighted_r2(all_targets, all_preds_biomass, OFFICIAL_WEIGHTS)
    
    num_batches = len(loader)
    return {
        'loss': running_loss / num_batches,
        'loss_biomass': running_loss_biomass / num_batches,
        'loss_aux': running_loss_aux / num_batches,
        'loss_species': running_loss_species / num_batches,
        'loss_month': running_loss_month / num_batches,
        'r2': official_avg_r2
    }

def run_training():
    # 1. Setup
    session_dir = setup_logging(file_name_part="Unified_Trainer")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    
    df = load_data(logger)
    
    # Log Global Stats
    log_dataset_stats(df, logger, "GLOBAL DATASET OVERVIEW")
    
    # 2. Save Metadata for Inference
    species_list = sorted(df['Species'].unique().tolist())
    metadata = {
        'species_list': species_list,
        'species_to_id': {s: i for i, s in enumerate(species_list)},
        'target_cols': TARGET_COLS,
        'aux_cols': ['Pre_GSHH_NDVI', 'Height_Ave_cm_log'],
        'official_weights': OFFICIAL_WEIGHTS,
        'image_size': IMAGE_SIZE,
        'backbone': BACKBONE
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'wb') as f:
        f.write(json.dumps(metadata, indent=4).encode('utf-8'))
    logger.info(f"Metadata saved to {os.path.join(session_dir, 'metadata.json')}")

    # 3. Nested Validation Strategy
    # Outer Loop: GroupKFold (Strictly separates Farms/Dates to prevent leakage)
    # Inner Loop: Stratified Split (Ensures stable optimization with balanced species)
    
    # Shuffle DF to ensure random groups for GroupKFold (which doesn't shuffle)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_transform, val_transform = get_image_data_transforms()    
    fold_results = []
    gkf = GroupKFold(n_splits=N_FOLDS)    
    df['group'] = df["State"] + "_" +   df["Sampling_Date"].astype(str) + "_" + df["season"].astype(str)        
    
    # Outer Split: Group-based
    for fold, (outer_train_idx, outer_holdout_idx) in enumerate(gkf.split(df,groups=df['group'])):
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        
        df_outer_train = df.iloc[outer_train_idx]
        df_holdout = df.iloc[outer_holdout_idx]
        check_group_leakage(df_outer_train, df_holdout)

        # create path and save outer split csvs        
        outer_df_path = os.path.join(session_dir,"splits", f"fold{fold+1}_train.csv")
        os.makedirs(os.path.dirname(outer_df_path), exist_ok=True)
        df_outer_train.to_csv(outer_df_path, index=False)
        df_holdout_path = os.path.join(session_dir, "splits", f"fold{fold}_holdout.csv")
        os.makedirs(os.path.dirname(df_holdout_path), exist_ok=True)
        df_holdout.to_csv(df_holdout_path, index=False)
        
        # Inner Split: Stratified by Species (Optimization Set)
        # We take 20% of the TRAINING data to act as the validation set for the scheduler/early stopping
        # This allows the model to learn from a balanced signal, even if it creates slight leakage vs 'true' generalization
        train_df, inner_val_df = train_test_split(
            df_outer_train, 
            test_size=0.2, 
            stratify=df_outer_train['season'],
            random_state=42
        )
        
        # Upsample the Inner Train set
        train_df = upsample_minority_classes(train_df, 'Species', logger)
        
        # Datasets
        train_ds = BiomassDataset(train_df, transform=train_transform, species_to_id=metadata['species_to_id'])
        val_ds = BiomassDataset(inner_val_df, transform=val_transform, species_to_id=metadata['species_to_id'])
        holdout_ds = BiomassDataset(df_holdout, transform=val_transform, species_to_id=metadata['species_to_id'])
        
        # Log Stats
        log_dataset_stats(inner_val_df, logger, f"FOLD {fold+1} INNER VAL STATS (Stratified)")
        log_dataset_stats(df_holdout, logger, f"FOLD {fold+1} OUTER HOLDOUT STATS (Groups)")
        
        # Loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=len(species_list)).to(DEVICE)
        initialize_weights(model)
        
        if FREEZE_BACKBONE:
            logger.info(f"Freezing backbone (Fraction: {BACKBONE_FREEZE_FRACTION})")
            model.freeze_backbone(freeze_fraction=BACKBONE_FREEZE_FRACTION)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion_biomass = nn.HuberLoss(reduction='none', delta=5.0) 
        criterion_aux = nn.HuberLoss(delta=5.0) 
        criterion_species = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_month = nn.HuberLoss(delta=1.0) 
        
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LEARNING_RATE,
            epochs=STAGE1_EPOCHS,
            steps_per_epoch=steps_per_epoch
        )
        
        best_r2 = -float('inf')
        history = {
            'train_loss': [], 'val_loss': [], 'val_r2': [], 'holdout_r2': [],
            'loss_biomass': [], 'loss_aux': [], 'loss_species': [], 'loss_month': [],
            'val_loss_biomass': [], 'val_loss_aux': [], 'val_loss_species': [], 'val_loss_month': []
        }
        
        for epoch in range(STAGE1_EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, 
                criterion_biomass, criterion_aux, criterion_species, criterion_month,
                DEVICE, scheduler
            )
            
            # 1. Validation on Stratified Inner Set (Optimization Target)
            val_metrics = validate(
                model, val_loader, 
                criterion_biomass, criterion_aux, criterion_species, criterion_month,
                DEVICE
            )
            
            # 2. Validation on Outer Holdout Set (Strict Monitoring)
            # We don't save based on this, but we log it to see if we are overfitting the stratification
            holdout_metrics = validate(
                model, holdout_loader,
                criterion_biomass, criterion_aux, criterion_species, criterion_month,
                DEVICE
            )
            
            # Store history
            history['train_loss'].append(train_metrics['loss'])
            history['loss_biomass'].append(train_metrics['loss_biomass'])
            history['loss_aux'].append(train_metrics['loss_aux'])
            history['loss_species'].append(train_metrics['loss_species'])
            history['loss_month'].append(train_metrics['loss_month'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_r2'].append(val_metrics['r2'])
            history['holdout_r2'].append(holdout_metrics['r2']) # Track strict performance
            
            history['val_loss_biomass'].append(val_metrics['loss_biomass'])
            history['val_loss_aux'].append(val_metrics['loss_aux'])
            history['val_loss_species'].append(val_metrics['loss_species'])
            history['val_loss_month'].append(val_metrics['loss_month'])
            
            logger.info(f"Epoch {epoch+1}/{STAGE1_EPOCHS} | "
                        f"T_Loss: {train_metrics['loss']/1000.0:.2f}k | "
                        f"V_Loss: {val_metrics['loss']/1000.0:.2f}k | "
                        f"V_R2 (Strat): {val_metrics['r2']:.4f} | "
                        f"H_R2 (Group): {holdout_metrics['r2']:.4f}")
            
            # Save based on Inner Stratified R2 (Optimization Goal)
            holdout_r2 = holdout_metrics['r2']
            if holdout_r2 > best_r2:
                best_r2 = holdout_r2
                logger.info(f"New Best R2: Holdout: {holdout_metrics['r2']:.4f} - Stratified: {val_metrics['r2']:.4f} ")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold}.pth"))
            
            plot_training_history(history, fold, session_dir)
                
        fold_results.append(best_r2)            
        
    logger.info(f"\nFinal Resume: Mean R2 across folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

if __name__ == "__main__":
    run_training()
