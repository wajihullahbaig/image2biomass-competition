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
import sys
from pathlib import Path

# Add project root to path for package imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from configs import *
from common import (
    check_group_leakage, load_data, setup_logging, set_seed, get_image_data_transforms_v1,
    calculate_global_weighted_r2, enforce_physical_constraints,
    plot_training_history, apply_tta, upsample_minority_classes
)
from dataset import BiomassDataset
from models import BiomassUnifiedModel, initialize_weights
from scripts.make_holdout import generate_holdout

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
        
        # 1. Biomass Loss (Huber on Decagrams)
        weights = COL_WEIGHTS_TENSOR.view(1, -1)
        
        loss_biomass = criterion_biomass(biomass_pred, targets)
        loss_biomass = (loss_biomass * weights).sum() / weights.sum()
        
        # 2. Auxiliary Loss (Huber on NDVI/Height)
        loss_aux = criterion_aux(aux_pred, aux_feats).mean()
        
        # 3. Species Loss (Cross Entropy)
        loss_species = criterion_species(species_logits, species_id) / 10.0 # Scale down for stability
        
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
            'L_Bio': f"{loss_biomass.item():.4f}", 
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
            
            # Loss Calculation (In Log Space)
            weights = COL_WEIGHTS_TENSOR.view(1, -1)
            loss_biomass = criterion_biomass(biomass_pred, targets)
            loss_biomass = (loss_biomass * weights).sum() / weights.sum()
            
            loss_aux = criterion_aux(aux_pred, aux_feats).mean()
            loss_species = criterion_species(species_logits, species_id)/10.0 # Scale down for stability
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
            
    # Calculate R2 Score (Convert log-space back to real scale Grams)
    all_targets_real = np.expm1(np.concatenate(all_targets))
    all_preds_real = np.expm1(np.concatenate(all_preds_biomass))
       
    # Apply physical constraints to real-scale predictions
    all_preds_real = enforce_physical_constraints(all_preds_real)

    # Calculate R2 Score (Official Weighted Global Metric)
    # Order: [Clover, Dead, Green, Total, GDM]
    official_avg_r2 = calculate_global_weighted_r2(all_targets_real, all_preds_real, OFFICIAL_WEIGHTS)
    
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
    logger.info(config_str())

    # --- AUTOMATED DATA PIPELINE ---
    logger.info("Starting Data Refresh (Scale -> Split)...")
    load_data(logger) # Scales to Decagrams and saves wide.csv
    h_len, t_len = generate_holdout() # Reads wide.csv and creates temporal splits
    logger.info(f"Data ready: {t_len} training samples, {h_len} independent holdout samples.")

    # Load the Temporally Honest Splits
    holdout_dir = os.path.join(os.getcwd(), 'holdout_outputs')
    train_csv_path = os.path.join(holdout_dir, 'train_filtered.csv')
    test_csv_path = os.path.join(holdout_dir, 'holdout.csv')

    holdout_csv_path = os.path.join(holdout_dir, 'holdout.csv')

    if not os.path.exists(train_csv_path) or not os.path.exists(holdout_csv_path):
        logger.error(f"Required files not found in {holdout_dir}. Run scripts/make_holdout.py first.")
        return

    df = pd.read_csv(train_csv_path)
    df_independent_test = pd.read_csv(holdout_csv_path)
    
    # Create Groups for Leakage Check
    df['cv_group'] = df['State'] + "_" + df['Sampling_Date'].astype(str)
    df_independent_test['cv_group'] = df_independent_test['State'] + "_" + df_independent_test['Sampling_Date'].astype(str)

    # GLOBAL LEAKAGE CHECK: Dev Set vs. Independent Holdout
    logger.info("--- Global Separation Check (Dev vs. Holdout) ---")
    check_group_leakage(df, df_independent_test, group_col='cv_group', logger=logger)
    check_group_leakage(df, df_independent_test, group_col='sample_id', logger=logger)
    
    # Ensure date parsing for both
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df_independent_test['Sampling_Date'] = pd.to_datetime(df_independent_test['Sampling_Date'])

    # Log Global Stats
    log_dataset_stats(df, logger, "DEVELOPMENT SET (FOR K-FOLD)")
    log_dataset_stats(df_independent_test, logger, "INDEPENDENT TEMPORAL HOLDOUT (FOR FINAL TESTING)")
    
    # 2. Save Metadata for Inference
    # Use global species list (from both sets) to ensure consistency
    all_combined = pd.concat([df, df_independent_test])
    species_list = sorted(all_combined['Species'].unique().tolist())
    metadata = {
        'species_list': species_list,
        'species_to_id': {s: i for i, s in enumerate(species_list)},
        'target_cols': TARGET_COLS,
        'aux_cols': ['Pre_GSHH_NDVI', 'Height_Ave_cm_log'],
        'official_weights': OFFICIAL_WEIGHTS,
        'image_size': IMAGE_SIZE,
        'backbone': BACKBONE
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {os.path.join(session_dir, 'metadata.json')}")

    # 3. K-Fold Training Cycle
    # We maintain temporal honesty by sorting by date.
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    # We group by State + Date to separate ENVIRONMENTS (Farms/Times)
    # This allows the model to learn Species-specific textures while testing 
    # spatial/temporal generalization.
    df['cv_group'] = df['State'] + "_" + df['Sampling_Date'].astype(str)
    
    train_transform, val_transform = get_image_data_transforms_v1()    
    
    fold_results = []
    independent_holdout_results = []
    
    gkf = GroupKFold(n_splits=N_FOLDS)    
    
    # Independent Loader (eval on this every epoch)
    independent_ds = BiomassDataset(df_independent_test, transform=val_transform, species_to_id=metadata['species_to_id'])
    independent_loader = DataLoader(independent_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['cv_group'])):
        logger.info(f"\n{'='*20} Fold {fold}/{N_FOLDS} (Environment-Grouped) {'='*20}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # 4. Leakage Check (Sanity Check)
        check_group_leakage(train_df, val_df, group_col='cv_group', logger=logger)
        check_group_leakage(train_df, val_df, group_col='sample_id', logger=logger)
        
        # Verify species distribution/overlap
        train_species = set(train_df['Species'])
        val_species = set(val_df['Species'])
        overlap = train_species.intersection(val_species)
        if overlap:
            logger.warning(f"Leakage detected: Species overlap in local CV: {overlap}")
        
        # Upsample the Train set to balance species representation
        train_df = upsample_minority_classes(train_df, 'Species', logger)
        
        # Datasets
        train_ds = BiomassDataset(train_df, transform=train_transform, species_to_id=metadata['species_to_id'])
        val_ds = BiomassDataset(val_df, transform=val_transform, species_to_id=metadata['species_to_id'])
        
        # Loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = BiomassUnifiedModel(backbone_name=BACKBONE, num_species=len(species_list)).to(DEVICE)
        initialize_weights(model)
        
        if FREEZE_BACKBONE:
            model.freeze_backbone(freeze_fraction=BACKBONE_FREEZE_FRACTION)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion_biomass = nn.MSELoss(reduction='none') 
        criterion_aux = nn.HuberLoss(delta=1.0) 
        criterion_species = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_month = nn.MSELoss() 
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader)
        )
        
        best_val_r2 = -float('inf')
        best_independent_r2 = -float('inf')
        
        history = {
            'train_loss': [], 'val_loss': [], 'ind_loss': [], 'val_r2': [], 'holdout_r2': [],
            'loss_biomass': [], 'loss_aux': [], 'loss_species': [], 'loss_month': [],
            'val_loss_biomass': [], 'val_loss_aux': [], 'val_loss_species': [], 'val_loss_month': [],
            'ind_loss_biomass': [], 'ind_loss_aux': [], 'ind_loss_species': [], 'ind_loss_month': []
        }
        
        for epoch in range(EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, 
                criterion_biomass, criterion_aux, criterion_species, criterion_month,
                DEVICE, scheduler
            )
            
            # 1. Validation on CV Set (Optimization Target)
            val_metrics = validate(
                model, val_loader, 
                criterion_biomass, criterion_aux, criterion_species, criterion_month,
                DEVICE
            )
            
            # 2. Evaluation on Independent Temporal Holdout
            ind_metrics = validate(
                model, independent_loader,
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
            history['val_loss_biomass'].append(val_metrics['loss_biomass'])
            history['val_loss_aux'].append(val_metrics['loss_aux'])
            history['val_loss_species'].append(val_metrics['loss_species'])
            history['val_loss_month'].append(val_metrics['loss_month'])

            history['ind_loss'].append(ind_metrics['loss'])
            history['ind_loss_biomass'].append(ind_metrics['loss_biomass'])
            history['ind_loss_aux'].append(ind_metrics['loss_aux'])
            history['ind_loss_species'].append(ind_metrics['loss_species'])
            history['ind_loss_month'].append(ind_metrics['loss_month'])

            history['val_r2'].append(val_metrics['r2'])
            history['holdout_r2'].append(ind_metrics['r2'])
            
            # Logging
            logger.info(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train L: {train_metrics['loss']:.4f} (Bio: {train_metrics['loss_biomass']:.4f}, Aux: {train_metrics['loss_aux']:.4f}, Sp: {train_metrics['loss_species']:.4f}, Mo: {train_metrics['loss_month']:.4f}) | "
                f"Val L: {val_metrics['loss']:.4f} (Bio: {val_metrics['loss_biomass']:.4f}, Aux: {val_metrics['loss_aux']:.4f}, Sp: {val_metrics['loss_species']:.4f}, Mo: {val_metrics['loss_month']:.4f}) | "
                f"Val R2: {val_metrics['r2']:.3f} | "
                f"IND-TEST R2: {ind_metrics['r2']:.3f}"
            )
            
            # Save based on Local CV R2
            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                best_independent_r2 = ind_metrics['r2']
                logger.info(f"  → New Best Val R2: {best_val_r2:.4f} (Ind-Test: {best_independent_r2:.4f})")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold}.pth"))
            
            plot_training_history(history, fold, session_dir)
                
        fold_results.append(best_val_r2)
        independent_holdout_results.append(best_independent_r2)
        
    logger.info(f"\nFinal Summary:")
    logger.info(f"Mean Val R2 (CV): {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    logger.info(f"Mean Ind-Test R2: {np.mean(independent_holdout_results):.4f} (+/- {np.std(independent_holdout_results):.4f})")
            
        
    logger.info(f"\nFinal Resume: Mean R2 across folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

if __name__ == "__main__":
    run_training()
