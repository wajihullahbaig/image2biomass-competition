# train_holdout.py
# train_holdout_tiled.py - Enhanced Training with Tile Augmentation
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
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
    CORE_SPECIES, USE_TTA,
    MIN_TRAIN_SAMPLES, BACKBONE_FREEZE_THRESHOLD,
    FREEZE_BACKBONE, BACKBONE_FREEZE_FRACTION,
    MAX_GRAD_NORM, BACKBONE_LR_FACTOR,
    TILE_PROB, MIXUP_PROB, MIXUP_ALPHA
)
from common import (
    load_data, get_image_data_transforms, save_batch_images, 
    set_seed, calculate_global_weighted_r2, smart_upsample,
    get_taxonomy_targets,
    rotate_crop_resize, add_cv_group
)
from log_and_plots import (
    setup_logging, plot_training_history, 
    log_fold_details
)

from dataset import TiledBiomassDataset, TiledMixupDataset
from models import BiomassUnifiedModel
from torchvision.utils import save_image


def save_tta_images(images, view_name, batch_idx, fold, epoch, session_dir):
    """Save TTA-augmented images for visualization."""
    if fold != 0 or epoch != 0 or batch_idx > 0:
        return
        
    save_dir = os.path.join(session_dir, 'tta_debug', f'fold{fold+1}_ep{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize
    mean = torch.tensor(configs.IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(configs.IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    save_path = os.path.join(save_dir, f'batch{batch_idx}_{view_name}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)


# -----------------------------------------------------------------------------
# TRAINING ENGINE (No changes needed - works with tiled data automatically)
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_ce, device, epoch, session_dir=None, logger=None):
    model.train()
    metrics = defaultdict(float)
    scaler = torch.amp.GradScaler('cuda')
    
    all_preds_log = []
    all_targets_g = []
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets_g = batch['targets'].to(device)
        
        # Proactive Safety Check
        if torch.isnan(targets_g).any():
            if logger:
                logger.warning(f"NaN TARGETS DETECTED in batch {batch_idx}. Skipping.")
            continue

        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(device)
        species_vec = batch['species_id'].to(device)
        
        if epoch == 0 and batch_idx < 5 and session_dir:
            save_batch_images(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)
        
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
        
        if torch.isnan(total_loss):
            if logger:
                logger.warning(f"!!! NAN TOTAL LOSS at Ep {epoch}, batch {batch_idx} !!!")
            optimizer.zero_grad()
            continue

        scaler.scale(total_loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
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
            metrics['train_loss_int_mul']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B
            metrics['train_loss_int_add']  += nn.functional.mse_loss(aux_out[:, 3], aux_feats[:, 3]).item() * B
        
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
def validate(model, loader, criterion_reg, criterion_ce, device, prefix='val', use_tta=False, epoch=0, fold=0, session_dir=None):
    model.eval()
    metrics = defaultdict(float)
    all_preds_log, all_targets_g = [], []
    
    tta_views = [
        ('identity', lambda x: x),
        ('hflip', lambda x: torch.flip(x, [3])),
        ('vflip', lambda x: torch.flip(x, [2])),
        ('rot5', lambda x: rotate_crop_resize(x, 5)),
        ('rot-5', lambda x: rotate_crop_resize(x, -5)),
    ]
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        targets_g = batch['targets'].to(device)
        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(device)
        species_vec = batch['species_id'].to(device)
        taxonomy_targets = get_taxonomy_targets(species_vec)
        
        if use_tta:
            # Accumulate TTA outputs in Linear Space
            accum_bio_linear = 0
            accum_aux = 0
            accum_sp_probs = 0
            accum_tax_probs = 0
            
            for view_name, transform_fn in tta_views:
                img_aug = transform_fn(images)
                
                # Save TTA Debug Images (Fold 0, Batch 0, Epoch 0 only)
                if session_dir is not None:
                     save_tta_images(img_aug, view_name, batch_idx, fold, epoch, session_dir)
                
                bio_out, aux_out, sp_logits, tax_logits = model(img_aug)
                
                # Convert to linear/prob space for averaging
                accum_bio_linear += torch.expm1(bio_out)
                accum_aux += aux_out
                accum_sp_probs += torch.softmax(sp_logits, dim=1)
                accum_tax_probs += torch.softmax(tax_logits, dim=1)
            
            # Average
            avg_bio_linear = accum_bio_linear / len(tta_views)
            avg_aux = accum_aux / len(tta_views)
            avg_sp_probs = accum_sp_probs / len(tta_views)
            avg_tax_probs = accum_tax_probs / len(tta_views)
            
            # Reconstruct for Loss
            biomass_out = torch.log1p(avg_bio_linear)
            aux_out = avg_aux 
            species_logits = torch.log(avg_sp_probs + 1e-9)
            taxonomy_logits = torch.log(avg_tax_probs + 1e-9)
            
        else:
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
        metrics[f'{prefix}_loss_int_mul']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B
        metrics[f'{prefix}_loss_int_add']  += nn.functional.mse_loss(aux_out[:, 3], aux_feats[:, 3]).item() * B

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
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tile_augmentation': 'enabled'
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    session_dir = setup_logging(file_name_part="tiled_stratified_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    
    logger.info("="*70)
    logger.info("TILE-BASED AUGMENTATION ENABLED")
    logger.info("Each training sample generates 6 views:")
    logger.info("  1x Original + 1x Stitched + 4x Divided Tiles")
    logger.info("Effective Training Set Size: N_samples × 6")
    logger.info("="*70)
    
    logger.info(config_str())
    
    # 1. Load Data
    df = load_data(logger)
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    df = add_cv_group(df)
    logger.info("Data sorted by Sampling_Date and grouped by State+Date (cv_group).")
    
    species_list = CORE_SPECIES
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    save_metadata(session_dir, species_list, target_cols)

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # 2. Strict temporal holdout by State+Date group (cv_group)
    unique_groups = (
        df[['cv_group', 'Sampling_Date']]
        .drop_duplicates()
        .sort_values('Sampling_Date')
        .reset_index(drop=True)
    )
    cutoff_gidx = int(len(unique_groups) * (1.0 - configs.SPLIT_CONFIG.get('holdout_pct', 0.15)))
    dev_groups = set(unique_groups.iloc[:cutoff_gidx]['cv_group'])
    holdout_groups = set(unique_groups.iloc[cutoff_gidx:]['cv_group'])

    dev_df = df[df['cv_group'].isin(dev_groups)].copy().reset_index(drop=True)
    global_holdout_df = df[df['cv_group'].isin(holdout_groups)].copy().reset_index(drop=True)

    logger.info(f"\n{'='*40}")
    logger.info("STRICT TEMPORAL HOLDOUT (grouped by State+Date)")
    logger.info(f"{'='*40}")
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Development Set: {len(dev_df)} ({dev_df['Sampling_Date'].min().date()} -> {dev_df['Sampling_Date'].max().date()})")
    logger.info(f"Global Holdout:  {len(global_holdout_df)} ({global_holdout_df['Sampling_Date'].min().date()} -> {global_holdout_df['Sampling_Date'].max().date()})")

    global_holdout_df.to_csv(os.path.join(splits_dir, "global_holdout.csv"), index=False)
    
    # 3. Prepare Data Transforms
    train_transform, val_transform = get_image_data_transforms()
    
    # 4. Holdout Dataset (NO TILING)
    holdout_ds = TiledBiomassDataset(
        global_holdout_df, 
        transform=val_transform,
        mode='holdout',  # Disables tiling
        tile_prob=0.0
    )
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 5. Group-wise Time-Series Split on Development Set
    group_index_df = (
        dev_df[['cv_group', 'Sampling_Date']]
        .drop_duplicates()
        .sort_values('Sampling_Date')
        .reset_index(drop=True)
    )
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    
    best_overall_score = -float('inf')
    stratification_col = 'StratifyKey'
    
    for fold, (g_train_idx, g_val_idx) in enumerate(tscv.split(group_index_df)):
        train_groups = set(group_index_df.iloc[g_train_idx]['cv_group'])
        val_groups = set(group_index_df.iloc[g_val_idx]['cv_group'])
        
        train_df = dev_df[dev_df['cv_group'].isin(train_groups)].copy().reset_index(drop=True)
        val_df = dev_df[dev_df['cv_group'].isin(val_groups)].copy().reset_index(drop=True)
        raw_n_train = len(train_df)
        
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")
        logger.info(f"Train: {train_df['Sampling_Date'].min().date()} -> {train_df['Sampling_Date'].max().date()} (n={len(train_df)})")
        logger.info(f"Val:   {val_df['Sampling_Date'].min().date()} -> {val_df['Sampling_Date'].max().date()} (n={len(val_df)})")
        
        log_fold_details(logger, train_df, val_df)

        if raw_n_train < MIN_TRAIN_SAMPLES:
            logger.info(f"\nSkipping Fold {fold+1}: Training set too small ({raw_n_train} < {MIN_TRAIN_SAMPLES})")
            continue
 
        # Upsampling
        logger.info("\nUpsampling Train Set")
        logger.info(f"Before Upsampling: {train_df['StratifyKey'].value_counts()}")
        train_df = smart_upsample(train_df, stratify_col='StratifyKey')
        logger.info(f"After Upsampling: {train_df['StratifyKey'].value_counts()}")
        
        # Calculate effective training size with tiling
        effective_train_size = len(train_df) * 6  # 6 views per sample
        logger.info(f"\n{'='*40}")
        logger.info(f"EFFECTIVE TRAINING SIZE WITH TILING")
        logger.info(f"Base Samples: {len(train_df)}")
        logger.info(f"With 6x Tile Augmentation: {effective_train_size}")
        logger.info(f"{'='*40}\n")
        
        # Save Fold Splits
        train_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_val.csv"), index=False)
        
        # ===== KEY CHANGE: Use TiledBiomassDataset =====
        train_ds_base = TiledBiomassDataset(
            train_df, 
            transform=train_transform,
            mode='training',  # Enables tiling
            tile_prob=TILE_PROB
        )
        train_ds = TiledMixupDataset(train_ds_base, prob=MIXUP_PROB, alpha=MIXUP_ALPHA)
        
        val_ds = TiledBiomassDataset(
            val_df, 
            transform=val_transform,
            mode='validation',  # Disables tiling
            tile_prob=0.0
        )
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation')
        n_aux = dummy_ds[0]['aux_feats'].shape[0]
        
        model = BiomassUnifiedModel(num_species=len(species_list), num_aux=n_aux).to(DEVICE)
        
        # Backbone Protection Logic
        n_upsampled = len(train_df)
        if n_upsampled < BACKBONE_FREEZE_THRESHOLD:
            logger.info(f"PROTECTION: Keeping backbone FROZEN for Fold {fold+1} (n_upsampled={n_upsampled} < {BACKBONE_FREEZE_THRESHOLD})")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            if FREEZE_BACKBONE:
                logger.info(f"STRATEGY: Applying Partial Freeze ({BACKBONE_FREEZE_FRACTION*100}%) for Fold {fold+1} (n_upsampled={n_upsampled})")
                all_params = list(model.backbone.parameters())
                freeze_until = int(len(all_params) * BACKBONE_FREEZE_FRACTION)
                for i, p in enumerate(all_params):
                    p.requires_grad = (i >= freeze_until)
            else:
                logger.info(f"STRATEGY: Full Backbone Unfreeze for Fold {fold+1}")
                for param in model.backbone.parameters():
                    param.requires_grad = True
        
        # Differential Learning Rates
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        
        param_groups = [
            {'params': backbone_params, 'lr': LEARNING_RATE * BACKBONE_LR_FACTOR},
            {'params': head_params, 'lr': LEARNING_RATE}
        ]
        
        optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-3, min_lr=1e-6)
        
        criterion_reg = nn.MSELoss() 
        criterion_ce = nn.CrossEntropyLoss()
        
        history = defaultdict(list)
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_h_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_ce, 
                DEVICE, epoch, session_dir=session_dir, logger=logger
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion_reg, criterion_ce, DEVICE, prefix='val', use_tta=False)
            
            # Holdout
            hol_metrics = validate(
                model, holdout_loader, criterion_reg, criterion_ce, DEVICE, 
                prefix='holdout', use_tta=USE_TTA,
                epoch=epoch, fold=fold, session_dir=session_dir
            )
            
            # Custom Score
            v_r2 = val_metrics['val_r2']
            h_r2 = hol_metrics['holdout_r2']
            
            avg_r2 = (v_r2 + h_r2) / 2
            consistency_penalty = 0.5 * abs(v_r2 - h_r2)
            current_score = avg_r2 - consistency_penalty
            
            score_gap = abs(v_r2 - h_r2)
            scheduler.step(current_score)
            
            log_msg = (f"Ep {epoch} | T_Loss: {train_metrics['train_loss']:.3f} | V_Loss: {val_metrics['val_loss']:.3f} | H_Loss: {hol_metrics['holdout_loss']:.3f} | "
                       f"T_R2: {train_metrics['train_r2']:.4f} | V_R2: {v_r2:.4f} | H_R2: {h_r2:.4f} | "
                       f"Score: {current_score:.4f} | Gap: {score_gap:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
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
                best_fold_v_r2 = v_r2
                best_fold_h_r2 = h_r2
                best_fold_epoch = epoch
                
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
        
        # End of Fold Summary
        logger.info(f"\n[Fold {fold+1} COMPLETE]")
        logger.info(f"Best Score: {best_fold_score:.4f} (at Epoch {best_fold_epoch})")
        logger.info(f"Best Val R2: {best_fold_v_r2:.4f}")
        logger.info(f"Best Holdout R2: {best_fold_h_r2:.4f}")
        logger.info("-" * 40)

if __name__ == '__main__':
    main()