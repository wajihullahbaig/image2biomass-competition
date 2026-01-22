
import os
import logging
import shutil
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json

from config.loader import cfg, yaml_path
from configs import config_str
from common import (
    load_data, get_image_data_transforms, save_batch_images,
    save_hsv_mask_batch, set_seed, calculate_global_weighted_r2,
    rotate_crop_resize, save_tta_images, build_weighted_sampler_from_df
)
from feature_transform import BiomassFeatureTransform, apply_deterministic_features
from log_and_plots import (
    get_formatted_loss_log, log_dataframe_details, setup_logging, plot_training_history,
    log_species_table, log_fold_summary_tables, log_aggregate_best_across_folds,
    log_fold_details
)
from dataset import TiledBiomassDataset, TiledMixupDataset
from models import BiomassUnifiedModel

# --- Custom Scoring Logic (Local to this script for full control) ---
def calculate_cv_score(train_r2, val_r2, ema_score_prev=None, ema_decay=0.9):
    """
    Score = Val_R2 - Penalty(Overfitting)
    More lenient penalty to allow learning on small datasets.
    """
    # 1. Base Score is Validation R2
    current_score = val_r2
    
    # 2. Overfitting Penalty (VERY RELAXED for small datasets)
    gap = train_r2 - val_r2
    
    # Only penalize if gap is VERY large and val_r2 is actually good
    # This prevents penalizing when model is still learning
    if val_r2 > 0.15 and gap > 0.15:  # Increased threshold
        penalty = (gap - 0.15) * 0.2  # Reduced penalty factor
        current_score -= penalty
        print(f"Applying overfitting penalty: {penalty:.4f}")
    
    # 3. EMA Smoothing (less aggressive for small datasets)
    if ema_score_prev is None:
        ema_score = current_score
    else:
        # Use less smoothing for faster adaptation
        ema_score = 0.7 * ema_score_prev + 0.3 * current_score
        
    return ema_score, gap, current_score


def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_species, cfg, epoch, session_dir=None, logger=None,
                    bio_mean=None, bio_std=None, aux_mean=None, aux_std=None, official_weights_t=None):
    model.train()
    metrics = defaultdict(float)
    scaler = torch.amp.GradScaler('cuda')
    all_preds_log = []
    all_targets_full = []

    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(cfg.device)
        targets_g = batch['targets'].to(cfg.device)

        if torch.isnan(targets_g).any():
            if logger: logger.warning(f"NaN TARGETS DETECTED in batch {batch_idx}. Skipping.")
            continue

        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        # Detailed Logging: Save HSV masks and images for debugging
        if epoch == 0 and batch_idx < 5 and session_dir:
            save_batch_images(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)
            save_hsv_mask_batch(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            biomass_out, aux_out, species_logits = model(images)

            targ_green_log = torch.log1p(targets_g[:, 0:1])
            targ_dead_log  = torch.log1p(targets_g[:, 1:2])
            targ_clover_log = torch.log1p(targets_g[:, 2:3])
            targ_gdm_log   = torch.log1p(targets_g[:, 3:4])
            targ_total_log = torch.log1p(targets_g[:, 4:5])

            reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
            
            p_green, t_green = biomass_out[:, 0:1], targ_green_log
            p_dead,  t_dead  = biomass_out[:, 1:2], targ_dead_log
            p_clover, t_clover = biomass_out[:, 2:3], targ_clover_log
            p_gdm,   t_gdm   = biomass_out[:, 3:4], targ_gdm_log
            p_total, t_total = biomass_out[:, 4:5], targ_total_log
            
            if cfg.loss.use_standardized_loss and bio_mean is not None:
                eps = 1e-9
                p_green = (p_green - bio_mean[:, 0:1]) / (bio_std[:, 0:1] + eps)
                t_green = (t_green - bio_mean[:, 0:1]) / (bio_std[:, 0:1] + eps)
                p_dead  = (p_dead  - bio_mean[:, 1:2]) / (bio_std[:, 1:2] + eps)
                t_dead  = (t_dead  - bio_mean[:, 1:2]) / (bio_std[:, 1:2] + eps)
                p_clover= (p_clover- bio_mean[:, 2:3]) / (bio_std[:, 2:3] + eps)
                t_clover= (t_clover- bio_mean[:, 2:3]) / (bio_std[:, 2:3] + eps)
                p_gdm   = (p_gdm   - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + eps)
                t_gdm   = (t_gdm   - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + eps)
                p_total = (p_total - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + eps)
                t_total = (t_total - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + eps)

            l_green  = reg(p_green, t_green)
            l_dead   = reg(p_dead,  t_dead)
            l_clover = reg(p_clover, t_clover)
            l_gdm    = reg(p_gdm,   t_gdm)
            l_total  = reg(p_total, t_total)
            
            if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
                l_green  = l_green  * official_weights_t[0]
                l_dead   = l_dead   * official_weights_t[1]
                l_clover = l_clover * official_weights_t[2]
                l_gdm    = l_gdm    * official_weights_t[3]
                l_total  = l_total  * official_weights_t[4]
            
            loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

            # --- Auxiliary Loss ---
            p_aux, t_aux = aux_out, aux_feats
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                eps = 1e-9
                p_aux = (p_aux - aux_mean) / (aux_std + eps)
                t_aux = (t_aux - aux_mean) / (aux_std + eps)
            loss_aux = nn.MSELoss()(p_aux, t_aux) * cfg.training.aux_feat_weight

            # --- Species Loss ---
            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight

            total_loss = loss_bio + loss_aux + loss_sp

        if torch.isnan(total_loss):
            optimizer.zero_grad()
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio']  += loss_bio.item() * B
        metrics['train_aux']  += loss_aux.item() * B
        metrics['train_sp']   += loss_sp.item() * B
        
        metrics['train_loss_green']  += l_green.item() * B
        metrics['train_loss_dead']   += l_dead.item() * B
        metrics['train_loss_clover'] += l_clover.item() * B
        metrics['train_loss_gdm']    += l_gdm.item() * B
        metrics['train_loss_total']  += l_total.item() * B
        
        # Flexible auxiliary feature loss tracking (NDVI, Height, Interactions, HSV)
        aux_feature_names = ['ndvi', 'height_log', 'interaction_mul', 'interaction_add', 'species_count',
                           'green_hsv', 'dead_hsv', 'clover_hsv', 'soil_hsv']
        for i in range(min(aux_out.shape[1], len(aux_feature_names))):
            feature_name = aux_feature_names[i]
            metrics[f'train_loss_{feature_name}'] += nn.functional.mse_loss(aux_out[:, i], aux_feats[:, i]).item() * B
        
        # Legacy/Shortcut HSV tracking
        if 'train_loss_green_hsv' in metrics:
            metrics['train_loss_hsv'] = metrics['train_loss_green_hsv']
        
        with torch.no_grad():
            all_preds_log.append(biomass_out.cpu().numpy())
            all_targets_full.append(torch.cat([targ_green_log, targ_dead_log, targ_clover_log, targ_gdm_log, targ_total_log], dim=1).cpu().numpy())

        pbar.set_postfix({'L': total_loss.item()})

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    preds_linear = np.expm1(preds_log)
    targets_log = np.concatenate(all_targets_full)
    targets_linear = np.expm1(targets_log)
    final_metrics['train_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, cfg.targets.official_weights)
    return final_metrics


@torch.no_grad()
def validate(model, loader, criterion_reg, criterion_species, cfg, prefix='val', use_tta=False, epoch=0, fold=0, session_dir=None,
             bio_mean=None, bio_std=None, aux_mean=None, aux_std=None, official_weights_t=None):
    model.eval()
    metrics = defaultdict(float)
    all_preds_log, all_targets_full = [], []

    tta_views = [
        ('id', lambda x: x), ('hflip', lambda x: torch.flip(x, [3])), ('vflip', lambda x: torch.flip(x, [2])),
        ('rot5', lambda x: rotate_crop_resize(x, 5)), ('rot-5', lambda x: rotate_crop_resize(x, -5)),
    ]

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(cfg.device)
        targets_g = batch['targets'].to(cfg.device)
        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        if use_tta:
            accum_bio_linear = 0
            accum_aux = 0
            accum_sp_probs = 0
            for view_name, transform_fn in tta_views:
                img_aug = transform_fn(images)
                if session_dir is not None and epoch % 5 == 0: # Save infrequently for TTA
                    save_tta_images(img_aug, view_name, batch_idx, fold, epoch, session_dir)
                bio_out_tta, aux_out_tta, sp_logits_tta = model(img_aug)
                accum_bio_linear += torch.expm1(bio_out_tta)
                accum_aux += aux_out_tta
                accum_sp_probs += torch.sigmoid(sp_logits_tta)
            
            biomass_out = torch.log1p(accum_bio_linear / len(tta_views))
            aux_out = accum_aux / len(tta_views)
            species_probs = accum_sp_probs / len(tta_views)
        else:
            biomass_out, aux_out, species_logits = model(images)

        targ_green_log = torch.log1p(targets_g[:, 0:1])
        targ_dead_log  = torch.log1p(targets_g[:, 1:2])
        targ_clover_log = torch.log1p(targets_g[:, 2:3])
        targ_gdm_log   = torch.log1p(targets_g[:, 3:4])
        targ_total_log = torch.log1p(targets_g[:, 4:5])

        reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
        
        p_green, t_green = biomass_out[:, 0:1], targ_green_log
        p_dead,  t_dead  = biomass_out[:, 1:2], targ_dead_log
        p_clover, t_clover = biomass_out[:, 2:3], targ_clover_log
        p_gdm,   t_gdm   = biomass_out[:, 3:4], targ_gdm_log
        p_total, t_total = biomass_out[:, 4:5], targ_total_log
        
        if cfg.loss.use_standardized_loss and bio_mean is not None:
            eps = 1e-9
            p_green = (p_green - bio_mean[:, 0:1]) / (bio_std[:, 0:1] + eps)
            t_green = (t_green - bio_mean[:, 0:1]) / (bio_std[:, 0:1] + eps)
            p_dead  = (p_dead  - bio_mean[:, 1:2]) / (bio_std[:, 1:2] + eps)
            t_dead  = (t_dead  - bio_mean[:, 1:2]) / (bio_std[:, 1:2] + eps)
            p_clover= (p_clover- bio_mean[:, 2:3]) / (bio_std[:, 2:3] + eps)
            t_clover= (t_clover- bio_mean[:, 2:3]) / (bio_std[:, 2:3] + eps)
            p_gdm   = (p_gdm   - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + eps)
            t_gdm   = (t_gdm   - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + eps)
            p_total = (p_total - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + eps)
            t_total = (t_total - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + eps)

        l_green  = reg(p_green, t_green)
        l_dead   = reg(p_dead,  t_dead)
        l_clover = reg(p_clover, t_clover)
        l_gdm    = reg(p_gdm,   t_gdm)
        l_total  = reg(p_total, t_total)
        
        if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
            l_green  = l_green  * official_weights_t[0]
            l_dead   = l_dead   * official_weights_t[1]
            l_clover = l_clover * official_weights_t[2]
            l_gdm    = l_gdm    * official_weights_t[3]
            l_total  = l_total  * official_weights_t[4]
        
        loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight
        loss_aux = nn.MSELoss()(aux_out, aux_feats) * cfg.training.aux_feat_weight
        
        if use_tta:
            loss_sp = nn.BCELoss()(species_probs, species_vec) * cfg.training.species_feat_weight
        else:
            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight

        total_loss = loss_bio + loss_aux + loss_sp

        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_sp']  += loss_sp.item() * B
        metrics[f'{prefix}_loss_green']  += l_green.item() * B
        metrics[f'{prefix}_loss_dead']   += l_dead.item() * B
        metrics[f'{prefix}_loss_clover'] += l_clover.item() * B
        metrics[f'{prefix}_loss_gdm']    += l_gdm.item() * B
        metrics[f'{prefix}_loss_total']  += l_total.item() * B

        # Flexible auxiliary feature loss tracking (NDVI, Height, Interactions, HSV)
        aux_feature_names = ['ndvi', 'height_log', 'interaction_mul', 'interaction_add', 'species_count',
                           'green_hsv', 'dead_hsv', 'clover_hsv', 'soil_hsv']
        for i in range(min(aux_out.shape[1], len(aux_feature_names))):
            feature_name = aux_feature_names[i]
            metrics[f'{prefix}_loss_{feature_name}'] += nn.functional.mse_loss(aux_out[:, i], aux_feats[:, i]).item() * B
        
        # Legacy/Shortcut HSV tracking
        if f'{prefix}_loss_green_hsv' in metrics:
            metrics[f'{prefix}_loss_hsv'] = metrics[f'{prefix}_loss_green_hsv']

        with torch.no_grad():
            all_preds_log.append(biomass_out.cpu().numpy())
            all_targets_full.append(torch.cat([targ_green_log, targ_dead_log, targ_clover_log, targ_gdm_log, targ_total_log], dim=1).cpu().numpy())

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    preds_linear = np.expm1(preds_log)
    targets_log = np.concatenate(all_targets_full)
    targets_linear = np.expm1(targets_log)
    final_metrics[f'{prefix}_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, cfg.targets.official_weights)
    return final_metrics

def main():
    session_dir = setup_logging(file_name_part="cv_stratified")
    logger = logging.getLogger("System Logger")
    set_seed(313, logger)
    shutil.copy(yaml_path, os.path.join(session_dir, 'used_config.yaml'))
    logger.info("="*70)
    logger.info("FULL STRATIFIED GROUP K-FOLD TRAINING (NO HOLDOUT)")
    logger.info("="*70)
    logger.info(config_str())

    # 1. Load Data
    df = load_data(logger)
    df = apply_deterministic_features(df)
    df.to_csv('wide_safe.csv', index=False)
    logger.info("Saved deterministic features to wide_safe.csv")

    train_transform, val_transform = get_image_data_transforms()

    # 2. Stratified Group K-Fold Setup
    n_folds = cfg.hyperparameters.n_folds
    strat_col = cfg.split.group_stratification_col  # e.g. "State"
    group_col = cfg.split.group_col                 # e.g. "SessionID"

    # ===== FIX 1: Robust stratification column handling =====
    available_cols = df.columns.tolist()
    logger.info(f"Available columns for stratification: {available_cols}")
    
    if strat_col not in df.columns:
        logger.warning(f"Stratification col '{strat_col}' not found in df. Searching for alternatives...")
        
        # Try alternatives in order of preference
        alternatives = ['State_Species', 'Season_State_Species', 'State', 'Species', 'Season']
        found = False
        for alt in alternatives:
            if alt in df.columns:
                strat_col = alt
                logger.info(f"✓ Using '{strat_col}' for stratification")
                found = True
                break
        
        if not found:
            logger.error(f"No suitable stratification column found in: {available_cols}")
            raise ValueError("Cannot proceed without stratification column")
    else:
        logger.info(f"✓ Using configured stratification column: '{strat_col}'")

    # Verify group column exists
    if group_col not in df.columns:
        logger.error(f"Group col '{group_col}' not found in columns: {available_cols}")
        raise ValueError(f"Group column {group_col} must exist for GroupKFold")
    else:
        logger.info(f"✓ Using group column: '{group_col}'")

    # Check for null values in critical columns
    null_strat = df[strat_col].isnull().sum()
    null_group = df[group_col].isnull().sum()
    if null_strat > 0:
        logger.warning(f"Found {null_strat} null values in {strat_col}. Filling with 'Unknown'")
        df[strat_col] = df[strat_col].fillna('Unknown')
    if null_group > 0:
        logger.error(f"Found {null_group} null values in {group_col}. Cannot proceed!")
        raise ValueError(f"Group column {group_col} contains null values")

    # Create stratification labels if needed
    logger.info(f"Stratification column '{strat_col}' value counts:")
    logger.info(f"\n{df[strat_col].value_counts()}")
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=313)
    splits = list(sgkf.split(df, df[strat_col], groups=df[group_col]))

    oof_preds = np.zeros((len(df), 5))
    oof_targets = np.zeros((len(df), 5))
    validation_mask = np.zeros(len(df), dtype=bool)
    per_fold_best = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*30} FOLD {fold+1}/{n_folds} {'='*30}")
        
        train_df_fold = df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = df.iloc[val_idx].reset_index(drop=True)
        
        # Fit Feature Transform on TRAIN ONLY
        ft = BiomassFeatureTransform(logger)
        train_df = ft.fit(train_df_fold)
        val_df = ft.transform(val_df_fold)
        
        # ===== FIX 2: Verify critical columns preserved =====
        critical_cols = [group_col, strat_col, 'Species', 'State']
        for col in critical_cols:
            if col not in train_df.columns and col in train_df_fold.columns:
                logger.warning(f"Column '{col}' was lost during transform! Restoring...")
                train_df[col] = train_df_fold[col].values
            if col not in val_df.columns and col in val_df_fold.columns:
                logger.warning(f"Column '{col}' was lost during transform! Restoring...")
                val_df[col] = val_df_fold[col].values
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
        logger.info(f"Train Groups ({group_col}): {train_df[group_col].nunique()}")
        logger.info(f"Val Groups ({group_col}): {val_df[group_col].nunique()}")
        
        # Verify No Leakage
        train_groups = set(train_df[group_col])
        val_groups = set(val_df[group_col])
        leakage = train_groups.intersection(val_groups)
        if leakage:
            logger.error(f"CRITICAL: GROUP LEAKAGE DETECTED! {len(leakage)} groups shared: {leakage}")
            raise ValueError("Group Leakage Detected")
        else:
            logger.info("✓ No group leakage detected")

        # Datasets
        train_ds_base = TiledBiomassDataset(
            train_df, transform=train_transform, mode='training', 
            tile_prob=cfg.augmentation.tile_prob,
            target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        )
        train_ds = TiledMixupDataset(train_ds_base, prob=cfg.augmentation.mixup_prob, alpha=cfg.augmentation.mixup_alpha)
        
        val_ds = TiledBiomassDataset(
            val_df, transform=val_transform, mode='validation', tile_prob=0.0,
            target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        )
        
        train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0)
        
        # Model
        n_aux = train_ds_base[0]['aux_feats'].shape[0]
        model = BiomassUnifiedModel(num_aux=n_aux, config=cfg).to(cfg.device)
        
        # ===== FIX 3: More aggressive backbone unfreezing =====
        n_upsampled = len(train_df)
        
        if n_upsampled < 200:
            # Very small dataset: freeze 90% of backbone
            freeze_frac = 0.9
            logger.info(f"PROTECTION: Freezing {freeze_frac*100}% of backbone (n={n_upsampled} < 200)")
            all_params = list(model.backbone.parameters())
            freeze_until = int(len(all_params) * freeze_frac)
            for i, p in enumerate(all_params):
                p.requires_grad = (i >= freeze_until)
            logger.info(f"  → {len(all_params) - freeze_until}/{len(all_params)} backbone params trainable")
                
        elif n_upsampled < cfg.hyperparameters.backbone_freeze_threshold:
            # Small dataset: freeze 70% of backbone (not 100%!)
            freeze_frac = 0.7
            logger.info(f"STRATEGY: Freezing {freeze_frac*100}% of backbone (n={n_upsampled} < {cfg.hyperparameters.backbone_freeze_threshold})")
            all_params = list(model.backbone.parameters())
            freeze_until = int(len(all_params) * freeze_frac)
            for i, p in enumerate(all_params):
                p.requires_grad = (i >= freeze_until)
            logger.info(f"  → {len(all_params) - freeze_until}/{len(all_params)} backbone params trainable")
                
        else:
            # Larger dataset: use config setting
            if cfg.training.freeze_backbone:
                logger.info(f"STRATEGY: Applying Partial Freeze ({cfg.training.backbone_freeze_fraction*100}%) for Fold {fold+1}")
                all_params = list(model.backbone.parameters())
                freeze_until = int(len(all_params) * cfg.training.backbone_freeze_fraction)
                for i, p in enumerate(all_params):
                    p.requires_grad = (i >= freeze_until)
                logger.info(f"  → {len(all_params) - freeze_until}/{len(all_params)} backbone params trainable")
            else:
                logger.info(f"STRATEGY: Full Backbone Unfreeze for Fold {fold+1}")
                for param in model.backbone.parameters():
                    param.requires_grad = True
        
        # Differential Learning Rates
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        
        # Count trainable params
        n_trainable_backbone = sum(p.numel() for p in backbone_params if p.requires_grad)
        n_trainable_head = sum(p.numel() for p in head_params if p.requires_grad)
        logger.info(f"Trainable params: Backbone={n_trainable_backbone:,}, Heads={n_trainable_head:,}")
        
        param_groups = [
            {'params': [p for p in backbone_params if p.requires_grad], 
             'lr': cfg.hyperparameters.learning_rate * cfg.hyperparameters.backbone_lr_factor},
            {'params': head_params, 'lr': cfg.hyperparameters.learning_rate}
        ]
        
        optimizer = AdamW(param_groups, weight_decay=cfg.hyperparameters.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, threshold=1e-3)
        
        # Stats
        bio_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        bio_train_log = np.log1p(train_df[bio_cols].astype(float).values)
        bio_mean_t = torch.tensor(bio_train_log.mean(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        bio_std_t = torch.tensor(bio_train_log.std(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        
        official_weights_t = torch.tensor(cfg.targets.official_weights, dtype=torch.float32, device=cfg.device)

        # Log species distribution (safe version)
        try:
            if 'Species' in train_df.columns and 'Species' in val_df.columns:
                log_species_table(logger, train_df, val_df, pd.DataFrame(), species_col='Species', title='Species in Fold')
            else:
                logger.info(f"Species column not available for logging")
        except Exception as e:
            logger.warning(f"Could not log species table: {e}")
        
        log_fold_details(logger, train_df, val_df)

        # Training Loop
        best_fold_score = -float('inf')
        best_fold_model_path = os.path.join(session_dir, f"best_model_fold{fold+1}.pth")
        ema_score_prev = None
        history = defaultdict(list)
        patience_counter = 0

        for epoch in range(cfg.hyperparameters.epochs):
            train_mets = train_one_epoch(
                model, train_loader, optimizer, None, None, cfg, epoch, 
                session_dir=session_dir, logger=logger,
                bio_mean=bio_mean_t, bio_std=bio_std_t, official_weights_t=official_weights_t
            )
            
            val_mets = validate(
                model, val_loader, None, None, cfg, prefix='val', epoch=epoch, fold=fold+1, session_dir=session_dir,
                bio_mean=bio_mean_t, bio_std=bio_std_t, official_weights_t=official_weights_t
            )
            
            t_r2 = train_mets['train_r2']
            v_r2 = val_mets['val_r2']
            
            # ===== FIX 4: More lenient scoring =====
            ema_score, gap, score_raw = calculate_cv_score(t_r2, v_r2, ema_score_prev=ema_score_prev)
            ema_score_prev = ema_score
            
            scheduler.step(ema_score)
            
            log_msg = get_formatted_loss_log(epoch, train_mets, val_mets, {}, ema_score, gap, optimizer.param_groups[0]['lr'], v_r2, 0.0)
            logger.info(log_msg)
            
            # Update history
            for k, v in train_mets.items(): history[k].append(v)
            for k, v in val_mets.items(): history[k].append(v)
            history['score'].append(ema_score)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Generate plots
            plot_training_history(history, fold+1, session_dir)
            
            # Save Best
            if ema_score > best_fold_score:
                best_fold_score = ema_score
                best_fold_epoch = epoch
                torch.save(model.state_dict(), best_fold_model_path)
                logger.info(f"*** Fold {fold+1} Improved Score: {best_fold_score:.4f} (Val R2: {v_r2:.4f}, Gap: {gap:.4f}) ***")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= cfg.hyperparameters.early_stop_patience:
                logger.info(f"Early Stopping triggered at epoch {epoch}")
                break
        
        # Log fold summary
        log_fold_summary_tables(logger, fold+1, history, best_fold_epoch)
        per_fold_best.append({
            'best_score': best_fold_score,
            'best_val_r2': history['val_r2'][best_fold_epoch],
            'best_epoch': best_fold_epoch
        })
        
        # Generate OOF predictions
        logger.info(f"Reloading best model for Fold {fold+1} to generate OOF predictions...")
        model.load_state_dict(torch.load(best_fold_model_path))
        model.eval()
        
        fold_probs = []
        fold_targs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(cfg.device)
                targs = batch['targets'].to(cfg.device)
                preds, _, _ = model(imgs)
                fold_probs.append(preds.cpu().numpy())
                fold_targs.append(targs.cpu().numpy())
        
        fold_probs = np.concatenate(fold_probs)
        fold_targs = np.concatenate(fold_targs)
        
        oof_preds[val_idx] = np.expm1(fold_probs)
        oof_targets[val_idx] = fold_targs
        validation_mask[val_idx] = True
        
        plot_training_history(history, fold+1, session_dir)
    
    # Global CV Score
    if validation_mask.sum() < len(df):
        logger.warning(f"Only {validation_mask.sum()}/{len(df)} samples were validated.")
    
    final_oof_preds = oof_preds[validation_mask]
    final_oof_targets = oof_targets[validation_mask]
    
    global_r2 = calculate_global_weighted_r2(final_oof_targets, final_oof_preds, cfg.targets.official_weights)
    
    # Save OOF
    oof_df = df[validation_mask].copy()
    for i, col in enumerate(['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']):
        oof_df[f'Pred_{col}'] = final_oof_preds[:, i]
    
    oof_df.to_csv(os.path.join(session_dir, 'oof_predictions.csv'), index=False)
    logger.info(f"OOF predictions saved. GLOBAL OOF R2: {global_r2:.4f}")

    log_aggregate_best_across_folds(logger, per_fold_best)
    
    logger.info("="*70)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info(f"Session Dir: {session_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()