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

from config.loader import cfg,yaml_path
from configs import config_str
from common import (
    calculate_scheduler_score, get_season, load_data, get_image_data_transforms, save_batch_images,
    save_hsv_mask_batch,
    set_seed, calculate_global_weighted_r2,
    rotate_crop_resize, save_tta_images, build_weighted_sampler_from_df
)
from feature_transform import BiomassFeatureTransform, apply_deterministic_features

from log_and_plots import (
    get_formatted_loss_log, log_dataframe_details, setup_logging, plot_training_history,
    log_fold_details, log_species_table
)

from log_and_plots import log_fold_summary_tables, log_aggregate_best_across_folds

from dataset import TiledBiomassDataset, TiledMixupDataset
from models import BiomassUnifiedModel


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
            if logger:
                logger.warning(f"NaN TARGETS DETECTED in batch {batch_idx}. Skipping.")
            continue

        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        if epoch == 0 and batch_idx < 5 and session_dir:
            save_batch_images(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)
            save_hsv_mask_batch(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            # Model now returns [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total]
            biomass_out, aux_out, species_logits = model(images)

            # Targets are [Green, Dead, Clover, GDM, Total] in linear grams from dataset
            # Convert ALL to log space at once for efficiency
            targets_log = torch.log1p(targets_g)
            
            # Direct prediction/target pairs (Log Space)
            reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
            
            p_bio = biomass_out
            t_bio = targets_log
            
            if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
                p_bio = (p_bio - bio_mean) / (bio_std + 1e-9)
                t_bio = (t_bio - bio_mean) / (bio_std + 1e-9)

            # Individual component losses from the standardized/processed tensors
            l_green  = reg(p_bio[:, 0:1], t_bio[:, 0:1])
            l_dead   = nn.HuberLoss(delta=cfg.training.huber_delta)(p_bio[:, 1:2], t_bio[:, 1:2]) if cfg.training.use_huber_loss_for_dead else reg(p_bio[:, 1:2], t_bio[:, 1:2])
            l_clover = reg(p_bio[:, 2:3], t_bio[:, 2:3])
            l_gdm    = reg(p_bio[:, 3:4], t_bio[:, 3:4])
            l_total  = reg(p_bio[:, 4:5], t_bio[:, 4:5])
            
            if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
                l_green  = l_green  * official_weights_t[0]
                l_dead   = l_dead   * official_weights_t[1]
                l_clover = l_clover * official_weights_t[2]
                l_gdm    = l_gdm    * official_weights_t[3]
                l_total  = l_total  * official_weights_t[4]
            
            loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

            # --- Physical Consistency Loss (Log-Space for Stability) ---
            # Use RAW model outputs (log-scale) for physics checks to avoid standardized distortions
            raw_green, raw_dead, raw_clover, raw_gdm, raw_total = [biomass_out[:, i:i+1] for i in range(5)]
            
            lin_green, lin_dead, lin_clover, lin_gdm, lin_total = [torch.expm1(v) for v in [raw_green, raw_dead, raw_clover, raw_gdm, raw_total]]
            
            # Predict sums in linear space, then evaluate penalty in log space
            p_gdm_sum_log = torch.log1p(torch.clamp(lin_green + lin_clover, min=1e-4))
            p_total_sum_log = torch.log1p(torch.clamp(lin_gdm + lin_dead, min=1e-4))
            
            # Use separate consistency weight if available (fallback to bio weight)
            cons_weight = getattr(cfg.training, 'consistency_weight', cfg.training.biomass_feat_weight * 0.1)
            
            l_consistency_gdm = reg(raw_gdm, p_gdm_sum_log)
            l_consistency_tot = reg(raw_total, p_total_sum_log)
            
            # --- Dead HSV Constraint (Strategy 2) ---
            # dead_hsv is at index 8 of aux_feats (visible fraction 0.0-1.0)
            t_dead_hsv = aux_feats[:, 8:9]
            
            # min_dead_g = visible_fraction * k (k=17.5 as found in EDA y=17.3x + 10.0)
            k_scaling = getattr(cfg.training, 'dead_hsv_min_k', 17.5)
            min_dead_lin = t_dead_hsv * k_scaling
            
            # Convert the requirement to log-space so it matches the magnitude of other losses
            # We compare it against the RAW (unstandardized) log prediction: biomass_out[:, 1:2]
            p_dead_log = biomass_out[:, 1:2]
            min_dead_log = torch.log1p(min_dead_lin)
            
            # Penalty for predicting less than visible minimum (in log space)
            l_hsv_constraint = torch.mean(torch.nn.functional.relu(min_dead_log - p_dead_log))
            
            l_consistency_bio = (l_consistency_gdm + l_consistency_tot) * cons_weight + l_hsv_constraint * getattr(cfg.training, 'hsv_constraint_weight', 10.0)

            # --- Auxiliary Loss ---
            p_aux, t_aux = aux_out, aux_feats
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                eps = 1e-9
                p_aux = (p_aux - aux_mean) / (aux_std + eps)
                t_aux = (t_aux - aux_mean) / (aux_std + eps)
            loss_aux = nn.MSELoss()(p_aux, t_aux) * cfg.training.aux_feat_weight

            # --- Species Loss ---
            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight

            total_loss = loss_bio + loss_aux + loss_sp + l_consistency_bio

        if torch.isnan(total_loss):
            if logger:
                logger.warning(f"!!! NAN TOTAL LOSS at Ep {epoch}, batch {batch_idx} !!!")
            optimizer.zero_grad()
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # biomass_out is now [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total]
            preds_5_log = biomass_out
            targs_5_log = targets_log
            
            all_preds_log.append(preds_5_log.cpu().numpy())
            all_targets_full.append(targs_5_log.cpu().numpy())

        # Update metrics
        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio']  += loss_bio.item() * B
        metrics['train_aux']  += loss_aux.item() * B
        metrics['train_sp']   += loss_sp.item() * B
        metrics['train_cons'] += l_consistency_bio.item() * B
        metrics['train_loss_hsv'] += l_hsv_constraint.item() * B
        
        # Individual losses for tracking
        metrics['train_loss_green']  += l_green.item() * B
        metrics['train_loss_dead']   += l_dead.item() * B
        metrics['train_loss_clover'] += l_clover.item() * B
        metrics['train_loss_gdm']    += l_gdm.item() * B
        metrics['train_loss_total']  += l_total.item() * B
        
        # Flexible auxiliary feature loss tracking (flattened)
        aux_names = ['ndvi', 'height', 'i_mul', 'i_add', 'sp_count', 'g_hsv', 'dg_hsv', 'c_hsv', 'd_hsv', 's_hsv']
        for i in range(min(aux_out.shape[1], len(aux_names))):
            metrics[f"train_l_{aux_names[i]}"] += nn.functional.mse_loss(aux_out[:, i], aux_feats[:, i]).item() * B

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
        ('identity', lambda x: x),
        ('hflip', lambda x: torch.flip(x, [3])),
        ('vflip', lambda x: torch.flip(x, [2])),
        ('rot5', lambda x: rotate_crop_resize(x, 5)),
        ('rot-5', lambda x: rotate_crop_resize(x, -5)),
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
                if session_dir is not None:
                    save_tta_images(img_aug, view_name, batch_idx, fold, epoch, session_dir)
                # Model now returns [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total]
                bio_out_tta, aux_out_tta, sp_logits_tta = model(img_aug)
                accum_bio_linear += torch.expm1(bio_out_tta) # Sum linear predictions
                accum_aux += aux_out_tta
                accum_sp_probs += torch.sigmoid(sp_logits_tta)
            
            avg_bio_linear = accum_bio_linear / len(tta_views)
            biomass_out = torch.log1p(avg_bio_linear) # Convert back to log for loss
            aux_out = accum_aux / len(tta_views)
            species_probs = accum_sp_probs / len(tta_views)
        else:
            # Model now returns [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total]
            biomass_out, aux_out, species_logits = model(images)

        # Targets are [Green, Dead, Clover, GDM, Total] in linear grams from dataset
        targets_log = torch.log1p(targets_g)
        
        # Direct prediction/target pairs (Log Space)
        reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
        
        p_bio = biomass_out
        t_bio = targets_log
        
        if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
            p_bio = (p_bio - bio_mean) / (bio_std + 1e-9)
            t_bio = (t_bio - bio_mean) / (bio_std + 1e-9)

        # Individual component losses
        l_green  = reg(p_bio[:, 0:1], t_bio[:, 0:1])
        l_dead   = nn.HuberLoss(delta=cfg.training.huber_delta)(p_bio[:, 1:2], t_bio[:, 1:2]) if cfg.training.use_huber_loss_for_dead else reg(p_bio[:, 1:2], t_bio[:, 1:2])
        l_clover = reg(p_bio[:, 2:3], t_bio[:, 2:3])
        l_gdm    = reg(p_bio[:, 3:4], t_bio[:, 3:4])
        l_total  = reg(p_bio[:, 4:5], t_bio[:, 4:5])
        
        if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
            l_green  = l_green  * official_weights_t[0]
            l_dead   = l_dead   * official_weights_t[1]
            l_clover = l_clover * official_weights_t[2]
            l_gdm    = l_gdm    * official_weights_t[3]
            l_total  = l_total  * official_weights_t[4]
        
        loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

        # --- Physical Consistency Loss (Log-Space) ---
        # Use RAW model outputs (log-scale) for physics checks
        raw_green, raw_dead, raw_clover, raw_gdm, raw_total = [biomass_out[:, i:i+1] for i in range(5)]
        lin_green, lin_dead, lin_clover, lin_gdm, lin_total = [torch.expm1(v) for v in [raw_green, raw_dead, raw_clover, raw_gdm, raw_total]]
        
        p_gdm_sum_log = torch.log1p(torch.clamp(lin_green + lin_clover, min=1e-4))
        p_total_sum_log = torch.log1p(torch.clamp(lin_gdm + lin_dead, min=1e-4))
        
        cons_weight = getattr(cfg.training, 'consistency_weight', cfg.training.biomass_feat_weight * 0.1)
        
        l_consistency_gdm = reg(raw_gdm, p_gdm_sum_log)
        l_consistency_tot = reg(raw_total, p_total_sum_log)
        
        # Dead HSV Constraint (Log Space)
        t_dead_hsv = aux_feats[:, 8:9]
        k_scaling = getattr(cfg.training, 'dead_hsv_min_k', 17.5)
        min_dead_lin = t_dead_hsv * k_scaling
        p_dead_log = biomass_out[:, 1:2]
        min_dead_log = torch.log1p(min_dead_lin)
        l_hsv_constraint = torch.mean(torch.nn.functional.relu(min_dead_log - p_dead_log))
        
        l_consistency_bio = (l_consistency_gdm + l_consistency_tot) * cons_weight + l_hsv_constraint * getattr(cfg.training, 'hsv_constraint_weight', 10.0)

        # --- Auxiliary Loss ---
        p_aux, t_aux = aux_out, aux_feats
        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            eps = 1e-9
            p_aux = (p_aux - aux_mean) / (aux_std + eps)
            t_aux = (t_aux - aux_mean) / (aux_std + eps)
        loss_aux = criterion_reg(p_aux, t_aux) * cfg.training.aux_feat_weight

        # --- Species Loss ---
        if use_tta:
            loss_sp = nn.BCELoss()(species_probs, species_vec) * cfg.training.species_feat_weight
        else:
            loss_sp = criterion_species(species_logits, species_vec) * cfg.training.species_feat_weight

        total_loss = loss_bio + loss_aux + loss_sp + l_consistency_bio

        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_sp']  += loss_sp.item() * B
        metrics[f'{prefix}_cons'] += l_consistency_bio.item() * B
        metrics[f'{prefix}_loss_hsv'] += l_hsv_constraint.item() * B
        
        # Individual losses for tracking
        metrics[f'{prefix}_loss_green']  += l_green.item() * B
        metrics[f'{prefix}_loss_dead']   += l_dead.item() * B
        metrics[f'{prefix}_loss_clover'] += l_clover.item() * B
        metrics[f'{prefix}_loss_gdm']    += l_gdm.item() * B
        metrics[f'{prefix}_loss_total']  += l_total.item() * B

        # Flexible auxiliary feature loss tracking
        aux_names = ['ndvi', 'height', 'i_mul', 'i_add', 'sp_count', 'g_hsv', 'dg_hsv', 'c_hsv', 'd_hsv', 's_hsv']
        for i in range(min(aux_out.shape[1], len(aux_names))):
            metrics[f"{prefix}_l_{aux_names[i]}"] += nn.functional.mse_loss(aux_out[:, i], aux_feats[:, i]).item() * B

        # Accumulate for R2 calculation
        with torch.no_grad():
            preds_5_log = biomass_out  # Now directly predicting all 5
            targs_5_log = targets_log
            all_preds_log.append(preds_5_log.cpu().numpy())
            all_targets_full.append(targs_5_log.cpu().numpy())

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    preds_linear = np.expm1(preds_log)
    targets_log = np.concatenate(all_targets_full)
    targets_linear = np.expm1(targets_log)
    final_metrics[f'{prefix}_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, cfg.targets.official_weights)
    return final_metrics


def save_metadata(session_dir, species_list, target_cols, num_aux):
    metadata = {
        'species_list': species_list,
        'target_cols': target_cols,
        'num_aux': num_aux,
        'backbone': cfg.hyperparameters.backbone,
        'image_height': cfg.preprocessing.image_height,
        'image_width': cfg.preprocessing.image_width,
        'imagenet_mean': cfg.preprocessing.imagenet_mean,
        'imagenet_std': cfg.preprocessing.imagenet_std,
        'num_species': len(species_list),
        'biomass_clamp': cfg.targets.biomass_clamp,
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tile_augmentation': 'enabled',
        'stratification_key': cfg.split.stratification_key,
        'holdout_pct': cfg.split.holdout_pct
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata


def ensure_fold_coverage(train_df, val_df, group_col='SessionID', logger=None):
    """
    Ensure every Species present in the validation set is also in training.
    If not, move the smallest session of that species from validation to training.
    """
    missed = set(val_df['Species'].unique()) - set(train_df['Species'].unique())
    if not missed:
        return train_df, val_df
        
    for sp in missed:
        sp_val = val_df[val_df['Species'] == sp]
        # Sort by number of samples to move the smallest session first to minimize impact on validation size
        sessions = sp_val.groupby(group_col).size().sort_values().index.tolist()
        if sessions:
            sess_to_move = sessions[0]
            mask = val_df[group_col] == sess_to_move
            train_df = pd.concat([train_df, val_df[mask]], ignore_index=True)
            val_df = val_df[~mask].reset_index(drop=True)
            if logger:
                logger.info(f"  [COVERAGE] Moved session {sess_to_move} ({sp}) from Val to Train.")
    return train_df, val_df

def main():
    session_dir = setup_logging(file_name_part="unified_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(cfg.hyperparameters.random_seed, logger)

    logger.info("="*70)
    logger.info("UNIFIED TRAIN/VAL + RANDOM HOLDOUT")
    logger.info("Tile-based augmentation enabled for training.")
    logger.info("="*70)

    logger.info(config_str())
    shutil.copy(yaml_path, os.path.join(session_dir, 'used_config.yaml'))


    # 1. Load raw wide + deterministic features (no learning)
    df = load_data(logger)
    df = apply_deterministic_features(df)

    species_list = cfg.species_taxonomy.core_species
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    train_transform, val_transform = get_image_data_transforms()

    # 2. Coverage-aware train/holdout split ensuring species×season coverage
    strat_key = cfg.split.stratification_key
    holdout_pct = cfg.split.holdout_pct
    min_train_per_combo = getattr(cfg.split, 'min_train_per_combo', 2)
    
    logger.info(f"Using coverage-aware split with stratification_key='{strat_key}'")
    
    # Import the coverage-aware splitting function
    from common import coverage_aware_split
    
    dev_df, hold_df = coverage_aware_split(
        df, 
        stratify_col=strat_key,
        min_train_per_combo=min_train_per_combo,
        holdout_pct=holdout_pct,
        random_state=313,
        logger=logger,
        ensure_species_train_coverage=cfg.split.ensure_species_train_coverage,
        species_col=cfg.split.species_col,
        min_train_per_species=cfg.split.min_train_per_species,
        ensure_combo_train_coverage=cfg.split.ensure_combo_train_coverage,
        combo_col=cfg.split.combo_col        
    )
    
    species_col = 'species_id' if 'species_id' in df.columns else ('Species' if 'Species' in df.columns else None)
    
    log_dataframe_details(logger, dev_df, name="Development Set")
    log_dataframe_details(logger, hold_df, name="Random Holdout Set")

    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Development Set: {len(dev_df)}")
    logger.info(f"Random Holdout: {len(hold_df)}")
    hold_df.to_csv(os.path.join(splits_dir, "global_holdout.csv"), index=False)

    # 3. StratifiedKFold CV on Species_Season for balanced folds
    best_overall_score = -float('inf')
    
    if strat_key not in dev_df.columns:
        raise ValueError(f"Stratification key '{strat_key}' not found in dev_df.")
    
    sgkf = StratifiedGroupKFold(n_splits=cfg.hyperparameters.n_folds, shuffle=True, random_state=313)
    # Use config keys directly — allow exceptions if keys/columns are missing
    strat_col_cfg = cfg.split.group_stratification_col
    group_col_cfg = cfg.split.group_col

    splitter = sgkf.split(dev_df, dev_df[strat_col_cfg], groups=dev_df[group_col_cfg])
    fold_iter = [(dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True))
                 for train, val in splitter]
    split_name = f"StratifiedGroupKFold-{strat_col_cfg}-groups-{group_col_cfg}"
    
    per_fold_best = []
    for fold, (train_df_raw, val_df_raw) in enumerate(fold_iter):
        logger.info(f"\n{'='*30} FOLD {fold+1}/{cfg.hyperparameters.n_folds} {'='*30}")
        
        # Ensure training coverage for this fold (Species-level)
        #train_df_raw, val_df_raw = ensure_fold_coverage(train_df_raw, val_df_raw, logger=logger)
        
        # Fit/Transform pipeline per fold
        ft = BiomassFeatureTransform(logger)
        train_df = ft.fit(train_df_raw)
        val_df = ft.transform(val_df_raw)
        hold_df = ft.transform(hold_df)
        raw_n_train = len(train_df)
        # Verify No Leakage
        train_groups = set(train_df[group_col_cfg])
        val_groups = set(val_df[group_col_cfg])
        leakage = train_groups.intersection(val_groups)
        if leakage:
            logger.error(f"CRITICAL: GROUP LEAKAGE DETECTED IN TRAIN/VAL! {len(leakage)} groups shared: {leakage}")
            raise ValueError("Group Leakage Detected")
        else:
            logger.info("✓ No group leakage in train/validation detected")
        logger.info(f"\n{'='*20} Fold {fold+1}/{cfg.hyperparameters.n_folds} ({split_name}) {'='*20}")
        logger.info(f"Train:   n={len(train_df)}, sessions={train_df['SessionID'].nunique() if 'SessionID' in train_df.columns else 'N/A'}")
        logger.info(f"Val:     n={len(val_df)}, sessions={val_df['SessionID'].nunique() if 'SessionID' in val_df.columns else 'N/A'}")
        logger.info(f"Holdout: n={len(hold_df)}, sessions={hold_df['SessionID'].nunique() if 'SessionID' in hold_df.columns else 'N/A'}")
   
        # Log species counts across train/val/hold using a formatted table helper
        log_species_table(logger, train_df, val_df, hold_df, species_col=species_col, title='Species in Fold')
        
        # Generate split analysis visualizations
        from visualize_splits import analyze_splits_in_training
        try:
            analyze_splits_in_training(
                train_df,
                val_df,
                hold_df,
                session_dir,
                fold,
                group_col=group_col_cfg
            )
            logger.info("Split analysis visualizations saved to split_analysis/")            
        except Exception as e:
            logger.warning(f"Could not generate split analysis: {e}")

        log_fold_details(logger, train_df, val_df)

        if raw_n_train < cfg.hyperparameters.min_train_samples:
            logger.info(f"\nSkipping Fold {fold+1}: Training set too small ({raw_n_train} < {cfg.hyperparameters.min_train_samples})")
            continue

        logger.info(f"Training fold {fold} size: {len(train_df)} (upsampling applied in fit())")
        if 'State_Species' in train_df.columns:
            logger.info(f"Training set distribution: {train_df['State_Species'].value_counts()}")

        effective_train_size = len(train_df) * 6
        logger.info(f"\n{'='*40}")
        logger.info(f"EFFECTIVE TRAINING SIZE WITH TILING")
        logger.info(f"Base Samples: {len(train_df)}")
        logger.info(f"With 6x Tile Augmentation: {effective_train_size}")
        logger.info(f"{'='*40}\n")

        train_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_val.csv"), index=False)

        # Train targets must align with model outputs: [Green, Dead, Clover]
        train_ds_base = TiledBiomassDataset(
            train_df,
            transform=train_transform,
            mode='training',
            tile_prob=cfg.augmentation.tile_prob,
            target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        )
        train_ds = TiledMixupDataset(train_ds_base, prob=cfg.augmentation.mixup_prob, alpha=cfg.augmentation.mixup_alpha)

        val_ds = TiledBiomassDataset(
            val_df,
            transform=val_transform,
            mode='validation',
            tile_prob=0.0,
            target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        )

        holdout_ds = TiledBiomassDataset(
            hold_df,
            transform=val_transform,
            mode='validation',
            tile_prob=0.0,
            target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation', target_cols=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'])
        n_aux = dummy_ds[0]['aux_feats'].shape[0]
        model = BiomassUnifiedModel(num_aux=n_aux, config=cfg).to(cfg.device)

        if fold == 0:
            save_metadata(session_dir, cfg.species_taxonomy.core_species, cfg.targets.cols, n_aux)

        n_upsampled = len(train_df)
        # With tile_prob=0.8, effective training size is ~5x larger
        effective_size = n_upsampled * (1 + 5 * cfg.augmentation.tile_prob)
        
        freeze_threshold = cfg.hyperparameters.backbone_freeze_threshold
        
        if n_upsampled < freeze_threshold:
            logger.info(f"PROTECTION: Keeping backbone FROZEN for Fold {fold+1} (n_upsampled={n_upsampled} < {freeze_threshold})")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            if cfg.training.freeze_backbone:
                logger.info(f"STRATEGY: Applying Partial Freeze ({cfg.training.backbone_freeze_fraction*100}%) for Fold {fold+1} (n_upsampled={n_upsampled}, effective={effective_size:.0f})")
                all_params = list(model.backbone.parameters())
                freeze_until = int(len(all_params) * cfg.training.backbone_freeze_fraction)
                for i, p in enumerate(all_params):
                    p.requires_grad = (i >= freeze_until)
            else:
                # Smart unfreezing for ViT models: unfreeze last transformer blocks
                logger.info(f"STRATEGY: Smart Partial Unfreeze for Fold {fold+1} (n_upsampled={n_upsampled}, effective={effective_size:.0f})")
                
                # For ViT models, unfreeze the last 4 transformer blocks (most important for adaptation)
                # Keep patch embedding and early blocks frozen (general features)
                if hasattr(model.backbone, 'blocks'):  # ViT architecture
                    total_blocks = len(model.backbone.blocks)
                    unfreeze_last_n = 1  # Ultra-conservative: only last block
                    
                    # Freeze patch embedding and early blocks
                    if hasattr(model.backbone, 'patch_embed'):
                        for param in model.backbone.patch_embed.parameters():
                            param.requires_grad = False
                    
                    # Freeze/unfreeze blocks
                    for i, block in enumerate(model.backbone.blocks):
                        freeze_block = i < (total_blocks - unfreeze_last_n)
                        for param in block.parameters():
                            param.requires_grad = not freeze_block
                    
                    # Unfreeze norm and head if they exist
                    if hasattr(model.backbone, 'norm'):
                        for param in model.backbone.norm.parameters():
                            param.requires_grad = True
                    
                    logger.info(f"  ViT: Unfroze last {unfreeze_last_n}/{total_blocks} transformer blocks")
                else:
                    # For non-ViT models, unfreeze all
                    logger.info(f"  Non-ViT: Full backbone unfreeze")
                    for param in model.backbone.parameters():
                        param.requires_grad = True


        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        param_groups = [
            {'params': backbone_params, 'lr': cfg.hyperparameters.learning_rate * cfg.hyperparameters.backbone_lr_factor},
            {'params': head_params, 'lr': cfg.hyperparameters.learning_rate}
        ]
        optimizer = AdamW(param_groups, weight_decay=cfg.hyperparameters.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, threshold=5e-4, min_lr=1e-6, verbose=True)

        criterion_reg = nn.MSELoss()
        criterion_species = nn.BCEWithLogitsLoss()

        history = defaultdict(list)
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_h_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0


        bio_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        bio_train_log = np.log1p(train_df[bio_cols].astype(float).values)
        
        bio_mean_np = bio_train_log.mean(axis=0)
        bio_std_np = bio_train_log.std(axis=0)
        bio_mean_t = torch.tensor(bio_mean_np, dtype=torch.float32, device=cfg.device).view(1, -1)
        bio_std_t = torch.tensor(bio_std_np, dtype=torch.float32, device=cfg.device).view(1, -1)
        official_weights_t = torch.tensor(cfg.targets.official_weights, dtype=torch.float32, device=cfg.device)

        base_aux = ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Interaction_Add']
        ordinal_cols = ['NDVI_Bin_Ordinal', 'Height_Bin_Ordinal']
        onehot_cols = [f'NDVI_Bin_OH_{k}' for k in range(4)] + [f'Height_Bin_OH_{k}' for k in range(4)]
        aux_cols = [c for c in base_aux if c in train_df.columns]
        for c in ordinal_cols + onehot_cols:
            if c in train_df.columns:
                aux_cols.append(c)
        if 'Species_Count' in train_df.columns:
            aux_cols.append('Species_Count')
        aux_data = train_df[aux_cols].astype(float).fillna(0.0).values if len(aux_cols) > 0 else np.zeros((len(train_df), 0), dtype=float)
        if aux_data.shape[1] > 0:
            aux_mean_np = aux_data.mean(axis=0)
            aux_std_np = aux_data.std(axis=0)
            # Add HSV stats for 5 biomass scores (mean=0.0, std=1.0 as they are already 0-1 scores)
            # Order: green, dry_green, clover, dead, soil
            aux_mean_np = np.append(aux_mean_np, [0.0, 0.0, 0.0, 0.0, 0.0])
            aux_std_np = np.append(aux_std_np, [1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            aux_mean_np = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            aux_std_np = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

        aux_mean_t = torch.tensor(aux_mean_np, dtype=torch.float32, device=cfg.device).view(1, -1)
        aux_std_t = torch.tensor(aux_std_np, dtype=torch.float32, device=cfg.device).view(1, -1)

        # EMA state for scheduler stability (epoch-level score smoothing)
        ema_score_prev = None
        ema_decay = cfg.training.ema_decay

        for epoch in range(cfg.hyperparameters.epochs):
            # --- Linear Warmup for first 5 epochs ---
            warmup_epochs = 5
            if epoch < warmup_epochs:
                # Calculate warmup factor (0.3 at ep 0, 1.0 at ep 5)
                warmup_factor = 0.3 + 0.7 * (epoch / warmup_epochs)
                base_lr = cfg.hyperparameters.learning_rate * warmup_factor
                optimizer.param_groups[0]['lr'] = base_lr * cfg.hyperparameters.backbone_lr_factor
                optimizer.param_groups[1]['lr'] = base_lr
                logger.info(f"  [Warmup] Epoch {epoch}: LR scales to {warmup_factor:.2f}x ({base_lr:.6f})")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_species,
                cfg, epoch, session_dir=session_dir, logger=logger,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t,
                official_weights_t=official_weights_t
            )

            val_metrics = validate(
                model, val_loader, criterion_reg, criterion_species, cfg, prefix='val', use_tta=False,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )

            hol_metrics = validate(
                model, holdout_loader, criterion_reg, criterion_species, cfg,
                prefix='holdout', use_tta=cfg.training.use_tta,
                epoch=epoch, fold=fold, session_dir=session_dir,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )            
            t_r2 = train_metrics['train_r2']
            v_r2 = val_metrics['val_r2']
            h_r2 = hol_metrics['holdout_r2']

            # Calculate score with selected method
            ema_score, score_gap, raw_score = calculate_scheduler_score(
                train_r2=t_r2,
                val_r2=v_r2,
                holdout_r2=h_r2,
                ema_score_prev=ema_score_prev,
                ema_decay=ema_decay,                
            )
            current_score = ema_score
            ema_score_prev = ema_score

            # Step the scheduler (only after warmup)
            if epoch >= warmup_epochs:
                scheduler.step(ema_score)

            log_msg = get_formatted_loss_log(epoch,
                                             train_metrics,
                                             val_metrics,
                                             hol_metrics,
                                             current_score, score_gap,
                                             scheduler.get_last_lr()[0],
                                             v_r2, h_r2)
            logger.info(log_msg)

            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            for k, v in hol_metrics.items(): history[k].append(v)
            history['score'].append(current_score)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            
            better_r2_val = v_r2 > best_fold_v_r2
            better_r2_holdout = h_r2 > best_fold_h_r2
            better_score = current_score > best_fold_score
            
            # Save when EITHER Val or Holdout R² improves
            save_model = better_r2_val and better_r2_holdout
            
            if save_model:
                # log what we have compared to what we had previously
                logger.info(f"*** Fold {fold+1} Improved R2 Metrics (V:{v_r2:.3f} <new vs old> {best_fold_v_r2:.3f}, H:{h_r2:.3f} <new vs old> {best_fold_h_r2:.3f}) ***")
                logger.info(f"*** Fold {fold+1} Improved Score (S:{current_score:.4f} <new vs old> {best_fold_score:.4f}) ***")

                # Unconditionally update all trackers when saving (they all improved by construction)
                best_fold_v_r2 = v_r2
                best_fold_h_r2 = h_r2
                best_fold_score = current_score
                
                best_fold_epoch = epoch
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold+1}.pth"))
                patience_counter = 0
                
                # Update overall best tracker
                if best_fold_score > best_overall_score:
                    best_overall_score = best_fold_score
                    torch.save(model.state_dict(), os.path.join(session_dir, "best_model_overall.pth"))
                    logger.info(f">>> NEW OVERALL BEST MODEL: Fold {fold+1}, Score {best_overall_score:.4f} <<<")
            else:
                patience_counter += 1

            if patience_counter >= cfg.hyperparameters.early_stop_patience:
                logger.info("Early Stopping Triggered")
                break

            plot_training_history(history, fold+1, session_dir)

        logger.info(f"\n[Fold {fold+1} COMPLETE]")
        logger.info(f"Best Score: {best_fold_score:.4f} (at Epoch {best_fold_epoch})")
        logger.info(f"Best Val R2: {best_fold_v_r2:.4f}")
        logger.info(f"Best Holdout R2: {best_fold_h_r2:.4f}")
        logger.info("-" * 40)
        # Safely derive best epoch (fallback to last epoch if none recorded)
        be = best_fold_epoch if best_fold_epoch >= 0 else (len(history.get('score', [])) - 1 if len(history.get('score', [])) > 0 else 0)
        try:
            best_val_loss = history.get('val_loss', [None])[be]
        except Exception:
            best_val_loss = None
        try:
            best_holdout_loss = history.get('holdout_loss', [None])[be]
        except Exception:
            best_holdout_loss = None
        try:
            best_train_loss = history.get('train_loss', [None])[be]
        except Exception:
            best_train_loss = None

        per_fold_best.append({
            'best_score': best_fold_score,
            'best_val_loss': best_val_loss,
            'best_holdout_loss': best_holdout_loss,
            'best_train_loss': best_train_loss,
            'best_val_r2': best_fold_v_r2,
            'best_holdout_r2': best_fold_h_r2,
            'best_epoch': be
        })

        # Log fold summary tables (best-epoch + epoch averages)
        log_fold_summary_tables(logger, fold+1, history, be)

    # After all folds complete, aggregate and log best metrics across folds
    try:
        log_aggregate_best_across_folds(logger, per_fold_best)
    except Exception:
        logger.warning("Failed to compute aggregated fold statistics")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - Using only fold-specific models")
    logger.info("="*80)


if __name__ == '__main__':
    main()