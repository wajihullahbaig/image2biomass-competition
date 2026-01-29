import os
import logging
import shutil
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.loader import cfg, yaml_path
from configs import config_str
from common import (
    calculate_scheduler_score, get_season, load_data, get_image_data_transforms, save_batch_images,
    save_hsv_mask_batch,
    set_seed, calculate_competition_r2,
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

        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            # Model returns [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total], aux_out (tabular), hsv_out (visual), species_logits
            biomass_out, aux_out, hsv_out, species_logits = model(images)

            n_tab = model.num_aux
            t_aux = aux_feats[:, :n_tab]
            t_hsv = aux_feats[:, n_tab:]

            targets_log = torch.log1p(targets_g)
            
            reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
            
            p_bio = biomass_out
            t_bio = targets_log
            
            if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
                p_bio = (p_bio - bio_mean) / (bio_std + 1e-9)
                t_bio = (t_bio - bio_mean) / (bio_std + 1e-9)

            l_green  = reg(p_bio[:, 0:1], t_bio[:, 0:1])
            l_dead   = nn.HuberLoss(delta=cfg.training.huber_delta)(p_bio[:, 1:2], t_bio[:, 1:2]) if cfg.training.use_huber_loss_for_dead else reg(p_bio[:, 1:2], t_bio[:, 1:2])
            l_clover = reg(p_bio[:, 2:3], t_bio[:, 2:3])
            l_gdm    = reg(p_bio[:, 3:4], t_bio[:, 3:4])
            l_total  = reg(p_bio[:, 4:5], t_bio[:, 4:5])
            
            if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
                official_weights_t = official_weights_t.to(cfg.device)
                l_green  = l_green  * official_weights_t[0]
                l_dead   = l_dead   * official_weights_t[1]
                l_clover = l_clover * official_weights_t[2]
                l_gdm    = l_gdm    * official_weights_t[3]
                l_total  = l_total  * official_weights_t[4]
            
            loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

            # Physical Consistency
            raw_green, raw_dead, raw_clover, raw_gdm, raw_total = [biomass_out[:, i:i+1] for i in range(5)]
            lin_green, lin_dead, lin_clover, lin_gdm, lin_total = [torch.expm1(v) for v in [raw_green, raw_dead, raw_clover, raw_gdm, raw_total]]
            p_gdm_sum_log = torch.log1p(torch.clamp(lin_green + lin_clover, min=1e-4))
            p_total_sum_log = torch.log1p(torch.clamp(lin_gdm + lin_dead, min=1e-4))
            
            cons_weight = getattr(cfg.training, 'consistency_weight', cfg.training.biomass_feat_weight * 0.1)
            l_consistency_gdm = reg(raw_gdm, p_gdm_sum_log)
            l_consistency_tot = reg(raw_total, p_total_sum_log)
            
            t_dead_hsv = t_hsv[:, 3:4]
            k_scaling = getattr(cfg.training, 'dead_hsv_min_k', 17.5)
            min_dead_lin = t_dead_hsv * k_scaling
            p_dead_log = biomass_out[:, 1:2]
            min_dead_log = torch.log1p(min_dead_lin)
            l_hsv_constraint = torch.mean(torch.nn.functional.relu(min_dead_log - p_dead_log))
            
            l_consistency_bio = (l_consistency_gdm + l_consistency_tot) * cons_weight + l_hsv_constraint * getattr(cfg.training, 'hsv_constraint_weight', 10.0)

            # Auxiliary Loss
            p_aux = aux_out
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                p_aux = (p_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + 1e-9)
                t_aux = (t_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + 1e-9)
            loss_aux = nn.MSELoss()(p_aux, t_aux) * cfg.training.aux_feat_weight

            # HSV Loss
            p_hsv = hsv_out
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                p_hsv = (p_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + 1e-9)
                t_hsv = (t_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + 1e-9)
            loss_hsv = nn.MSELoss()(p_hsv, t_hsv) * cfg.training.hsv_feat_weight

            # Species Loss with Label Smoothing
            # loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight
            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec * 0.9 + 0.05) * cfg.training.species_feat_weight

            # --- POPULATE GRANULAR LOSSES FOR PLOTTER ---
            # 1. HSV Components
            l_hsv_g = nn.MSELoss()(p_hsv[:, 0:1], t_hsv[:, 0:1])
            l_hsv_dg = nn.MSELoss()(p_hsv[:, 1:2], t_hsv[:, 1:2])
            l_hsv_c = nn.MSELoss()(p_hsv[:, 2:3], t_hsv[:, 2:3])
            l_hsv_d = nn.MSELoss()(p_hsv[:, 3:4], t_hsv[:, 3:4])
            l_hsv_s = nn.MSELoss()(p_hsv[:, 4:5], t_hsv[:, 4:5])

            # 2. Aux Components (if they exist)
            l_ndvi = nn.MSELoss()(p_aux[:, 0:1], t_aux[:, 0:1]) if n_tab > 0 else torch.tensor(0.0).to(cfg.device)
            l_height = nn.MSELoss()(p_aux[:, 1:2], t_aux[:, 1:2]) if n_tab > 1 else torch.tensor(0.0).to(cfg.device)
            l_imul = nn.MSELoss()(p_aux[:, 2:3], t_aux[:, 2:3]) if n_tab > 2 else torch.tensor(0.0).to(cfg.device)
            l_iadd = nn.MSELoss()(p_aux[:, 3:4], t_aux[:, 3:4]) if n_tab > 3 else torch.tensor(0.0).to(cfg.device)
            
            # Find Species_Count in aux_cols if it exists
            l_sp_count = torch.tensor(0.0).to(cfg.device)
            if 'Species_Count' in loader.dataset.dataset.aux_cols:
                sp_idx = loader.dataset.dataset.aux_cols.index('Species_Count')
                l_sp_count = nn.MSELoss()(p_aux[:, sp_idx:sp_idx+1], t_aux[:, sp_idx:sp_idx+1])

            total_loss = loss_bio + loss_aux + loss_hsv + loss_sp + l_consistency_bio

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
            all_preds_log.append(biomass_out.cpu().numpy())
            all_targets_full.append(targets_log.cpu().numpy())

        # Update metrics
        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio']  += loss_bio.item() * B
        metrics['train_aux']  += loss_aux.item() * B
        metrics['train_hsv']  += loss_hsv.item() * B
        metrics['train_sp']   += loss_sp.item() * B
        metrics['train_cons'] += l_consistency_bio.item() * B
        metrics['train_loss_hsv_constraint'] += l_hsv_constraint.item() * B
        
        metrics['train_loss_green']  += l_green.item() * B
        metrics['train_loss_dead']   += l_dead.item() * B
        metrics['train_loss_clover'] += l_clover.item() * B
        metrics['train_loss_gdm']    += l_gdm.item() * B
        metrics['train_loss_total']  += l_total.item() * B

        # Granular components for plotter
        metrics['train_loss_g_hsv'] += l_hsv_g.item() * B
        metrics['train_loss_dg_hsv'] += l_hsv_dg.item() * B
        metrics['train_loss_c_hsv'] += l_hsv_c.item() * B
        metrics['train_loss_d_hsv'] += l_hsv_d.item() * B
        metrics['train_loss_s_hsv'] += l_hsv_s.item() * B
        
        metrics['train_loss_ndvi'] += l_ndvi.item() * B
        metrics['train_loss_height'] += l_height.item() * B
        metrics['train_loss_i_mul'] += l_imul.item() * B
        metrics['train_loss_i_add'] += l_iadd.item() * B
        metrics['train_loss_sp_count'] += l_sp_count.item() * B

        pbar.set_postfix({'L': total_loss.item()})

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    targets_log = np.concatenate(all_targets_full)
    final_metrics['train_r2'] = calculate_competition_r2(np.expm1(targets_log), np.expm1(preds_log), cfg.targets.official_weights)
    return final_metrics

@torch.no_grad()
def validate(model, loader, criterion_reg, criterion_species, cfg, prefix='val', use_tta=False, 
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

    for batch in loader:
        images = batch['image'].to(cfg.device)
        targets_g = batch['targets'].to(cfg.device)
        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        if use_tta:
            accum_bio_linear = 0
            accum_aux, accum_hsv, accum_sp_probs = 0, 0, 0
            for _, transform_fn in tta_views:
                img_aug = transform_fn(images)
                bio_out_tta, aux_out_tta, hsv_out_tta, sp_logits_tta = model(img_aug)
                accum_bio_linear += torch.expm1(bio_out_tta)
                accum_aux += aux_out_tta
                accum_hsv += hsv_out_tta
                accum_sp_probs += torch.sigmoid(sp_logits_tta)
            
            biomass_out = torch.log1p(accum_bio_linear / len(tta_views))
            aux_out = accum_aux / len(tta_views)
            hsv_out = accum_hsv / len(tta_views)
            species_probs = torch.sigmoid(accum_sp_probs / len(tta_views))
        else:
            biomass_out, aux_out, hsv_out, species_logits = model(images)
            species_probs = torch.sigmoid(species_logits)

        n_tab = model.num_aux
        t_aux = aux_feats[:, :n_tab]
        t_hsv = aux_feats[:, n_tab:]
        targets_log = torch.log1p(targets_g)
        reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
        
        p_bio = biomass_out
        t_bio = targets_log
        if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
            p_bio = (p_bio - bio_mean) / (bio_std + 1e-9)
            t_bio = (t_bio - bio_mean) / (bio_std + 1e-9)

        l_green  = reg(p_bio[:, 0:1], t_bio[:, 0:1])
        l_dead   = nn.HuberLoss(delta=cfg.training.huber_delta)(p_bio[:, 1:2], t_bio[:, 1:2]) if cfg.training.use_huber_loss_for_dead else reg(p_bio[:, 1:2], t_bio[:, 1:2])
        l_clover = reg(p_bio[:, 2:3], t_bio[:, 2:3])
        l_gdm    = reg(p_bio[:, 3:4], t_bio[:, 3:4])
        l_total  = reg(p_bio[:, 4:5], t_bio[:, 4:5])
        
        if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
            official_weights_t = official_weights_t.to(cfg.device)
            l_green *= official_weights_t[0]; l_dead *= official_weights_t[1]
            l_clover *= official_weights_t[2]; l_gdm *= official_weights_t[3]; l_total *= official_weights_t[4]
        
        loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

        # Consistency
        raw_green, raw_dead, raw_clover, raw_gdm, raw_total = [biomass_out[:, i:i+1] for i in range(5)]
        lin_green, lin_dead, lin_clover, lin_gdm, lin_total = [torch.expm1(v) for v in [raw_green, raw_dead, raw_clover, raw_gdm, raw_total]]
        p_gdm_sum_log = torch.log1p(torch.clamp(lin_green + lin_clover, min=1e-4))
        p_total_sum_log = torch.log1p(torch.clamp(lin_gdm + lin_dead, min=1e-4))
        cons_weight = getattr(cfg.training, 'consistency_weight', cfg.training.biomass_feat_weight * 0.1)
        l_consistency_gdm = reg(raw_gdm, p_gdm_sum_log)
        l_consistency_tot = reg(raw_total, p_total_sum_log)
        
        t_dead_hsv = t_hsv[:, 3:4]
        k_scaling = getattr(cfg.training, 'dead_hsv_min_k', 17.5)
        min_dead_lin = t_dead_hsv * k_scaling
        p_dead_log = biomass_out[:, 1:2]
        min_dead_log = torch.log1p(min_dead_lin)
        l_hsv_constraint = torch.mean(torch.nn.functional.relu(min_dead_log - p_dead_log))
        l_consistency_bio = (l_consistency_gdm + l_consistency_tot) * cons_weight + l_hsv_constraint * getattr(cfg.training, 'hsv_constraint_weight', 10.0)

        # Aux & HSV
        p_aux = aux_out
        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            p_aux = (p_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + 1e-9)
            t_aux = (t_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + 1e-9)
        loss_aux = criterion_reg(p_aux, t_aux) * cfg.training.aux_feat_weight

        p_hsv = hsv_out
        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            p_hsv = (p_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + 1e-9)
            t_hsv = (t_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + 1e-9)
        loss_hsv = criterion_reg(p_hsv, t_hsv) * cfg.training.hsv_feat_weight

        if use_tta: loss_sp = nn.BCELoss()(species_probs, species_vec) * cfg.training.species_feat_weight
        else: loss_sp = criterion_species(species_logits, species_vec) * cfg.training.species_feat_weight

        total_loss = loss_bio + loss_aux + loss_hsv + loss_sp + l_consistency_bio
        
        # --- GRANULAR LOSSES FOR PLOTTER ---
        l_hsv_g = nn.MSELoss()(p_hsv[:, 0:1], t_hsv[:, 0:1])
        l_hsv_dg = nn.MSELoss()(p_hsv[:, 1:2], t_hsv[:, 1:2])
        l_hsv_c = nn.MSELoss()(p_hsv[:, 2:3], t_hsv[:, 2:3])
        l_hsv_d = nn.MSELoss()(p_hsv[:, 3:4], t_hsv[:, 3:4])
        l_hsv_s = nn.MSELoss()(p_hsv[:, 4:5], t_hsv[:, 4:5])

        l_ndvi = nn.MSELoss()(p_aux[:, 0:1], t_aux[:, 0:1]) if n_tab > 0 else torch.tensor(0.0).to(cfg.device)
        l_height = nn.MSELoss()(p_aux[:, 1:2], t_aux[:, 1:2]) if n_tab > 1 else torch.tensor(0.0).to(cfg.device)
        l_imul = nn.MSELoss()(p_aux[:, 2:3], t_aux[:, 2:3]) if n_tab > 2 else torch.tensor(0.0).to(cfg.device)
        l_iadd = nn.MSELoss()(p_aux[:, 3:4], t_aux[:, 3:4]) if n_tab > 3 else torch.tensor(0.0).to(cfg.device)
        
        l_sp_count = torch.tensor(0.0).to(cfg.device)
        if 'Species_Count' in loader.dataset.aux_cols:
            sp_idx = loader.dataset.aux_cols.index('Species_Count')
            l_sp_count = nn.MSELoss()(p_aux[:, sp_idx:sp_idx+1], t_aux[:, sp_idx:sp_idx+1])

        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_hsv'] += loss_hsv.item() * B
        metrics[f'{prefix}_sp'] += loss_sp.item() * B
        metrics[f'{prefix}_cons'] += l_consistency_bio.item() * B
        metrics[f'{prefix}_loss_hsv_constraint'] += l_hsv_constraint.item() * B
        
        # Individual biomass components for plotting
        metrics[f'{prefix}_loss_green'] += l_green.item() * B
        metrics[f'{prefix}_loss_dead'] += l_dead.item() * B
        metrics[f'{prefix}_loss_clover'] += l_clover.item() * B
        metrics[f'{prefix}_loss_gdm'] += l_gdm.item() * B
        metrics[f'{prefix}_loss_total'] += l_total.item() * B
        
        # Granular components for plotter
        metrics[f'{prefix}_loss_g_hsv'] += l_hsv_g.item() * B
        metrics[f'{prefix}_loss_dg_hsv'] += l_hsv_dg.item() * B
        metrics[f'{prefix}_loss_c_hsv'] += l_hsv_c.item() * B
        metrics[f'{prefix}_loss_d_hsv'] += l_hsv_d.item() * B
        metrics[f'{prefix}_loss_s_hsv'] += l_hsv_s.item() * B
        
        metrics[f'{prefix}_loss_ndvi'] += l_ndvi.item() * B
        metrics[f'{prefix}_loss_height'] += l_height.item() * B
        metrics[f'{prefix}_loss_i_mul'] += l_imul.item() * B
        metrics[f'{prefix}_loss_i_add'] += l_iadd.item() * B
        metrics[f'{prefix}_loss_sp_count'] += l_sp_count.item() * B
        
        all_preds_log.append(biomass_out.cpu().numpy())
        all_targets_full.append(targets_log.cpu().numpy())

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    targets_log = np.concatenate(all_targets_full)
    final_metrics[f'{prefix}_r2'] = calculate_competition_r2(np.expm1(targets_log), np.expm1(preds_log), cfg.targets.official_weights)
    return final_metrics

def save_metadata(session_dir, species_list, target_cols, num_aux):
    metadata = {
        'species_list': species_list,
        'target_cols': target_cols,
        'num_aux': num_aux,
        'fusion_dim': cfg.training.fusion_dim,
        'backbone': cfg.hyperparameters.backbone,
        'image_height': cfg.preprocessing.image_height,
        'image_width': cfg.preprocessing.image_width,
        'imagenet_mean': cfg.preprocessing.imagenet_mean,
        'imagenet_std': cfg.preprocessing.imagenet_std,
        'num_species': len(species_list),
        'biomass_clamp': cfg.targets.biomass_clamp,
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tile_augmentation': "enabled" if cfg.augmentation.tile_prob > 0 else "disabled",
        'stratification_key': cfg.split.stratification_key,
        'holdout_pct': cfg.split.holdout_pct
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

def check_leakage(train_df, val_df, hold_df, group_col, logger):
    """Exit the loop if any group is shared between splits."""
    t_groups = set(train_df[group_col])
    v_groups = set(val_df[group_col])
    h_groups = set(hold_df[group_col])
    
    leak_tv = t_groups & v_groups
    leak_th = t_groups & h_groups
    leak_vh = v_groups & h_groups
    
    if leak_tv or leak_th or leak_vh:
        logger.error(f"CRITICAL: LEAKAGE DETECTED!")
        if leak_tv: logger.error(f"Train/Val leak: {len(leak_tv)} groups shared")
        if leak_th: logger.error(f"Train/Hold leak: {len(leak_th)} groups shared")
        if leak_vh: logger.error(f"Val/Hold leak: {len(leak_vh)} groups shared")
        raise ValueError("Group Leakage Detected between dynamic splits.")
    logger.info(f"✓ Leakage Check: No shared {group_col} between Train, Val, and Holdout.")

def main():
    session_dir = setup_logging(file_name_part="unified_triplet")
    logger = logging.getLogger("System Logger")
    set_seed(cfg.hyperparameters.random_seed, logger)

    logger.info("="*70)
    logger.info("UNIFIED TRAIN/VAL/HOLD TRIPLET REGIME")
    logger.info("Dynamic splitting per fold with leakage verification")
    logger.info("="*70)

    logger.info(config_str())
    shutil.copy(yaml_path, os.path.join(session_dir, 'used_config.yaml'))

    # Load and Preprocess
    df = load_data(logger)
    df = apply_deterministic_features(df)

    species_list = cfg.species_taxonomy.core_species
    target_cols = cfg.targets.cols
    group_col = cfg.split.group_col
    strat_col = cfg.split.group_stratification_col
    n_folds = cfg.hyperparameters.n_folds

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    train_transform, val_transform = get_image_data_transforms()

    # 1. Stratified Group Splitting for Triplets
    # Using StratifiedGroupKFold on basic groups for the initial split
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.hyperparameters.random_seed)
    fold_indices = list(sgkf.split(df, df[strat_col], groups=df[group_col]))

    official_weights_t = torch.tensor(cfg.targets.official_weights, dtype=torch.float32, device=cfg.device)
    
    per_fold_best = []
    
    for fold_idx in range(n_folds):
        logger.info(f"\n{'='*30} FOLD {fold_idx+1}/{n_folds} {'='*30}")
        
        # --- Triplet Logic ---
        # Validation = Fold i
        # Holdout    = Fold (i+1) % N
        # Training   = Rest
        val_idx = fold_indices[fold_idx][1]
        hold_idx = fold_indices[(fold_idx + 1) % n_folds][1]
        
        # Remaining are Training
        all_indices = set(range(len(df)))
        train_idx = list(all_indices - set(val_idx) - set(hold_idx))
        
        train_df_raw = df.iloc[train_idx].reset_index(drop=True)
        val_df_raw = df.iloc[val_idx].reset_index(drop=True)
        hold_df_raw = df.iloc[hold_idx].reset_index(drop=True)

        # 2. IMMEDIATE LEAKAGE CHECK
        check_leakage(train_df_raw, val_df_raw, hold_df_raw, group_col, logger)

        # 3. Fit/Transform Feature Pipeline
        ft = BiomassFeatureTransform(logger)
        train_df = ft.fit(train_df_raw)
        val_df = ft.transform(val_df_raw)
        hold_df = ft.transform(hold_df_raw)

        logger.info(f"Fold Sizes: Train={len(train_df)}, Val={len(val_df)}, Hold={len(hold_df)}")
        log_species_table(logger, train_df, val_df, hold_df, species_col='Species', title=f'Species Distribution Fold {fold_idx+1}')

        train_df.to_csv(os.path.join(splits_dir, f"fold{fold_idx+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold_idx+1}_val.csv"), index=False)
        hold_df.to_csv(os.path.join(splits_dir, f"fold{fold_idx+1}_hold.csv"), index=False)

        # 4. Datasets & Loaders
        train_ds = TiledMixupDataset(
            TiledBiomassDataset(train_df, transform=train_transform, mode='training', tile_prob=cfg.augmentation.tile_prob),
            prob=cfg.augmentation.mixup_prob, alpha=cfg.augmentation.mixup_alpha,
            use_cutmix=getattr(cfg.augmentation, 'use_cutmix', False)
        )
        val_ds = TiledBiomassDataset(val_df, transform=val_transform, mode='validation')
        hold_ds = TiledBiomassDataset(hold_df, transform=val_transform, mode='validation')

        train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0)
        hold_loader = DataLoader(hold_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0)

        # 5. Model Initialization
        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation')
        n_total_aux = dummy_ds[0]['aux_feats'].shape[0]
        model = BiomassUnifiedModel(num_aux=n_total_aux-5, num_hsv=5, config=cfg).to(cfg.device)

        if fold_idx == 0:
            save_metadata(session_dir, species_list, target_cols, n_total_aux)

        # 6. Normalization Stats (Per-Fold)
        bio_train_log = np.log1p(train_df[target_cols].astype(float).values)
        bio_mean_t = torch.tensor(bio_train_log.mean(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        bio_std_t = torch.tensor(bio_train_log.std(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        
        # Use the columns discovered by the dataset to ensure perfect consistency
        # train_ds is TiledMixupDataset, so we get aux_cols from the inner TiledBiomassDataset
        aux_cols = train_ds.dataset.aux_cols
        
        aux_data = train_df[aux_cols].astype(float).fillna(0.0).values if aux_cols else np.zeros((len(train_df), 0))
        aux_mean_np = np.append(aux_data.mean(axis=0), [0.0]*5)
        aux_std_np = np.append(aux_data.std(axis=0), [1.0]*5)
        aux_mean_t = torch.tensor(aux_mean_np, dtype=torch.float32, device=cfg.device).view(1, -1)
        aux_std_t = torch.tensor(aux_std_np, dtype=torch.float32, device=cfg.device).view(1, -1)

        # 7. Optimizer & Backbone Logic
        if cfg.training.freeze_backbone:
            logger.info("Applying partial freeze on backbone...")
            for param in model.backbone.parameters(): param.requires_grad = False
            # Unfreeze last layer if requested by logic in existing script
            if hasattr(model.backbone, 'blocks'):
                for param in model.backbone.blocks[-1].parameters(): param.requires_grad = True
        
        # Dual-Group Optimizer: Separate backbone and head
        # Backbone LR = learning_rate * backbone_lr_factor
        # Head LR = learning_rate
        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': cfg.hyperparameters.learning_rate * cfg.hyperparameters.backbone_lr_factor},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': cfg.hyperparameters.learning_rate}
        ], weight_decay=cfg.hyperparameters.weight_decay)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion_reg = nn.MSELoss()
        criterion_species = nn.BCEWithLogitsLoss()

        # 8. Training Loop
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_h_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0
        ema_score_prev = None
        history = defaultdict(list)

        warmup_epochs = int(cfg.training.warmup_percentage_epochs * cfg.hyperparameters.epochs)
        for epoch in range(cfg.hyperparameters.epochs):
            # Status Logging
            is_unfrozen = epoch >= getattr(cfg.training, 'unfreeze_epoch', 999)
            is_warmup = epoch < warmup_epochs
            logger.info(f"--- Epoch {epoch} Status: Backbone={'UNFROZEN' if is_unfrozen else 'FROZEN'}, Warmup={'ACTIVE' if is_warmup else 'DONE'} ---")

            if epoch < warmup_epochs:
                warmup_factor = 0.3 + 0.7 * (epoch / warmup_epochs)
                base_lr = cfg.hyperparameters.learning_rate * warmup_factor
                optimizer.param_groups[0]['lr'] = base_lr * cfg.hyperparameters.backbone_lr_factor
                optimizer.param_groups[1]['lr'] = base_lr
                logger.info(f"  [Warmup] Epoch {epoch}: LR scales to {warmup_factor:.2f}x ({base_lr:.6f})")

            # --- Gradual Backbone Unfreezing ---
            if epoch == getattr(cfg.training, 'unfreeze_epoch', -1):
                unfreeze_layers = getattr(cfg.training, 'unfreeze_layers', 6)
                if hasattr(model.backbone, 'blocks'):
                    for param in model.backbone.blocks[-unfreeze_layers:].parameters():
                        param.requires_grad = True
                    logger.info(f"!!! Unfroze last {unfreeze_layers} backbone blocks at epoch {epoch} !!!")
                
                # Update optimizer with new backbone LR factor if unfreezing
                unfreeze_lr_factor = getattr(cfg.training, 'unfreeze_lr_factor', 0.01)
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr'] * unfreeze_lr_factor
                logger.info(f"Backbone LR factor updated to {unfreeze_lr_factor}x head LR")

            
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_species, cfg, epoch,
                session_dir, logger, bio_mean_t, bio_std_t, aux_mean_t, aux_std_t, official_weights_t
            )
            val_metrics = validate(
                model, val_loader, criterion_reg, criterion_species, cfg, prefix='val',
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )
            hold_metrics = validate(
                model, hold_loader, criterion_reg, criterion_species, cfg, prefix='holdout', use_tta=cfg.training.use_tta,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )

            # Scoring
            v_r2 = val_metrics['val_r2']
            h_r2 = hold_metrics['holdout_r2']
            ema_score, gap, _ = calculate_scheduler_score(
                train_metrics['train_r2'], v_r2, h_r2,
                ema_score_prev, cfg.training.ema_decay
            )
            ema_score_prev = ema_score
            # Scheduler Step (Cosine Annealing) - step per epoch
            scheduler.step(epoch)

            # Logging (Using optimizer.param_groups[1]['lr'] as the primary "Head" LR)
            logger.info(get_formatted_loss_log(
                epoch, train_metrics, val_metrics, hold_metrics, ema_score, gap,
                optimizer.param_groups[1]['lr'], v_r2, h_r2
            ))

            # History tracking
            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            for k, v in hold_metrics.items(): history[k].append(v)
            history['score'].append(ema_score)
            history['lr'].append(optimizer.param_groups[1]['lr'])


            # Plot every epoch for real-time monitoring
            plot_training_history(history, fold_idx+1, session_dir)

            better_r2_val = v_r2 > best_fold_v_r2
            better_r2_holdout = h_r2 > best_fold_h_r2
            
            
            if better_r2_val and better_r2_holdout:
                best_fold_score = ema_score
                best_fold_v_r2 = v_r2
                best_fold_h_r2 = h_r2
                best_fold_epoch = epoch
                logger.info(f"*** New Best Score for Fold {fold_idx+1}: {best_fold_score:.4f} (Val R2: {v_r2:.3f}, Hold R2: {h_r2:.3f}) ***")
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_fold{fold_idx+1}.pt"))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.hyperparameters.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        per_fold_best.append({
            'fold': fold_idx+1,
            'score': best_fold_score,
            'val_r2': best_fold_v_r2,
            'holdout_r2': best_fold_h_r2,
            'epoch': best_fold_epoch
        })
        
        # Log Summary for this fold
        log_fold_summary_tables(logger, fold_idx+1, history, best_fold_epoch)

    logger.info(f"\nFinal Triplet CV Score: {np.mean([f['score'] for f in per_fold_best]):.4f}")
    log_aggregate_best_across_folds(logger, per_fold_best)

if __name__ == "__main__":
    main()
