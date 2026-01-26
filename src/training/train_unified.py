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

        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)

        if epoch == 0 and batch_idx < 5 and session_dir:
            save_batch_images(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)
            save_hsv_mask_batch(images, fold=0, batch_idx=batch_idx, session_dir=session_dir, max_batches_to_save=5)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            # Model returns [Log_Green, Log_Dead, Log_Clover, Log_GDM, Log_Total], aux_out (tabular), hsv_out (visual), species_logits
            biomass_out, aux_out, hsv_out, species_logits = model(images)

            # Split aux_feats from dataset into tabular and hsv targets based on model architecture
            n_tab = model.num_aux
            t_aux = aux_feats[:, :n_tab]
            t_hsv = aux_feats[:, n_tab:]

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
            
            # dead_hsv is at index 3 of t_hsv
            t_dead_hsv = t_hsv[:, 3:4]
            
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

            # --- Auxiliary Loss (Tabular) ---
            p_aux = aux_out
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                eps = 1e-9
                # Tabular means/stds are stored in the first n_tab elements
                p_aux = (p_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + eps)
                t_aux = (t_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + eps)
            loss_aux = nn.MSELoss()(p_aux, t_aux) * cfg.training.aux_feat_weight

            # --- HSV Loss (Visual scores) ---
            p_hsv = hsv_out
            # HSV scores are usually already 0-1, so standardization may not be needed, 
            # but if it is, they are in indices 5-9
            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                eps = 1e-9
                p_hsv = (p_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + eps)
                t_hsv = (t_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + eps)
            loss_hsv = nn.MSELoss()(p_hsv, t_hsv) * cfg.training.hsv_feat_weight

            # --- Species Loss ---
            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight

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
            preds_5_log = biomass_out
            targs_5_log = targets_log
            
            all_preds_log.append(preds_5_log.cpu().numpy())
            all_targets_full.append(targs_5_log.cpu().numpy())

        # Update metrics
        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio']  += loss_bio.item() * B
        metrics['train_aux']  += loss_aux.item() * B
        metrics['train_hsv']  += loss_hsv.item() * B
        metrics['train_sp']   += loss_sp.item() * B
        metrics['train_cons'] += l_consistency_bio.item() * B
        metrics['train_loss_hsv_constraint'] += l_hsv_constraint.item() * B
        
        # Individual losses for tracking
        metrics['train_loss_green']  += l_green.item() * B
        metrics['train_loss_dead']   += l_dead.item() * B
        metrics['train_loss_clover'] += l_clover.item() * B
        metrics['train_loss_gdm']    += l_gdm.item() * B
        metrics['train_loss_total']  += l_total.item() * B
        
        # Tabular auxiliary feature loss tracking
        aux_names = ['ndvi', 'height', 'i_mul', 'i_add', 'sp_count']
        for i in range(min(aux_out.shape[1], len(aux_names))):
            metrics[f"train_loss_{aux_names[i]}"] += nn.functional.mse_loss(aux_out[:, i], t_aux[:, i]).item() * B
            
        # HSV feature loss tracking
        hsv_names = ['g_hsv', 'dg_hsv', 'c_hsv', 'd_hsv', 's_hsv']
        for i in range(min(hsv_out.shape[1], len(hsv_names))):
            metrics[f"train_loss_{hsv_names[i]}"] += nn.functional.mse_loss(hsv_out[:, i], t_hsv[:, i]).item() * B

        pbar.set_postfix({'L': total_loss.item()})

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = np.concatenate(all_preds_log)
    preds_linear = np.expm1(preds_log)
    targets_log = np.concatenate(all_targets_full)
    targets_linear = np.expm1(targets_log)
    final_metrics['train_r2'] = calculate_competition_r2(targets_linear, preds_linear, cfg.targets.official_weights)
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
            accum_hsv = 0
            accum_sp_probs = 0
            for view_name, transform_fn in tta_views:
                img_aug = transform_fn(images)
                bio_out_tta, aux_out_tta, hsv_out_tta, sp_logits_tta = model(img_aug)
                accum_bio_linear += torch.expm1(bio_out_tta)
                accum_aux += aux_out_tta
                accum_hsv += hsv_out_tta
                accum_sp_probs += torch.sigmoid(sp_logits_tta)
            
            avg_bio_linear = accum_bio_linear / len(tta_views)
            biomass_out = torch.log1p(avg_bio_linear)
            aux_out = accum_aux / len(tta_views)
            hsv_out = accum_hsv / len(tta_views)
            species_probs = accum_sp_probs / len(tta_views)
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
            l_green  = l_green  * official_weights_t[0]
            l_dead   = l_dead   * official_weights_t[1]
            l_clover = l_clover * official_weights_t[2]
            l_gdm    = l_gdm    * official_weights_t[3]
            l_total  = l_total  * official_weights_t[4]
        
        loss_bio = (l_green + l_dead + l_clover + l_gdm + l_total) * cfg.training.biomass_feat_weight

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

        p_aux = aux_out
        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            eps = 1e-9
            p_aux = (p_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + eps)
            t_aux = (t_aux - aux_mean[:, :n_tab]) / (aux_std[:, :n_tab] + eps)
        loss_aux = criterion_reg(p_aux, t_aux) * cfg.training.aux_feat_weight

        p_hsv = hsv_out
        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            eps = 1e-9
            p_hsv = (p_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + eps)
            t_hsv = (t_hsv - aux_mean[:, n_tab:]) / (aux_std[:, n_tab:] + eps)
        loss_hsv = criterion_reg(p_hsv, t_hsv) * cfg.training.hsv_feat_weight

        if use_tta:
            loss_sp = nn.BCELoss()(species_probs, species_vec) * cfg.training.species_feat_weight
        else:
            loss_sp = criterion_species(species_logits, species_vec) * cfg.training.species_feat_weight

        total_loss = loss_bio + loss_aux + loss_hsv + loss_sp + l_consistency_bio

        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_hsv'] += loss_hsv.item() * B
        metrics[f'{prefix}_sp']  += loss_sp.item() * B
        metrics[f'{prefix}_cons'] += l_consistency_bio.item() * B
        metrics[f'{prefix}_loss_hsv_constraint'] += l_hsv_constraint.item() * B
        
        metrics[f'{prefix}_loss_green']  += l_green.item() * B
        metrics[f'{prefix}_loss_dead']   += l_dead.item() * B
        metrics[f'{prefix}_loss_clover'] += l_clover.item() * B
        metrics[f'{prefix}_loss_gdm']    += l_gdm.item() * B
        metrics[f'{prefix}_loss_total']  += l_total.item() * B

        aux_names = ['ndvi', 'height', 'i_mul', 'i_add', 'sp_count']
        for i in range(min(aux_out.shape[1], len(aux_names))):
            metrics[f"{prefix}_loss_{aux_names[i]}"] += nn.functional.mse_loss(aux_out[:, i], t_aux[:, i]).item() * B
            
        hsv_names = ['g_hsv', 'dg_hsv', 'c_hsv', 'd_hsv', 's_hsv']
        for i in range(min(hsv_out.shape[1], len(hsv_names))):
            metrics[f"{prefix}_loss_{hsv_names[i]}"] += nn.functional.mse_loss(hsv_out[:, i], t_hsv[:, i]).item() * B

        with torch.no_grad():
            preds_5_log = biomass_out
            targs_5_log = targets_log
            all_preds_log.append(preds_5_log.cpu().numpy())
            all_targets_full.append(targs_5_log.cpu().numpy())

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()} if N > 0 else {}
    if N > 0:
        preds_log = np.concatenate(all_preds_log)
        preds_linear = np.expm1(preds_log)
        targets_log = np.concatenate(all_targets_full)
        targets_linear = np.expm1(targets_log)
        final_metrics[f'{prefix}_r2'] = calculate_competition_r2(targets_linear, preds_linear, cfg.targets.official_weights)
    else:
        final_metrics[f'{prefix}_r2'] = 0.0
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
        'holdout_pct': 0.0
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata


def main():
    session_dir = setup_logging(file_name_part="unified_no_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(cfg.hyperparameters.random_seed, logger)

    logger.info("="*70)
    logger.info("UNIFIED TRAIN/VAL (NO HOLDOUT VERSION)")
    logger.info("Entire dataset used for Cross-Validation.")
    logger.info("="*70)

    logger.info(config_str())
    shutil.copy(yaml_path, os.path.join(session_dir, 'used_config.yaml'))

    df = load_data(logger)
    df = apply_deterministic_features(df)

    species_list = cfg.species_taxonomy.core_species
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    train_transform, val_transform = get_image_data_transforms()

    # NO HOLDOUT: dev_df is the entire dataset
    dev_df = df
    hold_df = pd.DataFrame() # Empty holdout
    
    species_col = 'species_id' if 'species_id' in df.columns else ('Species' if 'Species' in df.columns else None)
    
    log_dataframe_details(logger, dev_df, name="Development Set (Full)")
    logger.info(f"Total Samples: {len(df)}")

    best_overall_score = -float('inf')
    strat_key = cfg.split.stratification_key or 'State_Species'
    
    sgkf = StratifiedGroupKFold(n_splits=cfg.hyperparameters.n_folds, shuffle=True, random_state=313)
    strat_col_cfg = cfg.split.group_stratification_col
    group_col_cfg = cfg.split.group_col

    splitter = sgkf.split(dev_df, dev_df[strat_col_cfg], groups=dev_df[group_col_cfg])
    fold_iter = [(dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True))
                 for train, val in splitter]
    
    per_fold_best = []
    for fold, (train_df_raw, val_df_raw) in enumerate(fold_iter):
        logger.info(f"\n{'='*30} FOLD {fold+1}/{cfg.hyperparameters.n_folds} {'='*30}")
        
        ft = BiomassFeatureTransform(logger)
        train_df = ft.fit(train_df_raw)
        val_df = ft.transform(val_df_raw)
        
        raw_n_train = len(train_df)
        logger.info(f"Train:   n={len(train_df)}, sessions={train_df['SessionID'].nunique()}")
        logger.info(f"Val:     n={len(val_df)}, sessions={val_df['SessionID'].nunique()}")
   
        log_species_table(logger, train_df, val_df, pd.DataFrame(), species_col=species_col, title='Species in Fold')
        log_fold_details(logger, train_df, val_df)

        if raw_n_train < cfg.hyperparameters.min_train_samples:
            continue

        train_ds_base = TiledBiomassDataset(
            train_df, transform=train_transform, mode='training',
            tile_prob=cfg.augmentation.tile_prob, target_cols=target_cols
        )
        train_ds = TiledMixupDataset(train_ds_base, prob=cfg.augmentation.mixup_prob, alpha=cfg.augmentation.mixup_alpha)

        val_ds = TiledBiomassDataset(
            val_df, transform=val_transform, mode='validation', target_cols=target_cols
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation', target_cols=target_cols)
        n_total_aux = dummy_ds[0]['aux_feats'].shape[0]
        n_hsv = 5
        n_tab = n_total_aux - n_hsv
        model = BiomassUnifiedModel(num_aux=n_tab, num_hsv=n_hsv, config=cfg).to(cfg.device)
        
        if fold == 0:
            save_metadata(session_dir, cfg.species_taxonomy.core_species, target_cols, n_total_aux)

        # Partial Freeze / Unfreeze logic
        n_upsampled = len(train_df)
        freeze_threshold = cfg.hyperparameters.backbone_freeze_threshold
        if n_upsampled < freeze_threshold:
            for param in model.backbone.parameters(): param.requires_grad = False
        else:
            if cfg.training.freeze_backbone:
                all_params = list(model.backbone.parameters())
                freeze_until = int(len(all_params) * cfg.training.backbone_freeze_fraction)
                for i, p in enumerate(all_params): p.requires_grad = (i >= freeze_until)
            else:
                for param in model.backbone.parameters(): param.requires_grad = True

        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': cfg.hyperparameters.learning_rate * cfg.hyperparameters.backbone_lr_factor},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': cfg.hyperparameters.learning_rate}
        ], weight_decay=cfg.hyperparameters.weight_decay)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=4, threshold=5e-4, min_lr=1e-6)
        criterion_reg = nn.MSELoss()
        criterion_species = nn.BCEWithLogitsLoss()

        history = defaultdict(list)
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0

        # Feature Stats
        bio_train_log = np.log1p(train_df[target_cols].astype(float).values)
        bio_mean_t = torch.tensor(bio_train_log.mean(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        bio_std_t = torch.tensor(bio_train_log.std(axis=0), dtype=torch.float32, device=cfg.device).view(1, -1)
        official_weights_t = torch.tensor(cfg.targets.official_weights, dtype=torch.float32, device=cfg.device)

        base_aux = ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Interaction_Add']
        aux_cols = [c for c in base_aux if c in train_df.columns]
        for c in ['NDVI_Bin_Ordinal', 'Height_Bin_Ordinal'] + [f'NDVI_Bin_OH_{k}' for k in range(4)] + [f'Height_Bin_OH_{k}' for k in range(4)]:
            if c in train_df.columns: aux_cols.append(c)
        if 'Species_Count' in train_df.columns: aux_cols.append('Species_Count')
        
        aux_data = train_df[aux_cols].astype(float).fillna(0.0).values if aux_cols else np.zeros((len(train_df), 0))
        aux_mean_np = np.append(aux_data.mean(axis=0), [0.0]*5) if aux_data.shape[1] > 0 else np.array([0.0]*5)
        aux_std_np = np.append(aux_data.std(axis=0), [1.0]*5) if aux_data.shape[1] > 0 else np.array([1.0]*5)
        aux_mean_t = torch.tensor(aux_mean_np, dtype=torch.float32, device=cfg.device).view(1, -1)
        aux_std_t = torch.tensor(aux_std_np, dtype=torch.float32, device=cfg.device).view(1, -1)

        ema_score_prev = None
        ema_decay = cfg.training.ema_decay

        for epoch in range(cfg.hyperparameters.epochs):
            # Warmup
            if epoch < 5:
                wf = 0.3 + 0.7 * (epoch / 5)
                optimizer.param_groups[0]['lr'] = cfg.hyperparameters.learning_rate * wf * cfg.hyperparameters.backbone_lr_factor
                optimizer.param_groups[1]['lr'] = cfg.hyperparameters.learning_rate * wf

            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion_reg, criterion_species, cfg, epoch, session_dir, logger, bio_mean_t, bio_std_t, aux_mean_t, aux_std_t, official_weights_t)
            val_metrics = validate(model, val_loader, criterion_reg, criterion_species, cfg, 'val', False, epoch, fold, session_dir, bio_mean_t, bio_std_t, aux_mean_t, aux_std_t, official_weights_t)
            
            t_r2 = train_metrics['train_r2']
            v_r2 = val_metrics['val_r2']

            # Use val_r2 as holdout_r2 for the score calculation in no-holdout mode
            ema_score, score_gap, raw_score = calculate_scheduler_score(train_r2=t_r2, val_r2=v_r2, holdout_r2=v_r2, ema_score_prev=ema_score_prev, ema_decay=ema_decay)
            ema_score_prev = ema_score
            
            if epoch >= 5: scheduler.step(ema_score)

            logger.info(get_formatted_loss_log(epoch, train_metrics, val_metrics, {}, ema_score, score_gap, optimizer.param_groups[1]['lr'], v_r2, 0.0))

            for k, v in train_metrics.items(): history[k].append(v)
            for k, v in val_metrics.items(): history[k].append(v)
            history['score'].append(ema_score); history['lr'].append(optimizer.param_groups[1]['lr'])

            if v_r2 > best_fold_v_r2:
                logger.info(f"*** Fold {fold+1} Improved Val R2: {v_r2:.4f} (Score: {ema_score:.4f}) ***")
                best_fold_v_r2 = v_r2
                best_fold_score = ema_score
                best_fold_epoch = epoch
                torch.save(model.state_dict(), os.path.join(session_dir, f"best_model_fold{fold+1}.pth"))
                if best_fold_score > best_overall_score:
                    best_overall_score = best_fold_score
                    torch.save(model.state_dict(), os.path.join(session_dir, "best_model_overall.pth"))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.hyperparameters.early_stop_patience: break
            plot_training_history(history, fold+1, session_dir)

        per_fold_best.append({
            'best_score': best_fold_score, 'best_val_loss': history.get('val_loss', [0]*100)[best_fold_epoch],
            'best_train_loss': history.get('train_loss', [0]*100)[best_fold_epoch],
            'best_val_r2': best_fold_v_r2, 'best_epoch': best_fold_epoch
        })
        log_fold_summary_tables(logger, fold+1, history, best_fold_epoch)

    log_aggregate_best_across_folds(logger, per_fold_best)
    logger.info("\nTRAINING COMPLETE (NO HOLDOUT)")

if __name__ == '__main__':
    main()