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
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGF = True
except Exception:
    StratifiedGroupKFold = None
    _HAS_SGF = False
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json

from config.loader import cfg,yaml_path
from configs import config_str
from common import (
    get_season, load_data, get_image_data_transforms, save_batch_images,
    set_seed, calculate_global_weighted_r2,
    get_taxonomy_targets,
    rotate_crop_resize, save_tta_images
)
from feature_transform import BiomassFeatureTransform, apply_deterministic_features

from log_and_plots import (
    get_formatted_loss_log, log_dataframe_details, setup_logging, plot_training_history,
    log_fold_details, log_species_table
)

from log_and_plots import log_fold_summary_tables, log_aggregate_best_across_folds

from dataset import TiledBiomassDataset, TiledMixupDataset
from models import BiomassUnifiedModel


def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_species, criterion_tax, cfg, epoch, session_dir=None, logger=None,
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

        taxonomy_targets = get_taxonomy_targets(species_vec)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            biomass_out, aux_out, species_logits, taxonomy_logits = model(images)

            bio_out_comp = biomass_out[:, :3]
            targ_comp = targets_log[:, :3]
            if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
                bio_out_comp = (bio_out_comp - bio_mean[:, :3]) / (bio_std[:, :3] + 1e-9)
                targ_comp = (targ_comp - bio_mean[:, :3]) / (bio_std[:, :3] + 1e-9)

            if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
                comp_weights = official_weights_t[:3]
                per_el = (
                    torch.nn.functional.smooth_l1_loss(bio_out_comp, targ_comp, reduction='none')
                    if cfg.loss.reg_loss_type == 'smoothl1'
                    else torch.nn.functional.mse_loss(bio_out_comp, targ_comp, reduction='none')
                )
                per_target_mean = per_el.mean(dim=0)
                loss_bio_comp = (per_target_mean * comp_weights).sum() * cfg.training.biomass_feat_weight
            else:
                loss_bio_comp = nn.MSELoss()(bio_out_comp, targ_comp) * cfg.training.biomass_feat_weight

            if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
                aux_out_std = (aux_out - aux_mean) / (aux_std + 1e-9)
                aux_targ_std = (aux_feats - aux_mean) / (aux_std + 1e-9)
                loss_aux = nn.MSELoss()(aux_out_std, aux_targ_std) * cfg.training.aux_feat_weight
            else:
                loss_aux = nn.MSELoss()(aux_out, aux_feats) * cfg.training.aux_feat_weight

            loss_sp = nn.BCEWithLogitsLoss()(species_logits, species_vec) * cfg.training.species_feat_weight
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            tax_log_probs = torch.nn.functional.log_softmax(taxonomy_logits, dim=1)
            loss_tax = nn.KLDivLoss(reduction='batchmean')(tax_log_probs, tax_targets_norm) * cfg.training.taxonomy_feat_weight

            pred_c = torch.expm1(biomass_out[:, 0:1])
            pred_d = torch.expm1(biomass_out[:, 1:2])
            pred_g = torch.expm1(biomass_out[:, 2:3])
            derived_log_total = torch.log1p(pred_c + pred_d + pred_g + 1e-8)
            derived_log_gdm = torch.log1p(pred_c + pred_g + 1e-8)

            targ_c = targets_g[:, 0:1]
            targ_d = targets_g[:, 1:2]
            targ_g = targets_g[:, 2:3]
            targ_total = torch.log1p(targ_c + targ_d + targ_g + 1e-8)
            targ_gdm = torch.log1p(targ_c + targ_g + 1e-8)
            if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
                derived_log_total_std = (derived_log_total - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + 1e-9)
                targ_total_std = (targ_total - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + 1e-9)
                derived_log_gdm_std = (derived_log_gdm - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + 1e-9)
                targ_gdm_std = (targ_gdm - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + 1e-9)
                reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
                loss_total = reg(derived_log_total_std, targ_total_std)
                loss_gdm = reg(derived_log_gdm_std, targ_gdm_std)
            else:
                reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
                loss_total = reg(derived_log_total, targ_total)
                loss_gdm = reg(derived_log_gdm, targ_gdm)

            if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
                loss_total = loss_total * official_weights_t[3]
                loss_gdm = loss_gdm * official_weights_t[4]
            loss_total = loss_total * cfg.training.biomass_feat_weight
            loss_gdm = loss_gdm * cfg.training.biomass_feat_weight

            total_loss = loss_bio_comp + loss_aux + loss_sp + loss_tax + loss_total + loss_gdm

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

        B = images.size(0)
        metrics['train_loss'] += total_loss.item() * B
        metrics['train_bio'] += loss_bio_comp.item() * B
        metrics['train_aux'] += loss_aux.item() * B
        metrics['train_sp']  += loss_sp.item() * B
        metrics['train_tax'] += loss_tax.item() * B

        with torch.no_grad():
            metrics['train_loss_c'] += nn.functional.mse_loss(biomass_out[:, 0], targets_log[:, 0]).item() * B
            metrics['train_loss_d'] += nn.functional.mse_loss(biomass_out[:, 1], targets_log[:, 1]).item() * B
            metrics['train_loss_g'] += nn.functional.mse_loss(biomass_out[:, 2], targets_log[:, 2]).item() * B
            metrics['train_loss_t'] += nn.functional.mse_loss(derived_log_total.squeeze(1), targ_total.squeeze(1)).item() * B
            metrics['train_loss_gdm'] += nn.functional.mse_loss(derived_log_gdm.squeeze(1), targ_gdm.squeeze(1)).item() * B
            metrics['train_loss_ndvi'] += nn.functional.mse_loss(aux_out[:, 0], aux_feats[:, 0]).item() * B
            metrics['train_loss_h']    += nn.functional.mse_loss(aux_out[:, 1], aux_feats[:, 1]).item() * B
            metrics['train_loss_int_mul']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B
            metrics['train_loss_int_add']  += nn.functional.mse_loss(aux_out[:, 3], aux_feats[:, 3]).item() * B

        preds_full = torch.zeros((biomass_out.size(0), 5), device=biomass_out.device, dtype=biomass_out.dtype)
        preds_full[:, 0:3] = biomass_out
        preds_full[:, 3:4] = derived_log_total
        preds_full[:, 4:5] = derived_log_gdm
        all_preds_log.append(preds_full.detach().cpu())

        targ_c = targets_g[:, 0:1]
        targ_d = targets_g[:, 1:2]
        targ_g = targets_g[:, 2:3]
        targ_total_lin = targ_c + targ_d + targ_g
        targ_gdm_lin = targ_c + targ_g
        targets_full_lin = torch.cat([targ_c, targ_d, targ_g, targ_total_lin, targ_gdm_lin], dim=1)
        all_targets_full.append(targets_full_lin.detach().cpu())

        pbar.set_postfix({'L': total_loss.item()})

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_full).numpy()
    final_metrics['train_r2'] = calculate_global_weighted_r2(targets_linear, preds_linear, cfg.targets.official_weights)
    return final_metrics


@torch.no_grad()
def validate(model, loader, criterion_reg, criterion_species, criterion_tax, cfg, prefix='val', use_tta=False, epoch=0, fold=0, session_dir=None,
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
        targets_log = torch.log1p(targets_g)
        aux_feats = batch['aux_feats'].to(cfg.device)
        species_vec = batch['species_id'].to(cfg.device)
        taxonomy_targets = get_taxonomy_targets(species_vec)

        if use_tta:
            accum_bio_linear = 0
            accum_aux = 0
            accum_sp_probs = 0
            accum_tax_probs = 0
            for view_name, transform_fn in tta_views:
                img_aug = transform_fn(images)
                if session_dir is not None:
                    save_tta_images(img_aug, view_name, batch_idx, fold, epoch, session_dir)
                bio_out, aux_out, sp_logits, tax_logits = model(img_aug)
                accum_bio_linear += torch.expm1(bio_out)
                accum_aux += aux_out
                accum_sp_probs += torch.sigmoid(sp_logits)
                accum_tax_probs += torch.softmax(tax_logits, dim=1)
            avg_bio_linear = accum_bio_linear / len(tta_views)
            avg_aux = accum_aux / len(tta_views)
            avg_sp_probs = accum_sp_probs / len(tta_views)
            avg_tax_probs = accum_tax_probs / len(tta_views)
            biomass_out = torch.log1p(avg_bio_linear)
            aux_out = avg_aux
            species_probs = avg_sp_probs
            taxonomy_log_probs = torch.log(avg_tax_probs + 1e-9)
        else:
            biomass_out, aux_out, species_logits, taxonomy_logits = model(images)

        bio_out_comp = biomass_out[:, :3]
        targ_comp = targets_log[:, :3]
        if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
            bio_out_comp = (bio_out_comp - bio_mean[:, :3]) / (bio_std[:, :3] + 1e-9)
            targ_comp = (targ_comp - bio_mean[:, :3]) / (bio_std[:, :3] + 1e-9)
        if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
            comp_weights = official_weights_t[:3]
            per_el = (
                torch.nn.functional.smooth_l1_loss(bio_out_comp, targ_comp, reduction='none')
                if cfg.loss.reg_loss_type == 'smoothl1'
                else torch.nn.functional.mse_loss(bio_out_comp, targ_comp, reduction='none')
            )
            per_target_mean = per_el.mean(dim=0)
            loss_bio = (per_target_mean * comp_weights).sum() * cfg.training.biomass_feat_weight
        else:
            loss_bio = criterion_reg(bio_out_comp, targ_comp) * cfg.training.biomass_feat_weight

        if cfg.loss.use_standardized_loss and aux_mean is not None and aux_std is not None:
            aux_out_std = (aux_out - aux_mean) / (aux_std + 1e-9)
            aux_targ_std = (aux_feats - aux_mean) / (aux_std + 1e-9)
            loss_aux = criterion_reg(aux_out_std, aux_targ_std) * cfg.training.aux_feat_weight
        else:
            loss_aux = criterion_reg(aux_out, aux_feats) * cfg.training.aux_feat_weight

        if use_tta:
            loss_sp = nn.BCELoss()(species_probs, species_vec) * cfg.training.species_feat_weight
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            loss_tax = criterion_tax(taxonomy_log_probs, tax_targets_norm) * cfg.training.taxonomy_feat_weight
        else:
            loss_sp = criterion_species(species_logits, species_vec) * cfg.training.species_feat_weight
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            tax_log_probs = torch.nn.functional.log_softmax(taxonomy_logits, dim=1)
            loss_tax = criterion_tax(tax_log_probs, tax_targets_norm) * cfg.training.taxonomy_feat_weight

        pred_c = torch.expm1(biomass_out[:, 0:1])
        pred_d = torch.expm1(biomass_out[:, 1:2])
        pred_g = torch.expm1(biomass_out[:, 2:3])
        derived_log_total = torch.log1p(pred_c + pred_d + pred_g + 1e-8)
        derived_log_gdm = torch.log1p(pred_c + pred_g + 1e-8)

        targ_c = targets_g[:, 0:1]
        targ_d = targets_g[:, 1:2]
        targ_g = targets_g[:, 2:3]
        targ_total = torch.log1p(targ_c + targ_d + targ_g + 1e-8)
        targ_gdm = torch.log1p(targ_c + targ_g + 1e-8)
        if cfg.loss.use_standardized_loss and bio_mean is not None and bio_std is not None:
            derived_log_total_std = (derived_log_total - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + 1e-9)
            targ_total_std = (targ_total - bio_mean[:, 3:4]) / (bio_std[:, 3:4] + 1e-9)
            derived_log_gdm_std = (derived_log_gdm - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + 1e-9)
            targ_gdm_std = (targ_gdm - bio_mean[:, 4:5]) / (bio_std[:, 4:5] + 1e-9)
            reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
            loss_total = reg(derived_log_total_std, targ_total_std)
            loss_gdm = reg(derived_log_gdm_std, targ_gdm_std)
        else:
            reg = torch.nn.functional.smooth_l1_loss if cfg.loss.reg_loss_type == 'smoothl1' else torch.nn.functional.mse_loss
            loss_total = reg(derived_log_total, targ_total)
            loss_gdm = reg(derived_log_gdm, targ_gdm)

        if cfg.loss.use_weighted_regression_loss and official_weights_t is not None:
            loss_total = loss_total * official_weights_t[3]
            loss_gdm = loss_gdm * official_weights_t[4]
        total_loss = loss_bio + loss_aux + loss_sp + loss_tax + (loss_total + loss_gdm) * cfg.training.biomass_feat_weight

        B = images.size(0)
        metrics[f'{prefix}_loss'] += total_loss.item() * B
        metrics[f'{prefix}_bio'] += loss_bio.item() * B
        metrics[f'{prefix}_aux'] += loss_aux.item() * B
        metrics[f'{prefix}_sp']  += loss_sp.item() * B
        metrics[f'{prefix}_tax'] += loss_tax.item() * B
        metrics[f'{prefix}_loss_c'] += nn.functional.mse_loss(biomass_out[:, 0], targets_log[:, 0]).item() * B
        metrics[f'{prefix}_loss_d'] += nn.functional.mse_loss(biomass_out[:, 1], targets_log[:, 1]).item() * B
        metrics[f'{prefix}_loss_g'] += nn.functional.mse_loss(biomass_out[:, 2], targets_log[:, 2]).item() * B
        metrics[f'{prefix}_loss_t'] += nn.functional.mse_loss(derived_log_total.squeeze(1), targ_total.squeeze(1)).item() * B
        metrics[f'{prefix}_loss_gdm'] += nn.functional.mse_loss(derived_log_gdm.squeeze(1), targ_gdm.squeeze(1)).item() * B
        metrics[f'{prefix}_loss_ndvi'] += nn.functional.mse_loss(aux_out[:, 0], aux_feats[:, 0]).item() * B
        metrics[f'{prefix}_loss_h']    += nn.functional.mse_loss(aux_out[:, 1], aux_feats[:, 1]).item() * B
        metrics[f'{prefix}_loss_int_mul']  += nn.functional.mse_loss(aux_out[:, 2], aux_feats[:, 2]).item() * B
        metrics[f'{prefix}_loss_int_add']  += nn.functional.mse_loss(aux_out[:, 3], aux_feats[:, 3]).item() * B

        preds_full = torch.zeros((biomass_out.size(0), 5), device=biomass_out.device, dtype=biomass_out.dtype)
        preds_full[:, 0:3] = biomass_out
        preds_full[:, 3:4] = derived_log_total
        preds_full[:, 4:5] = derived_log_gdm
        all_preds_log.append(preds_full.cpu())
        targ_c = targets_g[:, 0:1]
        targ_d = targets_g[:, 1:2]
        targ_g = targets_g[:, 2:3]
        targ_total_lin = targ_c + targ_d + targ_g
        targ_gdm_lin = targ_c + targ_g
        targets_full_lin = torch.cat([targ_c, targ_d, targ_g, targ_total_lin, targ_gdm_lin], dim=1)
        all_targets_full.append(targets_full_lin.cpu())

    N = len(loader.dataset)
    final_metrics = {k: v / N for k, v in metrics.items()}
    preds_log = torch.cat(all_preds_log).numpy()
    preds_linear = np.expm1(preds_log)
    targets_linear = torch.cat(all_targets_full).numpy()
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
        'num_species': len(species_list),
        'session_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tile_augmentation': 'enabled',
        'groupby_key': cfg.split.groupby_key,
        'stratification_key': cfg.split.stratification_key,
        'holdout_pct': cfg.split.holdout_pct
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata


def main():
    session_dir = setup_logging(file_name_part="unified_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)

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
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    train_transform, val_transform = get_image_data_transforms()

    # 2. Random holdout (global fraction from the entire dataset)
    holdout_pct = cfg.split.holdout_pct
    species_col = 'species_id' if 'species_id' in df.columns else ('Species' if 'Species' in df.columns else None)
    if species_col is None:
        # Species column is not required for global random holdout; proceed.
        logger.warning("No species column found (expected 'species_id' or 'Species'). Proceeding with global random holdout.")

    k = max(1, int(np.ceil(len(df) * holdout_pct)))
    hold_idx = df.sample(n=k, random_state=42, replace=False).index
    hold_df = df.loc[hold_idx].copy().reset_index(drop=True)
    dev_df = df.drop(hold_idx).copy().reset_index(drop=True)

    log_dataframe_details(logger, dev_df, name="Development Set")
    log_dataframe_details(logger, hold_df, name="Random Holdout Set")

    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Development Set: {len(dev_df)}")
    logger.info(f"Random Holdout: {len(hold_df)}")
    hold_df.to_csv(os.path.join(splits_dir, "global_holdout.csv"), index=False)

    # 3. Choose fold strategy from config keys
    groupby_key = cfg.split.groupby_key
    strat_key = cfg.split.stratification_key
    logger.info(f"Using groupby_key='{groupby_key}' and stratification_key='{strat_key}'")

    # dev/hold already have deterministic grouping keys from pre-split step

    best_overall_score = -float('inf')

    if groupby_key and strat_key:
        # Prefer StratifiedGroupKFold when available
        if groupby_key not in dev_df.columns:
            raise ValueError(f"Configured groupby_key '{groupby_key}' not found in dev_df.")
        # Build strat labels inline if needed (avoid learned features here)
        if strat_key in dev_df.columns:
            y_strat = dev_df[strat_key]
        elif strat_key == 'biomass_binned_composite':
            from sklearn.preprocessing import KBinsDiscretizer
            wts = cfg.targets.official_weights
            tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            comp = np.zeros(len(dev_df), dtype=float)
            for col, wt in zip(tgt_cols, wts):
                if col in dev_df.columns:
                    comp += wt * dev_df[col].astype(float).values
            kbd = KBinsDiscretizer(n_bins=int(getattr(cfg.features, 'biomass_composite_bins', 5)), encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
            y_strat = kbd.fit_transform(comp.reshape(-1, 1)).astype(int).ravel()
        else:
            raise ValueError(f"Configured stratification_key '{strat_key}' not found in dev_df and no inline builder is defined.")
        if _HAS_SGF:
            sgkf = StratifiedGroupKFold(n_splits=cfg.hyperparameters.n_folds, shuffle=True, random_state=42)
            splitter = sgkf.split(dev_df, y=y_strat, groups=dev_df[groupby_key])
            split_name = 'StratifiedGroupKFold'
        else:
            logging.getLogger("System Logger").warning("StratifiedGroupKFold not available; falling back to GroupKFold (no strat balance across folds). Consider upgrading scikit-learn >= 1.1.")
            gkf = GroupKFold(n_splits=cfg.hyperparameters.n_folds)
            splitter = gkf.split(dev_df, groups=dev_df[groupby_key])
            split_name = 'GroupKFold'
        fold_iter = ((dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True)) for train, val in splitter)
    elif groupby_key:
        # GroupKFold path
        if groupby_key not in dev_df.columns:
            raise ValueError(f"Configured groupby_key '{groupby_key}' not found in dev_df.")
        gkf = GroupKFold(n_splits=cfg.hyperparameters.n_folds)
        splitter = gkf.split(dev_df, groups=dev_df[groupby_key])
        fold_iter = ((dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True)) for train, val in splitter)
        split_name = 'GroupKFold'
    elif strat_key:
        # StratifiedKFold path
        if strat_key not in dev_df.columns:
            raise ValueError(f"Configured stratification_key '{strat_key}' not found in dev_df.")
        skf = StratifiedKFold(n_splits=cfg.hyperparameters.n_folds)
        splitter = skf.split(dev_df, dev_df[strat_key])
        fold_iter = ((dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True)) for train, val in splitter)
        split_name = 'StratifiedKFold'
    else:
        # Fallback: simple KFold when no grouping or stratification key is provided
        logger.info("No groupby/stratification key configured; falling back to simple KFold.")
        kf = KFold(n_splits=cfg.hyperparameters.n_folds, shuffle=True, random_state=42)
        splitter = kf.split(dev_df)
        fold_iter = ((dev_df.iloc[train].reset_index(drop=True), dev_df.iloc[val].reset_index(drop=True)) for train, val in splitter)
        split_name = 'KFold'

    per_fold_best = []
    for fold, (train_df_raw, val_df_raw) in enumerate(fold_iter):
        # Fit/Transform pipeline per fold
        ft = BiomassFeatureTransform(logger)
        train_df = ft.fit(train_df_raw)
        val_df = ft.transform(val_df_raw)
        hold_df = ft.transform(hold_df)
        raw_n_train = len(train_df)
        logger.info(f"\n{'='*20} Fold {fold+1}/{cfg.hyperparameters.n_folds} ({split_name}) {'='*20}")
        logger.info(f"Train:   n={len(train_df)}, sessions={train_df['SessionID'].nunique() if 'SessionID' in train_df.columns else 'N/A'}")
        logger.info(f"Val:     n={len(val_df)}, sessions={val_df['SessionID'].nunique() if 'SessionID' in val_df.columns else 'N/A'}")
        logger.info(f"Holdout: n={len(hold_df)}, sessions={hold_df['SessionID'].nunique() if 'SessionID' in hold_df.columns else 'N/A'}")

        if 'State' in train_df.columns:
            logger.info(f"States in Train: {sorted(train_df['State'].unique())}")
            logger.info(f"States in Val:   {sorted(val_df['State'].unique())}")
            logger.info(f"States in Hold:  {sorted(hold_df['State'].unique())}")
            
        # Log species counts across train/val/hold using a formatted table helper
        log_species_table(logger, train_df, val_df, hold_df, species_col=species_col, title='Species in Fold')

        log_fold_details(logger, train_df, val_df)

        if raw_n_train < cfg.hyperparameters.min_train_samples:
            logger.info(f"\nSkipping Fold {fold+1}: Training set too small ({raw_n_train} < {cfg.hyperparameters.min_train_samples})")
            continue

        logger.info(f"Training fold {fold} size: {len(train_df)} (upsampling applied in fit())")
        if 'State_Specie' in train_df.columns:
            logger.info(f"Training set distribution: {train_df['State_Specie'].value_counts()}")

        effective_train_size = len(train_df) * 6
        logger.info(f"\n{'='*40}")
        logger.info(f"EFFECTIVE TRAINING SIZE WITH TILING")
        logger.info(f"Base Samples: {len(train_df)}")
        logger.info(f"With 6x Tile Augmentation: {effective_train_size}")
        logger.info(f"{'='*40}\n")

        train_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir, f"fold{fold+1}_val.csv"), index=False)

        train_ds_base = TiledBiomassDataset(
            train_df,
            transform=train_transform,
            mode='validation',
            tile_prob=cfg.augmentation.tile_prob,
            target_cols=['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
        )
        train_ds = TiledMixupDataset(train_ds_base, prob=cfg.augmentation.mixup_prob, alpha=cfg.augmentation.mixup_alpha)

        val_ds = TiledBiomassDataset(
            val_df,
            transform=val_transform,
            mode='validation',
            tile_prob=0.2,
            target_cols=['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
        )

        holdout_ds = TiledBiomassDataset(
            hold_df,
            transform=val_transform,
            mode='validation',
            tile_prob=0.2,
            target_cols=['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        holdout_loader = DataLoader(holdout_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation', target_cols=['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g'])
        n_aux = dummy_ds[0]['aux_feats'].shape[0]
        model = BiomassUnifiedModel(num_aux=n_aux, config=cfg).to(cfg.device)

        if fold == 0 or fold == 1:
            save_metadata(session_dir, cfg.species_taxonomy.core_species, cfg.targets.cols, n_aux)

        n_upsampled = len(train_df)
        if n_upsampled < cfg.hyperparameters.backbone_freeze_threshold:
            logger.info(f"PROTECTION: Keeping backbone FROZEN for Fold {fold+1} (n_upsampled={n_upsampled} < {cfg.hyperparameters.backbone_freeze_threshold})")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            if cfg.training.freeze_backbone:
                logger.info(f"STRATEGY: Applying Partial Freeze ({cfg.training.backbone_freeze_fraction*100}%) for Fold {fold+1} (n_upsampled={n_upsampled})")
                all_params = list(model.backbone.parameters())
                freeze_until = int(len(all_params) * cfg.training.backbone_freeze_fraction)
                for i, p in enumerate(all_params):
                    p.requires_grad = (i >= freeze_until)
            else:
                logger.info(f"STRATEGY: Full Backbone Unfreeze for Fold {fold+1}")
                for param in model.backbone.parameters():
                    param.requires_grad = True

        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        param_groups = [
            {'params': backbone_params, 'lr': cfg.hyperparameters.learning_rate * cfg.hyperparameters.backbone_lr_factor},
            {'params': head_params, 'lr': cfg.hyperparameters.learning_rate}
        ]
        optimizer = AdamW(param_groups, weight_decay=cfg.hyperparameters.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=5, threshold=1e-3, min_lr=1e-6)

        criterion_reg = nn.MSELoss()
        criterion_species = nn.BCEWithLogitsLoss()
        criterion_tax = nn.KLDivLoss(reduction='batchmean')

        history = defaultdict(list)
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_h_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0

        bio_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
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
        else:
            aux_mean_np = np.array([], dtype=float)
            aux_std_np = np.array([], dtype=float)
        aux_mean_t = torch.tensor(aux_mean_np, dtype=torch.float32, device=cfg.device).view(1, -1) if aux_data.shape[1] > 0 else None
        aux_std_t = torch.tensor(aux_std_np, dtype=torch.float32, device=cfg.device).view(1, -1) if aux_data.shape[1] > 0 else None

        for epoch in range(cfg.hyperparameters.epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_species, criterion_tax,
                cfg, epoch, session_dir=session_dir, logger=logger,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t,
                official_weights_t=official_weights_t
            )

            val_metrics = validate(
                model, val_loader, criterion_reg, criterion_species, criterion_tax, cfg, prefix='val', use_tta=False,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )

            hol_metrics = validate(
                model, holdout_loader, criterion_reg, criterion_species, criterion_tax, cfg,
                prefix='holdout', use_tta=cfg.training.use_tta,
                epoch=epoch, fold=fold, session_dir=session_dir,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t, official_weights_t=official_weights_t
            )

            v_r2 = val_metrics['val_r2']
            h_r2 = hol_metrics['holdout_r2']
            avg_r2 = (v_r2 + h_r2) / 2
            consistency_penalty = 0.5 * abs(v_r2 - h_r2)
            current_score = avg_r2 - consistency_penalty
            score_gap = abs(v_r2 - h_r2)
            scheduler.step(current_score)

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


if __name__ == '__main__':
    main()