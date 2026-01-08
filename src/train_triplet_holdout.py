# train_triplet_holdout.py
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json

# Local Imports
from config.loader import cfg

DEVICE = cfg.device
BATCH_SIZE = cfg.hyperparameters.batch_size
EPOCHS = cfg.hyperparameters.epochs
LEARNING_RATE = cfg.hyperparameters.learning_rate
TAXONOMY_FEAT_WEIGHT = cfg.training.taxonomy_feat_weight
WEIGHT_DECAY = cfg.hyperparameters.weight_decay
EARLY_STOP_PATIENCE = cfg.hyperparameters.early_stop_patience
N_FOLDS = cfg.hyperparameters.n_folds
BIOMASS_FEAT_WEIGHT = cfg.training.biomass_feat_weight
AUX_FEAT_WEIGHT = cfg.training.aux_feat_weight
SPECIES_FEAT_WEIGHT = cfg.training.species_feat_weight
PHYSICS_FEAT_WEIGHT = cfg.training.physics_feat_weight
OFFICIAL_WEIGHTS = cfg.targets.official_weights
CORE_SPECIES = cfg.species_taxonomy.core_species
USE_TTA = cfg.training.use_tta
MIN_TRAIN_SAMPLES = cfg.hyperparameters.min_train_samples
BACKBONE_FREEZE_THRESHOLD = cfg.hyperparameters.backbone_freeze_threshold
FREEZE_BACKBONE = cfg.training.freeze_backbone
BACKBONE_FREEZE_FRACTION = cfg.training.backbone_freeze_fraction
MAX_GRAD_NORM = cfg.hyperparameters.max_grad_norm
BACKBONE_LR_FACTOR = cfg.hyperparameters.backbone_lr_factor
TILE_PROB = cfg.augmentation.tile_prob
MIXUP_PROB = cfg.augmentation.mixup_prob
MIXUP_ALPHA = cfg.augmentation.mixup_alpha

# Loss configuration from YAML
USE_STANDARDIZED_LOSS = cfg.loss.use_standardized_loss
USE_WEIGHTED_REGRESSION_LOSS = cfg.loss.use_weighted_regression_loss
REG_LOSS_TYPE = cfg.loss.reg_loss_type  # 'smoothl1' or 'mse'
OFFICIAL_WEIGHTS_T = None  # initialized in main() with device
from common import (
    load_data, engineer_features, get_image_data_transforms, save_batch_images, 
    set_seed, calculate_global_weighted_r2,
    get_taxonomy_targets,save_tta_images,
    rotate_crop_resize, smart_temporal_split, triple_moving_time_series_split
)

from log_and_plots import (
    log_dataframe_details, setup_logging, plot_training_history, 
    log_fold_details
)

from dataset import TiledBiomassDataset, TiledMixupDataset
from models import BiomassUnifiedModel
from torchvision.utils import save_image


def build_weighted_sampler_from_df(df, key='StratifyKey', cap_quantile=0.95):
    if key not in df.columns or len(df) == 0:
        return None
    counts = df[key].value_counts()
    if counts.empty:
        return None
    w_map = (1.0 / counts).to_dict()
    weights = df[key].map(w_map).astype(float).values
    cap = np.quantile(weights, cap_quantile) if len(weights) > 4 else None
    if cap is not None and np.isfinite(cap):
        weights = np.minimum(weights, cap)
    w_tensor = torch.as_tensor(weights, dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(w_tensor, num_samples=len(df), replacement=True)
    return sampler

def save_tta_images(images, view_name, batch_idx, fold, epoch, session_dir):
    """Save TTA-augmented images for visualization."""
    if fold != 0 or epoch != 0 or batch_idx > 0:
        return
        
    save_dir = os.path.join(session_dir, 'tta_debug', f'fold{fold+1}_ep{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize
    mean = torch.tensor(cfg.preprocessing.imagenet_mean).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(cfg.preprocessing.imagenet_std).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    save_path = os.path.join(save_dir, f'batch{batch_idx}_{view_name}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)


# -----------------------------------------------------------------------------
# TRAINING ENGINE (No changes needed - works with tiled data automatically)
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion_reg, criterion_species, criterion_tax, device, epoch, session_dir=None, logger=None,
                    bio_mean=None, bio_std=None, aux_mean=None, aux_std=None):
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
            # Standardize for loss if enabled
            # Biomass loss (optionally standardized + per-target weighted)
            bio_out_for_loss = biomass_out
            targ_for_loss = targets_log
            if USE_STANDARDIZED_LOSS and bio_mean is not None and bio_std is not None:
                bio_out_for_loss = (biomass_out - bio_mean) / (bio_std + 1e-9)
                targ_for_loss = (targets_log - bio_mean) / (bio_std + 1e-9)

            if USE_WEIGHTED_REGRESSION_LOSS and OFFICIAL_WEIGHTS_T is not None:
                if REG_LOSS_TYPE == 'smoothl1':
                    per_el = F.smooth_l1_loss(bio_out_for_loss, targ_for_loss, reduction='none')  # [B,5]
                else:
                    per_el = F.mse_loss(bio_out_for_loss, targ_for_loss, reduction='none')  # [B,5]
                # Mean over batch, weight across targets, sum → scalar
                per_target_mean = per_el.mean(dim=0)  # [5]
                loss_bio = (per_target_mean * OFFICIAL_WEIGHTS_T).sum() * BIOMASS_FEAT_WEIGHT
            else:
                loss_bio = criterion_reg(bio_out_for_loss, targ_for_loss) * BIOMASS_FEAT_WEIGHT

            if USE_STANDARDIZED_LOSS and aux_mean is not None and aux_std is not None:
                aux_out_std = (aux_out - aux_mean) / (aux_std + 1e-9)
                aux_targ_std = (aux_feats - aux_mean) / (aux_std + 1e-9)
                loss_aux = criterion_reg(aux_out_std, aux_targ_std) * AUX_FEAT_WEIGHT
            else:
                loss_aux = criterion_reg(aux_out, aux_feats) * AUX_FEAT_WEIGHT
            # Species: multi-label → BCEWithLogitsLoss
            loss_sp = criterion_species(species_logits, species_vec) * SPECIES_FEAT_WEIGHT
            # Taxonomy: soft 3-class distribution → KLDivLoss on log-softmax vs normalized targets
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            tax_log_probs = F.log_softmax(taxonomy_logits, dim=1)
            loss_tax = criterion_tax(tax_log_probs, tax_targets_norm) * TAXONOMY_FEAT_WEIGHT
            
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
def validate(model, loader, criterion_reg, criterion_species, criterion_tax, device, prefix='val', use_tta=False, epoch=0, fold=0, session_dir=None,
             bio_mean=None, bio_std=None, aux_mean=None, aux_std=None):
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
            # For species (multi-label), we have probabilities; use BCELoss on probs
            species_probs = avg_sp_probs
            # For taxonomy, use KLDiv on log-probs
            taxonomy_log_probs = torch.log(avg_tax_probs + 1e-9)
            
        else:
            biomass_out, aux_out, species_logits, taxonomy_logits = model(images)
        
        bio_out_for_loss = biomass_out
        targ_for_loss = targets_log
        if USE_STANDARDIZED_LOSS and bio_mean is not None and bio_std is not None:
            bio_out_for_loss = (biomass_out - bio_mean) / (bio_std + 1e-9)
            targ_for_loss = (targets_log - bio_mean) / (bio_std + 1e-9)
        if USE_WEIGHTED_REGRESSION_LOSS and OFFICIAL_WEIGHTS_T is not None:
            if REG_LOSS_TYPE == 'smoothl1':
                per_el = F.smooth_l1_loss(bio_out_for_loss, targ_for_loss, reduction='none')
            else:
                per_el = F.mse_loss(bio_out_for_loss, targ_for_loss, reduction='none')
            per_target_mean = per_el.mean(dim=0)
            loss_bio = (per_target_mean * OFFICIAL_WEIGHTS_T).sum() * BIOMASS_FEAT_WEIGHT
        else:
            loss_bio = criterion_reg(bio_out_for_loss, targ_for_loss) * BIOMASS_FEAT_WEIGHT

        if USE_STANDARDIZED_LOSS and aux_mean is not None and aux_std is not None:
            aux_out_std = (aux_out - aux_mean) / (aux_std + 1e-9)
            aux_targ_std = (aux_feats - aux_mean) / (aux_std + 1e-9)
            loss_aux = criterion_reg(aux_out_std, aux_targ_std) * AUX_FEAT_WEIGHT
        else:
            loss_aux = criterion_reg(aux_out, aux_feats) * AUX_FEAT_WEIGHT
        if use_tta:
            # BCE on probabilities for species when using TTA-averaged probs
            loss_sp = nn.BCELoss()(species_probs, species_vec) * SPECIES_FEAT_WEIGHT
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            loss_tax = criterion_tax(taxonomy_log_probs, tax_targets_norm) * TAXONOMY_FEAT_WEIGHT
        else:
            # BCEWithLogits on raw logits for species
            loss_sp = criterion_species(species_logits, species_vec) * SPECIES_FEAT_WEIGHT
            # KLDiv on log-softmax for taxonomy
            tax_targets_norm = taxonomy_targets / (taxonomy_targets.sum(dim=1, keepdim=True) + 1e-9)
            tax_log_probs = F.log_softmax(taxonomy_logits, dim=1)
            loss_tax = criterion_tax(tax_log_probs, tax_targets_norm) * TAXONOMY_FEAT_WEIGHT
        
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
        'tile_augmentation': 'enabled'
    }
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    session_dir = setup_logging(file_name_part="triplet_holdout")
    logger = logging.getLogger("System Logger")
    set_seed(42, logger)
    
    logger.info("="*70)
    logger.info("TILE-BASED AUGMENTATION ENABLED")
    logger.info("Each training sample generates 6 views:")
    logger.info("  1x Original + 1x Stitched + 4x Divided Tiles")
    logger.info("Effective Training Set Size: N_samples × 6")
    logger.info("="*70)
    logger.info(cfg)
    
    # 1. Load Data
    df = load_data(logger)
    df = engineer_features(df, logger)

    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    logger.info("Data sorted by Sampling_Date.")
    
    species_list = CORE_SPECIES
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

    # Sort full df by Sampling_Date
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    
    splits_dir = os.path.join(session_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # 2. Prepare Data Transforms
    train_transform, val_transform = get_image_data_transforms()
    
    best_overall_score = -float('inf')
    stratification_col = 'StratifyKey'
    
    # 3. Triple Moving Window Split
    for fold, (train_idx, val_idx, hold_idx) in enumerate(triple_moving_time_series_split(df, n_splits=N_FOLDS)):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        hold_df = df.iloc[hold_idx].copy()
        
        raw_n_train = len(train_df)
        
        logger.info(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} (Triple Moving Window) {'='*20}")
        logger.info(f"Train:   {train_df['Sampling_Date'].min().date()} -> {train_df['Sampling_Date'].max().date()} (n={len(train_df)}, sessions={train_df['SessionID'].nunique()})")
        logger.info(f"Val:     {val_df['Sampling_Date'].min().date()} -> {val_df['Sampling_Date'].max().date()} (n={len(val_df)}, sessions={val_df['SessionID'].nunique()})")
        logger.info(f"Holdout: {hold_df['Sampling_Date'].min().date()} -> {hold_df['Sampling_Date'].max().date()} (n={len(hold_df)}, sessions={hold_df['SessionID'].nunique()})")
        
        logger.info(f"States in Train: {train_df['State'].unique().tolist()}")
        logger.info(f"States in Val:   {val_df['State'].unique().tolist()}")
        logger.info(f"States in Hold:  {hold_df['State'].unique().tolist()}")
        
        log_dataframe_details(logger, hold_df, name="Holdout Set")
        log_fold_details(logger, train_df, val_df)

        if raw_n_train < MIN_TRAIN_SAMPLES:
            logger.info(f"\nSkipping Fold {fold+1}: Training set too small ({raw_n_train} < {MIN_TRAIN_SAMPLES})")
            continue
 
        # Upsampling is now handled in load_data function
        logger.info(f"Training set size: {len(train_df)} (upsampling applied in load_data)")
        logger.info(f"Training set distribution: {train_df['StratifyKey'].value_counts()}")
        
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
        
        # Compute per-fold means/stds for standardized-loss (biomass in log space, aux features)
        bio_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        bio_train_log = np.log1p(train_df[bio_cols].astype(float).values)
        bio_mean_np = bio_train_log.mean(axis=0)
        bio_std_np = bio_train_log.std(axis=0)
        bio_mean_t = torch.tensor(bio_mean_np, dtype=torch.float32, device=DEVICE).view(1, -1)
        bio_std_t = torch.tensor(bio_std_np, dtype=torch.float32, device=DEVICE).view(1, -1)
        # Prepare official weights tensor on device
        global OFFICIAL_WEIGHTS_T
        OFFICIAL_WEIGHTS_T = torch.tensor(OFFICIAL_WEIGHTS, dtype=torch.float32, device=DEVICE)

        # Reconstruct aux column list similar to dataset
        base_aux = ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Interaction_Add']
        ordinal_cols = ['NDVI_Bin_Ordinal', 'Height_Bin_Ordinal']
        onehot_cols = [f'NDVI_Bin_OH_{k}' for k in range(4)] + [f'Height_Bin_OH_{k}' for k in range(4)]
        aux_cols = [c for c in base_aux if c in train_df.columns]
        for c in ordinal_cols + onehot_cols:
            if c in train_df.columns:
                aux_cols.append(c)
        if 'Species_Count' in train_df.columns:
            aux_cols.append('Species_Count')
        # Compute aux stats robustly
        aux_data = train_df[aux_cols].astype(float).fillna(0.0).values if len(aux_cols) > 0 else np.zeros((len(train_df), 0), dtype=float)
        if aux_data.shape[1] > 0:
            aux_mean_np = aux_data.mean(axis=0)
            aux_std_np = aux_data.std(axis=0)
        else:
            aux_mean_np = np.array([], dtype=float)
            aux_std_np = np.array([], dtype=float)
        aux_mean_t = torch.tensor(aux_mean_np, dtype=torch.float32, device=DEVICE).view(1, -1) if aux_data.shape[1] > 0 else None
        aux_std_t = torch.tensor(aux_std_np, dtype=torch.float32, device=DEVICE).view(1, -1) if aux_data.shape[1] > 0 else None

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
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        holdout_ds = TiledBiomassDataset(
            hold_df, 
            transform=val_transform,
            mode='holdout',  # Disables tiling
            tile_prob=0.0
        )
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        dummy_ds = TiledBiomassDataset(train_df[:1], transform=train_transform, mode='validation')
        n_aux = dummy_ds[0]['aux_feats'].shape[0]
        
        model = BiomassUnifiedModel(num_aux=n_aux, config=cfg).to(DEVICE)
        
        if fold == 0:
            save_metadata(session_dir, CORE_SPECIES, cfg.targets.cols, n_aux)
        
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
        # Species: multi-label classification over CORE_SPECIES
        criterion_species = nn.BCEWithLogitsLoss()
        # Taxonomy: soft distribution across 3 groups
        criterion_tax = nn.KLDivLoss(reduction='batchmean')
        
        history = defaultdict(list)
        best_fold_score = -float('inf')
        best_fold_v_r2 = -float('inf')
        best_fold_h_r2 = -float('inf')
        best_fold_epoch = -1
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion_reg, criterion_species, criterion_tax,
                DEVICE, epoch, session_dir=session_dir, logger=logger,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t
            )
            
            # Validate
            val_metrics = validate(
                model, val_loader, criterion_reg, criterion_species, criterion_tax, DEVICE,
                prefix='val', use_tta=False,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t
            )
            
            # Holdout
            hol_metrics = validate(
                model, holdout_loader, criterion_reg, criterion_species, criterion_tax, DEVICE,
                prefix='holdout', use_tta=USE_TTA,
                epoch=epoch, fold=fold, session_dir=session_dir,
                bio_mean=bio_mean_t, bio_std=bio_std_t, aux_mean=aux_mean_t, aux_std=aux_std_t
            )
            
            # Custom Score
            v_r2 = val_metrics['val_r2']
            h_r2 = hol_metrics['holdout_r2']
            
            avg_r2 = (v_r2 + h_r2) / 2
            consistency_penalty = 0.5 * abs(v_r2 - h_r2)
            current_score = avg_r2 - consistency_penalty
            
            score_gap = abs(v_r2 - h_r2)
            scheduler.step(current_score)
            
            log_msg = (
                f"Ep {epoch} | "
                f"T_Loss: {train_metrics['train_loss']:.3f} "
                f"(bio:{train_metrics.get('train_bio', 0.0):.3f}, aux:{train_metrics.get('train_aux', 0.0):.3f}, "
                f"sp:{train_metrics.get('train_sp', 0.0):.3f}, tax:{train_metrics.get('train_tax', 0.0):.3f}, phy:{train_metrics.get('train_phy', 0.0):.3f}) | "
                f"V_Loss: {val_metrics['val_loss']:.3f} "
                f"(bio:{val_metrics.get('val_bio', 0.0):.3f}, aux:{val_metrics.get('val_aux', 0.0):.3f}, "
                f"sp:{val_metrics.get('val_sp', 0.0):.3f}, tax:{val_metrics.get('val_tax', 0.0):.3f}, phy:{val_metrics.get('val_phy', 0.0):.3f}) | "
                f"H_Loss: {hol_metrics['holdout_loss']:.3f} "
                f"(bio:{hol_metrics.get('holdout_bio', 0.0):.3f}, aux:{hol_metrics.get('holdout_aux', 0.0):.3f}, "
                f"sp:{hol_metrics.get('holdout_sp', 0.0):.3f}, tax:{hol_metrics.get('holdout_tax', 0.0):.3f}, phy:{hol_metrics.get('holdout_phy', 0.0):.3f}) | "
                f"T_R2: {train_metrics['train_r2']:.4f} | V_R2: {v_r2:.4f} | H_R2: {h_r2:.4f} | "
                f"Score: {current_score:.4f} | Gap: {score_gap:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}"
            )
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
                    # Log high priority save
                    logger.info(f"!!!!! NEW OVERALL BEST MODEL: Fold {fold+1}, Score {best_overall_score:.4f} !!!!!")
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