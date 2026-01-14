# common.py
import os
import math
import random
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from PIL import ImageFilter
from sklearn.preprocessing import KBinsDiscretizer

from config.loader import cfg

IMAGENET_DEFAULT_MEAN = cfg.preprocessing.imagenet_mean
IMAGENET_DEFAULT_STD = cfg.preprocessing.imagenet_std
IMAGE_HEIGHT = cfg.preprocessing.image_height
IMAGE_WIDTH = cfg.preprocessing.image_width
CORE_SPECIES = cfg.species_taxonomy.core_species
GROUP_DEFINITIONS = cfg.species_taxonomy.groups
N_FOLDS = cfg.hyperparameters.n_folds
TAXONOMY_IDXS = cfg.species_taxonomy.taxonomy_idxs
UPSAMPLE_CONFIG = {
    'enabled': cfg.upsample.enabled,
    'target_min_samples': cfg.upsample.target_min_samples,
    'method': cfg.upsample.method,
    'noise_scale': cfg.upsample.noise_scale,
    'seasonal_drift': cfg.upsample.seasonal_drift,
    'day_shift_prob': cfg.upsample.day_shift_prob,
    'drift_strength': cfg.upsample.drift_strength
}
SPLIT_CONFIG = {
    'holdout_pct': cfg.split.holdout_pct,
}
SEASON_MONTH_MAP = cfg.seasons.month_map
SEASONAL_DRIFT = cfg.seasons.drift
USE_BIN_FEATURES = cfg.features.use_bin_features
BIN_ENCODING = cfg.features.bin_encoding
USE_SPECIES_COUNT_FEATURE = cfg.features.use_species_count_feature

from configs import get_key1_specie_pair

def get_largest_rotated_crop(h: int, w: int, angle: float) -> Tuple[int, int]:
    """
    Calculates the dimensions of the largest axis-aligned rectangle 
    that fits inside a rotated image without including any borders/artifacts.
    
    Math: W_new = W / (sin(a) + cos(a)) for a square.
    """
    angle_rad = math.radians(abs(angle))
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    
    # Calculate scale factor to remain within the valid image area
    scale = 1.0 / (cos_a + sin_a)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    return new_h, new_w

def rotate_crop_resize(img: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Deterministic transformation for TTA.
    1. Rotates the image.
    2. Crops to the largest valid center (zooming in).
    3. Resizes back to original dimensions.
    """
    # Handle inputs
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2:]
    else:
        # Fallback for PIL (though we usually pass Tensors in TTA)
        w, h = img.size

    # 1. Rotate (Bilinear matches training behavior)
    img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
    
    # 2. Calculate valid crop
    ch, cw = get_largest_rotated_crop(h, w, angle)
    
    # 3. Center Crop (The "Zoom")
    img_crop = TF.center_crop(img_rot, [ch, cw])
    
    # 4. Resize back (The "upsample")
    # align_corners=False prevents sub-pixel phase shifts
    if img.ndim == 4:
        # Batched input (B, C, H, W) -> Directly interpolate
        img_resized = torch.nn.functional.interpolate(
            img_crop, size=(h, w), mode='bilinear', align_corners=False
        )
    else:
        # Single image (C, H, W) -> Unsqueeze to (1, C, H, W)
        img_resized = torch.nn.functional.interpolate(
            img_crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
    
    return img_resized

class RandomRotateCropResize(nn.Module):
    """
    Biomass-Safe Rotation for Training.
    
    Standard rotation introduces black corners.
    Standard RandomResizedCrop changes density (mass/pixel) too aggressively.
    
    This transform mimics the Inference TTA:
    It rotates and 'zooms' into the valid area. This forces the model to learn
    density estimation even when the field of view changes slightly.
    """
    def __init__(self, degrees=30):
        super().__init__()
        self.degrees = degrees

    def forward(self, img):
        # 1. Pick Random Angle
        angle = random.uniform(-self.degrees, self.degrees)
        
        # 2. Rotate (Fill with Mean Color ~ Gray/Brown to match TTA/ImageNet Mean)
        # ImageNet Mean (0.485, 0.456, 0.406) * 255 ~= (124, 116, 104)
        img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=(124, 116, 104))
        
        # 3. Get Dimensions
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size
            
        # 4. Crop Valid Area
        ch, cw = get_largest_rotated_crop(h, w, angle)
        img_crop = TF.center_crop(img_rot, [ch, cw])
        
        # 5. Resize Back
        img_final = TF.resize(
            img_crop, [h, w], 
            interpolation=transforms.InterpolationMode.BILINEAR, 
            antialias=True
        )
        
        return img_final

class SubtleSharpen:
    """
    Applies subtle sharpening to grass images.
    Uses PIL's UnsharpMask filter with conservative parameters.
    """
    def __init__(self, probability=0.5, radius=1, percent=50, threshold=3):
        """
        Args:
            probability: Chance to apply sharpening (0.0 to 1.0)
            radius: Sharpening radius (1-2 is subtle for grass)
            percent: Sharpening strength (50-100 is gentle)
            threshold: Minimum brightness change to sharpen (higher = less aggressive)
        """
        self.probability = probability
        self.radius = radius
        self.percent = percent
        self.threshold = threshold
    
    def __call__(self, img):
        if random.random() < self.probability:
            return img.filter(ImageFilter.UnsharpMask(
                radius=self.radius,
                percent=self.percent,
                threshold=self.threshold
            ))
        return img
    

def get_image_data_transforms():
    """
    Safe Transforms.
    """
    train_transform = transforms.Compose([
        # 1. Ensure Baseline Resolution
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        SubtleSharpen(probability=0.5, radius=1, percent=50, threshold=3),
        # 2. Geometry (Manifold Alignment with TTA)
        transforms.RandomApply([
            RandomRotateCropResize(degrees=5),    
            RandomRotateCropResize(degrees=-5),
        ], p=0.5),     
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        # 3. Spatial 
        # Hue/Sat are sensitive for "Dead vs Green" classification.
        # Brightness/Contrast simulate time-of-day/clouds safely.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),

        # 4. Normalization
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return train_transform, val_transform

def load_data(logger):
    """
    Load and preprocess train.csv.
    This creates the initial wide dataframe with all raw features and targets.
    """
    logger.info("Loading and Pivoting Data...")
    if not os.path.exists('train.csv'):
        raise FileNotFoundError("train.csv not found in current directory")
    
    df = pd.read_csv('train.csv')
    df['clean_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
    # Pivot Targets
    targets = df.pivot_table(
        index='clean_id', 
        columns='target_name', 
        values='target',
        aggfunc='max' 
    ).reset_index()
    
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for col in target_cols:
        if col not in targets.columns: 
            targets[col] = 0.0
    targets[target_cols] = targets[target_cols].fillna(0.0)

    # Merge Metadata
    meta_cols = ['clean_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_meta_cols = [c for c in meta_cols if c in df.columns]
    
    meta = df[valid_meta_cols].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # Parse Dates
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'], format='mixed', dayfirst=False)
    
    # Basic numeric conversion
    wide['Height_Ave_cm'] = pd.to_numeric(wide['Height_Ave_cm'], errors='coerce').fillna(0)
    wide['Pre_GSHH_NDVI'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce').fillna(0)
    
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    wide[target_cols] = wide[target_cols].astype(float)

    logger.info(f"Data Loaded. Rows: {len(wide)}")
    return wide

def engineer_features(wide, logger):
    """
    Perform feature engineering on the wide dataframe.
    This includes species parsing, upsampling, and derived features.
    """
    logger.info("Engineering Features...")
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

    logger.info("Parsing species to vectors...")
    # Initialize columns for each core species
    wide['Species'] = wide['Species'].str.lower()
    for sp in CORE_SPECIES:
        wide[f'Species_{sp}'] = 0.0

    # Apply species parsing
    species_vectors = wide['Species'].apply(parse_species_to_vector)
    species_matrix = np.stack(species_vectors.values)
    for i, sp in enumerate(CORE_SPECIES):
        wide[f'Species_{sp}'] = species_matrix[:, i]

    wide = assign_functional_groups(wide)
    
    wide['State_Species'] = wide.apply(lambda row: get_key1_specie_pair(row, key1='State'), axis=1)
    
    logger.info("Applying smart upsampling with NDVI and Height augmentation...")
    wide = apply_smart_upsample_with_features(wide, logger)
    
    # Feature Engineering (Auxiliary Inputs)
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'])
    wide['Interaction_Mul'] = wide['Pre_GSHH_NDVI'] * wide['Height_Ave_cm_log']
    wide['Interaction_Add'] = wide['Pre_GSHH_NDVI'] + wide['Height_Ave_cm_log']

    # Species richness (per-sample count) and soft labels
    # Always compute an internal count for soft-label normalization
    species_cols_all = [f'Species_{sp}' for sp in CORE_SPECIES if f'Species_{sp}' in wide.columns]
    if len(species_cols_all) > 0:
        # Count of present species (sum of one-hot/multi-hot entries)
        species_count_internal = wide[species_cols_all].sum(axis=1).astype(float)
        # Avoid divide-by-zero; if zero, fall back to count=1 so probs remain 0
        species_count_internal = species_count_internal.replace(0.0, 1.0)
        # Create soft probability columns SpeciesProb_{sp}
        for sp in CORE_SPECIES:
            col = f'Species_{sp}'
            if col in wide.columns:
                wide[f'SpeciesProb_{sp}'] = wide[col].astype(float) / species_count_internal
            else:
                wide[f'SpeciesProb_{sp}'] = 0.0

        # Optionally expose Species_Count feature according to config (richness excludes generic 'clover')
        if USE_SPECIES_COUNT_FEATURE:
            richness_cols = [f'Species_{sp}' for sp in CORE_SPECIES if sp != 'clover' and f'Species_{sp}' in wide.columns]
            if len(richness_cols) > 0:
                wide['Species_Count'] = wide[richness_cols].sum(axis=1).astype(float)

    # Composite Biomass Stratification Bins (KBinsDiscretizer: ordinal/quantile)
    try:
        wts = cfg.targets.official_weights
        tgt_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        comp = np.zeros(len(wide), dtype=float)
        for col, wt in zip(tgt_cols, wts):
            if col in wide.columns:
                comp += wt * wide[col].astype(float).values
        n_bins = int(getattr(cfg.features, 'biomass_composite_bins', 5))
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
        bins = kbd.fit_transform(comp.reshape(-1, 1)).astype(int).ravel()
        # Guard against fewer effective bins due to duplicates; still store as int labels
        wide['biomass_binned_composite'] = bins
    except Exception:
        # Fallback: single-bin if distribution too uniform/small
        wide['biomass_binned_composite'] = 0

    # Quantile Bin Features
    if USE_BIN_FEATURES:
        try:
            ndvi_bins_ord = None
            height_bins_ord = None
            if 'Pre_GSHH_NDVI' in wide.columns:
                ndvi_bins_ord = pd.qcut(wide['Pre_GSHH_NDVI'], q=4, labels=False, duplicates='drop').astype(int)
            if 'Height_Ave_cm' in wide.columns:
                height_bins_ord = pd.qcut(wide['Height_Ave_cm'], q=4, labels=False, duplicates='drop').astype(int)

            if BIN_ENCODING == 'ordinal':
                if ndvi_bins_ord is not None:
                    wide['NDVI_Bin_Ordinal'] = ndvi_bins_ord.astype(float)
                if height_bins_ord is not None:
                    wide['Height_Bin_Ordinal'] = height_bins_ord.astype(float)
            elif BIN_ENCODING == 'onehot':
                if ndvi_bins_ord is not None:
                    for k in range(4):
                        wide[f'NDVI_Bin_OH_{k}'] = (ndvi_bins_ord == k).astype(float)
                if height_bins_ord is not None:
                    for k in range(4):
                        wide[f'Height_Bin_OH_{k}'] = (height_bins_ord == k).astype(float)
        except Exception:
            if BIN_ENCODING == 'ordinal':
                if 'Pre_GSHH_NDVI' in wide.columns and 'NDVI_Bin_Ordinal' not in wide.columns:
                    wide['NDVI_Bin_Ordinal'] = 1.0
                if 'Height_Ave_cm' in wide.columns and 'Height_Bin_Ordinal' not in wide.columns:
                    wide['Height_Bin_Ordinal'] = 1.0
            elif BIN_ENCODING == 'onehot':
                for k in range(4):
                    wide[f'NDVI_Bin_OH_{k}'] = 1.0 if k == 1 else 0.0
                    wide[f'Height_Bin_OH_{k}'] = 1.0 if k == 1 else 0.0
    
    wide['SessionID'] = wide.apply(lambda r: f"{r['State']}_{pd.to_datetime(r['Sampling_Date']).strftime('%Y%m%d')}", axis=1)
    wide['Season'] = wide['Sampling_Date'].apply(get_season)
    wide['State_Species'] = wide.apply(lambda row: get_key1_specie_pair(row, key1='State'), axis=1)
    wide["Season_State_Species"] = wide.apply(lambda r: f"{r['Season']}_{r['State_Species']}", axis=1)
    wide["State_Season"] = wide.apply(lambda r: f"{r['State']}_{r['Season']}", axis=1)
    wide['Season_Species'] = wide.apply(lambda row: get_key1_specie_pair(row, key1='Season'), axis=1)
    wide['Species_Season'] = wide.apply(lambda row: get_key1_specie_pair(row, key1='Season', flip=True), axis=1)
    wide['Species_Sampling_Date'] = wide.apply(lambda row: get_key1_specie_pair(row, key1='Sampling_Date',flip=True), axis=1)
    wide['State_Sampling_Date'] = wide.apply(lambda r: f"{r['State']}_{r['Sampling_Date']}", axis=1)
    wide['Season_Sampling_Date'] = wide.apply(lambda r: f"{r['Season']}_{r['Sampling_Date']}", axis=1)
                
    logger.info(f"Feature Engineering Complete. Rows: {len(wide)}")
    wide.to_csv('wide.csv', index=False)
    return wide


def parse_species_to_vector(species_str):
    """
    Parse species string to 14-dim binary vector.
    Handles mixtures like 'Ryegrass_Clover' and 'Mixed'.
    """
    if pd.isna(species_str):
        return np.zeros(14, dtype=np.float32)
    
    species_str = str(species_str).lower().replace(' ', '')
    vec = np.zeros(14, dtype=np.float32)
    
    # Special case: Mixed = all species
    if species_str == 'mixed':
        return np.ones(14, dtype=np.float32)
    
    # Split by underscore and expand 'clover'
    parts = species_str.split('_')
    for part in parts:
        if part == 'clover':
            # Expand to 4 sub-types
            for idx, sp in enumerate(CORE_SPECIES):
                if 'clover' in sp:
                    vec[idx] = 1.0
        else:
            # Direct match
            for idx, sp in enumerate(CORE_SPECIES):
                if part == sp:
                    vec[idx] = 1.0
    
    return vec

def add_species_columns(df):
    """
    Add Species_{name} columns for each core species.
    This is used by the dataset class.
    """
    for idx, sp in enumerate(CORE_SPECIES):
        df[f'Species_{sp}'] = df['Species'].apply(
            lambda x: parse_species_to_vector(x)[idx]
        )
    return df


def coverage_aware_split(df, stratify_col='Species_Season', 
                         min_train_per_combo=2, holdout_pct=0.15, 
                         random_state=42, logger=None,
                         ensure_species_train_coverage=True, species_col='Species',
                         min_train_per_species=1,
                         ensure_combo_train_coverage=True, combo_col='Season_State_Species',
                         min_train_per_combo_key=1):
    """
    Coverage-prioritized splitting that guarantees minimum training 
    representation for every stratification group.
    
    Algorithm:
     1. Group by stratify_col (e.g., Species_Season)
    3. For each group:
       - If group size <= min_train_per_combo: ALL samples go to training
         - Else: Reserve min_train_per_combo for training (randomized), rest are holdout candidates
     4. Sample holdout from candidates (randomized)
    5. Remaining candidates + reserved samples = training
    
    Args:
        df: Input dataframe with all samples
        stratify_col: Column for stratification (default: 'Species_Season')
        min_train_per_combo: Minimum samples per combo reserved for training
        holdout_pct: Target percentage for holdout set
        random_state: Random seed for reproducibility
        logger: Optional logger for diagnostics
        
    Returns:
        train_df, holdout_df
    """
    np.random.seed(random_state)
    
    # Ensure stratify column exists
    if stratify_col not in df.columns:
        if logger:
            logger.warning(f"Column '{stratify_col}' not found. Creating it from Species + Season.")
        if 'Species' in df.columns and 'Season' in df.columns:
            df = df.copy()
            df[stratify_col] = df['Species'].astype(str) + '_' + df['Season'].astype(str)
        else:
            raise ValueError(f"Cannot create {stratify_col}: missing Species or Season columns")
    
    # Get unique stratification groups
    groups = df[stratify_col].unique()
    n_total = len(df)
    target_holdout = int(n_total * holdout_pct)
    
    reserved_train_idx = []  # Guaranteed training samples
    holdout_candidates_idx = []  # Pool for holdout selection
    
    coverage_stats = {'full_coverage': [], 'partial_coverage': [], 'sparse': []}
    
    for group in groups:
        group_mask = df[stratify_col] == group
        group_idx = df[group_mask].index.tolist()
        n_group = len(group_idx)
        
        if n_group <= min_train_per_combo:
            # Sparse group: ALL go to training to ensure coverage
            reserved_train_idx.extend(group_idx)
            coverage_stats['sparse'].append((group, n_group))
        else:
            # Reserve minimum for training (randomized), rest are holdout candidates
            np.random.shuffle(group_idx)
            reserved_train_idx.extend(group_idx[:min_train_per_combo])
            holdout_candidates_idx.extend(group_idx[min_train_per_combo:])
            
            if n_group >= 2 * min_train_per_combo:
                coverage_stats['full_coverage'].append((group, n_group))
            else:
                coverage_stats['partial_coverage'].append((group, n_group))
    
    # Species-level training coverage enforcement before holdout sampling
    # Ensures no species ends up unseen in training
    if ensure_species_train_coverage and species_col in df.columns:
        # Build current coverage sets
        species_series = df[species_col].astype(str)
        train_species_now = set(species_series.loc[reserved_train_idx].unique())
        all_species = set(species_series.unique())
        missing_species = [s for s in all_species if s not in train_species_now]

        moved_count = 0
        moved_detail = []
        ordered_candidates = holdout_candidates_idx[:]

        # Index mapping for quick lookups
        species_by_idx = species_series.to_dict()

        for sp in missing_species:
            # Collect candidates of this species
            sp_candidates = [idx for idx in ordered_candidates if species_by_idx.get(idx) == sp]
            if len(sp_candidates) == 0:
                # If a species has no candidates, it may already be fully reserved due to sparsity
                continue
            take_n = min(min_train_per_species, len(sp_candidates))
            take_idxs = sp_candidates[:take_n]

            # Move selected indices from candidate pool into reserved train
            reserved_train_idx.extend(take_idxs)
            holdout_candidates_idx = [idx for idx in holdout_candidates_idx if idx not in take_idxs]
            moved_count += len(take_idxs)
            moved_detail.append((sp, len(take_idxs)))

        if logger and moved_count > 0:
            logger.info(f"\nSpecies coverage enforcement: moved {moved_count} samples into training to cover missing species.")
            for sp, cnt in moved_detail[:10]:
                logger.info(f"  + {sp}: {cnt} sample(s)")
            if len(moved_detail) > 10:
                logger.info(f"  ... and {len(moved_detail)-10} more species")

    # Species-level training coverage enforcement before holdout sampling
    n_candidates = len(holdout_candidates_idx)
    n_holdout = min(target_holdout, n_candidates)
    
    if n_holdout > 0:
        holdout_idx = np.random.choice(holdout_candidates_idx, size=n_holdout, replace=False).tolist()
        remaining_candidates = [idx for idx in holdout_candidates_idx if idx not in holdout_idx]
    else:
        holdout_idx = []
        remaining_candidates = holdout_candidates_idx
    
    # Combo-level (Season-State-Species) coverage enforcement
    if ensure_combo_train_coverage:
        # Ensure the combo column exists; create if possible
        if combo_col not in df.columns:
            # Try to construct from Season, State, Species
            needed = ['Season', 'State', 'Species']
            if all(c in df.columns for c in needed):
                df = df.copy()
                df[combo_col] = df.apply(lambda r: f"{r['Season']}_{r['State']}_{str(r['Species'])}", axis=1)
            else:
                if logger:
                    logger.warning(f"Combo coverage requested but '{combo_col}' missing and cannot be constructed; skipping.")
                ensure_combo_train_coverage = False

    if ensure_combo_train_coverage and combo_col in df.columns:
        # Determine missing combos in training
        all_combos = set(df[combo_col].astype(str).unique())
        train_combos_now = set(df.loc[reserved_train_idx, combo_col].astype(str).unique())
        missing_combos = [c for c in all_combos if c not in train_combos_now]

        # Build map from index to combo for fast lookup
        combo_by_idx = df[combo_col].astype(str).to_dict()

        moved_combo_cnt = 0
        moved_combo_detail = []
        for cmb in missing_combos:
            # Find candidates in remaining pool with this combo
            cmb_candidates = [idx for idx in remaining_candidates if combo_by_idx.get(idx) == cmb]
            if len(cmb_candidates) == 0:
                # Try also from holdout_idx (if absolutely necessary, pull back one)
                alt_candidates = [idx for idx in holdout_idx if combo_by_idx.get(idx) == cmb]
                if len(alt_candidates) == 0:
                    continue
                take_n = min(min_train_per_combo_key, len(alt_candidates))
                take_idxs = alt_candidates[:take_n]
                # Move from holdout back to train
                reserved_train_idx.extend(take_idxs)
                holdout_idx = [idx for idx in holdout_idx if idx not in take_idxs]
            else:
                take_n = min(min_train_per_combo_key, len(cmb_candidates))
                take_idxs = cmb_candidates[:take_n]
                reserved_train_idx.extend(take_idxs)
                remaining_candidates = [idx for idx in remaining_candidates if idx not in take_idxs]
            moved_combo_cnt += len(take_idxs)
            moved_combo_detail.append((cmb, len(take_idxs)))

        if logger and moved_combo_cnt > 0:
            logger.info(f"\nCombo coverage enforcement: moved {moved_combo_cnt} samples into training to cover missing {combo_col} combos.")
            for cmb, cnt in moved_combo_detail[:10]:
                logger.info(f"  + {cmb}: {cnt} sample(s)")
            if len(moved_combo_detail) > 10:
                logger.info(f"  ... and {len(moved_combo_detail)-10} more combos")

    # Training = reserved + remaining candidates
    train_idx = reserved_train_idx + remaining_candidates
    
    # Create dataframes
    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    holdout_df = df.loc[holdout_idx].copy().reset_index(drop=True) if holdout_idx else pd.DataFrame()
    
    # Log coverage statistics
    if logger:
        logger.info(f"\n{'='*50}")
        logger.info(f"COVERAGE-AWARE SPLIT RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Total samples: {n_total}")
        logger.info(f"Training samples: {len(train_df)} ({100*len(train_df)/n_total:.1f}%)")
        logger.info(f"Holdout samples: {len(holdout_df)} ({100*len(holdout_df)/n_total:.1f}%)")
        logger.info(f"\nStratification groups ({stratify_col}): {len(groups)}")
        logger.info(f"  Full coverage (n >= {2*min_train_per_combo}): {len(coverage_stats['full_coverage'])}")
        logger.info(f"  Partial coverage: {len(coverage_stats['partial_coverage'])}")
        logger.info(f"  Sparse (all in train): {len(coverage_stats['sparse'])}")
        
        if coverage_stats['sparse']:
            logger.info(f"\nSparse groups (100% in training):")
            for grp, cnt in coverage_stats['sparse'][:10]:
                logger.info(f"    {grp}: {cnt} samples")
            if len(coverage_stats['sparse']) > 10:
                logger.info(f"    ... and {len(coverage_stats['sparse'])-10} more")
        
        # Verify coverage
        train_groups = set(train_df[stratify_col].unique())
        missing_in_train = set(groups) - train_groups
        if missing_in_train:
            logger.warning(f"WARNING: Groups missing from training: {missing_in_train}")
        else:
            logger.info(f"\n✓ All {len(groups)} groups represented in training")
        # Verify combo coverage
        if ensure_combo_train_coverage and combo_col in df.columns:
            all_combos = set(df[combo_col].astype(str).unique())
            train_combos = set(train_df[combo_col].astype(str).unique())
            missing_train_combos = all_combos - train_combos
            if missing_train_combos:
                logger.warning(f"WARNING: {combo_col} combos missing from training despite enforcement: {missing_train_combos}")
            else:
                logger.info(f"\n✓ Combo coverage: all {len(all_combos)} {combo_col} combos represented in training")

        # Verify species-level coverage if requested
        if ensure_species_train_coverage and species_col in train_df.columns and species_col in df.columns:
            species_all = set(df[species_col].astype(str).unique())
            species_train = set(train_df[species_col].astype(str).unique())
            missing_species_train = species_all - species_train
            if missing_species_train:
                logger.warning(f"WARNING: Species missing from training despite enforcement: {missing_species_train}")
            else:
                logger.info(f"\n✓ Species coverage: all {len(species_all)} species represented in training")
    
    return train_df, holdout_df


def get_season(date_val):
    """
    Map a timestamp to Australian meteorological seasons.
    Returns one of: 'summer','autumn','winter','spring'.
    """
    if pd.isna(date_val):
        return 'spring'
    month = int(pd.to_datetime(date_val).month)
    return SEASON_MONTH_MAP.get(month, 'spring')

def apply_seasonal_drift(row: pd.Series, season: str, drift_strength: float) -> pd.Series:
    """
    Adjust biomass targets according to seasonal tendencies.
    - Applies multiplicative drift to selected components.
    - Recomputes `Dry_Total_g` as Clover + Dead + Green.
    - Scales `GDM_g` proportionally to total change when possible.
    """
    tendencies = SEASONAL_DRIFT.get(season, {})
    # Copy to avoid mutating original during pandas operations
    new_row = row.copy()

    # Components potentially present
    components = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
    for comp in components:
        if comp in new_row.index:
            base = float(new_row.get(comp, 0.0))
            t = float(tendencies.get(comp, 0.0))
            # Randomize magnitude slightly to avoid monotony
            mag = np.random.uniform(0.5, 1.0)
            factor = 1.0 + (t * drift_strength * mag)
            # Clamp factor to reasonable bounds
            factor = max(0.5, min(1.5, factor))
            new_row[comp] = max(0.0, base * factor)

    # Recompute total
    if all(c in new_row.index for c in components):
        total_old = float(row.get('Dry_Total_g', 0.0))
        total_new = float(new_row['Dry_Clover_g']) + float(new_row['Dry_Dead_g']) + float(new_row['Dry_Green_g'])
        new_row['Dry_Total_g'] = max(0.0, total_new)

        # Scale GDM proportionally if present
        if 'GDM_g' in new_row.index:
            if total_old > 0:
                scale = total_new / total_old
                new_row['GDM_g'] = max(0.0, float(new_row['GDM_g']) * scale)
            else:
                # fallback: track green dominance
                g = float(new_row['Dry_Green_g'])
                new_row['GDM_g'] = max(0.0, g)

    return new_row

def apply_smart_upsample_with_features(wide_df, logger):
    """
    Apply smart upsampling with enhanced feature augmentation.
    Uses new target order: [Green, Dead, Clover, GDM, Total]
    Includes proper biomass constraints: GDM = Clover + Green, Total = Clover + Dead + Green
    """
    if not UPSAMPLE_CONFIG['enabled']:
        logger.info("Upsampling disabled, skipping...")
        return wide_df
    
    target_min = UPSAMPLE_CONFIG['target_min_samples']
    noise_scale = UPSAMPLE_CONFIG['noise_scale']
    use_seasonal = UPSAMPLE_CONFIG.get('seasonal_drift', False)
    day_shift_prob = UPSAMPLE_CONFIG.get('day_shift_prob', 0.0)
    drift_strength = UPSAMPLE_CONFIG.get('drift_strength', 0.0)
    
    logger.info(f"Smart upsampling config: target_min={target_min}, noise_scale={noise_scale}, seasonal_drift={use_seasonal}")
    
    groups = []
    original_count = len(wide_df)
    groups_processed = 0
    total_synthetic_added = 0
    
    for key in wide_df['State_Species'].unique():
        key_df = wide_df[wide_df['State_Species'] == key].copy()
        n = len(key_df)
        groups_processed += 1
        
        if n >= target_min:
            # Already sufficient
            key_df['is_synthetic'] = False
            groups.append(key_df)
            logger.info(f"  [{groups_processed:2d}] {key}: {n} samples (sufficient, no upsampling)")
        else:
            # Upsample to target_min
            n_needed = target_min - n
            upsampled = key_df.sample(n=n_needed, replace=True, random_state=313).copy()
            total_synthetic_added += n_needed
            logger.info(f"  [{groups_processed:2d}] {key}: {n} → {target_min} samples (+{n_needed} synthetic)")
            
            # Apply date shifting and seasonal drift if enabled
            if day_shift_prob > 0:
                shift_mask = np.random.rand(len(upsampled)) < day_shift_prob
                if shift_mask.any():
                    offsets = np.random.choice([-1, 1], size=int(shift_mask.sum()))
                    shifted_dates = pd.to_datetime(upsampled.loc[shift_mask, 'Sampling_Date']) + pd.to_timedelta(offsets, unit='D')
                    upsampled.loc[shift_mask, 'Sampling_Date'] = shifted_dates

            if use_seasonal:
                seasons = upsampled['Sampling_Date'].apply(get_season)
                rand_mag = np.random.uniform(0.5, 1.0, size=len(upsampled))
                
                # Apply drift to component biomass (correct order: [Green, Dead, Clover])
                components = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
                for comp in components:
                    if comp in upsampled.columns:
                        t_series = seasons.map(lambda s: SEASONAL_DRIFT.get(s, {}).get(comp, 0.0)).astype(float)
                        factors = np.clip(1.0 + t_series.values * drift_strength * rand_mag, 0.5, 1.5)
                        base = upsampled[comp].astype(float).values
                        upsampled[comp] = np.maximum(0.0, base * factors)

                # Recompute derived targets: Total = Green + Dead + Clover, GDM = Green + Clover  
                if all(c in upsampled.columns for c in components):
                    green = upsampled['Dry_Green_g'].astype(float).values
                    dead = upsampled['Dry_Dead_g'].astype(float).values 
                    clover = upsampled['Dry_Clover_g'].astype(float).values
                    
                    new_total = green + dead + clover
                    new_gdm = green + clover
                    
                    if 'Dry_Total_g' in upsampled.columns:
                        upsampled['Dry_Total_g'] = np.maximum(0.0, new_total)
                    if 'GDM_g' in upsampled.columns:
                        upsampled['GDM_g'] = np.maximum(0.0, new_gdm)

            # Add noise to component biomass, then recompute derived targets
            comp_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
            
            for col in comp_cols:
                if col in upsampled.columns:
                    s = upsampled[col].std()
                    if np.isnan(s) or s == 0:
                        s = upsampled[col].mean()
                    noise = np.random.normal(0, s * noise_scale, size=len(upsampled))
                    upsampled[col] = np.maximum(0.0, upsampled[col] + noise)

            # Recompute derived targets with proper constraints
            if all(c in upsampled.columns for c in comp_cols):
                green = upsampled['Dry_Green_g'].astype(float).values
                dead = upsampled['Dry_Dead_g'].astype(float).values
                clover = upsampled['Dry_Clover_g'].astype(float).values
                
                new_total = green + dead + clover
                new_gdm = green + clover
                
                upsampled['Dry_Total_g'] = np.maximum(0.0, new_total)
                if 'GDM_g' in upsampled.columns:
                    upsampled['GDM_g'] = np.maximum(0.0, new_gdm)

            # Clamp to competition target limits 
            clamp_val = float(cfg.targets.biomass_clamp)
            biomass_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
            for col in biomass_cols:
                if col in upsampled.columns:
                    upsampled[col] = np.clip(upsampled[col].astype(float).values, 0.0, clamp_val)
            
            # Add noise to auxiliary features (NDVI and Height) 
            if 'Pre_GSHH_NDVI' in upsampled.columns:
                ndvi_noise_std = 0.015  # Fixed small noise for NDVI
                ndvi_noise = np.random.normal(0, ndvi_noise_std, size=len(upsampled))
                upsampled['Pre_GSHH_NDVI'] = np.clip(upsampled['Pre_GSHH_NDVI'] + ndvi_noise, 0.0, 1.0)
                
            if 'Height_Ave_cm' in upsampled.columns:
                height_std = upsampled['Height_Ave_cm'].std()
                if np.isnan(height_std) or height_std == 0: height_std = upsampled['Height_Ave_cm'].mean()
                height_noise_std = height_std * 0.03  # 3% relative noise for height
                height_noise = np.random.normal(0, height_noise_std, size=len(upsampled))
                upsampled['Height_Ave_cm'] = np.maximum(0.1, upsampled['Height_Ave_cm'] + height_noise)  # Minimum 0.1 cm
            
            # Mark samples and combine
            upsampled['is_synthetic'] = True
            key_df['is_synthetic'] = False
            
            groups.append(pd.concat([key_df, upsampled], ignore_index=True))
    
    result_df = pd.concat(groups, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=313).reset_index(drop=True)  # Shuffle
    result_df = result_df.sort_values(by=['Sampling_Date']).reset_index(drop=True)
    
    synthetic_count = len(result_df[result_df.get('is_synthetic', False)])
    total_count = len(result_df)
    upsampling_ratio = synthetic_count / total_count if total_count > 0 else 0
    
    logger.info(f"Upsampling summary:")
    logger.info(f"  Original: {original_count:,} samples")
    logger.info(f"  Final: {total_count:,} samples")
    logger.info(f"  Synthetic: {synthetic_count:,} samples ({upsampling_ratio:.1%})")
    logger.info(f"  Groups processed: {groups_processed}")
    logger.info(f"  Total synthetic added: {total_synthetic_added:,}")
    
    return result_df


def assign_functional_groups(df):
    """
    Classifies samples into biological categories for model learning.
    Note: This is NOT used for stratification anymore (we use State+Species).
    """
    col_legumes = [f'Species_{x}' for x in GROUP_DEFINITIONS['legume'] if f'Species_{x}' in df.columns]
    col_grasses = [f'Species_{x}' for x in GROUP_DEFINITIONS['grass']  if f'Species_{x}' in df.columns]
    col_weeds   = [f'Species_{x}' for x in GROUP_DEFINITIONS['weed']   if f'Species_{x}' in df.columns]

    s_legume = df[col_legumes].sum(axis=1)
    s_grass  = df[col_grasses].sum(axis=1)
    s_weed   = df[col_weeds].sum(axis=1)

    groups = []
    for l, g, w in zip(s_legume, s_grass, s_weed):
        # Hierarchy: Weed > Legume > Grass (Prioritize rare groups in ties/mixes)
        if w > 0 and w >= g and w >= l:
            groups.append('weed')
        elif l >= g: 
            groups.append('legume')
        else: 
            groups.append('grass')

    df['FunctionalGroup'] = groups
    return df


def calculate_global_weighted_r2(y_true, y_pred, weights):
    """
    Competition Metric: Global Weighted R2.
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    weights = np.array(weights, dtype=float)
    
    # Repeat weights for flattened arrays
    n_targets = len(weights)
    n_samples = len(y_true) // n_targets
    w_flat = np.tile(weights, n_samples)
    
    y_weighted_mean = np.sum(y_true * w_flat) / np.sum(w_flat)
    
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    ss_tot = np.sum(w_flat * (y_true - y_weighted_mean)**2)
    
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if logger: logger.info(f"Seed set to {seed}")

def save_batch_images(images, fold, batch_idx, session_dir, max_batches_to_save=5):
    """
    Save a batch of images as a grid to disk.
    """
    if batch_idx >= max_batches_to_save:
        return
    
    images_dir = os.path.join(session_dir, 'images', f'fold{fold}')
    os.makedirs(images_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    save_path = os.path.join(images_dir, f'batch_{batch_idx:03d}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)

def get_taxonomy_targets(species_vec):
    """
    Converts 14-dim species probability vector to 3-dim Taxonomy vector.
    Order: [Legume, Grass, Weed]
    
    Uses indices defined in configs.py to ensure consistency with CORE_SPECIES.
    """
    legume_prob = species_vec[:, TAXONOMY_IDXS['legume']].sum(dim=1, keepdim=True)
    grass_prob  = species_vec[:, TAXONOMY_IDXS['grass']].sum(dim=1, keepdim=True)
    weed_prob   = species_vec[:, TAXONOMY_IDXS['weed']].sum(dim=1, keepdim=True)
    
    return torch.cat([legume_prob, grass_prob, weed_prob], dim=1)

def save_tta_images(images, view_name, batch_idx, fold, epoch, session_dir):
    """Save TTA-augmented images for visualization."""
    if fold != 0 or epoch != 0 or batch_idx > 0:
        return
        
    save_dir = os.path.join(session_dir, 'tta_analysis', f'fold{fold+1}_ep{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    save_path = os.path.join(save_dir, f'batch{batch_idx}_{view_name}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)

def build_weighted_sampler_from_df(df, key='State_Species', cap_quantile=0.95):
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

