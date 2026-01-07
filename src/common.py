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

# Local Imports
from configs import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, 
    IMAGE_HEIGHT, IMAGE_WIDTH, CORE_SPECIES, GROUP_DEFINITIONS, 
    N_FOLDS, TAXONOMY_IDXS, get_stratify_key, 
    UPSAMPLE_CONFIG, SPLIT_CONFIG, SEASON_MONTH_MAP, SEASONAL_DRIFT,
    USE_BIN_FEATURES, BIN_ENCODING, USE_SPECIES_COUNT_FEATURE
)

# -----------------------------------------------------------------------------
# 1. MATH & GEOMETRY HELPERS (The Core of the Strategy)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 2. CUSTOM TRAINING TRANSFORM (Manifold Alignment)
# -----------------------------------------------------------------------------

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
    
# -----------------------------------------------------------------------------
# 3. DATA AUGMENTATION PIPELINES
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 4. DATA LOADING & STRATIFICATION
# -----------------------------------------------------------------------------

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

    # Assign functional groups
    wide = assign_functional_groups(wide)
    
    # Region-Aware Stratification & Session ID ===
    wide['StratifyKey'] = wide.apply(get_stratify_key, axis=1)
    
    # Apply upsampling with enhanced NDVI and Height noise
    logger.info("Applying smart upsampling with NDVI and Height augmentation...")
    wide = apply_smart_upsample_with_features(wide, logger)
    
    # Feature Engineering (Auxiliary Inputs)
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'])
    wide['Interaction_Mul'] = wide['Pre_GSHH_NDVI'] * wide['Height_Ave_cm_log']
    wide['Interaction_Add'] = wide['Pre_GSHH_NDVI'] + wide['Height_Ave_cm_log']

    # Species richness (per-sample count)
    if USE_SPECIES_COUNT_FEATURE:
        # Exclude the generic 'clover' bucket
        richness_cols = [f'Species_{sp}' for sp in CORE_SPECIES if sp != 'clover' and f'Species_{sp}' in wide.columns]
        if len(richness_cols) > 0:
            wide['Species_Count'] = wide[richness_cols].sum(axis=1).astype(float)

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
    
    # Session ID
    wide['SessionID'] = wide.apply(lambda r: f"{r['State']}_{pd.to_datetime(r['Sampling_Date']).strftime('%Y%m%d')}", axis=1)
    
    # Log distribution
    logger.info("\n" + "="*70)
    logger.info("STRATIFICATION KEY DISTRIBUTION")
    logger.info("="*70)
    key_counts = wide['StratifyKey'].value_counts().sort_index()
    for key, count in key_counts.items():
        logger.info(f"  {key:<30}: {count:>3} samples")
    logger.info("="*70 + "\n")
        
    logger.info(f"Feature Engineering Complete. Rows: {len(wide)}")
    wide.to_csv('wide.csv', index=False)
    return wide


# -----------------------------------------------------------------------------
# 5. SPECIES PARSING (Existing Logic)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 6. IMPROVED TEMPORAL SPLIT
# -----------------------------------------------------------------------------

def smart_temporal_split(df, stratify_col='StratifyKey'):
    """
    Hybrid Temporal Split:
    1. STRICT: evaluation_dates > training_dates (Zero leakage across all groups).
    2. SMART: Adjusts the split cutoff to ensure every group is represented in training.
    """
    df = df.sort_values('Sampling_Date').reset_index(drop=True)
    n_total = len(df)
    
    # Configuration
    holdout_pct = SPLIT_CONFIG.get('holdout_pct', 0.15)
    target_split_idx = int(n_total * (1.0 - holdout_pct))
    
    # 1. Representation Guard: Every group must have at least one sample in dev
    # We find the MIN date for each group, and then the MAX of those.
    # This date is the absolute earliest we can split to include everyone.
    min_dates_per_group = df.groupby(stratify_col)['Sampling_Date'].min()
    safe_cutoff_date = min_dates_per_group.max()
    
    # 2. Target Cutoff: The date at our target split percentile
    target_cutoff_date = df.iloc[target_split_idx]['Sampling_Date']
    
    # Final Choice: Use the later of the two dates to satisfy both constraints
    final_cutoff = max(target_cutoff_date, safe_cutoff_date)
    
    # Session Guard: Ensure the final cutoff doesn't split a session
    # (Though typically sessions are all on the same date, this is safer)
    session_at_cutoff = df[df['Sampling_Date'] == final_cutoff]['SessionID'].unique()
    
    # Split
    dev_df = df[df['Sampling_Date'] <= final_cutoff].copy()
    holdout_df = df[df['Sampling_Date'] > final_cutoff].copy()
    
    # Verification: If the last session in dev_df is also in holdout_df, move it entirely to one side
    # But with strict temporal sorting, this shouldn't happen unless dates are identical.
    
    # Emergency fallback: If holdout is empty (rare), take the last 5% regardless
    if len(holdout_df) == 0:
        split_idx = int(len(df) * 0.95)
        dev_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()
        
    return dev_df, holdout_df

def triple_moving_time_series_split(df, n_splits=5, window_pct=0.25):
    """
    Implements a Triple Moving Window TimeSeriesSplit with MULTI-SESSION windows.
    Each fold:
    1. Val Set: Window of sessions (~window_pct of total).
    2. Holdout Set: Subsequent window of sessions (~window_pct of total).
    3. Train Set: All sessions before the Val window.
    """
    # Get unique sessions and their dates
    session_info = df.groupby('SessionID')['Sampling_Date'].min().sort_values()
    sessions = session_info.index.tolist()
    n_sessions = len(sessions)
    
    # Calculate window size (at least 1 session)
    window_size = max(1, int(n_sessions * window_pct))
    
    # We want the LAST fold to have Holdout as the very latest sessions
    # and Val as the sessions just before those.
    # Total eval sessions per fold = 2 * window_size
    # Total sessions required to 'move' across n_splits: 
    # Let's use a simpler proportional shift
    
    # Calculate indices for the windows
    # For fold i, the holdout ends at some index.
    # To ensure we utilize the whole dataset, let's fix the windows for the LATEST fold
    # and move them backwards for earlier folds.
    
    for i in range(n_splits):
        # Shift the eval windows backwards from the end
        # Fold (n_splits-1) should have holdout ending at n_sessions
        # Fold i should have holdout ending at (n_sessions - (n_splits - 1 - i))
        
        end_idx = n_sessions - (n_splits - 1 - i)
        hold_start = end_idx - window_size
        val_start = hold_start - window_size
        
        # Guard against index overflow/underflow
        if val_start < 1: 
             # Not enough data for this n_splits/window_size combo
             # Adjust val_start to at least 1 session for train
             val_start = max(1, i + 1) # Ensure train grows by at least 1 session per fold
             hold_start = val_start + window_size
             end_idx = hold_start + window_size
             
             if end_idx > n_sessions:
                 # Last resort fallback if total sessions < required
                 # Divide remaining sessions between Val and Hold
                 remaining = n_sessions - val_start
                 w = max(1, remaining // 2)
                 hold_start = val_start + w
                 end_idx = n_sessions

        train_sessions = sessions[:val_start]
        val_sessions = sessions[val_start:hold_start]
        hold_sessions = sessions[hold_start:end_idx]
        
        train_indices = df[df['SessionID'].isin(train_sessions)].index.tolist()
        val_indices = df[df['SessionID'].isin(val_sessions)].index.tolist()
        hold_indices = df[df['SessionID'].isin(hold_sessions)].index.tolist()
        
        yield train_indices, val_indices, hold_indices

# -----------------------------------------------------------------------------
# 7. SEASONAL HELPERS & SMART UPSAMPLING
# -----------------------------------------------------------------------------

def add_cv_group(df: pd.DataFrame, date_col: str = 'Sampling_Date') -> pd.DataFrame:
    """
    Add a group identifier combining State and Sampling_Date (YYYY-MM-DD) to avoid leakage
    across splits when the same state-date appears.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['cv_group'] = df['State'].astype(str) + "_" + df[date_col].dt.strftime('%Y-%m-%d')
    return df

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

def smart_upsample(train_df, stratify_col='StratifyKey',date_col='Sampling_Date'):
    """
    Upsample only sparse groups to minimum threshold.
    
    Improvements over blanket upsampling:
    1. Only upsample groups below target (preserves natural distribution)
    2. Add noise to biomass targets (prevents exact duplicates → overfitting)
    3. Mark synthetic samples for monitoring
    
    Why this matters:
    - Prevents overwhelming large groups (e.g., Ryegrass_Clover)
    - Reduces overfitting on repeated samples
    - Balances class distribution without destroying signal
    """
    if not UPSAMPLE_CONFIG['enabled']:
        return train_df
    
    target_min = UPSAMPLE_CONFIG['target_min_samples']
    noise_scale = UPSAMPLE_CONFIG['noise_scale']
    use_seasonal = UPSAMPLE_CONFIG.get('seasonal_drift', False)
    day_shift_prob = UPSAMPLE_CONFIG.get('day_shift_prob', 0.0)
    drift_strength = UPSAMPLE_CONFIG.get('drift_strength', 0.0)
    
    groups = []
    
    for key in train_df[stratify_col].unique():
        key_df = train_df[train_df[stratify_col] == key].copy()
        n = len(key_df)
        
        if n >= target_min:
            # Already sufficient
            key_df['is_synthetic'] = False
            groups.append(key_df)
        else:
            # Upsample to target_min
            n_needed = target_min - n
            upsampled = key_df.sample(n=n_needed, replace=True, random_state=42).copy()
            
            # Optionally shift dates by ±1 day (vectorized) and apply seasonal drift
            if day_shift_prob > 0:
                shift_mask = np.random.rand(len(upsampled)) < day_shift_prob
                if shift_mask.any():
                    offsets = np.random.choice([-1, 1], size=int(shift_mask.sum()))
                    shifted_dates = pd.to_datetime(upsampled.loc[shift_mask, date_col]) + pd.to_timedelta(offsets, unit='D')
                    upsampled.loc[shift_mask, date_col] = shifted_dates

            if use_seasonal:
                seasons = upsampled[date_col].apply(get_season)
                rand_mag = np.random.uniform(0.5, 1.0, size=len(upsampled))
                components = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
                for comp in components:
                    if comp in upsampled.columns:
                        t_series = seasons.map(lambda s: SEASONAL_DRIFT.get(s, {}).get(comp, 0.0)).astype(float)
                        factors = np.clip(1.0 + t_series.values * drift_strength * rand_mag, 0.5, 1.5)
                        base = upsampled[comp].astype(float).values
                        upsampled[comp] = np.maximum(0.0, base * factors)

                # Recompute totals and adjust GDM proportionally
                if all(c in upsampled.columns for c in components):
                    total_new = (
                        upsampled['Dry_Clover_g'].astype(float).values +
                        upsampled['Dry_Dead_g'].astype(float).values +
                        upsampled['Dry_Green_g'].astype(float).values
                    )
                    if 'Dry_Total_g' in upsampled.columns:
                        upsampled['Dry_Total_g'] = np.maximum(0.0, total_new)
                    if 'GDM_g' in upsampled.columns:
                        total_old = upsampled['Dry_Total_g'].astype(float).values if 'Dry_Total_g' in upsampled.columns else np.zeros_like(total_new)
                        scale = np.divide(total_new, total_old, out=np.ones_like(total_new), where=total_old > 0)
                        gdm = upsampled['GDM_g'].astype(float).values
                        gdm_scaled = np.maximum(0.0, gdm * scale)
                        # Fallback to green component where old total is zero
                        gdm_final = np.where(total_old > 0, gdm_scaled, np.maximum(0.0, upsampled['Dry_Green_g'].astype(float).values))
                        upsampled['GDM_g'] = gdm_final

            # Add noise to biomass targets (avoid exact duplicates)
            biomass_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            for col in biomass_cols:
                if col in upsampled.columns:
                    s = upsampled[col].std()
                    if np.isnan(s) or s == 0: s = upsampled[col].mean() # Fallback if std is undefined
                    noise = np.random.normal(0, s * noise_scale, size=len(upsampled))
                    upsampled[col] = np.maximum(0, upsampled[col] + noise)  # Ensure non-negative
            
            # Mark samples
            upsampled['is_synthetic'] = True
            key_df['is_synthetic'] = False
            
            groups.append(pd.concat([key_df, upsampled], ignore_index=True))
    
    result_df = pd.concat(groups, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    result_df = result_df.sort_values(by=[date_col]).reset_index(drop=True)
    return result_df

def apply_smart_upsample_with_features(wide_df, logger):
    """
    Apply smart upsampling with enhanced feature augmentation.
    Includes proper noise for NDVI (bounded 0-1) and Height (cm scale) in linear space.
    This should be called before any log transformations.
    """
    if not UPSAMPLE_CONFIG['enabled']:
        logger.info("Upsampling disabled, skipping...")
        return wide_df
    
    target_min = UPSAMPLE_CONFIG['target_min_samples']
    noise_scale = UPSAMPLE_CONFIG['noise_scale']
    use_seasonal = UPSAMPLE_CONFIG.get('seasonal_drift', False)
    day_shift_prob = UPSAMPLE_CONFIG.get('day_shift_prob', 0.0)
    drift_strength = UPSAMPLE_CONFIG.get('drift_strength', 0.0)
    
    groups = []
    original_count = len(wide_df)
    
    for key in wide_df['StratifyKey'].unique():
        key_df = wide_df[wide_df['StratifyKey'] == key].copy()
        n = len(key_df)
        
        if n >= target_min:
            # Already sufficient
            key_df['is_synthetic'] = False
            groups.append(key_df)
            logger.info(f"  {key}: {n} samples (sufficient, no upsampling)")
        else:
            # Upsample to target_min
            n_needed = target_min - n
            upsampled = key_df.sample(n=n_needed, replace=True, random_state=42).copy()
            logger.info(f"  {key}: {n} → {target_min} samples (added {n_needed})")
            
            # Optionally shift dates by ±1 day (vectorized) and apply seasonal drift
            if day_shift_prob > 0:
                shift_mask = np.random.rand(len(upsampled)) < day_shift_prob
                if shift_mask.any():
                    offsets = np.random.choice([-1, 1], size=int(shift_mask.sum()))
                    shifted_dates = pd.to_datetime(upsampled.loc[shift_mask, 'Sampling_Date']) + pd.to_timedelta(offsets, unit='D')
                    upsampled.loc[shift_mask, 'Sampling_Date'] = shifted_dates

            if use_seasonal:
                seasons = upsampled['Sampling_Date'].apply(get_season)
                rand_mag = np.random.uniform(0.5, 1.0, size=len(upsampled))
                components = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
                for comp in components:
                    if comp in upsampled.columns:
                        t_series = seasons.map(lambda s: SEASONAL_DRIFT.get(s, {}).get(comp, 0.0)).astype(float)
                        factors = np.clip(1.0 + t_series.values * drift_strength * rand_mag, 0.5, 1.5)
                        base = upsampled[comp].astype(float).values
                        upsampled[comp] = np.maximum(0.0, base * factors)

                # Recompute totals and adjust GDM proportionally
                if all(c in upsampled.columns for c in components):
                    total_new = (
                        upsampled['Dry_Clover_g'].astype(float).values +
                        upsampled['Dry_Dead_g'].astype(float).values +
                        upsampled['Dry_Green_g'].astype(float).values
                    )
                    if 'Dry_Total_g' in upsampled.columns:
                        upsampled['Dry_Total_g'] = np.maximum(0.0, total_new)
                    if 'GDM_g' in upsampled.columns:
                        total_old = upsampled['Dry_Total_g'].astype(float).values if 'Dry_Total_g' in upsampled.columns else np.zeros_like(total_new)
                        scale = np.divide(total_new, total_old, out=np.ones_like(total_new), where=total_old > 0)
                        gdm = upsampled['GDM_g'].astype(float).values
                        gdm_scaled = np.maximum(0.0, gdm * scale)
                        # Fallback to green component where old total is zero
                        gdm_final = np.where(total_old > 0, gdm_scaled, np.maximum(0.0, upsampled['Dry_Green_g'].astype(float).values))
                        upsampled['GDM_g'] = gdm_final

            # Add noise to biomass targets (existing logic)
            biomass_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
            for col in biomass_cols:
                if col in upsampled.columns:
                    s = upsampled[col].std()
                    if np.isnan(s) or s == 0: s = upsampled[col].mean() # Fallback if std is undefined
                    noise = np.random.normal(0, s * noise_scale, size=len(upsampled))
                    upsampled[col] = np.maximum(0, upsampled[col] + noise)  # Ensure non-negative
            
            # NEW: Add noise to NDVI (bounded 0-1) and Height (cm scale) in linear space
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
            
            # Mark samples
            upsampled['is_synthetic'] = True
            key_df['is_synthetic'] = False
            
            groups.append(pd.concat([key_df, upsampled], ignore_index=True))
    
    result_df = pd.concat(groups, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    result_df = result_df.sort_values(by=['Sampling_Date']).reset_index(drop=True)
    
    synthetic_count = len(result_df[result_df.get('is_synthetic', False)])
    total_count = len(result_df)
    logger.info(f"Upsampling complete: {original_count} → {total_count} samples ({synthetic_count} synthetic)")
    
    return result_df

# -----------------------------------------------------------------------------
# 8. FUNCTIONAL GROUP ASSIGNMENT (For Model Features)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 9. LEGACY UPSAMPLING (Keep for backward compatibility)
# -----------------------------------------------------------------------------

def upsample_minority_classes(df, target_col='FunctionalGroup', date_col='Sampling_Date'):
    """
    OLD upsampling strategy using temporal neighbors.
    Kept for backward compatibility but NOT recommended.
    Use smart_upsample() instead.
    """
    if target_col == 'FunctionalGroup':
        df = assign_functional_groups(df)
        
    counts = df[target_col].value_counts()
    target_count = int(counts.max())
    
    dfs = [df]
    
    for cls, count in counts.items():
        if count < target_count:
            n_needed = target_count - count
            
            cls_mask = df[target_col] == cls
            cls_df = df[cls_mask].copy()
            
            existing_dates = set(cls_df[date_col].dt.date)
            candidates = []
            
            for _, row in cls_df.iterrows():
                for offset in [-1, 1]:
                    d_new = row[date_col] + pd.Timedelta(days=offset)
                    if d_new.date() not in existing_dates:
                        new_row = row.copy()
                        new_row[date_col] = d_new
                        candidates.append(new_row)
            
            cand_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
            
            if len(cand_df) > 0:
                if len(cand_df) >= n_needed:
                    dfs.append(cand_df.sample(n=n_needed, replace=False, random_state=42))
                else:
                    dfs.append(cand_df)
                    rem = n_needed - len(cand_df)
                    if rem > 0:
                        dfs.append(cls_df.sample(n=rem, replace=True, random_state=42))
            else:
                dfs.append(cls_df.sample(n=n_needed, replace=True, random_state=42))

    dfs = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    dfs = dfs.sort_values(by=[date_col]).reset_index(drop=True)
    return dfs

# -----------------------------------------------------------------------------
# 10. METRICS & UTILS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 11. SEED SETTING
# -----------------------------------------------------------------------------
def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if logger: logger.info(f"Seed set to {seed}")

# -----------------------------------------------------------------------------
# 12. SAVE BATCH IMAGES
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 13. TAXONOMY TARGET GENERATION (For Model)
# -----------------------------------------------------------------------------
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
