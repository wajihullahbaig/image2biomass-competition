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

# Local Imports
from configs import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, 
    IMAGE_HEIGHT, IMAGE_WIDTH, CORE_SPECIES,GROUP_DEFINITIONS, N_FOLDS
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
        _, h, w = img.shape
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
        
        # 2. Rotate
        img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        
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

from PIL import ImageFilter
import random

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
# 4. STRATIFICATION LOGIC (Composite Key)
# -----------------------------------------------------------------------------

def create_stratify_key(df):
    """
    Creates a composite key to ensure every fold gets a fair distribution of
    Geographies (State) and Biology (FunctionalGroup).
    
    Crucial because State is temporally disjoint (NSW=Jan, WA=Sept).
    Random Stratified split is the ONLY way to ensure Fold 1 sees WA soil.
    """
    # 1. Ensure Functional Group exists
    if 'FunctionalGroup' not in df.columns:
        df = assign_functional_groups(df)
        
    # 2. Composite Key: State + Group
    # e.g., "NSW_Legume", "WA_Grass", "Vic_Weed"
    df['StratifyKey'] = df['State'].astype(str) + "_" + df['FunctionalGroup'].astype(str)
    
    # 3. Handle Rare Combinations
    # If a combo appears < N_FOLDS, StratifiedKFold will crash.
    # We map them to just 'State' or just 'Group' to allow splitting.
    counts = df['StratifyKey'].value_counts()
    rare_keys = counts[counts < N_FOLDS].index 
    
    # Fallback for rare items: Just use FunctionalGroup (Biology is more important than State for mass)
    df.loc[df['StratifyKey'].isin(rare_keys), 'StratifyKey'] = df['FunctionalGroup'].astype(str)
    
    return df


# -----------------------------------------------------------------------------
# 5. DATA LOADING & PREP (UPDATED)
# -----------------------------------------------------------------------------
def load_data(logger: logging.Logger) -> pd.DataFrame:
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
        if col not in targets.columns: targets[col] = 0.0
    targets[target_cols] = targets[target_cols].fillna(0.0)

    # Merge Metadata
    meta_cols = ['clean_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_meta_cols = [c for c in meta_cols if c in df.columns]
    
    meta = df[valid_meta_cols].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # Dates
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'], format='mixed', dayfirst=False)
    
    # Feature Engineering (Auxiliary Inputs)
    wide['Height_Ave_cm'] = pd.to_numeric(wide['Height_Ave_cm'], errors='coerce').fillna(0)
    wide['Pre_GSHH_NDVI'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce').fillna(0)
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'])
    wide['Interaction_Mul'] = wide['Pre_GSHH_NDVI'] * wide['Height_Ave_cm_log']
    
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    
    wide[target_cols] = wide[target_cols].astype(float)

    logger.info("Performing global species breakup...")

    # Initialize columns for each core species
    wide['Species'] = wide['Species'].str.lower()
    for sp in CORE_SPECIES:
        wide[f'Species_{sp}'] = 0.0

    # Parsing Logic
    def parse_species_string(s):
        vec = np.zeros(len(CORE_SPECIES))
        if not isinstance(s, str):
            return vec
        
        s_lower = s.lower().replace(' ', '')
        
        # Mapping specific cases
        # Note: 'clover' in string matches 'Clover', 'WhiteClover' etc. logic below handles exact/substring
        parts = s_lower.split('_')
        
        found_indices = set()
        
        for p in parts:
            # Check against core species
            for idx, core in enumerate(CORE_SPECIES):
                c_lower = core.lower()
                # Check for match. 
                # p="ryegrass" matches c="ryegrass"
                # p="whiteclover" matches c="whiteclover"
                # p="clover" matches c="clover"
                if c_lower == p or (p in c_lower and len(p) > 3) or (c_lower in p and len(c_lower) > 3):
                    found_indices.add(idx)
        
        if found_indices:
            # Distribute probability uniformly among found species
            prob = 1.0 / len(found_indices)
            for idx in found_indices:
                vec[idx] = prob
        else:
            # Fallback for "Mixed" or unknown -> Uniform across all
            vec[:] = 1.0 / len(CORE_SPECIES)
            
        return vec

    # Apply to dataframe
    # We iterate to assign to new columns
    species_vectors = wide['Species'].apply(parse_species_string)
    
    # Stack vectors into a matrix and assign to columns
    species_matrix = np.stack(species_vectors.values)
    for i, sp in enumerate(CORE_SPECIES):
        wide[f'Species_{sp}'] = species_matrix[:, i]

    wide = assign_functional_groups(wide)
    wide = create_stratify_key(wide)
        
    logger.info(f"Data Loaded and Parsed. Rows: {len(wide)}")
    wide.to_csv('wide.csv', index=False)
    return wide

# -----------------------------------------------------------------------------
# 6. UPSAMPLING LOGIC (Temporal Neighbor)
# -----------------------------------------------------------------------------
def upsample_minority_classes(df, target_col= None, date_col='Sampling_Date'):
    """
    Temporal Neighbor Upsampling.
    Tries to find samples from D-1 or D+1 to fill the class quota before
    resorting to exact duplication.
    """
    counts = df[target_col].value_counts()
    target = int(counts.max())
    dfs = [df]
    
    for cls, count in counts.items():
        if count < target:
            n_needed = target - count
            cls_mask = df[target_col] == cls
            cls_df = df[cls_mask].copy()
            existing_dates = set(cls_df[date_col].dt.date)
            candidates = []
            
            for _, row in cls_df.iterrows():
                # Check neighbors
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
    dfs =pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    # sort to keep temportal order
    dfs = dfs.sort_values(by=[date_col]).reset_index(drop=True)
    return 

# -----------------------------------------------------------------------------
# 7. METRICS & UTILS
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
# 8. SEED SETTING
# -----------------------------------------------------------------------------
def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if logger: logger.info(f"Seed set to {seed}")


# -----------------------------------------------------------------------------
# 9. FUNCTIONAL GROUP LOGIC 
# -----------------------------------------------------------------------------

def assign_functional_groups(df):
    """
    Classifies samples based on config definitions.
    """
    # Use definitions from configs.py
    col_legumes = [f'Species_{x}' for x in GROUP_DEFINITIONS['legume'] if f'Species_{x}' in df.columns]
    col_grasses = [f'Species_{x}' for x in GROUP_DEFINITIONS['grass']  if f'Species_{x}' in df.columns]
    col_weeds   = [f'Species_{x}' for x in GROUP_DEFINITIONS['weed']   if f'Species_{x}' in df.columns]

    s_legume = df[col_legumes].sum(axis=1)
    s_grass  = df[col_grasses].sum(axis=1)
    s_weed   = df[col_weeds].sum(axis=1)

    groups = []
    for l, g, w in zip(s_legume, s_grass, s_weed):
        if l >= g and l >= w: groups.append('legume')
        elif w > g: groups.append('weed')
        else: groups.append('grass')

    df['FunctionalGroup'] = groups
    return df

# -----------------------------------------------------------------------------
# 10. FUNCTIONAL GROUP UPSAMPLING LOGIC
# -----------------------------------------------------------------------------
def upsample_minority_classes(df, target_col='FunctionalGroup', date_col='Sampling_Date'):
    """
    Upsampling Strategy:
    1. Assigns Functional Groups (Grass/Legume/Weed).
    2. Upsamples based on these groups .
    3. Uses 'Temporal Neighbors' (D-1, D+1) to create variety instead of exact duplicates.
    """
    # 1. Assign Groups if not present (or if target_col is 'FunctionalGroup')
    if target_col == 'FunctionalGroup':
        df = assign_functional_groups(df)
        
    # 2. Calculate Targets
    counts = df[target_col].value_counts()
    target_count = int(counts.max())
    
    dfs = [df]
    
    # 3. Iterate Minority Classes
    for cls, count in counts.items():
        if count < target_count:
            n_needed = target_count - count
            
            # Get minority data
            cls_mask = df[target_col] == cls
            cls_df = df[cls_mask].copy()
            
            # --- Temporal Neighbor Search ---
            existing_dates = set(cls_df[date_col].dt.date)
            candidates = []
            
            for _, row in cls_df.iterrows():
                # Look for D-1 and D+1
                for offset in [-1, 1]:
                    d_new = row[date_col] + pd.Timedelta(days=offset)
                    # Only add if this specific date isn't already in the training set for this class
                    # (Prevents data leakage if we actually had data that day, though rare here)
                    if d_new.date() not in existing_dates:
                        new_row = row.copy()
                        new_row[date_col] = d_new
                        # We keep the same Image ID. 
                        # Rationale: "This image COULD have been taken yesterday."
                        candidates.append(new_row)
            
            # --- Selection Logic ---
            cand_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
            
            if len(cand_df) > 0:
                if len(cand_df) >= n_needed:
                    # We have enough temporal neighbors to fill the gap completely!
                    dfs.append(cand_df.sample(n=n_needed, replace=False, random_state=42))
                else:
                    # Use all temporal neighbors
                    dfs.append(cand_df)
                    # Fill remainder with standard duplicates
                    rem = n_needed - len(cand_df)
                    if rem > 0:
                        dfs.append(cls_df.sample(n=rem, replace=True, random_state=42))
            else:
                # No temporal neighbors found, fallback to standard duplication
                dfs.append(cls_df.sample(n=n_needed, replace=True, random_state=42))

    # 5 Shuffle and then sort to keep temporal order
    dfs = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    dfs = dfs.sort_values(by=[date_col]).reset_index(drop=True)
    return dfs


# -----------------------------------------------------------------------------
# 11. SAVE BATCH IMAGES
# -----------------------------------------------------------------------------
def save_batch_images(images, fold, batch_idx, session_dir, max_batches_to_save=5):
    """
    Save a batch of images as a grid to disk.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        fold: Current fold number
        batch_idx: Current batch index
        session_dir: Root session directory
        max_batches_to_save: Only save first N batches per epoch to avoid too many files
    """
    if batch_idx >= max_batches_to_save:
        return
    
    # Create directory structure: session_dir/images/fold<N>/
    images_dir = os.path.join(session_dir, 'images', f'fold{fold}')
    os.makedirs(images_dir, exist_ok=True)
    
    # Denormalize images if they were normalized
    # Assuming ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Save as grid
    save_path = os.path.join(images_dir, f'batch_{batch_idx:03d}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)
