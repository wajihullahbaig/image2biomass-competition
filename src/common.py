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

# Local Imports
from configs import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, 
    IMAGE_HEIGHT, IMAGE_WIDTH
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
        
        # 2. Geometry (Manifold Alignment with TTA)
        RandomRotateCropResize(degrees=30),
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        # 3. Physics Simulation (Drone Altitude Noise)
        # Keep scale conservative (0.9-1.1) to preserve Mass-to-Pixel relationship.
        transforms.RandomAffine(
            degrees=0,              # Rotation handled above
            translate=(0.05, 0.05), # Slight shift
            scale=(0.9, 1.1),       # Conservative scaling
            shear=5
        ),

        # 4. Color Physics
        # Hue/Sat are sensitive for "Dead vs Green" classification.
        # Brightness/Contrast simulate time-of-day/clouds safely.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),

        # 5. Normalization
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
# 4. INFERENCE TTA ENGINE
# -----------------------------------------------------------------------------

def apply_tta(model, image, device):
    """
    Applies Test Time Augmentation.
    
    CRITICAL PHYSICS:
    We average in LINEAR SPACE (Grams), not Log Space.
    Averaging in Log space = Geometric Mean (underestimates biomass).
    Averaging in Linear space = Arithmetic Mean (maximizes R2).
    """
    model.eval()
    
    # Store predictions in LINEAR GRAMS
    all_biomass_linear = [] 
    all_aux = []
    all_species = []

    # TTA Policy: 7 Views
    # Includes standard flips and the "Zoom+Rotate" views
    tta_transforms = [
        lambda x: x,                           # 1. Identity
        lambda x: torch.flip(x, [3]),          # 2. H-Flip
        lambda x: torch.flip(x, [2]),          # 3. V-Flip
        lambda x: rotate_crop_resize(x, 15),   # 4. Rot +15 (Zoom ~1.2x)
        lambda x: rotate_crop_resize(x, -15),  # 5. Rot -15
        lambda x: rotate_crop_resize(x, 30),   # 6. Rot +30 (Zoom ~1.4x)
        lambda x: rotate_crop_resize(x, -30),  # 7. Rot -30
    ]

    for t in tta_transforms:
        with torch.no_grad():
            img_aug = t(image)
            
            # Forward Pass (Outputs are Log1p)
            log_bio, aux, sp = model(img_aug) 
            
            # Convert to Linear Grams IMMEDIATELY
            lin_bio = torch.expm1(log_bio)
            
            all_biomass_linear.append(lin_bio)
            all_aux.append(aux)
            all_species.append(sp)            

    # --- AGGREGATION ---
    
    # 1. Biomass: Arithmetic Mean in Linear Space
    avg_bio_linear = torch.stack(all_biomass_linear).mean(0)
    # Convert back to Log Space for consistency with training loop/loss wrappers
    avg_bio_log = torch.log1p(avg_bio_linear)

    # 2. Aux & Species: Mean in Logit/Raw space is fine
    avg_aux = torch.stack(all_aux).mean(0)
    avg_species = torch.stack(all_species).mean(0)
            
    return avg_bio_log, avg_aux, avg_species

# -----------------------------------------------------------------------------
# 5. DATA LOADING & PREP
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
    
    # Log Height often correlates better with Log Biomass
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'])
    
    # Interaction Term (Volume Proxy)
    wide['Interaction_Mul'] = wide['Pre_GSHH_NDVI'] * wide['Height_Ave_cm_log']
    
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    
    # Targets stay in Raw Grams here.
    # Conversion to Log1p happens inside the Dataset/Training loop.
    wide[target_cols] = wide[target_cols].astype(float)
    
    logger.info(f"Data Loaded. Rows: {len(wide)}")
    return wide

def upsample_minority_classes(df, target_col, date_col='Sampling_Date'):
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

    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 6. METRICS & UTILS
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

def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if logger: logger.info(f"Seed set to {seed}")