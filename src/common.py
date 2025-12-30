# common.py
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import logging
from datetime import datetime
from typing import Optional
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from configs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, BIOMASS_FEAT_WEIGHT, IMAGE_HEIGHT, IMAGE_WIDTH

def load_data(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading and Pivoting Data...")
    if not os.path.exists('train.csv'):
        raise FileNotFoundError("train.csv not found in current directory")
        
    df = pd.read_csv('train.csv')
    df['clean_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
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

    meta_cols = ['clean_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_meta_cols = [c for c in meta_cols if c in df.columns]
    
    meta = df[valid_meta_cols].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'], format='mixed', dayfirst=False)
    
    wide['Height_Ave_cm'] = pd.to_numeric(wide['Height_Ave_cm'], errors='coerce')
    wide['Pre_GSHH_NDVI'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce')
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'].fillna(0))
    # Feature Engineering from Visual Analysis
    wide['Interaction_Mul'] = wide['Pre_GSHH_NDVI'].fillna(0) * wide['Height_Ave_cm_log']
    
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    
    # Scale Targets: Use Raw Grams for training (range ~0-250)
    # Log1p of ~60g is ~4.1, which is much better for gradients than Log1p(0.06)
    wide[target_cols] = wide[target_cols].astype(float)
    
    logger.info(f"Data Loaded. Rows: {len(wide)}")
    return wide
    
def get_image_data_transforms_v2():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            scale=(0.85, 1.0),
            ratio=(1.7, 2.3) # Matches 512/224 approx 2.28
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        # Rotation with limited black corners
        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0  # black corners allowed
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),

        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomGrayscale(p=0.2),

        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return train_transform, val_transform


def rotate_and_center_crop(x, angle):
    """
    x: (B, C, H, W)
    """
    B, C, H, W = x.shape

    # rotate
    x = TF.rotate(
        x,
        angle=angle,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0
    )

    # crop central region (avoid corners)
    crop_frac = 0.9
    ch, cw = int(H * crop_frac), int(W * crop_frac)

    top = (H - ch) // 2
    left = (W - cw) // 2

    x = x[:, :, top:top+ch, left:left+cw]

    # resize back
    x = torch.nn.functional.interpolate(
        x,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    return x

def apply_tta(model, image, device):
    model.eval()
    all_biomass, all_aux, all_species = [], [], []

    transforms_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: rotate_and_center_crop(x, 10),
        lambda x: rotate_and_center_crop(x, -10),
    ]

    for t in transforms_list:
        with torch.no_grad():
            img_aug = t(image)
            b, a, s = model(img_aug)
            all_biomass.append(b)
            all_aux.append(a)
            all_species.append(s)            

    return (
        torch.stack(all_biomass).mean(0),
        torch.stack(all_aux).mean(0),
        torch.stack(all_species).mean(0),
            
    )


def enforce_physical_constraints(predictions_real_scale):
    preds = np.maximum(predictions_real_scale, 0)
    clover, dead, green = preds[:, 0], preds[:, 1], preds[:, 2]
    preds[:, 3] = clover + dead + green  # Total
    preds[:, 4] = clover + green    # GDM
    return preds

def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if logger: logger.info(f"Seed set to {seed}")

def calculate_global_weighted_r2(y_true, y_pred, weights):
    """
    Calculates the Globally Weighted Coefficient of Determination (R2).
    Follows Competition Formula: R2 = 1 - (SS_res / SS_tot)
    where SS_tot is calculated using the globally weighted mean.
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    weights = np.array(weights, dtype=float)
    
    # Repeat weights for each sample [w1, w2, w3, w4, w5, w1, w2, ...]
    n_targets = len(weights)
    n_samples = len(y_true) // n_targets
    w_flat = np.tile(weights, n_samples)
    
    # Weighted Mean: sum(w * y) / sum(w)
    y_weighted_mean = np.sum(y_true * w_flat) / np.sum(w_flat)
    
    # Residual Sum of Squares
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    
    # Total Sum of Squares (Weighted variance from global mean)
    ss_tot = np.sum(w_flat * (y_true - y_weighted_mean)**2)
    
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

def check_group_leakage(train_df, holdout_df, group_col, logger):
    overlap = set(train_df[group_col]) & set(holdout_df[group_col])
    if overlap: logger.warning(f"Leakage detected: {overlap}")

def upsample_minority_classes(df, target_col, date_col='Sampling_Date'):
    """
    Upsamples minority classes to match the count of the majority class.
    Strategy: "Temporal Neighbor Upsampling"
    1. Identify 'gaps' or valid neighbors at D-1 and D+1 for existing samples.
    2. Prioritize filling the quota with these temporal clones (modifying date).
    3. If quota not met, fill remainder with standard random duplication.
    """
    counts = df[target_col].value_counts()
    target = int(counts.max())
    dfs = [df]
    
    for cls, count in counts.items():
        if count < target:
            n_needed = target - count
            cls_mask = df[target_col] == cls
            cls_df = df[cls_mask].copy()
            
            # Existing dates for this class (set for fast lookup)
            existing_dates = set(cls_df[date_col].dt.date)
            
            candidates = []
            
            # Identify valid temporal neighbors
            for _, row in cls_df.iterrows():
                # D-1 Candidate
                d_minus = row[date_col] - pd.Timedelta(days=1)
                if d_minus.date() not in existing_dates:
                    new_row = row.copy()
                    new_row[date_col] = d_minus
                    # We keep sample_id same, or could modify it. 
                    # Keeping it implies 'same image, slightly different time context'
                    candidates.append(new_row)
                    
                # D+1 Candidate
                d_plus = row[date_col] + pd.Timedelta(days=1)
                if d_plus.date() not in existing_dates:
                    new_row = row.copy()
                    new_row[date_col] = d_plus
                    candidates.append(new_row)
            
            # Convert candidates to DataFrame
            if candidates:
                cand_df = pd.DataFrame(candidates)
            else:
                cand_df = pd.DataFrame()
            
            # Fill Logic
            if len(cand_df) > 0:
                if len(cand_df) >= n_needed:
                    # Enough temporal neighbors to fill quota
                    sampled = cand_df.sample(n=n_needed, replace=False, random_state=42)
                    dfs.append(sampled)
                else:
                    # Take all temporal neighbors
                    dfs.append(cand_df)
                    remaining = n_needed - len(cand_df)
                    
                    # Fill remainder with standard duplication
                    if remaining > 0:
                        filled = cls_df.sample(n=remaining, replace=True, random_state=42)
                        dfs.append(filled)
            else:
                # No temporal candidates possible, fallback to full duplication
                filled = cls_df.sample(n=n_needed, replace=True, random_state=42)
                dfs.append(filled)

    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
