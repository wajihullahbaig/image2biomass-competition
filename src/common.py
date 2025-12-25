# common.py
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import logging
from datetime import datetime
from typing import Optional
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from configs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGE_SIZE

# ====================== DATA PREP ======================
def load_data(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading and Pivoting Data...")
    df = pd.read_csv('train.csv')
    
    # 1. Clean sample_id (Remove __target_name suffix if it exists)
    # This converts 'ID123__Dry_Clover_g' -> 'ID123'
    df['clean_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
    # 2. HARD PIVOT: Ensure exactly one row per clean_id
    # We use 'max' to aggregate because the other rows have 0 or NaN for that target
    targets = df.pivot_table(
        index='clean_id', 
        columns='target_name', 
        values='target',
        aggfunc='max' 
    ).reset_index()
    
    # Fill missing targets with 0.0 
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for col in target_cols:
        if col not in targets.columns: targets[col] = 0.0
    targets[target_cols] = targets[target_cols].fillna(0.0)

    # 3. Extract Metadata (Take the first entry for each clean_id)
    # We drop 'target_name' and 'target' and 'sample_id' from meta to avoid dupes
    meta_cols = ['clean_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    # Ensure we only check columns that actually exist in the csv
    valid_meta_cols = [c for c in meta_cols if c in df.columns]
    
    meta = df[valid_meta_cols].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
    
    # 4. Merge
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # 5. Feature Engineering
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['month'] = wide['Sampling_Date'].dt.month
    wide['season'] = wide['month'].apply(get_season)
    
    wide['Height_Ave_cm'] = pd.to_numeric(wide['Height_Ave_cm'], errors='coerce')
    wide['Pre_GSHH_NDVI'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce')
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'].fillna(0))
    
    # Rename clean_id back to sample_id for consistency
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    
    # 6. SCALE TARGETS: Log-Space Scaling (log1p)
    # This compresses the range [0, 250] grams into [0, 5.5] log units.
    # It addresses the skewness and prevents magnitude bias.
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    wide[target_cols] = np.log1p(wide[target_cols].astype(float))
    
    logger.info(f"Data Loaded Successfully. Rows: {len(wide)} (Targets Log-Scaled)")
    
    # SANITY CHECK
    # Dry_Total should roughly equal components. 
    # If there's a massive mismatch, print warning.
    calc_total = wide['Dry_Clover_g'] + wide['Dry_Dead_g'] + wide['Dry_Green_g']
    diff = (wide['Dry_Total_g'] - calc_total).abs().mean()
    logger.info(f"Average Physics Consistency Error (Total vs Sum): {diff:.4f}g")
    wide.to_csv('wide.csv', index=False)
    return wide
    
def get_image_data_transforms()->tuple:
    """
    Returns the training and validation data augmentation transforms.
    Focus on geometric invariance while preserving photometric signal (greenness).
    """
    # Data Augmentation Transforms for small dataset (357 samples)
    train_transform = transforms.Compose([
        # 1. Structural/Scale (Resizing happens here)
        # Using scale >= 0.7 to avoid losing the plot context
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        
        # 2. Geometric (Full Invariance)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # 90-degree rotations are often cleaner for plant layouts than arbitrary degrees
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        
        # 3. Photometric (Conservative)
        # CRITICAL: Keep hue jitter very low (<= 0.02) to maintain biomass-greenness relationship
        transforms.ColorJitter(
            brightness=0.15, 
            contrast=0.15, 
            saturation=0.1, 
            hue=0.01 
        ),
        
        # 4. Noise/Blur
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        
        # 5. For texture
        transforms.RandomGrayscale(p=0.25),

        # 6. Conversion
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        
        # 7. Occlusion (Post-Tensor)
        # RandomErasing / Cutout forces model to learn global features
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))

        
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return train_transform, val_transform       

def get_image_data_transforms_v1()->tuple:
    """
    Returns the training and validation data augmentation transforms.
    """
    # Data Augmentation Transforms
    train_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAutocontrast(),
                transforms.RandomEqualize(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomGrayscale(p=0.25),
                transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])

    val_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
    return train_transform, val_transform  

def apply_tta(model, image, device, n_passes=1):
    """
    Performs Test-Time Augmentation (TTA) using 5-Crop + Flips.
    """
    model.eval()
    all_biomass = []
    all_aux = []
    all_species = []
    all_month = []
    
    # 1. Standard Views (Original, Flips, Rotations)
    # We create a list of augmentation functions
    aug_fns = [
        lambda x: x,                        # Original
        lambda x: torch.flip(x, [3]),       # H-Flip
        lambda x: torch.flip(x, [2]),       # V-Flip
        lambda x: torch.rot90(x, 1, [2, 3]),# Rot90
        lambda x: torch.rot90(x, 3, [2, 3]) # Rot270
    ]
    
    for i, aug_fn in enumerate(aug_fns):
        if i >= n_passes: break
        
        with torch.no_grad():
            img_aug = aug_fn(image)
            b, a, s, m = model(img_aug)
            all_biomass.append(b)
            all_aux.append(a)
            all_species.append(s)
            all_month.append(m)
            
    # Average predictions
    avg_biomass = torch.stack(all_biomass).mean(0)
    avg_aux = torch.stack(all_aux).mean(0)
    avg_species = torch.stack(all_species).mean(0)
    avg_month = torch.stack(all_month).mean(0)
    
    return avg_biomass, avg_aux, avg_species, avg_month

def setup_logging(logger_name="System Logger", log_dir='logs', file_name_part=None) -> str:
    """
    Set up logging to both console and file.
    Creates a new session directory 'logs/{timestamp}' and saves 'session.log' there.
    Returns the path to the session directory.
    """
    # Create timestamped session directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if file_name_part:
        # e.g. logs/Unified_Shared_20251217_230000
        session_dir_name = f"{file_name_part}_{timestamp}"
    else:
        session_dir_name = timestamp
        
    session_dir = os.path.join(log_dir, session_dir_name)
    os.makedirs(session_dir, exist_ok=True)
    
    # Create plots directory inside session directory
    os.makedirs(os.path.join(session_dir, 'plots'), exist_ok=True)
    
    # Log file path
    log_file = os.path.join(session_dir, 'session.log')
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simpler logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Session Directory: {session_dir}")
    return session_dir


def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

def enforce_physical_constraints(predictions_real_scale):
    """
    Enforces strict mass balance post-processing.
    Input: Numpy array of predictions in REAL GRAMS (not log).
    Order: [Clover, Dead, Green, Total, GDM]
    """
    # 1. Enforce Non-Negativity
    preds = np.maximum(predictions_real_scale, 0)
    
    # 2. Extract components
    clover = preds[:, 0]
    dead = preds[:, 1]
    green = preds[:, 2]
    
    # 3. Recalculate aggregates based on components
    new_gdm = clover + green
    new_total = clover + dead + green
    
    # 4. Update the prediction array
    preds[:, 3] = new_total  # Total
    preds[:, 4] = new_gdm    # GDM
    
    return preds


def set_seed(seed: Optional[int] = 42, logger=None) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        if logger:
            logger.info(f"Random seed set to {seed} for reproducibility")

def plot_training_history(history, fold, session_dir):
    """
    Plots training and validation metrics for the unified model.
    history: dict with keys 'train_loss', 'val_loss', 'val_r2', 'loss_biomass', 'loss_aux'.
    """
    save_dir = os.path.join(session_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    if 'train_loss scaled - biomass loss x(1/100)' in history:
        plt.plot(np.array(history['train_loss']), label='Total Train Loss', linewidth=2, color='tab:blue')
    if 'loss_biomass' in history:
        plt.plot(np.array(history['loss_biomass']) * 100.0, label='Biomass Train Loss (x100)', linestyle='--', alpha=0.7)
    if 'loss_aux' in history:
        plt.plot(history['loss_aux'], label='Aux Train Loss', linestyle=':', alpha=0.7)
    if 'loss_species' in history:
        plt.plot(history['loss_species'], label='Species Train Loss', linestyle='-.', alpha=0.7)
    if 'loss_month' in history:
        plt.plot(history['loss_month'], label='Month Train Loss', linestyle='-', alpha=0.4, color='gray')
        
    if 'val_loss' in history:
        plt.plot(np.array(history['val_loss']), label='Total Val Loss', linewidth=2, color='tab:red')
    if 'val_loss_biomass' in history:
        plt.plot(np.array(history['val_loss_biomass']) * 100.0, label='Biomass Val Loss (x100)', linestyle='--', color='tab:orange', alpha=0.7)
    if 'val_loss_aux' in history:
        plt.plot(history['val_loss_aux'], label='Aux Val Loss', linestyle=':', color='magenta', alpha=0.7)
    if 'val_loss_species' in history:
        plt.plot(history['val_loss_species'], label='Species Val Loss', linestyle='-.', color='tab:brown', alpha=0.7)
    if 'val_loss_month' in history:
        plt.plot(history['val_loss_month'], label='Month Val Loss', linestyle='-', alpha=0.4, color='purple')
        
    plt.title(f'Fold {fold} - Training Progress (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- R2 Plot ---
    plt.subplot(1, 2, 2)
    if 'val_r2' in history:
        plt.plot(history['val_r2'], label='Val R2 (Special)', color='green', linewidth=2)
    
    plt.title(f'Fold {fold} - Validation Metric (R2)')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    if 'holdout_r2' in history:
        plt.plot(history['holdout_r2'], label='Holdout R2 (Strict)', color='red', linewidth=2, linestyle=':')
    
    plt.title(f'Fold {fold} - Validation Metric (R2)')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.ylim(-2.0, 2.0)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_metrics.png"))
    plt.close()

def calculate_global_weighted_r2(y_true, y_pred, weights):
    """
    Official Metric Implementation.
    y_true, y_pred: (N, 5) arrays.
    weights: Pattern [0.1, 0.1, 0.1, 0.5, 0.2] for [Clover, Dead, Green, Total, GDM]
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    n_samples = len(y_true) // 5
    w_flat = np.tile(weights, n_samples)
    
    # Avoid division by zero in average if weights sum to 0
    if np.sum(w_flat) == 0:
        return 0.0
        
    global_mean = np.average(y_true, weights=w_flat)
    
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    ss_tot = np.sum(w_flat * (y_true - global_mean)**2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def upsample_minority_classes(df, target_col, logger):
    """
    Upsamples under-represented classes in target_col to the MEDIAN count.
    This creates a more balanced dataset without over-representing rare classes.
    Only upsamples classes that exist in the current fold.
    """
    counts = df[target_col].value_counts()
    
    if len(counts) == 0:
        logger.info(f"No classes found in '{target_col}'. Skipping upsampling.")
        return df
    
    # Calculate target threshold: boost EVERYTHING to the MAX count for perfect fairness
    target_threshold = int(counts.max())
    
    # Only upsample classes below the maximum
    minority_classes = counts[counts < target_threshold].index
    
    if len(minority_classes) == 0:
        logger.info(f"All '{target_col}' classes are already perfectly balanced at {target_threshold} samples.")
        return df
    
    # Log before upsampling
    logger.info(f"\n--- Balancing to Max Count ---")
    logger.info(f"Total Samples (Initial): {len(df)}")
    logger.info(f"Target count (Max): {target_threshold}")
    logger.info(f"Upsampling Regime: Boosting {len(minority_classes)} '{target_col}' classes to match the max ({target_threshold} samples).")
    
    upsampled_dfs = [df]
    total_added = 0
    
    for cls in minority_classes:
        cls_df = df[df[target_col] == cls]
        current_count = len(cls_df)
        num_to_add = target_threshold - current_count
        
        if num_to_add > 0:
            # Sample with replacement to reach threshold
            added_df = cls_df.sample(n=num_to_add, replace=True, random_state=42)
            upsampled_dfs.append(added_df)
            total_added += num_to_add
            logger.info(f"  + Added {num_to_add} samples for '{cls}'")
    
    new_df = pd.concat(upsampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Log after upsampling
    logger.info(f"\n--- After Upsampling ---")
    logger.info(f"Total Samples: {len(new_df)} (added {total_added})")
    new_counts = new_df[target_col].value_counts()
    logger.info(f"Updated class distribution:")
    for cls in sorted(new_counts.index):
        logger.info(f"  - {cls}: {new_counts[cls]} samples")
    
    return new_df


def check_group_leakage(train_df, holdout_df, group_col, logger):
    train_groups = set(train_df[group_col])
    holdout_groups = set(holdout_df[group_col])
    overlap = train_groups & holdout_groups

    logger.info(f"Group overlap count: {len(overlap)}")
    if len(overlap) > 0:
        logger.info("❌ LEAKAGE DETECTED")
        logger.info(f"Overlapping groups: {overlap}")
    else:
        logger.info("✅ No group leakage")