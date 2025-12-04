# common.py
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, List, Tuple
import torch
from torchvision import transforms
from torch.utils.data import Sampler
import random

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
    
    logger.info(f"Data Loaded Successfully. Rows: {len(wide)}")
    
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

def setup_logging(logger_name = "System Logger",log_dir='logs',file_name_part =None) -> logging.Logger:
    """
    Set up logging to both console and file with timestamps.
    Creates a new log file for each run.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{file_name_part}_{timestamp}.log')
    
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
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


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

def calculate_sample_weights_mean(df, group_col, weight_col='sample_weight', logger=None):
    proportions = df[group_col].value_counts(normalize=True)    
    inverse_proportions = 1 / proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse
    weight_map = normalized_weights.to_dict()
    df[weight_col] = df[group_col].map(weight_map)
    
    if logger:
        # Calculate average weight per group for logging verification
        avg_weights = df.groupby(group_col)[weight_col].mean().to_dict()
        logger.info(f"Sample weights calculated based on '{group_col}'")
        for group, avg_weight in avg_weights.items():
            logger.info(f"  {group}: {avg_weight:.4f}")
        
        # Global normalization stats
        logger.info(
            f"Normalized Weights — min: {df[weight_col].min():.4f}, "
            f"max: {df[weight_col].max():.4f}, "
            f"mean: {df[weight_col].mean():.4f}, "
            f"sum: {df[weight_col].sum():.4f}"
        )
    
    return df, weight_col


def calculate_sample_weights(df, group_col, weight_col='sample_weight', smooth=10.0, logger=None):
    """
    Calculates sample weights using smoothed inverse frequency.
    Normalizes by median so the majority class has weight ~1.0.
    """
    counts = df[group_col].value_counts()    
    weights = 1.0 / (df[group_col].map(counts) + smooth)    
    weights = weights / weights.median()
    df[weight_col] = weights.astype('float32')
    
    if logger:
        # Calculate average weight per group for logging verification
        avg_weights = df.groupby(group_col)[weight_col].mean().to_dict()
        logger.info(f"Sample weights calculated based on '{group_col}' with smoothing={smooth}")
        for group, avg_weight in avg_weights.items():
            logger.info(f"  {group}: {avg_weight:.4f}")
        
    return df, weight_col

def calculate_count_frequency_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    group_col: str, 
    local_group_col: Optional[str] = None, 
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calculates and adds global and optional local count/frequency features.
    
    - Statistics are calculated ONLY from the training set to prevent data leakage.
    - The validation set is imputed using the training set statistics.
    - Unseen categories in validation are handled by filling with 0 for log-counts
      and the mean of training frequencies for frequencies.
    
    Returns:
        A tuple containing (processed_train_df, processed_val_df, list_of_new_features).
    """
    if logger:
        logger.info("-" * 50)
        logger.info("Calculating count and frequency features...")
        
    # --- Create copies to avoid SettingWithCopyWarning ---
    train_df = train_df.copy()
    val_df = val_df.copy()

    # --- Feature Names ---
    global_count_feat = f'{group_col.lower()}_count_global'
    global_freq_feat = f'{group_col.lower()}_freq_global'
    
    # --- 1. GLOBAL FEATURES (calculated from training set) ---
    if logger: logger.info(f"Calculating global features for '{group_col}'...")
    
    global_counts = train_df[group_col].value_counts()
    global_freq = global_counts / len(train_df)
    global_counts_log = np.log1p(global_counts)
    
    # Map to train_df
    train_df[global_count_feat] = train_df[group_col].map(global_counts_log)
    train_df[global_freq_feat] = train_df[group_col].map(global_freq)
    
    # Map to val_df and impute unseen values
    val_df[global_count_feat] = val_df[group_col].map(global_counts_log).fillna(0)
    val_df[global_freq_feat] = val_df[group_col].map(global_freq).fillna(global_freq.mean())

    new_features = [global_count_feat, global_freq_feat]

    # --- 2. LOCAL FEATURES (optional, calculated from training set) ---
    if local_group_col:
        local_count_feat = f'{group_col.lower()}_count_{local_group_col.lower()}'
        local_freq_feat = f'{group_col.lower()}_freq_{local_group_col.lower()}'
        new_features.extend([local_count_feat, local_freq_feat])
        
        if logger: logger.info(f"Calculating local features for '{group_col}' grouped by '{local_group_col}'...")

        # Calculate stats from training set
        local_counts = train_df.groupby([local_group_col, group_col]).size()
        local_counts_log = np.log1p(local_counts)
        
        # Calculate frequency within each local group
        local_group_totals = train_df.groupby(local_group_col).size()
        local_freq = local_counts / local_counts.index.map(lambda x: local_group_totals[x[0]])

        # Map to train_df
        train_df[local_count_feat] = train_df.apply(
            lambda row: local_counts_log.get((row[local_group_col], row[group_col]), 0),
            axis=1
        )
        train_df[local_freq_feat] = train_df.apply(
            lambda row: local_freq.get((row[local_group_col], row[group_col]), 0),
            axis=1
        )
        
        # Map to val_df and impute unseen values
        val_df[local_count_feat] = val_df.apply(
            lambda row: local_counts_log.get((row[local_group_col], row[group_col]), 0),
            axis=1
        )
        # Use the mean of all local frequencies as a robust fallback for unseen combinations
        val_df[local_freq_feat] = val_df.apply(
            lambda row: local_freq.get((row[local_group_col], row[group_col]), local_freq.mean()),
            axis=1
        )

    if logger:
        logger.info(f"Successfully added features: {new_features}")
        logger.info("-" * 50)
        
    return train_df, val_df, new_features


def print_stratification_stats(df, train_df, val_df,start_col=None, logger=None):
    """Prints stratification statistics for the start_col column."""
    if logger:
        logger.info(f"\n--- {start_col} Distribution Verification ---")
    
    # Original Dataset Counts
    original_counts = df[start_col].value_counts()
    original_proportions = df[start_col].value_counts(normalize=True).mul(100).round(2)
    original_stats = pd.DataFrame({'Count': original_counts, 'Proportion (%)': original_proportions})
    
    if logger:
        logger.info(f"Original Dataset:\n{original_stats}")
    
    # Training Split Counts
    train_counts = train_df[start_col].value_counts()
    train_proportions = train_df[start_col].value_counts(normalize=True).mul(100).round(2)
    train_stats = pd.DataFrame({'Count': train_counts, 'Proportion (%)': train_proportions})
    
    if logger:
        logger.info(f"\nTraining Split (80%):\n{train_stats}")
    
    # Validation Split Counts
    val_counts = val_df[start_col].value_counts()
    val_proportions = val_df[start_col].value_counts(normalize=True).mul(100).round(2)
    val_stats = pd.DataFrame({'Count': val_counts, 'Proportion (%)': val_proportions})
    
    if logger:
        logger.info(f"\nValidation Split (20%):\n{val_stats}")

# Simple Australian Seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'        
    
class SeasonalCurriculumSampler(Sampler):
    """
    Samples indices in seasonal order: Summer → Autumn → Winter → Spring.
    Yields individual indices. The DataLoader handles the batching.
    """
    def __init__(self, data_df, shuffle_within_season=False, seed=42):
        self.data_df = data_df
        self.shuffle_within_season = shuffle_within_season
        self.seed = seed
        self.season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
        self._generate_indices()

    def _generate_indices(self):
        # Reset seeds so order is deterministic per epoch if needed
        # (Move this to __iter__ if you want different shuffles every epoch)
        random.seed(self.seed) 
        
        # 1. Group indices by season
        indices_by_season = {s: [] for s in self.season_order}
        
        # Iterate efficiently
        for idx in range(len(self.data_df)):
            # Ensure we access the 'season' column safely
            # We use iloc to get the row by integer position, regardless of DataFrame index
            season = self.data_df.iloc[idx]['season']
            if season in indices_by_season:
                indices_by_season[season].append(idx)

        # 2. Shuffle within seasons and flatten list
        self.ordered_indices = []
        for season in self.season_order:
            season_ind = indices_by_season[season]
            if self.shuffle_within_season:
                random.shuffle(season_ind)
            self.ordered_indices.extend(season_ind)

    def __iter__(self):
        return iter(self.ordered_indices)

    def __len__(self):
        return len(self.data_df)
    

def enforce_physical_constraints(predictions_real_scale):
    """
    OPTION B: Post-processing to enforce strict mass balance.
    Input: Numpy array of predictions in REAL GRAMS (not log).
    Order: [Clover, Dead, Green, Total, GDM]
    """
    # 1. Enforce Non-Negativity (Safety net)
    preds = np.maximum(predictions_real_scale, 0)
    
    # 2. Extract components
    clover = preds[:, 0]
    dead = preds[:, 1]
    green = preds[:, 2]
    
    # 3. Recalculate Aggregates based on components
    new_gdm = clover + green
    new_total = clover + dead + green
    
    # 4. Update the prediction array
    preds[:, 3] = new_total  # Total
    preds[:, 4] = new_gdm    # GDM
    
    return preds    



def calculate_global_weighted_r2(y_true, y_pred, weights):
    """
    Official Metric Implementation.
    y_true, y_pred: Flattened or (N, 5) arrays.
    weights: List of 5 weights [0.1, 0.1, 0.1, 0.5, 0.2]
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Validation
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) % 5 != 0:
        raise ValueError(f"Input length {len(y_true)} must be divisible by 5")
    
    if len(weights) != 5:
        raise ValueError("weights must have exactly 5 elements")
    
    # Repeat weights pattern for every sample
    n_samples = len(y_true) // 5
    w_flat = np.tile(weights, n_samples)
    
    # Global Weighted Mean
    global_mean = np.average(y_true, weights=w_flat)
    
    # Weighted SS_res and SS_tot
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    ss_tot = np.sum(w_flat * (y_true - global_mean)**2)
    
    if ss_tot == 0:
        return np.nan  # or 1.0, depending on interpretation
    
    return 1 - (ss_res / ss_tot)