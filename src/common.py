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


# CONFIGURATIONS
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])

    val_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
    return train_transform, val_transform       

def setup_logging(log_dir='logs',file_name_part =None) -> logging.Logger:
    """
    Set up logging to both console and file with timestamps.
    Creates a new log file for each run.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{file_name_part}_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('Stage1Logger')
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

def calculate_sample_weights(df, proportions=None, prop_col=None, weight_col='sample_weight', logger=None):
    """
    Calculates inverse-frequency sample weights based on prop_col distribution.
    The weights are normalized so the mean weight is 1.0.
    """
    if proportions is None:
        proportions = df[prop_col].value_counts(normalize=True)

    # Calculate inverse proportions and normalize
    inverse_proportions = 1 / proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse

    # Create the weight map
    weight_map = normalized_weights.to_dict()

    # Apply the weight to the DataFrame
    df[weight_col] = df[prop_col].map(weight_map)
    
    if logger:
        logger.info(f"Sample weights calculated based on '{prop_col}'")
        logger.debug(f"Weight distribution: {weight_map}")

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
    def __init__(self, data_df, shuffle_within_season=True, seed=42):
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