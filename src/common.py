# common.py
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional
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