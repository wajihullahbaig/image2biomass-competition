import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import timm
import joblib
from tqdm import tqdm
import warnings
from sklearn.metrics import r2_score, accuracy_score, f1_score


def setup_logging(log_dir='logs'):
    """
    Set up logging to both console and file with timestamps.
    Creates a new log file for each run.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'stage1_training_{timestamp}.log')
    
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