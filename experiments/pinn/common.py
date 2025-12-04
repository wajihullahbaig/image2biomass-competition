import os
import pandas as pd
import numpy as np
import logging
import torch
import random
from torchvision import transforms
from configs import IMAGE_SIZE, SEED

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def load_data(logger=None):
    if logger: logger.info("Loading Data...")
    df = pd.read_csv('train.csv')
    
    # Clean ID
    df['clean_id'] = df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
    
    # Pivot Targets
    targets = df.pivot_table(
        index='clean_id', 
        columns='target_name', 
        values='target',
        aggfunc='max'
    ).reset_index()
    
    # Fill missing targets with 0.0
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    for c in target_cols:
        if c not in targets.columns: targets[c] = 0.0
    targets = targets.fillna(0.0)

    # Metadata
    meta_cols = ['clean_id', 'image_path', 'Sampling_Date', 'State', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    valid_cols = [c for c in meta_cols if c in df.columns]
    meta = df[valid_cols].drop_duplicates(subset=['clean_id'])
    
    # Merge
    wide = pd.merge(meta, targets, on='clean_id', how='left')
    
    # Season/Date Engineering
    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['month'] = wide['Sampling_Date'].dt.month
    wide['season'] = wide['month'].apply(get_season)
    
    # Auxiliary features for Multi-task learning (optional, helps backbone)
    wide['aux_ndvi'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce').fillna(0)
    wide['aux_height'] = np.log1p(pd.to_numeric(wide['Height_Ave_cm'], errors='coerce').fillna(0))
    
    if logger: logger.info(f"Loaded {len(wide)} unique samples.")
    return wide

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/pinn_train.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def calculate_weighted_average_r2(y_true, y_pred, weights):
    """
    Calculates R2 for each column, then returns the weighted average.
    y_true, y_pred: numpy arrays of shape (N, 5)
    weights: list or array of shape (5,)
    """
    r2_scores = []
    
    # Loop through each of the 5 targets
    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        
        # Handle constant target case (avoid divide by zero)
        if ss_tot < 1e-6:
            r2 = 0.0 # Or 1.0 if ss_res is also small, but usually 0 for safety
        else:
            r2 = 1 - (ss_res / ss_tot)
            
        r2_scores.append(r2)
    
    # Calculate weighted average
    weighted_r2 = np.average(r2_scores, weights=weights)
    
    return weighted_r2, r2_scores