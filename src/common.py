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

from configs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGE_SIZE, BIOMASS_FEAT_WEIGHT

class EWC:
    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.device = device
        self.importance = importance
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            images = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            biomass_pred, _, _, _ = self.model(images)
            loss = nn.functional.huber_loss(biomass_pred, targets) * BIOMASS_FEAT_WEIGHT
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.importance * loss

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
    wide = wide.rename(columns={'clean_id': 'sample_id'})
    
    # Scale Targets to KG (kilogra scale)
    wide[target_cols] = wide[target_cols].astype(float) / 1000.0
    
    logger.info(f"Data Loaded. Rows: {len(wide)}")
    return wide
    
def get_image_data_transforms_v2():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1)
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
    all_biomass, all_aux, all_species, all_month = [], [], [], []

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
            b, a, s, m = model(img_aug)
            all_biomass.append(b)
            all_aux.append(a)
            all_species.append(s)
            all_month.append(m)

    return (
        torch.stack(all_biomass).mean(0),
        torch.stack(all_aux).mean(0),
        torch.stack(all_species).mean(0),
        torch.stack(all_month).mean(0),
    )



def setup_logging(logger_name="System Logger", log_dir='logs', file_name_part=None) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir_name = f"{file_name_part}_{timestamp}" if file_name_part else timestamp
    session_dir = os.path.join(log_dir, session_dir_name)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, 'plots'), exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    file_handler = logging.FileHandler(os.path.join(session_dir, 'session.log'), encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return session_dir

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
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    n_samples = len(y_true) // 5
    w_flat = np.tile(weights, n_samples)
    
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    ss_tot = np.sum(w_flat * (y_true - np.average(y_true, weights=w_flat))**2)
    
    return 1 - (ss_res / (ss_tot + 1e-8))

def check_group_leakage(train_df, holdout_df, group_col, logger):
    overlap = set(train_df[group_col]) & set(holdout_df[group_col])
    if overlap: logger.warning(f"Leakage detected: {overlap}")

def upsample_minority_classes(df, target_col):
    counts = df[target_col].value_counts()
    target = int(counts.max())
    dfs = [df]
    for cls, count in counts.items():
        if count < target:
            add = target - count
            dfs.append(df[df[target_col] == cls].sample(n=add, replace=True, random_state=42))
    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def plot_training_history(history, fold, session_dir):
    """
    Plots metrics including Independent Fold components.
    """
    save_dir = os.path.join(session_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    def try_plot(ax_idx, key, label, color, style='-'):
        if key in history and len(history[key]) > 0:
            axes[ax_idx].plot(history[key], label=label, color=color, linestyle=style)

    # 1. Total Loss
    try_plot(0, 'train_loss', 'Train', 'tab:blue')
    try_plot(0, 'val_loss', 'Val', 'tab:red')
    try_plot(0, 'ind_loss', 'Holdout', 'tab:green', ':')
    axes[0].set_title('Total Loss')

    # 2. Biomass Loss
    try_plot(1, 'loss_biomass', 'Train', 'tab:blue')
    try_plot(1, 'val_loss_biomass', 'Val', 'tab:red')
    try_plot(1, 'ind_loss_biomass', 'Holdout', 'tab:green', ':')
    axes[1].set_title('Biomass Loss')

    # 3. Aux Loss (ADDED IND)
    try_plot(2, 'loss_aux', 'Train', 'tab:blue')
    try_plot(2, 'val_loss_aux', 'Val', 'tab:red')
    try_plot(2, 'ind_loss_aux', 'Holdout', 'tab:green', ':') 
    axes[2].set_title('Aux Loss')

    # 4. Species Loss (ADDED IND)
    try_plot(3, 'loss_species', 'Train', 'tab:blue')
    try_plot(3, 'val_loss_species', 'Val', 'tab:red')
    try_plot(3, 'ind_loss_species', 'Holdout', 'tab:green', ':')
    axes[3].set_title('Species Loss')

    # 5. Month Loss (ADDED IND)
    try_plot(4, 'loss_month', 'Train', 'tab:blue')
    try_plot(4, 'val_loss_month', 'Val', 'tab:red')
    try_plot(4, 'ind_loss_month', 'Holdout', 'tab:green', ':')
    axes[4].set_title('Month Loss')

    # 6. R2
    try_plot(5, 'val_r2', 'Val R2', 'red')
    try_plot(5, 'holdout_r2', 'Holdout R2', 'green')
    axes[5].set_title('R2 Metrics')
    axes[5].axhline(0, color='black', alpha=0.3)
    axes[5].set_ylim(-1.5, 1.1)

    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_metrics.png"))
    plt.close()