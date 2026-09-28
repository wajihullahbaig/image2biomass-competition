"""
CSIRO Image2Biomass: Dual-Stream DINO Vision Pipeline with Interval Classification
Unified 5-Fold Training Script (train.py)

Key Principles & Solutions Heritage:
1. 1st-Place Solution:
   - 1000x1000 Centerline 1:1 view split preserving natural quadrat aspect ratio.
   - Shared DINO ViT backbone with Cross-View Multi-Head Attention layer.
   - Auxiliary 7-interval classification heads (UEPNet crowd counting formulation).
   - 2-Stage Staged Training: Stage 1 (Backbone frozen) -> Stage 2 (Differential LR).
   - Test-Time Augmentation (TTA) with mirrored horizontal-flip panoramic views.
2. 2nd-Place Solution:
   - ColorJitter (brightness, contrast, saturation, hue) for pasture sunlight variations.
   - Dead biomass fringe expansion (>20 * 1.1, <10 * 0.9) in post-processing.
3. 3rd-Place Solution:
   - Vertical 4-strip permutation (p=0.5): mass strictly conserved while breaking position bias.
   - Random grayscale (p=0.2): forces leaf texture & morphology learning over color shortcuts.
   - View swap (p=0.5): swaps Left and Right views into cross-view attention.
   - Camera focal scale simulation (p=0.2).
4. Competition-Winning Cross-Validation & Anti-Leakage:
   - StratifiedGroupKFold on 'Sampling_Date' stratified by 'State' (seed=223).
   - Host-confirmed: prevents temporal leakage across unseen flight dates.
   - Full 439 samples from wide.csv (including high-biomass synthetic pasture augmentations).
5. Evaluation Metric & Soft Physics:
   - Official competition R2 on log1p scale.
   - Soft physics calibration (clover_scale=0.8, dead fringe expansion, mass conservation blending).
"""

import os
import sys
import time
import math
import random
import argparse
import logging
from datetime import datetime

# Ensure standard UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UserWarning, module="timm.layers.attention")

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import timm

# ==============================================================================
# Constants & Competition Targets
# ==============================================================================
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Non-uniform 7-interval thresholds from 1st-place solution (UEPNet crowd-counting formulation)
BORDERS_DICT = {
    'Dry_Green_g':  [1.6e-05, 13.4232, 27.0782, 45.5236, 79.834, 157.9836],
    'Dry_Dead_g':   [1.6e-05, 6.1407, 13.1192, 23.277, 38.8581, 83.8407],
    'Dry_Clover_g': [1.6e-05, 3.9, 10.5353, 20.6523, 37.5911, 71.7865],
    'GDM_g':        [1.6e-05, 16.5143, 30.507, 49.5585, 81.0, 157.9836],
    'Dry_Total_g':  [1.6e-05, 23.4907, 41.1, 61.1, 96.8288, 185.7],
}


def set_seed(seed=223):
    """Sets deterministic random seeds for full reproducibility and accelerates cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# Cross-Validation: Host-Confirmed Anti-Temporal-Leakage Grouping (0.59825 Peak)
# ==============================================================================
def create_grouped_stratified_folds(df, n_splits=5, seed=223, group_col='Sampling_Date', strat_col='State'):
    """
    Creates 5 cross-validation folds grouped by Sampling_Date and stratified by State.
    Competition test images are captured on completely unseen flight dates. Grouping by
    Sampling_Date guarantees that train and validation folds never share images from the
    same flight, weather conditions, or sun angles.
    """
    df = df.copy().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, df[strat_col], groups=df[group_col])):
        df.loc[val_idx, 'fold'] = fold_idx
    return df


def align_img_size_to_backbone(img_size, backbone_name):
    """Ensures input image size is cleanly divisible by the ViT patch size."""
    if 'patch14' in backbone_name:
        patch = 14
    elif 'patch16' in backbone_name:
        patch = 16
    else:
        patch = 14 if 'dinov2' in backbone_name else 16

    if img_size % patch != 0:
        return int(round(img_size / patch)) * patch
    return img_size


def get_interval_labels(targets_np):
    """Discretizes continuous biomass values (grams) into 7 classes (0..6)."""
    labels = np.zeros_like(targets_np, dtype=np.int64)
    for col_idx, col_name in enumerate(TARGET_NAMES):
        borders = BORDERS_DICT.get(col_name)
        if borders is not None:
            labels[:, col_idx] = np.digitize(targets_np[:, col_idx], borders)
        else:
            labels[:, col_idx] = np.clip(np.digitize(targets_np[:, col_idx], [0, 5, 15, 30, 60, 120]), 0, 6)
    return labels


# ==============================================================================
# Augmentations (1st + 2nd + 3rd Place Solutions)
# ==============================================================================
def apply_camera_scale_simulation(image_np, prob=0.2):
    """Simulates focal distance variation by downscaling and zero-padding (1st/3rd place)."""
    if prob > 0 and random.random() < prob:
        h, w = image_np.shape[:2]
        bg = np.zeros_like(image_np)
        scale = random.uniform(0.85, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        bg[top:top + new_h, left:left + new_w] = resized
        return bg
    return image_np


def permute_vertical_strips(image_np, n_strips=4, prob=0.5):
    """
    Randomly permutes N vertical strips of the pasture quadrat sub-image.
    Because biomass is purely additive mass, shuffling vertical slices
    conserves total grams in the frame while preventing spatial overfitting.
    From 3rd Place Solution (+0.02 gain).
    """
    if prob > 0 and random.random() < prob:
        strips = np.array_split(image_np, n_strips, axis=1)
        random.shuffle(strips)
        return np.concatenate(strips, axis=1)
    return image_np


class DualStreamBiomassDataset(Dataset):
    """
    Dual-Stream Dataset for Panoramic (2:1) Pasture Images.
    Splits wide images into Left and Right 1:1 views for shared ViT feature extraction.
    """
    def __init__(self, df, img_size=384, is_training=True,
                 camera_scaling_prob=0.2, strip_shuffle_prob=0.5, view_swap_prob=0.5, grayscale_prob=0.2):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_training = is_training
        self.camera_scaling_prob = camera_scaling_prob
        self.strip_shuffle_prob = strip_shuffle_prob
        self.view_swap_prob = view_swap_prob

        if self.is_training:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=grayscale_prob),  # 3rd place: learns morphology over color
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        self.has_targets = all(t in self.df.columns for t in TARGET_NAMES)
        if self.has_targets:
            self.targets_reg = self.df[TARGET_NAMES].values.astype(np.float32)
            self.targets_cls = get_interval_labels(self.targets_reg)

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, rel_path):
        if os.path.exists(rel_path):
            return rel_path
        fname = os.path.basename(rel_path)
        for cand_dir in ['train', 'test', 'images', os.path.join('..', 'train'), os.path.join('..', 'test'), './data']:
            cand = os.path.join(cand_dir, fname)
            if os.path.exists(cand):
                return cand
        for root, _, files in os.walk('.'):
            if fname in files:
                return os.path.join(root, fname)
        raise FileNotFoundError(f"Cannot find image: {rel_path}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row['image_path'])

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = raw_rgb.shape
        mid_w = w // 2

        left_np = raw_rgb[:, :mid_w].copy()
        right_np = raw_rgb[:, mid_w:].copy()

        # 3rd-Place Augmentations during training
        if self.is_training:
            # 1. Left/Right view swap (symmetry)
            if self.view_swap_prob > 0 and random.random() < self.view_swap_prob:
                left_np, right_np = right_np, left_np

            # 2. Camera focal/scaling simulation
            if self.camera_scaling_prob > 0:
                left_np = apply_camera_scale_simulation(left_np, prob=self.camera_scaling_prob)
                right_np = apply_camera_scale_simulation(right_np, prob=self.camera_scaling_prob)

            # 3. Vertical 4-strip permutation (mass-conserving spatial decorrelation)
            if self.strip_shuffle_prob > 0:
                left_np = permute_vertical_strips(left_np, n_strips=4, prob=self.strip_shuffle_prob)
                right_np = permute_vertical_strips(right_np, n_strips=4, prob=self.strip_shuffle_prob)

        tensor_l = self.transform(Image.fromarray(left_np))
        tensor_r = self.transform(Image.fromarray(right_np))

        item = {
            'image_left': tensor_l,
            'image_right': tensor_r,
            'sample_id': row.get('sample_id', row.get('clean_id', f'sample_{idx}')),
            'state': row.get('State', 'Unknown')
        }

        if self.has_targets:
            item['targets_reg'] = torch.tensor(self.targets_reg[idx], dtype=torch.float32)
            item['targets_cls'] = torch.tensor(self.targets_cls[idx], dtype=torch.long)

        return item


# ==============================================================================
# Model Architecture: Dual-Stream DINO ViT with Cross-View Attention
# ==============================================================================
class DualStreamBiomassModel(nn.Module):
    """
    Dual-Stream Vision Transformer:
    1. Shared DINO ViT Backbone (vit_small_patch16_dinov3_qkvb / vit_base_patch16_dinov3_qkvb).
    2. Multi-Head Cross-View Attention layer across Left and Right sub-quadrats.
    3. Fusion MLP projecting joint representations.
    4. 5 Independent Continuous Regression Heads (Softplus activation).
    5. 5 Auxiliary Interval Classification Heads (7 classes each).
    """
    def __init__(self, backbone_name="vit_small_patch16_dinov3_qkvb", num_targets=5, num_intervals=7, fusion_dim=384, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.fusion_dim = fusion_dim
        self.num_targets = num_targets
        self.num_intervals = num_intervals

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True
        )
        self.backbone_dim = self.backbone.num_features

        # Cross-View Multi-Head Attention
        num_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=self.backbone_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.backbone_dim)

        # Joint Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5 Continuous Regression Heads
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(fusion_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(self.num_targets)
        ])

        # 5 Auxiliary Interval Classification Heads
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, self.num_intervals)
            ) for _ in range(self.num_targets)
        ])

    def extract_features(self, x):
        feats = self.backbone(x)
        if len(feats.shape) == 3:
            return feats.mean(dim=1)
        elif len(feats.shape) == 4:
            return feats.mean(dim=[2, 3])
        return feats

    def forward(self, img_left, img_right):
        feat_l = self.extract_features(img_left)
        feat_r = self.extract_features(img_right)

        # Cross-View Attention
        tokens = torch.stack([feat_l, feat_r], dim=1)
        attn_out, _ = self.cross_view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Concat & Fusion
        fused = self.fusion_mlp(torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1))

        reg_preds = [F.softplus(head(fused)) for head in self.reg_heads]
        cls_preds = [head(fused) for head in self.cls_heads]

        return reg_preds, cls_preds


# ==============================================================================
# Dual-Objective Loss Function
# ==============================================================================
class WeightedBiomassLoss(nn.Module):
    """Combines SmoothL1 Continuous Regression with Auxiliary Interval Classification."""
    def __init__(self, loss_weights=OFFICIAL_WEIGHTS, cls_weight=0.3):
        super().__init__()
        self.weights = loss_weights
        self.cls_weight = cls_weight
        self.criterion_reg = nn.SmoothL1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, predictions_reg, predictions_cls, targets_reg, targets_cls=None):
        device = targets_reg.device
        w = torch.tensor(self.weights, device=device, dtype=torch.float32)

        loss_reg = torch.tensor(0.0, device=device)
        for i in range(5):
            pred_i = predictions_reg[i].squeeze(-1) if isinstance(predictions_reg, list) else predictions_reg[:, i]
            loss_reg += w[i] * self.criterion_reg(pred_i, targets_reg[:, i])

        loss_cls = torch.tensor(0.0, device=device)
        if predictions_cls is not None and targets_cls is not None:
            for i in range(5):
                loss_cls += w[i] * self.criterion_cls(predictions_cls[i], targets_cls[:, i].long())

        total_loss = loss_reg + (self.cls_weight * loss_cls)
        return total_loss, loss_reg, loss_cls


# ==============================================================================
# Evaluation Metric & Physical Post-Processing (Exact Kaggle Benchmark Recipe)
# ==============================================================================
def calculate_competition_r2(y_true, y_pred, weights=OFFICIAL_WEIGHTS):
    """
    Calculates official weighted multi-target R2 metric on log1p(biomass) scale.
    Matches the Kaggle competition evaluation metric.
    """
    y_true = np.array(y_true, dtype=float).reshape(-1, 5)
    y_pred = np.array(y_pred, dtype=float).reshape(-1, 5)
    w = np.array(weights, dtype=float)

    yt = np.log1p(np.maximum(0, y_true))
    yp = np.log1p(np.maximum(0, y_pred))

    r2_scores = []
    for i in range(5):
        ss_res = np.sum((yt[:, i] - yp[:, i]) ** 2)
        ss_tot = np.sum((yt[:, i] - np.mean(yt[:, i])) ** 2)
        score = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
        r2_scores.append(score)

    return float(np.sum(w * np.array(r2_scores))), r2_scores


def soft_physics_postprocess(preds_np, clover_scale=0.8, dead_upper_thresh=20.0, dead_upper_scale=1.1, dead_lower_thresh=10.0, dead_lower_scale=0.9, gdm_weight=0.5, total_weight=0.5):
    """
    Enforces physical mass relationships, thatch fringe expansion, and clover calibration:
    1. Clover downscaling by 0.8 (corrects systematic overprediction).
    2. Dead biomass fringe expansion (3rd Place calibration: overcomes regression compression).
    3. Mass-conservation blends:
       - GDM = 0.5 * GDM + 0.5 * (Green + Clover)
       - Total = 0.5 * Total + 0.5 * (Green + Clover + Dead)
    4. Non-negativity clamp (val >= 0.0).
    """
    preds = np.maximum(preds_np.copy(), 0.0)
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2] * clover_scale
    gdm = preds[:, 3]
    total = preds[:, 4]

    # Dead biomass fringe expansion
    dead = np.where(dead > dead_upper_thresh, dead * dead_upper_scale,
           np.where(dead < dead_lower_thresh, dead * dead_lower_scale, dead))

    gdm_blended = gdm_weight * gdm + (1.0 - gdm_weight) * (green + clover)
    total_blended = total_weight * total + (1.0 - total_weight) * (green + clover + dead)

    return np.maximum(np.column_stack([green, dead, clover, gdm_blended, total_blended]), 0.0)


# ==============================================================================
# Evaluation Function with TTA
# ==============================================================================
def evaluate(model, val_loader, criterion, device, use_tta=True):
    model.eval()
    val_loss_total, val_loss_reg, val_loss_cls = 0.0, 0.0, 0.0
    val_preds_list, val_true_list = [], []
    correct_cls, total_cls = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            img_l = batch['image_left'].to(device)
            img_r = batch['image_right'].to(device)
            t_reg = batch['targets_reg'].to(device)
            t_cls = batch['targets_cls'].to(device)

            with torch.amp.autocast('cuda'):
                if use_tta:
                    # Standard view
                    r1, c1 = model(img_l, img_r)
                    # Mirrored horizontal view (flip right view as left, flip left view as right)
                    r2, c2 = model(torch.flip(img_r, [3]), torch.flip(img_l, [3]))
                    r = [(a + b) * 0.5 for a, b in zip(r1, r2)]
                    c = [(a + b) * 0.5 for a, b in zip(c1, c2)]
                else:
                    r, c = model(img_l, img_r)

                loss, loss_reg, loss_cls = criterion(r, c, t_reg, t_cls)

            val_loss_total += loss.item() * len(img_l)
            val_loss_reg += loss_reg.item() * len(img_l)
            val_loss_cls += loss_cls.item() * len(img_l)

            val_preds_list.append(torch.cat(r, dim=1).cpu().numpy())
            val_true_list.append(t_reg.cpu().numpy())

            if c is not None and t_cls is not None:
                for i in range(5):
                    preds_i = c[i].argmax(dim=-1)
                    correct_cls += (preds_i == t_cls[:, i]).sum().item()
                total_cls += len(t_cls) * 5

    n_samples = len(val_loader.dataset)
    val_loss = val_loss_total / n_samples
    val_reg = val_loss_reg / n_samples
    val_cls = val_loss_cls / n_samples
    cls_acc = (correct_cls / total_cls) if total_cls > 0 else 0.0

    v_true = np.concatenate(val_true_list, axis=0)
    v_pred_raw = np.concatenate(val_preds_list, axis=0)
    v_pred_post = soft_physics_postprocess(v_pred_raw)

    r2_raw, per_target_raw = calculate_competition_r2(v_true, v_pred_raw)
    r2_post, per_target_post = calculate_competition_r2(v_true, v_pred_post)

    return val_loss, val_reg, val_cls, r2_raw, r2_post, per_target_post, cls_acc, v_pred_raw, v_pred_post


# ==============================================================================
# Logging Setup
# ==============================================================================
def setup_logging(output_dir="models", log_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(log_dir, f"dual_stream_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    logger = logging.getLogger("DualStreamTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Session log file
    session_log_path = os.path.join(session_dir, "session.log")
    fh1 = logging.FileHandler(session_log_path, mode='w', encoding='utf-8')
    fh1.setLevel(logging.INFO)
    fh1.setFormatter(file_formatter)
    logger.addHandler(fh1)

    # Output log file
    output_log_path = os.path.join(output_dir, "train.log")
    fh2 = logging.FileHandler(output_log_path, mode='w', encoding='utf-8')
    fh2.setLevel(logging.INFO)
    fh2.setFormatter(file_formatter)
    logger.addHandler(fh2)

    return logger, session_dir, session_log_path


# ==============================================================================
# Data Loading & Verification (Always uses Wide Formatting)
# ==============================================================================
def load_and_pivot_data(data_path='wide.csv', train_csv_path='train.csv', logger=None):
    """
    Loads wide format dataset.
    1. If wide.csv exists, loads all 439 samples directly (including synthetic samples).
    2. If missing, automatically pivots train.csv into wide format.
    """
    if os.path.exists(data_path):
        if logger:
            logger.info(f"  [DATA] Successfully loaded existing wide dataset: '{data_path}'")
        df = pd.read_csv(data_path)
    elif os.path.exists(train_csv_path):
        if logger:
            logger.info(f"  [DATA] '{data_path}' not found. Auto-pivoting '{train_csv_path}' into wide format...")
        raw_df = pd.read_csv(train_csv_path)
        if 'target_name' in raw_df.columns:
            raw_df['clean_id'] = raw_df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
            targets = raw_df.pivot_table(index='clean_id', columns='target_name', values='target', aggfunc='max').reset_index()
            meta = raw_df[['clean_id', 'image_path', 'Sampling_Date', 'State', 'Species']].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
            df = pd.merge(meta, targets, on='clean_id', how='left')
            df['sample_id'] = df['clean_id']
        else:
            df = raw_df.copy()
        df.to_csv('wide.csv', index=False)
    else:
        raise FileNotFoundError(f"Neither '{data_path}' nor '{train_csv_path}' could be found.")

    if 'clean_id' not in df.columns:
        df['clean_id'] = df['sample_id']

    for col in TARGET_NAMES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Derived targets if missing
    if df['GDM_g'].sum() == 0 and 'Dry_Green_g' in df.columns and 'Dry_Clover_g' in df.columns:
        df['GDM_g'] = df['Dry_Green_g'] + df['Dry_Clover_g']
    if df['Dry_Total_g'].sum() == 0 and 'Dry_Dead_g' in df.columns:
        df['Dry_Total_g'] = df['GDM_g'] + df['Dry_Dead_g']

    return df


# ==============================================================================
# CLI Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream DINO Training")
    parser.add_argument('--data_path', type=str, default='wide.csv', help='Path to wide format dataset CSV (wide.csv with 439 samples)')
    parser.add_argument('--backbone', type=str, default='vit_small_patch16_dinov3_qkvb',
                        help='Vision Transformer backbone (vit_small_patch16_dinov3_qkvb / vit_base_patch16_dinov3_qkvb)')
    parser.add_argument('--img_size', type=int, default=384, help='Input resolution for each sub-image (384 for ultra-fast, 512 for max res)')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps (effective BS = batch_size * grad_accum)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate for heads')
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1, help='Differential LR factor for backbone during Stage 2')
    parser.add_argument('--stage1_epochs', type=int, default=8, help='Stage 1: Frozen backbone warm-up epochs (default: 8)')
    parser.add_argument('--stage2_epochs', type=int, default=25, help='Stage 2: Full model fine-tuning epochs (default: 25)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--start_fold', type=int, default=1, help='Starting fold index (1-based, e.g. 2 to resume from Fold 2)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save checkpoints and OOF predictions')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save timestamped session logs')
    parser.add_argument('--seed', type=int, default=223, help='Random seed (223 gives optimal balanced date splits)')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Enable TTA for checkpoint validation and final OOF')
    return parser.parse_args()


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def run_training():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, session_dir, session_log_path = setup_logging(args.output_dir, args.log_dir)

    # Automatically align image size to ViT patch size
    aligned_size = align_img_size_to_backbone(args.img_size, args.backbone)
    if aligned_size != args.img_size:
        logger.info(f"[RESCALE] Aligned img_size from {args.img_size} to {aligned_size} (divisible by backbone patch size)")
        args.img_size = aligned_size

    # --------------------------------------------------------------------------
    # STEP 1: Initialization & Configuration
    # --------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("[STEP 1/7] INITIALIZATION & CONFIGURATION")
    logger.info(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"  Backbone: {args.backbone} (ViT Small DINOv3, dim=384, ~21.6M params)")
    logger.info(f"  Sub-Image Resolution: {args.img_size}x{args.img_size} per view (Effective panoramic field: {args.img_size*2}x{args.img_size})")
    logger.info(f"  Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}) | Effective Batch Size: {args.batch_size * args.grad_accum}")
    logger.info(f"  Learning Rate: {args.lr:.1e} (Differential factor: {args.backbone_lr_factor}x for backbone in Stage 2)")
    logger.info(f"  Schedule: Stage 1 = {args.stage1_epochs} eps (Heads) | Stage 2 = {args.stage2_epochs} eps (Full Fine-Tuning) | Total = {args.stage1_epochs + args.stage2_epochs} eps/fold")
    logger.info(f"  Random Seed: {args.seed} | Validation TTA: {args.use_tta}")
    logger.info(f"  Checkpoints Directory: '{args.output_dir}'")
    logger.info(f"  Session Logs Directory: '{session_dir}'")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 2: Data Loading & Verification (Always uses Wide Formatting)
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[STEP 2/7] DATA LOADING & DATASET VERIFICATION")
    df = load_and_pivot_data(data_path=args.data_path, train_csv_path='train.csv', logger=logger)
    logger.info(f"  Total Samples Loaded: {len(df)}")
    if 'is_synthetic' in df.columns:
        n_real = (df['is_synthetic'] == False).sum()
        n_syn = (df['is_synthetic'] == True).sum()
        logger.info(f"  Sample Composition: {n_real} Real Pasture + {n_syn} Synthetic Pasture Samples")
    logger.info(f"  Unique Flight Dates: {df['Sampling_Date'].nunique()} | States: {dict(df['State'].value_counts())}")
    for t in TARGET_NAMES:
        logger.info(f"    - {t:15s}: Mean={df[t].mean():6.2f}g | Min={df[t].min():6.2f}g | Max={df[t].max():6.2f}g")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 3: Anti-Temporal Leakage Cross-Validation Grouping
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[STEP 3/7] CROSS-VALIDATION GROUPING (ANTI-TEMPORAL LEAKAGE)")
    logger.info(f"  Splitting {len(df)} samples into {args.n_folds} folds grouped by 'Sampling_Date', stratified by 'State' (seed={args.seed})")
    df = create_grouped_stratified_folds(df, n_splits=args.n_folds, seed=args.seed)

    for f in range(args.n_folds):
        sub_val = df[df['fold'] == f]
        n_train = len(df) - len(sub_val)
        st_dict = dict(sub_val['State'].value_counts())
        n_dates = sub_val['Sampling_Date'].nunique()
        logger.info(f"  Fold {f+1}: Train={n_train} | Val={len(sub_val)} (Dates={n_dates} | States={st_dict} | Total Mean={sub_val['Dry_Total_g'].mean():.1f}g)")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 4: Augmentations Logging
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[STEP 4/7] MULTI-TIER AUGMENTATIONS (1st + 2nd + 3rd PLACE HERITAGE)")
    logger.info("  1st-Place: 1:1 view split, H/V flips, 90-deg rotations, Cross-View Attention, TTA")
    logger.info("  2nd-Place: ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) for lighting variation")
    logger.info("  3rd-Place: Vertical 4-strip permutation (p=0.5), RandomGrayscale (p=0.2), View Swap (p=0.5), Camera Scaling (p=0.2)")
    logger.info("  Post-Process: soft_physics_postprocess (clover_scale=0.8, dead fringe expansion, mass conservation blends)")
    logger.info("=" * 70)

    oof_preds_post = np.zeros((len(df), 5), dtype=np.float32)
    oof_preds_raw = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = df[TARGET_NAMES].values.astype(np.float32)

    fold_scores = []
    start_total_time = time.time()
    criterion = WeightedBiomassLoss(loss_weights=OFFICIAL_WEIGHTS, cls_weight=0.3).to(device)

    # --------------------------------------------------------------------------
    # STEP 5 & 6: Staged Training Across Folds
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"[STEP 5/7] EXECUTING {args.n_folds}-FOLD STAGED TRAINING PIPELINE")
    logger.info("=" * 70)

    for fold in range(args.n_folds):
        fold_num = fold + 1
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        val_indices = df[df['fold'] == fold].index.values
        ckpt_path = os.path.join(args.output_dir, f"best_model_fold{fold_num}.pt")

        # Resume / Cache check
        if fold_num < args.start_fold:
            if os.path.exists(ckpt_path):
                logger.info(f"\n{'='*30} FOLD {fold_num} / {args.n_folds} [CACHED] {'='*30}")
                logger.info(f"[RESUME] Loading existing checkpoint for Fold {fold_num}: {ckpt_path}")
                val_ds = DualStreamBiomassDataset(val_df, img_size=args.img_size, is_training=False)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                cached_model = DualStreamBiomassModel(backbone_name=args.backbone, pretrained=False).to(device)
                cached_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                _, _, _, r2_raw, r2_post, per_target, _, p_raw, p_post = evaluate(
                    cached_model, val_loader, criterion, device, use_tta=args.use_tta
                )
                logger.info(f"[RESUME] Fold {fold_num} Verified: Post R2 = {r2_post:.4f} (Raw R2 = {r2_raw:.4f})")
                oof_preds_post[val_indices] = p_post
                oof_preds_raw[val_indices] = p_raw
                fold_scores.append(r2_post)
                del cached_model
                torch.cuda.empty_cache()
                continue
            else:
                logger.warning(f"[RESUME] Checkpoint {ckpt_path} not found for Fold {fold_num}. Training from scratch.")

        logger.info(f"\n{'='*30} FOLD {fold_num} / {args.n_folds} {'='*30}")
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        logger.info(f"Training on {len(train_df)} samples | Validation on {len(val_df)} samples.")

        train_ds = DualStreamBiomassDataset(train_df, img_size=args.img_size, is_training=True)
        val_ds = DualStreamBiomassDataset(val_df, img_size=args.img_size, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = DualStreamBiomassModel(backbone_name=args.backbone, pretrained=True).to(device)
        scaler = torch.amp.GradScaler('cuda')

        best_fold_r2 = -float('inf')
        best_preds_post = None
        best_preds_raw = None

        # ----------------------------------------------------------------------
        # STAGE 1: Warm-up Heads (Backbone FROZEN)
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold_num}] STAGE 1: Training Heads ({args.stage1_epochs} epochs | Backbone FROZEN | LR: {args.lr:.1e}) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)

        for epoch in range(1, args.stage1_epochs + 1):
            t0 = time.time()
            model.train()
            tr_loss_sum, tr_reg_sum, tr_cls_sum = 0.0, 0.0, 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f'Fold {fold_num} [S1 Ep {epoch:02d}/{args.stage1_epochs:02d}]', leave=False)
            for step, batch in enumerate(pbar):
                img_l = batch['image_left'].to(device)
                img_r = batch['image_right'].to(device)
                t_reg = batch['targets_reg'].to(device)
                t_cls = batch['targets_cls'].to(device)

                with torch.amp.autocast('cuda'):
                    r, c = model(img_l, img_r)
                    loss, l_reg, l_cls = criterion(r, c, t_reg, t_cls)
                    loss_scaled = loss / args.grad_accum

                scaler.scale(loss_scaled).backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                tr_loss_sum += loss.item() * len(img_l)
                tr_reg_sum += l_reg.item() * len(img_l)
                tr_cls_sum += l_cls.item() * len(img_l)
                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'reg': f'{l_reg.item():.2f}', 'cls': f'{l_cls.item():.2f}'})

            n_tr = len(train_loader.dataset)
            tr_loss, tr_reg, tr_cls = tr_loss_sum / n_tr, tr_reg_sum / n_tr, tr_cls_sum / n_tr
            val_loss, _, _, r2_raw, r2_post, per_target, cls_acc, v_raw, v_post = evaluate(
                model, val_loader, criterion, device, use_tta=args.use_tta
            )
            elapsed = time.time() - t0

            is_best = r2_post > best_fold_r2
            if is_best:
                best_fold_r2 = r2_post
                best_preds_post = v_post
                best_preds_raw = v_raw
                torch.save(model.state_dict(), ckpt_path)

            star = '  ★ Best Model Saved' if is_best else ''
            logger.info(f"[S1 Ep {epoch:02d}/{args.stage1_epochs:02d}] Train: {tr_loss:.3f} (reg:{tr_reg:.2f}, cls:{tr_cls:.2f}) | "
                        f"Val: {val_loss:.3f} | R2 Raw: {r2_raw:.4f} | R2 SoftBlend: {r2_post:.4f} | Cls Acc: {cls_acc:.1%} ({elapsed:.0f}s){star}")

        # ----------------------------------------------------------------------
        # STAGE 2: Full End-to-End Fine-Tuning (Differential LR + CosineAnnealing)
        # ----------------------------------------------------------------------
        backbone_lr = args.lr * args.backbone_lr_factor
        logger.info(f"\n--- [Fold {fold_num}] STAGE 2: Full Fine-Tuning ({args.stage2_epochs} epochs | Backbone LR: {backbone_lr:.1e}, Heads LR: {args.lr:.1e}) ---")
        for p in model.backbone.parameters():
            p.requires_grad = True

        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': args.lr}
        ], weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=args.lr * 0.01)

        for epoch in range(1, args.stage2_epochs + 1):
            t0 = time.time()
            model.train()
            tr_loss_sum, tr_reg_sum, tr_cls_sum = 0.0, 0.0, 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f'Fold {fold_num} [S2 Ep {epoch:02d}/{args.stage2_epochs:02d}]', leave=False)
            for step, batch in enumerate(pbar):
                img_l = batch['image_left'].to(device)
                img_r = batch['image_right'].to(device)
                t_reg = batch['targets_reg'].to(device)
                t_cls = batch['targets_cls'].to(device)

                with torch.amp.autocast('cuda'):
                    r, c = model(img_l, img_r)
                    loss, l_reg, l_cls = criterion(r, c, t_reg, t_cls)
                    loss_scaled = loss / args.grad_accum

                scaler.scale(loss_scaled).backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                tr_loss_sum += loss.item() * len(img_l)
                tr_reg_sum += l_reg.item() * len(img_l)
                tr_cls_sum += l_cls.item() * len(img_l)
                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'reg': f'{l_reg.item():.2f}', 'cls': f'{l_cls.item():.2f}'})

            scheduler.step()
            n_tr = len(train_loader.dataset)
            tr_loss, tr_reg, tr_cls = tr_loss_sum / n_tr, tr_reg_sum / n_tr, tr_cls_sum / n_tr
            val_loss, _, _, r2_raw, r2_post, per_target, cls_acc, v_raw, v_post = evaluate(
                model, val_loader, criterion, device, use_tta=args.use_tta
            )
            elapsed = time.time() - t0

            is_best = r2_post > best_fold_r2
            if is_best:
                best_fold_r2 = r2_post
                best_preds_post = v_post
                best_preds_raw = v_raw
                torch.save(model.state_dict(), ckpt_path)

            star = '  ★ Best Model Saved' if is_best else ''
            logger.info(f"[S2 Ep {epoch:02d}/{args.stage2_epochs:02d}] Train: {tr_loss:.3f} (reg:{tr_reg:.2f}, cls:{tr_cls:.2f}) | "
                        f"Val: {val_loss:.3f} | R2 Raw: {r2_raw:.4f} | R2 SoftBlend: {r2_post:.4f} | Cls Acc: {cls_acc:.1%} ({elapsed:.0f}s){star}")

        logger.info(f"\n>>> Fold {fold_num} Finished! Best SoftBlend R2: {best_fold_r2:.4f} <<<\n")
        oof_preds_post[val_indices] = best_preds_post
        oof_preds_raw[val_indices] = best_preds_raw
        fold_scores.append(best_fold_r2)

    # --------------------------------------------------------------------------
    # STEP 7: Final Out-Of-Fold Evaluation & Export
    # --------------------------------------------------------------------------
    overall_post_r2, per_target_post = calculate_competition_r2(oof_targets, oof_preds_post)
    overall_raw_r2, per_target_raw = calculate_competition_r2(oof_targets, oof_preds_raw)
    total_time_min = (time.time() - start_total_time) / 60.0

    logger.info("\n" + "=" * 70)
    logger.info("[STEP 7/7] FINAL OUT-OF-FOLD (OOF) COMPETITION RESULTS")
    logger.info("=" * 70)
    logger.info(f"[METRIC] OVERALL OOF COMPETITION R2 (SoftBlend): {overall_post_r2:.4f}")
    logger.info(f"         Overall OOF Competition R2 (Raw):       {overall_raw_r2:.4f}")
    logger.info(f"         Per-Fold Scores: {[round(s, 4) for s in fold_scores]}")
    logger.info("         Per-Target Breakdown (SoftBlend):")
    for t_name, score, w in zip(TARGET_NAMES, per_target_post, OFFICIAL_WEIGHTS):
        logger.info(f"           - {t_name:15s} (Weight: {w:.1f}): R2 = {score:.4f}")
    logger.info(f"Total CV Training Time: {total_time_min:.1f} minutes")
    logger.info("=" * 70)

    # Save Out-Of-Fold Predictions CSV
    oof_df = df[['sample_id', 'State', 'Species', 'Sampling_Date'] + TARGET_NAMES].copy()
    for idx, t in enumerate(TARGET_NAMES):
        oof_df[f'pred_{t}'] = oof_preds_post[:, idx]
        oof_df[f'pred_raw_{t}'] = oof_preds_raw[:, idx]

    oof_path = os.path.join(args.output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    oof_session_path = os.path.join(session_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_session_path, index=False)

    logger.info(f"[SAVED] Saved OOF predictions to {oof_path} and {oof_session_path}")
    logger.info(f"[LOG] Complete session log saved to {session_log_path}")


if __name__ == '__main__':
    run_training()
