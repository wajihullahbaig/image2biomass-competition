"""
CSIRO Image2Biomass: Dual-Stream DINO Vision Pipeline with Interval Classification
Unified 5-Fold Training Script (train.py)

Key Features:
1. Balanced Stratification: StratifiedKFold on State + Binned Composite Biomass to eliminate fold distortion.
2. Dual-Stream Architecture: Shared DINO ViT backbone with Cross-View Multi-Head Attention.
3. 5 Direct Regression Heads (Green, Dead, Clover, GDM, Total) - no compounding summation errors.
4. 5 Auxiliary Interval Classification Heads (7 bins from UEPNet crowd counting formulation, +0.03 LB/PB).
5. Dual-Objective Loss: SmoothL1 + 0.3 * CrossEntropy weighted by official competition metric [0.1, 0.1, 0.1, 0.2, 0.5].
6. Robust 2-Stage Training: Stage 1 (Backbone frozen head warm-up) -> Stage 2 (Differential LR full fine-tuning).
7. Soft Physical Post-Processing: Blends direct total/GDM predictions with physical identities + WA zero-dead clipping.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
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


def set_seed(seed=42):
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Balanced Stratification (Fixes the Major Stratification Imbalance)
# ==============================================================================
def create_balanced_stratified_folds(df, n_splits=5, seed=42):
    """
    Creates balanced cross-validation folds by stratifying on both geographic State
    and composite biomass quintiles.

    Why this is critical:
    With only 357 images and 4 states (NSW: 75, Tas: 138, Vic: 112, WA: 32), grouping
    strictly by Sampling_Date causes extreme imbalance where WA dates and rare species
    are completely isolated in single folds, distorting label distributions.
    This joint State + Biomass stratification guarantees that every fold has:
    1. Equal sample counts (~71-72 samples).
    2. Proportional state representation (NSW ~15, Tas ~28, Vic ~22, WA ~6-7).
    3. Balanced label means and standard deviations across all 5 targets.
    """
    df = df.copy().reset_index(drop=True)
    
    # 1. Compute official-weighted composite biomass score
    composite = sum(df[t] * w for t, w in zip(TARGET_NAMES, OFFICIAL_WEIGHTS))
    
    # 2. Discretize composite biomass into 5 quantiles
    try:
        biomass_bins = pd.qcut(composite, q=5, labels=False, duplicates='drop')
    except ValueError:
        biomass_bins = pd.qcut(composite, q=3, labels=False, duplicates='drop')

    # 3. Create composite stratification key
    strat_key = df['State'].astype(str) + '_' + biomass_bins.astype(str)

    # 4. Group any class with fewer members than n_splits to avoid split warnings
    counts = strat_key.value_counts()
    rare_classes = counts[counts < n_splits].index
    strat_key = strat_key.apply(lambda k: k.split('_')[0] + '_other' if k in rare_classes else k)

    # 5. Stratified split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, strat_key)):
        df.loc[val_idx, 'fold'] = fold_idx

    return df


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
# Augmentations & Dataset
# ==============================================================================
def apply_camera_scaling(image_np, prob=0.2):
    """
    Simulates focal distance variation by downscaling image (0.85 - 1.0) and padding
    with black pixels (1st-place solution).
    """
    if random.random() < prob:
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


def apply_vertical_strip_shuffle(image_np, n_strips=4, prob=0.3):
    """
    Shuffles vertical slices of pasture quadrat. Conserves total grams of biomass
    while breaking spatial memorization (3rd-place solution).
    """
    if random.random() < prob:
        strips = np.array_split(image_np, n_strips, axis=1)
        random.shuffle(strips)
        return np.concatenate(strips, axis=1)
    return image_np


def apply_clahe(image_np, prob=0.3):
    """Applies Contrast Limited Adaptive Histogram Equalization in LAB space."""
    if random.random() < prob:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image_np


def apply_gaussian_noise(image_np, prob=0.3, var_limit=(10.0, 50.0)):
    """Injects gentle Gaussian sensor noise."""
    if random.random() < prob:
        row, col, ch = image_np.shape
        var = random.uniform(var_limit[0], var_limit[1])
        sigma = var ** 0.5
        gauss = np.random.normal(0, sigma, (row, col, ch)).astype(np.float32)
        noisy = np.clip(image_np.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy
    return image_np


class DualStreamBiomassDataset(Dataset):
    """
    Dual-Stream Dataset for Panoramic (2:1) Pasture Images.
    Splits 2000x1000 pasture images into Left and Right 1000x1000 views.
    Applies independent augmentations to each view before feeding them
    into a shared Vision Transformer backbone.
    """
    def __init__(self, df, img_size=512, is_training=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_training = is_training

        if self.is_training:
            self.pil_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.pil_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
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
        for cand_dir in ['train', 'test', 'images', os.path.join('..', 'train')]:
            cand = os.path.join(cand_dir, fname)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"Cannot find image: {rel_path}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row['image_path'])

        # Load image (RGB)
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = raw_rgb.shape
        mid_w = w // 2

        # Split into Left and Right views
        left_np = raw_rgb[:, :mid_w].copy()
        right_np = raw_rgb[:, mid_w:].copy()

        if self.is_training:
            # 1. Left/Right view swap (50% prob)
            if random.random() < 0.5:
                left_np, right_np = right_np, left_np

            # 2. Camera focal downscaling (20% prob)
            left_np = apply_camera_scaling(left_np, prob=0.2)
            right_np = apply_camera_scaling(right_np, prob=0.2)

            # 3. Vertical strip permutation (30% prob)
            left_np = apply_vertical_strip_shuffle(left_np, n_strips=4, prob=0.3)
            right_np = apply_vertical_strip_shuffle(right_np, n_strips=4, prob=0.3)

            # 4. CLAHE & Gaussian Noise
            left_np = apply_clahe(left_np, prob=0.3)
            right_np = apply_clahe(right_np, prob=0.3)
            left_np = apply_gaussian_noise(left_np, prob=0.3)
            right_np = apply_gaussian_noise(right_np, prob=0.3)

        # Independent PIL transforms
        left_tensor = self.pil_transform(Image.fromarray(left_np))
        right_tensor = self.pil_transform(Image.fromarray(right_np))

        item = {
            'image_left': left_tensor,
            'image_right': right_tensor,
            'sample_id': row.get('sample_id', row.get('image_path', f'sample_{idx}')),
            'state': row.get('State', 'Unknown')
        }

        if self.has_targets:
            item['targets_reg'] = torch.tensor(self.targets_reg[idx], dtype=torch.float32)
            item['targets_cls'] = torch.tensor(self.targets_cls[idx], dtype=torch.long)

        return item


# ==============================================================================
# Model Architecture: Dual-Stream DINO ViT with Cross-View Attention
# ==============================================================================
class DualStreamDINO(nn.Module):
    """
    Dual-Stream Vision Transformer with:
    1. Shared DINO ViT Backbone extracting global patch embeddings from Left and Right views.
    2. Multi-Head Self-Attention Cross-View Interaction layer across the seam.
    3. Fusion MLP projecting joint representations.
    4. 5 Independent Continuous Regression Heads (3-layer MLPs).
    5. 5 Auxiliary Interval Classification Heads (7 classes each).
    """
    def __init__(self, backbone_name="vit_base_patch14_dinov2", fusion_dim=384, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.fusion_dim = fusion_dim
        self.num_targets = 5
        self.num_intervals = 7

        # 1. Shared Vision Transformer Backbone
        kwargs = {}
        if 'dinov2' in backbone_name or 'patch14' in backbone_name or 'patch16' in backbone_name:
            kwargs['dynamic_img_size'] = True

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs
        )
        self.backbone_dim = self.backbone.num_features

        # 2. Cross-View Interaction: Multi-Head Attention on [Batch, 2 views, embed_dim]
        n_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=self.backbone_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.backbone_dim)

        # 3. Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. 5 Independent Regression Heads (3-layer MLPs)
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim // 2),
                nn.LayerNorm(self.fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(self.fusion_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(self.num_targets)
        ])

        # 5. 5 Auxiliary Classification Heads (7 classes each)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, self.num_intervals)
            ) for _ in range(self.num_targets)
        ])

        self._init_heads()

    def _init_heads(self):
        for m in list(self.reg_heads) + list(self.cls_heads) + [self.fusion_mlp]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def extract_features(self, x):
        feats = self.backbone(x)
        if len(feats.shape) == 3:  # ViT tokens [B, N, C]
            return feats.mean(dim=1)
        elif len(feats.shape) == 4:  # Spatial feature maps [B, C, H, W]
            return feats.mean(dim=[2, 3])
        return feats

    def forward(self, img_left, img_right):
        # 1. Feature extraction through shared backbone
        feat_l = self.extract_features(img_left)
        feat_r = self.extract_features(img_right)

        # 2. Cross-view interaction via self-attention
        tokens = torch.stack([feat_l, feat_r], dim=1)  # [B, 2, C]
        attn_out, _ = self.cross_view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # 3. Concatenate and project through fusion MLP
        fused = torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)
        fused = self.fusion_mlp(fused)

        # 4. Continuous regression predictions (grams)
        # Softplus ensures non-negative biomass predictions
        reg_preds = [F.softplus(head(fused)) for head in self.reg_heads]

        # 5. Discrete interval classification logits
        cls_preds = [head(fused) for head in self.cls_heads]

        return reg_preds, cls_preds


# ==============================================================================
# Dual-Objective Loss & Competition Metric
# ==============================================================================
class WeightedBiomassLoss(nn.Module):
    """
    Weighted SmoothL1 regression + Cross-Entropy auxiliary interval classification.
    Loss weights strictly match the official competition metric weights:
    Dry_Total_g: 0.50, GDM_g: 0.20, Green: 0.10, Dead: 0.10, Clover: 0.10.
    """
    def __init__(self, cls_weight=0.3):
        super().__init__()
        self.criterion_reg = nn.SmoothL1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight
        self.weights = torch.tensor(OFFICIAL_WEIGHTS, dtype=torch.float32)

    def forward(self, reg_preds, cls_preds, targets_reg, targets_cls=None):
        device = targets_reg.device
        w = self.weights.to(device)

        loss_reg = torch.tensor(0.0, device=device)
        for i in range(5):
            pred_i = reg_preds[i].squeeze(-1)
            true_i = targets_reg[:, i]
            loss_reg += w[i] * self.criterion_reg(pred_i, true_i)

        loss_cls = torch.tensor(0.0, device=device)
        if cls_preds is not None and targets_cls is not None:
            for i in range(5):
                pred_c_i = cls_preds[i]
                true_c_i = targets_cls[:, i]
                loss_cls += w[i] * self.criterion_cls(pred_c_i, true_c_i)

        total_loss = loss_reg + (self.cls_weight * loss_cls)
        return total_loss, loss_reg, loss_cls


def calculate_competition_r2(y_true, y_pred, weights=OFFICIAL_WEIGHTS):
    """
    Official CSIRO Competition Metric:
    Sum of individual R2 scores computed in log-space: log(1 + y).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    yt = np.log1p(np.maximum(0, y_true))
    yp = np.log1p(np.maximum(0, y_pred))

    r2_scores = []
    for i in range(5):
        t = yt[:, i]
        p = yp[:, i]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_scores.append(r2)

    return float(np.sum(w * np.array(r2_scores))), r2_scores


def apply_soft_blend_postprocess(preds_5, states=None):
    """
    1st + 2nd Place Post-Processing Formula:
    1. Clover scale by 0.8 (adjusts for test distribution shift).
    2. Dead fringe correction (>20 * 1.1, <10 * 0.9).
    3. Soft physical blend:
       - GDM = 0.5 * pred_GDM + 0.5 * (Green + Clover)
       - Total = 0.5 * pred_Total + 0.5 * (Green + Clover + Dead)
    4. WA State zero-dead correction (pasture thatch in WA is strictly 0.0g).
    5. Non-negative clipping.
    """
    preds = np.maximum(np.asarray(preds_5, dtype=np.float32).copy(), 0.0)
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2] * 0.8
    gdm = preds[:, 3]
    total = preds[:, 4]

    # Dead fringe adjustment
    dead = np.where(dead > 20.0, dead * 1.1,
           np.where(dead < 10.0, dead * 0.9, dead))

    # WA zero-dead physical correction
    if states is not None:
        for idx, st in enumerate(states):
            if str(st).strip() == 'WA':
                dead[idx] = 0.0

    # Soft physical blending
    derived_gdm = green + clover
    gdm_blended = 0.5 * gdm + 0.5 * derived_gdm

    derived_total = green + clover + dead
    total_blended = 0.5 * total + 0.5 * derived_total

    result = np.column_stack([green, dead, clover, gdm_blended, total_blended])
    return np.maximum(result, 0.0)


# ==============================================================================
# Training Engine
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps=1):
    model.train()
    loss_sum, reg_sum, cls_sum, samples = 0.0, 0.0, 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        img_l = batch['image_left'].to(device)
        img_r = batch['image_right'].to(device)
        t_reg = batch['targets_reg'].to(device)
        t_cls = batch['targets_cls'].to(device)
        bs = img_l.size(0)

        with torch.amp.autocast('cuda'):
            reg_preds, cls_preds = model(img_l, img_r)
            loss, loss_reg, loss_cls = criterion(reg_preds, cls_preds, t_reg, t_cls)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_sum += loss.item() * grad_accum_steps * bs
        reg_sum += loss_reg.item() * bs
        cls_sum += loss_cls.item() * bs
        samples += bs

    return loss_sum / max(1, samples), reg_sum / max(1, samples), cls_sum / max(1, samples)


def validate(model, loader, criterion, device, val_df, use_tta=True):
    model.eval()
    loss_sum, samples = 0.0, 0
    all_preds_raw, all_targets_raw = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val", leave=False):
            img_l = batch['image_left'].to(device)
            img_r = batch['image_right'].to(device)
            t_reg = batch['targets_reg'].to(device)
            t_cls = batch['targets_cls'].to(device)
            bs = img_l.size(0)

            if use_tta:
                # TTA: Standard + Horizontal flip
                reg1, cls1 = model(img_l, img_r)
                img_l_flip = torch.flip(img_l, [3])
                img_r_flip = torch.flip(img_r, [3])
                reg2, cls2 = model(img_r_flip, img_l_flip)
                reg_preds = [(r1 + r2) * 0.5 for r1, r2 in zip(reg1, reg2)]
                cls_preds = [(c1 + c2) * 0.5 for c1, c2 in zip(cls1, cls2)]
            else:
                reg_preds, cls_preds = model(img_l, img_r)

            loss, _, _ = criterion(reg_preds, cls_preds, t_reg, t_cls)
            loss_sum += loss.item() * bs
            samples += bs

            preds_matrix = torch.cat(reg_preds, dim=1).cpu().numpy()
            all_preds_raw.append(preds_matrix)
            all_targets_raw.append(t_reg.cpu().numpy())

    preds_raw = np.concatenate(all_preds_raw, axis=0)
    targets_true = np.concatenate(all_targets_raw, axis=0)

    states = val_df['State'].tolist() if 'State' in val_df.columns else None
    preds_post = apply_soft_blend_postprocess(preds_raw, states=states)

    r2_raw, per_target_raw = calculate_competition_r2(targets_true, preds_raw)
    r2_post, per_target_post = calculate_competition_r2(targets_true, preds_post)
    avg_loss = loss_sum / max(1, samples)

    return avg_loss, r2_raw, r2_post, per_target_post, preds_raw, preds_post


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream DINO Training")
    parser.add_argument('--data_path', type=str, default='train_converted.csv', help='Path to processed train CSV')
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2', help='Vision Transformer backbone')
    parser.add_argument('--img_size', type=int, default=512, help='Input resolution for each sub-image (512 or 1024)')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps (effective BS = batch_size * grad_accum)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate for heads')
    parser.add_argument('--stage1_epochs', type=int, default=8, help='Stage 1: Frozen backbone warm-up epochs')
    parser.add_argument('--stage2_epochs', type=int, default=26, help='Stage 2: Full model fine-tuning epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save checkpoints and OOF predictions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Enable TTA during validation')
    return parser.parse_args()


def run_training():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("[INIT] CSIRO IMAGE2BIOMASS: DUAL-STREAM DINO + INTERVAL CLASSIFICATION")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {args.backbone} | Resolution: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}) | Base LR: {args.lr}")
    print(f"Schedule: Stage 1 = {args.stage1_epochs} eps (Heads) | Stage 2 = {args.stage2_epochs} eps (Full FT)")
    print(f"Cross-Validation: {args.n_folds}-Fold Balanced Stratified Split (State + Biomass Quantiles)")
    print("=" * 70)

    # 1. Load Data
    data_path = args.data_path if os.path.exists(args.data_path) else 'train_converted.csv'
    if not os.path.exists(data_path):
        data_path = 'wide.csv'
    df = pd.read_csv(data_path)
    print(f"[DATA] Loaded {len(df)} samples from {data_path}")

    # 2. Balanced Stratification
    df = create_balanced_stratified_folds(df, n_splits=args.n_folds, seed=args.seed)
    print(f"[SPLIT] Generated {args.n_folds}-fold balanced stratification. Fold summary:")
    for f in range(args.n_folds):
        sub = df[df['fold'] == f]
        st_dict = dict(sub['State'].value_counts())
        print(f"   Fold {f+1}: N={len(sub)} | States={st_dict} | Total Biomass Mean={sub['Dry_Total_g'].mean():.1f}g")

    oof_preds_post = np.zeros((len(df), 5), dtype=np.float32)
    oof_preds_raw = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = df[TARGET_NAMES].values.astype(np.float32)

    fold_scores = []
    start_total_time = time.time()

    # 3. Iterate Folds
    for fold in range(args.n_folds):
        print(f"\n{'='*30} FOLD {fold + 1} / {args.n_folds} {'='*30}")
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        val_indices = df[df['fold'] == fold].index.values

        # Datasets & Loaders
        train_ds = DualStreamBiomassDataset(train_df, img_size=args.img_size, is_training=True)
        val_ds = DualStreamBiomassDataset(val_df, img_size=args.img_size, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Model & Loss
        model = DualStreamDINO(backbone_name=args.backbone, pretrained=True).to(device)
        criterion = WeightedBiomassLoss(cls_weight=0.3).to(device)
        scaler = torch.amp.GradScaler('cuda')

        best_fold_r2 = -float('inf')
        best_preds_post = None
        best_preds_raw = None
        ckpt_path = os.path.join(args.output_dir, f"best_model_fold{fold + 1}.pt")

        # ------------------------------------------------------------------
        # STAGE 1: Warm-up Heads (Backbone FROZEN)
        # ------------------------------------------------------------------
        print(f"\n--- [Fold {fold+1}] STAGE 1: Warm-up Heads ({args.stage1_epochs} epochs | Backbone FROZEN) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs, eta_min=1e-5)

        for ep in range(1, args.stage1_epochs + 1):
            tr_loss, tr_reg, tr_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps=args.grad_accum
            )
            scheduler.step()
            va_loss, r2_raw, r2_post, per_target, p_raw, p_post = validate(
                model, val_loader, criterion, device, val_df, use_tta=args.use_tta
            )
            print(f"[S1 Ep {ep:02d}] Train: {tr_loss:.4f} (reg:{tr_reg:.3f}, cls:{tr_cls:.3f}) | "
                  f"Val: {va_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f}")

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                best_preds_post = p_post
                best_preds_raw = p_raw
                torch.save(model.state_dict(), ckpt_path)

        # ------------------------------------------------------------------
        # STAGE 2: Full End-to-End Fine-Tuning (Differential LR + Warmup)
        # ------------------------------------------------------------------
        print(f"\n--- [Fold {fold+1}] STAGE 2: Full Fine-Tuning ({args.stage2_epochs} epochs | Differential LR) ---")
        for p in model.backbone.parameters():
            p.requires_grad = True

        backbone_lr = args.lr * 0.1  # 3e-5 for backbone
        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': args.lr}
        ], weight_decay=0.01)

        warmup_epochs = 3
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.stage2_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

        for ep in range(1, args.stage2_epochs + 1):
            curr_ep = args.stage1_epochs + ep
            tr_loss, tr_reg, tr_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps=args.grad_accum
            )
            scheduler.step()
            va_loss, r2_raw, r2_post, per_target, p_raw, p_post = validate(
                model, val_loader, criterion, device, val_df, use_tta=args.use_tta
            )
            print(f"[S2 Ep {curr_ep:02d}] Train: {tr_loss:.4f} (reg:{tr_reg:.3f}, cls:{tr_cls:.3f}) | "
                  f"Val: {va_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f} "
                  f"| Total R2: {per_target[4]:.3f} GDM R2: {per_target[3]:.3f}")

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                best_preds_post = p_post
                best_preds_raw = p_raw
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [BEST] New Best Model for Fold {fold+1} Saved (R2 Post: {best_fold_r2:.4f})")

        oof_preds_post[val_indices] = best_preds_post
        oof_preds_raw[val_indices] = best_preds_raw
        fold_scores.append(best_fold_r2)
        print(f"[OK] Fold {fold+1} Complete. Best Post R2: {best_fold_r2:.4f}")

    # 4. Final Out-Of-Fold Evaluation
    overall_post_r2, per_target_post = calculate_competition_r2(oof_targets, oof_preds_post)
    overall_raw_r2, per_target_raw = calculate_competition_r2(oof_targets, oof_preds_raw)
    total_time_min = (time.time() - start_total_time) / 60.0

    print("\n" + "=" * 70)
    print("[RESULTS] FINAL OUT-OF-FOLD (OOF) COMPETITION RESULTS")
    print("=" * 70)
    print(f"[METRIC] OVERALL OOF COMPETITION R2 (Post-Processed): {overall_post_r2:.4f}")
    print(f"         Overall OOF Competition R2 (Raw):            {overall_raw_r2:.4f}")
    print(f"         Per-Fold Scores: {[round(s, 4) for s in fold_scores]}")
    print("         Per-Target Breakdown (Post-Processed):")
    for t_name, score, w in zip(TARGET_NAMES, per_target_post, OFFICIAL_WEIGHTS):
        print(f"           - {t_name:15s} (Weight: {w:.1f}): R2 = {score:.4f}")
    print(f"Total CV Training Time: {total_time_min:.1f} minutes")
    print("=" * 70)

    # 5. Save Out-Of-Fold Predictions CSV
    oof_df = df[['sample_id', 'State', 'Species', 'Sampling_Date'] + TARGET_NAMES].copy()
    for idx, t in enumerate(TARGET_NAMES):
        oof_df[f'pred_{t}'] = oof_preds_post[:, idx]
        oof_df[f'pred_raw_{t}'] = oof_preds_raw[:, idx]
    oof_path = os.path.join(args.output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[SAVED] Saved OOF predictions to {oof_path}")


if __name__ == '__main__':
    run_training()
