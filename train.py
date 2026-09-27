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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Cross-Validation: Host-Confirmed Anti-Temporal-Leakage Grouping (0.59825 Peak)
# ==============================================================================
def create_grouped_stratified_folds(df, n_splits=5, seed=223, group_col='Sampling_Date', strat_col='State'):
    """
    Creates 5 cross-validation folds grouped by Sampling_Date and stratified by State.
    
    Why this achieves highest Kaggle LB/PB generalization:
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
def apply_camera_scaling(image_np, prob=0.2):
    """Simulates focal distance variation by downscaling and zero-padding (1st/3rd place)."""
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


def apply_vertical_strip_shuffle(image_np, n_strips=4, prob=0.5):
    """
    Shuffles vertical slices of pasture quadrat (3rd-place solution).
    Biomass is strictly conserved while eliminating spatial position bias.
    """
    if random.random() < prob:
        strips = np.array_split(image_np, n_strips, axis=1)
        random.shuffle(strips)
        return np.concatenate(strips, axis=1)
    return image_np


class DualStreamBiomassDataset(Dataset):
    """
    Dual-Stream Dataset for Panoramic (2:1) Pasture Images.
    Splits wide images into Left and Right 1:1 views for shared ViT feature extraction.
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
                # 2nd-place: Lighting robustness
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # 3rd-place: Texture/morphology learning
                transforms.RandomGrayscale(p=0.20),
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

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = raw_rgb.shape
        mid_w = w // 2

        # Split 2000x1000 into Left and Right 1000x1000 views
        left_np = raw_rgb[:, :mid_w].copy()
        right_np = raw_rgb[:, mid_w:].copy()

        if self.is_training:
            # 1. View Swap (50% prob) - models seam continuity
            if random.random() < 0.5:
                left_np, right_np = right_np, left_np

            # 2. Camera Focal Scaling (20% prob)
            left_np = apply_camera_scaling(left_np, prob=0.2)
            right_np = apply_camera_scaling(right_np, prob=0.2)

            # 3. Vertical 4-Strip Permutation (50% prob)
            left_np = apply_vertical_strip_shuffle(left_np, n_strips=4, prob=0.5)
            right_np = apply_vertical_strip_shuffle(right_np, n_strips=4, prob=0.5)

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
    Dual-Stream Vision Transformer:
    1. Shared DINO ViT Backbone (DINOv3 / DINOv2).
    2. Multi-Head Cross-View Attention layer across Left and Right sub-quadrats.
    3. Fusion MLP projecting joint representations.
    4. 5 Independent Continuous Regression Heads (Softplus activation).
    5. 5 Auxiliary Interval Classification Heads (7 classes each).
    """
    def __init__(self, backbone_name="vit_base_patch16_dinov3_qkvb", fusion_dim=384, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.fusion_dim = fusion_dim
        self.num_targets = 5
        self.num_intervals = 7

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

        # Cross-View Multi-Head Attention
        n_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=self.backbone_dim,
            num_heads=n_heads,
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
        fused = torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)
        fused = self.fusion_mlp(fused)

        reg_preds = [F.softplus(head(fused)) for head in self.reg_heads]
        cls_preds = [head(fused) for head in self.cls_heads]

        return reg_preds, cls_preds


# ==============================================================================
# Dual-Objective Loss Function
# ==============================================================================
class WeightedBiomassLoss(nn.Module):
    """Combines SmoothL1 Continuous Regression with Auxiliary Interval Classification."""
    def __init__(self, official_weights=OFFICIAL_WEIGHTS, cls_weight=0.3):
        super().__init__()
        self.weights = official_weights
        self.cls_weight = cls_weight
        self.reg_loss = nn.SmoothL1Loss(beta=1.0)
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, reg_preds, cls_preds, targets_reg, targets_cls):
        total_reg_loss = 0.0
        total_cls_loss = 0.0

        for i in range(len(reg_preds)):
            r_pred = reg_preds[i].squeeze(-1)
            r_true = targets_reg[:, i]
            w = self.weights[i]

            loss_r = self.reg_loss(r_pred, r_true)
            total_reg_loss += w * loss_r

            c_pred = cls_preds[i]
            c_true = targets_cls[:, i]
            loss_c = self.cls_loss(c_pred, c_true)
            total_cls_loss += w * loss_c

        total_loss = total_reg_loss + (self.cls_weight * total_cls_loss)
        return total_loss, total_reg_loss, total_cls_loss


# ==============================================================================
# Evaluation Metric & Physical Post-Processing
# ==============================================================================
def calculate_competition_r2(y_true, y_pred, weights=OFFICIAL_WEIGHTS):
    """Calculates official weighted multi-target R2 metric."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

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
    Decoupled Post-Processing Soft Blending (Winning Formulations):
    1. Pure physics for clover (unscaled, no artificial 0.8 downscale penalty).
    2. Dead fringe expansion: >20 * 1.1, <10 * 0.9 (2nd/3rd place solution).
    3. Soft physical blending (1st place solution):
       - GDM = 0.5 * GDM_pred + 0.5 * (Green + Clover)
       - Total = 0.5 * Total_pred + 0.5 * (Green + Clover + Dead)
    4. Range clipping to pasture bounds (Clover <= 71.79, Dead <= 83.84, Green <= 157.98).
    5. WA zero-dead correction (if state metadata is available).
    6. Non-negativity clamp (val >= 0.0).
    """
    preds = np.maximum(np.asarray(preds_5, dtype=np.float32).copy(), 0.0)
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2]
    gdm = preds[:, 3]
    total = preds[:, 4]

    # Dead fringe adjustment (from 2nd & 3rd place solutions)
    dead = np.where(dead > 20.0, dead * 1.1,
           np.where(dead < 10.0, dead * 0.9, dead))

    # WA zero-dead physical correction
    if states is not None:
        for idx, st in enumerate(states):
            if str(st).strip() == 'WA':
                dead[idx] = 0.0

    # Range clipping to pasture bounds
    clover = np.clip(clover, 0.0, 71.79)
    dead = np.clip(dead, 0.0, 83.84)
    green = np.clip(green, 0.0, 157.98)

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

    n = max(1, samples)
    return loss_sum / n, reg_sum / n, cls_sum / n


def validate(model, loader, criterion, device, val_df, use_tta=True):
    model.eval()
    loss_sum, samples = 0.0, 0
    all_preds_raw = []
    all_targets_raw = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val", leave=False):
            img_l = batch['image_left'].to(device)
            img_r = batch['image_right'].to(device)
            t_reg = batch['targets_reg'].to(device)
            t_cls = batch['targets_cls'].to(device)
            bs = img_l.size(0)

            with torch.amp.autocast('cuda'):
                if use_tta:
                    # Standard view
                    reg1, cls1 = model(img_l, img_r)
                    # Mirrored view: horizontal flip & swap left/right
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

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Session log
    session_log_path = os.path.join(session_dir, "session.log")
    fh1 = logging.FileHandler(session_log_path, mode='w', encoding='utf-8')
    fh1.setLevel(logging.INFO)
    fh1.setFormatter(file_formatter)
    logger.addHandler(fh1)

    # Output log
    output_log_path = os.path.join(output_dir, "train.log")
    fh2 = logging.FileHandler(output_log_path, mode='w', encoding='utf-8')
    fh2.setLevel(logging.INFO)
    fh2.setFormatter(file_formatter)
    logger.addHandler(fh2)

    return logger, session_dir, session_log_path


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream DINO Training")
    parser.add_argument('--data_path', type=str, default='wide.csv', help='Path to full dataset CSV (wide.csv with 439 samples)')
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_dinov3_qkvb', help='Vision Transformer backbone (vit_base_patch16_dinov3_qkvb / vit_base_patch14_dinov2)')
    parser.add_argument('--img_size', type=int, default=512, help='Input resolution for each sub-image (512 for patch16, 518 for patch14)')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps (effective BS = batch_size * grad_accum)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate for heads')
    parser.add_argument('--stage1_epochs', type=int, default=8, help='Stage 1: Frozen backbone warm-up epochs')
    parser.add_argument('--stage2_epochs', type=int, default=27, help='Stage 2: Full model fine-tuning epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--start_fold', type=int, default=1, help='Starting fold index (1-based, e.g. 2 to resume from Fold 2)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save checkpoints and OOF predictions')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save timestamped session logs')
    parser.add_argument('--seed', type=int, default=223, help='Random seed (223 gives optimal balanced date splits)')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Enable TTA during validation')
    return parser.parse_args()


def run_training():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, session_dir, session_log_path = setup_logging(args.output_dir, args.log_dir)

    # Automatically align image size to ViT patch size (divisibility assertion)
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
    logger.info(f"  Backbone: {args.backbone} | Sub-Image Resolution: {args.img_size}x{args.img_size}")
    logger.info(f"  Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}) | Effective Batch Size: {args.batch_size * args.grad_accum}")
    logger.info(f"  Learning Rate: {args.lr} (Differential factor: 0.1x for backbone in Stage 2)")
    logger.info(f"  Schedule: Stage 1 = {args.stage1_epochs} eps (Heads) | Stage 2 = {args.stage2_epochs} eps (Full FT)")
    logger.info(f"  Random Seed: {args.seed} | Validation TTA: {args.use_tta}")
    logger.info(f"  Logging to: {session_log_path} and {os.path.join(args.output_dir, 'train.log')}")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 2: Data Loading & Verification
    # --------------------------------------------------------------------------
    data_path = args.data_path if os.path.exists(args.data_path) else 'wide.csv'
    if not os.path.exists(data_path):
        data_path = 'train_converted.csv'
    df = pd.read_csv(data_path)

    logger.info("\n" + "=" * 70)
    logger.info("[STEP 2/7] DATA LOADING & DATASET VERIFICATION")
    logger.info(f"  Dataset Source: {data_path}")
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
    logger.info(f"  Splitting 439 samples into {args.n_folds} folds grouped by 'Sampling_Date', stratified by 'State' (seed={args.seed})")
    df = create_grouped_stratified_folds(df, n_splits=args.n_folds, seed=args.seed)

    for f in range(args.n_folds):
        sub_val = df[df['fold'] == f]
        n_train = len(df) - len(sub_val)
        st_dict = dict(sub_val['State'].value_counts())
        n_dates = sub_val['Sampling_Date'].nunique()
        logger.info(f"  Fold {f+1}: Train={n_train} | Val={len(sub_val)} (Dates={n_dates} | States={st_dict} | Total Mean={sub_val['Dry_Total_g'].mean():.1f}g)")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 4: Augmentation & Solution Items Logging
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[STEP 4/7] MULTI-TIER AUGMENTATIONS (1st + 2nd + 3rd PLACE HERITAGE)")
    logger.info("  1st-Place: 1000x1000 Centerline 1:1 view split, H/V flips, Cross-View Attention")
    logger.info("  2nd-Place: ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) for sunlight variation")
    logger.info("  3rd-Place: Vertical 4-strip permutation (p=0.5), RandomGrayscale (p=0.2), View Swap (p=0.5), Camera Scaling (p=0.2)")
    logger.info("=" * 70)

    oof_preds_post = np.zeros((len(df), 5), dtype=np.float32)
    oof_preds_raw = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = df[TARGET_NAMES].values.astype(np.float32)

    fold_scores = []
    start_total_time = time.time()

    # --------------------------------------------------------------------------
    # STEP 5 & 6: Iterate Folds & Train
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
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                cached_model = DualStreamDINO(backbone_name=args.backbone, pretrained=False).to(device)
                cached_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                criterion = WeightedBiomassLoss().to(device)
                va_loss, r2_raw, r2_post, per_target, p_raw, p_post = validate(
                    cached_model, val_loader, criterion, device, val_df, use_tta=args.use_tta
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
        logger.info(f"[DATA] Train samples: {len(train_df)} | Val samples: {len(val_df)}")

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

        # ----------------------------------------------------------------------
        # STAGE 1: Warm-up Heads (Backbone FROZEN)
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold_num}] STAGE 1: Training Heads ({args.stage1_epochs} epochs | Backbone FROZEN) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=args.lr, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs, eta_min=args.lr * 0.1)

        for ep in range(1, args.stage1_epochs + 1):
            tr_loss, tr_reg, tr_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps=args.grad_accum
            )
            scheduler.step()
            va_loss, r2_raw, r2_post, per_target, p_raw, p_post = validate(
                model, val_loader, criterion, device, val_df, use_tta=args.use_tta
            )
            logger.info(f"[S1 Ep {ep:02d}/{args.stage1_epochs:02d}] Train: {tr_loss:.4f} (reg:{tr_reg:.3f}, cls:{tr_cls:.3f}) | "
                        f"Val: {va_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f}")

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                best_preds_post = p_post
                best_preds_raw = p_raw
                torch.save(model.state_dict(), ckpt_path)

        # ----------------------------------------------------------------------
        # STAGE 2: Full End-to-End Fine-Tuning (Differential LR + Warmup)
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold_num}] STAGE 2: Full Fine-Tuning ({args.stage2_epochs} epochs | Differential LR) ---")
        for p in model.backbone.parameters():
            p.requires_grad = True

        backbone_lr = args.lr * 0.1  # 3e-5 for backbone
        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': args.lr}
        ], weight_decay=0.05)

        warmup_epochs = 3
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.stage2_epochs - warmup_epochs), eta_min=backbone_lr * 0.05)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

        for ep in range(1, args.stage2_epochs + 1):
            curr_ep = args.stage1_epochs + ep
            total_eps = args.stage1_epochs + args.stage2_epochs
            tr_loss, tr_reg, tr_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps=args.grad_accum
            )
            scheduler.step()
            va_loss, r2_raw, r2_post, per_target, p_raw, p_post = validate(
                model, val_loader, criterion, device, val_df, use_tta=args.use_tta
            )
            logger.info(f"[S2 Ep {curr_ep:02d}/{total_eps:02d}] Train: {tr_loss:.4f} (reg:{tr_reg:.3f}, cls:{tr_cls:.3f}) | "
                        f"Val: {va_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f} "
                        f"| Total R2: {per_target[4]:.3f} GDM R2: {per_target[3]:.3f}")

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                best_preds_post = p_post
                best_preds_raw = p_raw
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"  ✓ [SAVED] New Best Model Saved for Fold {fold_num} (R2 Post: {best_fold_r2:.4f}) -> {ckpt_path}")

        oof_preds_post[val_indices] = best_preds_post
        oof_preds_raw[val_indices] = best_preds_raw
        fold_scores.append(best_fold_r2)
        logger.info(f"[OK] Fold {fold_num} Complete. Final Best Post R2: {best_fold_r2:.4f}")

    # --------------------------------------------------------------------------
    # STEP 7: Final Out-Of-Fold Evaluation & Export
    # --------------------------------------------------------------------------
    overall_post_r2, per_target_post = calculate_competition_r2(oof_targets, oof_preds_post)
    overall_raw_r2, per_target_raw = calculate_competition_r2(oof_targets, oof_preds_raw)
    total_time_min = (time.time() - start_total_time) / 60.0

    logger.info("\n" + "=" * 70)
    logger.info("[STEP 7/7] FINAL OUT-OF-FOLD (OOF) COMPETITION RESULTS")
    logger.info("=" * 70)
    logger.info(f"[METRIC] OVERALL OOF COMPETITION R2 (Post-Processed): {overall_post_r2:.4f}")
    logger.info(f"         Overall OOF Competition R2 (Raw):            {overall_raw_r2:.4f}")
    logger.info(f"         Per-Fold Scores: {[round(s, 4) for s in fold_scores]}")
    logger.info("         Per-Target Breakdown (Post-Processed):")
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
