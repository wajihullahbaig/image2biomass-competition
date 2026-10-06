"""
CSIRO Image2Biomass: Dual-Stream DINOv3 + Interval Classification (train.py)

Recipe (1st-place solution, with our CV fixes):
- Panorama split into Left/Right 1:1 views -> shared DINOv3 backbone -> cross-view attention -> fusion MLP.
- 5 regression heads + 5 auxiliary 7-interval classification heads (UEPNet).
- Regression loss: epsilon-insensitive L1 (1st place's best single model) or SmoothL1.
- Two stages: frozen backbone (heads warm-up) -> full fine-tune with differential LR.
- Fixed epoch budget; the saved model is the SWA average of the last `swa_epochs` epochs
  (no best-epoch picking on ~70 validation images).
- CV: StratifiedGroupKFold grouped by Sampling_Date, stratified by State, real images only.
- Metric: official globally weighted R2 over all (image, target) pairs on raw grams.
- Post-processing (1st place: clover x0.8, dead fringe, mass blends) is reported side-by-side, never used for selection.
"""

import os
import sys
import time
import random
import argparse
import logging
from datetime import datetime

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
from torch.optim.swa_utils import AveragedModel
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import timm

# ==============================================================================
# Constants
# ==============================================================================
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Non-uniform 7-interval thresholds from the 1st-place solution (UEPNet formulation)
BORDERS_DICT = {
    'Dry_Green_g':  [1.6e-05, 13.4232, 27.0782, 45.5236, 79.834, 157.9836],
    'Dry_Dead_g':   [1.6e-05, 6.1407, 13.1192, 23.277, 38.8581, 83.8407],
    'Dry_Clover_g': [1.6e-05, 3.9, 10.5353, 20.6523, 37.5911, 71.7865],
    'GDM_g':        [1.6e-05, 16.5143, 30.507, 49.5585, 81.0, 157.9836],
    'Dry_Total_g':  [1.6e-05, 23.4907, 41.1, 61.1, 96.8288, 185.7],
}


def set_seed(seed=223):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Data & Cross-Validation
# ==============================================================================
def load_data(data_path, logger=None):
    """Loads wide.csv or long-format train.csv (pivoted). Generated/synthetic rows are dropped:
    they duplicate real images and leak across date-grouped folds."""
    df = pd.read_csv(data_path)
    if 'target_name' in df.columns:
        df['sample_id'] = df['sample_id'].astype(str).str.split('__').str[0]
        targets = df.pivot_table(index='sample_id', columns='target_name', values='target', aggfunc='max').reset_index()
        meta = df[['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species']].drop_duplicates('sample_id')
        df = meta.merge(targets, on='sample_id')
    if 'is_synthetic' in df.columns:
        n_syn = int(df['is_synthetic'].astype(bool).sum())
        df = df[~df['is_synthetic'].astype(bool)]
        if logger and n_syn:
            logger.info(f"  [DATA] Dropped {n_syn} synthetic rows (duplicated images -> fold leakage)")
    df[TARGET_NAMES] = df[TARGET_NAMES].fillna(0.0)
    return df.reset_index(drop=True)


def create_grouped_stratified_folds(df, n_splits=5, seed=223, group_col='Sampling_Date', strat_col='State'):
    """Folds grouped by Sampling_Date (test dates are unseen) and stratified by State."""
    df = df.copy()
    df['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df[strat_col], groups=df[group_col])):
        df.loc[val_idx, 'fold'] = fold_idx
    return df


def align_img_size_to_backbone(img_size, backbone_name):
    """Ensures input image size is divisible by the ViT patch size."""
    patch = 14 if 'patch14' in backbone_name else 16
    return int(round(img_size / patch)) * patch


def get_interval_labels(targets_np):
    """Discretizes continuous biomass values (grams) into 7 classes (0..6)."""
    labels = np.zeros_like(targets_np, dtype=np.int64)
    for col_idx, col_name in enumerate(TARGET_NAMES):
        labels[:, col_idx] = np.digitize(targets_np[:, col_idx], BORDERS_DICT[col_name])
    return labels


# ==============================================================================
# Augmentations (applied independently to each sub-image view)
# ==============================================================================
def apply_camera_scale_simulation(image_np, prob=0.2):
    """Simulates focal distance variation by downscaling and zero-padding (1st place)."""
    if random.random() < prob:
        h, w = image_np.shape[:2]
        bg = np.zeros_like(image_np)
        scale = random.uniform(0.85, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top, left = random.randint(0, h - new_h), random.randint(0, w - new_w)
        bg[top:top + new_h, left:left + new_w] = resized
        return bg
    return image_np


def permute_vertical_strips(image_np, n_strips=4, prob=0.5):
    """Shuffles vertical strips: biomass is additive so mass is conserved (3rd place)."""
    if random.random() < prob:
        strips = np.array_split(image_np, n_strips, axis=1)
        random.shuffle(strips)
        return np.concatenate(strips, axis=1)
    return image_np


def apply_clahe(image_np, prob=0.3):
    """CLAHE on the L channel (1st place: clip_limit=2.0, tile 8x8)."""
    if random.random() < prob:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image_np


def apply_gauss_noise(image_np, prob=0.3):
    """Additive Gaussian pixel noise, std 3-7 on the 0-255 scale (1st place GaussNoise)."""
    if random.random() < prob:
        noise = np.random.normal(0, random.uniform(3, 7), image_np.shape)
        return np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return image_np


class DualStreamBiomassDataset(Dataset):
    """Splits each 2:1 panorama into Left and Right 1:1 views for the shared backbone."""
    def __init__(self, df, img_size=512, is_training=True, img_root='.',
                 camera_scaling_prob=0.2, strip_shuffle_prob=0.5, view_swap_prob=0.5, grayscale_prob=0.2):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.is_training = is_training
        self.camera_scaling_prob = camera_scaling_prob
        self.strip_shuffle_prob = strip_shuffle_prob
        self.view_swap_prob = view_swap_prob

        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=grayscale_prob),
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

    def _augment_view(self, view):
        view = apply_camera_scale_simulation(view, self.camera_scaling_prob)
        view = permute_vertical_strips(view, 4, self.strip_shuffle_prob)
        view = apply_clahe(view)
        return apply_gauss_noise(view)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['image_path'])
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        mid_w = raw_rgb.shape[1] // 2
        left_np, right_np = raw_rgb[:, :mid_w].copy(), raw_rgb[:, mid_w:].copy()

        if self.is_training:
            if random.random() < self.view_swap_prob:
                left_np, right_np = right_np, left_np
            left_np, right_np = self._augment_view(left_np), self._augment_view(right_np)

        item = {
            'image_left': self.transform(Image.fromarray(left_np)),
            'image_right': self.transform(Image.fromarray(right_np)),
        }
        if self.has_targets:
            item['targets_reg'] = torch.tensor(self.targets_reg[idx], dtype=torch.float32)
            item['targets_cls'] = torch.tensor(self.targets_cls[idx], dtype=torch.long)
        return item


# ==============================================================================
# Model: Dual-Stream DINOv3 with Cross-View Attention
# ==============================================================================
class DualStreamBiomassModel(nn.Module):
    def __init__(self, backbone_name="vit_large_patch16_dinov3_qkvb", num_targets=5, num_intervals=7,
                 fusion_dim=384, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.backbone_dim = self.backbone.num_features

        num_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(self.backbone_dim, num_heads, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.backbone_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(fusion_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            ) for _ in range(num_targets)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_intervals),
            ) for _ in range(num_targets)
        ])

    def extract_features(self, x):
        feats = self.backbone(x)
        if feats.dim() == 3:
            return feats.mean(dim=1)
        if feats.dim() == 4:
            return feats.mean(dim=[2, 3])
        return feats

    def forward(self, img_left, img_right):
        tokens = torch.stack([self.extract_features(img_left), self.extract_features(img_right)], dim=1)
        attn_out, _ = self.cross_view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        fused = self.fusion_mlp(torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1))
        reg_preds = [F.softplus(head(fused)) for head in self.reg_heads]
        cls_preds = [head(fused) for head in self.cls_heads]
        return reg_preds, cls_preds


def save_checkpoint(model, path, backbone, img_size):
    """Stores fp16 weights plus the backbone/img_size needed to rebuild the model at inference."""
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({'state_dict': state, 'backbone': backbone, 'img_size': img_size}, path)


# ==============================================================================
# Losses
# ==============================================================================
class EpsilonInsensitiveLoss(nn.Module):
    """1st-place L1 with a label-dependent dead zone: eps=1 for y<=20 g, else 0.1*y capped at 5."""
    def __init__(self, eps_point=20.0, scale_ratio=0.1, max_eps=5.0):
        super().__init__()
        self.eps_point, self.scale_ratio, self.max_eps = eps_point, scale_ratio, max_eps

    def forward(self, pred, target):
        eps = torch.where(target <= self.eps_point, torch.ones_like(target), target * self.scale_ratio)
        eps = torch.clamp(eps, max=self.max_eps)
        return torch.relu(torch.abs(pred - target) - eps).mean()


class WeightedBiomassLoss(nn.Module):
    """Metric-weighted regression loss + auxiliary interval cross-entropy."""
    def __init__(self, reg_loss='eps', cls_weight=0.3):
        super().__init__()
        self.weights = OFFICIAL_WEIGHTS
        self.cls_weight = cls_weight
        self.criterion_reg = EpsilonInsensitiveLoss() if reg_loss == 'eps' else nn.SmoothL1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, preds_reg, preds_cls, targets_reg, targets_cls):
        loss_reg = sum(w * self.criterion_reg(p.squeeze(-1).float(), targets_reg[:, i])
                       for i, (w, p) in enumerate(zip(self.weights, preds_reg)))
        loss_cls = sum(w * self.criterion_cls(c.float(), targets_cls[:, i])
                       for i, (w, c) in enumerate(zip(self.weights, preds_cls)))
        return loss_reg + self.cls_weight * loss_cls, loss_reg, loss_cls


# ==============================================================================
# Metric & Post-Processing
# ==============================================================================
def calculate_competition_r2(y_true, y_pred, weights=OFFICIAL_WEIGHTS):
    """Official metric: one R2 over all (image, target) pairs on raw grams, each pair weighted by
    its target weight. Also returns plain per-target R2 for diagnostics."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 5)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 5)
    w = np.broadcast_to(np.asarray(weights, dtype=float), y_true.shape)
    y_bar = (w * y_true).sum() / w.sum()
    score = 1.0 - (w * (y_true - y_pred) ** 2).sum() / (w * (y_true - y_bar) ** 2).sum()
    per_target = 1.0 - ((y_true - y_pred) ** 2).sum(0) / ((y_true - y_true.mean(0)) ** 2).sum(0)
    return float(score), per_target.tolist()


def soft_physics_postprocess(preds_np, clover_scale=0.8, dead_upper_thresh=20.0, dead_upper_scale=1.1,
                             dead_lower_thresh=10.0, dead_lower_scale=0.9, gdm_weight=0.5, total_weight=0.5):
    """1st-place post-processing: clover x0.8, dead fringe expansion, GDM/Total mass-conservation blends."""
    preds = np.maximum(preds_np.copy(), 0.0)
    green, dead, gdm, total = preds[:, 0], preds[:, 1], preds[:, 3], preds[:, 4]
    clover = preds[:, 2] * clover_scale
    dead = np.where(dead > dead_upper_thresh, dead * dead_upper_scale,
           np.where(dead < dead_lower_thresh, dead * dead_lower_scale, dead))
    gdm_blended = gdm_weight * gdm + (1.0 - gdm_weight) * (green + clover)
    total_blended = total_weight * total + (1.0 - total_weight) * (green + clover + dead)
    return np.maximum(np.column_stack([green, dead, clover, gdm_blended, total_blended]), 0.0)


# ==============================================================================
# Train / Predict Loops
# ==============================================================================
@torch.no_grad()
def predict(model, loader, device, use_tta=True):
    """Raw regression predictions (N, 5). TTA = mirrored panorama: flip both views and swap them."""
    model.eval()
    preds = []
    for batch in loader:
        img_l, img_r = batch['image_left'].to(device), batch['image_right'].to(device)
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            reg, _ = model(img_l, img_r)
            if use_tta:
                reg_flip, _ = model(torch.flip(img_r, [3]), torch.flip(img_l, [3]))
                reg = [(a + b) * 0.5 for a, b in zip(reg, reg_flip)]
        preds.append(torch.cat(reg, dim=1).float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum, desc):
    model.train()
    sums, n = np.zeros(3), 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        img_l, img_r = batch['image_left'].to(device), batch['image_right'].to(device)
        t_reg, t_cls = batch['targets_reg'].to(device), batch['targets_cls'].to(device)
        with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
            r, c = model(img_l, img_r)
            loss, l_reg, l_cls = criterion(r, c, t_reg, t_cls)
        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        sums += np.array([loss.item(), l_reg.item(), l_cls.item()]) * len(img_l)
        n += len(img_l)
    return sums / n


def fit(model, train_loader, val_loader, val_targets, args, device, logger, tag):
    """Stage 1 (frozen backbone) -> Stage 2 (full fine-tune). Returns the model loaded with the
    SWA average of the last `args.swa_epochs` Stage-2 epochs. Validation is only logged."""
    criterion = WeightedBiomassLoss(reg_loss=args.reg_loss, cls_weight=0.3)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone')]

    def log_epoch(stage, epoch, total, losses, t0, swa_note=''):
        msg = f"[{tag} {stage} Ep {epoch:02d}/{total:02d}] Train: {losses[0]:.3f} (reg:{losses[1]:.2f}, cls:{losses[2]:.2f})"
        if val_loader is not None:
            raw = predict(model, val_loader, device, args.tta)
            msg += (f" | Val R2 raw: {calculate_competition_r2(val_targets, raw)[0]:.4f}"
                    f" | post: {calculate_competition_r2(val_targets, soft_physics_postprocess(raw))[0]:.4f}")
        logger.info(f"{msg} ({time.time() - t0:.0f}s){swa_note}")

    # Stage 1: heads warm-up with the backbone frozen
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = AdamW(head_params, lr=args.lr, weight_decay=0.05)
    for epoch in range(1, args.stage1_epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.grad_accum,
                                 f'{tag} S1 Ep {epoch}')
        log_epoch('S1', epoch, args.stage1_epochs, losses, t0)

    # Stage 2: full fine-tune, differential LR, cosine decay, SWA over the final epochs
    for p in model.backbone.parameters():
        p.requires_grad = True
    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * args.backbone_lr_factor},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=args.lr * 0.01)
    swa_model, swa_start = None, args.stage2_epochs - args.swa_epochs + 1
    for epoch in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.grad_accum,
                                 f'{tag} S2 Ep {epoch}')
        scheduler.step()
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model)
            swa_model.update_parameters(model)
        log_epoch('S2', epoch, args.stage2_epochs, losses, t0, '  [SWA]' if epoch >= swa_start else '')

    if swa_model is not None:
        model.load_state_dict(swa_model.module.state_dict())
        del swa_model
    model.backbone.set_grad_checkpointing(False)
    return model


# ==============================================================================
# Logging & CLI
# ==============================================================================
def setup_logging(output_dir="models", log_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    session_dir = os.path.join(log_dir, f"dual_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)

    logger = logging.getLogger("DualStreamTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for path in [os.path.join(session_dir, "session.log"), os.path.join(output_dir, "train.log")]:
        fh = logging.FileHandler(path, mode='w', encoding='utf-8')
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)
    return logger, session_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream DINOv3 Training")
    parser.add_argument('--data_path', type=str, default='wide.csv', help='wide.csv or long-format train.csv')
    parser.add_argument('--img_root', type=str, default='.', help='Directory that image_path entries are relative to')
    parser.add_argument('--backbone', type=str, default='vit_large_patch16_dinov3_qkvb')
    parser.add_argument('--img_size', type=int, default=512, help='Resolution of each 1:1 view')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4, help='Effective batch = batch_size * grad_accum')
    parser.add_argument('--grad_ckpt', action=argparse.BooleanOptionalAction, default=True,
                        help='Gradient checkpointing in Stage 2 (needed for ViT-L on 16 GB)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Heads learning rate')
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1, help='Backbone LR = lr * factor in Stage 2')
    parser.add_argument('--stage1_epochs', type=int, default=8)
    parser.add_argument('--stage2_epochs', type=int, default=25)
    parser.add_argument('--swa_epochs', type=int, default=5, help='Average weights of the last N Stage-2 epochs')
    parser.add_argument('--reg_loss', choices=['eps', 'smoothl1'], default='eps')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--start_fold', type=int, default=1, help='Resume: folds before this load existing checkpoints')
    parser.add_argument('--full_train', action='store_true', help='Train one model on all data (no CV) for submission')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=223)
    parser.add_argument('--tta', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


# ==============================================================================
# Main
# ==============================================================================
def run_training(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, session_dir = setup_logging(args.output_dir, args.log_dir)
    args.img_size = align_img_size_to_backbone(args.img_size, args.backbone)

    logger.info("=" * 70)
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Config: {vars(args)}")
    logger.info(f"Session logs: {session_dir}")

    df = load_data(args.data_path, logger)
    logger.info(f"Samples: {len(df)} | Dates: {df['Sampling_Date'].nunique()} | States: {df['State'].value_counts().to_dict()}")
    loader_kw = dict(num_workers=0)

    def make_model(pretrained=True):
        return DualStreamBiomassModel(backbone_name=args.backbone, pretrained=pretrained).to(device)

    if args.full_train:
        logger.info("=" * 70 + "\nFULL-DATA TRAINING (no validation)")
        train_loader = DataLoader(DualStreamBiomassDataset(df, args.img_size, True, args.img_root),
                                  batch_size=args.batch_size, shuffle=True, **loader_kw)
        model = fit(make_model(), train_loader, None, None, args, device, logger, 'Full')
        ckpt_path = os.path.join(args.output_dir, f"model_full_seed{args.seed}.pt")
        save_checkpoint(model, ckpt_path, args.backbone, args.img_size)
        logger.info(f"[SAVED] {ckpt_path}")
        return

    df = create_grouped_stratified_folds(df, n_splits=args.n_folds, seed=args.seed)
    for f in range(args.n_folds):
        v = df[df['fold'] == f]
        logger.info(f"  Fold {f + 1}: Val={len(v)} | Dates={v['Sampling_Date'].nunique()} | "
                    f"States={v['State'].value_counts().to_dict()} | Total mean={v['Dry_Total_g'].mean():.1f}g")

    targets = df[TARGET_NAMES].values.astype(np.float32)
    oof_raw = np.zeros_like(targets)
    fold_scores = []
    start_time = time.time()

    for fold in range(args.n_folds):
        fold_num = fold + 1
        val_mask = (df['fold'] == fold).values
        val_loader = DataLoader(DualStreamBiomassDataset(df[val_mask], args.img_size, False, args.img_root),
                                batch_size=args.batch_size, shuffle=False, **loader_kw)
        ckpt_path = os.path.join(args.output_dir, f"model_fold{fold_num}.pt")

        if fold_num < args.start_fold and os.path.exists(ckpt_path):
            logger.info(f"\n[RESUME] Fold {fold_num}: loading {ckpt_path}")
            model = make_model(pretrained=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)['state_dict'])
        else:
            logger.info(f"\n{'=' * 30} FOLD {fold_num} / {args.n_folds} {'=' * 30}")
            train_loader = DataLoader(DualStreamBiomassDataset(df[~val_mask], args.img_size, True, args.img_root),
                                      batch_size=args.batch_size, shuffle=True, **loader_kw)
            model = fit(make_model(), train_loader, val_loader, targets[val_mask], args, device, logger, f'F{fold_num}')
            save_checkpoint(model, ckpt_path, args.backbone, args.img_size)

        oof_raw[val_mask] = predict(model, val_loader, device, args.tta)
        r2_raw = calculate_competition_r2(targets[val_mask], oof_raw[val_mask])[0]
        r2_post = calculate_competition_r2(targets[val_mask], soft_physics_postprocess(oof_raw[val_mask]))[0]
        fold_scores.append(r2_raw)
        logger.info(f">>> Fold {fold_num} (SWA model): R2 raw {r2_raw:.4f} | post {r2_post:.4f}")
        del model
        torch.cuda.empty_cache()

    oof_post = soft_physics_postprocess(oof_raw)
    r2_raw, per_target_raw = calculate_competition_r2(targets, oof_raw)
    r2_post, per_target_post = calculate_competition_r2(targets, oof_post)
    logger.info("\n" + "=" * 70)
    logger.info(f"[METRIC] OOF R2 (official, raw preds):       {r2_raw:.4f}")
    logger.info(f"[METRIC] OOF R2 (official, post-processed):  {r2_post:.4f}")
    logger.info(f"         Per-fold (raw): {[round(s, 4) for s in fold_scores]}")
    for name, a, b in zip(TARGET_NAMES, per_target_raw, per_target_post):
        logger.info(f"         {name:13s} R2 raw {a:.4f} | post {b:.4f}")
    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")

    oof_df = df[['sample_id', 'State', 'Species', 'Sampling_Date', 'fold'] + TARGET_NAMES].copy()
    for i, t in enumerate(TARGET_NAMES):
        oof_df[f'pred_raw_{t}'] = oof_raw[:, i]
        oof_df[f'pred_{t}'] = oof_post[:, i]
    for path in [os.path.join(args.output_dir, "oof_predictions.csv"), os.path.join(session_dir, "oof_predictions.csv")]:
        oof_df.to_csv(path, index=False)
    logger.info(f"[SAVED] OOF predictions -> {args.output_dir} and {session_dir}")


if __name__ == '__main__':
    run_training(parse_args())
