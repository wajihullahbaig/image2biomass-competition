# common.py - Core Constants, Losses, Metrics, and Post-Processing
import os
import random
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# Constants & Targets
# ==============================================================================
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]

# UEPNet (CVPR 2021) 7-interval partition thresholds for each target
BORDERS_DICT = {
    'Dry_Green_g':  [1.6e-05, 13.4232, 27.0782, 45.5236, 79.834, 157.9836],
    'Dry_Dead_g':   [1.6e-05, 6.1407, 13.1192, 23.277, 38.8581, 83.8407],
    'Dry_Clover_g': [1.6e-05, 3.9, 10.5353, 20.6523, 37.5911, 71.7865],
    'GDM_g':        [1.6e-05, 16.5143, 30.507, 49.5585, 81.0, 157.9836],
    'Dry_Total_g':  [1.6e-05, 23.4907, 41.1, 61.1, 96.8288, 185.7],
}


def set_seed(seed=42):
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Label Binning (UEPNet Interval Partitioning)
# ==============================================================================
def get_interval_labels(targets_np, target_cols=TARGET_ORDER):
    """
    Discretizes continuous biomass values (grams) into 7 intervals (classes 0..6)
    using the non-uniform borders derived from UEPNet crowd counting.
    """
    labels_cls = np.zeros_like(targets_np, dtype=np.int64)
    for col_idx, col_name in enumerate(target_cols):
        borders = BORDERS_DICT.get(col_name)
        if borders is not None:
            labels_cls[:, col_idx] = np.digitize(targets_np[:, col_idx], borders)
        else:
            labels_cls[:, col_idx] = np.clip(
                np.digitize(targets_np[:, col_idx], [0, 5, 15, 30, 60, 120]), 0, 6
            )
    return labels_cls


# ==============================================================================
# Dual-Objective Loss (SmoothL1 + Cross-Entropy)
# ==============================================================================
class WeightedBiomassLoss(nn.Module):
    """
    Dual-Objective Biomass Loss:
    1. SmoothL1 Loss for continuous regression on raw continuous biomass (grams)
    2. CrossEntropy Loss for auxiliary interval classification
    3. Scaled according to official competition weights [0.1, 0.1, 0.1, 0.2, 0.5]
    """
    def __init__(self, loss_weights=None, cls_weight=0.3):
        super().__init__()
        self.criterion_reg = nn.SmoothL1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight
        self.weights = loss_weights if loss_weights is not None else OFFICIAL_WEIGHTS

    def forward(self, predictions_reg, predictions_cls, targets_reg, targets_cls=None):
        device = targets_reg.device
        w = torch.tensor(self.weights, device=device, dtype=torch.float32)
        
        # 1. Continuous Regression Loss
        loss_reg_total = torch.tensor(0.0, device=device)
        for i in range(5):
            pred_i = predictions_reg[i].squeeze(-1) if isinstance(predictions_reg, list) else predictions_reg[:, i]
            true_i = targets_reg[:, i]
            loss_i = self.criterion_reg(pred_i, true_i)
            loss_reg_total += w[i] * loss_i

        # 2. Auxiliary Interval Classification Loss
        loss_cls_total = torch.tensor(0.0, device=device)
        if predictions_cls is not None and targets_cls is not None:
            for i in range(5):
                pred_cls_i = predictions_cls[i]
                true_cls_i = targets_cls[:, i].long()
                loss_cls_i = self.criterion_cls(pred_cls_i, true_cls_i)
                loss_cls_total += w[i] * loss_cls_i

        total_loss = loss_reg_total + (self.cls_weight * loss_cls_total)
        return total_loss, loss_reg_total, loss_cls_total


# ==============================================================================
# Competition Metric: Weighted R^2 in Log-Space
# ==============================================================================
def calculate_competition_r2(y_true, y_pred, weights=OFFICIAL_WEIGHTS):
    """
    Official Competition Metric: Weighted sum of individual log-space R^2 scores.
    Formula: FinalScore = sum(w_i * R2_i) where R2_i is computed on log(1 + y).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    w = np.array(weights, dtype=float)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 5)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 5)
        
    yt = np.log1p(np.maximum(0, y_true))
    yp = np.log1p(np.maximum(0, y_pred))
    
    r2_scores = []
    for i in range(5):
        target_true = yt[:, i]
        target_pred = yp[:, i]
        
        ss_res = np.sum((target_true - target_pred) ** 2)
        ss_tot = np.sum((target_true - np.mean(target_true)) ** 2)
        
        if ss_tot == 0:
            score = 1.0 if ss_res == 0 else 0.0
        else:
            score = 1.0 - (ss_res / ss_tot)
        r2_scores.append(score)
        
    return float(np.sum(w * np.array(r2_scores)))


# ==============================================================================
# Soft Physical Post-Processing & Calibration
# ==============================================================================
def soft_physics_postprocess(preds_np):
    """
    Decoupled Post-Processing Soft Blending:
    Aligns raw predictions with physical identities without imposing rigid
    constraints during backpropagation.
    """
    preds = np.maximum(preds_np.copy(), 0.0)
    
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2] * 0.8  # Correction for clover overestimation
    gdm = preds[:, 3]
    total = preds[:, 4]
    
    # Dead biomass piecewise calibration
    dead = np.where(dead > 20.0, dead * 1.1, np.where(dead < 10.0, dead * 0.9, dead))
    
    # Soft Blending of composite quantities
    derived_gdm = green + clover
    gdm_blended = 0.5 * gdm + 0.5 * derived_gdm
    
    derived_total = green + clover + dead
    total_blended = 0.5 * total + 0.5 * derived_total
    
    blended = np.column_stack([green, dead, clover, gdm_blended, total_blended])
    return np.maximum(blended, 0.0)