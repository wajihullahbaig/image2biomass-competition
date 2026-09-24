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
    3. Supports both 3 base targets [Green, Dead, Clover] and full 5 targets.
    """
    def __init__(self, loss_weights=None, cls_weight=0.2, num_targets=3):
        super().__init__()
        self.criterion_reg = nn.SmoothL1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight
        self.num_targets = num_targets
        self.weights = loss_weights

    def forward(self, predictions_reg, predictions_cls, targets_reg, targets_cls=None):
        device = targets_reg.device
        n_t = len(predictions_reg) if isinstance(predictions_reg, list) else predictions_reg.shape[1]
        
        if self.weights is not None and len(self.weights) >= n_t:
            w = torch.tensor(self.weights[:n_t], device=device, dtype=torch.float32)
            w = w / w.sum()  # normalize
        else:
            w = torch.ones(n_t, device=device, dtype=torch.float32) / n_t
        
        # 1. Continuous Regression Loss
        loss_reg_total = torch.tensor(0.0, device=device)
        for i in range(n_t):
            pred_i = predictions_reg[i].squeeze(-1) if isinstance(predictions_reg, list) else predictions_reg[:, i]
            true_i = targets_reg[:, i]
            loss_i = self.criterion_reg(pred_i, true_i)
            loss_reg_total += w[i] * loss_i

        # 2. Auxiliary Interval Classification Loss
        loss_cls_total = torch.tensor(0.0, device=device)
        if predictions_cls is not None and targets_cls is not None:
            for i in range(n_t):
                pred_cls_i = predictions_cls[i]
                true_cls_i = targets_cls[:, i].long()
                loss_cls_i = self.criterion_cls(pred_cls_i, true_cls_i)
                loss_cls_total += w[i] * loss_cls_i

        total_loss = loss_reg_total + (self.cls_weight * loss_cls_total)
        return total_loss, loss_reg_total, loss_cls_total


# ==============================================================================
# Target Derivation (3 Base -> 5 Full Competition Targets)
# ==============================================================================
def derive_5_targets(preds_3):
    """
    Derives GDM = Green + Clover and Total = Green + Dead + Clover from 3 base targets.
    Input: [N, 3] corresponding to [Green, Dead, Clover].
    Output: [N, 5] corresponding to [Green, Dead, Clover, GDM, Total].
    """
    if isinstance(preds_3, torch.Tensor):
        green = preds_3[:, 0:1]
        dead = preds_3[:, 1:2]
        clover = preds_3[:, 2:3]
        gdm = green + clover
        total = green + dead + clover
        return torch.cat([green, dead, clover, gdm, total], dim=-1)
    else:
        preds_np = np.asarray(preds_3, dtype=np.float32)
        if preds_np.ndim == 1:
            preds_np = preds_np.reshape(1, -1)
        green = preds_np[:, 0:1]
        dead = preds_np[:, 1:2]
        clover = preds_np[:, 2:3]
        gdm = green + clover
        total = green + dead + clover
        return np.concatenate([green, dead, clover, gdm, total], axis=-1)


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
# Post-Processing (Physical Consistency: WA Zero-Dead + Boundary Clipping + Identities)
# ==============================================================================
def apply_2nd_place_postprocess(preds_5, states=None):
    """
    Physical Post-Processing:
    1. WA Dead Zeroing: Ground truth in Western Australia is strictly 0.0g dead biomass.
    2. Target Range Clipping to Training Bounds:
       - Clover in [0, 71.7865]
       - Dead in [0, 83.8407]
       - Green in [0, 157.9836]
    3. Recompute Physical Identities:
       - GDM = Green + Clover
       - Total = Green + Dead + Clover
    Note: Artificial state scalar multipliers (e.g. WA clover *= 0.80) are omitted
    as empirical validation proved they severely under-predict large clover plots.
    """
    preds = np.maximum(np.asarray(preds_5, dtype=np.float32).copy(), 0.0)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
        
    green = preds[:, 0].copy()
    dead = preds[:, 1].copy()
    clover = preds[:, 2].copy()

    # 1. State-specific ground-truth physical correction
    if states is not None:
        for idx, st in enumerate(states):
            st_str = str(st).strip()
            if st_str == 'WA':
                dead[idx] = 0.0  # WA pasture thatch is strictly 0.0g

    # 2. Clipping to training set boundaries
    clover = np.clip(clover, 0.0, 71.7865)
    dead = np.clip(dead, 0.0, 83.8407)
    green = np.clip(green, 0.0, 157.9836)

    # 3. Recompute physical composite identities
    gdm = green + clover
    total = green + dead + clover

    return np.column_stack([green, dead, clover, gdm, total])


# ==============================================================================
# Soft Physical Post-Processing & Calibration (1st Place Soft Blend)
# ==============================================================================
def soft_physics_postprocess(preds_np, 
                             clover_scale=0.8, 
                             dead_upper_thresh=20.0, 
                             dead_upper_scale=1.1, 
                             dead_lower_thresh=10.0, 
                             dead_lower_scale=0.9,
                             gdm_weight=0.5,
                             total_weight=0.5):
    preds = np.maximum(preds_np.copy(), 0.0)
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2] * clover_scale
    gdm = preds[:, 3]
    total = preds[:, 4]
    
    dead = np.where(dead > dead_upper_thresh, dead * dead_upper_scale,
           np.where(dead < dead_lower_thresh, dead * dead_lower_scale, dead))
    
    derived_gdm = green + clover
    gdm_blended = gdm_weight * gdm + (1.0 - gdm_weight) * derived_gdm
    
    derived_total = green + clover + dead
    total_blended = total_weight * total + (1.0 - total_weight) * derived_total
    
    blended = np.column_stack([green, dead, clover, gdm_blended, total_blended])
    return np.maximum(blended, 0.0)