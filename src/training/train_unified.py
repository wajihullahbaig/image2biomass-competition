# train_unified.py - Dual-Stream DINO Vision Model Training with Interval Classification
import os
import sys
import logging
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# Add training module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config.loader import cfg
from configs import config_str
from common import (
    WeightedBiomassLoss,
    soft_physics_postprocess,
    calculate_competition_r2,
    TARGET_ORDER,
    set_seed
)
from dataset import DualStreamBiomassDataset
from models import DualStreamBiomassModel
from log_and_plots import setup_logging


def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg, epoch, stage=1, logger=None):
    """
    Trains the DualStream model for one epoch using mixed precision.
    """
    model.train()
    total_loss_sum = 0.0
    reg_loss_sum = 0.0
    cls_loss_sum = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Stage {stage} Ep {epoch:02d}", leave=False)
    for batch in pbar:
        img_l = batch['image_left'].to(cfg.device)
        img_r = batch['image_right'].to(cfg.device)
        targets_reg = batch['targets'].to(cfg.device)
        targets_cls = batch['targets_cls'].to(cfg.device)

        batch_size = img_l.size(0)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            reg_preds, cls_preds = model(img_l, img_r)
            loss, loss_reg, loss_cls = criterion(reg_preds, cls_preds, targets_reg, targets_cls)

        if torch.isnan(loss):
            if logger:
                logger.warning(f"NaN loss encountered at epoch {epoch}! Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item() * batch_size
        reg_loss_sum += loss_reg.item() * batch_size
        cls_loss_sum += loss_cls.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'reg': f"{loss_reg.item():.4f}",
            'cls': f"{loss_cls.item():.4f}"
        })

    avg_loss = total_loss_sum / max(1, total_samples)
    avg_reg = reg_loss_sum / max(1, total_samples)
    avg_cls = cls_loss_sum / max(1, total_samples)
    return avg_loss, avg_reg, avg_cls


def validate(model, loader, criterion, cfg, use_tta=False):
    """
    Evaluates model on validation set. Returns competition metrics and loss.
    """
    model.eval()
    val_loss_sum = 0.0
    total_samples = 0
    
    all_targets_reg = []
    all_preds_reg_raw = []
    
    all_targets_cls = []
    all_preds_cls = []

    with torch.no_grad():
        for batch in loader:
            img_l = batch['image_left'].to(cfg.device)
            img_r = batch['image_right'].to(cfg.device)
            targets_reg = batch['targets'].to(cfg.device)
            targets_cls = batch['targets_cls'].to(cfg.device)
            batch_size = img_l.size(0)

            if use_tta:
                # TTA: Standard + Horizontal flip
                reg1, cls1 = model(img_l, img_r)
                # Flip views horizontally
                img_l_flip = torch.flip(img_l, [3])
                img_r_flip = torch.flip(img_r, [3])
                reg2, cls2 = model(img_r_flip, img_l_flip)

                # Average predictions
                reg_preds = [(r1 + r2) * 0.5 for r1, r2 in zip(reg1, reg2)]
                cls_preds = [(c1 + c2) * 0.5 for c1, c2 in zip(cls1, cls2)]
            else:
                reg_preds, cls_preds = model(img_l, img_r)

            loss, _, _ = criterion(reg_preds, cls_preds, targets_reg, targets_cls)
            val_loss_sum += loss.item() * batch_size
            total_samples += batch_size

            # Stack continuous predictions: [B, 5]
            preds_linear = torch.cat(reg_preds, dim=1).cpu().numpy()
            all_preds_reg_raw.append(preds_linear)
            all_targets_reg.append(targets_reg.cpu().numpy())

            # Stack classification predictions: list of [B, 7]
            cls_indices = torch.stack([torch.argmax(c, dim=1) for c in cls_preds], dim=1).cpu().numpy()
            all_preds_cls.append(cls_indices)
            all_targets_cls.append(targets_cls.cpu().numpy())

    y_true = np.concatenate(all_targets_reg, axis=0)
    y_pred_raw = np.concatenate(all_preds_reg_raw, axis=0)
    
    y_true_cls = np.concatenate(all_targets_cls, axis=0)
    y_pred_cls = np.concatenate(all_preds_cls, axis=0)
    
    # 1. Raw Competition R2
    weights = cfg.targets.official_weights
    r2_raw = calculate_competition_r2(y_true, y_pred_raw, weights)
    
    # 2. Soft Postprocessed Competition R2 (1st place soft blend)
    y_pred_post = soft_physics_postprocess(y_pred_raw)
    r2_post = calculate_competition_r2(y_true, y_pred_post, weights)
    
    # 3. Interval Classification Accuracy
    cls_acc = np.mean(y_true_cls == y_pred_cls)

    avg_loss = val_loss_sum / max(1, total_samples)
    return avg_loss, r2_raw, r2_post, cls_acc, y_pred_raw, y_pred_post


def run_training():
    # 1. Setup session and logging
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("logs", f"dual_stream_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    setup_logging(logger_name="System Logger", log_dir=session_dir, file_name_part=f"dual_stream_{session_id}")
    logger = logging.getLogger("System Logger")

    logger.info("=" * 60)
    logger.info("DUAL-STREAM DINO + INTERVAL CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(config_str())

    set_seed(cfg.hyperparameters.random_seed)

    # 2. Load dataset
    data_path = 'wide.csv'
    if not os.path.exists(data_path):
        logger.error(f"Cannot find {data_path} in current directory!")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")

    # Ensure required grouping columns exist
    if 'State_Sampling_Date' not in df.columns:
        df['State_Sampling_Date'] = df['State'].astype(str) + "_" + df['Sampling_Date'].astype(str)
    if 'State_Species' not in df.columns:
        df['State_Species'] = df['State'].astype(str) + "_" + df['Species'].astype(str)

    # 3. Cross-Validation Split: StratifiedGroupKFold on State_Sampling_Date
    n_folds = cfg.hyperparameters.n_folds
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.hyperparameters.random_seed)
    
    group_col = cfg.split.group_col or 'State_Sampling_Date'
    strat_col = cfg.split.group_stratification_col or 'State_Species'
    
    logger.info(f"Cross-Validation: {n_folds} folds grouped by '{group_col}', stratified by '{strat_col}'")
    
    oof_predictions_raw = np.zeros((len(df), 5), dtype=np.float32)
    oof_predictions_post = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = np.zeros((len(df), 5), dtype=np.float32)

    fold_scores = []
    fold_scores_post = []

    img_size = cfg.preprocessing.image_height
    batch_size = cfg.hyperparameters.batch_size
    base_lr = cfg.hyperparameters.learning_rate
    stage1_epochs = getattr(cfg.training, 'stage1_epochs', 8)
    stage2_epochs = getattr(cfg.training, 'stage2_epochs', 27)
    total_epochs = stage1_epochs + stage2_epochs

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df[strat_col], groups=df[group_col])):
        logger.info(f"\n{'='*25} FOLD {fold + 1} / {n_folds} {'='*25}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

        # Datasets & Loaders
        train_ds = DualStreamBiomassDataset(
            train_df, 
            img_size=img_size, 
            is_training=True, 
            camera_scaling_prob=getattr(cfg.augmentation, 'camera_scaling_prob', 0.2),
            strip_shuffle_prob=getattr(cfg.augmentation, 'strip_shuffle_prob', 0.5),
            view_swap_prob=getattr(cfg.augmentation, 'view_swap_prob', 0.5)
        )
        val_ds = DualStreamBiomassDataset(
            val_df, 
            img_size=img_size, 
            is_training=False
        )

        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        # Initialize Model
        model = DualStreamBiomassModel(
            backbone_name=cfg.hyperparameters.backbone,
            num_targets=5,
            num_intervals=cfg.loss.num_intervals,
            fusion_dim=cfg.training.fusion_dim,
            dropout=getattr(cfg.training, 'dropout', 0.3),
            pretrained=True
        ).to(cfg.device)

        criterion = WeightedBiomassLoss(
            loss_weights=cfg.targets.official_weights,
            cls_weight=cfg.loss.cls_weight
        )

        scaler = torch.amp.GradScaler('cuda')

        # ----------------------------------------------------------------------
        # STAGE 1: Freeze Backbone -> Train Cross-View Fusion & MLP Heads
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold+1}] STAGE 1: Training Heads ({stage1_epochs} epochs) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=base_lr, weight_decay=cfg.hyperparameters.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=base_lr * 0.1)

        best_fold_r2 = -float('inf')
        best_fold_model_path = os.path.join(session_dir, f"best_model_fold{fold+1}.pt")
        val_post_preds_best = None

        for epoch in range(1, stage1_epochs + 1):
            train_loss, train_reg, train_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, cfg, epoch, stage=1, logger=logger
            )
            scheduler.step()
            val_loss, r2_raw, r2_post, cls_acc, val_raw, val_post = validate(
                model, val_loader, criterion, cfg, use_tta=False
            )

            logger.info(
                f"[S1 Ep {epoch:02d}] Train: {train_loss:.4f} (reg:{train_reg:.4f}, cls:{train_cls:.4f}) | "
                f"Val: {val_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 SoftBlend: {r2_post:.4f} | Cls Acc: {cls_acc:.2%}"
            )

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                val_post_preds_best = val_post
                torch.save(model.state_dict(), best_fold_model_path)

        # ----------------------------------------------------------------------
        # STAGE 2: Unfreeze Backbone -> Full End-to-End Fine-Tuning
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold+1}] STAGE 2: Full Fine-Tuning ({stage2_epochs} epochs) ---")
        for p in model.backbone.parameters():
            p.requires_grad = True

        backbone_lr = base_lr * getattr(cfg.training, 'stage2_backbone_lr_factor', 0.1)
        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': base_lr}
        ], weight_decay=cfg.hyperparameters.weight_decay)

        scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=backbone_lr * 0.05)

        for epoch in range(1, stage2_epochs + 1):
            curr_epoch = stage1_epochs + epoch
            train_loss, train_reg, train_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, cfg, curr_epoch, stage=2, logger=logger
            )
            scheduler.step()
            val_loss, r2_raw, r2_post, cls_acc, val_raw, val_post = validate(
                model, val_loader, criterion, cfg, use_tta=cfg.training.use_tta
            )

            logger.info(
                f"[S2 Ep {curr_epoch:02d}] Train: {train_loss:.4f} (reg:{train_reg:.4f}, cls:{train_cls:.4f}) | "
                f"Val: {val_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 SoftBlend: {r2_post:.4f} | Cls Acc: {cls_acc:.2%}"
            )

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                val_post_preds_best = val_post
                torch.save(model.state_dict(), best_fold_model_path)
                logger.info(f"  ✓ New Best Model Saved for Fold {fold+1} (R2: {best_fold_r2:.4f})")

        # Save OOF for this fold
        oof_predictions_post[val_idx] = val_post_preds_best
        oof_targets[val_idx] = val_df[TARGET_ORDER].values

        fold_scores.append(best_fold_r2)
        logger.info(f"Fold {fold+1} Final Best R2: {best_fold_r2:.4f}")

    # 4. Overall Out-Of-Fold (OOF) Score
    overall_oof_r2 = calculate_competition_r2(oof_targets, oof_predictions_post, cfg.targets.official_weights)
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL OUT-OF-FOLD COMPETITION R2 SCORE: {overall_oof_r2:.4f}")
    logger.info(f"Per-fold scores: {[round(s, 4) for s in fold_scores]}")
    logger.info("=" * 60)

    # Save OOF predictions CSV
    oof_df = df[['sample_id', 'Species', 'State', 'Sampling_Date'] + TARGET_ORDER].copy()
    for idx, col in enumerate(TARGET_ORDER):
        oof_df[f'pred_{col}'] = oof_predictions_post[:, idx]
    oof_csv_path = os.path.join(session_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_csv_path}")


if __name__ == "__main__":
    run_training()