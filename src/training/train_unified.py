# train_unified.py - Fast 2-Stage Training with Stratified Splits & 2nd Place Post-Processing
import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import StratifiedGroupKFold

# Ensure both workspace root, src, and src/training are on sys.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)
parent_dir = os.path.dirname(cur_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from config.loader import cfg
except ImportError:
    from src.training.config.loader import cfg

from configs import config_str
from common import (
    set_seed, 
    TARGET_ORDER, 
    OFFICIAL_WEIGHTS, 
    WeightedBiomassLoss, 
    derive_5_targets,
    apply_2nd_place_postprocess,
    calculate_competition_r2
)
from dataset import DualStreamBiomassDataset
from models import DualStreamBiomassModel
from log_and_plots import setup_logging
from visualize_batch import save_sample_batch


def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg, epoch, stage=1, logger=None):
    """
    Trains model for one epoch.
    """
    model.train()
    total_loss_sum = 0.0
    reg_loss_sum = 0.0
    cls_loss_sum = 0.0
    total_samples = 0

    grad_accum_steps = getattr(cfg.hyperparameters, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Stage {stage} Ep {epoch:02d}", leave=False)
    for step, batch in pbar:
        img_l = batch['image_left'].to(cfg.device)
        img_r = batch['image_right'].to(cfg.device)
        targets_reg = batch['targets'].to(cfg.device)
        targets_cls = batch['targets_cls'].to(cfg.device)

        batch_size = img_l.size(0)

        with torch.amp.autocast('cuda'):
            reg_preds, cls_preds = model(img_l, img_r)
            loss, loss_reg, loss_cls = criterion(reg_preds, cls_preds, targets_reg, targets_cls)
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            if logger:
                logger.warning(f"NaN loss encountered at epoch {epoch}! Skipping batch.")
            continue

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss_sum += loss.item() * grad_accum_steps * batch_size
        reg_loss_sum += loss_reg.item() * batch_size
        cls_loss_sum += loss_cls.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            'loss': f"{loss.item() * grad_accum_steps:.4f}",
            'reg': f"{loss_reg.item():.4f}",
            'cls': f"{loss_cls.item():.4f}"
        })

    avg_loss = total_loss_sum / max(1, total_samples)
    avg_reg = reg_loss_sum / max(1, total_samples)
    avg_cls = cls_loss_sum / max(1, total_samples)
    return avg_loss, avg_reg, avg_cls


def validate(model, loader, criterion, cfg, val_df, use_tta=False):
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

            # Stack continuous predictions
            preds_linear = torch.cat(reg_preds, dim=1).cpu().numpy()
            all_preds_reg_raw.append(preds_linear)
            all_targets_reg.append(targets_reg.cpu().numpy())

            # Stack classification predictions
            cls_indices = torch.stack([torch.argmax(c, dim=1) for c in cls_preds], dim=1).cpu().numpy()
            all_preds_cls.append(cls_indices)
            all_targets_cls.append(targets_cls.cpu().numpy())

    y_pred_base = np.concatenate(all_preds_reg_raw, axis=0)
    
    y_true_cls = np.concatenate(all_targets_cls, axis=0)
    y_pred_cls = np.concatenate(all_preds_cls, axis=0)

    # Convert 3 base targets -> 5 competition targets (GDM = Green + Clover, Total = Green + Dead + Clover)
    if y_pred_base.shape[1] == 3:
        y_pred_5 = derive_5_targets(y_pred_base)
    else:
        y_pred_5 = y_pred_base
        
    y_true_5 = val_df[TARGET_ORDER].values.astype(np.float32)

    # 1. Raw Competition R2 (Physical derivation)
    weights = cfg.targets.official_weights
    r2_raw = calculate_competition_r2(y_true_5, y_pred_5, weights)
    
    # 2. 2nd-Place Solution Postprocessing (WA Dead zeroing + State Multipliers + Boundary Clipping)
    states = val_df['State'].values if 'State' in val_df.columns else None
    y_pred_post = apply_2nd_place_postprocess(y_pred_5, states=states)
    r2_post = calculate_competition_r2(y_true_5, y_pred_post, weights)
    
    # 3. Interval Classification Accuracy
    cls_acc = np.mean(y_true_cls == y_pred_cls)

    avg_loss = val_loss_sum / max(1, total_samples)
    return avg_loss, r2_raw, r2_post, cls_acc, y_pred_5, y_pred_post


def run_training():
    # 1. Logging and Session Initialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"dual_stream_{timestamp}"
    session_dir = os.path.join("logs", session_id)
    os.makedirs(session_dir, exist_ok=True)

    setup_logging(logger_name="System Logger", log_dir=session_dir, file_name_part=f"dual_stream_{session_id}")
    logger = logging.getLogger("System Logger")

    logger.info("=" * 60)
    logger.info("FAST DUAL-STREAM DINO + 2ND PLACE POST-PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(config_str())

    set_seed(cfg.hyperparameters.random_seed)

    # 2. Load dataset (Prioritize train_converted.csv with clean 357 samples)
    data_path = 'train_converted.csv'
    if not os.path.exists(data_path):
        data_path = 'wide.csv'
    if not os.path.exists(data_path):
        logger.error(f"Cannot find dataset at train_converted.csv or wide.csv!")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")

    # 3. Anti-Leakage Cross-Validation Split: StratifiedGroupKFold on Sampling_Date & State
    n_folds = cfg.hyperparameters.n_folds
    group_col = getattr(cfg.split, 'group_col', 'Sampling_Date')
    strat_col = getattr(cfg.split, 'group_stratification_col', 'State')
    seed = cfg.hyperparameters.random_seed

    logger.info(f"Generating {n_folds}-fold StratifiedGroupKFold split grouped by '{group_col}', stratified by '{strat_col}' (seed={seed})")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_iter = list(sgkf.split(df, y=df[strat_col], groups=df[group_col]))

    for f_idx, (tr_idx, val_idx) in enumerate(folds_iter):
        tr_dates = set(df.iloc[tr_idx][group_col])
        va_dates = set(df.iloc[val_idx][group_col])
        overlap = tr_dates.intersection(va_dates)
        logger.info(f"  Fold {f_idx + 1}: {len(val_idx)} val samples | {len(va_dates)} dates | Overlapping dates with train: {len(overlap)}")

    oof_predictions_raw = np.zeros((len(df), 5), dtype=np.float32)
    oof_predictions_post = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = np.zeros((len(df), 5), dtype=np.float32)

    fold_scores = []
    fold_scores_raw = []

    img_size = cfg.preprocessing.image_height
    batch_size = cfg.hyperparameters.batch_size
    base_lr = cfg.hyperparameters.learning_rate
    stage1_epochs = getattr(cfg.training, 'stage1_epochs', 8)
    stage2_epochs = getattr(cfg.training, 'stage2_epochs', 26)
    stage3_epochs = getattr(cfg.training, 'stage3_epochs', 0)
    target_cols = cfg.targets.cols

    start_time_all = time.time()

    for fold, (train_idx, val_idx) in enumerate(folds_iter):
        set_seed(cfg.hyperparameters.random_seed + fold)
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
            view_swap_prob=getattr(cfg.augmentation, 'view_swap_prob', 0.5),
            target_cols=target_cols
        )
        val_ds = DualStreamBiomassDataset(
            val_df, 
            img_size=img_size, 
            is_training=False, 
            target_cols=target_cols
        )

        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )

        # Save visual sample batches for this fold
        sample_batch_dir = os.path.join(session_dir, "sample_batches")
        try:
            tr_sample = next(iter(train_loader))
            va_sample = next(iter(val_loader))
            save_sample_batch(tr_sample, os.path.join(sample_batch_dir, f"fold_{fold+1}_train_batch.png"), max_samples=4, title_prefix=f"Fold {fold+1} Train")
            save_sample_batch(va_sample, os.path.join(sample_batch_dir, f"fold_{fold+1}_val_batch.png"), max_samples=4, title_prefix=f"Fold {fold+1} Val")
            logger.info(f"  ✓ Saved visual sample batch to {sample_batch_dir}")
        except Exception as e:
            logger.warning(f"Could not save sample batch: {e}")

        # Initialize Model
        model = DualStreamBiomassModel(
            backbone_name=cfg.hyperparameters.backbone,
            num_targets=len(target_cols),
            num_intervals=cfg.loss.num_intervals,
            fusion_dim=cfg.training.fusion_dim,
            dropout=cfg.training.dropout,
            pretrained=True
        ).to(cfg.device)

        criterion = WeightedBiomassLoss(
            cls_weight=cfg.loss.cls_weight,
            num_targets=len(target_cols)
        ).to(cfg.device)

        scaler = torch.amp.GradScaler('cuda')

        best_fold_r2 = -float('inf')
        val_post_preds_best = None
        val_raw_preds_best = None
        best_fold_model_path = os.path.join(session_dir, f"best_model_fold{fold+1}.pt")
        best_s2_model_path = os.path.join(session_dir, f"best_s2_model_fold{fold+1}.pt")
        best_s2_r2 = -float('inf')

        # ----------------------------------------------------------------------
        # STAGE 1: Freeze Backbone -> Warm up Heads
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold+1}] STAGE 1: Warm-up Heads ({stage1_epochs} epochs | Backbone FROZEN) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=base_lr, weight_decay=cfg.hyperparameters.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-5)

        for epoch in range(1, stage1_epochs + 1):
            train_loss, train_reg, train_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, cfg, epoch, stage=1, logger=logger
            )
            scheduler.step()
            val_loss, r2_raw, r2_post, cls_acc, val_raw, val_post = validate(
                model, val_loader, criterion, cfg, val_df, use_tta=cfg.training.use_tta
            )

            logger.info(
                f"[S1 Ep {epoch:02d}] Train: {train_loss:.4f} (reg:{train_reg:.4f}, cls:{train_cls:.4f}) | "
                f"Val: {val_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f} | Cls Acc: {cls_acc:.2%}"
            )

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                val_post_preds_best = val_post
                val_raw_preds_best = val_raw
                torch.save(model.state_dict(), best_fold_model_path)

        # ----------------------------------------------------------------------
        # STAGE 2: Unfreeze Backbone -> Full End-to-End Fine-Tuning with Warmup
        # ----------------------------------------------------------------------
        logger.info(f"\n--- [Fold {fold+1}] STAGE 2: Full Fine-Tuning ({stage2_epochs} epochs | Differential LR + Warmup) ---")
        for p in model.backbone.parameters():
            p.requires_grad = True

        backbone_lr = base_lr * getattr(cfg.training, 'stage2_backbone_lr_factor', 0.1)
        optimizer = AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': base_lr}
        ], weight_decay=cfg.hyperparameters.weight_decay)

        warmup_epochs = getattr(cfg.training, 'stage2_warmup_epochs', 3)
        if warmup_epochs > 0 and stage2_epochs > warmup_epochs:
            warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=stage2_epochs - warmup_epochs, eta_min=backbone_lr * 0.05)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=backbone_lr * 0.05)

        for epoch in range(1, stage2_epochs + 1):
            curr_epoch = stage1_epochs + epoch
            train_loss, train_reg, train_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, cfg, curr_epoch, stage=2, logger=logger
            )
            scheduler.step()
            val_loss, r2_raw, r2_post, cls_acc, val_raw, val_post = validate(
                model, val_loader, criterion, cfg, val_df, use_tta=cfg.training.use_tta
            )

            logger.info(
                f"[S2 Ep {curr_epoch:02d}] Train: {train_loss:.4f} (reg:{train_reg:.4f}, cls:{train_cls:.4f}) | "
                f"Val: {val_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f} | Cls Acc: {cls_acc:.2%}"
            )

            # Track Stage 2 best model independently (guarantees adapted backbone for Stage 3)
            if r2_post > best_s2_r2:
                best_s2_r2 = r2_post
                torch.save(model.state_dict(), best_s2_model_path)

            if r2_post > best_fold_r2:
                best_fold_r2 = r2_post
                val_post_preds_best = val_post
                val_raw_preds_best = val_raw
                torch.save(model.state_dict(), best_fold_model_path)
                logger.info(f"  ✓ New Best Model Saved for Fold {fold+1} (R2: {best_fold_r2:.4f})")

        # ----------------------------------------------------------------------
        # STAGE 3 (Optional): Re-Freeze Backbone
        # ----------------------------------------------------------------------
        if stage3_epochs > 0:
            logger.info(f"\n--- [Fold {fold+1}] STAGE 3: Head Calibration ({stage3_epochs} epochs | Backbone RE-FROZEN) ---")
            # CRITICAL: Always load the adapted backbone checkpoint from Stage 2!
            load_path = best_s2_model_path if os.path.exists(best_s2_model_path) else best_fold_model_path
            if os.path.exists(load_path):
                model.load_state_dict(torch.load(load_path, map_location=cfg.device, weights_only=True))
                logger.info(f"  Loaded adapted backbone checkpoint from {os.path.basename(load_path)} for Stage 3")

            for p in model.backbone.parameters():
                p.requires_grad = False

            stage3_lr = base_lr * getattr(cfg.training, 'stage3_lr_factor', 0.1)
            head_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(head_params, lr=stage3_lr, weight_decay=cfg.hyperparameters.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=stage3_epochs, eta_min=1e-6)

            for epoch in range(1, stage3_epochs + 1):
                curr_epoch = stage1_epochs + stage2_epochs + epoch
                train_loss, train_reg, train_cls = train_one_epoch(
                    model, train_loader, optimizer, criterion, scaler, cfg, curr_epoch, stage=3, logger=logger
                )
                scheduler.step()
                val_loss, r2_raw, r2_post, cls_acc, val_raw, val_post = validate(
                    model, val_loader, criterion, cfg, val_df, use_tta=cfg.training.use_tta
                )

                logger.info(
                    f"[S3 Ep {curr_epoch:02d}] Train: {train_loss:.4f} (reg:{train_reg:.4f}, cls:{train_cls:.4f}) | "
                    f"Val: {val_loss:.4f} | R2 Raw: {r2_raw:.4f} | R2 Post: {r2_post:.4f} | Cls Acc: {cls_acc:.2%}"
                )

                if r2_post > best_fold_r2:
                    best_fold_r2 = r2_post
                    val_post_preds_best = val_post
                    val_raw_preds_best = val_raw
                    torch.save(model.state_dict(), best_fold_model_path)
                    logger.info(f"  ★ New Best Model Saved for Fold {fold+1} (R2: {best_fold_r2:.4f})")

        # Save OOF for this fold (fallback safety if val_post_preds_best was never assigned)
        if val_post_preds_best is None:
            val_post_preds_best = val_post
            val_raw_preds_best = val_raw

        oof_predictions_post[val_idx] = val_post_preds_best
        oof_predictions_raw[val_idx] = val_raw_preds_best
        oof_targets[val_idx] = val_df[TARGET_ORDER].values

        fold_scores.append(best_fold_r2)
        logger.info(f"Fold {fold+1} Final Best Post R2: {best_fold_r2:.4f}")

    # 4. Overall Out-Of-Fold (OOF) Score
    overall_oof_r2 = calculate_competition_r2(oof_targets, oof_predictions_post, cfg.targets.official_weights)
    overall_raw_r2 = calculate_competition_r2(oof_targets, oof_predictions_raw, cfg.targets.official_weights)
    elapsed_min = (time.time() - start_time_all) / 60.0

    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL OUT-OF-FOLD COMPETITION R2 SCORE (Post-Processed): {overall_oof_r2:.4f}")
    logger.info(f"FINAL OUT-OF-FOLD COMPETITION R2 SCORE (Raw Physics):    {overall_raw_r2:.4f}")
    logger.info(f"Per-fold scores: {[round(s, 4) for s in fold_scores]}")
    logger.info(f"Total CV Training Time: {elapsed_min:.2f} minutes")
    logger.info("=" * 60)

    # Save OOF predictions CSV
    oof_df = df[['sample_id', 'Species', 'State', 'Sampling_Date'] + TARGET_ORDER].copy()
    for idx, col in enumerate(TARGET_ORDER):
        oof_df[f'pred_{col}'] = oof_predictions_post[:, idx]
        oof_df[f'pred_raw_{col}'] = oof_predictions_raw[:, idx]
    oof_csv_path = os.path.join(session_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_csv_path}")


if __name__ == "__main__":
    run_training()