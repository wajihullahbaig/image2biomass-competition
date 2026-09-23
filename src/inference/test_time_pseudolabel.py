# test_time_pseudolabel.py - 1st-Place Test-Time Online Training / Pseudo-Labeling
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add training module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..', 'training'))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from config.loader import cfg
from common import (
    WeightedBiomassLoss,
    derive_5_targets,
    soft_physics_postprocess,
    set_seed,
    TARGET_ORDER
)
from dataset import DualStreamBiomassDataset
from models import DualStreamBiomassModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_pseudo_labels(model_checkpoint_dir, test_df, test_img_dir="test", img_size=512):
    """
    Step 1: Ensembles high-performing models to generate initial pseudo-labels for the test set.
    """
    model_paths = sorted(glob.glob(os.path.join(model_checkpoint_dir, "best_model_fold*.pt")))
    if not model_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {model_checkpoint_dir}")
        
    print(f"Generating pseudo-labels using {len(model_paths)} models...")
    test_dataset = DualStreamBiomassDataset(
        test_df,
        img_dir=test_img_dir,
        img_size=img_size,
        is_training=False,
        camera_scaling_prob=0.0
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    all_preds = []
    for model_path in model_paths:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        head_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('reg_heads.') and '.0.weight' in k]
        ckpt_num_targets = max(head_indices) + 1 if head_indices else 5

        model = DualStreamBiomassModel(
            backbone_name=cfg.hyperparameters.backbone,
            num_targets=ckpt_num_targets,
            num_intervals=cfg.loss.num_intervals,
            fusion_dim=cfg.training.fusion_dim,
            pretrained=False
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                img_l = batch['image_left'].to(DEVICE)
                img_r = batch['image_right'].to(DEVICE)
                reg_preds, _ = model(img_l, img_r)
                p = torch.cat(reg_preds, dim=1).cpu().numpy()
                fold_preds.append(p)
        fold_preds_np = np.concatenate(fold_preds, axis=0)
        if ckpt_num_targets == 3:
            fold_preds_np = derive_5_targets(fold_preds_np)
        all_preds.append(fold_preds_np)

    avg_preds = np.mean(all_preds, axis=0)
    # Apply soft physical calibration to pseudo-labels
    calibrated_pseudo = soft_physics_postprocess(avg_preds)
    
    pseudo_df = test_df.copy()
    for idx, col in enumerate(TARGET_ORDER):
        pseudo_df[col] = calibrated_pseudo[:, idx]
        
    return pseudo_df


def online_fine_tune(
    train_df,
    pseudo_test_df,
    epochs=12,
    lr=1e-4,
    swa_epochs=4,
    output_model_path="logs/online_trained_model.pt"
):
    """
    Step 2: Online Training on Combined Train + Test (pseudo-labeled) dataset.
    Uses Stochastic Weight Averaging (SWA) across the final epochs.
    """
    print("=" * 60)
    print(f"TEST-TIME ONLINE TRAINING (Train: {len(train_df)}, Pseudo-Test: {len(pseudo_test_df)})")
    print("=" * 60)

    combined_df = pd.concat([train_df, pseudo_test_df], ignore_index=True)
    img_size = cfg.preprocessing.image_height
    
    dataset = DualStreamBiomassDataset(
        combined_df,
        img_size=img_size,
        is_training=True,
        camera_scaling_prob=cfg.augmentation.camera_scaling_prob
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg.hyperparameters.batch_size, 
        shuffle=True, 
        pin_memory=True
    )

    model = DualStreamBiomassModel(
        backbone_name=cfg.hyperparameters.backbone,
        num_targets=5,
        num_intervals=cfg.loss.num_intervals,
        fusion_dim=cfg.training.fusion_dim,
        dropout=cfg.training.dropout,
        pretrained=True
    ).to(DEVICE)

    criterion = WeightedBiomassLoss(
        loss_weights=cfg.targets.official_weights,
        cls_weight=cfg.loss.cls_weight
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda')

    # SWA state storage
    swa_weights = None
    swa_count = 0

    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        pbar = tqdm(loader, desc=f"Online Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            img_l = batch['image_left'].to(DEVICE)
            img_r = batch['image_right'].to(DEVICE)
            t_reg = batch['targets'].to(DEVICE)
            t_cls = batch['targets_cls'].to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                reg, cls = model(img_l, img_r)
                loss, _, _ = criterion(reg, cls, t_reg, t_cls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * img_l.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = loss_sum / len(dataset)
        print(f"Epoch {epoch:02d} | Train + Test Loss: {avg_loss:.4f}")

        # Accumulate weights for SWA in final epochs
        if epoch > (epochs - swa_epochs):
            swa_count += 1
            if swa_weights is None:
                swa_weights = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    swa_weights[k] += v

    # Finalize SWA model
    if swa_weights is not None and swa_count > 0:
        for k in swa_weights:
            swa_weights[k] = swa_weights[k] / swa_count
        model.load_state_dict(swa_weights)
        print(f"✓ Applied SWA across last {swa_count} epochs.")

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"✓ Online trained model saved to: {output_model_path}")
    return model


if __name__ == "__main__":
    set_seed(42)
    candidate_logs = sorted(glob.glob("logs/dual_stream_*"), reverse=True)
    if candidate_logs and os.path.exists("test.csv") and os.path.exists("wide.csv"):
        chk_dir = candidate_logs[0]
        test_df = pd.read_csv("test.csv")
        train_df = pd.read_csv("wide.csv")
        
        # 1. Generate pseudo-labels
        pseudo_test = generate_pseudo_labels(chk_dir, test_df)
        
        # 2. Fine-tune on Train + Pseudo-labeled Test
        online_fine_tune(train_df, pseudo_test)
    else:
        print("Please train a model first using train_unified.py.")
