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
    apply_2nd_place_postprocess,
    soft_physics_postprocess,
    set_seed,
    TARGET_ORDER
)
from dataset import DualStreamBiomassDataset
from models import DualStreamBiomassModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_pseudo_labels(model_checkpoint_dir, test_df, test_img_dir="test", img_size=None):
    """
    Step 1: Ensembles high-performing models to generate initial pseudo-labels for the test set.
    """
    if img_size is None:
        img_size = cfg.preprocessing.image_height
        
    model_paths = sorted(glob.glob(os.path.join(model_checkpoint_dir, "best_model_fold*.pt")))
    if not model_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {model_checkpoint_dir}")
        
    print(f"Generating pseudo-labels using {len(model_paths)} models from {model_checkpoint_dir}...")
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
        ckpt_num_targets = max(head_indices) + 1 if head_indices else 3

        # Smart detection of backbone architecture from checkpoint weights
        clean_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if 'cross_view_attn.in_proj_weight' in clean_sd:
            dim = clean_sd['cross_view_attn.in_proj_weight'].shape[1]
            if dim == 1536:
                backbone = 'convnextv2_large'
            elif dim == 768:
                backbone = 'vit_base_patch16_dinov3_qkvb'
            elif dim == 384:
                backbone = 'vit_small_patch14_dinov2'
            else:
                backbone = cfg.hyperparameters.backbone
        else:
            backbone = cfg.hyperparameters.backbone

        model = DualStreamBiomassModel(
            backbone_name=backbone,
            num_targets=ckpt_num_targets,
            num_intervals=cfg.loss.num_intervals,
            fusion_dim=cfg.training.fusion_dim,
            pretrained=False
        ).to(DEVICE)
        model.load_state_dict(clean_sd, strict=False)
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
    # Apply physical ground-truth post-processing to pseudo-labels
    states = test_df['State'].tolist() if 'State' in test_df.columns else None
    calibrated_pseudo = apply_2nd_place_postprocess(avg_preds, states=states)
    
    pseudo_df = test_df.copy()
    for idx, col in enumerate(TARGET_ORDER):
        pseudo_df[col] = calibrated_pseudo[:, idx]
        
    return pseudo_df


def online_fine_tune(
    train_df,
    pseudo_test_df,
    model_checkpoint_dir=None,
    epochs=6,
    lr=3e-5,
    swa_epochs=3,
    output_model_path="logs/online_trained_model.pt",
    output_submission_path="submission_online_trained.csv"
):
    """
    Step 2: Online Training on Combined Train + Test (pseudo-labeled) dataset.
    Initializes from the best trained checkpoint and applies SWA across the final epochs.
    """
    print("=" * 60)
    print(f"TEST-TIME ONLINE TRAINING (Train: {len(train_df)}, Pseudo-Test: {len(pseudo_test_df)})")
    print("=" * 60)

    # 1. Discover initial checkpoint to fine-tune from
    if model_checkpoint_dir is None:
        candidate_logs = sorted(glob.glob("logs/dual_stream_*"), reverse=True)
        model_checkpoint_dir = candidate_logs[0] if candidate_logs else None
        
    model_paths = sorted(glob.glob(os.path.join(model_checkpoint_dir, "best_model_fold*.pt"))) if model_checkpoint_dir else []
    init_ckpt_path = model_paths[0] if model_paths else None
    
    ckpt_num_targets = 3
    backbone = cfg.hyperparameters.backbone
    clean_sd = None
    if init_ckpt_path and os.path.exists(init_ckpt_path):
        print(f"Initializing online fine-tuning from checkpoint: {init_ckpt_path}")
        raw_sd = torch.load(init_ckpt_path, map_location=DEVICE, weights_only=True)
        clean_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
        head_indices = [int(k.split('.')[1]) for k in clean_sd.keys() if k.startswith('reg_heads.') and '.0.weight' in k]
        ckpt_num_targets = max(head_indices) + 1 if head_indices else 3
        if 'cross_view_attn.in_proj_weight' in clean_sd:
            dim = clean_sd['cross_view_attn.in_proj_weight'].shape[1]
            if dim == 1536:
                backbone = 'convnextv2_large'
            elif dim == 768:
                backbone = 'vit_base_patch16_dinov3_qkvb'
            elif dim == 384:
                backbone = 'vit_small_patch14_dinov2'

    # Align target columns with model architecture (3 base vs 5 full)
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g'] if ckpt_num_targets == 3 else TARGET_ORDER
    print(f"Target count: {len(target_cols)} ({target_cols}) | Backbone: {backbone}")

    combined_df = pd.concat([train_df, pseudo_test_df], ignore_index=True)
    img_size = cfg.preprocessing.image_height
    
    dataset = DualStreamBiomassDataset(
        combined_df,
        img_size=img_size,
        is_training=True,
        camera_scaling_prob=cfg.augmentation.camera_scaling_prob,
        target_cols=target_cols
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg.hyperparameters.batch_size, 
        shuffle=True, 
        pin_memory=True
    )

    model = DualStreamBiomassModel(
        backbone_name=backbone,
        num_targets=len(target_cols),
        num_intervals=cfg.loss.num_intervals,
        fusion_dim=cfg.training.fusion_dim,
        dropout=cfg.training.dropout,
        pretrained=clean_sd is None
    ).to(DEVICE)

    if clean_sd is not None:
        model.load_state_dict(clean_sd, strict=False)

    criterion = WeightedBiomassLoss(
        cls_weight=cfg.loss.cls_weight,
        num_targets=len(target_cols)
    ).to(DEVICE)

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
            t_cls = batch['targets_cls'].to(DEVICE) if 'targets_cls' in batch and batch['targets_cls'] is not None else None

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

    # Predict test set using the online-adapted model
    test_ds = DualStreamBiomassDataset(
        pseudo_test_df,
        img_size=img_size,
        is_training=False,
        camera_scaling_prob=0.0
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            l = batch['image_left'].to(DEVICE)
            r = batch['image_right'].to(DEVICE)
            reg1, _ = model(l, r)
            reg2, _ = model(torch.flip(r, [3]), torch.flip(l, [3]))
            avg_r = [(a + b) * 0.5 for a, b in zip(reg1, reg2)]
            test_preds.append(torch.cat(avg_r, dim=1).cpu().numpy())

    test_preds_np = np.concatenate(test_preds, axis=0)
    if len(target_cols) == 3:
        test_preds_np = derive_5_targets(test_preds_np)
        
    states = pseudo_test_df['State'].tolist() if 'State' in pseudo_test_df.columns else None
    final_post = apply_2nd_place_postprocess(test_preds_np, states=states)

    # Format Kaggle submission
    clean_ids = pseudo_test_df['clean_id'].tolist() if 'clean_id' in pseudo_test_df.columns else pseudo_test_df['sample_id'].tolist()
    pred_dict = {
        clean_ids[i]: {col: final_post[i, c_idx] for c_idx, col in enumerate(TARGET_ORDER)}
        for i in range(len(clean_ids))
    }

    test_df_raw = pd.read_csv("test.csv")
    if 'target_name' in test_df_raw.columns:
        sub_df = test_df_raw.copy()
        clean_id_col = sub_df['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
        sub_df['target'] = [pred_dict.get(cid, {}).get(tname, 0.0) for cid, tname in zip(clean_id_col, sub_df['target_name'])]
        sub_res = sub_df[['sample_id', 'target']]
    else:
        records = []
        for cid in clean_ids:
            for col in TARGET_ORDER:
                records.append({'sample_id': f"{cid}__{col}", 'target': pred_dict[cid][col]})
        sub_res = pd.DataFrame(records)

    sub_res.to_csv(output_submission_path, index=False)
    print(f"✓ Online trained submission saved to: {output_submission_path}")
    return model


if __name__ == "__main__":
    set_seed(42)
    candidate_logs = sorted(glob.glob("logs/dual_stream_*"), reverse=True)
    train_csv = "train_converted.csv" if os.path.exists("train_converted.csv") else "wide.csv"
    if candidate_logs and os.path.exists("test.csv") and os.path.exists(train_csv):
        chk_dir = candidate_logs[0]
        test_df = pd.read_csv("test.csv")
        train_df = pd.read_csv(train_csv)
        
        # 1. Generate pseudo-labels
        pseudo_test = generate_pseudo_labels(chk_dir, test_df)
        
        # 2. Fine-tune on Train + Pseudo-labeled Test
        online_fine_tune(train_df, pseudo_test, model_checkpoint_dir=chk_dir)
    else:
        print("Please train a model first using train_unified.py.")
