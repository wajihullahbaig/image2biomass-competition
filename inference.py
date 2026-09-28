"""
CSIRO Image2Biomass: Dual-Stream DINO Vision Pipeline
Unified Multi-Fold Inference & Submission Script (inference.py)

Key Features:
1. Multi-Fold Ensembling: Automatically discovers and averages all fold checkpoints (best_model_fold*.pt).
2. Dual-Stream Evaluation: Splits test images into Left and Right 1:1 square views.
3. Test-Time Augmentation (TTA): Dual-stream standard + horizontal flip averaging:
   - View 1: model(img_l, img_r)
   - View 2: model(torch.flip(img_r, [3]), torch.flip(img_l, [3]))
   - Average: (r1 + r2) * 0.5
4. Soft Physical Blend Post-Processing:
   - Clover downscale (0.8)
   - Dead fringe expansion (>20 * 1.1, <10 * 0.9)
   - Mass-conservation blending (GDM and Total)
5. Automatic Architecture Detection:
   - Supports ViT Small (dim=384), ViT Base (dim=768), and ConvNeXt-V2 Large (dim=1536).
6. Formats output strictly to competition submission format: sample_id,target.
"""

import os
import sys
import glob
import argparse
import numpy as np

# Ensure standard UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ==============================================================================
# Constants
# ==============================================================================
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==============================================================================
# Dataset for Inference
# ==============================================================================
class DualStreamTestDataset(Dataset):
    """
    Test Dataset for 2:1 Panoramic Pasture Images.
    Splits image into Left and Right views for dual-stream feature extraction.
    """
    def __init__(self, df, img_dir=None, img_size=384):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, rel_path):
        if self.img_dir:
            fname = os.path.basename(rel_path)
            candidate = os.path.join(self.img_dir, fname)
            if os.path.exists(candidate):
                return candidate
        if os.path.exists(rel_path):
            return rel_path
        fname = os.path.basename(rel_path)
        for cand_dir in ['test', 'train', 'images', os.path.join('..', 'test'), os.path.join('..', 'train'), './data']:
            cand = os.path.join(cand_dir, fname)
            if os.path.exists(cand):
                return cand
        for root, _, files in os.walk('.'):
            if fname in files:
                return os.path.join(root, fname)
        return rel_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row['image_path'])

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            # Fallback black image if corrupted
            raw_rgb = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = raw_rgb.shape
        mid_w = w // 2

        left_np = raw_rgb[:, :mid_w].copy()
        right_np = raw_rgb[:, mid_w:].copy()

        tensor_l = self.transform(Image.fromarray(left_np))
        tensor_r = self.transform(Image.fromarray(right_np))

        return {
            'image_left': tensor_l,
            'image_right': tensor_r,
            'image_path': img_path,
            'clean_id': row.get('clean_id', os.path.splitext(os.path.basename(img_path))[0]),
            'sample_id': row.get('sample_id', row.get('clean_id', f'sample_{idx}')),
        }


# ==============================================================================
# Model Architecture
# ==============================================================================
class DualStreamBiomassModel(nn.Module):
    def __init__(self, backbone_name="vit_small_patch16_dinov3_qkvb", num_targets=5, num_intervals=7, fusion_dim=384, dropout=0.3, pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.fusion_dim = fusion_dim
        self.num_targets = num_targets
        self.num_intervals = num_intervals

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.backbone_dim = self.backbone.num_features

        num_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=self.backbone_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.backbone_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(fusion_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_targets)
        ])

        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_intervals)
            ) for _ in range(num_targets)
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
        tokens = torch.stack([feat_l, feat_r], dim=1)
        attn_out, _ = self.cross_view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        fused = self.fusion_mlp(torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1))
        reg_preds = [F.softplus(head(fused)) for head in self.reg_heads]
        cls_preds = [head(fused) for head in self.cls_heads]
        return reg_preds, cls_preds


# ==============================================================================
# Model Architecture Detector
# ==============================================================================
def detect_model_architecture(state_dict):
    """
    Inspects state_dict tensor dimensions and keys to automatically determine
    whether the model is DINO ViT-Small (dim=384), ViT-Base (dim=768), or ConvNeXt-V2 Large.
    """
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        if 'backbone' in state_dict:
            return state_dict['backbone'], state_dict['state_dict']
        state_dict = state_dict['state_dict']

    clean_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 1. Primary check: cross-view attention projection embed dimension
    if 'cross_view_attn.in_proj_weight' in clean_sd:
        dim = clean_sd['cross_view_attn.in_proj_weight'].shape[1]
        if dim == 384:
            return 'vit_small_patch16_dinov3_qkvb', clean_sd
        elif dim == 768:
            return 'vit_base_patch16_dinov3_qkvb', clean_sd
        elif dim == 1536:
            return 'convnextv2_large', clean_sd

    # 2. Secondary check: patch projection or convolutional stages
    keys = list(clean_sd.keys())
    if any('stages' in k for k in keys) or any('convnext' in k.lower() for k in keys):
        return 'convnextv2_large', clean_sd

    return 'vit_small_patch16_dinov3_qkvb', clean_sd


# ==============================================================================
# Soft Physical Post-Processing & Calibration
# ==============================================================================
def soft_physics_postprocess(preds_np, clover_scale=0.8, dead_upper_thresh=20.0, dead_upper_scale=1.1, dead_lower_thresh=10.0, dead_lower_scale=0.9, gdm_weight=0.5, total_weight=0.5):
    """Enforces physical mass relationships and thatch fringe expansion:"""
    preds = np.maximum(preds_np.copy(), 0.0)
    green = preds[:, 0]
    dead = preds[:, 1]
    clover = preds[:, 2] * clover_scale
    gdm = preds[:, 3]
    total = preds[:, 4]

    # Dead thatch fringe expansion
    dead = np.where(dead > dead_upper_thresh, dead * dead_upper_scale,
           np.where(dead < dead_lower_thresh, dead * dead_lower_scale, dead))

    # Mass-conservation blends
    gdm_blended = gdm_weight * gdm + (1.0 - gdm_weight) * (green + clover)
    total_blended = total_weight * total + (1.0 - total_weight) * (green + clover + dead)

    return np.maximum(np.column_stack([green, dead, clover, gdm_blended, total_blended]), 0.0)


# ==============================================================================
# Checkpoint Discovery
# ==============================================================================
def find_model_checkpoints(model_dir="models"):
    """Finds all model checkpoints across specified directory, root, and subfolders."""
    candidates = []
    if os.path.exists(model_dir):
        candidates.extend(glob.glob(os.path.join(model_dir, "*.pt")) + glob.glob(os.path.join(model_dir, "*.pth")))
    candidates.extend(glob.glob("best_model_fold*.pt") + glob.glob("*.pt") + glob.glob("*.pth"))

    seen = set()
    unique = []
    for c in sorted(candidates):
        norm = os.path.abspath(c)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)

    # Filter to model checkpoints
    ckpts = [f for f in unique if 'model' in os.path.basename(f).lower() or 'fold' in os.path.basename(f).lower()]
    return sorted(ckpts if ckpts else unique)


# ==============================================================================
# Inference Runner
# ==============================================================================
def run_inference():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream DINO Inference")
    parser.add_argument('--model_dir', type=str, default='models', help='Directory with trained fold checkpoints')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='Path to test.csv')
    parser.add_argument('--img_dir', type=str, default=None, help='Optional directory containing test images')
    parser.add_argument('--img_size', type=int, default=384, help='Image resolution (384 for ViT Small, 512 for ViT Base)')
    parser.add_argument('--batch_size', type=int, default=8, help='Inference batch size')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Output submission file path')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Enable mirrored horizontal-flip TTA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("CSIRO Image2Biomass Inference & Submission Generation")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Image Size: {args.img_size}x{args.img_size} | Batch Size: {args.batch_size} | TTA: {args.use_tta}")
    print("=" * 60)

    # 1. Discover Checkpoints
    checkpoints = find_model_checkpoints(args.model_dir)
    print(f"\nDiscovered {len(checkpoints)} Model Checkpoint(s):")
    for cp in checkpoints:
        print(f"  - {cp} ({os.path.getsize(cp) / (1024*1024):.1f} MB)")

    if len(checkpoints) == 0:
        print(f"\n[ERROR] No model checkpoints found in '{args.model_dir}' or working directory!")
        sys.exit(1)

    # 2. Read Test Data
    if not os.path.exists(args.test_csv):
        print(f"[ERROR] Test CSV not found at '{args.test_csv}'!")
        sys.exit(1)

    test_df_raw = pd.read_csv(args.test_csv)
    if 'target_name' in test_df_raw.columns:
        test_df_raw['clean_id'] = test_df_raw['sample_id'].astype(str).apply(lambda x: x.split('__')[0])
        unique_test = test_df_raw[['clean_id', 'image_path']].drop_duplicates(subset=['clean_id']).reset_index(drop=True)
    else:
        unique_test = test_df_raw.copy()
        if 'clean_id' not in unique_test.columns:
            unique_test['clean_id'] = unique_test['sample_id']

    print(f"\nTest Dataset: {len(unique_test)} unique pasture images to predict.")

    # 3. Create DataLoader
    test_ds = DualStreamTestDataset(unique_test, img_dir=args.img_dir, img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Predict across all checkpoints
    all_fold_preds = []
    for cp_idx, cp_path in enumerate(checkpoints):
        cp_name = os.path.basename(cp_path)
        raw_state = torch.load(cp_path, map_location=device)
        backbone_name, clean_sd = detect_model_architecture(raw_state)

        print(f"\n[{cp_idx + 1}/{len(checkpoints)}] Evaluating {cp_name} (Architecture: {backbone_name})...")
        model = DualStreamBiomassModel(backbone_name=backbone_name, pretrained=False).to(device)
        model.load_state_dict(clean_sd)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  Inference {cp_name}", leave=False):
                img_l = batch['image_left'].to(device)
                img_r = batch['image_right'].to(device)

                if args.use_tta:
                    # Standard view
                    r_std, _ = model(img_l, img_r)
                    # Mirrored horizontal view
                    r_flip, _ = model(torch.flip(img_r, [3]), torch.flip(img_l, [3]))
                    avg_r = [(a + b) * 0.5 for a, b in zip(r_std, r_flip)]
                else:
                    avg_r, _ = model(img_l, img_r)

                pred_batch = torch.cat(avg_r, dim=1).cpu().numpy()
                fold_preds.append(pred_batch)

        all_fold_preds.append(np.concatenate(fold_preds, axis=0))

    # 5. Ensemble Average & Soft Physical Post-Processing
    print("\n" + "=" * 60)
    print(f"Ensembling {len(all_fold_preds)} checkpoint prediction(s)...")
    avg_raw = np.mean(all_fold_preds, axis=0)
    avg_post = soft_physics_postprocess(avg_raw)
    print("Applied soft physics post-processing (clover_scale=0.8, dead fringe expansion, mass conservation)")
    print("=" * 60)

    # 6. Build Submission DataFrame
    clean_ids = unique_test['clean_id'].tolist()
    pred_dict = {
        clean_ids[i]: {col: float(avg_post[i, c_idx]) for c_idx, col in enumerate(TARGET_NAMES)}
        for i in range(len(clean_ids))
    }

    if 'target_name' in test_df_raw.columns:
        sub_df = test_df_raw.copy()
        sub_df['target'] = sub_df.apply(
            lambda row: pred_dict.get(row['clean_id'], {}).get(row['target_name'], 0.0), axis=1
        )
        res = sub_df[['sample_id', 'target']].copy()
    else:
        records = []
        for cid in clean_ids:
            for col in TARGET_NAMES:
                records.append({'sample_id': f"{cid}__{col}", 'target': pred_dict[cid][col]})
        res = pd.DataFrame(records)

    assert res['target'].isna().sum() == 0, "ERROR: Submission contains NaN values!"
    assert len(res) > 0, "ERROR: Submission is empty!"

    res.to_csv(args.output_csv, index=False)
    print(f"\n[SUCCESS] Submission saved to '{args.output_csv}' ({len(res)} rows).")
    print("\nTarget Statistics Preview:")
    print(res['target'].describe())
    print("\nFirst 10 Rows:")
    print(res.head(10))


if __name__ == '__main__':
    run_inference()
