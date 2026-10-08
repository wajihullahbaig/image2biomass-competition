"""
CSIRO Image2Biomass: Multi-Checkpoint Inference & Submission (inference.py)

- Averages raw predictions of every checkpoint found in --model_dir (fold models and/or full-data models).
- Each checkpoint carries its backbone and img_size; legacy raw state_dicts are detected by embedding dim.
- Mirrored-panorama TTA. 1st-place post-processing is off by default: it lowered OOF R2 in every run so far.
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import (TARGET_NAMES, DualStreamBiomassDataset, DualStreamBiomassModel,
                   predict, soft_physics_postprocess)

LEGACY_BACKBONES = {384: 'vit_small_patch16_dinov3_qkvb', 768: 'vit_base_patch16_dinov3_qkvb',
                    1024: 'vit_large_patch16_dinov3_qkvb', 1536: 'convnextv2_large'}


def load_checkpoint(path, legacy_img_size=512):
    """Returns (state_dict, backbone, img_size) for new-format or legacy checkpoints."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt and 'backbone' in ckpt:
        return ckpt['state_dict'], ckpt['backbone'], ckpt['img_size']
    state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    dim = state['cross_view_attn.in_proj_weight'].shape[1]
    return state, LEGACY_BACKBONES[dim], legacy_img_size


def load_test_images(test_csv):
    """One row per unique image from the long-format test.csv."""
    df = pd.read_csv(test_csv)
    df['image_id'] = df['sample_id'].astype(str).str.split('__').str[0]
    return df, df[['image_id', 'image_path']].drop_duplicates('image_id').reset_index(drop=True)


def build_submission(test_long, images, preds):
    """Maps per-image predictions back onto the long-format sample_id rows."""
    wide = pd.DataFrame(preds, columns=TARGET_NAMES)
    wide['image_id'] = images['image_id']
    long = wide.melt(id_vars='image_id', var_name='target_name', value_name='target')
    sub = test_long[['sample_id', 'image_id', 'target_name']].merge(long, on=['image_id', 'target_name'], how='left')
    assert sub['target'].notna().all(), "Submission contains NaN values"
    return sub[['sample_id', 'target']]


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = sorted(glob.glob(os.path.join(args.model_dir, '**', '*.pt'), recursive=True))
    if not checkpoints:
        sys.exit(f"[ERROR] No .pt checkpoints found under '{args.model_dir}'")

    test_long, images = load_test_images(args.test_csv)
    print(f"Device: {device} | Test images: {len(images)} | Checkpoints: {len(checkpoints)} | Post-process: {args.postprocess}")

    all_preds = []
    for path in checkpoints:
        state, backbone, img_size = load_checkpoint(path)
        print(f"  {os.path.basename(path)}: {backbone} @ {img_size}")
        model = DualStreamBiomassModel(backbone_name=backbone, pretrained=False).to(device)
        model.load_state_dict(state)
        loader = DataLoader(DualStreamBiomassDataset(images, img_size, False, args.img_root),
                            batch_size=8, shuffle=False, num_workers=0)
        all_preds.append(predict(model, loader, device))
        del model
        torch.cuda.empty_cache()

    preds = np.mean(all_preds, axis=0)
    if args.postprocess == 'first_place':
        preds = soft_physics_postprocess(preds)

    sub = build_submission(test_long, images, preds)
    sub.to_csv(args.output_csv, index=False)
    print(f"[SUCCESS] {args.output_csv} ({len(sub)} rows)")
    print(sub.head(10))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass Dual-Stream Inference")
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--img_root', type=str, default='.', help='Directory that image_path entries are relative to')
    parser.add_argument('--postprocess', choices=['none', 'first_place'], default='none')
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run_inference(parse_args())
