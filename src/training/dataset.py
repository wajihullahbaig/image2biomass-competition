# dataset.py - Dual-Stream High-Resolution Pasture Dataset
import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from common import (
    IMAGENET_DEFAULT_MEAN, 
    IMAGENET_DEFAULT_STD, 
    TARGET_ORDER,
    get_interval_labels
)


def apply_camera_scale_simulation(image_np, prob=0.2):
    """
    Simulates camera focal length and viewing distance variation by randomly
    downscaling the image (0.85 - 1.0) and padding with black pixels.
    From 1st Place Solution (+0.01 PB gain).
    """
    if random.random() < prob:
        h, w = image_np.shape[:2]
        background = np.zeros_like(image_np)
        scale = random.uniform(0.85, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        background[top:top + new_h, left:left + new_w] = resized
        return background
    return image_np


def permute_vertical_strips(image_np, n_strips=4, prob=0.5):
    """
    Randomly permutes N vertical strips of the pasture quadrat sub-image.
    Because biomass is purely additive mass, shuffling vertical slices
    conserves total grams in the frame while preventing spatial overfitting.
    From 3rd Place Solution (+0.02 gain).
    """
    if random.random() < prob:
        strips = np.array_split(image_np, n_strips, axis=1)
        random.shuffle(strips)
        return np.concatenate(strips, axis=1)
    return image_np


def get_dual_stream_transforms(img_size=512, is_training=True, camera_scaling_prob=0.2):
    """
    Transforms applied independently to each sub-image (view).
    Preserves natural square aspect ratio.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),  # 3rd place: learns leaf morphology over color
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])


class DualStreamBiomassDataset(Dataset):
    """
    Dual-Stream Dataset for Panoramic (2:1) Pasture Images.
    Splits 2000x1000 image down the vertical centerline into two natural 1:1 square
    sub-images (Left view: 0..1000, Right view: 1000..2000).
    """
    def __init__(self, 
                 df, 
                 img_dir=None,
                 img_size=512,
                 is_training=True,
                 camera_scaling_prob=0.2,
                 strip_shuffle_prob=0.5,
                 view_swap_prob=0.5,
                 target_cols=TARGET_ORDER):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_training = is_training
        self.camera_scaling_prob = camera_scaling_prob
        self.strip_shuffle_prob = strip_shuffle_prob
        self.view_swap_prob = view_swap_prob
        self.target_cols = target_cols
        
        self.transform = get_dual_stream_transforms(
            img_size=self.img_size, 
            is_training=self.is_training,
            camera_scaling_prob=self.camera_scaling_prob
        )
        
        # Precompute targets if present in dataframe
        self.has_targets = all(c in self.df.columns for c in self.target_cols)
        if self.has_targets:
            self.targets_reg = self.df[self.target_cols].values.astype(np.float32)
            self.targets_cls = get_interval_labels(self.targets_reg, self.target_cols)
        else:
            self.targets_reg = None
            self.targets_cls = None

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
        # Check standard root directories
        fname = os.path.basename(rel_path)
        for d in ['train', 'test', 'images']:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Image not found for path: {rel_path}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row['image_path'])
        
        # Load raw image
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        
        h, w, _ = raw_rgb.shape
        mid_w = w // 2
        
        # Split into Left and Right views
        left_np = raw_rgb[:, :mid_w].copy()
        right_np = raw_rgb[:, mid_w:].copy()
        
        # 3rd-Place Augmentations during training
        if self.is_training:
            # 1. Left/Right view swap (50% probability)
            if self.view_swap_prob > 0 and random.random() < self.view_swap_prob:
                left_np, right_np = right_np, left_np

            # 2. Camera focal/scaling simulation
            if self.camera_scaling_prob > 0:
                left_np = apply_camera_scale_simulation(left_np, prob=self.camera_scaling_prob)
                right_np = apply_camera_scale_simulation(right_np, prob=self.camera_scaling_prob)

            # 3. Vertical 4-strip permutation (conserves total biomass while breaking spatial artifacts)
            if self.strip_shuffle_prob > 0:
                left_np = permute_vertical_strips(left_np, n_strips=4, prob=self.strip_shuffle_prob)
                right_np = permute_vertical_strips(right_np, n_strips=4, prob=self.strip_shuffle_prob)
            
        left_pil = Image.fromarray(left_np)
        right_pil = Image.fromarray(right_np)
        
        # Independent transforms
        tensor_l = self.transform(left_pil)
        tensor_r = self.transform(right_pil)
        
        item = {
            'image_left': tensor_l,
            'image_right': tensor_r,
            'sample_id': row.get('sample_id', row.get('clean_id', f'sample_{idx}')),
            'image_path': img_path
        }
        
        if self.has_targets:
            item['targets'] = torch.tensor(self.targets_reg[idx], dtype=torch.float32)
            item['targets_cls'] = torch.tensor(self.targets_cls[idx], dtype=torch.long)
            
        return item

