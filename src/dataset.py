# dataset.py
# dataset_tiled.py - Enhanced Dataset with Tile-Based Augmentation
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

# Local Imports
from common import (
    CORE_SPECIES,
    get_season, load_data, get_image_data_transforms,
    get_hsv_green_mask
)

class TiledBiomassDataset(Dataset):
    """
    Biomass Dataset with Tile-Based Augmentation.
    
    Augmentation Modes:
    1. STITCH: Split into 4 tiles, apply transforms, stitch back (targets unchanged)
    2. DIVIDE: Split into 4 tiles, treat each as separate sample (targets divided by 4)
    3. NORMAL: Original image as-is
    
    This creates 1 + 4 = 5 samples per original image in training.
    """
    def __init__(self, df, transform=None, target_cols=None, aux_cols=None, 
                 is_test=False, mode='training', tile_prob=0.8):
        """
        Args:
            df: DataFrame with image paths and targets
            transform: Torchvision transforms (applied AFTER tiling)
            mode: 'training' (use augmentation), 'validation' (no augmentation), 'holdout' (no augmentation)
            tile_prob: Probability of applying tiling (only for training)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols or ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Interaction_Add']
        self.is_test = is_test
        self.mode = mode
        self.tile_prob = tile_prob
        
        self.core_species = CORE_SPECIES
        # Prefer soft probability labels if present; fallback to binary/multi-hot
        prob_cols = [f'SpeciesProb_{sp}' for sp in self.core_species]
        hard_cols = [f'Species_{sp}' for sp in self.core_species]
        if all(c in self.df.columns for c in prob_cols):
            self.species_cols = prob_cols
        else:
            self.species_cols = hard_cols

        # Auto-include bin features if present (computed in preprocessing)
        extra_aux = []
        ordinal_cols = ['NDVI_Bin_Ordinal', 'Height_Bin_Ordinal']
        onehot_cols = [f'NDVI_Bin_OH_{k}' for k in range(4)] + [f'Height_Bin_OH_{k}' for k in range(4)]
        for c in ordinal_cols + onehot_cols:
            if c in self.df.columns:
                extra_aux.append(c)
        # Include species richness if computed
        if 'Species_Count' in self.df.columns:
            extra_aux.append('Species_Count')
        if extra_aux:
            self.aux_cols = self.aux_cols + extra_aux
        
        # Validation check
        if not all(c in self.df.columns for c in self.species_cols):
            for c in self.species_cols:
                if c not in self.df.columns: 
                    self.df[c] = 0.0
        
        # Pre-expand dataset index for training mode
        # Each original sample becomes: 1 (original) + 1 (stitch) + 4 (divided tiles) = 6 augmented views
        if mode == 'training':
            self.effective_length = len(self.df) * 6  # Original + Stitch + 4 Tiles
        else:
            self.effective_length = len(self.df)  # Validation/Holdout: No augmentation
    
    def __len__(self):
        return self.effective_length
    
    def _load_image(self, img_path):
        """Load and convert image to RGB."""
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Image not found: {img_path}")
    
    def _split_image_into_tiles(self, image):
        """
        Split image into 4 quadrants (2x2 grid).
        Returns: [top_left, top_right, bottom_left, bottom_right]
        """
        w, h = image.size
        mid_w, mid_h = w // 2, h // 2
        
        tiles = [
            image.crop((0, 0, mid_w, mid_h)),           # Top-Left
            image.crop((mid_w, 0, w, mid_h)),           # Top-Right
            image.crop((0, mid_h, mid_w, h)),           # Bottom-Left
            image.crop((mid_w, mid_h, w, h))            # Bottom-Right
        ]
        return tiles
    
    def _stitch_tiles(self, tiles):
        """
        Reconstruct image from 4 tiles back into original dimensions.
        tiles: [TL, TR, BL, BR] as PIL Images
        """
        # Get tile dimensions (all tiles should be same size)
        tile_w, tile_h = tiles[0].size
        
        # Create canvas
        full_w = tile_w * 2
        full_h = tile_h * 2
        stitched = Image.new('RGB', (full_w, full_h))
        
        # Paste tiles
        stitched.paste(tiles[0], (0, 0))           # TL
        stitched.paste(tiles[1], (tile_w, 0))      # TR
        stitched.paste(tiles[2], (0, tile_h))      # BL
        stitched.paste(tiles[3], (tile_w, tile_h)) # BR
        
        return stitched
    
    def _apply_tile_transforms(self, tiles):
        """
        Apply random horizontal/vertical flips to each tile independently.
        This creates texture variation while maintaining spatial realism.
        """
        transformed_tiles = []
        for tile in tiles:
            # Random horizontal flip
            if random.random() > 0.5:
                tile = TF.hflip(tile)
            # Random vertical flip
            if random.random() > 0.5:
                tile = TF.vflip(tile)
            transformed_tiles.append(tile)
        return transformed_tiles
    
    def __getitem__(self, idx):
        # Decode augmentation strategy from index
        if self.mode == 'training':
            base_idx = idx // 6  # Which original sample
            aug_type = idx % 6    # Which augmentation: 0=original, 1=stitch, 2-5=tiles
        else:
            base_idx = idx
            aug_type = 0  # Always use original for validation/holdout
        
        row = self.df.iloc[base_idx]
        img_path = row['image_path']
        
        # Load base image
        image = self._load_image(img_path)
        
        # Determine augmentation strategy
        use_tiling = (self.mode == 'training' and 
                      aug_type > 0 and 
                      random.random() < self.tile_prob)
        
        # ===== AUGMENTATION LOGIC =====
        if not use_tiling or aug_type == 0:
            # MODE 0: Original Image (No Tiling)
            final_image = image
            targets_scale = 1.0
            sample_id_suffix = ""
            
        elif aug_type == 1:
            # MODE 1: STITCH (Split → Transform → Reconstruct)
            tiles = self._split_image_into_tiles(image)
            tiles = self._apply_tile_transforms(tiles)
            final_image = self._stitch_tiles(tiles)
            targets_scale = 1.0  # Full targets
            sample_id_suffix = "_STITCH"
            
        else:
            # MODE 2: DIVIDE (Each tile becomes independent sample)
            tile_idx = aug_type - 2  # 0, 1, 2, 3
            tiles = self._split_image_into_tiles(image)
            
            # Apply transforms to all tiles (maintain consistency)
            tiles = self._apply_tile_transforms(tiles)
            
            # Use specific tile
            final_image = tiles[tile_idx]
            targets_scale = 0.25  # Each tile has 1/4 of total biomass
            sample_id_suffix = f"_TILE{tile_idx}"
        
        # Calculate HSV Green Score before normalization/tensorization
        hsv_score = self._get_green_mask_count(final_image, sample_id=row['sample_id'] + sample_id_suffix)

        # Apply downstream transforms (rotation, color jitter, normalization)
        if self.transform:
            final_transformed_image = self.transform(final_image)
        else:
            raise ValueError("No transform provided.")
        
        # Handle test mode
        if self.is_test:
            return {
                'image': final_transformed_image, 
                'sample_id': row['sample_id'] + sample_id_suffix,
                'hsv_score': hsv_score
            }
        
        # ===== TARGETS & FEATURES =====
        # Scale targets based on augmentation type
        raw_targets = row[self.target_cols].values.astype(np.float32)
        scaled_targets = raw_targets * targets_scale
        targets = torch.tensor(scaled_targets)
        
        # Auxiliary features + HSV Green Score
        hsv_score = self._get_green_mask_count(final_image, sample_id=row['sample_id'] + sample_id_suffix)
        aux_values = np.append(row[self.aux_cols].values.astype(np.float32), [hsv_score])
        aux_feats = torch.tensor(np.nan_to_num(aux_values), dtype=torch.float32)
        
        # Species vector (unchanged by tiling)
        species_vec = torch.tensor(row[self.species_cols].values.astype(np.float32))
        
        return {
            'image': final_transformed_image,
            'targets': targets,
            'aux_feats': aux_feats,
            'species_id': species_vec,
            'sample_id': row['sample_id'] + sample_id_suffix,
            'is_tiled': use_tiling,
            'tile_scale': targets_scale,
            'hsv_score': hsv_score
        }

    def _get_green_mask_count(self, image, sample_id=None):
        """
        Calculates green pixel percentage using HSV color space.
        Uses shared logic from common.py
        """
        img_np = np.array(image)
        _, hsv_score = get_hsv_green_mask(img_np)
        return hsv_score


class TiledMixupDataset(Dataset):
    """
    MixUp wrapper for TiledBiomassDataset.
    Applies texture blending AFTER tiling augmentation.
    """
    def __init__(self, dataset, prob=0.5, alpha=0.4):
        self.dataset = dataset
        self.prob = prob
        self.alpha = alpha
        self.indices = list(range(len(dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Coin flip: Apply MixUp?
        if np.random.rand() >= self.prob:
            return self.dataset[idx]
        
        # Select partner
        idx2 = np.random.choice(self.indices)
        sample1 = self.dataset[idx]
        sample2 = self.dataset[idx2]
        
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_img = (lam * sample1['image'] + (1 - lam) * sample2['image']).to(torch.float32)
        
        # Mix targets (handles tiled targets correctly)
        mixed_targets = (lam * sample1['targets'] + (1 - lam) * sample2['targets']).to(torch.float32)
        
        # Mix auxiliary features
        mixed_aux = (lam * sample1['aux_feats'] + (1 - lam) * sample2['aux_feats']).to(torch.float32)
        mixed_species = (lam * sample1['species_id'] + (1 - lam) * sample2['species_id']).to(torch.float32)
        
        # Mix HSV Green Score
        mixed_hsv = float(lam * sample1.get('hsv_score', 0.0) + (1 - lam) * sample2.get('hsv_score', 0.0))

        return {
            'image': mixed_img,
            'targets': mixed_targets,
            'aux_feats': mixed_aux,
            'species_id': mixed_species,
            'sample_id': f"{sample1['sample_id']}_MIX_{sample2['sample_id']}",
            'is_tiled': sample1.get('is_tiled', False) or sample2.get('is_tiled', False),
            'tile_scale': (sample1.get('tile_scale', 1.0) + sample2.get('tile_scale', 1.0)) / 2,
            'hsv_score': mixed_hsv
        }


# ============================================================================
# BACKWARD COMPATIBILITY: Drop-in replacements for old dataset classes
# ============================================================================

class BiomassDataset(TiledBiomassDataset):
    """Legacy wrapper - automatically disables tiling for validation/holdout."""
    def __init__(self, df, transform=None, target_cols=None, aux_cols=None, is_test=False):
        # Auto-detect mode: If transform has augmentation, assume training
        mode = 'validation'  # Default to safe mode
        super().__init__(
            df=df, 
            transform=transform, 
            target_cols=target_cols, 
            aux_cols=aux_cols,
            is_test=is_test,
            mode=mode,
            tile_prob=0.0  # Disable tiling for legacy compatibility
        )

class MixupDataset(TiledMixupDataset):
    """Legacy wrapper."""
    pass