# dataset.py
# dataset_tiled.py - Enhanced Dataset with Tile-Based Augmentation
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

# Local Imports
from common import CORE_SPECIES

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
        self.target_cols = target_cols or ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Interaction_Add']
        self.is_test = is_test
        self.mode = mode
        self.tile_prob = tile_prob
        
        self.core_species = CORE_SPECIES
        self.species_cols = [f'Species_{sp}' for sp in self.core_species]
        
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
        
        # Apply downstream transforms (rotation, color jitter, normalization)
        if self.transform:
            final_image = self.transform(final_image)
        else:
            raise ValueError("No transform provided.")
        
        # Handle test mode
        if self.is_test:
            return {
                'image': final_image, 
                'sample_id': row['sample_id'] + sample_id_suffix
            }
        
        # ===== TARGETS & FEATURES =====
        # Scale targets based on augmentation type
        raw_targets = row[self.target_cols].values.astype(np.float32)
        scaled_targets = raw_targets * targets_scale
        targets = torch.tensor(scaled_targets)
        
        # Auxiliary features (unchanged by tiling)
        aux_values = row[self.aux_cols].values
        aux_feats = torch.tensor(np.nan_to_num(aux_values.astype(np.float32)))
        
        # Species vector (unchanged by tiling)
        species_vec = torch.tensor(row[self.species_cols].values.astype(np.float32))
        
        return {
            'image': final_image,
            'targets': targets,
            'aux_feats': aux_feats,
            'species_id': species_vec,
            'sample_id': row['sample_id'] + sample_id_suffix,
            'is_tiled': use_tiling,
            'tile_scale': targets_scale
        }


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
        mixed_img = lam * sample1['image'] + (1 - lam) * sample2['image']
        
        # Mix targets (handles tiled targets correctly)
        mixed_targets = lam * sample1['targets'] + (1 - lam) * sample2['targets']
        
        # Mix auxiliary features
        mixed_aux = lam * sample1['aux_feats'] + (1 - lam) * sample2['aux_feats']
        mixed_species = lam * sample1['species_id'] + (1 - lam) * sample2['species_id']
        
        return {
            'image': mixed_img,
            'targets': mixed_targets,
            'aux_feats': mixed_aux,
            'species_id': mixed_species,
            'sample_id': f"{sample1['sample_id']}_MIX_{sample2['sample_id']}",
            'is_tiled': sample1.get('is_tiled', False) or sample2.get('is_tiled', False),
            'tile_scale': (sample1.get('tile_scale', 1.0) + sample2.get('tile_scale', 1.0)) / 2
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