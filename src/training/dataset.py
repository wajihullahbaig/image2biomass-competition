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
    get_hsv_green_mask, get_hsv_biomass_scores, SubtleSharpen
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
            final_image = image
            targets_scale = 1.0
            sample_id_suffix = ""
            
            # Use PRE-COMPUTED scores from dataframe if available (Priority 1)
            # This reduces redundant compute and ensures consistency
            if all(f'hsv_{k}_score' in row.index for k in ['green', 'dry_green', 'clover', 'dead', 'soil']):
                biomass_scores = {
                    'green_score': row['hsv_green_score'],
                    'dry_green_score': row['hsv_dry_green_score'],
                    'clover_score': row['hsv_clover_score'],
                    'dead_score': row['hsv_dead_score'],
                    'soil_score': row['hsv_soil_score']
                }
            else:
                # Calculate biomass scores for the whole image (Fallback)
                biomass_scores = self._get_biomass_scores(final_image, sample_id=row['sample_id'] + sample_id_suffix)
            
        elif aug_type == 1:
            # MODE 1: STITCH (Split → Transform → Reconstruct)
            tiles = self._split_image_into_tiles(image)
            tiles = self._apply_tile_transforms(tiles)
            final_image = self._stitch_tiles(tiles)
            targets_scale = 1.0  # Full targets
            sample_id_suffix = "_STITCH"
            # Calculate biomass scores for the stitched image
            biomass_scores = self._get_biomass_scores(final_image, sample_id=row['sample_id'] + sample_id_suffix)
            
        else:
            # MODE 2: DIVIDE (Each tile becomes independent sample)
            tile_idx = aug_type - 2  # 0, 1, 2, 3
            tiles = self._split_image_into_tiles(image)
            
            # Apply transforms to all tiles (maintain consistency)
            tiles = self._apply_tile_transforms(tiles)
            
            # Calculate biomass scores for ALL tiles to enable intelligent scaling
            tile_scores = []
            for tile in tiles:
                tile_biomass_scores = self._get_biomass_scores(tile)
                tile_scores.append(tile_biomass_scores)
            
            # Use specific tile
            final_image = tiles[tile_idx]
            biomass_scores = tile_scores[tile_idx]
            
            # TILE SHARPENING: Apply slightly more aggressive sharpening to individual tiles
            # to recover details lost during the crop/resize process.
            tile_sharpener = SubtleSharpen(probability=1.0, radius=1, percent=130, threshold=1)
            final_image = tile_sharpener(final_image)
            
            # INTELLIGENT TILING: Use HSV scores to weight targets appropriately
            targets_scale = self._calculate_intelligent_tile_scale(tile_scores, tile_idx, row[self.target_cols])
            sample_id_suffix = f"_TILE{tile_idx}"
        
        # Legacy green score for compatibility
        hsv_score = biomass_scores['green_score']

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
                'hsv_score': hsv_score,
                'green_score': biomass_scores['green_score'],
                'dry_green_score': biomass_scores['dry_green_score'],
                'clover_score': biomass_scores['clover_score'],
                'dead_score': biomass_scores['dead_score'],
                'soil_score': biomass_scores['soil_score']
            }
        
        # ===== TARGETS & FEATURES =====
        # Scale targets based on augmentation type
        raw_targets = row[self.target_cols].values.astype(np.float32)
        
        # Apply scaling - either uniform or intelligent
        if isinstance(targets_scale, dict):
            # Intelligent scaling per target type
            scaled_targets = np.array([
                raw_targets[i] * targets_scale[self.target_cols[i]] 
                for i in range(len(self.target_cols))
            ])
        else:
            # Uniform scaling (original/stitch mode)
            scaled_targets = raw_targets * targets_scale
            
        targets = torch.tensor(scaled_targets)
        
        # Auxiliary features + All 5 HSV Biomass Scores
        hsv_features = [
            biomass_scores['green_score'], biomass_scores['dry_green_score'],
            biomass_scores['clover_score'], biomass_scores['dead_score'], 
            biomass_scores['soil_score']
        ]
        aux_values = np.append(row[self.aux_cols].values.astype(np.float32), hsv_features)
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
            'tile_scale': targets_scale if not isinstance(targets_scale, dict) else np.mean(list(targets_scale.values())),
            'hsv_score': hsv_score,
            'green_score': biomass_scores['green_score'],
            'dry_green_score': biomass_scores['dry_green_score'],
            'clover_score': biomass_scores['clover_score'],
            'dead_score': biomass_scores['dead_score'],
            'soil_score': biomass_scores['soil_score']
        }

    def _calculate_intelligent_tile_scale(self, tile_scores, selected_tile_idx, raw_targets):
        """
        Calculate intelligent scaling factors for tiled targets based on HSV biomass scores.
        
        The idea: Instead of naive 1/4 scaling, weight each tile's targets based on its
        relative content of different matter types.
        
        Args:
            tile_scores: List of 4 biomass score dicts (one per tile)
            selected_tile_idx: Index of the tile we're using (0-3)
            raw_targets: Array of target values [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
            
        Returns:
            Dictionary of scaling factors for each target
        """
        # Extract scores for each tile
        green_scores = [scores['green_score'] for scores in tile_scores]
        dry_green_scores = [scores['dry_green_score'] for scores in tile_scores]
        clover_scores = [scores['clover_score'] for scores in tile_scores]
        dead_scores = [scores['dead_score'] for scores in tile_scores]
        soil_scores = [scores['soil_score'] for scores in tile_scores]
        
        # Normalize so that sum across tiles = 1.0 for each matter type
        def safe_normalize(scores):
            total = sum(scores)
            if total > 0:
                return [s / total for s in scores]
            else:
                return [0.25, 0.25, 0.25, 0.25]  # Fall back to equal split
        
        green_weights = safe_normalize(green_scores)
        dry_green_weights = safe_normalize(dry_green_scores)
        dead_weights = safe_normalize(dead_scores)
        clover_weights = safe_normalize(clover_scores)
        
        # For GDM (Green Dry Matter), combine green, dry green, and clover
        combined_green_clover = [g + dg + c for g, dg, c in zip(green_scores, dry_green_scores, clover_scores)]
        gdm_weights = safe_normalize(combined_green_clover)
        
        # For Dry_Total, use a weighted combination of all vegetation types
        total_veg_scores = [g + dg + c + d for g, dg, c, d in zip(green_scores, dry_green_scores, clover_scores, dead_scores)]
        total_weights = safe_normalize(total_veg_scores)
        
        # Apply weights to the selected tile
        selected_tile_weights = {
            'Dry_Green_g': green_weights[selected_tile_idx],
            'Dry_Dead_g': dead_weights[selected_tile_idx], 
            'Dry_Clover_g': clover_weights[selected_tile_idx],
            'GDM_g': gdm_weights[selected_tile_idx],
            'Dry_Total_g': total_weights[selected_tile_idx]
        }
        
        # Handle edge cases: if weights are too extreme, blend with uniform scaling
        uniform_weight = 0.25
        blend_factor = 0.7  # How much to trust HSV vs uniform scaling
        
        final_weights = {}
        for target_name, hsv_weight in selected_tile_weights.items():
            # Blend HSV-based weight with uniform weight
            blended_weight = blend_factor * hsv_weight + (1 - blend_factor) * uniform_weight
            # Clamp to reasonable bounds [0.05, 0.8] to prevent extreme values
            final_weights[target_name] = max(0.05, min(0.8, blended_weight))
        
        return final_weights

    def _get_green_mask_count(self, image, sample_id=None):
        """
        Calculates green pixel percentage using HSV color space.
        Uses shared logic from common.py
        
        DEPRECATED: Use _get_biomass_scores for comprehensive analysis
        """
        img_np = np.array(image)
        _, hsv_score = get_hsv_green_mask(img_np)
        return hsv_score
    
    def _get_biomass_scores(self, image, sample_id=None):
        """
        Calculate comprehensive HSV-based biomass matter scores.
        Returns dict with green, dead, dry_clover, and soil scores.
        """
        img_np = np.array(image)
        biomass_scores = get_hsv_biomass_scores(img_np)
        
        return {
            'green_score': biomass_scores['green_score'],
            'dry_green_score': biomass_scores['dry_green_score'],
            'clover_score': biomass_scores['clover_score'],
            'dead_score': biomass_scores['dead_score'], 
            'soil_score': biomass_scores['soil_score']
        }


class TiledMixupDataset(Dataset):
    """
    MixUp and CutMix wrapper for TiledBiomassDataset.
    Applies texture blending or cutting AFTER tiling augmentation.
    """
    def __init__(self, dataset, prob=0.5, alpha=0.4, use_cutmix=False):
        self.dataset = dataset
        self.prob = prob
        self.alpha = alpha
        self.use_cutmix = use_cutmix
        self.indices = list(range(len(dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Coin flip: Apply MixUp/CutMix?
        if np.random.rand() >= self.prob:
            return self.dataset[idx]
        
        # Select partner
        idx2 = np.random.choice(self.indices)
        sample1 = self.dataset[idx]
        sample2 = self.dataset[idx2]
        
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        if self.use_cutmix:
            # --- CutMix Logic ---
            mixed_img = sample1['image'].clone()
            W, H = sample1['image'].shape[1], sample1['image'].shape[2]
            
            # Draw random box coordinates
            r_x = np.random.randint(W)
            r_y = np.random.randint(H)
            r_w = int(W * np.sqrt(1 - lam))
            r_h = int(H * np.sqrt(1 - lam))
            
            # Clip bounds
            x1 = np.clip(r_x - r_w // 2, 0, W)
            y1 = np.clip(r_y - r_h // 2, 0, H)
            x2 = np.clip(r_x + r_w // 2, 0, W)
            y2 = np.clip(r_y + r_h // 2, 0, H)
            
            # Patch from image 2 into image 1
            mixed_img[:, x1:x2, y1:y2] = sample2['image'][:, x1:x2, y1:y2]
            
            # Adjust lambda to be the actual pixel ratio
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        else:
            # --- Standard MixUp ---
            mixed_img = (lam * sample1['image'] + (1 - lam) * sample2['image']).to(torch.float32)
        
        # Mix targets (handles tiled targets correctly)
        mixed_targets = (lam * sample1['targets'] + (1 - lam) * sample2['targets']).to(torch.float32)
        
        # Mix auxiliary features
        mixed_aux = (lam * sample1['aux_feats'] + (1 - lam) * sample2['aux_feats']).to(torch.float32)
        mixed_species = (lam * sample1['species_id'] + (1 - lam) * sample2['species_id']).to(torch.float32)
        
        # Mix HSV Green Score and comprehensive biomass scores
        mixed_hsv = float(lam * sample1.get('hsv_score', 0.0) + (1 - lam) * sample2.get('hsv_score', 0.0))
        
        # Mix individual biomass scores
        mixed_green_score = float(lam * sample1.get('green_score', 0.0) + (1 - lam) * sample2.get('green_score', 0.0))
        mixed_dry_green_score = float(lam * sample1.get('dry_green_score', 0.0) + (1 - lam) * sample2.get('dry_green_score', 0.0))
        mixed_clover_score = float(lam * sample1.get('clover_score', 0.0) + (1 - lam) * sample2.get('clover_score', 0.0))
        mixed_dead_score = float(lam * sample1.get('dead_score', 0.0) + (1 - lam) * sample2.get('dead_score', 0.0))
        mixed_soil_score = float(lam * sample1.get('soil_score', 0.0) + (1 - lam) * sample2.get('soil_score', 0.0))

        return {
            'image': mixed_img,
            'targets': mixed_targets,
            'aux_feats': mixed_aux,
            'species_id': mixed_species,
            'sample_id': f"{sample1['sample_id']}_MIX_{sample2['sample_id']}",
            'is_tiled': sample1.get('is_tiled', False) or sample2.get('is_tiled', False),
            'tile_scale': (sample1.get('tile_scale', 1.0) + sample2.get('tile_scale', 1.0)) / 2,
            'hsv_score': mixed_hsv,
            'green_score': mixed_green_score,
            'dry_green_score': mixed_dry_green_score,
            'clover_score': mixed_clover_score,
            'dead_score': mixed_dead_score,
            'soil_score': mixed_soil_score
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