import yaml
import torch
import os
from pathlib import Path
from .schemas import (
    Config, PreprocessingConfig, HyperparametersConfig, 
    TrainingConfig, AugmentationConfig, FeatureConfig, LossConfig,
    TargetConfig, SpeciesTaxonomyConfig, UpsampleConfig, 
    SplitConfig, SeasonsConfig, HSVBiomassConfig, HSVMatterConfig, IntelligentTilingConfig
)

def load_config(yaml_path: str = None) -> Config:
    if yaml_path is None:
        # Default to the config.yaml in the same directory as this file
        yaml_path = Path(__file__).parent / "config.yaml"
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle HSV biomass configuration if present
    features_data = data.get('features', {})
    if 'hsv_biomass_scores' in features_data:
        hsv_data = features_data['hsv_biomass_scores']
        
        # Build HSVMatterConfig objects for each matter type
        def build_hsv_matter_config(matter_data):
            return HSVMatterConfig(
                hue_range=matter_data.get('hue_range'),
                saturation_min=matter_data.get('saturation_min'),
                saturation_max=matter_data.get('saturation_max'),
                saturation_range=matter_data.get('saturation_range'),
                value_min=matter_data.get('value_min'),
                value_range=matter_data.get('value_range')
            )
        
        # Build IntelligentTilingConfig
        tiling_data = hsv_data.get('intelligent_tiling', {})
        intelligent_tiling = IntelligentTilingConfig(
            enabled=tiling_data.get('enabled', True),
            hsv_blend_factor=tiling_data.get('hsv_blend_factor', 0.7),
            min_tile_weight=tiling_data.get('min_tile_weight', 0.05),
            max_tile_weight=tiling_data.get('max_tile_weight', 0.8)
        )
        
        # Build HSVBiomassConfig
        hsv_biomass_config = HSVBiomassConfig(
            enabled=hsv_data.get('enabled', True),
            green_vegetation=build_hsv_matter_config(hsv_data.get('green_vegetation', {})),
            dry_green_vegetation=build_hsv_matter_config(hsv_data.get('dry_green_vegetation', {})),
            clover=build_hsv_matter_config(hsv_data.get('clover', {})),
            dead_matter=build_hsv_matter_config(hsv_data.get('dead_matter', {})),
            soil=build_hsv_matter_config(hsv_data.get('soil', {})),
            intelligent_tiling=intelligent_tiling
        )
        
        # Remove nested config from features data and add the processed config
        features_data_clean = {k: v for k, v in features_data.items() if k != 'hsv_biomass_scores'}
        features_data_clean['hsv_biomass_scores'] = hsv_biomass_config
    else:
        features_data_clean = features_data

    config = Config(
        preprocessing=PreprocessingConfig(**data['preprocessing']),
        hyperparameters=HyperparametersConfig(**data['hyperparameters']),
        training=TrainingConfig(**data['training']),
        augmentation=AugmentationConfig(**data['augmentation']),
        features=FeatureConfig(**features_data_clean),
        loss=LossConfig(**data['loss']),
        targets=TargetConfig(**data['targets']),
        species_taxonomy=SpeciesTaxonomyConfig(**data['species_taxonomy']),
        upsample=UpsampleConfig(**data['upsample']),
        split=SplitConfig(**data['split']),
        seasons=SeasonsConfig(**data['seasons']),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return config,yaml_path

# Singleton instance
cfg,yaml_path = load_config()
