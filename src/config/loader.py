import yaml
import torch
import os
from pathlib import Path
from .schemas import (
    Config, PreprocessingConfig, HyperparametersConfig, 
    TrainingConfig, AugmentationConfig, FeatureConfig, 
    TargetConfig, SpeciesTaxonomyConfig, UpsampleConfig, 
    SplitConfig, SeasonsConfig
)

def load_config(yaml_path: str = None) -> Config:
    if yaml_path is None:
        # Default to the config.yaml in the same directory as this file
        yaml_path = Path(__file__).parent / "config.yaml"
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    config = Config(
        preprocessing=PreprocessingConfig(**data['preprocessing']),
        hyperparameters=HyperparametersConfig(**data['hyperparameters']),
        training=TrainingConfig(**data['training']),
        augmentation=AugmentationConfig(**data['augmentation']),
        features=FeatureConfig(**data['features']),
        targets=TargetConfig(**data['targets']),
        species_taxonomy=SpeciesTaxonomyConfig(**data['species_taxonomy']),
        upsample=UpsampleConfig(**data['upsample']),
        split=SplitConfig(**data['split']),
        seasons=SeasonsConfig(**data['seasons']),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return config

# Singleton instance
cfg = load_config()
