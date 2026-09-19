# loader.py - Clean Configuration Loader
import yaml
import torch
from pathlib import Path
from .schemas import (
    Config, PreprocessingConfig, HyperparametersConfig,
    TrainingConfig, AugmentationConfig, LossConfig,
    TargetConfig, SplitConfig
)

def load_config(yaml_path: str = None) -> tuple:
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "config.yaml"
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    config = Config(
        preprocessing=PreprocessingConfig(**data.get('preprocessing', {})),
        hyperparameters=HyperparametersConfig(**data.get('hyperparameters', {})),
        training=TrainingConfig(**data.get('training', {})),
        augmentation=AugmentationConfig(**data.get('augmentation', {})),
        loss=LossConfig(**data.get('loss', {})),
        targets=TargetConfig(**data.get('targets', {})),
        split=SplitConfig(**data.get('split', {})),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return config, yaml_path

cfg, yaml_path = load_config()
