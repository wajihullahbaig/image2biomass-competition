from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

@dataclass
class PreprocessingConfig:
    imagenet_mean: Tuple[float, float, float]
    imagenet_std: Tuple[float, float, float]
    image_height: int
    image_width: int

@dataclass
class HyperparametersConfig:
    batch_size: int
    learning_rate: float
    n_folds: int
    epochs: int
    weight_decay: float
    early_stop_patience: int
    backbone: str
    min_train_samples: int
    backbone_freeze_threshold: int
    max_grad_norm: float
    backbone_lr_factor: float

@dataclass
class TrainingConfig:
    freeze_backbone: bool
    backbone_freeze_fraction: float
    use_tta: bool
    fusion_dim: int
    biomass_feat_weight: float
    aux_feat_weight: float
    species_feat_weight: float
    taxonomy_feat_weight: float
    physics_feat_weight: float

@dataclass
class AugmentationConfig:
    tile_prob: float
    mixup_prob: float
    mixup_alpha: float

@dataclass
class FeatureConfig:
    use_bin_features: bool
    bin_encoding: str
    use_species_count_feature: bool

@dataclass
class TargetConfig:
    cols: List[str]
    official_weights: List[float]
    biomass_clamp: float

@dataclass
class SpeciesTaxonomyConfig:
    core_species: List[str]
    groups: Dict[str, List[str]]
    
    @property
    def taxonomy_idxs(self) -> Dict[str, List[int]]:
        return {
            group: [i for i, species in enumerate(self.core_species) if species in names]
            for group, names in self.groups.items()
        }

@dataclass
class UpsampleConfig:
    enabled: bool
    target_min_samples: int
    method: str
    noise_scale: float
    seasonal_drift: bool
    day_shift_prob: float
    drift_strength: float

@dataclass
class SplitConfig:
    holdout_pct: float
    sparse_threshold: int
    small_threshold: int

@dataclass
class SeasonsConfig:
    month_map: Dict[int, str]
    drift: Dict[str, Dict[str, float]]

@dataclass
class Config:
    preprocessing: PreprocessingConfig
    hyperparameters: HyperparametersConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    features: FeatureConfig
    targets: TargetConfig
    species_taxonomy: SpeciesTaxonomyConfig
    upsample: UpsampleConfig
    split: SplitConfig
    seasons: SeasonsConfig
    device: str = "cpu" # Default, will be updated during loading
