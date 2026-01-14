from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any
import json

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
    biomass_composite_bins: int

@dataclass
class LossConfig:
    use_standardized_loss: bool
    use_weighted_regression_loss: bool
    reg_loss_type: str

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
    stratification_key: str  # Key for coverage-aware splitting (e.g., 'Species_Season')
    min_train_per_combo: int = 2  # Minimum samples per combo guaranteed in training
    # Coverage enforcement flags
    ensure_species_train_coverage: bool = True
    species_col: str = "Species"
    min_train_per_species: int = 1
    ensure_combo_train_coverage: bool = True
    combo_col: str = "Season_State_Species"
    min_train_per_combo_key: int = 1

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
    loss: LossConfig
    targets: TargetConfig
    species_taxonomy: SpeciesTaxonomyConfig
    upsample: UpsampleConfig
    split: SplitConfig
    seasons: SeasonsConfig
    device: str = "cpu" # Default, will be updated during loading

    def __str__(self):
        return json.dumps(asdict(self), indent=4)
