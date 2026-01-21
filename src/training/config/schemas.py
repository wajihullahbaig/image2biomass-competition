from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
import json

@dataclass
class HSVMatterConfig:
    """Configuration for HSV ranges of different biomass matter types"""
    hue_range: Optional[List[int]] = None
    saturation_min: Optional[int] = None
    saturation_max: Optional[int] = None
    saturation_range: Optional[List[int]] = None
    value_min: Optional[int] = None
    value_range: Optional[List[int]] = None

@dataclass
class IntelligentTilingConfig:
    """Configuration for intelligent HSV-based tiling"""
    enabled: bool = True
    hsv_blend_factor: float = 0.7
    min_tile_weight: float = 0.05
    max_tile_weight: float = 0.8

@dataclass
class HSVBiomassConfig:
    """Configuration for HSV-based biomass detection"""
    enabled: bool = True
    green_vegetation: Optional[HSVMatterConfig] = None
    dead_matter: Optional[HSVMatterConfig] = None
    dry_clover: Optional[HSVMatterConfig] = None
    soil: Optional[HSVMatterConfig] = None
    intelligent_tiling: Optional[IntelligentTilingConfig] = None

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
    overfitting_penalty_alpha: float = 0.5
    # Exponential moving average decay for smoothing scheduler score (0..1)
    # Higher values -> smoother (less responsive). Default: 0.9
    ema_decay: float = 0.9

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
    hsv_biomass_scores: Optional[HSVBiomassConfig] = None
    hsv_biomass_scores: Optional[HSVBiomassConfig] = None

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
    min_train_per_combo_key: int = 2
    ensure_holdout_coverage: bool = True
    min_holdout_per_species: int = 1
    min_holdout_per_combo: int = 1
    group_stratification_col: str = "State"
    group_col: str = "Sampling_Date"

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
