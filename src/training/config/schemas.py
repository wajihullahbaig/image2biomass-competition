# schemas.py - Configuration Schemas for Dual-Stream DINO Pipeline
from dataclasses import dataclass
from typing import List

@dataclass
class PreprocessingConfig:
    image_height: int = 512
    image_width: int = 512
    dual_stream: bool = True

@dataclass
class HyperparametersConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0003
    n_folds: int = 5
    epochs: int = 35
    weight_decay: float = 0.05
    early_stop_patience: int = 8
    backbone: str = "vit_base_patch16_dinov3_qkvb"
    max_grad_norm: float = 1.0
    random_seed: int = 42

@dataclass
class TrainingConfig:
    stage1_epochs: int = 14
    stage2_epochs: int = 16
    stage3_epochs: int = 5
    stage2_backbone_lr_factor: float = 0.1
    stage3_lr_factor: float = 0.1
    fusion_dim: int = 384
    dropout: float = 0.3
    use_tta: bool = True

@dataclass
class AugmentationConfig:
    camera_scaling_prob: float = 0.2
    strip_shuffle_prob: float = 0.5
    view_swap_prob: float = 0.5

@dataclass
class LossConfig:
    reg_loss_type: str = "smoothl1"
    num_intervals: int = 7
    cls_weight: float = 0.3

@dataclass
class TargetConfig:
    cols: List[str]
    official_weights: List[float]
    all_cols: List[str] = None

@dataclass
class SplitConfig:
    n_splits: int = 5
    group_col: str = "State_Sampling_Date"
    group_stratification_col: str = "State_Species"

@dataclass
class Config:
    preprocessing: PreprocessingConfig
    hyperparameters: HyperparametersConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    loss: LossConfig
    targets: TargetConfig
    split: SplitConfig
    device: str = "cuda"
