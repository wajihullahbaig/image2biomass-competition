# configs.py
from config.loader import cfg

# ====================== CONFIG ======================
# CONFIGURATIONS
DEVICE = cfg.device

# Preprocessing
IMAGENET_DEFAULT_MEAN = cfg.preprocessing.imagenet_mean
IMAGENET_DEFAULT_STD = cfg.preprocessing.imagenet_std
IMAGE_HEIGHT = cfg.preprocessing.image_height
IMAGE_WIDTH = cfg.preprocessing.image_width

# Training Hyperparameters
BATCH_SIZE = cfg.hyperparameters.batch_size
LEARNING_RATE = cfg.hyperparameters.learning_rate
N_FOLDS = cfg.hyperparameters.n_folds
EPOCHS = cfg.hyperparameters.epochs
WEIGHT_DECAY = cfg.hyperparameters.weight_decay
EARLY_STOP_PATIENCE = cfg.hyperparameters.early_stop_patience
BACKBONE = cfg.hyperparameters.backbone
MIN_TRAIN_SAMPLES = cfg.hyperparameters.min_train_samples
BACKBONE_FREEZE_THRESHOLD = cfg.hyperparameters.backbone_freeze_threshold
MAX_GRAD_NORM = cfg.hyperparameters.max_grad_norm
BACKBONE_LR_FACTOR = cfg.hyperparameters.backbone_lr_factor

# Use Official Weights for Loss Calculation
TARGET_COLS = cfg.targets.cols
OFFICIAL_WEIGHTS = cfg.targets.official_weights

# training settings
FREEZE_BACKBONE = cfg.training.freeze_backbone
BACKBONE_FREEZE_FRACTION = cfg.training.backbone_freeze_fraction
USE_TTA = cfg.training.use_tta
FUSION_DIM = cfg.training.fusion_dim
BIOMASS_FEAT_WEIGHT = cfg.training.biomass_feat_weight
AUX_FEAT_WEIGHT = cfg.training.aux_feat_weight
SPECIES_FEAT_WEIGHT = cfg.training.species_feat_weight
TAXONOMY_FEAT_WEIGHT = cfg.training.taxonomy_feat_weight

# --- Augmentation & Tiling Settings ---
TILE_PROB = cfg.augmentation.tile_prob
MIXUP_PROB = cfg.augmentation.mixup_prob
MIXUP_ALPHA = cfg.augmentation.mixup_alpha

# --- Feature Engineering Toggles ---
USE_BIN_FEATURES = cfg.features.use_bin_features
BIN_ENCODING = cfg.features.bin_encoding
USE_SPECIES_COUNT_FEATURE = cfg.features.use_species_count_feature


# ====================== SPECIES & TAXONOMY ======================
CORE_SPECIES = cfg.species_taxonomy.core_species
GROUP_DEFINITIONS = cfg.species_taxonomy.groups
TAXONOMY_IDXS = cfg.species_taxonomy.taxonomy_idxs

# ====================== IMPROVED STRATIFICATION ======================
def get_key1_specie_pair(row, key1='State',flip=False) -> str:
    """
    Create region-aware stratification key: State + Primary Species Pattern.
    """
    col1 = row[key1]
    species = str(row['Species']).lower().replace(' ', '')
    
    dominant = None
    if 'phalaris' in species:
        dominant = 'Phalaris'
    elif 'ryegrass' in species:
        dominant = 'Ryegrass'
    elif 'fescue' in species:
        dominant = 'Fescue'
    elif 'lucerne' in species:
        dominant = 'Lucerne'
    elif 'clover' in species or 'whiteclover' in species:
        dominant = 'Clover'
    elif 'mixed' in species:
        dominant = 'Mixed'
    else:
        dominant = 'Other'

    if flip:
        return f"{dominant}_{col1}"
    return f"{col1}_{dominant}"

# ===== UPSAMPLING STRATEGY =====
UPSAMPLE_CONFIG = {
    'enabled': cfg.upsample.enabled,
    'target_min_samples': cfg.upsample.target_min_samples,
    'method': cfg.upsample.method,
    'noise_scale': cfg.upsample.noise_scale,
    'seasonal_drift': cfg.upsample.seasonal_drift,
    'day_shift_prob': cfg.upsample.day_shift_prob,
    'drift_strength': cfg.upsample.drift_strength
}

# ===== TEMPORAL SPLIT STRATEGY =====
SPLIT_CONFIG = {
    'holdout_pct': cfg.split.holdout_pct,
    'sparse_threshold': cfg.split.sparse_threshold,
    'small_threshold': cfg.split.small_threshold,
}

# ====================== AUSTRALIAN SEASONS ======================
SEASON_MONTH_MAP = cfg.seasons.month_map
SEASONAL_DRIFT = cfg.seasons.drift

# return a string representation of the configuration
def config_str():
    config_items = [
        f"DEVICE: {DEVICE}",
        f"IMAGE_HEIGHT: {IMAGE_HEIGHT}",
        f"IMAGE_WIDTH: {IMAGE_WIDTH}",
        f"BATCH_SIZE: {BATCH_SIZE}",
        f"LEARNING_RATE: {LEARNING_RATE}",
        f"N_FOLDS: {N_FOLDS}",
        f"EPOCHS: {EPOCHS}",
        f"WEIGHT_DECAY: {WEIGHT_DECAY}",
        f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}",  
        f"BACKBONE: {BACKBONE}",
        f"TARGET_COLS: {TARGET_COLS}",
        f"OFFICIAL_WEIGHTS: {OFFICIAL_WEIGHTS}",
        f"FREEZE_BACKBONE: {FREEZE_BACKBONE}",
        f"BACKBONE_FREEZE_FRACTION: {BACKBONE_FREEZE_FRACTION}",
        f"USE_TTA: {USE_TTA}",
        f"TILE_PROB: {TILE_PROB}",
        f"MIXUP_PROB: {MIXUP_PROB}",
        f"MIXUP_ALPHA: {MIXUP_ALPHA}",
        f"FUSION_DIM: {FUSION_DIM}",
        f"AUX_FEAT_WEIGHT: {AUX_FEAT_WEIGHT}",
        f"SPECIES_FEAT_WEIGHT: {SPECIES_FEAT_WEIGHT}",
        f"TAXONOMY_FEAT_WEIGHT: {TAXONOMY_FEAT_WEIGHT}",
        f"BIOMASS_FEAT_WEIGHT: {BIOMASS_FEAT_WEIGHT}",
        f"STRATIFY: State + Dominant Species (Region-Aware)",
        f"UPSAMPLE: Enabled={UPSAMPLE_CONFIG['enabled']} (target={UPSAMPLE_CONFIG['target_min_samples']})",
        f"SPLIT: Adaptive temporal (sparse_threshold={SPLIT_CONFIG['sparse_threshold']})",
        f"SEASONS: AU (Summer/Autumn/Winter/Spring)",
        f"SEASONAL_DRIFT: {UPSAMPLE_CONFIG['seasonal_drift']} (strength={UPSAMPLE_CONFIG['drift_strength']})",
        f"MIN_TRAIN_SAMPLES: {MIN_TRAIN_SAMPLES}",
        f"BACKBONE_FREEZE_THRESHOLD: {BACKBONE_FREEZE_THRESHOLD}",
        f"MAX_GRAD_NORM: {MAX_GRAD_NORM}",
        f"BACKBONE_LR_FACTOR: {BACKBONE_LR_FACTOR}"
    ]
    return "\n".join(config_items)
