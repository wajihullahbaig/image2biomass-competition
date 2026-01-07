# configs.py
import torch

# ====================== CONFIG ======================
# CONFIGURATIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# We use 256x512 to respect the ~2.33 aspect ratio of the 70cm x 30cm quadrats.
# This prevents "squashing" the grass which destroys density features.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

# Training Hyperparameters
BATCH_SIZE = 32 
LEARNING_RATE = 2e-4 
N_FOLDS = 3
EPOCHS = 40 
WEIGHT_DECAY = 0.05
EARLY_STOP_PATIENCE = 10
BACKBONE = 'timm/tf_efficientnet_b3.ns_jft_in1k'  
MIN_TRAIN_SAMPLES = 50 # Skip folds with too little data
BACKBONE_FREEZE_THRESHOLD = 200 # Unfreeze earlier to allow domain adaptation
MAX_GRAD_NORM = 2.0 # Gradient clipping
BACKBONE_LR_FACTOR = 0.85 # Fine-tune backbone at the BACKBONE_LR_FACTOR*LR of LR
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 

# training settings
FREEZE_BACKBONE = False
BACKBONE_FREEZE_FRACTION = 0.7
USE_TTA = True
FUSION_DIM = 256
BIOMASS_FEAT_WEIGHT = 80.0 
AUX_FEAT_WEIGHT = 15.0
SPECIES_FEAT_WEIGHT = 20.0
TAXONOMY_FEAT_WEIGHT = 25.0
PHYSICS_FEAT_WEIGHT = 30.0 

# --- Augmentation & Tiling Settings ---
TILE_PROB = 0.25        # Prob of applying tiling (Mode 1 or 2)
MIXUP_PROB = 0.05      # Prob of applying MixUp
MIXUP_ALPHA = 0.40     # Higher alpha for more aggressive regularization on small data

# --- Feature Engineering Toggles ---
USE_BIN_FEATURES = True
BIN_ENCODING = 'ordinal'
USE_SPECIES_COUNT_FEATURE = True


# ====================== SPECIES & TAXONOMY ======================
CORE_SPECIES = [
    'clover', 'whiteclover', 'subcloverdalkeith', 'subcloverlosa',  # 0-3
    'ryegrass', 'phalaris', 'fescue', 'lucerne',                    # 4-7
    'barleygrass', 'silvergrass', 'speargrass', 'bromegrass',       # 8-11
    'capeweed', 'crumbweed'                                         # 12-13
]

# ===== FUNCTIONAL GROUPS (for model features - biological categories) =====
# These are what the MODEL learns, not what we stratify by
GROUP_DEFINITIONS = {
    'legume': ['clover', 'whiteclover', 'subcloverdalkeith', 'subcloverlosa', 'lucerne'],
    'grass':  ['ryegrass', 'phalaris', 'fescue', 'barleygrass', 'silvergrass', 'speargrass', 'bromegrass'],
    'weed':   ['capeweed', 'crumbweed']
}

# Dynamically Generate Indices Dictionary for model
# Structure: {'legume': [0, 1, 2, 3, 7], 'grass': [...], 'weed': [...]}
TAXONOMY_IDXS = {
    group: [i for i, species in enumerate(CORE_SPECIES) if species in names]
    for group, names in GROUP_DEFINITIONS.items()
}

# ====================== IMPROVED STRATIFICATION ======================
# Key insight: Different states have different species distributions
# Stratify by State+DominantSpecies instead of coarse functional groups

def get_stratify_key(row):
    """
    Create region-aware stratification key: State + Primary Species Pattern.
    
    Why this matters:
    - WA only has subclovers (sparse, 8 samples) -> needs special handling
    - Vic has complex mixtures -> needs granular splits
    - NSW/Tas have distinct monocultures -> region-specific
    
    Examples:
        (WA, SubcloverDalkeith) -> 'WA_Clover'
        (Vic, Phalaris_Ryegrass_Clover) -> 'Vic_Phalaris_Mix'
        (NSW, Fescue) -> 'NSW_Fescue'
    
    This ensures temporal splits preserve regional diversity.
    """
    state = row['State']
    species = str(row['Species']).lower().replace(' ', '')
    
    # === WA Special Case ===
    # Only 8 samples, all subclovers -> must keep in training
    if state == 'WA':
        return 'WA_Clover'
    
    # === Identify Dominant Species ===
    # Check which species appear in the name (handles mixtures)
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
    
    # === Return Composite Key ===
    # Format: State_DominantSpecies
    return f"{state}_{dominant}"

# ===== UPSAMPLING STRATEGY =====
UPSAMPLE_CONFIG = {
    'enabled': True,
    'target_min_samples': 25,  # Minimum samples per stratify key
    'method': 'smart',  # Only upsample sparse groups
    'noise_scale': 0.025,  # Add 5% noise to biomass targets (prevents overfitting)
    'seasonal_drift': True,  # Apply season-aware target adjustments
    'day_shift_prob': 0.6,   # Probability to shift ±1 day when synthesizing
    'drift_strength': 0.15   # Base amplitude for seasonal drift multipliers
}

# ===== TEMPORAL SPLIT STRATEGY =====
SPLIT_CONFIG = {
    'holdout_pct': 0.10,  # 20% holdout for groups with enough data
    'sparse_threshold': 4,  # Groups ≤4 samples: keep all in training
    'small_threshold': 6,   # Groups 5-9: take 1-2 for holdout
}

# ====================== AUSTRALIAN SEASONS ======================
# Meteorological seasons for Australia
# Summer: Dec-Feb, Autumn: Mar-May, Winter: Jun-Aug, Spring: Sep-Nov
SEASON_MONTH_MAP = {
    1: 'summer', 2: 'summer', 12: 'summer',
    3: 'autumn', 4: 'autumn', 5: 'autumn',
    6: 'winter', 7: 'winter', 8: 'winter',
    9: 'spring', 10: 'spring', 11: 'spring',
}

# Target-wise drift tendencies by season.
# Values denote directional tendency (signed) that will be scaled by UPSAMPLE_CONFIG['drift_strength']
SEASONAL_DRIFT = {
    'spring': {
        'Dry_Green_g': +1.0,   # greenness increases
        'Dry_Dead_g':  -0.4,   # dead material decreases
        'Dry_Clover_g': +0.4,  # clover often flushes in spring
    },
    'summer': {
        'Dry_Green_g': +0.4,   # early/mid-summer growth pulses
        'Dry_Dead_g':  +0.2,   # some senescence toward late summer
        'Dry_Clover_g': +0.2,
    },
    'autumn': {
        'Dry_Green_g': -0.2,   # greenness recedes
        'Dry_Dead_g':  +0.2,   # dead material builds
        'Dry_Clover_g': -0.2,
    },
    'winter': {
        'Dry_Green_g': -0.2,   # minimal green biomass
        'Dry_Dead_g':  +1.0,   # dead biomass peaks
        'Dry_Clover_g': -0.2,
    },
}

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
        f"PHYSICS_FEAT_WEIGHT: {PHYSICS_FEAT_WEIGHT}",
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
