# config.py
import torch
# ====================== CONFIG ======================
# CONFIGURATIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prerocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# We use 224x512 to respect the ~2.33 aspect ratio of the 70cm x 30cm quadrats.
# This prevents "squashing" the grass which destroys density features.
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 768
# Training Hyperparameters
BATCH_SIZE = 32 
LEARNING_RATE = 1e-4 
N_FOLDS = 3
EPOCHS = 40 
WEIGHT_DECAY = 0.05
EARLY_STOP_PATIENCE = 20
BACKBONE = 'timm/convnext_tiny.fb_in1k'  
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
# training settings
FREEZE_BACKBONE = False
BACKBONE_FREEZE_FRACTION = 0.5
USE_TTA = True

# --- Model Settings ---
FUSION_DIM = 256
# Balanced Weights: Scaling optimized for Gram-scale Log-space
BIOMASS_FEAT_WEIGHT = 80.0 
AUX_FEAT_WEIGHT = 15.0
SPECIES_FEAT_WEIGHT = 20.0
TAXONOMY_FEAT_WEIGHT = 25.0
PHYSICS_FEAT_WEIGHT = 30.0 
PREDICT_DEAD_RATIO = True 

# ====================== TAXONOMY & SPECIES ======================
CORE_SPECIES = [
    'clover', 'whiteclover', 'subcloverdalkeith', 'subcloverlosa',  # 0-3
    'ryegrass', 'phalaris', 'fescue', 'lucerne',                    # 4-7
    'barleygrass', 'silvergrass', 'speargrass', 'bromegrass',       # 8-11
    'capeweed', 'crumbweed'                                         # 12-13
]

# Define Groups by Name (Safer than indices)
GROUP_DEFINITIONS = {
    'legume': ['clover', 'whiteclover', 'subcloverdalkeith', 'subcloverlosa', 'lucerne'],
    'grass':  ['ryegrass', 'phalaris', 'fescue', 'barleygrass', 'silvergrass', 'speargrass', 'bromegrass'],
    'weed':   ['capeweed', 'crumbweed']
}

# Dynamically Generate Indices Dictionary
# Structure: {'Legume': [0, 1, 2, 3, 7], 'Grass': [...], 'Weed': [...]}
TAXONOMY_IDXS = {
    group: [i for i, species in enumerate(CORE_SPECIES) if species in names]
    for group, names in GROUP_DEFINITIONS.items()
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
        f"FUSION_DIM: {FUSION_DIM}",
        f"AUX_FEAT_WEIGHT: {AUX_FEAT_WEIGHT}",
        f"SPECIES_FEAT_WEIGHT: {SPECIES_FEAT_WEIGHT}",
        f"TAXONOMY_FEAT_WEIGHT: {TAXONOMY_FEAT_WEIGHT}",
        f"BIOMASS_FEAT_WEIGHT: {BIOMASS_FEAT_WEIGHT}",
        f"PHYSICS_FEAT_WEIGHT: {PHYSICS_FEAT_WEIGHT}"
    ]
    return "\n".join(config_items)