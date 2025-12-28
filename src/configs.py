# config.py
import torch
# ====================== CONFIG ======================
# CONFIGURATIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prerocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 384
# Training Hyperparameters
BATCH_SIZE = 32 
LEARNING_RATE = 1e-4 
N_FOLDS = 4
EPOCHS = 80 
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 30
BACKBONE = 'timm/tf_efficientnet_b3.ns_jft_in1k'  
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
COL_WEIGHTS_TENSOR = torch.tensor(OFFICIAL_WEIGHTS, device=DEVICE)
# training settings
FREEZE_BACKBONE = True
BACKBONE_FREEZE_FRACTION = 0.5
USE_TTA = True

# --- Model Settings ---
FUSION_DIM = 256
# Balanced Weights: Biomass is still king but Aux/Phys are loud enough to matter
BIOMASS_FEAT_WEIGHT = 200.0 
AUX_FEAT_WEIGHT = 5.0
SPECIES_FEAT_WEIGHT = 0.01
MONTH_FEAT_WEIGHT = 0.01
PHYSICS_FEAT_WEIGHT = 15.0 # Define this properly in config

# --- Regularization ---

# return a string representation of the configuration
def config_str():
    config_items = [
        f"DEVICE: {DEVICE}",
        f"IMAGE_SIZE: {IMAGE_SIZE}",
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
        f"MONTH_FEAT_WEIGHT: {MONTH_FEAT_WEIGHT}",
        f"BIOMASS_FEAT_WEIGHT: {BIOMASS_FEAT_WEIGHT}",
        f"PHYSICS_FEAT_WEIGHT: {PHYSICS_FEAT_WEIGHT}"
    ]
    return "\n".join(config_items)