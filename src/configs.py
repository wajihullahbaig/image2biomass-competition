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
BATCH_SIZE = 16 
LEARNING_RATE = 15e-5 
N_FOLDS = 5
EPOCHS = 50 
BACKBONE = 'timm/tf_efficientnet_b3.ns_jft_in1k'  
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
COL_WEIGHTS_TENSOR = torch.tensor(OFFICIAL_WEIGHTS, device=DEVICE)
# training settings
FREEZE_BACKBONE = True
BACKBONE_FREEZE_FRACTION = 0.75
USE_TTA = True

# --- Model Settings ---
FUSION_DIM = 256
# Targets are small (KG scale, < 1.0). Large weights cause exploding gradients.
BIOMASS_FEAT_WEIGHT = 100.0 
AUX_FEAT_WEIGHT = 1.0
SPECIES_FEAT_WEIGHT = 1.0
MONTH_FEAT_WEIGHT = 1.0
# --- Regularization ---
EWC_IMPORTANCE = 1000
EARLY_STOP_PATIENCE = 15
ACCUMULATION_STEPS = 1  # Keep at 1 for stability unless OOM

# return a string representation of the configuration
def config_str():
    config_items = [
        f"DEVICE: {DEVICE}",
        f"IMAGE_SIZE: {IMAGE_SIZE}",
        f"BATCH_SIZE: {BATCH_SIZE}",
        f"LEARNING_RATE: {LEARNING_RATE}",
        f"N_FOLDS: {N_FOLDS}",
        f"EPOCHS: {EPOCHS}",
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
        f"EWC_IMPORTANCE: {EWC_IMPORTANCE}",
        f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}",
        f"ACCUMULATION_STEPS: {ACCUMULATION_STEPS}"
    ]
    return "\n".join(config_items)