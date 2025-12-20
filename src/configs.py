# config.py
import torch
# ====================== CONFIG ======================
# CONFIGURATIONS
IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prerocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Training Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
N_FOLDS = 4
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 40
BACKBONE_S1 = 'timm/tf_efficientnet_b0.ns_jft_in1k' 
BACKBONE_S2 = 'efficientnet_b4' 
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
COL_WEIGHTS_TENSOR = torch.tensor(OFFICIAL_WEIGHTS, device=DEVICE)
# training settings
FREEZE_BACKBONE = True
BACKBONE_FREEZE_FRACTION = 0.8
USE_TTA = True

# Model Settings
FUSION_DIM = 512
AUX_FEAT_WEIGHT = 0.5
SPECIES_FEAT_WEIGHT = 0.2
BIOMASS_FEAT_WEIGHT = 1.0