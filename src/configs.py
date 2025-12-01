# config.py
import torch
# ====================== CONFIG ======================
# CONFIGURATIONS
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prerocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Training
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 100
BACKBONE_S1 = 'tf_efficientnet_b3_ns'        
BACKBONE_S2 = 'swin_base_patch4_window7_224' 
# Feature Flags
USE_COUNT_FEATURES = True        # Use Global/Seasonal counts in Stage 2
USE_SAMPLE_WEIGHTS_S1 = True     # Use Hard Balancing for Stage 1
# Use Official Weights for Loss Calculation
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
COL_WEIGHTS_TENSOR = torch.tensor(OFFICIAL_WEIGHTS, device=DEVICE)
N_FOLDS = 5
TEST_SPLIT_RATIO = 0.20     # 10% data for validation in each fold
STAGE1_STRATIFICATION_COLUMN = 'Species'  
STAGE2_STRATIFICATION_COLUMN = 'pred_species'   