import torch

# ====================== CONFIG ======================
IMAGE_SIZE = 256 # Slightly larger for better detail
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 30
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
BACKBONE = 'tf_efficientnet_b4_ns' # Good balance of speed/accuracy
DROPOUT = 0.2

# Data
STRATIFY_COL = 'season' # or 'State'
NUM_WORKERS = 4
SEED = 42

# Targets
# We predict components: Clover, Dead, Green.
# We derive: Total, GDM.
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

# Loss Weights (Official Competition Weights)
# C, D, G, Total, GDM
LOSS_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2]