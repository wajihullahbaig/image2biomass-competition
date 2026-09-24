# configs.py - Configuration Exporter
import os
import sys

# Ensure both workspace root, src, and src/training are on PYTHONPATH
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)
parent_dir = os.path.dirname(cur_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config.loader import cfg
except ImportError:
    from src.training.config.loader import cfg

DEVICE = cfg.device
IMAGE_HEIGHT = cfg.preprocessing.image_height
IMAGE_WIDTH = cfg.preprocessing.image_width
DUAL_STREAM = cfg.preprocessing.dual_stream

BATCH_SIZE = cfg.hyperparameters.batch_size
GRADIENT_ACCUMULATION_STEPS = getattr(cfg.hyperparameters, 'gradient_accumulation_steps', 1)
LEARNING_RATE = cfg.hyperparameters.learning_rate
N_FOLDS = cfg.hyperparameters.n_folds
EPOCHS = cfg.hyperparameters.epochs
WEIGHT_DECAY = cfg.hyperparameters.weight_decay
EARLY_STOP_PATIENCE = cfg.hyperparameters.early_stop_patience
BACKBONE = cfg.hyperparameters.backbone
MAX_GRAD_NORM = cfg.hyperparameters.max_grad_norm

TARGET_COLS = cfg.targets.cols
OFFICIAL_WEIGHTS = cfg.targets.official_weights

STAGE1_EPOCHS = cfg.training.stage1_epochs
STAGE2_EPOCHS = cfg.training.stage2_epochs
STAGE3_EPOCHS = getattr(cfg.training, 'stage3_epochs', 0)
STAGE2_BACKBONE_LR_FACTOR = cfg.training.stage2_backbone_lr_factor
STAGE3_LR_FACTOR = getattr(cfg.training, 'stage3_lr_factor', 0.1)
FUSION_DIM = cfg.training.fusion_dim
DROPOUT = cfg.training.dropout
USE_TTA = cfg.training.use_tta

CAMERA_SCALING_PROB = cfg.augmentation.camera_scaling_prob
NUM_INTERVALS = cfg.loss.num_intervals
CLS_WEIGHT = cfg.loss.cls_weight

GROUP_COL = cfg.split.group_col
GROUP_STRAT_COL = cfg.split.group_stratification_col


def config_str():
    return (
        f"DEVICE: {DEVICE}\n"
        f"IMAGE_SIZE: {IMAGE_HEIGHT}x{IMAGE_WIDTH} (Dual-Stream={DUAL_STREAM})\n"
        f"BACKBONE: {BACKBONE}\n"
        f"BATCH_SIZE: {BATCH_SIZE} (Grad Accum: {GRADIENT_ACCUMULATION_STEPS}) | LR: {LEARNING_RATE} | WEIGHT_DECAY: {WEIGHT_DECAY}\n"
        f"2-STAGE SCHEDULE: S1={STAGE1_EPOCHS} eps (heads warm-up) | S2={STAGE2_EPOCHS} eps (full FT)\n"
        f"TARGETS ({len(TARGET_COLS)} base): {TARGET_COLS}\n"
        f"FUSION_DIM: {FUSION_DIM} | DROPOUT: {DROPOUT} | USE_TTA: {USE_TTA}\n"
        f"INTERVALS: {NUM_INTERVALS} bins | CLS_WEIGHT: {CLS_WEIGHT}\n"
        f"CAMERA_SCALE_PROB: {CAMERA_SCALING_PROB}\n"
        f"CROSS-VALIDATION: {N_FOLDS} folds grouped by '{GROUP_COL}' & stratified by '{GROUP_STRAT_COL}'"
    )
