import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from tqdm import tqdm
import json
import torch.nn as nn
import timm
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import math

import random
random.seed(313)
np.random.seed(313)
torch.manual_seed(313)
torch.cuda.manual_seed_all(313)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================== CONFIGURATION ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS (Update MODEL_DIR to your upload location)
TEST_CSV_PATH = './test.csv'  
TEST_IMG_DIR = './test/' 
MODEL_DIR = './logs/unified_triplet_20260128_124643'

# DEFAULTS
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
FUSION_DIM = 384
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16
BIOMASS_CLAMP = 2500.0  # grams - max realistic biomass for dense pasture (~1-2 m² plot)

# FEATURE FLAGS
USE_TTA = True              # Enable/Disable Test-Time Augmentation
SAVE_IMAGES = True          # Save augmented images for debugging
MAX_IMAGES_TO_SAVE = 10     # Only save first N batches to avoid disk fill

# ====================== IMAGE SAVING HELPER ======================
def save_tta_images(images, batch_idx, view_name, output_dir='./inference_images_routed', max_to_save=MAX_IMAGES_TO_SAVE):
    """
    Save TTA-augmented images for visualization.
    """
    if not SAVE_IMAGES or batch_idx >= max_to_save:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize images
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Save as grid
    save_path = os.path.join(output_dir, f'batch_{batch_idx:03d}_{view_name}.png')
    save_image(images_denorm, save_path, nrow=4, padding=2)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return x + self.dropout(self.block(x))

# ====================== MODEL ARCHITECTURE (HARDCODED) ======================
class BiomassUnifiedModel(nn.Module):
    def __init__(self, 
                 num_aux=5, 
                 num_hsv=5,
                 backbone_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                 num_species=14,
                 fusion_dim=192,  # Matches new training default
                 img_h=224,
                 img_w=224,
                 biomass_clamp=2500.0):
        super(BiomassUnifiedModel, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_species = num_species
        self.fusion_dim = fusion_dim
        self.img_h = img_h
        self.img_w = img_w
        self.num_aux = num_aux
        self.biomass_clamp_val = biomass_clamp
        
        self.backbone = timm.create_model(self.backbone_name, pretrained=False, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_h, self.img_w)
            feats = self.backbone(dummy_input)
            
            if len(feats.shape) == 4:  # CNN: [B, C, H, W]
                self.backbone_dim = feats.shape[1]
            elif len(feats.shape) == 3:  # ViT: [B, seq_len, embed_dim]
                self.backbone_dim = feats.shape[2]
            else:  # Already pooled
                self.backbone_dim = feats.shape[1]
                
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),  
            nn.ReLU(),
            nn.Dropout(0.5), # Matches new training
            nn.Linear(64, self.num_aux)
        )
        
        self.num_hsv = num_hsv
        self.hsv_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),  
            nn.ReLU(),
            nn.Dropout(0.5), # Matches new training
            nn.Linear(64, self.num_hsv)
        )
        
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.LayerNorm(32),  
            nn.ReLU(),
            nn.Dropout(0.5), # Matches new training
            nn.Linear(32, self.num_species)
        )
        
        # 5. Biomass Head (Enhanced Complexity)
        input_dim = self.backbone_dim + self.num_aux + self.num_hsv + self.num_species
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim), 
            nn.GELU(),
            nn.Dropout(0.5),
            ResidualBlock(self.fusion_dim, dropout=0.5),
            ResidualBlock(self.fusion_dim, dropout=0.5),
            nn.Linear(self.fusion_dim, 5), # Predicting 5 targets directly
        )

        self.log_clamp = torch.log1p(torch.tensor(float(self.biomass_clamp_val)))
        self._init_biomass_head()
        
    def _init_biomass_head(self):
        # Recursive initialization for all sub-modules
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.aux_head.apply(init_weights)
        self.hsv_head.apply(init_weights)
        self.species_head.apply(init_weights)
        self.biomass_head.apply(init_weights)

        # Final layer specific init
        last_layer = self.biomass_head[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        
        with torch.no_grad():
            last_layer.bias[0] = 3.0 # Green (~20g)
            last_layer.bias[1] = 2.0 # Dead (~7g)
            last_layer.bias[2] = 2.5 # Clover (~12g)
            last_layer.bias[3] = 3.2 # GDM (~25g)
            last_layer.bias[4] = 3.5 # Total (~30g)

    def forward(self, x):
        feat_map = self.backbone(x)
        
        if len(feat_map.shape) == 4:  # CNN
            img_feats = self.global_pool(feat_map).flatten(1)
        elif len(feat_map.shape) == 3:  # ViT
            img_feats = feat_map.mean(dim=1)
        else:
            img_feats = feat_map
            
        # Add stochastic depth after backbone (matches training)
        img_feats = nn.functional.dropout(img_feats, p=0.3, training=self.training)
        
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        aux_out = self.aux_head(img_feats)
        hsv_out = self.hsv_head(img_feats)
        
        combined_feats = torch.cat([img_feats, aux_out, hsv_out, species_probs], dim=1)
        log_preds_raw = self.biomass_head(combined_feats)
        
        p_green        = torch.clamp(nn.functional.softplus(log_preds_raw[:, 0:1]), 0.0, self.log_clamp)
        p_dead_direct  = torch.clamp(nn.functional.softplus(log_preds_raw[:, 1:2]), 0.0, self.log_clamp)
        p_clover       = torch.clamp(nn.functional.softplus(log_preds_raw[:, 2:3]), 0.0, self.log_clamp)
        p_gdm_direct   = torch.clamp(nn.functional.softplus(log_preds_raw[:, 3:4]), 0.0, self.log_clamp)
        p_total_direct = torch.clamp(nn.functional.softplus(log_preds_raw[:, 4:5]), 0.0, self.log_clamp)
        
        lin_green = torch.expm1(p_green)
        lin_clover = torch.expm1(p_clover)
        lin_gdm_derived = torch.clamp(lin_green + lin_clover, min=1e-4)
        p_gdm_derived = torch.log1p(lin_gdm_derived)
        
        p_gdm = 0.7 * p_gdm_direct + 0.3 * p_gdm_derived
        
        lin_total = torch.expm1(p_total_direct)
        lin_dead_derived = torch.clamp(lin_total - (lin_green + lin_clover), min=1e-4)
        p_dead_derived = torch.log1p(lin_dead_derived)
        
        if hsv_out.shape[1] > 3:
            # Gentler blending
            vis_weight = torch.clamp(hsv_out[:, 3:4] * 5.0, 0.0, 1.0)
            p_dead = vis_weight * p_dead_direct + (1 - vis_weight) * p_dead_derived
        else:
            p_dead = 0.5 * p_dead_direct + 0.5 * p_dead_derived
            
        lin_gdm_final = torch.expm1(p_gdm)
        lin_dead_final = torch.expm1(p_dead)
        p_total_final = torch.log1p(torch.clamp(lin_gdm_final + lin_dead_final, min=1e-4))
        
        biomass_out = torch.cat([p_green, p_dead, p_clover, p_gdm, p_total_final], dim=1)
        
        return biomass_out, aux_out, hsv_out, species_logits

# ====================== TTA HELPERS ======================
def get_largest_rotated_crop(h, w, angle):
    angle_rad = math.radians(abs(angle))
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    scale = 1.0 / (cos_a + sin_a)
    return int(h * scale), int(w * scale)

def rotate_crop_resize(img, angle):
    """
    Handles both Single Image (C, H, W) and Batch (B, C, H, W).
    For 90/270 degree rotations, just rotate without cropping.
    For small angles, rotate, crop to remove black corners, and resize back.
    """
    h, w = img.shape[-2:]
    
    # For 90° and 270°, just rotate (dimensions swap)
    if abs(angle) in [90, 270]:
        img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        # Resize back to original dimensions (since rotation swaps H and W)
        if img.ndim == 3:
            img_resized = torch.nn.functional.interpolate(
                img_rot.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            img_resized = torch.nn.functional.interpolate(
                img_rot, size=(h, w), mode='bilinear', align_corners=False
            )
        return img_resized
    
    # For small angles, use crop method
    img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
    
    ch, cw = get_largest_rotated_crop(h, w, angle)
    img_crop = TF.center_crop(img_rot, [ch, cw])
    
    if img.ndim == 3:
        img_resized = torch.nn.functional.interpolate(
            img_crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
    else:
        img_resized = torch.nn.functional.interpolate(
            img_crop, size=(h, w), mode='bilinear', align_corners=False
        )
        
    return img_resized

def no_tta(model, image, batch_idx=0):
    """
    Single pass inference.
    """
    model.eval()
    
    if SAVE_IMAGES:
        save_tta_images(image, batch_idx, 'original')
    
    with torch.no_grad():
        biomass_out, aux, hsv, sp_logits = model(image)
        # Extract Confidence from Species
        probs = torch.sigmoid(sp_logits)
        conf, _ = torch.max(probs, dim=1)
        
    return biomass_out, conf
    
def apply_tta(model, image, batch_idx=0):
    """
    TTA with HARD PHYSICS BARRIER and image saving.
    """
    model.eval()
    
    all_biomass_linear = [] 
    all_confidences = []
    # TTA Policy: 5 Views with names for saving
    tta_views = [
        ('identity', lambda x: x),
        ('hflip', lambda x: torch.flip(x, [3])),
        ('vflip', lambda x: torch.flip(x, [2])),
        ('rot5', lambda x: rotate_crop_resize(x, 5)),
        ('rot-5', lambda x: rotate_crop_resize(x, -5)),
        ('rot90', lambda x: rotate_crop_resize(x, 90)),
        ('rot270', lambda x: rotate_crop_resize(x, 270)),
    ]

    for view_name, transform_fn in tta_views:
        with torch.no_grad():
            img_aug = transform_fn(image)
            
            # Save augmented view
            if SAVE_IMAGES:
                save_tta_images(img_aug, batch_idx, view_name)
            
            # Predict all targets directly
            biomass_out, aux, hsv, sp_logits = model(img_aug) 
            
            # Linear space for all 5 targets
            bio_lin = torch.expm1(biomass_out)
            green_lin = bio_lin[:, 0:1]
            dead_lin  = bio_lin[:, 1:2]
            clover_lin = bio_lin[:, 2:3]
            gdm_lin = bio_lin[:, 3:4]
            total_lin = bio_lin[:, 4:5]
            
            # Resulting 5 linear targets (direct predictions)
            full_bio_lin = torch.cat([green_lin, dead_lin, clover_lin, gdm_lin, total_lin], dim=1)
            all_biomass_linear.append(full_bio_lin)
            
            # Extract Confidence from Species
            probs = torch.sigmoid(sp_logits) 
            conf, _ = torch.max(probs, dim=1) 
            all_confidences.append(conf)

    # AGGREGATION
    avg_bio_linear = torch.stack(all_biomass_linear).mean(0)
    avg_confidence = torch.stack(all_confidences).mean(0)
            
    return avg_bio_linear, avg_confidence

# ====================== DATASET ======================
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get just the filename (e.g., ID1001187975.jpg)
        img_filename = os.path.basename(row['image_path'])
        img_path = os.path.join(self.img_dir, img_filename)
        
        # CRITICAL: Check if path exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

def get_inference_transforms(h, w, mean=None, std=None):
    """
    Inference transforms matches validation (Resize + Norm).
    Uses dynamic mean/std from metadata for consistency.
    """
    if mean is None or std is None:
        raise ValueError("mean and std must be provided - no fallbacks allowed")
        
    return transforms.Compose([
        transforms.Resize((h, w)),        
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def load_model(fold_path, device, num_species, backbone_name, num_aux=5, num_hsv=5, biomass_clamp=None, fusion_dim=192):
    model = BiomassUnifiedModel(
        backbone_name=backbone_name, 
        num_species=num_species, 
        num_aux=num_aux, 
        num_hsv=num_hsv,
        biomass_clamp=biomass_clamp,
        fusion_dim=fusion_dim
    ).to(device)
    
    state_dict = torch.load(fold_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_model] Non-strict load. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()
    return model

# ====================== MAIN INFERENCE ======================
def run_inference(USE_TTA=True):
    print("="*80)
    print("ROUTED INFERENCE (Weighted Ensemble)")
    print(f"MODE: {'TTA ENABLED' if USE_TTA else 'NO TTA'}")
    print(f"IMAGE SAVING: {'ENABLED' if SAVE_IMAGES else 'DISABLED'}")
    print(f"SHARPENING: DISABLED")
    print("="*80 + "\n")
    
    # 1. LOAD TEST DATA
    if not os.path.exists(TEST_CSV_PATH):
        print("Warning: Test CSV not found. Creating dummy.")
        df_wide = pd.DataFrame({'clean_id':['test_1', 'test_2'], 'image_path':['t1.jpg', 't2.jpg']})
    else:
        df = pd.read_csv(TEST_CSV_PATH)
        if 'target_name' in df.columns:
            df['clean_id'] = df['sample_id'].str.split('__').str[0]
            df_wide = df[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
        else:
            df_wide = df
            if 'clean_id' not in df_wide.columns: 
                df_wide['clean_id'] = df_wide['sample_id']
                 
    print(f"Test Images: {len(df_wide)}")
    
    # 2. LOAD METADATA - STRICT MODE (no fallbacks)
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json missing in {MODEL_DIR}")
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract required parameters (no fallbacks - fail if missing)
    num_species = metadata['num_species']
    backbone_name = metadata['backbone']
    img_h = metadata['image_height']
    img_w = metadata['image_width']
    imagenet_mean = tuple(metadata['imagenet_mean'])
    imagenet_std = tuple(metadata['imagenet_std'])
    num_aux_total = metadata['num_aux']
    # Split models have 9 or 10 total aux features (4 or 5 tabular + 5 hsv)
    num_aux = num_aux_total - 5 if num_aux_total >= 9 else num_aux_total
    num_hsv = 5 if num_aux_total >= 9 else 0
    fusion_dim = metadata.get('fusion_dim', 192)

    biomass_clamp = metadata['biomass_clamp']
    print(f"Config: {backbone_name} | {img_w}x{img_h} | num_aux: {num_aux}, num_hsv: {num_hsv} | fusion_dim: {fusion_dim} | clamp: {biomass_clamp}g")
    print(f"Normalization: mean={imagenet_mean}, std={imagenet_std}")

    # 3. DISCOVER MODELS (Only fold-specific models)
    found_folds = []
    # Check both "best_model_foldX.pth" (Standard) and "best_foldX.pt" (Triplet)
    for f in range(1, 11):
        for pattern in [f"best_fold{f}.pt", f"best_model_fold{f}.pth"]:
            p = os.path.join(MODEL_DIR, pattern)
            if os.path.exists(p): 
                found_folds.append(p)
                break # Found one for this fold, move to next
                
    if not found_folds:
        # Check overall models
        for pattern in ["best_model_overall.pth", "best_model.pt"]:
            p = os.path.join(MODEL_DIR, pattern)
            if os.path.exists(p): 
                found_folds.append(p)
                break
            
    if not found_folds: 
        raise FileNotFoundError(f"No fold models found in {MODEL_DIR}")
    print(f"Found {len(found_folds)} checkpoints.\n")

    # 4. RUN INFERENCE LOOP
    val_transform = get_inference_transforms(h=img_h, w=img_w, mean=imagenet_mean, std=imagenet_std)
    ds = TestDataset(df_wide, TEST_IMG_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    ensemble_preds = []   # List of [N_Samples, 5]
    ensemble_confs = []   # List of [N_Samples]
    final_clean_ids = []
    
    for i, model_path in enumerate(found_folds):
        fold_name = os.path.basename(model_path)
        print(f"-> Processing {fold_name}...")
        model = load_model(model_path, DEVICE, num_species, backbone_name, num_aux=num_aux, num_hsv=num_hsv, biomass_clamp=biomass_clamp, fusion_dim=fusion_dim)
        
        fold_preds = []
        fold_confs = []
        
        with torch.no_grad():
            for batch_idx, (imgs, ids) in enumerate(tqdm(loader, desc=f"  Fold {i+1}", leave=False)):
                imgs = imgs.to(DEVICE)
                
                # Choose TTA or no TTA
                if USE_TTA:
                    preds_linear_5, conf = apply_tta(model, imgs, batch_idx)
                    preds_linear_5 = preds_linear_5.cpu().numpy()
                else:
                    # No-TTA: use direct predictions
                    biomass_out, conf = no_tta(model, imgs, batch_idx)
                    bio_lin = torch.expm1(biomass_out)
                    green_lin = bio_lin[:, 0:1]
                    dead_lin  = bio_lin[:, 1:2]
                    clover_lin = bio_lin[:, 2:3]
                    gdm_lin = bio_lin[:, 3:4]
                    total_lin = bio_lin[:, 4:5]
                    preds_linear_5 = torch.cat([green_lin, dead_lin, clover_lin, gdm_lin, total_lin], dim=1).cpu().numpy()
                
                # Store [green, dead, clover, gdm, total]
                fold_preds.append(preds_linear_5)
                fold_confs.append(conf.cpu().numpy())
                
                if i == 0: 
                    final_clean_ids.extend(ids)
                    
        ensemble_preds.append(np.concatenate(fold_preds, axis=0))
        ensemble_confs.append(np.concatenate(fold_confs, axis=0))
        print(f"  ✓ Completed\n")
    
    # Convert to Numpy for Analysis: [N_Models, N_Samples]
    W_raw = np.stack(ensemble_confs, axis=0)
    
    # ====================== ROUTER DIAGNOSTICS ======================
    print("\n" + "="*30 + " ROUTER DIAGNOSTICS " + "="*30)
    
    # 1. Who is winning?
    winners = np.argmax(W_raw, axis=0) 
    print(f"Total Samples: {len(final_clean_ids)}")
    for model_idx in range(len(found_folds)):
        win_count = np.sum(winners == model_idx)
        win_pct = (win_count / len(final_clean_ids)) * 100
        fname = os.path.basename(found_folds[model_idx])
        print(f"  {fname:<25} | Selected {win_count:>4} times ({win_pct:.1f}%)")
        
    # 2. Print First 5 Samples Detail
    print("\n--- Sample Selection Preview ---")
    for i in range(min(5, len(final_clean_ids))):
        sid = final_clean_ids[i]
        scores = W_raw[:, i]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        print(f"Sample {sid:<15}: Trusting Fold {best_idx+1} (Conf: {best_score:.4f}) | Others: {[f'{s:.2f}' for s in scores]}")
    print("="*80 + "\n")
    # ================================================================

    # 5. WEIGHTED ENSEMBLE CALCULATION 
    print("Calculating Species-Weighted Ensemble...")
    
    # Shape: [N_Models, N_Samples, 4] (Total, GDM, Green, Dead)
    E = np.stack(ensemble_preds, axis=0)
    # Shape: [N_Models, N_Samples]
    W = W_raw
    
    # Expand Weights for broadcasting: [N_Models, N_Samples, 1]
    W_expanded = W[:, :, np.newaxis]
    
    # Weighted Average: Sum(Pred * Weight) / Sum(Weight)
    numerator = np.sum(E * W_expanded, axis=0)
    denominator = np.sum(W_expanded, axis=0) + 1e-8
    
    avg_out = numerator / denominator  # [N_Samples, 5]
    
    # KAGGLE SAFETY: Explicit bounds [0, 2500g] after weighted ensemble
    # Model already clamps during forward pass, but this provides extra safety against numerical errors
    avg_out = np.clip(avg_out, 0.0, 2500.0)

    # 6. ENFORCE PHYSICS (Total = Green + Dead + Clover, GDM = Green + Clover)
    print("Enforcing physics constraints...")
    # Order: [Green, Dead, Clover, GDM, Total]
    pred_green = np.maximum(0, avg_out[:, 0])
    pred_dead = np.maximum(0, avg_out[:, 1])
    pred_clover = np.maximum(0, avg_out[:, 2])
    
    # GDM = Green + Clover
    pred_gdm = pred_green + pred_clover
    # Total = GDM + Dead
    pred_total = pred_gdm + pred_dead

    # 7. EXPORT [green, dead, clover, gdm, total]
    final_df = pd.DataFrame({
        'Dry_Green_g': pred_green,
        'Dry_Dead_g': pred_dead,
        'Dry_Clover_g': pred_clover,
        'GDM_g': pred_gdm,
        'Dry_Total_g': pred_total,
    })
    final_df['clean_id'] = final_clean_ids
    
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    submission_rows = []
    for _, row in final_df.iterrows():
        cid = row['clean_id']
        for col in target_cols:
            submission_rows.append({
                'sample_id': f"{cid}__{col}", 
                'target': row[col]
            })
            
    sub_df = pd.DataFrame(submission_rows)
    
    # Save with appropriate filename
    output_filename ='submission.csv'
    sub_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*80)
    print("ROUTED INFERENCE COMPLETE")
    print("="*80)
    print(sub_df.head(10))
    print(f"\nSaved {len(sub_df)} rows to {output_filename}")
    if SAVE_IMAGES:
        print(f"Saved augmented images to ./inference_images_routed/")
    print("="*80)

if __name__ == '__main__':
    run_inference(USE_TTA)