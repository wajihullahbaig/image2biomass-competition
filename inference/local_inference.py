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
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====================== CONFIGURATION ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS (Update MODEL_DIR to your upload location)
TEST_CSV_PATH = './test.csv'  
TEST_IMG_DIR = './test/' 
MODEL_DIR = './logs/unified_holdout_20260115_195150'

# DEFAULTS
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
FUSION_DIM = 256
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
BIOMASS_CLAMP = 1000.0

# FEATURE FLAGS
USE_TTA = True            # Enable/Disable Test-Time Augmentation
SAVE_IMAGES = True        # Set to False to disable image saving
MAX_IMAGES_TO_SAVE = 10   # Only save first N batches

# ====================== SHARPENING TRANSFORM ======================


# ====================== IMAGE SAVING HELPER ======================
def save_tta_images(images, batch_idx, view_name, output_dir='./inference_images', max_to_save=MAX_IMAGES_TO_SAVE):
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
# ====================== UPDATED MODEL ARCHITECTURE ======================
class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name, num_aux=5, num_species=14, pretrained=False):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            feats = self.backbone(dummy_input)
            self.backbone_dim = feats.shape[1]
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_aux)
        )
        
        # 3. Species Head (Fine-Grained: 14 classes)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, num_species)
        )

        # 4. Taxonomy Head (Coarse-Grained: 3 classes - Legume, Grass, Weed)
        self.taxonomy_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(16, 3) 
        )
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux(3) + Species(14) + Taxonomy(3)
        input_dim = self.backbone_dim + num_aux + num_species + 3
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.6),
            # Transition to 128
            nn.Linear(FUSION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # [Log_Total, Log_GDM, Log_Green]
        )
        # Dead Ratio Head (predicts Dead_to_Total in [0,1])
        self.dead_ratio_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
        self.log_clamp = torch.log1p(torch.tensor(BIOMASS_CLAMP))

        self._init_biomass_head()
        
    def _init_biomass_head(self):
        last_layer = self.biomass_head[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        with torch.no_grad():
            last_layer.bias.fill_(0)
            last_layer.bias[0] = 4.0 # ~50g total
            last_layer.bias[1] = 3.5 # ~30g gdm
            last_layer.bias[2] = 3.0 # ~20g green

    def forward(self, x):
        feat_map = self.backbone(x)
        img_feats = self.global_pool(feat_map).flatten(1)
        
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        taxonomy_logits = self.taxonomy_head(img_feats)
        taxonomy_probs = torch.softmax(taxonomy_logits, dim=1)
        
        aux_out = self.aux_head(img_feats) 
            
        combined_feats = torch.cat([img_feats, aux_out, species_probs, taxonomy_probs], dim=1)
        
        # Biomass Prediction
        log_preds_raw = self.biomass_head(combined_feats)
        # softplus ensures positivity, clamp ensures we don't blow up expm1
        log_preds = torch.clamp(nn.functional.softplus(log_preds_raw), 0.0, self.log_clamp)
        
        log_total = log_preds[:, 0:1]
        log_gdm = log_preds[:, 1:2]
        log_green = log_preds[:, 2:3]
        biomass_out = torch.cat([log_total, log_gdm, log_green], dim=1)
        # Ratio in [0,1]
        dead_ratio = torch.sigmoid(self.dead_ratio_head(combined_feats))
        
        return biomass_out, aux_out, species_logits, taxonomy_logits, dead_ratio


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
    """
    h, w = img.shape[-2:]
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
    Single pass inference with PHYSICS BARRIER.
    Safe against exploding gradients or hallucinations.
    """
    model.eval()
    
    # Save original image
    if SAVE_IMAGES:
        save_tta_images(image, batch_idx, 'original')
    
    with torch.no_grad():
        log_bio, aux, sp, tax_logits, dead_ratio = model(image)
        # Extract Confidence
        probs = torch.softmax(tax_logits, dim=1)
        conf, _ = torch.max(probs, dim=1)
        
    return log_bio, conf, dead_ratio

def apply_tta(model, image, batch_idx=0):
    """
    TTA with HARD PHYSICS BARRIER and image saving.
    """
    model.eval()
    
    all_biomass_linear = [] 
    all_confidences = []
    all_dead_ratios = []

    # TTA Policy: 5 Views with names for saving
    tta_views = [
        ('identity', lambda x: x),
        ('hflip', lambda x: torch.flip(x, [3])),
        ('vflip', lambda x: torch.flip(x, [2])),
        ('rot5', lambda x: rotate_crop_resize(x, 5)),
        ('rot-5', lambda x: rotate_crop_resize(x, -5)),
    ]

    for view_name, transform_fn in tta_views:
        with torch.no_grad():
            img_aug = transform_fn(image)
            
            # Save augmented view
            if SAVE_IMAGES:
                save_tta_images(img_aug, batch_idx, view_name)
            
            # Forward Pass
            log_bio, aux, sp, tax_logits, dead_ratio = model(img_aug) 
            
            # Convert to Linear Grams
            lin_bio = torch.expm1(log_bio)
            
            all_biomass_linear.append(lin_bio)
            
            # Extract Confidence
            probs = torch.softmax(tax_logits, dim=1) 
            conf, _ = torch.max(probs, dim=1) 
            all_confidences.append(conf)
            all_dead_ratios.append(dead_ratio)

    # AGGREGATION
    # Average Biomass (Linear Space)
    avg_bio_linear = torch.stack(all_biomass_linear).mean(0)
    
    # Average Confidence
    avg_confidence = torch.stack(all_confidences).mean(0)
    avg_dead_ratio = torch.stack(all_dead_ratios).mean(0)
            
    # Convert to Log for consistency with return signature
    avg_bio_log = torch.log1p(avg_bio_linear)
            
    return avg_bio_log, avg_confidence, avg_dead_ratio
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

def get_inference_transforms(h, w):
    """
    Inference transforms matches validation (Resize + Norm).
    """
    return transforms.Compose([
        transforms.Resize((h, w)),        
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

def load_model(fold_path, device, num_species, backbone_name, num_aux=7):
    model = BiomassUnifiedModel(backbone_name=backbone_name, num_species=num_species, num_aux=num_aux).to(device)
    state_dict = torch.load(fold_path, map_location=device, weights_only=True)
    # Allow loading older checkpoints with 4-output biomass head by ignoring mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_model] Non-strict load. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()
    return model

# ====================== MAIN INFERENCE ======================
def run_inference(use_tta=False):
    """
    Main inference function.
    
    Args:
        use_tta: Whether to apply test-time augmentation
    """
    print("="*60)
    print(f" INFERENCE MODE: {'TTA ENABLED' if use_tta else 'NO TTA'}")
    print(f" IMAGE SAVING: {'ENABLED' if SAVE_IMAGES else 'DISABLED'}")
    print("="*60 + "\n")
    
    # 1. LOAD TEST DATA
    if not os.path.exists(TEST_CSV_PATH):
        print("Warning: Test CSV not found. Creating dummy.")
        df_wide = pd.DataFrame({'clean_id':['test'], 'image_path':['test.jpg']})
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
    
    # 2. LOAD METADATA
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json missing in {MODEL_DIR}")
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_species = metadata.get('num_species')
    backbone_name = metadata.get('backbone')
    img_h = metadata.get('image_height', IMAGE_HEIGHT)
    img_w = metadata.get('image_width', IMAGE_WIDTH)
    num_aux = metadata.get('num_aux', 5)  # Default to 5 (NDVI, Height, Int_Mul, Int_Add, SpCount)
    print(f"Config: {backbone_name} | {img_w}x{img_h} | num_aux: {num_aux}")

    # 3. DISCOVER MODELS
    found_folds = []
    for f in range(10):
        p = os.path.join(MODEL_DIR, f"best_model_fold{f+1}.pth")
        if os.path.exists(p): 
            found_folds.append(p)
    if not found_folds:
        if os.path.exists(os.path.join(MODEL_DIR, "best_model_overall.pth")):
            found_folds.append(os.path.join(MODEL_DIR, "best_model_overall.pth"))
            
    if not found_folds: 
        raise FileNotFoundError(f"No models found in {MODEL_DIR}")
    print(f"Found {len(found_folds)} checkpoints.\n")

    # 4. RUN INFERENCE
    val_transform = get_inference_transforms(h=img_h, w=img_w)
    ds = TestDataset(df_wide, TEST_IMG_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    ensemble_preds_g = []
    final_clean_ids = []
    
    for i, model_path in enumerate(found_folds):
        fold_name = os.path.basename(model_path)
        print(f"-> Processing {fold_name}...")
        model = load_model(model_path, DEVICE, num_species, backbone_name, num_aux=num_aux)
        
        fold_preds = []
        with torch.no_grad():
            for batch_idx, (imgs, ids) in enumerate(tqdm(loader, desc=f"  Fold {i+1}", leave=False)):
                imgs = imgs.to(DEVICE)
                
                # Choose TTA or no TTA
                if use_tta:
                    # Returns: (log_pred, conf, dead_ratio)
                    log_pred, _, dead_ratio = apply_tta(model, imgs, batch_idx)
                else:
                    log_pred, _, dead_ratio = no_tta(model, imgs, batch_idx)
                
                # Convert to Linear for averaging
                preds_linear = torch.expm1(log_pred)  # [total, gdm, green]
                dead_ratio_np = dead_ratio.cpu().numpy()
                

                pred_total = preds_linear[:, 0:1].cpu().numpy()
                pred_dead = pred_total * dead_ratio_np
                
                # Store [total, gdm, green, dead] for ensemble averaging
                fold_preds.append(np.concatenate([preds_linear.cpu().numpy(), pred_dead], axis=1))
                if i == 0: 
                    final_clean_ids.extend(ids)
                    
        ensemble_preds_g.append(np.concatenate(fold_preds, axis=0))
        print(f"  ✓ Completed\n")
        
    # 5. AVERAGE ENSEMBLE (Linear Space) over 4 predictions
    print("Averaging ensemble predictions...")
    all_preds = np.mean(ensemble_preds_g, axis=0)  # shape [N,4] => [Total,GDM,Green,Dead] (order from model)
    all_preds = np.maximum(all_preds, 0)
    avg_total = all_preds[:, 0]
    avg_gdm = all_preds[:, 1]
    avg_green = all_preds[:, 2]
    avg_dead = all_preds[:, 3]  # Already computed as total*ratio per model

    # 6. DERIVE 5 targets from 4 predictions (Dead already computed per model)
    pred_total = avg_total
    pred_gdm = avg_gdm
    pred_green = avg_green
    pred_dead = avg_dead  # Already computed as total*ratio per model
    
    # Ensure no negative values
    pred_clover = np.maximum(0, pred_gdm - pred_green)

    # 7. EXPORT in correct order [green, dead, clover, gdm, total]
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
    
    output_filename = 'submission.csv'
    sub_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(sub_df.head(10))
    print(f"\nSaved {len(sub_df)} rows to {output_filename}")
    if SAVE_IMAGES:
        print(f"Saved augmented images to ./inference_images/")
    print("="*60)

if __name__ == '__main__':
    run_inference(use_tta=USE_TTA)