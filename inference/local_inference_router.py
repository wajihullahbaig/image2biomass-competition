import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import json
import torch.nn as nn
import timm
from torchvision import transforms
import torchvision.transforms.functional as TF
import math

# ====================== CONFIGURATION ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS (Update MODEL_DIR to your upload location)
TEST_CSV_PATH = './test.csv'  
TEST_IMG_DIR = './test/' 
MODEL_DIR = './logs/mixup_taxonomy_20251231_114427'


# DEFAULTS
DEFAULT_HEIGHT = 320
DEFAULT_WIDTH = 768
FUSION_DIM = 256
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16

import matplotlib.pyplot as plt


# ====================== UPDATED MODEL ARCHITECTURE ======================
class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name, num_aux=3, num_species=14, pretrained=False):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) 
            feats = self.backbone(dummy_input)
            self.backbone_dim = feats.shape[1]
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
            
        # 2. Auxiliary Head
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_aux)
        )
        
        # 3. Species Head (Fine-grained)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5), # Matched training dropout
            nn.Linear(64, num_species)
        )

        # 4. Taxonomy Head (Coarse-grained: Legume/Grass/Weed) - NEW
        self.taxonomy_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )
                
        # 5. Biomass Head
        # Inputs: Backbone + Aux(3) + Species(14) + Taxonomy(3)
        input_dim = self.backbone_dim + num_aux + num_species + 3
        
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # [Log_C, Log_D, Log_G, Log_T]
        )

    def forward(self, x):
        feat_map = self.backbone(x)
        img_feats = self.global_pool(feat_map).flatten(1)
        
        # Heads
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        taxonomy_logits = self.taxonomy_head(img_feats)
        taxonomy_probs = torch.softmax(taxonomy_logits, dim=1)
        
        aux_out = self.aux_head(img_feats) 
            
        # Fusion: Include Taxonomy Probs
        combined_feats = torch.cat([img_feats, aux_out, species_probs, taxonomy_probs], dim=1)
        
        # Log-Space Predictions
        log_preds_raw = self.biomass_head(combined_feats)
        log_preds = nn.functional.softplus(log_preds_raw)
        
        log_c = log_preds[:, 0:1]
        log_d = log_preds[:, 1:2]
        log_g = log_preds[:, 2:3]
        log_t = log_preds[:, 3:4]
        
        # Derived GDM
        c = torch.expm1(log_c)
        g = torch.expm1(log_g)
        log_gdm = torch.log1p(c + g + 1e-8)
        
        biomass_out = torch.cat([log_c, log_d, log_g, log_t, log_gdm], dim=1)
        
        # RETURN 4 VALUES
        return biomass_out, aux_out, species_logits, taxonomy_logits

# ====================== TTA HELPERS (INLINED) ======================
# ====================== TTA HELPERS (FIXED) ======================
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
    # 1. Robustly get Height and Width (last two dims)
    h, w = img.shape[-2:]
    
    # 2. Rotate (TF.rotate handles batches automatically)
    img_rot = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
    
    # 3. Calculate Valid Crop
    ch, cw = get_largest_rotated_crop(h, w, angle)
    
    # 4. Center Crop
    img_crop = TF.center_crop(img_rot, [ch, cw])
    
    # 5. Resize (Interpolate)
    # Check if batch (4D) or single (3D) to handle unsqueeze logic if needed
    if img.ndim == 3:
        # Single image: needs unsqueeze to be (1, C, H, W) for interpolate
        img_resized = torch.nn.functional.interpolate(
            img_crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
    else:
        # Batch (B, C, H, W): works directly
        img_resized = torch.nn.functional.interpolate(
            img_crop, size=(h, w), mode='bilinear', align_corners=False
        )
        
    return img_resized

def no_tta(model, image):
    """
    Single pass inference with PHYSICS BARRIER.
    Safe against exploding gradients or hallucinations.
    """
    model.eval()
    print("No TTA Inference")
    # 1. Forward Pass
    # log_bio: [Batch, 5]
    # tax_logits: [Batch, 3]
    log_bio, aux, sp, tax_logits = model(image)
    
    # 2. Convert to Linear Grams
    lin_bio = torch.expm1(log_bio)
    
    # 3. PHYSICS BARRIER 
    # Force values to be between 0g and 3000g (3kg).
    # This guarantees no '1e+25' errors.
    lin_bio = torch.clamp(lin_bio, min=0.0, max=3000.0)
    
    # 4. Convert back to Log Space
    # (Because your run_inference loop expects log inputs to perform the expm1 later)
    log_bio_clamped = torch.log1p(lin_bio)
    
    # 5. Extract Confidence (Softmax Fix)
    # Convert Raw Logits (e.g., 19.4) -> Probabilities (0.0 - 1.0)
    probs = torch.softmax(tax_logits, dim=1)
    conf, _ = torch.max(probs, dim=1)
    
    return log_bio_clamped, conf
    
def apply_tta(model, image):
    """
    TTA with HARD PHYSICS BARRIER.
    Prevents any single view from predicting mass > 20,000g (20kg).
    """
    model.eval()
    print("Applying TTA Inference")
    
    all_biomass_linear = [] 
    all_confidences = []

    # TTA Policy: 7 Views
    transforms_list = [
        lambda x: x,                           
        lambda x: torch.flip(x, [3]),          
        lambda x: torch.flip(x, [2]),          
        lambda x: rotate_crop_resize(x, 15),   
        lambda x: rotate_crop_resize(x, -15),  
        lambda x: rotate_crop_resize(x, 30),   
        lambda x: rotate_crop_resize(x, -30),  
    ]

    for t in transforms_list:
        with torch.no_grad():
            img_aug = t(image)
            
            # Forward Pass
            log_bio, aux, sp, tax_logits = model(img_aug) 
            
            # 1. Convert to Linear Grams
            lin_bio = torch.expm1(log_bio)
            
            # --- THE PHYSICS BARRIER ---
            # Anything above 3000g (3kg) in a 70cm plot is a black hole, not grass.
            # We clamp heavily here to stop 1e+25 from polluting the average.
            lin_bio = torch.clamp(lin_bio, min=0.0, max=2000.0)
            
            all_biomass_linear.append(lin_bio)
            
            # 2. Extract Confidence
            probs = torch.softmax(tax_logits, dim=1) 
            conf, _ = torch.max(probs, dim=1) 
            all_confidences.append(conf)

    # --- AGGREGATION ---
    
    # 1. Average Biomass (Linear Space)
    avg_bio_linear = torch.stack(all_biomass_linear).mean(0)
    
    # 2. Average Confidence
    avg_confidence = torch.stack(all_confidences).mean(0)
            
    # Return Linear directly to avoid log/exp conversions in the loop
    # We will log1p it only if we need to return log, but your loop expects linear now
    # Based on your previous code, let's return LOG to match signature, 
    # OR change the return to linear. 
    # Let's return LOG to match your 'run_inference' expectation:
    avg_bio_log = torch.log1p(avg_bio_linear)
            
    return avg_bio_log, avg_confidence

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
        img_path = os.path.join(self.img_dir, os.path.basename(row['image_path']))
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (512, 512)) # Fallback
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

def get_inference_transforms(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

def load_model(fold_path, device, num_species, backbone_name):
    model = BiomassUnifiedModel(backbone_name=backbone_name, num_species=num_species).to(device)
    state_dict = torch.load(fold_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ====================== MAIN INFERENCE ======================
def run_inference():
    print("="*60 + "\nGRANDMASTER INFERENCE (Weighted Ensemble)\n" + "="*60)
    
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
            if 'clean_id' not in df_wide.columns: df_wide['clean_id'] = df_wide['sample_id']
                 
    print(f"Test Images: {len(df_wide)}")
    
    # 2. LOAD METADATA
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json missing in {MODEL_DIR}")
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_species = metadata.get('num_species', 14)
    backbone_name = metadata.get('backbone')
    img_h = metadata.get('image_height', DEFAULT_HEIGHT)
    img_w = metadata.get('image_width', DEFAULT_WIDTH)
    print(f"Config: {backbone_name} | {img_h}x{img_w}")

    # 3. DISCOVER MODELS
    found_folds = []
    for f in range(10):
        p = os.path.join(MODEL_DIR, f"best_model_fold{f+1}.pth")
        if os.path.exists(p): found_folds.append(p)
            
    if not found_folds: 
        if os.path.exists(os.path.join(MODEL_DIR, "best_model_overall.pth")):
            found_folds.append(os.path.join(MODEL_DIR, "best_model_overall.pth"))
            
    if not found_folds: raise FileNotFoundError(f"No models found in {MODEL_DIR}")
    print(f"Found {len(found_folds)} checkpoints.")

    # 4. RUN INFERENCE LOOP
    val_transform = get_inference_transforms(h=img_h, w=img_w)
    ds = TestDataset(df_wide, TEST_IMG_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    ensemble_preds = []   # List of [N_Samples, 5]
    ensemble_confs = []   # List of [N_Samples]
    final_clean_ids = []
    
    for i, model_path in enumerate(found_folds):
        fold_name = os.path.basename(model_path)
        print(f"-> Processing {fold_name}...")
        model = load_model(model_path, DEVICE, num_species, backbone_name)
        
        fold_preds = []
        fold_confs = []
        
        with torch.no_grad():
            for imgs, ids in tqdm(loader, leave=False):
                imgs = imgs.to(DEVICE)
                
                # --- APPLY TTA ---
                # Returns Log Preds and Confidence Score
                log_pred, conf = no_tta(model, imgs)
                
                # Convert to Linear for averaging
                lin_pred = torch.expm1(log_pred)
                
                fold_preds.append(lin_pred.cpu().numpy())
                fold_confs.append(conf.cpu().numpy())
                
                if i == 0: final_clean_ids.extend(ids)
                    
        ensemble_preds.append(np.concatenate(fold_preds, axis=0))
        ensemble_confs.append(np.concatenate(fold_confs, axis=0))
    
    # Convert to Numpy for Analysis: [N_Models, N_Samples]
    W_raw = np.stack(ensemble_confs, axis=0)
    
    # ====================== ROUTER DIAGNOSTICS ======================
    print("\n" + "="*30 + " ROUTER DIAGNOSTICS " + "="*30)
    
    # 1. Who is winning?
    # Find which model index has max confidence for each sample
    winners = np.argmax(W_raw, axis=0) 
    # Print Stats
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
    print("Calculating Taxonomy-Weighted Ensemble...")
    
    # Shape: [N_Models, N_Samples, 5]
    E = np.stack(ensemble_preds, axis=0)
    # Shape: [N_Models, N_Samples]
    W = W_raw
    
    # Expand Weights for broadcasting: [N_Models, N_Samples, 1]
    W_expanded = W[:, :, np.newaxis]
    
    # Sharpen weights (Square them) to favor the expert model more heavily
    W_expanded = W_expanded ** 2
    
    # Weighted Average: Sum(Pred * Weight) / Sum(Weight)
    numerator = np.sum(E * W_expanded, axis=0)
    denominator = np.sum(W_expanded, axis=0) + 1e-8
    
    avg_preds_g = numerator / denominator
    avg_preds_g = np.maximum(avg_preds_g, 0) # Clip negatives
    
    # 6. EXPORT
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    final_df = pd.DataFrame(avg_preds_g, columns=target_cols)
    final_df['clean_id'] = final_clean_ids
    
    submission_rows = []
    for _, row in final_df.iterrows():
        cid = row['clean_id']
        for col in target_cols:
            submission_rows.append({
                'sample_id': f"{cid}__{col}", 
                'target': row[col]
            })
            
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv('submission.csv', index=False)
    print(sub_df)
    print(f"Saved {len(sub_df)} rows to submission.csv")
    


if __name__ == '__main__':
    run_inference()            
