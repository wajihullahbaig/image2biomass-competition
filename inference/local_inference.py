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

# ====================== INLINED CONFIG & MODEL ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# These defaults are fallback; metadata.json takes precedence
DEFAULT_HEIGHT = 224
DEFAULT_WIDTH = 512
FUSION_DIM = 256
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32

class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name, num_aux=3, num_species=14, pretrained=False):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            # Use a generic size to determine feature dim; exact input size doesn't change channel count
            dummy_input = torch.randn(1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH) 
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
        
        # 3. Species Head
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_species)
        )
                
        # 4. Biomass Head
        input_dim = self.backbone_dim + num_aux + num_species
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # [Log_Clover, Log_Dead, Log_Green, Log_Total]
        )

    def forward(self, x):
        feat_map = self.backbone(x)
        img_feats = self.global_pool(feat_map).flatten(1)
        
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        aux_out = self.aux_head(img_feats) 
            
        combined_feats = torch.cat([img_feats, aux_out, species_probs], dim=1)
        
        # Log-Space Predictions
        log_preds_raw = self.biomass_head(combined_feats)
        log_preds = nn.functional.softplus(log_preds_raw)
        
        log_c = log_preds[:, 0:1]
        log_d = log_preds[:, 1:2]
        log_g = log_preds[:, 2:3]
        log_t = log_preds[:, 3:4]
        
        # Derive Log(GDM) = Log(1 + C + G)
        c = torch.expm1(log_c)
        g = torch.expm1(log_g)
        log_gdm = torch.log1p(c + g + 1e-8)
        
        biomass_out = torch.cat([log_c, log_d, log_g, log_t, log_gdm], dim=1)
        
        return biomass_out, aux_out, species_logits

def get_inference_transforms(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

# ====================== CONFIGURATION ======================
TEST_CSV_PATH = './test.csv'  
TEST_IMG_DIR = './test/' 
# UPDATE THIS PATH TO YOUR SESSION FOLDER
MODEL_DIR = './logs/ts_split_train_phy_20251229_235832' 

if not os.path.exists(TEST_CSV_PATH):
    print(f"Warning: {TEST_CSV_PATH} not found.")

# ====================== TEST DATASET ======================
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.basename(row['image_path'])
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback for missing images (rare)
            print(f"Warning: Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

def load_model(fold_path, device, num_species, backbone_name):
    """Load a single fold model with correct config."""
    model = BiomassUnifiedModel(backbone_name=backbone_name, num_species=num_species).to(device)
    # weights_only=True is safer, but ensure timm version matches
    state_dict = torch.load(fold_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference():
    print("="*70 + "\nBIOMASS UNIFIED MODEL INFERENCE\n" + "="*70)
    
    # 1. LOAD TEST DATA
    print("\n[1/5] Loading test data...")
    if not os.path.exists(TEST_CSV_PATH):
        print("Test CSV not found. Creating dummy for dry run...")
        # Create dummy df for testing script logic if file missing
        df_wide = pd.DataFrame({'sample_id': ['test_1'], 'image_path': ['test_1.jpg'], 'clean_id': ['test_1']})
    else:
        df = pd.read_csv(TEST_CSV_PATH)
        if 'target_name' in df.columns:
            print("   Detected long format, extracting unique images...")
            df['clean_id'] = df['sample_id'].str.split('__').str[0]
            df_wide = df[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
        else:
            df_wide = df
            if 'clean_id' not in df_wide.columns and 'sample_id' in df_wide.columns:
                 df_wide['clean_id'] = df_wide['sample_id']
                 
    print(f"   Test Images: {len(df_wide)}")
    
    # 2. LOAD METADATA
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        print(f"\n[2/5] Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_species = metadata.get('num_species')
        backbone_name = metadata.get('backbone')
        img_h = metadata.get('image_height', DEFAULT_HEIGHT)
        img_w = metadata.get('image_width', DEFAULT_WIDTH)
        print(f"   Config: Backbone={backbone_name}, Species={num_species}, H={img_h}, W={img_w}")
    else:
        print(f"\n[2/5] WARNING: metadata.json not found in {MODEL_DIR}. Using defaults.")
        raise FileNotFoundError(f"metadata.json not found in {MODEL_DIR}")

    # 3. DISCOVER MODELS
    print(f"\n[3/5] Discovering models in {MODEL_DIR}...")
    found_folds = []
    # Check for fold models
    for f in range(10):
        p = os.path.join(MODEL_DIR, f"best_model_fold{f+1}.pth")
        if os.path.exists(p):
            found_folds.append(p)
            
    # If no folds, check for overall best
    if not found_folds:
        p = os.path.join(MODEL_DIR, "best_model_overall.pth")
        if os.path.exists(p):
            found_folds.append(p)
            
    if not found_folds:
         raise FileNotFoundError(f"No .pth models found in {MODEL_DIR}")
         
    print(f"   Found {len(found_folds)} model checkpoints.")

    # 4. RUN INFERENCE
    print(f"\n[4/5] Running inference...")
    
    val_transform = get_inference_transforms(h=img_h, w=img_w)
    
    ds = TestDataset(df_wide, TEST_IMG_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    ensemble_preds_g = []
    final_clean_ids = []
    
    for i, model_path in enumerate(found_folds):
        print(f"   -> Processing model {i+1}/{len(found_folds)}: {os.path.basename(model_path)}")
        model = load_model(model_path, DEVICE, num_species, backbone_name)
        
        fold_preds = []
        
        with torch.no_grad():
            for imgs, ids in tqdm(loader, leave=False):
                imgs = imgs.to(DEVICE)
                
                # FIXED: Unpack 3 values, not 4
                biomass_out, _, _ = model(imgs)
                
                # Convert Log-Space -> Linear Grams
                # Model output is log1p(grams)
                pred_g = torch.expm1(biomass_out)
                
                fold_preds.append(pred_g.cpu().numpy())
                
                if i == 0:
                    final_clean_ids.extend(ids)
                    
        ensemble_preds_g.append(np.concatenate(fold_preds, axis=0))
        
    # 5. ENSEMBLE (Average in Linear Space)
    print("\n[5/5] Averaging and post-processing...")
    avg_preds_g = np.mean(ensemble_preds_g, axis=0) # (N, 5)
    
    # Clip negative values (just in case)
    avg_preds_g = np.maximum(avg_preds_g, 0)
    
    # 6. EXPORT
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    final_df = pd.DataFrame(avg_preds_g, columns=target_cols)
    final_df['clean_id'] = final_clean_ids
    
    # Convert to Submission Format (Long)
    submission_rows = []
    for _, row in final_df.iterrows():
        cid = row['clean_id']
        for col in target_cols:
            sid = f"{cid}__{col}"
            submission_rows.append({'sample_id': sid, 'target': row[col]})
            
    sub_df = pd.DataFrame(submission_rows)
    out_file = 'submission.csv'
    sub_df.to_csv(out_file, index=False)
    print(f"   Saved {len(sub_df)} rows to {out_file}")
    print(sub_df.head())

if __name__ == '__main__':
    run_inference() 