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
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 512
FUSION_DIM = 256
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 

class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name='timm/tf_efficientnet_b3.ns_jft_in1k', num_aux=3, num_species=14, pretrained=False):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            feats = self.backbone(dummy_input)
            self.backbone_dim = feats.shape[1]
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
            
        # 2. Auxiliary Head (NDVI, LogHeight, Interaction)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_aux)
        )
        
        # Species Head
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_species)
        )
        
        # 4. Month Head (Sin/Cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )
        
        # 5. Biomass Head
        input_dim = self.backbone_dim + num_aux + num_species + 2
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
        month_logits = self.month_head(img_feats)
        aux_out = self.aux_head(img_feats) 
            
        combined_feats = torch.cat([img_feats, aux_out, species_probs, month_logits], dim=1)
        
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
        
        return biomass_out, aux_out, species_logits, month_logits

def get_inference_transforms(h=IMAGE_HEIGHT, w=IMAGE_WIDTH):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

# ====================== CONFIGURATION ======================
# Adjust these paths as needed for your local environment
TEST_CSV_PATH = './test.csv'  # Local path assumption
TEST_IMG_DIR = './test/' # Local path assumption
MODEL_DIR = './logs/ts_split_train_20251228_182529' # User must point this to the correct session
BATCH_SIZE = 32

# Interactive override if these don't exist
if not os.path.exists(TEST_CSV_PATH):
    print(f"Warning: {TEST_CSV_PATH} not found. Please ensure data is present.")

print(f"Device: {DEVICE}")

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
            print(f"Warning: Image not found: {img_path}")
            img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT)) # Fallback
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

def load_model(fold_path, device, num_species):
    """Load a single fold model with correct num_species."""
    model = BiomassUnifiedModel(num_species=num_species).to(device)
    state_dict = torch.load(fold_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference():
    print("="*70 + "\nBIOMASS UNIFIED MODEL INFERENCE (LOCAL)\n" + "="*70)
    
    # 1. LOAD TEST DATA
    print("\n[1/5] Loading test data...")
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV_PATH}")
        
    df = pd.read_csv(TEST_CSV_PATH)
    # Convert long to wide if needed
    if 'target_name' in df.columns:
        print("   Detected long format, extracting unique images...")
        df['clean_id'] = df['sample_id'].str.split('__').str[0]
        df_wide = df[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
    else:
        # Assume wide or simple list
        df_wide = df
        if 'clean_id' not in df_wide.columns and 'sample_id' in df_wide.columns:
             df_wide['clean_id'] = df_wide['sample_id'] # Fallback
             
    print(f"   Test Images: {len(df_wide)}")
    
    # 2. LOAD METADATA
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        print(f"\n[2/5] Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_species = metadata.get('num_species', 15)
        species_list = metadata.get('species_list', [])
        species_to_id = {s: i for i, s in enumerate(species_list)}
        print(f"   Detected {num_species} species classes.")
    else:
        print(f"\n[2/5] WARNING: metadata.json not found in {MODEL_DIR}. Using defaults.")
        num_species = 15
        species_to_id = {}

    # 3. DISCOVER MODELS
    print(f"\n[3/5] Discovering models in {MODEL_DIR}...")
    models = []
    # Look for best_model_foldX.pth or best_model_overall.pth
    # Priority: if overall exists, maybe just use that? Or ensemble folds if available.
    # Standard practice: Ensemble all folds found.
    
    found_folds = []
    for f in range(10):
        p = os.path.join(MODEL_DIR, f"best_model_fold{f+1}.pth") # train.py saves as fold+1
        if os.path.exists(p):
            found_folds.append(p)
            
    if not found_folds:
        # Try overall
        p = os.path.join(MODEL_DIR, "best_model_overall.pth")
        if os.path.exists(p):
            found_folds.append(p)
            
    if not found_folds:
         raise FileNotFoundError(f"No .pth models found in {MODEL_DIR}")
         
    print(f"   Found {len(found_folds)} model checkpoints.")

    # 4. RUN INFERENCE
    print(f"\n[4/5] Running inference...")
    
    img_h = metadata.get('image_height', IMAGE_HEIGHT)
    img_w = metadata.get('image_width', IMAGE_WIDTH)
    val_transform = get_inference_transforms(h=img_h, w=img_w)
    
    ds = TestDataset(df_wide, TEST_IMG_DIR, transform=val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # We will accumulate predictions: (N_Samples, 5_Targets)
    # Targets order in models.py Output: [Clover, Dead, Green, Total, GDM]
    # NOTE: Output is Log-Space (log1p(Kg)).
    
    ensemble_preds_kg = []
    final_clean_ids = []
    
    for i, model_path in enumerate(found_folds):
        print(f"   -> Processing {os.path.basename(model_path)}...")
        model = load_model(model_path, DEVICE, num_species)
        
        fold_preds = []
        
        with torch.no_grad():
            for imgs, ids in tqdm(loader, leave=False):
                imgs = imgs.to(DEVICE)
                
                # Forward
                # biomass_out channels: 0:C, 1:D, 2:G, 3:T, 4:GDM
                biomass_out, _, _, _ = model(imgs)
                
                # Convert Log-Space -> Linear KG
                # Model predicts log1p(x_kg)
                pred_kg = torch.expm1(biomass_out)
                
                fold_preds.append(pred_kg.cpu().numpy())
                
                # Capture IDs only during the first model's pass
                if i == 0:
                    final_clean_ids.extend(ids)
                    
        ensemble_preds_kg.append(np.concatenate(fold_preds, axis=0))
        
    # 5. ENSEMBLE (Average in Linear KG Space)
    print("\n[5/5] Averaging and post-processing...")
    avg_preds_kg = np.mean(ensemble_preds_kg, axis=0) # (N, 5)
    
    # 5. POST-PROCESSING (Kg -> Grams)
    # Competition expects Grams. 
    # Our model was trained on Kg/1000? No wait.
    # common.load_data: wide[target_cols] = wide[target_cols] / 1000.0
    # So training inputs were KG.
    # So model output expm1 is KG.
    # So we must multiply by 1000 to get Grams.
    
    avg_preds_g = avg_preds_kg * 1000.0
    
    # Clip negative
    avg_preds_g = np.maximum(avg_preds_g, 0)
    
    # 6. EXPORT
    print("\n[5/5] Saving submission...")
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Create final df
    final_df = pd.DataFrame(avg_preds_g, columns=target_cols)
    final_df['clean_id'] = final_clean_ids
    
    # Melanize to long format (sample_id, target)
    submission_rows = []
    for _, row in final_df.iterrows():
        cid = row['clean_id']
        for col in target_cols:
            sid = f"{cid}__{col}"
            submission_rows.append({'sample_id': sid, 'target': row[col]})
            
    sub_df = pd.DataFrame(submission_rows)
    out_file = 'submission.csv'
    sub_df.to_csv(out_file, index=False)
    print(f"   Saved to {out_file}")
    print(sub_df.head())

if __name__ == '__main__':
    
    run_inference()
