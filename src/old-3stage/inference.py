import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ====================== CONFIG ======================
# UPDATE THESE PATHS FOR KAGGLE
TEST_CSV_PATH = '/kaggle/input/image2biomass/test.csv'
TEST_IMG_DIR = '/kaggle/input/image2biomass/test_images' 
STAGE1_PATH = '/kaggle/input/your-model-dataset/stage1_package.pth'
STAGE2_PATH = '/kaggle/input/your-model-dataset/stage2_package.pth'

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== HELPER FUNCTIONS ======================
def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    elif month in [3, 4, 5]: return 'Autumn'
    elif month in [6, 7, 8]: return 'Winter'
    else: return 'Spring'

def get_test_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# ====================== MODEL DEFINITIONS ======================
class Stage1Model(nn.Module):
    def __init__(self, num_species, num_months=12, backbone_name='tf_efficientnet_b3_ns'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.species_head = nn.Linear(feat, num_species)
        self.ndvi_head = nn.Linear(feat, 1)
        self.height_head = nn.Linear(feat, 1)
        self.month_head = nn.Linear(feat, num_months)

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.species_head(f),
            self.ndvi_head(f).squeeze(1),
            self.height_head(f).squeeze(1),
            self.month_head(f)
        )

class Stage2Model(nn.Module):
    def __init__(self, tab_size, backbone_name='swin_base_patch4_window7_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        img_feat = self.backbone.num_features
        
        # Match training architecture exactly
        self.mlp = nn.Sequential(
            nn.Linear(img_feat + tab_size, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, 3)

    def forward(self, img, tab):
        f = self.backbone(img)
        if len(f.shape) > 2: f = f.mean([2, 3])
        x = torch.cat([f, tab], dim=1)
        feat = self.mlp(x)
        components = F.softplus(self.head(feat))
        
        clover = components[:, 0:1]
        dead   = components[:, 1:2]
        green  = components[:, 2:3]
        total = clover + dead + green
        gdm   = clover + green
        return torch.cat([clover, dead, green, total, gdm], dim=1)

# ====================== TEST DATASET ======================
class TestDataset(Dataset):
    def __init__(self, df, img_dir, tabular_cols=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tabular_cols = tabular_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Robust Image Path Logic
        # 1. Try 'image_path' column
        if 'image_path' in row and pd.notna(row['image_path']):
            img_name = str(row['image_path']).split('/')[-1]
        else:
            # 2. Fallback: Assume filename is {sample_id}.jpg or similar
            # Adjust extension if necessary (.png, .jpeg)
            img_name = f"{row['sample_id']}.jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Fallback for broken/missing images (prevents submission crash)
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            img = self.transform(img)

        # Stage 2 Mode
        if self.tabular_cols is not None:
            tab_vals = row[self.tabular_cols].values.astype(np.float32)
            return img, torch.from_numpy(tab_vals), row['sample_id']
        
        # Stage 1 Mode
        return img, row['sample_id']

# ====================== MAIN INFERENCE ======================
def run_inference():
    print("Loading Test Data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    transform = get_test_transform()

    # ================= STAGE 1 =================
    print("Loading Stage 1 Model...")
    s1_pkg = torch.load(STAGE1_PATH, map_location=DEVICE, weights_only=False)
    species_le = s1_pkg['species_encoder']
    
    model_s1 = Stage1Model(
        num_species=len(species_le.classes_), 
        backbone_name=s1_pkg.get('backbone_name', 'tf_efficientnet_b3_ns')
    ).to(DEVICE)
    model_s1.load_state_dict(s1_pkg['model_state_dict'])
    model_s1.eval()
    
    print("Running Stage 1 Inference (Predicting Metadata)...")
    ds_s1 = TestDataset(test_df, TEST_IMG_DIR, transform=transform)
    loader_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    preds_s1 = {'species': [], 'ndvi': [], 'height_log': [], 'month': []}
    
    with torch.no_grad():
        for img, _ in tqdm(loader_s1):
            img = img.to(DEVICE)
            sp, nd, h, m = model_s1(img)
            
            # Decode predictions
            sp_idx = torch.argmax(sp, 1).cpu().numpy()
            preds_s1['species'].extend(species_le.inverse_transform(sp_idx))
            preds_s1['ndvi'].extend(nd.cpu().numpy())
            preds_s1['height_log'].extend(h.cpu().numpy())
            preds_s1['month'].extend((torch.argmax(m, 1).cpu().numpy() + 1))
            
    # --- FILL METADATA ---
    # Since we have no date, we rely 100% on the predicted month
    test_df['pred_season'] = [get_season(m) for m in preds_s1['month']]
    test_df['season_final'] = test_df['pred_season'] # Predicted season is the final season
    
    test_df['Species_final'] = preds_s1['species']
    test_df['NDVI_final'] = preds_s1['ndvi']
    test_df['Height_final_log'] = preds_s1['height_log']

    # ================= STAGE 2 =================
    print("Loading Stage 2 Model...")
    s2_pkg = torch.load(STAGE2_PATH, map_location=DEVICE, weights_only=False)
    required_cols = s2_pkg['tabular_cols']
    
    # 1. Feature Engineering (Test Set)
    test_df['ndvi_h_mul'] = test_df['NDVI_final'] * test_df['Height_final_log']
    test_df['ndvi_h_ratio'] = test_df['NDVI_final'] / (test_df['Height_final_log'] + 1e-6)
    
    # 2. Count Features (Approximation using PREDICTED values)
    # This matches the training distribution logic dynamically
    needs_counts = any('count' in c or 'freq' in c for c in required_cols)
    if needs_counts:
        print("Generating Count Features from Predictions...")
        
        # Global Counts (based on current test set predictions)
        g_counts = test_df['Species_final'].value_counts()
        test_df['species_count_global'] = test_df['Species_final'].map(lambda x: np.log1p(g_counts.get(x, 0)))
        test_df['species_freq_global'] = test_df['Species_final'].map(lambda x: g_counts.get(x, 0) / len(test_df))
        
        # Seasonal Counts
        l_counts = test_df.groupby(['season_final', 'Species_final']).size()
        test_df['species_count_season'] = test_df.apply(
            lambda r: np.log1p(l_counts.get((r['season_final'], r['Species_final']), 0)), axis=1
        )
        season_totals = test_df['season_final'].value_counts()
        test_df['species_freq_season'] = test_df.apply(
            lambda r: l_counts.get((r['season_final'], r['Species_final']), 0) / season_totals.get(r['season_final'], 1), 
            axis=1
        )

    # 3. Clean Tabular Inputs
    print(f"Preparing {len(required_cols)} tabular features...")
    for col in required_cols:
        if col not in test_df.columns:
            print(f"Warning: Feature {col} missing in Test DF. Filling 0.")
            test_df[col] = 0.0
        # Force float32
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0).astype(np.float32)

    # 4. Inference
    model_s2 = Stage2Model(
        tab_size=len(required_cols),
        backbone_name=s2_pkg.get('backbone_name', 'swin_base_patch4_window7_224')
    ).to(DEVICE)
    model_s2.load_state_dict(s2_pkg['model_state_dict'])
    model_s2.eval()
    
    ds_s2 = TestDataset(test_df, TEST_IMG_DIR, tabular_cols=required_cols, transform=transform)
    loader_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    all_preds = []
    all_ids = []
    
    print("Running Stage 2 Inference...")
    with torch.no_grad():
        for img, tab, sample_ids in tqdm(loader_s2):
            img = img.to(DEVICE)
            tab = tab.to(DEVICE)
            pred = model_s2(img, tab) # [B, 5]
            all_preds.append(pred.cpu().numpy())
            all_ids.extend(sample_ids)
            
    # ================= SUBMISSION =================
    print("Saving Submission...")
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    preds_arr = np.concatenate(all_preds, axis=0)
    
    sub_df = pd.DataFrame(preds_arr, columns=target_cols)
    sub_df.insert(0, 'sample_id', all_ids)
    
    # Clip to 0 (No negative mass)
    sub_df[target_cols] = sub_df[target_cols].clip(lower=0)
    
    sub_df.to_csv('submission.csv', index=False)
    print("submission.csv saved successfully!")
    print(sub_df.head())

if __name__ == '__main__':
    run_inference()