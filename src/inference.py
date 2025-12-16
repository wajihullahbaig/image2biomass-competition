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
from torchvision import transforms

# ====================== CONFIGURATION ======================
TEST_CSV_PATH = './test.csv'
TEST_IMG_DIR = './test'
STAGE1_MODEL_DIR = './models_stage1/'
STAGE2_MODEL_PATH = './models_stage2/best_model.pth'

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_S1 = 'tf_efficientnet_b3_ns'
N_FOLDS = 5
BACKBONE_S2 = 'swin_base_patch4_window7_224'
USE_COUNT_FEATURES = False

print(f"Device: {DEVICE}")

# ====================== HELPER FUNCTIONS ======================
def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    if month in [3, 4, 5]: return 'Autumn'
    if month in [6, 7, 8]: return 'Winter'
    return 'Spring'

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ====================== STAGE 1 MODEL DEFINITION ======================
class Stage1Model(nn.Module):
    """Multi-task model predicting Species, NDVI, Height, Month"""
    def __init__(self, num_species, num_months=12):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_S1, pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        
        # Multi-Heads
        self.species_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, num_species))
        self.ndvi_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, 1))
        self.height_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, 1))
        self.month_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, num_months))

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.species_head(f),
            self.ndvi_head(f).squeeze(1),
            self.height_head(f).squeeze(1),
            self.month_head(f)
        )

# ====================== STAGE 2 MODEL DEFINITION ======================
class Stage2ModelLog(nn.Module):
    def __init__(self, tab_dim, stage_index=1):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_S2, pretrained=False, features_only=True, out_indices=(stage_index,)
        )
        img_feature_size = self.backbone.feature_info[stage_index]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(img_feature_size + tab_dim, 512),
            nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.SiLU(),
        )
        self.head = nn.Linear(256, 3)

    def forward(self, img, tab):
        f = self.backbone(img)[0]
        if f.dim() == 4 and f.shape[1] != self.backbone.feature_info[0]['num_chs']:
            f = f.permute(0, 3, 1, 2)
        f = self.pool(f).squeeze(-1).squeeze(-1)
        x = torch.cat([f, tab], dim=1)
        log_comp = F.softplus(self.head(self.mlp(x)))
        l_c, l_d, l_g = log_comp[:, 0:1], log_comp[:, 1:2], log_comp[:, 2:3]
        r_c, r_d, r_g = torch.expm1(l_c), torch.expm1(l_d), torch.expm1(l_g)
        r_tot, r_gdm = r_c + r_d + r_g, r_c + r_g
        return torch.cat([r_c, r_d, r_g, r_tot, r_gdm], dim=1)

# ====================== TEST DATASET DEFINITION ======================
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
        img_path = os.path.join(self.img_dir, os.path.basename(row['image_path']))
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            img = self.transform(img)
        
        if self.tabular_cols:
            tab_vals = row[self.tabular_cols].values.astype(np.float32)
            return img, torch.from_numpy(tab_vals), row['sample_id']
        
        return img, row['sample_id']

# ====================== MAIN INFERENCE LOGIC ======================
def run_inference():
    print("="*70 + "\nSTARTING TWO-STAGE BIOMASS PREDICTION\n" + "="*70)
    
    # 1. LOAD DATA
    print("\n[1/6] Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    unique_images_df = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    print(f"   Found {len(test_df)} total rows, corresponding to {len(unique_images_df)} unique images.")
    transform = get_test_transform()

    # 2. LOAD STAGE 1 MODELS (ALL FOLDS FOR ENSEMBLE)
    print("\n[2/6] Loading Stage 1 models...")
    metadata_path = os.path.join(STAGE1_MODEL_DIR, 'stage1_metadata.pth')
    metadata = torch.load(metadata_path, map_location=DEVICE, weights_only=False)
    species_le, num_species = metadata['species_encoder'], metadata['num_species']
    print(f"   Loaded metadata. Found {num_species} species classes.")
    
    fold_models = []
    for fold in range(1, N_FOLDS + 1):
        fold_path = os.path.join(STAGE1_MODEL_DIR, f'stage1_fold{fold}.pth')
        if os.path.exists(fold_path):
            model = Stage1Model(num_species=num_species).to(DEVICE)
            model.load_state_dict(torch.load(fold_path, map_location=DEVICE, weights_only=True))
            model.eval()
            fold_models.append(model)
            print(f"   ✓ Loaded Fold {fold}")
        else:
            print(f"   ⚠ Fold {fold} not found at {fold_path}")
    
    if len(fold_models) == 0:
        raise FileNotFoundError("No Stage 1 fold models found!")
    
    print(f"   Total folds loaded: {len(fold_models)}")
    
    # 3. RUN STAGE 1 INFERENCE (ENSEMBLE ACROSS FOLDS)
    print(f"\n[3/6] Running Stage 1 Inference (ensemble across {len(fold_models)} folds)...")
    ds_s1 = TestDataset(unique_images_df, TEST_IMG_DIR, transform=transform)
    loader_s1 = DataLoader(
        ds_s1, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True
    )
    
    preds = {'species': [], 'ndvi': [], 'height': [], 'month': []}
    
    with torch.no_grad():
        for img, _ in tqdm(loader_s1, desc="Stage 1"):
            img = img.to(DEVICE)
            
            # Ensemble predictions across all folds (average)
            species_preds = torch.stack([m(img)[0] for m in fold_models]).mean(0).cpu()
            ndvi_preds = torch.stack([m(img)[1] for m in fold_models]).mean(0).cpu()
            height_preds = torch.stack([m(img)[2] for m in fold_models]).mean(0).cpu()
            month_preds = torch.stack([m(img)[3] for m in fold_models]).mean(0).cpu()
            
            preds['species'].append(species_preds)
            preds['ndvi'].append(ndvi_preds)
            preds['height'].append(height_preds)
            preds['month'].append(month_preds)

    # Aggregate predictions
    unique_images_df['pred_species_idx'] = torch.argmax(torch.cat(preds['species']), 1).numpy()
    unique_images_df['pred_species'] = species_le.inverse_transform(unique_images_df['pred_species_idx'])
    unique_images_df['pred_ndvi'] = torch.cat(preds['ndvi']).numpy()
    unique_images_df['pred_height_log'] = torch.cat(preds['height']).numpy()
    unique_images_df['pred_month'] = torch.argmax(torch.cat(preds['month']), 1).numpy() + 1
    unique_images_df['pred_season'] = unique_images_df['pred_month'].apply(get_season)
    
    print(f"   ✓ Stage 1 predictions complete.")

    # 4. PREPARE STAGE 2 FEATURES
    print("\n[4/6] Preparing Stage 2 features...")
    merge_cols = ['image_path', 'pred_species', 'pred_ndvi', 'pred_height_log', 'pred_month', 'pred_season']
    test_df = test_df.merge(unique_images_df[merge_cols], on='image_path', how='left')
    
    # Feature engineering (matching your local code)
    test_df['NDVI_final'] = test_df['pred_ndvi']
    test_df['Height_final_log'] = test_df['pred_height_log']
    test_df['ndvi_h_mul'] = test_df['NDVI_final'] * test_df['Height_final_log']
    test_df['ndvi_h_ratio'] = test_df['NDVI_final'] / (test_df['Height_final_log'] + 1e-6)
   
    tab_cols = ['NDVI_final', 'Height_final_log', 'ndvi_h_mul', 'ndvi_h_ratio']
    
    # Optional count features
    if USE_COUNT_FEATURES:
        test_df['pred_species_count'] = test_df['pred_species'].map(
            np.log1p(test_df['pred_species'].value_counts())
        ).fillna(0)
        tab_cols.append('pred_species_count')
    
    # Ensure all tabular columns are numeric and handle NaN
    for col in tab_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0).astype(np.float32)
    
    print(f"   ✓ Using {len(tab_cols)} tabular features for Stage 2: {tab_cols}")

    # 5. RUN STAGE 2 INFERENCE
    print(f"\n[5/6] Running Stage 2 Inference...")
    model_s2 = Stage2ModelLog(tab_dim=len(tab_cols), stage_index=1).to(DEVICE)
    model_s2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model_s2.eval()
    
    ds_s2 = TestDataset(test_df, TEST_IMG_DIR, tabular_cols=tab_cols, transform=transform)
    loader_s2 = DataLoader(
        ds_s2, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True
    )
    
    all_preds, all_ids = [], []
    
    with torch.no_grad():
        for img, tab, sample_ids in tqdm(loader_s2, desc="Stage 2"):
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            preds_batch = model_s2(img, tab).cpu().numpy()
            all_preds.append(preds_batch)
            all_ids.extend(sample_ids)
    
    print(f"   ✓ Stage 2 predictions complete. Total predictions: {len(all_ids)}")

    # 6. CREATE SUBMISSION IN CORRECT FORMAT
    print("\n[6/6] Creating submission file...")
    
    # Target column names
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Create predictions dataframe
    preds_df = pd.DataFrame(np.concatenate(all_preds, axis=0), columns=target_cols)
    preds_df['sample_id'] = all_ids
    
    # Extract target_name from sample_id (format: ID####__TargetName)
    preds_df['target_name'] = preds_df['sample_id'].str.split('__').str[1]
    
    # Convert to long format
    submission_long = preds_df.melt(
        id_vars=['sample_id', 'target_name'], 
        value_vars=target_cols, 
        var_name='predicted_target', 
        value_name='target'
    )
    
    # Filter to match sample_id target with predicted target
    final_submission = submission_long[
        submission_long['target_name'] == submission_long['predicted_target']
    ].copy()
    
    # Clip negative values
    final_submission['target'] = final_submission['target'].clip(lower=0)
    
    # CRITICAL: Save only sample_id and target columns (Kaggle format)
    final_submission[['sample_id', 'target']].to_csv('submission.csv', index=False)
    
    print("="*70 + "\n✅ SUBMISSION CREATED SUCCESSFULLY!\n" + "="*70)
    print("\nSubmission format (first 10 rows):")
    print(final_submission[['sample_id', 'target']].head(10))
    print(f"\nTotal rows in submission: {len(final_submission)}")
    print(f"Expected format: sample_id, target")
    print(f"Actual columns saved: {list(final_submission[['sample_id', 'target']].columns)}")

if __name__ == '__main__':
    run_inference()