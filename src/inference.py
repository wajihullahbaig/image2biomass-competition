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
from torchvision import transforms

# ====================== CONFIGURATION ======================
TEST_CSV_PATH = './test.csv'
TEST_IMG_DIR = './test'
MODEL_DIR = './models_unified/'

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model hyperparameters (must match training config)
IMG_FEAT_WEIGHT = 0.3
TAB_FEAT_WEIGHT = 0.7
FUSION_DIM = 512

# Single fold model to load
FOLD_TO_LOAD = 1

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

def convert_test_to_wide(test_df):
    """
    Convert test.csv from long format to wide format (like training data).
    
    Input (long format):
        sample_id, image_path, target_name
        ID123__Dry_Clover_g, test/ID123.jpg, Dry_Clover_g
        ID123__Dry_Dead_g, test/ID123.jpg, Dry_Dead_g
        ...
    
    Output (wide format):
        clean_id, image_path
        ID123, test/ID123.jpg
    """
    # Extract clean ID from sample_id (remove __target_name suffix)
    test_df['clean_id'] = test_df['sample_id'].str.split('__').str[0]
    
    # Get unique images (one row per image, not per target)
    wide_test = test_df[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
    
    print(f"   Converted from {len(test_df)} rows (long) to {len(wide_test)} unique images (wide)")
    
    return wide_test

# ====================== UNIFIED MODEL DEFINITION ======================
class UnifiedSharedModel(nn.Module):
    """
    Unified model with shared backbone for both Stage 1 and Stage 2 predictions.
    This EXACTLY matches the training architecture.
    """
    def __init__(self, num_species, backbone_name):
        super().__init__()
        # 1. SHARED BACKBONE
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        
        # 2. STAGE 1 HEADS (Metadata predictions)
        self.s1_species = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, num_species))
        self.s1_ndvi = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 1))
        self.s1_height = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 1))
        self.s1_month = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 12))

        # 3. STAGE 2 HEAD (Biomass predictions with feature fusion)
        # Species Embedding (for S2 conditioning)
        self.species_emb = nn.Embedding(num_species, 16)
        
        self.img_adapter = nn.Sequential(
            nn.Linear(feat_dim, FUSION_DIM),
            nn.BatchNorm1d(FUSION_DIM),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        self.tab_adapter = nn.Sequential(
            nn.Linear(4, FUSION_DIM),  # 4 tabular features
            nn.BatchNorm1d(FUSION_DIM),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # Input to MLP is now FUSION_DIM + 16 (Species Emb)
        s2_input_dim = FUSION_DIM + 16

        self.s2_mlp = nn.Sequential(
            nn.Linear(s2_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Predicts log-scale components: C, D, G
        )

    def forward(self, img, species_idx=None):
        # --- Shared Forward Pass ---
        feats = self.backbone(img)

        # --- Stage 1 Outputs ---
        sp_logits = self.s1_species(feats)
        ndvi_pred = self.s1_ndvi(feats).squeeze(1)
        h_pred = self.s1_height(feats).squeeze(1)
        month_logits = self.s1_month(feats)
        
        # --- Differentiable Feature Engineering ---
        ndvi_vec = ndvi_pred
        h_vec = h_pred
        ndvi_h_mul = ndvi_vec * h_vec
        
        # Robust ratio calculation
        h_safe = F.relu(h_vec) + 0.1
        ndvi_h_ratio = ndvi_vec / h_safe
        
        # Stack tabular features
        tab_features = torch.stack([ndvi_vec, h_vec, ndvi_h_mul, ndvi_h_ratio], dim=1)
        
        # --- Stage 2 Feature Fusion ---
        img_emb = self.img_adapter(feats)
        tab_emb = self.tab_adapter(tab_features)
        
        # Weighted blend
        s2_main = (IMG_FEAT_WEIGHT * img_emb) + (TAB_FEAT_WEIGHT * tab_emb)
        
        # Determine Species for Conditioning (INFERENCE: Defaults to Prediction)
        if species_idx is not None:
             sp_emb = self.species_emb(species_idx)
        else:
             sp_pred_idx = torch.argmax(sp_logits, dim=1)
             sp_emb = self.species_emb(sp_pred_idx)
             
        s2_input = torch.cat([s2_main, sp_emb], dim=1)
        
        # --- Stage 2 Biomass Predictions ---
        log_components = F.softplus(self.s2_mlp(s2_input))
        log_components = torch.clamp(log_components, max=15.0)
        
        # Physics reconstruction (C + D + G = Total, C + G = GDM)
        l_c, l_d, l_g = log_components[:, 0:1], log_components[:, 1:2], log_components[:, 2:3]
        
        # Convert from log1p space to real grams
        r_c = torch.expm1(l_c)
        r_d = torch.expm1(l_d)
        r_g = torch.expm1(l_g)
        r_tot = r_c + r_d + r_g
        r_gdm = r_c + r_g
        
        # Concatenate all predictions: [Clover, Dead, Green, Total, GDM]
        bio_real_pred = torch.cat([r_c, r_d, r_g, r_tot, r_gdm], dim=1)
        
        # For compatibility, also return log predictions
        bio_log_pred = torch.cat([l_c, l_d, l_g, torch.log1p(r_tot), torch.log1p(r_gdm)], dim=1)
        
        return sp_logits, ndvi_pred, h_pred, month_logits, bio_log_pred, bio_real_pred

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
        img_path = os.path.join(self.img_dir, os.path.basename(row['image_path']))
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Create blank image if file not found
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            print(f"Warning: Image not found: {img_path}")
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

# ====================== MAIN INFERENCE LOGIC ======================
def run_inference():
    print("="*70 + "\nUNIFIED MODEL INFERENCE (Single Fold)\n" + "="*70)
    
    # 1. LOAD AND CONVERT TEST DATA TO WIDE FORMAT
    print("\n[1/5] Loading test data...")
    test_df_long = pd.read_csv(TEST_CSV_PATH)
    print(f"   Original test data: {len(test_df_long)} rows (long format)")
    
    # Convert to wide format (one row per image)
    test_df_wide = convert_test_to_wide(test_df_long)
    print(f"   Converted to wide format: {len(test_df_wide)} unique images")
    
    # 2. LOAD MODEL METADATA
    print("\n[2/5] Loading model metadata...")
    metadata_path = os.path.join(MODEL_DIR, 'unified_metadata.pth')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    metadata = torch.load(metadata_path, map_location=DEVICE, weights_only=False)
    species_le = metadata['species_encoder']
    num_species = metadata['num_species']
    backbone_name = metadata['backbone_s1']
    
    print(f"   Backbone: {backbone_name}")
    print(f"   Number of species classes: {num_species}")
    print(f"   Is shared backbone: {metadata.get('is_shared', False)}")
    
    # 3. LOAD SINGLE FOLD MODEL
    print(f"\n[3/5] Loading fold {FOLD_TO_LOAD} model...")
    
    fold_path = os.path.join(MODEL_DIR, f'fold{FOLD_TO_LOAD}.pth')
    
    if not os.path.exists(fold_path):
        raise FileNotFoundError(f"Model not found at {fold_path}")
    
    model = UnifiedSharedModel(num_species=num_species, backbone_name=backbone_name).to(DEVICE)
    model.load_state_dict(torch.load(fold_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"   ✓ Loaded Fold {FOLD_TO_LOAD} successfully")
    
    # 4. RUN INFERENCE ON WIDE FORMAT DATA
    print(f"\n[4/5] Running inference...")
    transform = get_test_transform()
    dataset = TestDataset(test_df_wide, TEST_IMG_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    all_preds = []
    all_clean_ids = []
    
    with torch.no_grad():
        for img, clean_ids in tqdm(loader, desc="Inference"):
            img = img.to(DEVICE)
            
            # Single model prediction
            _, _, _, _, _, bio_real_pred = model(img)
            
            all_preds.append(bio_real_pred.cpu().numpy())
            all_clean_ids.extend(clean_ids)
    
    print(f"   ✓ Inference complete. Total images processed: {len(all_clean_ids)}")
    
    # 5. CREATE SUBMISSION IN KAGGLE FORMAT
    print("\n[5/5] Creating submission file...")
    
    # Target column names
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Concatenate all predictions
    preds_array = np.concatenate(all_preds, axis=0)
    
    # Clip negative values (safety check)
    preds_array = np.maximum(preds_array, 0)
    
    # Create wide predictions dataframe (one row per image)
    preds_wide = pd.DataFrame(preds_array, columns=target_cols)
    preds_wide['clean_id'] = all_clean_ids
    
    print(f"   Wide predictions shape: {preds_wide.shape}")
    
    # Convert predictions to long format to match submission requirements
    # Each image should have 5 rows (one per target)
    submission_rows = []
    
    for _, row in preds_wide.iterrows():
        clean_id = row['clean_id']
        for target_col in target_cols:
            sample_id = f"{clean_id}__{target_col}"
            target_value = row[target_col]
            submission_rows.append({
                'sample_id': sample_id,
                'target': target_value
            })
    
    # Create final submission dataframe
    submission_df = pd.DataFrame(submission_rows)
    
    # Ensure we have all the expected sample_ids from the original test.csv
    expected_sample_ids = set(test_df_long['sample_id'].unique())
    actual_sample_ids = set(submission_df['sample_id'].unique())
    
    missing = expected_sample_ids - actual_sample_ids
    if missing:
        print(f"   ⚠ Warning: {len(missing)} sample_ids are missing from predictions")
        print(f"   Missing examples: {list(missing)[:5]}")
    
    extra = actual_sample_ids - expected_sample_ids
    if extra:
        print(f"   ⚠ Warning: {len(extra)} extra sample_ids in predictions")
        print(f"   Extra examples: {list(extra)[:5]}")
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    
    print("="*70 + "\n✅ SUBMISSION CREATED SUCCESSFULLY!\n" + "="*70)
    print("\nSubmission format (first 10 rows):")
    print(submission_df.head(10))
    print(f"\nTotal rows in submission: {len(submission_df)}")
    print(f"Expected rows: {len(test_df_long)}")
    print(f"Submission file: submission.csv")
    
    # Print basic statistics
    print("\n--- Prediction Statistics (Per Target) ---")
    for target_col in target_cols:
        target_preds = preds_wide[target_col]
        print(f"{target_col:20s}: min={target_preds.min():8.2f}, max={target_preds.max():8.2f}, mean={target_preds.mean():8.2f}")
    
    # Verify physics constraints
    print("\n--- Physics Constraints Verification ---")
    total_calc = preds_wide['Dry_Clover_g'] + preds_wide['Dry_Dead_g'] + preds_wide['Dry_Green_g']
    total_diff = (preds_wide['Dry_Total_g'] - total_calc).abs().mean()
    print(f"Total vs (C+D+G) difference: {total_diff:.6f}g (should be ~0)")
    
    gdm_calc = preds_wide['Dry_Clover_g'] + preds_wide['Dry_Green_g']
    gdm_diff = (preds_wide['GDM_g'] - gdm_calc).abs().mean()
    print(f"GDM vs (C+G) difference: {gdm_diff:.6f}g (should be ~0)")

if __name__ == '__main__':
    run_inference()