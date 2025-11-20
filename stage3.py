import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import joblib
import os
from tqdm import tqdm

# --- CONFIG ---
# Must match Stage 1 training config
BACKBONE_SIZE = 'b3' 
MODEL_NAME_S1 = f'tf_efficientnet_{BACKBONE_SIZE}_ns'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32

# --- MODEL CLASSES ---

class InputPredictorModel(nn.Module):
    def __init__(self, num_species, model_name=MODEL_NAME_S1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        
        # Head 1: Species Classification
        self.species_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_species)
        )

        # Head 2: Regression (NDVI & Log-Height)
        self.reg_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2) 
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.species_head(features), self.reg_head(features)

class MultiModalModel(nn.Module):
    def __init__(self, tab_size, output_size=5):
        super().__init__()
        # This must match Stage 2 model name
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        img_dim = self.backbone.num_features
        
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + tab_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, img, tab):
        img_feat = self.backbone(img)
        # Flatten if output is not already 1D (depending on model)
        if len(img_feat.shape) > 2:
            img_feat = img_feat.mean(dim=[2, 3])
            
        combined = torch.cat([img_feat, tab], dim=1)
        return self.mlp(combined)

class InferenceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['sample_id']

# --- MAIN INFERENCE ---
def run_inference():
    print("--- Starting Inference Pipeline ---")
    print(f"Using Stage 1 Backbone: {MODEL_NAME_S1}")
    
    # 1. Load Test Data (Collapse to unique images)
    test_csv = pd.read_csv('test.csv')
    # The test.csv has multiple rows per sample_id (one for each target name).
    # We only need to predict inputs once per sample_id.
    df_unique = test_csv.drop_duplicates(subset=['sample_id']).copy().reset_index(drop=True)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- STAGE 1: PREDICT MISSING INPUTS ---
    print("1. Predicting Missing Inputs (Stage 1)...")
    
    le = joblib.load('stage1_species_encoder.pkl')
    model_s1 = InputPredictorModel(len(le.classes_)).to(DEVICE)
    
    # Load weights
    try:
        model_s1.load_state_dict(torch.load('stage1_model.pth', map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading Stage 1 model: {e}")
        print("Ensure BACKBONE_SIZE matches the trained model.")
        return

    model_s1.eval()
    
    ds_s1 = InferenceDataset(df_unique, 'test', transform)
    loader_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE, shuffle=False)
    
    species_preds = []
    ndvi_preds = []
    height_preds = []
    
    with torch.no_grad():
        for imgs, _ in tqdm(loader_s1):
            imgs = imgs.to(DEVICE)
            s_logits, r_out = model_s1(imgs)
            
            # Species
            s_idx = torch.argmax(s_logits, dim=1).cpu().numpy()
            species_preds.extend(le.inverse_transform(s_idx))
            
            # Regression
            r_out = r_out.cpu().numpy()
            ndvi_preds.extend(r_out[:, 0])
            # Reverse log transform for height
            height_preds.extend(np.expm1(r_out[:, 1]))
            
    df_unique['Species'] = species_preds
    df_unique['Pre_GSHH_NDVI'] = ndvi_preds
    df_unique['Height_Ave_cm'] = np.maximum(height_preds, 0.1) # Ensure non-negative
    
    print("   Inputs predicted.")
    
    # --- FEATURE ENGINEERING (Match Stage 2) ---
    print("2. Feature Engineering...")
    
    # Date Features
    # NOTE: If test.csv doesn't have Sampling_Date, you cannot run this model. 
    # Assuming it exists as per prompt context.
    df_unique['Sampling_Date'] = pd.to_datetime(df_unique['Sampling_Date'])
    df_unique['month'] = df_unique['Sampling_Date'].dt.month
    df_unique['month_sin'] = np.sin(2 * np.pi * df_unique['month'] / 12)
    df_unique['month_cos'] = np.cos(2 * np.pi * df_unique['month'] / 12)
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        else: return 'Spring'
    df_unique['season'] = df_unique['month'].apply(get_season)
    
    # Interactions
    df_unique['NDVI_Height_MUL'] = df_unique['Pre_GSHH_NDVI'] * df_unique['Height_Ave_cm']
    df_unique['NDVI_Height_ADD'] = df_unique['Pre_GSHH_NDVI'] + df_unique['Height_Ave_cm']
    df_unique['NDVI_Height_Ratio'] = df_unique['Pre_GSHH_NDVI'] / (df_unique['Height_Ave_cm'] + 1e-5)
    
    # --- STAGE 2: PREDICT BIOMASS ---
    print("3. Predicting Biomass (Stage 2)...")
    
    preprocessor = joblib.load('stage2_preprocessor.pkl')
    
    # Transform tabular features
    # IMPORTANT: Columns must match order used in fit. ColumnTransformer does this by name if passed DF, 
    # but let's ensure strict compliance.
    X_tab = preprocessor.transform(df_unique)
    
    # Model Setup
    model_s2 = MultiModalModel(tab_size=X_tab.shape[1]).to(DEVICE)
    model_s2.load_state_dict(torch.load('stage2_model.pth', map_location=DEVICE))
    model_s2.eval()
    
    # Dataset that returns processed tabular data
    class Stage2InferenceDataset(Dataset):
        def __init__(self, img_paths, tab_data, image_dir, transform):
            self.img_paths = img_paths
            self.tab_data = tab_data
            self.image_dir = image_dir
            self.transform = transform
        def __len__(self): return len(self.img_paths)
        def __getitem__(self, idx):
            path = os.path.join(self.image_dir, self.img_paths[idx].split('/')[-1])
            img = self.transform(Image.open(path).convert('RGB'))
            return img, torch.tensor(self.tab_data[idx], dtype=torch.float32)

    ds_s2 = Stage2InferenceDataset(df_unique['image_path'].values, X_tab, 'test', transform)
    loader_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for img, tab in tqdm(loader_s2):
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            preds = model_s2(img, tab)
            all_preds.append(preds.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # Shape (N_samples, 5)
    
    # --- FORMAT SUBMISSION ---
    print("4. Formatting Submission...")
    
    # Targets in order of Stage 2 Output
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Map predictions back to sample_id
    pred_map = {}
    for i, sample_id in enumerate(df_unique['sample_id']):
        pred_map[sample_id] = all_preds[i]
        
    # Fill the original test.csv structure
    submission = []
    for idx, row in test_csv.iterrows():
        sid = row['sample_id']
        target_name = row['target_name']
        
        # Get prediction vector for this sample
        pred_vec = pred_map[sid]
        
        # Find which index this target corresponds to
        target_idx = target_cols.index(target_name)
        value = pred_vec[target_idx]
        
        # Clip negative predictions (biomass can't be negative)
        value = max(0.0, value)
        
        submission.append({'sample_id': f"{sid}__{target_name}", 'target': value}) # Kaggle ID format check
        # Note: Check exact sample_id format in sample_submission.csv. 
        # Based on your provided snippet: "ID1001187975__Dry_Clover_g" is the sample_id column in submission
        
    sub_df = pd.DataFrame(submission)
    
    # Double check against sample_submission structure
    sample_sub = pd.read_csv('sample_submission.csv')
    
    # If sample_submission 'sample_id' matches our constructed ID, we are good.
    # If test.csv 'sample_id' column is just the ID (e.g. ID1001187975) and we need to construct the key:
    # The logic above does `sample_id` + `__` + `target_name`.
    
    # Ensure sorting or index matching
    sub_df = sub_df.set_index('sample_id').reindex(sample_sub['sample_id']).reset_index()
    
    sub_df.to_csv('submission.csv', index=False)
    print("✓ submission.csv created successfully.")

if __name__ == '__main__':
    run_inference()