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
import json

# ====================== MODEL CONFIGURATIONS ======================
FUSION_DIM = 256

class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name='timm/tf_efficientnet_b3.ns_jft_in1k', num_targets=5, num_aux=2, num_species=11, num_months=12, pretrained=True):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)  # Use default size for dummy
            self.backbone_dim = self.backbone(dummy_input).shape[1]
            
        # 2. Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_aux)
        )
        
        # Multi-task heads 
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_species)
        )
        
        # 4. Month Head (Cyclical Regression)
        # Forces backbone to learn seasonal cycles (sin/cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Sin, Cos
        )
        
        # 5. Biomass Head
        fusion_dim = FUSION_DIM
        
        self.biomass_head = nn.Sequential(
            nn.Linear(self.backbone_dim + num_aux + num_species + 2, fusion_dim), # +2 for Month Sin/Cos
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # OUTPUT: [Clover, Dead, Green] ONLY
            nn.Softplus() # Ensures positive outputs
        )

    def freeze_backbone(self, freeze_fraction=0.80):
        """Freezes a fraction of the backbone layers."""
        params = list(self.backbone.parameters())
        num_to_freeze = int(len(params) * freeze_fraction)
        
        for i, param in enumerate(params):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        if freeze_fraction > 0.9:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()

    def forward(self, x):
        # Extract features from image
        img_feats = self.backbone(x) # (B, backbone_dim)
        
        # Predict species (categorical logits)
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        # Predict month (continous logits)
        month_logits = self.month_head(img_feats)
        
        # Predict auxiliary features (NDVI, Height)
        aux_out = self.aux_head(img_feats) 
        #aux_out = torch.clamp(aux_out, 0.0, 10.0)
            
        # --- FUSION OF ALL FEATURES ---
        combined_feats = torch.cat([
            img_feats, 
            aux_out, 
            species_probs,
            month_logits
        ], dim=1)
        
        # --- PHYSICS-INFORMED HEAD ---
        # 1. Predict ONLY Components (Clover, Dead, Green)
        # We use Softplus ensuring non-negative raw mass (0 to inf)
        components_pred = self.biomass_head(combined_feats) # (B, 3)
        #components_pred = torch.clamp(components_pred, 0.0, 256.0)
        
        c = components_pred[:, 0:1] # Clover
        d = components_pred[:, 1:2] # Dead
        g = components_pred[:, 2:3] # Green
        
        # 2. Physics Constraints (Performed in Computational Graph)
        # Total = Clover + Dead + Green
        # GDM   = Clover + Green
        total = c + d + g
        gdm   = c + g
        
        # 3. Concatenate for Loss Calculation (Order: C, D, G, Total, GDM)
        # This allows gradients from 'Total' loss to flow back to C, D, G
        biomass_out = torch.cat([c, d, g, total, gdm], dim=1)
        
        return biomass_out, aux_out, species_logits, month_logits

# Weight Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# ====================== CONFIGURATION ======================
TEST_CSV_PATH = './test.csv'
TEST_IMG_DIR = './test'
MODEL_DIR = './logs/F2_071_EN_B3_Unified_Trainer_20251225_002636'  # Change this to the desired model folder
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Single fold model to load (USER REQUEST: Load Fold 0 as default)
FOLD_TO_LOAD = 0

print(f"Device: {DEVICE}")

# ====================== HELPER FUNCTIONS ======================
def get_season(month):
    if month in [12, 1, 2]: return 'Summer'
    if month in [3, 4, 5]: return 'Autumn'
    if month in [6, 7, 8]: return 'Winter'
    return 'Spring'

def get_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def convert_test_to_wide(test_df):
    """
    Convert test.csv from long format to wide format.
    """
    test_df['clean_id'] = test_df['sample_id'].str.split('__').str[0]
    wide_test = test_df[['clean_id', 'image_path']].drop_duplicates().reset_index(drop=True)
    print(f"   Converted from {len(test_df)} rows (long) to {len(wide_test)} unique images (wide)")
    return wide_test

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
            img = Image.new('RGB', (224, 224))  # Default size if error
            print(f"Warning: Image not found: {img_path}")
        
        if self.transform:
            img = self.transform(img)
        
        return img, row['clean_id']

# ====================== MAIN INFERENCE LOGIC ======================
def run_inference():
    print("="*70 + "\nBIOMASS UNIFIED MODEL INFERENCE\n" + "="*70)
    
    # 1. LOAD AND CONVERT TEST DATA
    print("\n[1/5] Loading test data...")
    test_df_long = pd.read_csv(TEST_CSV_PATH)
    test_df_wide = convert_test_to_wide(test_df_long)
    
    # 2. LOAD METADATA
    print("\n[2/5] Loading model metadata...")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    num_species = len(metadata['species_list'])
    backbone_name = metadata['backbone']
    image_size = metadata['image_size']
    
    print(f"   Backbone: {backbone_name}")
    print(f"   Image Size: {image_size}")
    print(f"   Num Species: {num_species}")
    
    # 3. LOAD MODEL
    print(f"\n[3/5] Loading fold {FOLD_TO_LOAD} model...")
    fold_path = os.path.join(MODEL_DIR, f'best_model_fold{FOLD_TO_LOAD}.pth')
    
    model = BiomassUnifiedModel(
        backbone_name=backbone_name, 
        num_species=num_species, 
        pretrained=False
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(fold_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"   ✓ Loaded Fold {FOLD_TO_LOAD} successfully")
    
    # 4. RUN INFERENCE
    print(f"\n[4/5] Running inference...")
    transform = get_test_transform(image_size)
    dataset = TestDataset(test_df_wide, TEST_IMG_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    all_preds = []
    all_clean_ids = []
    
    with torch.no_grad():
        for img, clean_ids in tqdm(loader, desc="Inference"):
            img = img.to(DEVICE)
            biomass_pred, _, _, _ = model(img)
            all_preds.append(biomass_pred.cpu().numpy())
            all_clean_ids.extend(clean_ids)
    
    # 5. CREATE SUBMISSION
    print("\n[5/5] Creating submission file...")
    target_cols = metadata['target_cols']  # ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    preds_array = np.concatenate(all_preds, axis=0)
    preds_array = np.maximum(preds_array, 0)  # Safety clip
    
    preds_wide = pd.DataFrame(preds_array, columns=target_cols)
    preds_wide['clean_id'] = all_clean_ids
    
    submission_rows = []
    for _, row in preds_wide.iterrows():
        clean_id = row['clean_id']
        for target_col in target_cols:
            sample_id = f"{clean_id}__{target_col}"
            submission_rows.append({
                'sample_id': sample_id,
                'target': row[target_col]
            })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('submission.csv', index=False)
    
    print("="*70 + "\n✅ SUBMISSION CREATED: submission.csv\n" + "="*70)
    print(submission_df.head())

if __name__ == '__main__':
    run_inference()