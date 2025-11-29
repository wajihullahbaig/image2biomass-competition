# 3 stage train.py
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
from tqdm import tqdm

# ====================== COMMON IMPORTS ======================
from common import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 
    calculate_sample_weights_mean, calculate_sample_weights,
    get_image_data_transforms, setup_logging, set_seed,
    SeasonalCurriculumSampler, get_season, calculate_count_frequency_features
)

set_seed(42)

# ====================== CONFIG ======================
STAGE1_EPOCHS = 2
STAGE2_EPOCHS = 15
BACKBONE_S1 = 'tf_efficientnet_b3_ns'        
BACKBONE_S2 = 'swin_base_patch4_window7_224' 

# Feature Flags
USE_COUNT_FEATURES = True        # Use Global/Seasonal counts in Stage 2
USE_SAMPLE_WEIGHTS_S1 = True     # Use Hard Balancing for Stage 1

TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Official Weights: Clover, Dead, Green, Total, GDM
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 
COL_WEIGHTS_TENSOR = torch.tensor(OFFICIAL_WEIGHTS, device=DEVICE)

# Logger: Get instance globally, but configure file in __main__ (Multiprocessing safe)
logger = logging.getLogger('Stage1Logger')
logger.setLevel(logging.INFO)

# ====================== DATA PREP ======================
def load_and_pivot_train(csv_path='train.csv'):
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded train.csv: {len(df)} rows")

    wide = df.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    wide['Sampling_Date'] = pd.to_datetime(wide['Sampling_Date'])
    wide['month'] = wide['Sampling_Date'].dt.month
    wide['season'] = wide['month'].apply(get_season)

    # Force base numeric types to avoid object issues later
    wide['Height_Ave_cm'] = pd.to_numeric(wide['Height_Ave_cm'], errors='coerce')
    wide['Pre_GSHH_NDVI'] = pd.to_numeric(wide['Pre_GSHH_NDVI'], errors='coerce')
    
    # Log height for Stage 1 features
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'].fillna(0))

    logger.info(f"Pivoted to wide format: {len(wide)} unique samples")
    return wide

# ====================== STAGE 1: AUXILIARY MULTI-TASK ======================
class Stage1Dataset(Dataset):
    def __init__(self, df, transform=None, species_le=None, fit_le=False, use_weights=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_weights = use_weights

        # 1. Label Encoding
        if fit_le:
            self.species_le = LabelEncoder()
            self.df['species_label'] = self.species_le.fit_transform(self.df['Species'].fillna('Unknown'))
        else:
            self.species_le = species_le
            self.df['species_label'] = self.df['Species'].fillna('Unknown').map(
                lambda x: self.species_le.transform([x])[0] if x in self.species_le.classes_ else -1
            )

        # 2. Sample Weights (Hard Balancing for Classification)
        if self.use_weights:
            self.df, _ = calculate_sample_weights(self.df, group_col='Species',smooth=5.0, logger=logger)
        else:
            self.df['sample_weight'] = 1.0

        self.df['Height_Ave_cm_log'] = np.log1p(self.df['Height_Ave_cm'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            img_path = f"train/{row['image_path'].split('/')[-1]}"
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            img = self.transform(img)

        species_label = int(row['species_label'])
        ndvi = row['Pre_GSHH_NDVI'] if pd.notna(row['Pre_GSHH_NDVI']) else 0.5
        height_log = row['Height_Ave_cm_log'] if pd.notna(row['Height_Ave_cm_log']) else 0.0
        month = int(row['month'] - 1)
        
        weight = torch.tensor(row['sample_weight'], dtype=torch.float32)

        mask = torch.tensor([
            species_label != -1,
            pd.notna(row['Pre_GSHH_NDVI']),
            pd.notna(row['Height_Ave_cm']),
            True 
        ], dtype=torch.bool)

        return (
            img,
            torch.tensor(species_label, dtype=torch.long),
            torch.tensor(ndvi, dtype=torch.float32),
            torch.tensor(height_log, dtype=torch.float32),
            torch.tensor(month, dtype=torch.long),
            weight,
            mask
        )

    def get_species_encoder(self):
        return self.species_le

class Stage1Model(nn.Module):
    def __init__(self, num_species, num_months=12):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_S1, pretrained=True, num_classes=0)
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


def train_stage1(df_wide):
    logger.info(f"=== STAGE 1: Training (Weighted={USE_SAMPLE_WEIGHTS_S1}) ===")
    
    # 1. Stratified Split
    train_df, val_df = train_test_split(df_wide, test_size=0.2, stratify=df_wide['season'], random_state=42)
    
    # 2. Datasets
    train_dataset = Stage1Dataset(train_df, transform=get_image_data_transforms()[0], 
                                  fit_le=True, use_weights=USE_SAMPLE_WEIGHTS_S1)
    val_dataset = Stage1Dataset(val_df, transform=get_image_data_transforms()[1], 
                                species_le=train_dataset.get_species_encoder(), fit_le=False, use_weights=False)
    
    species_le = train_dataset.get_species_encoder()
    
    # 3. Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              sampler=SeasonalCurriculumSampler(train_dataset.df, shuffle_within_season=False),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model & Optimizer
    model = Stage1Model(num_species=len(species_le.classes_)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    
    ce_loss_none = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    mse_loss_none = nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    save_path = 'stage1_package.pth' # Standardized filename

    for epoch in range(STAGE1_EPOCHS):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            img, sp, ndvi, hlog, month, weight, mask = [x.to(DEVICE) for x in batch]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                
                loss_vec = torch.zeros(img.size(0), device=DEVICE)
                if mask[:,0].any(): loss_vec += 0.4 * ce_loss_none(sp_pred, sp) * mask[:,0].float()
                if mask[:,1].any(): loss_vec += 0.3 * mse_loss_none(ndvi_pred, ndvi) * mask[:,1].float()
                if mask[:,2].any(): loss_vec += 0.2 * mse_loss_none(h_pred, hlog) * mask[:,2].float()
                loss_vec += 0.1 * ce_loss_none(month_pred, month)

                loss = (loss_vec * weight).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()            

        # === VAL ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img, sp, ndvi, hlog, month, _, mask = [x.to(DEVICE) for x in batch]
                with torch.amp.autocast('cuda'):
                    sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                    l = 0.0
                    if mask[:,0].any(): l += 0.4 * F.cross_entropy(sp_pred[mask[:,0]], sp[mask[:,0]], ignore_index=-1)
                    if mask[:,1].any(): l += 0.3 * F.mse_loss(ndvi_pred[mask[:,1]], ndvi[mask[:,1]])
                    if mask[:,2].any(): l += 0.2 * F.mse_loss(h_pred[mask[:,2]], hlog[mask[:,2]])
                    l += 0.1 * F.cross_entropy(month_pred, month)
                    val_loss += l.item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # === SAVE ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best val loss: {best_val_loss:.4f}")
            # Save the full package needed for inference
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'species_encoder': species_le,
                'backbone_name': BACKBONE_S1
            }
            torch.save(checkpoint, save_path)
            logger.info(f"New best model saved to {save_path}")

    # === LOAD BEST ===
    if os.path.exists(save_path):
        # We must load with weights_only=False because we are loading the LabelEncoder (pickle)
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {save_path} (Val Loss: {best_val_loss:.4f})")
    else:
        logger.warning(f"File {save_path} not found! (Did training fail or produce NaNs?). Returning last epoch model.")
    
    return model, species_le

# ====================== PSEUDO-LABEL GENERATION ======================
def generate_pseudo_labels(model, df_wide, species_le):
    logger.info("=== Generating Pseudo-Labels ===")
    model.eval()
    
    dataset = Stage1Dataset(df_wide.copy(), transform=get_image_data_transforms()[1], species_le=species_le, fit_le=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    preds = {'species': [], 'ndvi': [], 'height_log': [], 'month': []}
    
    with torch.no_grad():
        for img, _, _, _, _, _, _ in tqdm(loader, desc="Pseudo-labeling"):
            img = img.to(DEVICE)
            sp, nd, h, m = model(img)
            
            preds['species'].extend(species_le.inverse_transform(torch.argmax(sp, 1).cpu().numpy()))
            preds['ndvi'].extend(nd.cpu().numpy())
            preds['height_log'].extend(h.cpu().numpy())
            preds['month'].extend((torch.argmax(m, 1).cpu().numpy() + 1))

    df = df_wide.copy()
    df['pred_species'] = preds['species']
    df['pred_ndvi'] = preds['ndvi']
    df['pred_height_log'] = preds['height_log']
    df['pred_season'] = [get_season(m) for m in preds['month']]

    df['season_final'] = df['season'].fillna(df['pred_season'])
    df['Species_final'] = df['Species'].fillna(df['pred_species'])
    df['NDVI_final'] = df['Pre_GSHH_NDVI'].fillna(df['pred_ndvi'])
    df['Height_final_log'] = np.log1p(df['Height_Ave_cm']).fillna(df['pred_height_log'])

    return df

# ====================== STAGE 2: PHYSICS-INFORMED BIOMASS ======================
class Stage2Dataset(Dataset):
    def __init__(self, df, extra_features=None, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        
        # 1. Base Engineering
        self.df['NDVI_final'] = pd.to_numeric(self.df['NDVI_final'], errors='coerce').fillna(0.0)
        self.df['Height_final_log'] = pd.to_numeric(self.df['Height_final_log'], errors='coerce').fillna(0.0)

        self.df['ndvi_h_mul'] = self.df['NDVI_final'] * self.df['Height_final_log']
        self.df['ndvi_h_ratio'] = self.df['NDVI_final'] / (self.df['Height_final_log'] + 1e-6)
        
        self.tabular_cols = ['NDVI_final', 'Height_final_log', 'ndvi_h_mul', 'ndvi_h_ratio']
        
        # 2. Add Count Features
        if extra_features:
            self.tabular_cols.extend(extra_features)
            
        # 3. CRITICAL: Force all tab cols to float32 NOW to prevent TypeError in __getitem__
        for col in self.tabular_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0).astype(np.float32)
        
        # 4. Sample Weights (Smooth Balancing for Regression on SPECIES)
        self.df, _ = calculate_sample_weights_mean(self.df, group_col='Species_final')
        # Clip to prevent exploding gradients
        self.df['sample_weight'] = self.df['sample_weight'].clip(0.1, 10.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            img_path = f"train/{row['image_path'].split('/')[-1]}"
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            img = self.transform(img)

        # Fast and type-safe now
        # Tabular
        tab = torch.from_numpy(row[self.tabular_cols].values.astype(np.float32))        
        targets = torch.tensor(row[TARGET_COLS].values.astype(np.float32), dtype=torch.float32)
        weight = torch.tensor(row['sample_weight'], dtype=torch.float32)

        return img, tab, targets, weight

class Stage2Model(nn.Module):
    def __init__(self, tab_size):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_S2, pretrained=True, num_classes=0)
        img_feat = self.backbone.num_features

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
        self.head = nn.Linear(256, 3) # Predict Clover, Dead, Green

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

# ====================== IMPUTATION HELPERS ======================
def get_imputation_stats(df, target_cols):
    """
    Calculates median statistics from the TRAIN set to save for Inference.
    """
    group_cols = ['Species', 'State', 'season']
    
    # Calculate stats
    median_map = df.groupby(group_cols)[target_cols].median()
    global_medians = df[target_cols].median()
    
    return median_map, global_medians

def apply_imputation(df, median_map, global_medians, target_cols):
    """
    Applies saved stats to a dataframe (Train, Val, or Test).
    """
    df = df.copy()
    group_cols = ['Species', 'State', 'season']
    
    # Set index to group_cols for fast mapping
    # Note: We reset index at the end to return original structure
    df_idx = df.set_index(group_cols)
    
    for col in target_cols:
        # Map specific medians based on the index (Species, State, season)
        mapped_vals = median_map[col].reindex(df_idx.index)
        
        # Create Series with original dataframe index to satisfy pandas type check
        fill_values = pd.Series(mapped_vals.values, index=df.index)
        
        # Fill logic: 1. Try Group Median, 2. Fallback to Global Median
        df[col] = df[col].fillna(fill_values)
        df[col] = df[col].fillna(global_medians[col])
        
    return df

# ====================== STAGE 2 TRAINING LOOP ======================
def train_stage2(df_enhanced):
    logger.info(f"=== STAGE 2: Physics-Informed Training (CountFeats={USE_COUNT_FEATURES}) ===")
    
    # 1. Stratified Split
    train_df_raw, val_df_raw = train_test_split(
        df_enhanced, test_size=0.2, stratify=df_enhanced['season_final'], random_state=42
    )

    # 2. Feature Engineering (Count features on Train only, map to Val)
    extra_feats = []
    if USE_COUNT_FEATURES:
        logger.info("Generating Count/Frequency features...")
        # Note: We use 'Species_final' and 'season_final' because they have no NaNs
        train_df_raw, val_df_raw, extra_feats = calculate_count_frequency_features(
            train_df=train_df_raw,
            val_df=val_df_raw,
            group_col='Species_final',
            local_group_col='season_final',
            logger=logger
        )

    # 3. LEAKAGE-FREE IMPUTATION (Refactored for Inference Saving)
    logger.info("Calculating imputation stats on Train set...")
    # A. Calculate Stats on TRAIN only
    median_map, global_medians = get_imputation_stats(train_df_raw, TARGET_COLS)
    
    # B. Apply to Train and Val
    train_df = apply_imputation(train_df_raw, median_map, global_medians, TARGET_COLS)
    val_df = apply_imputation(val_df_raw, median_map, global_medians, TARGET_COLS)

    # 4. Data Loaders
    train_ds = Stage2Dataset(train_df, extra_features=extra_feats, transform=get_image_data_transforms()[0])
    val_ds = Stage2Dataset(val_df, extra_features=extra_feats, transform=get_image_data_transforms()[1])
    
    # Capture the exact column order for inference!
    final_tabular_cols = train_ds.tabular_cols
    logger.info(f"Final Tabular Columns ({len(final_tabular_cols)}): {final_tabular_cols}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, 
        sampler=SeasonalCurriculumSampler(train_ds.df), 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model Setup
    model = Stage2Model(tab_size=len(final_tabular_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") 

    best_score = -float('inf')

    # 6. Training Loop
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for img, tab, targets, sample_weight in train_loader:
            img = img.to(DEVICE)
            tab = tab.to(DEVICE)
            targets = targets.to(DEVICE)
            sample_weight = sample_weight.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use new AMP syntax
            with torch.amp.autocast('cuda'):
                pred = model(img, tab)
                
                # --- DOUBLE WEIGHTING STRATEGY ---
                # 1. Squared Error [Batch, 5]
                squared_err = (pred - targets) ** 2
                
                # 2. Column Weighting (Official Metric) [Batch, 5]
                col_weighted = squared_err * COL_WEIGHTS_TENSOR
                
                # 3. Sum Elements [Batch]
                loss_per_img = col_weighted.sum(dim=1)
                
                # 4. Sample Weighting (Rare Species)
                loss = (loss_per_img * sample_weight).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds, all_trues = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for img, tab, targets, sample_weight in val_loader:
                img = img.to(DEVICE)
                tab = tab.to(DEVICE)
                targets = targets.to(DEVICE)
                sample_weight = sample_weight.to(DEVICE)

                with torch.amp.autocast('cuda'):
                    pred = model(img, tab)
                    # Track validation loss using the same weighted metric
                    sq_err = (pred - targets) ** 2
                    val_loss += ((sq_err * COL_WEIGHTS_TENSOR).sum(1) * sample_weight).mean().item()

                all_preds.append(pred.float().cpu().numpy())
                all_trues.append(targets.float().cpu().numpy())

        val_loss /= len(val_loader)
        pred_arr = np.concatenate(all_preds, axis=0)
        true_arr = np.concatenate(all_trues, axis=0)
        
        # Calculate R2 using Official Weights for final score
        weights_flat = np.tile(OFFICIAL_WEIGHTS, (len(true_arr), 1)).flatten()
        score = r2_score(true_arr.flatten(), pred_arr.flatten(), sample_weight=weights_flat)

        logger.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Weighted R²: {score:.5f}")

        # Save Best Model Package
        if score > best_score:
            best_score = score
            logger.info(f"New best score: {best_score:.4f}")
            # === SAVING FULL INFERENCE PACKAGE ===
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tabular_cols': final_tabular_cols,     # Save exact column order
                'imputation_stats': {                   # Save math for Test set
                    'median_map': median_map,
                    'global_medians': global_medians
                },
                'backbone_name': BACKBONE_S2,
                'score': best_score
            }
            torch.save(checkpoint, 'stage2_package.pth')
            # =====================================
            
            logger.info(f"NEW BEST R²: {best_score:.5f} (Saved to stage2_package.pth)")
            
        scheduler.step()

# ====================== MAIN EXECUTION ======================
if __name__ == '__main__':
    # Initialize Logging File HERE (Main Process Only)
    setup_logging(log_dir='logs', file_name_part='Final_Pipeline')
    
    logger.info("Starting Full Training Pipeline")
    df_wide = load_and_pivot_train()

    s1_model, species_le = train_stage1(df_wide)
    df_enhanced = generate_pseudo_labels(s1_model, df_wide, species_le)
    train_stage2(df_enhanced)