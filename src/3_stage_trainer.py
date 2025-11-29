# train.py
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
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====================== COMMON IMPORTS ======================
from common import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, USE_COUNT_FEATURES, calculate_sample_weights_mean, 
    get_image_data_transforms, print_stratification_stats, setup_logging, set_seed,
    SeasonalCurriculumSampler, get_season, calculate_sample_weights
)

set_seed(42)

# ====================== CONFIG ======================
STAGE1_EPOCHS = 2
STAGE2_EPOCHS = 2
BACKBONE_S1 = 'tf_efficientnet_b3_ns'        
BACKBONE_S2 = 'swin_base_patch4_window7_224' 
TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
OFFICIAL_WEIGHTS = [0.1, 0.1, 0.1, 0.5, 0.2] 

# Initialize Logger (Global Scope - Get instance only)
logger = logging.getLogger('3StageTrainer')
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

    # Log height for Stage 1 features
    wide['Height_Ave_cm_log'] = np.log1p(wide['Height_Ave_cm'])

    logger.info(f"Pivoted to wide format: {len(wide)} unique samples")
    return wide

# ====================== STAGE 1: AUXILIARY MULTI-TASK ======================
class Stage1Dataset(Dataset):
    def __init__(self, df, transform=None, species_le=None, fit_le=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if fit_le:
            self.species_le = LabelEncoder()
            self.df['species_label'] = self.species_le.fit_transform(self.df['Species'].fillna('Unknown'))
        else:
            self.species_le = species_le
            self.df['species_label'] = self.df['Species'].fillna('Unknown').map(
                lambda x: self.species_le.transform([x])[0] if x in self.species_le.classes_ else -1
            )

        self.df['Height_Ave_cm_log'] = np.log1p(self.df['Height_Ave_cm'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            img_path = f"train/{row['image_path'].split('/')[-1]}"
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)) # Fallback

        if self.transform:
            img = self.transform(img)

        species_label = int(row['species_label'])
        ndvi = row['Pre_GSHH_NDVI'] if pd.notna(row['Pre_GSHH_NDVI']) else 0.5
        height_log = row['Height_Ave_cm_log'] if pd.notna(row['Height_Ave_cm_log']) else 0.0
        month = int(row['month'] - 1)

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
    logger.info("=== STAGE 1: Training Auxiliary Multi-Task Model ===")
    
    # 1. Stratified Split
    train_df, val_df = train_test_split(df_wide, test_size=0.2, stratify=df_wide['season'], random_state=42)
    print_stratification_stats(df_wide, train_df, val_df, start_col='season', logger=logger)
    # 2. Datasets
    # Train fits the encoder, Val reuses it
    train_dataset = Stage1Dataset(train_df, transform=get_image_data_transforms()[0], fit_le=True)
    val_dataset = Stage1Dataset(val_df, transform=get_image_data_transforms()[1], 
                                species_le=train_dataset.get_species_encoder(), fit_le=False)
    
    species_le = train_dataset.get_species_encoder()
    
    # 3. Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              sampler=SeasonalCurriculumSampler(train_dataset.df, shuffle_within_season=False),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model & Optimization
    model = Stage1Model(num_species=len(species_le.classes_)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler() # AMP
    
    # Losses
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(STAGE1_EPOCHS):
        # ================= TRAIN =================
        model.train()
        train_meters = {'loss': 0.0, 'sp_acc': 0.0, 'month_acc': 0.0}
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{STAGE1_EPOCHS}")
        for batch in pbar:
            img, sp, ndvi, hlog, month, mask = [x.to(DEVICE) for x in batch]
            
            optimizer.zero_grad()
            
            with autocast():
                sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                
                loss = 0.0
                # Weighted Multi-Task Loss
                if mask[:,0].any(): loss += 0.4 * ce_loss(sp_pred[mask[:,0]], sp[mask[:,0]]) # Species
                if mask[:,1].any(): loss += 0.3 * mse_loss(ndvi_pred[mask[:,1]], ndvi[mask[:,1]]) # NDVI
                if mask[:,2].any(): loss += 0.2 * mse_loss(h_pred[mask[:,2]], hlog[mask[:,2]]) # Height
                loss += 0.1 * ce_loss(month_pred, month) # Month (Season)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            train_meters['loss'] += loss.item()
            
            # Simple accuracy tracking for progress bar
            valid_sp = mask[:, 0]
            if valid_sp.any():
                acc = (sp_pred[valid_sp].argmax(1) == sp[valid_sp]).float().mean()
                train_meters['sp_acc'] += acc.item()
            
            acc_m = (month_pred.argmax(1) == month).float().mean()
            train_meters['month_acc'] += acc_m.item()
            steps += 1
            
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_meters['loss'] / steps

        # ================= VAL =================
        model.eval()
        val_loss = 0.0
        metrics = {'sp_acc': [], 'month_acc': [], 'ndvi_mse': [], 'h_mse': []}
        
        with torch.no_grad():
            for batch in val_loader:
                img, sp, ndvi, hlog, month, mask = [x.to(DEVICE) for x in batch]
                
                with autocast():
                    sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                    
                    # Calc Loss
                    batch_loss = 0.0
                    if mask[:,0].any(): batch_loss += 0.4 * ce_loss(sp_pred[mask[:,0]], sp[mask[:,0]])
                    if mask[:,1].any(): batch_loss += 0.3 * mse_loss(ndvi_pred[mask[:,1]], ndvi[mask[:,1]])
                    if mask[:,2].any(): batch_loss += 0.2 * mse_loss(h_pred[mask[:,2]], hlog[mask[:,2]])
                    batch_loss += 0.1 * ce_loss(month_pred, month)
                    val_loss += batch_loss.item()
                
                # Metrics
                if mask[:, 0].any():
                    metrics['sp_acc'].append((sp_pred[mask[:,0]].argmax(1) == sp[mask[:,0]]).float().mean().item())
                
                metrics['month_acc'].append((month_pred.argmax(1) == month).float().mean().item())
                
                if mask[:, 1].any():
                    metrics['ndvi_mse'].append(F.mse_loss(ndvi_pred[mask[:,1]], ndvi[mask[:,1]]).item())
                    
                if mask[:, 2].any():
                    metrics['h_mse'].append(F.mse_loss(h_pred[mask[:,2]], hlog[mask[:,2]]).item())

        avg_val_loss = val_loss / len(val_loader)
        
        # Summarize Validation Metrics
        sp_acc = np.mean(metrics['sp_acc']) * 100 if metrics['sp_acc'] else 0
        month_acc = np.mean(metrics['month_acc']) * 100
        ndvi_rmse = np.sqrt(np.mean(metrics['ndvi_mse'])) if metrics['ndvi_mse'] else 0
        h_rmse = np.sqrt(np.mean(metrics['h_mse'])) if metrics['h_mse'] else 0

        logger.info(
            f"Epoch {epoch+1} Summary:\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  [Metrics] Species Acc: {sp_acc:.1f}% | Season Acc: {month_acc:.1f}% | "
            f"NDVI RMSE: {ndvi_rmse:.4f} | Height RMSE: {h_rmse:.4f}"
        )

        # ================= SAVE BEST =================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'stage1_best.pth')
            logger.info(f"  >>> NEW BEST STAGE 1 MODEL (Loss: {best_val_loss:.4f})")
            
        scheduler.step(avg_val_loss)

    logger.info("Stage 1 Training Complete.")
    
    # Load best model before returning
    model.load_state_dict(torch.load('stage1_best.pth'))
    return model, species_le

# ====================== PSEUDO-LABEL GENERATION ======================
def generate_pseudo_labels(model, df_wide, species_le):
    logger.info("=== Generating Pseudo-Labels ===")
    model.eval()
    
    dataset = Stage1Dataset(df_wide.copy(), transform=get_image_data_transforms()[1], species_le=species_le, fit_le=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    preds = {'species': [], 'ndvi': [], 'height_log': [], 'month': []}
    
    with torch.no_grad():
        for img, _, _, _, _, _ in tqdm(loader, desc="Pseudo-labeling"):
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

    # Fill Missing Values with Predictions
    df['season_final'] = df['season'].fillna(df['pred_season'])
    df['Species_final'] = df['Species'].fillna(df['pred_species'])
    df['NDVI_final'] = df['Pre_GSHH_NDVI'].fillna(df['pred_ndvi'])
    df['Height_final_log'] = np.log1p(df['Height_Ave_cm']).fillna(df['pred_height_log'])

    return df

# ====================== STAGE 2: PHYSICS-INFORMED BIOMASS ======================

def safe_impute(train_df, val_df, target_cols):
    """
    LEAKAGE-FREE IMPUTATION
    Calculates medians ONLY on train_df, applies to both.
    Uses pd.Series wrapper to fix TypeError.
    """
    logger.info("Performing leakage-free imputation...")
    group_cols = ['Species', 'State', 'season']
    
    # 1. Calculate Imputation Map (Train Only)
    median_map = train_df.groupby(group_cols)[target_cols].median()
    global_medians = train_df[target_cols].median()
    
    def apply_fill(df):
        df = df.copy()
        # Temp index for mapping
        df_idx = df.set_index(group_cols)
        
        for col in target_cols:
            # Get median values aligned to df rows
            mapped_vals_series = median_map[col].reindex(df_idx.index)
            
            # Create a Series with the ORIGINAL df index to satisfy fillna()
            fill_values = pd.Series(mapped_vals_series.values, index=df.index)
            
            # Fill
            df[col] = df[col].fillna(fill_values)
            df[col] = df[col].fillna(global_medians[col])
            
        return df

    return apply_fill(train_df), apply_fill(val_df)

class Stage2Dataset(Dataset):
    def __init__(self, df, extra_features=None, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        
        # Base Engineering
        self.df['ndvi_h_mul'] = self.df['NDVI_final'] * self.df['Height_final_log']
        self.df['ndvi_h_ratio'] = self.df['NDVI_final'] / (self.df['Height_final_log'] + 1e-6)
        
        # Base Features
        self.tabular_cols = ['NDVI_final', 'Height_final_log', 'ndvi_h_mul', 'ndvi_h_ratio']
        
        # === DYNAMICALLY ADD EXTRA FEATURES ===
        if extra_features:
            self.tabular_cols.extend(extra_features)
            
        # Ensure they are float32 for the network
        for col in self.tabular_cols:
            self.df[col] = self.df[col].astype(np.float32)
        
        # Sample Weights
        self.df, _ = calculate_sample_weights(self.df, group_col='season_final', smooth=10.0)
        for col in self.tabular_cols:
            self.df[col] = self.df[col].astype(np.float32)
            
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
            nn.SiLU(inplace=True), # Swish > ReLU for regression
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # PREDICT ONLY 3 COMPONENTS: Clover, Dead, Green
        self.head = nn.Linear(256, 3)

    def forward(self, img, tab):
        f = self.backbone(img)
        if len(f.shape) > 2: f = f.mean([2, 3])

        x = torch.cat([f, tab], dim=1)
        feat = self.mlp(x)
        
        # Softplus ensures non-negative mass
        components = F.softplus(self.head(feat))
        
        clover = components[:, 0:1]
        dead   = components[:, 1:2]
        green  = components[:, 2:3]
        
        # Physics Sum
        total = clover + dead + green
        gdm   = clover + green
        
        return torch.cat([clover, dead, green, total, gdm], dim=1)

# Make sure to import calculate_count_frequency_features from common
from common import calculate_count_frequency_features 

def train_stage2(df_enhanced):
    logger.info(f"=== STAGE 2: Physics-Informed Biomass Training (Count Feats={USE_COUNT_FEATURES}) ===")
    
    # 1. STRATIFIED SPLIT
    train_df_raw, val_df_raw = train_test_split(
        df_enhanced, test_size=0.2, stratify=df_enhanced['season_final'], random_state=42
    )

    # 2. FEATURE ENGINEERING (COUNT FEATURES)
    # We do this AFTER split to prevent leakage (stats calculated on train, mapped to val)
    extra_feats = []
    if USE_COUNT_FEATURES:
        logger.info("Generating Count/Frequency features...")
        # Note: We use 'Species_final' because it has no NaNs (imputed by Stage 1)
        train_df_raw, val_df_raw, extra_feats = calculate_count_frequency_features(
            train_df=train_df_raw,
            val_df=val_df_raw,
            group_col='Species_final',      # Use the filled column
            local_group_col='season_final', # Use the filled column
            logger=logger
        )

    # 3. APPLY LEAKAGE-FREE IMPUTATION (TARGETS)
    train_df, val_df = safe_impute(train_df_raw, val_df_raw, TARGET_COLS)

    # 4. DATASETS & LOADERS
    # Pass the list of new features to the dataset
    train_ds = Stage2Dataset(train_df, extra_features=extra_feats, transform=get_image_data_transforms()[0])
    val_ds = Stage2Dataset(val_df, extra_features=extra_feats, transform=get_image_data_transforms()[1])
    
    logger.info(f"Tabular Input Size: {len(train_ds.tabular_cols)} features")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, 
        sampler=SeasonalCurriculumSampler(train_ds.df),
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. MODEL & OPTIMIZER
    # Model auto-adjusts input size based on tab_size
    model = Stage2Model(tab_size=len(train_ds.tabular_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)
    scaler = GradScaler() 

    best_score = -float('inf')

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"S2 Epoch {epoch+1}")
        for img, tab, targets, sample_weight in pbar:
            img, tab, targets = img.to(DEVICE), tab.to(DEVICE), targets.to(DEVICE)
            sample_weight = sample_weight.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                pred = model(img, tab)
                loss_sample = F.mse_loss(pred, targets, reduction='none').mean(dim=1)
                loss = (loss_sample * sample_weight).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # VALIDATION
        model.eval()
        all_preds, all_trues = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for img, tab, targets, sample_weight in val_loader:
                img, tab, targets = img.to(DEVICE), tab.to(DEVICE), targets.to(DEVICE)
                sample_weight = sample_weight.to(DEVICE)

                with autocast():
                    pred = model(img, tab)
                    loss_sample = F.mse_loss(pred, targets, reduction='none').mean(dim=1)
                    val_loss += (loss_sample * sample_weight).mean().item()

                all_preds.append(pred.float().cpu().numpy())
                all_trues.append(targets.float().cpu().numpy())

        val_loss /= len(val_loader)
        pred_arr = np.concatenate(all_preds, axis=0)
        true_arr = np.concatenate(all_trues, axis=0)
        
        weights_flat = np.tile(OFFICIAL_WEIGHTS, (len(true_arr), 1)).flatten()
        score = r2_score(true_arr.flatten(), pred_arr.flatten(), sample_weight=weights_flat)

        logger.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Weighted R²: {score:.5f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'stage2_best.pth')
            logger.info(f"NEW BEST R²: {best_score:.5f}")
            
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