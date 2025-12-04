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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
from tqdm import tqdm
from sklearn.model_selection import KFold

# ====================== COMMON IMPORTS ======================
from configs import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, 
    BACKBONE_S1, BACKBONE_S2, LEARNING_RATE, 
    N_FOLDS, STAGE1_STRATIFICATION_COLUMN, STAGE2_STRATIFICATION_COLUMN, TEST_SPLIT_RATIO,
    USE_SAMPLE_WEIGHTS_S1, 
    STAGE1_EPOCHS, 
    STAGE2_EPOCHS, 
    TARGET_COLS, 
    USE_COUNT_FEATURES,
    OFFICIAL_WEIGHTS,
    COL_WEIGHTS_TENSOR, USE_SAMPLE_WEIGHTS_S2
)
from common import (
    calculate_global_weighted_r2, calculate_sample_weights_01_normalized, calculate_sample_weights_smooth,
    get_image_data_transforms, load_data, print_stratification_stats, setup_logging, set_seed,
    get_season, calculate_count_frequency_features
)

set_seed(42)



# Logger: Get instance globally, but configure file in __main__ (Multiprocessing safe)
logger = logging.getLogger('System Logger')
logger.setLevel(logging.INFO)

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

        # 2. Sample Weights 
        if self.use_weights:
            # Note: We calculate weights based on Species balance
            self.df, _ = calculate_sample_weights_01_normalized(self.df, group_col='pred_species',logger=logger)
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
            # Fallback for broken paths
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            img = self.transform(img)

        species_label = int(row['species_label'])
        ndvi = row['Pre_GSHH_NDVI'] 
        height_log = row['Height_Ave_cm_log'] 
        month = int(row['month'] - 1) # 0-11 for class index
        
        weight = torch.tensor(row['sample_weight'], dtype=torch.float32)

        # Mask to ignore missing values in loss calculation
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

# ====================== STAGE 1 MODEL ======================
class Stage1Model(nn.Module):
    def __init__(self, num_species, num_months=12):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_S1, pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        
        # Multi-Heads
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


# ====================== TRAINING FUNCTION ======================
def train_stage1_kfold(df_wide):
    logger.info(f"=== STAGE 1: K-Fold Training  ===")
    logger.info(f"Total Samples: {len(df_wide)}")
    logger.info(f"Using Sample Weights: {USE_SAMPLE_WEIGHTS_S1}")
    logger.info(f"Stage 1 Epochs: {STAGE1_EPOCHS}")
    logger.info(f"Backbone: {BACKBONE_S1}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Stratification Column: {STAGE1_STRATIFICATION_COLUMN}")
    logger.info(f"Device: {DEVICE}")

    # 1. Global Label Encoder (Must fit on ALL data to ensure consistency across folds)
    species_le = LabelEncoder()
    species_le.fit(df_wide['Species'].fillna('Unknown'))
    num_species = len(species_le.classes_)
    
    
    

    
    # Initialize OOF columns
    oof_df = df_wide.copy()
    oof_df['pred_species_idx'] = -1
    oof_df['pred_ndvi'] = np.nan
    oof_df['pred_height_log'] = np.nan
    oof_df['pred_month'] = -1
    oof_df['pred_mon_sin'] = np.nan
    oof_df['pred_mon_cos'] = np.nan
    
    model_save_dir = 'models_stage1'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save Metadata for Inference
    metadata = {
        'species_encoder': species_le,
        'backbone_name': BACKBONE_S1,
        'num_species': num_species
    }
    torch.save(metadata, os.path.join(model_save_dir, 'stage1_metadata.pth'))


    stratify_col = None
    if STAGE1_STRATIFICATION_COLUMN not in df_wide.columns:
        logger.warning(f"Stratification column '{STAGE1_STRATIFICATION_COLUMN}' not found in dataframe!")
        logger.info("Proceeding without stratification...")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splitter = kf.split(df_wide)
    else:
        stratify_col = df_wide[STAGE1_STRATIFICATION_COLUMN]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)    
        splitter = skf.split(df_wide, stratify_col)
            

    for fold, (train_idx, val_idx) in enumerate(splitter):
        logger.info(f"\n--- Starting Fold {fold+1}/{N_FOLDS} ---")
        
        train_df = df_wide.iloc[train_idx]
        val_df = df_wide.iloc[val_idx]
        
        # Datasets
        # Train: Use weights, Augmentation
        train_dataset = Stage1Dataset(train_df, transform=get_image_data_transforms()[0], 
                                      species_le=species_le, fit_le=False, use_weights=USE_SAMPLE_WEIGHTS_S1)
        # Val: No weights, No Augmentation
        val_dataset = Stage1Dataset(val_df, transform=get_image_data_transforms()[1], 
                                    species_le=species_le, fit_le=False, use_weights=False)

        # Loaders - SHUFFLE=TRUE for Train - we dont have date/time in test set, so we dont care of date/time ordering
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                  shuffle=True, # No more Curriculum Sampler
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=4, pin_memory=True)

        # Model & Optimization
        model = Stage1Model(num_species=num_species).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        
        ce_loss_none = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        mse_loss_none = nn.MSELoss(reduction='none')

        best_val_loss = float('inf')
        fold_save_path = os.path.join(model_save_dir, f'stage1_fold{fold+1}.pth')
        if stratify_col is not None:
            print_stratification_stats(df_wide,train_df,val_df, STAGE1_STRATIFICATION_COLUMN,logger)    

        # --- EPOCH LOOP ---
        for epoch in range(STAGE1_EPOCHS):
            model.train()
            
            # Tracking metrics
            running_losses = {'total': 0, 'sp': 0, 'ndvi': 0, 'h': 0, 'mon': 0}
            
            # TQDM Progress Bar
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
            
            for batch in pbar:
                img, sp, ndvi, hlog, month, weight, mask = [x.to(DEVICE) for x in batch]
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                    
                    # Individual losses
                    loss_vec = torch.zeros(img.size(0), device=DEVICE)
                    
                    l_sp = torch.zeros_like(loss_vec)
                    l_ndvi = torch.zeros_like(loss_vec)
                    l_h = torch.zeros_like(loss_vec)
                    l_mon = torch.zeros_like(loss_vec)

                    if mask[:,0].any(): 
                        l_sp = ce_loss_none(sp_pred, sp) * mask[:,0].float()
                        loss_vec += 0.4 * l_sp
                    if mask[:,1].any(): 
                        l_ndvi = mse_loss_none(ndvi_pred, ndvi) * mask[:,1].float()
                        loss_vec += 0.3 * l_ndvi
                    if mask[:,2].any(): 
                        l_h = mse_loss_none(h_pred, hlog) * mask[:,2].float()
                        loss_vec += 0.2 * l_h
                    
                    l_mon = ce_loss_none(month_pred, month)
                    loss_vec += 0.1 * l_mon

                    # Apply Sample Weights
                    loss = (loss_vec * weight).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Logging updates
                running_losses['total'] += loss.item()
                running_losses['sp'] += (l_sp * weight).mean().item()
                running_losses['ndvi'] += (l_ndvi * weight).mean().item()
                running_losses['h'] += (l_h * weight).mean().item()
                running_losses['mon'] += (l_mon * weight).mean().item()

                pbar.set_postfix({
                    'L': f"{loss.item():.3f}",
                    'Sp': f"{(l_sp * weight).mean().item():.3f}"
                })

            # Scale running losses by number of batches
            num_batches = len(train_loader)
            train_log = {k: v/num_batches for k, v in running_losses.items()}

            # --- VALIDATION ---
            model.eval()
            val_losses = {'total': 0, 'sp': 0, 'ndvi': 0, 'h': 0, 'mon': 0}
            
            with torch.no_grad():
                for batch in val_loader:
                    img, sp, ndvi, hlog, month, _, mask = [x.to(DEVICE) for x in batch]
                    with torch.amp.autocast('cuda'):
                        sp_pred, ndvi_pred, h_pred, month_pred = model(img)
                        
                        # Unweighted raw losses for validation
                        l_sp = F.cross_entropy(sp_pred, sp, ignore_index=-1) if mask[:,0].any() else 0.0
                        l_ndvi = F.mse_loss(ndvi_pred[mask[:,1]], ndvi[mask[:,1]]) if mask[:,1].any() else 0.0
                        l_h = F.mse_loss(h_pred[mask[:,2]], hlog[mask[:,2]]) if mask[:,2].any() else 0.0
                        l_mon = F.cross_entropy(month_pred, month)
                        
                        total = (0.4 * l_sp) + (0.3 * l_ndvi) + (0.2 * l_h) + (0.1 * l_mon)
                        
                        val_losses['total'] += total.item() if isinstance(total, torch.Tensor) else total
                        val_losses['sp'] += l_sp.item() if isinstance(l_sp, torch.Tensor) else l_sp
                        val_losses['ndvi'] += l_ndvi.item() if isinstance(l_ndvi, torch.Tensor) else l_ndvi
                        val_losses['h'] += l_h.item() if isinstance(l_h, torch.Tensor) else l_h
                        val_losses['mon'] += l_mon.item() if isinstance(l_mon, torch.Tensor) else l_mon

            num_val = len(val_loader)
            val_log = {k: v/num_val for k, v in val_losses.items()}
            
            # Log Epoch Stats
            logger.info(
                f"Stage1 F{fold+1} E{epoch+1} | "
                f"Train: [Tot:{train_log['total']:.4f} Sp:{train_log['sp']:.3f} Nd:{train_log['ndvi']:.3f} H:{train_log['h']:.3f}] | "
                f"Val: [Tot:{val_log['total']:.4f} Sp:{val_log['sp']:.3f} Nd:{val_log['ndvi']:.3f} H:{val_log['h']:.3f}]"
            )

            # Checkpoint
            if val_log['total'] < best_val_loss:
                best_val_loss = val_log['total']
                torch.save(model.state_dict(), fold_save_path)
        
        logger.info(f"Fold {fold+1} Complete. Best Loss: {best_val_loss:.4f}. Saved to {fold_save_path}")

        # --- GENERATE OOF PREDICTIONS FOR THIS FOLD ---
        logger.info("Generating OOF predictions for current fold...")
        model.load_state_dict(torch.load(fold_save_path, weights_only=True))
        model.eval()
        
        preds = {'sp': [], 'ndvi': [], 'h': [], 'mon': [], 'sin': [], 'cos': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="OOF Preds"):
                img = batch[0].to(DEVICE)
                sp_p, nd_p, h_p, m_p = model(img)
                
                preds['sp'].extend(torch.argmax(sp_p, 1).cpu().numpy())
                preds['ndvi'].extend(nd_p.cpu().numpy())
                preds['h'].extend(h_p.cpu().numpy())
                preds['mon'].extend(torch.argmax(m_p, 1).cpu().numpy())
                preds['sin'].extend(m_p[:, 0].cpu().numpy())
                preds['cos'].extend(m_p[:, 1].cpu().numpy())
        
        # Assign to OOF dataframe
        oof_df.loc[val_idx, 'pred_species_idx'] = preds['sp']
        oof_df.loc[val_idx, 'pred_ndvi'] = preds['ndvi']
        oof_df.loc[val_idx, 'pred_height_log'] = preds['h']
        oof_df.loc[val_idx, 'pred_month'] = preds['mon']
        oof_df.loc[val_idx, 'pred_mon_sin'] = preds['sin']
        oof_df.loc[val_idx, 'pred_mon_cos'] = preds['cos']

    # --- FINALIZE OOF DATAFRAME ---
    # Convert indices back to human readable
    oof_df['pred_species'] = species_le.inverse_transform(oof_df['pred_species_idx'].astype(int))
    oof_df['pred_season'] = [get_season(m+1) for m in oof_df['pred_month'].astype(int)]
    
    # Fill Final Columns (Purely Prediction Based for Stage 2)
    oof_df['Species_final'] = oof_df['pred_species']
    oof_df['season_final'] = oof_df['pred_season']
    oof_df['NDVI_final'] = oof_df['pred_ndvi']
    oof_df['Height_final_log'] = oof_df['pred_height_log']
    
    # Save the OOF dataframe for inspection or Stage 2 reloading
    oof_df.to_csv('train_with_oof_predictions_kf.csv', index=False)
    logger.info("Saved OOF predictions to train_with_oof_predictions_kf.csv")

    return oof_df

# ====================== STAGE 2: PHYSICS-INFORMED BIOMASS ======================
class Stage2Dataset(Dataset):
    def __init__(self, df, transform=None, extra_features=None,use_weights=False):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        self.use_weights = use_weights
        # Use OOF Predictions for training features
        self.df['NDVI_final'] = pd.to_numeric(self.df['pred_ndvi'], errors='coerce').fillna(0.0).astype(np.float32)
        self.df['Height_final_log'] = pd.to_numeric(self.df['pred_height_log'], errors='coerce').fillna(0.0).astype(np.float32)
        
        # Engineering
        self.df['ndvi_h_mul'] = self.df['NDVI_final'] * self.df['Height_final_log']
        self.df['ndvi_h_ratio'] = self.df['NDVI_final'] / (self.df['Height_final_log'] + 1e-6)
        
        self.df['mon_sin'] = pd.to_numeric(self.df['pred_mon_sin'], errors='coerce').fillna(0.0).astype(np.float32)
        self.df['mon_cos'] = pd.to_numeric(self.df['pred_mon_cos'], errors='coerce').fillna(0.0).astype(np.float32)
        
        # Base Features
        self.tab_cols = ['NDVI_final', 'Height_final_log', 'ndvi_h_mul', 'ndvi_h_ratio', 'mon_sin', 'mon_cos']
        
        # --- FEATURE SELECTION: Seamless Integration ---
        if extra_features:
            # We assume these are already calculated in the dataframe passed to init
            self.tab_cols.extend(extra_features)
            
        # Handle NaNs in tabular inputs
        if self.df[self.tab_cols].isnull().any().any():
            logger.warnning ("WARNING: NaNs found in tabular inputs! Filling with 0.")
            self.df[self.tab_cols] = self.df[self.tab_cols].fillna(0.0)

        # Targets: Real and Log1p
        self.y_real = self.df[TARGET_COLS].values.astype(np.float32)
        self.y_log = np.log1p(self.y_real)
        
        # Weights for Balancing (Optional, but good for stability)
         # 2. Sample Weights 
        if self.use_weights:
            # Note: We calculate weights based on pred_species balance
            self.df, _ = calculate_sample_weights_01_normalized(self.df, group_col='pred_species', logger=logger)
        else:
            self.df['sample_weight'] = 1.0
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(f"train/{row['image_path'].split('/')[-1]}").convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform: img = self.transform(img)
        
        tab = torch.tensor(row[self.tab_cols].values.astype(np.float32))
        y_log = torch.tensor(self.y_log[idx])
        y_real = torch.tensor(self.y_real[idx])
        w = torch.tensor(row['sample_weight'], dtype=torch.float32)
        
        return img, tab, y_log, y_real, w

class Stage2ModelLog(nn.Module):
    def __init__(self, tab_dim, stage_index=None):
        super().__init__()
        
        if stage_index is not None:
            # Use features_only mode to extract from specific stage
            assert stage_index in [0, 1, 2, 3], \
                f"Invalid stage_index={stage_index}. Models have stages [0,1,2,3]."
            self.stage_index = stage_index
            self.use_features_only = True
            
            self.backbone = timm.create_model(
                BACKBONE_S2, 
                pretrained=True, 
                features_only=True,
                out_indices=(stage_index,)
            )
            
            # Get feature dimension from the specific stage
            img_feature_size = self.backbone.feature_info[stage_index]['num_chs']
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # Use standard mode (original behavior)
            self.use_features_only = False
            self.backbone = timm.create_model(BACKBONE_S2, pretrained=True, num_classes=0)
            img_feature_size = self.backbone.num_features
            self.pool = None
        
        self.mlp = nn.Sequential(
            nn.Linear(img_feature_size + tab_dim, 512),
            nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.SiLU()
        )
        self.head = nn.Linear(256, 3) # Log(1+C), Log(1+D), Log(1+G)

    def forward(self, img, tab):
        if self.use_features_only:
            # Extract from specific stage
            f = self.backbone(img)[0]
            f = f.permute(0, 3, 1, 2)  # Adjust dimensions if needed
            f = self.pool(f).squeeze(-1).squeeze(-1)
        else:
            # Standard extraction (original behavior)
            f = self.backbone(img)
            if len(f.shape) > 2: 
                f = f.mean([2, 3])
        
        x = torch.cat([f, tab], dim=1)
        # Softplus ensures positive mass output
        log_comp = F.softplus(self.head(self.mlp(x)))
        l_c, l_d, l_g = log_comp[:, 0:1], log_comp[:, 1:2], log_comp[:, 2:3]
        
        # Physics: Log -> Real
        r_c = torch.expm1(l_c)
        r_d = torch.expm1(l_d)
        r_g = torch.expm1(l_g)
        
        # Physics: Summation
        r_tot = r_c + r_d + r_g
        r_gdm = r_c + r_g
        
        # Physics: Real -> Log (For Loss)
        l_tot = torch.log1p(r_tot)
        l_gdm = torch.log1p(r_gdm)
        
        # Concatenate outputs
        pred_log = torch.cat([l_c, l_d, l_g, l_tot, l_gdm], dim=1)
        pred_real = torch.cat([r_c, r_d, r_g, r_tot, r_gdm], dim=1)
        
        return pred_log, pred_real

def train_stage2(df):
    logger.info("=== STAGE 2: Physics-Informed Biomass Regression ===")
    logger.info(f"Total Training Samples: {len(df)}")
    logger.info(f"Test Split Ratio: {TEST_SPLIT_RATIO}, Stratify Column: {STAGE2_STRATIFICATION_COLUMN}")
    logger.info(f"Using Count/Frequency Features: {USE_COUNT_FEATURES}")
    logger.info(f"Stage 2 Epochs: {STAGE2_EPOCHS}")
    logger.info(f"Official Weights: {OFFICIAL_WEIGHTS}")
    logger.info(f"Column Weights Tensor: {COL_WEIGHTS_TENSOR}")
    logger.info(f"Backbone: {BACKBONE_S2}")    
    logger.info(f"Device: {DEVICE}")
    
    if STAGE2_STRATIFICATION_COLUMN not in df.columns:
        logger.warning(f"Stratification column '{STAGE2_STRATIFICATION_COLUMN}' not found in dataframe!")
        logger.info("Proceeding without stratification...")
        tr_df, val_df = train_test_split(df, test_size=TEST_SPLIT_RATIO, random_state=42)
    else:
        logger.info(f"Stratifying Stage 2 on column: {STAGE2_STRATIFICATION_COLUMN}")
        tr_df, val_df = train_test_split(df, test_size=TEST_SPLIT_RATIO, random_state=42,stratify=df[STAGE2_STRATIFICATION_COLUMN])
        print_stratification_stats(df,tr_df,val_df, STAGE2_STRATIFICATION_COLUMN,logger)    

    # --- DYNAMIC FEATURE CALCULATION ---
    extra_feats = []
    if USE_COUNT_FEATURES:
        logger.info("Enabled: Calculating Count/Frequency features for Stage 2...")
        # We calculate statistics on Train and apply to Val to prevent leakage
        tr_df, val_df, extra_feats = calculate_count_frequency_features(
            tr_df, val_df, 
            group_col='pred_species',      # Count occurrences of this Predicted Species
            local_group_col='season_final', # ... specifically within this Predicted Season
            logger=logger
        )
    
    tr_ds = Stage2Dataset(tr_df, get_image_data_transforms()[0], extra_features=extra_feats,use_weights=USE_SAMPLE_WEIGHTS_S2)
    val_ds = Stage2Dataset(val_df, get_image_data_transforms()[1], extra_features=extra_feats,use_weights=False)
    
    tr_load = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    val_load = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model input dimension adapts automatically
    model = Stage2ModelLog(len(tr_ds.tab_cols),stage_index=1).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=STAGE2_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    
    # Recommended: MSELoss for R2 maximization
    criterion = nn.MSELoss(reduction='none') 
    
    best_r2 = -float('inf')
    component_names = ['Clover', 'Dead', 'Green', 'Total', 'GDM']
    
    for ep in range(STAGE2_EPOCHS):
        model.train()
        
        # Initialize trackers for each component
        # We track the "Weighted Loss" to see how much each component contributes to the optimization
        train_metrics = {name: 0.0 for name in component_names}
        train_metrics['Global_Loss'] = 0.0
        
        pbar = tqdm(tr_load, leave=False, desc=f"Ep {ep+1}")
        
        for img, tab, y_log, _, w in pbar:
            img, tab, y_log, w = img.to(DEVICE), tab.to(DEVICE), y_log.to(DEVICE), w.to(DEVICE)
            
            optim.zero_grad()
            with torch.amp.autocast('cuda'):
                p_log, _ = model(img, tab)
                
                # Calculate Raw MSE per component (B, 5)
                raw_loss = criterion(p_log, y_log)
                
                # Apply Component Weights (B, 5)
                weighted_loss_components = raw_loss * COL_WEIGHTS_TENSOR
                
                # Sum components to get sample loss (B)
                sample_loss = weighted_loss_components.sum(dim=1)
                
                # Apply Sample Weight (Class balancing) and Mean
                final_loss = (sample_loss * w).mean()
                
            scaler.scale(final_loss).backward()
            scaler.step(optim)
            scaler.update()
            
            # --- Detailed Logging Accumulation ---
            train_metrics['Global_Loss'] += final_loss.item()
            
            # Detach and calculate mean weighted loss per component for logging
            with torch.no_grad():
                # We average over the batch, respecting sample weights 'w'
                # shape: (B, 5) * (B, 1) -> (B, 5) -> mean(0) -> (5,)
                batch_comp_loss = (weighted_loss_components * w.unsqueeze(1)).mean(dim=0)
                
                for i, name in enumerate(component_names):
                    train_metrics[name] += batch_comp_loss[i].item()
            
            # Simple progress bar
            pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})
            
        # Normalize Train Metrics
        num_batches = len(tr_load)
        train_log_str = " | ".join([f"{k}: {v/num_batches:.4f}" for k, v in train_metrics.items()])
        
        # --- VALIDATION ---
        model.eval()
        all_pred, all_true = [], []
        
        # We also track validation loss breakdown
        val_metrics = {name: 0.0 for name in component_names}
        val_metrics['Global_Loss'] = 0.0
        
        with torch.no_grad():
            for img, tab, y_log, y_real, _ in val_load:
                img, tab, y_log = img.to(DEVICE), tab.to(DEVICE), y_log.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    p_log, p_real = model(img, tab)
                    
                    # Loss Calculation (Same as train)
                    raw_loss = criterion(p_log, y_log)
                    weighted_loss = raw_loss * COL_WEIGHTS_TENSOR
                    loss_val = weighted_loss.sum(1).mean() # No sample weights in val
                    
                    # Accumulate
                    val_metrics['Global_Loss'] += loss_val.item()
                    mean_comp_loss = weighted_loss.mean(dim=0) # Average over batch
                    for i, name in enumerate(component_names):
                        val_metrics[name] += mean_comp_loss[i].item()
                
                all_pred.append(p_real.float().cpu().numpy())
                all_true.append(y_real.float().cpu().numpy())
        
        # Normalize Validation Loss
        num_val = len(val_load)
        val_loss_str = " | ".join([f"{k}: {v/num_val:.4f}" for k, v in val_metrics.items()])
        
        # --- METRICS CALCULATION ---
        y_p_arr = np.concatenate(all_pred)
        y_t_arr = np.concatenate(all_true)
        
        # 1. Global Weighted R2
        r2 = calculate_global_weighted_r2(y_t_arr, y_p_arr, OFFICIAL_WEIGHTS)
        
        # 2. MAE per column (In Grams)
        mae_per_col = np.abs(y_t_arr - y_p_arr).mean(0)
        mae_str = " | ".join([f"{name}: {mae:.1f}g" for name, mae in zip(component_names, mae_per_col)])
        
        # --- FINAL LOGGING ---
        logger.info(
                f"Stage2 E{ep+1} | "
                f"Train: [{train_log_str}] "
        )
        logger.info(
                f"Stage2 E{ep+1} | "
                f"Val:   [{val_loss_str}]"
        )
        logger.info(f"MAE:   [{mae_str}]")
        logger.info(f"Global Weighted R²: {r2:.5f}")
        
        if r2 > best_r2:
            best_r2 = r2
            os.makedirs('models_stage2', exist_ok=True)
            torch.save(model.state_dict(), 'models_stage2/best_model.pth')
            logger.info(f">> NEW BEST MODEL SAVED! ({r2:.5f})")
            
        sched.step()

# ====================== MAIN EXECUTION ======================
if __name__ == '__main__':
    setup_logging(logger_name="System Logger",log_dir='logs', file_name_part='KFold')
    set_seed(42)
    
    if os.path.exists('train_with_oof_predictions_kf.csv'):
        df_oof = pd.read_csv('train_with_oof_predictions_kf.csv')
        train_stage2(df_oof)
    else:
        logger.error("OOF file not found. Training run Stage 1 first.")
        train_stage1_kfold(load_data(logger))
        df_oof = pd.read_csv('train_with_oof_predictions_kf.csv')
        train_stage2(df_oof)