# 2-Stage K-Fold Trainer with Physics-Constrained Architecture
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
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
from tqdm import tqdm

# ====================== IMPORTS ======================
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
    COL_WEIGHTS_TENSOR
)
from common import (
    calculate_global_weighted_r2, calculate_sample_weights_smooth,
    get_image_data_transforms, load_data, print_stratification_stats, setup_logging, set_seed,
    calculate_count_frequency_features
)

set_seed(42)

logger = logging.getLogger('System Logger')
logger.setLevel(logging.INFO)

# ====================== STAGE 1: AUXILIARY MULTI-TASK ======================
class Stage1Dataset(Dataset):
    """
    Stage 1 focuses on learning auxiliary features from images:
    - Species classification
    - NDVI regression  
    - Height regression
    """
    def __init__(self, df, transform=None, species_le=None, fit_le=False, use_weights=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_weights = use_weights

        # Label Encoding for Species
        if fit_le:
            self.species_le = LabelEncoder()
            self.df['species_label'] = self.species_le.fit_transform(self.df['Species'].fillna('Unknown'))
        else:
            self.species_le = species_le
            self.df['species_label'] = self.df['Species'].fillna('Unknown').map(
                lambda x: self.species_le.transform([x])[0] if x in self.species_le.classes_ else -1
            )

        # Sample Weights for class balancing
        if self.use_weights:
            self.df, _ = calculate_sample_weights_smooth(self.df, group_col='Species', smooth=5.0, logger=None)
        else:
            self.df['sample_weight'] = 1.0

        # Log transform height for better distribution
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
        ndvi = row['Pre_GSHH_NDVI'] 
        height_log = row['Height_Ave_cm_log']
        weight = torch.tensor(row['sample_weight'], dtype=torch.float32)

        # Mask for handling missing values in loss
        mask = torch.tensor([
            species_label != -1,
            pd.notna(row['Pre_GSHH_NDVI']),
            pd.notna(row['Height_Ave_cm'])
        ], dtype=torch.bool)

        return (
            img,
            torch.tensor(species_label, dtype=torch.long),
            torch.tensor(ndvi, dtype=torch.float32),
            torch.tensor(height_log, dtype=torch.float32),
            weight,
            mask
        )

    def get_species_encoder(self):
        return self.species_le


class Stage1Model(nn.Module):
    """
    Multi-task model predicting auxiliary features:
    - Species (classification)
    - NDVI (regression)
    - Height (regression)
    """
    def __init__(self, num_species):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_S1, pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        
        # Multi-task heads
        self.species_head = nn.Linear(feat, num_species)
        self.ndvi_head = nn.Linear(feat, 1)
        self.height_head = nn.Linear(feat, 1)

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.species_head(f),
            self.ndvi_head(f).squeeze(1),
            self.height_head(f).squeeze(1)
        )


def train_stage1_kfold(df_wide):
    """Train Stage 1 with K-Fold cross-validation to generate OOF predictions"""
    logger.info(f"=== STAGE 1: K-Fold Multi-Task Training ===")
    logger.info(f"Total Samples: {len(df_wide)}")
    logger.info(f"Using Sample Weights: {USE_SAMPLE_WEIGHTS_S1}")
    logger.info(f"Epochs: {STAGE1_EPOCHS}")
    logger.info(f"Backbone: {BACKBONE_S1}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Stratification Column: {STAGE1_STRATIFICATION_COLUMN}")
    logger.info(f"Device: {DEVICE}")

    # Global Label Encoder (fit on all data for consistency across folds)
    species_le = LabelEncoder()
    species_le.fit(df_wide['Species'].fillna('Unknown'))
    num_species = len(species_le.classes_)
    
    # Initialize OOF prediction columns
    oof_df = df_wide.copy()
    oof_df['pred_species_idx'] = -1
    oof_df['pred_ndvi'] = np.nan
    oof_df['pred_height_log'] = np.nan
    
    model_save_dir = 'models_stage1'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save metadata for inference
    metadata = {
        'species_encoder': species_le,
        'backbone_name': BACKBONE_S1,
        'num_species': num_species
    }
    torch.save(metadata, os.path.join(model_save_dir, 'stage1_metadata.pth'))

    # Setup K-Fold splitter
    if STAGE1_STRATIFICATION_COLUMN and STAGE1_STRATIFICATION_COLUMN in df_wide.columns:
        stratify_col = df_wide[STAGE1_STRATIFICATION_COLUMN]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)    
        splitter = skf.split(df_wide, stratify_col)
    else:
        logger.warning(f"Stratification column '{STAGE1_STRATIFICATION_COLUMN}' not found. Using regular KFold.")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splitter = kf.split(df_wide)
        stratify_col = None

    # K-Fold training loop
    for fold, (train_idx, val_idx) in enumerate(splitter):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold+1}/{N_FOLDS}")
        logger.info(f"{'='*60}")
        
        train_df = df_wide.iloc[train_idx]
        val_df = df_wide.iloc[val_idx]
        
        # Create datasets
        train_dataset = Stage1Dataset(
            train_df, 
            transform=get_image_data_transforms()[0], 
            species_le=species_le, 
            fit_le=False, 
            use_weights=USE_SAMPLE_WEIGHTS_S1
        )
        val_dataset = Stage1Dataset(
            val_df, 
            transform=get_image_data_transforms()[1], 
            species_le=species_le, 
            fit_le=False, 
            use_weights=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        # Initialize model and optimization
        model = Stage1Model(num_species=num_species).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        
        ce_loss_none = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        mse_loss_none = nn.MSELoss(reduction='none')

        best_val_loss = float('inf')
        fold_save_path = os.path.join(model_save_dir, f'stage1_fold{fold+1}.pth')
        
        if stratify_col is not None:
            print_stratification_stats(df_wide, train_df, val_df, STAGE1_STRATIFICATION_COLUMN, logger)

        # Training epochs
        for epoch in range(STAGE1_EPOCHS):
            model.train()
            running_losses = {'total': 0, 'sp': 0, 'ndvi': 0, 'h': 0}
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False)
            
            for batch in pbar:
                img, sp, ndvi, hlog, weight, mask = [x.to(DEVICE) for x in batch]
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    sp_pred, ndvi_pred, h_pred = model(img)
                    
                    # Compute individual losses
                    loss_vec = torch.zeros(img.size(0), device=DEVICE)
                    l_sp = torch.zeros_like(loss_vec)
                    l_ndvi = torch.zeros_like(loss_vec)
                    l_h = torch.zeros_like(loss_vec)

                    if mask[:, 0].any(): 
                        l_sp = ce_loss_none(sp_pred, sp) * mask[:, 0].float()
                        loss_vec += 0.4 * l_sp
                    if mask[:, 1].any(): 
                        l_ndvi = mse_loss_none(ndvi_pred, ndvi) * mask[:, 1].float()
                        loss_vec += 0.3 * l_ndvi
                    if mask[:, 2].any(): 
                        l_h = mse_loss_none(h_pred, hlog) * mask[:, 2].float()
                        loss_vec += 0.3 * l_h

                    # Apply sample weights
                    loss = (loss_vec * weight).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                running_losses['total'] += loss.item()
                running_losses['sp'] += (l_sp * weight).mean().item()
                running_losses['ndvi'] += (l_ndvi * weight).mean().item()
                running_losses['h'] += (l_h * weight).mean().item()

                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            # Calculate average training losses
            num_batches = len(train_loader)
            train_log = {k: v / num_batches for k, v in running_losses.items()}

            # Validation
            model.eval()
            val_losses = {'total': 0, 'sp': 0, 'ndvi': 0, 'h': 0}
            
            with torch.no_grad():
                for batch in val_loader:
                    img, sp, ndvi, hlog, _, mask = [x.to(DEVICE) for x in batch]
                    with torch.amp.autocast('cuda'):
                        sp_pred, ndvi_pred, h_pred = model(img)
                        
                        # Compute unweighted losses for validation
                        l_sp = F.cross_entropy(sp_pred, sp, ignore_index=-1) if mask[:, 0].any() else 0.0
                        l_ndvi = F.mse_loss(ndvi_pred[mask[:, 1]], ndvi[mask[:, 1]]) if mask[:, 1].any() else 0.0
                        l_h = F.mse_loss(h_pred[mask[:, 2]], hlog[mask[:, 2]]) if mask[:, 2].any() else 0.0
                        
                        total = (0.4 * l_sp) + (0.3 * l_ndvi) + (0.3 * l_h)
                        
                        val_losses['total'] += total.item() if isinstance(total, torch.Tensor) else total
                        val_losses['sp'] += l_sp.item() if isinstance(l_sp, torch.Tensor) else l_sp
                        val_losses['ndvi'] += l_ndvi.item() if isinstance(l_ndvi, torch.Tensor) else l_ndvi
                        val_losses['h'] += l_h.item() if isinstance(l_h, torch.Tensor) else l_h

            num_val = len(val_loader)
            val_log = {k: v / num_val for k, v in val_losses.items()}
            
            # Log epoch results
            logger.info(
                f"F{fold+1} E{epoch+1:02d} | "
                f"Train [Tot:{train_log['total']:.4f} Sp:{train_log['sp']:.3f} "
                f"NDVI:{train_log['ndvi']:.3f} H:{train_log['h']:.3f}] | "
                f"Val [Tot:{val_log['total']:.4f} Sp:{val_log['sp']:.3f} "
                f"NDVI:{val_log['ndvi']:.3f} H:{val_log['h']:.3f}]"
            )

            # Save best model
            if val_log['total'] < best_val_loss:
                best_val_loss = val_log['total']
                torch.save(model.state_dict(), fold_save_path)
        
        logger.info(f"✓ Fold {fold+1} complete. Best Val Loss: {best_val_loss:.4f}")

        # Generate OOF predictions
        logger.info("Generating OOF predictions...")
        model.load_state_dict(torch.load(fold_save_path, weights_only=True))
        model.eval()
        
        preds = {'sp': [], 'ndvi': [], 'h': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="OOF Inference", leave=False):
                img = batch[0].to(DEVICE)
                sp_p, nd_p, h_p = model(img)
                
                preds['sp'].extend(torch.argmax(sp_p, 1).cpu().numpy())
                preds['ndvi'].extend(nd_p.cpu().numpy())
                preds['h'].extend(h_p.cpu().numpy())
        
        # Store OOF predictions
        oof_df.loc[val_idx, 'pred_species_idx'] = preds['sp']
        oof_df.loc[val_idx, 'pred_ndvi'] = preds['ndvi']
        oof_df.loc[val_idx, 'pred_height_log'] = preds['h']

    # Convert predictions to human-readable format
    oof_df['pred_species'] = species_le.inverse_transform(oof_df['pred_species_idx'].astype(int))
    
    # Finalize features for Stage 2
    oof_df['Species_final'] = oof_df['pred_species']
    oof_df['NDVI_final'] = oof_df['pred_ndvi']
    oof_df['Height_final_log'] = oof_df['pred_height_log']
    
    # Save OOF predictions
    oof_df.to_csv('train_with_oof_predictions.csv', index=False)
    logger.info(f"\n✓ Stage 1 Complete. OOF predictions saved to 'train_with_oof_predictions.csv'")

    return oof_df


# ====================== STAGE 2: PHYSICS-CONSTRAINED BIOMASS ======================
class Stage2Dataset(Dataset):
    """
    Stage 2 uses OOF predictions from Stage 1 as features to predict biomass components.
    Physics constraint: Total = Clover + Dead + Green
    """
    def __init__(self, df, transform=None, extra_features=None):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        
        # Use OOF predictions as features
        self.df['NDVI_final'] = pd.to_numeric(self.df['pred_ndvi'], errors='coerce').fillna(0.0).astype(np.float32)
        self.df['Height_final_log'] = pd.to_numeric(self.df['pred_height_log'], errors='coerce').fillna(0.0).astype(np.float32)
        
        # Feature engineering: interactions
        self.df['ndvi_h_mul'] = self.df['NDVI_final'] * self.df['Height_final_log']
        self.df['ndvi_h_ratio'] = self.df['NDVI_final'] / (self.df['Height_final_log'] + 1e-6)
        
        # Base tabular features
        self.tab_cols = ['NDVI_final', 'Height_final_log', 'ndvi_h_mul', 'ndvi_h_ratio']
        
        # Add extra features if provided
        if extra_features:
            self.tab_cols.extend(extra_features)
            
        # Handle NaNs
        if self.df[self.tab_cols].isnull().any().any():
            logger.warning("NaNs found in tabular inputs! Filling with 0.")
            self.df[self.tab_cols] = self.df[self.tab_cols].fillna(0.0)

        # Targets
        self.y_real = self.df[TARGET_COLS].values.astype(np.float32)
        self.y_log = np.log1p(self.y_real)
        
        # Sample weights for balancing
        self.df, _ = calculate_sample_weights_smooth(self.df, 'pred_species', smooth=5.0, logger=logger)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(f"train/{row['image_path'].split('/')[-1]}").convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            img = self.transform(img)
        
        tab = torch.tensor(row[self.tab_cols].values.astype(np.float32))
        y_log = torch.tensor(self.y_log[idx])
        y_real = torch.tensor(self.y_real[idx])
        w = torch.tensor(row['sample_weight'], dtype=torch.float32)
        
        return img, tab, y_log, y_real, w


class Stage2ModelLog(nn.Module):
    """
    Physics-constrained model with auxiliary heads.
    
    Primary heads predict: Clover, Dead, Green
    Auxiliary heads help Dead prediction: Unexplained Mass, Total-Green
    Physics constraints: Total = C+D+G, GDM = C+G
    """
    def __init__(self, tab_dim, stage_index=None):
        super().__init__()
        
        # Image backbone
        if stage_index is not None:
            assert stage_index in [0, 1, 2, 3], f"Invalid stage_index={stage_index}"
            self.stage_index = stage_index
            self.use_features_only = True
            
            self.backbone = timm.create_model(
                BACKBONE_S2, 
                pretrained=True, 
                features_only=True,
                out_indices=(stage_index,)
            )
            
            img_feature_size = self.backbone.feature_info[stage_index]['num_chs']
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.use_features_only = False
            self.backbone = timm.create_model(BACKBONE_S2, pretrained=True, num_classes=0)
            img_feature_size = self.backbone.num_features
            self.pool = None
        
        # MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(img_feature_size + tab_dim, 512),
            nn.BatchNorm1d(512), 
            nn.SiLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.SiLU()
        )
        
        # Primary prediction heads (base components)
        self.head_clover = nn.Linear(256, 1)
        self.head_dead = nn.Linear(256, 1)
        self.head_green = nn.Linear(256, 1)
        
        # Auxiliary heads (help Dead learning via residual relationships)
        self.head_unexplained = nn.Linear(256, 1)  # Should ≈ Dead
        self.head_total_minus_green = nn.Linear(256, 1)  # Should ≈ Dead + Clover

    def forward(self, img, tab):
        # Extract image features
        if self.use_features_only:
            f = self.backbone(img)[0]
            f = f.permute(0, 3, 1, 2)
            f = self.pool(f).squeeze(-1).squeeze(-1)
        else:
            f = self.backbone(img)
            if len(f.shape) > 2: 
                f = f.mean([2, 3])
        
        # Fuse image and tabular features
        x = torch.cat([f, tab], dim=1)
        features = self.mlp(x)
        
        # Primary predictions (log space, Softplus ensures positivity)
        l_c = F.softplus(self.head_clover(features).squeeze(1))
        l_d = F.softplus(self.head_dead(features).squeeze(1))
        l_g = F.softplus(self.head_green(features).squeeze(1))
        
        # Auxiliary predictions (for better Dead learning)
        l_unexplained = F.softplus(self.head_unexplained(features).squeeze(1))
        l_total_minus_green = F.softplus(self.head_total_minus_green(features).squeeze(1))
        
        # Convert to real scale
        r_c = torch.expm1(l_c)
        r_d = torch.expm1(l_d)
        r_g = torch.expm1(l_g)
        
        # Physics constraints: Calculate aggregates
        r_tot = r_c + r_d + r_g  # Total = Clover + Dead + Green
        r_gdm = r_c + r_g         # GDM = Clover + Green
        
        # Back to log space for loss calculation
        l_tot = torch.log1p(r_tot)
        l_gdm = torch.log1p(r_gdm)
        
        # Stack predictions: [Clover, Dead, Green, Total, GDM]
        pred_log = torch.stack([l_c, l_d, l_g, l_tot, l_gdm], dim=1)
        pred_real = torch.stack([r_c, r_d, r_g, r_tot, r_gdm], dim=1)
        
        # Auxiliary predictions
        aux_unexplained = l_unexplained
        aux_tmg = l_total_minus_green
        
        return pred_log, pred_real, aux_unexplained, aux_tmg


def train_stage2(df):
    """Train Stage 2 with physics-constrained architecture"""
    logger.info(f"\n{'='*60}")
    logger.info("STAGE 2: Physics-Constrained Biomass Regression")
    logger.info(f"{'='*60}")
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Test Split Ratio: {TEST_SPLIT_RATIO}")
    logger.info(f"Stratification: {STAGE2_STRATIFICATION_COLUMN if STAGE2_STRATIFICATION_COLUMN else 'None'}")
    logger.info(f"Count Features: {USE_COUNT_FEATURES}")
    logger.info(f"Epochs: {STAGE2_EPOCHS}")
    logger.info(f"Backbone: {BACKBONE_S2}")
    logger.info(f"Official Weights: {OFFICIAL_WEIGHTS}")
    logger.info(f"Device: {DEVICE}")
    
    # Train/Val split
    if STAGE2_STRATIFICATION_COLUMN and STAGE2_STRATIFICATION_COLUMN in df.columns:
        logger.info(f"Stratifying on: {STAGE2_STRATIFICATION_COLUMN}")
        tr_df, val_df = train_test_split(
            df, 
            test_size=TEST_SPLIT_RATIO, 
            random_state=42,
            stratify=df[STAGE2_STRATIFICATION_COLUMN]
        )
        print_stratification_stats(df, tr_df, val_df, STAGE2_STRATIFICATION_COLUMN, logger)
    else:
        logger.info("No stratification applied")
        tr_df, val_df = train_test_split(df, test_size=TEST_SPLIT_RATIO, random_state=42)

    # Calculate count/frequency features if enabled
    extra_feats = []
    if USE_COUNT_FEATURES:
        logger.info("Calculating count/frequency features...")
        tr_df, val_df, extra_feats = calculate_count_frequency_features(
            tr_df, val_df, 
            group_col='pred_species',
            local_group_col='season',
            logger=logger
        )
    
    # Create datasets
    tr_ds = Stage2Dataset(tr_df, get_image_data_transforms()[0], extra_features=extra_feats)
    val_ds = Stage2Dataset(val_df, get_image_data_transforms()[1], extra_features=extra_feats)
    
    tr_load = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_load = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = Stage2ModelLog(len(tr_ds.tab_cols), stage_index=1).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=STAGE2_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    
    criterion = nn.MSELoss(reduction='none')
    
    best_r2 = -float('inf')
    component_names = ['Clover', 'Dead', 'Green', 'Total', 'GDM']
    
    logger.info(f"\n{'='*60}")
    logger.info("Starting Training...")
    logger.info(f"{'='*60}")
    
    for ep in range(STAGE2_EPOCHS):
        model.train()
        train_metrics = {
            'primary_loss': 0.0,
            'aux_unexplained': 0.0,
            'aux_tmg': 0.0,
            'total_loss': 0.0
        }
        
        pbar = tqdm(tr_load, leave=False, desc=f"Epoch {ep+1:02d}")
        
        for img, tab, y_log, y_real, w in pbar:
            img, tab, y_log, y_real, w = img.to(DEVICE), tab.to(DEVICE), y_log.to(DEVICE), y_real.to(DEVICE), w.to(DEVICE)
            
            optim.zero_grad()
            with torch.amp.autocast('cuda'):
                p_log, _, aux_unexp, aux_tmg = model(img, tab)
                
                # Primary loss: Weighted MSE on all 5 targets
                raw_loss = criterion(p_log, y_log)  # (B, 5)
                weighted_loss = raw_loss * COL_WEIGHTS_TENSOR  # Apply official weights
                sample_loss = weighted_loss.sum(dim=1)  # (B,)
                primary_loss = (sample_loss * w).mean()
                
                # Auxiliary losses (help Dead learning)
                # Target: Dead is at index 1 in y_log
                aux_loss_unexp = criterion(aux_unexp, y_log[:, 1]).mean()
                # Target: Dead + Clover = Total - Green
                target_tmg = torch.log1p(torch.expm1(y_log[:, 0]) + torch.expm1(y_log[:, 1]))  # C + D in log
                aux_loss_tmg = criterion(aux_tmg, target_tmg).mean()
                
                # Combined loss
                total_loss = 0.8 * primary_loss + 0.1 * aux_loss_unexp + 0.1 * aux_loss_tmg
                
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
            
            # Track metrics
            train_metrics['primary_loss'] += primary_loss.item()
            train_metrics['aux_unexplained'] += aux_loss_unexp.item()
            train_metrics['aux_tmg'] += aux_loss_tmg.item()
            train_metrics['total_loss'] += total_loss.item()
            
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
        
        # Average training metrics
        num_batches = len(tr_load)
        for k in train_metrics:
            train_metrics[k] /= num_batches
        
        # Validation
        model.eval()
        all_pred_real = []
        all_true_real = []
        val_losses = {'primary': 0.0, 'aux_unexp': 0.0, 'aux_tmg': 0.0, 'total': 0.0}
        
        with torch.no_grad():
            for img, tab, y_log, y_real, _ in val_load:
                img, tab, y_log, y_real = img.to(DEVICE), tab.to(DEVICE), y_log.to(DEVICE), y_real.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    p_log, p_real, aux_unexp, aux_tmg = model(img, tab)
                    
                    # Primary loss
                    raw_loss = criterion(p_log, y_log)
                    weighted_loss = raw_loss * COL_WEIGHTS_TENSOR
                    val_primary = weighted_loss.sum(1).mean()
                    
                    # Auxiliary losses
                    val_aux_unexp = criterion(aux_unexp, y_log[:, 1]).mean()
                    target_tmg = torch.log1p(torch.expm1(y_log[:, 0]) + torch.expm1(y_log[:, 1]))
                    val_aux_tmg = criterion(aux_tmg, target_tmg).mean()
                    
                    val_total = 0.8 * val_primary + 0.1 * val_aux_unexp + 0.1 * val_aux_tmg
                    
                    val_losses['primary'] += val_primary.item()
                    val_losses['aux_unexp'] += val_aux_unexp.item()
                    val_losses['aux_tmg'] += val_aux_tmg.item()
                    val_losses['total'] += val_total.item()
                
                all_pred_real.append(p_real.float().cpu().numpy())
                all_true_real.append(y_real.float().cpu().numpy())
        
        # Average validation losses
        num_val = len(val_load)
        for k in val_losses:
            val_losses[k] /= num_val
        
        # Calculate metrics
        y_pred = np.concatenate(all_pred_real)
        y_true = np.concatenate(all_true_real)
        
        # Global Weighted R²
        r2 = calculate_global_weighted_r2(y_true, y_pred, OFFICIAL_WEIGHTS)
        
        # Per-component R²
        r2_per_component = []
        for i in range(5):
            r2_comp = r2_score(y_true[:, i], y_pred[:, i])
            r2_per_component.append(r2_comp)
        
        # MAE per component
        mae_per_col = np.abs(y_true - y_pred).mean(0)
        
        # Logging
        logger.info(
            f"E{ep+1:02d} | Train [Primary:{train_metrics['primary_loss']:.4f} "
            f"AuxU:{train_metrics['aux_unexplained']:.4f} AuxTMG:{train_metrics['aux_tmg']:.4f} "
            f"Total:{train_metrics['total_loss']:.4f}]"
        )
        logger.info(
            f"     | Val   [Primary:{val_losses['primary']:.4f} "
            f"AuxU:{val_losses['aux_unexp']:.4f} AuxTMG:{val_losses['aux_tmg']:.4f} "
            f"Total:{val_losses['total']:.4f}]"
        )
        
        # Component-wise metrics
        r2_str = " | ".join([f"{name}: {r2_per_component[i]:.3f}" 
                             for i, name in enumerate(component_names)])
        logger.info(f"     | R² per component: [{r2_str}]")
        
        mae_str = " | ".join([f"{name}: {mae:.1f}g" 
                              for name, mae in zip(component_names, mae_per_col)])
        logger.info(f"     | MAE: [{mae_str}]")
        
        logger.info(f"     | ★ Global Weighted R²: {r2:.5f}")
        
        # Save best model
        if r2 > best_r2:
            best_r2 = r2
            os.makedirs('models_stage2', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'tab_cols': tr_ds.tab_cols,
                'r2': r2,
                'epoch': ep
            }, 'models_stage2/best_model.pth')
            logger.info(f"     | ✓ NEW BEST MODEL SAVED! (R²={r2:.5f})")
        
        sched.step()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Stage 2 Complete. Best Global Weighted R²: {best_r2:.5f}")
    logger.info(f"{'='*60}")


# ====================== MAIN EXECUTION ======================
if __name__ == '__main__':
    setup_logging(logger_name="System Logger", log_dir='logs', file_name_part='2Stage_Clean')
    set_seed(42)
    
    # Check if OOF predictions exist
    if os.path.exists('train_with_oof_predictions.csv'):
        logger.info("Found existing OOF predictions. Loading for Stage 2...")
        df_oof = pd.read_csv('train_with_oof_predictions.csv')
        train_stage2(df_oof)
    else:
        logger.info("No OOF predictions found. Running Stage 1 first...")
        df_wide = load_data(logger)
        df_oof = train_stage1_kfold(df_wide)
        train_stage2(df_oof)