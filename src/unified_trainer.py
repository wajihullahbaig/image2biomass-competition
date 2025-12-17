import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import os
from tqdm import tqdm

# ====================== CONFIG IMPORTS ======================
from configs import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, 
    BACKBONE_S1, LEARNING_RATE, 
    N_FOLDS, STAGE1_STRATIFICATION_COLUMN,
    STAGE1_EPOCHS, TARGET_COLS, OFFICIAL_WEIGHTS,
    COL_WEIGHTS_TENSOR, USE_SAMPLE_WEIGHTS_S1,
    IMG_FEAT_WEIGHT, TAB_FEAT_WEIGHT, FUSION_DIM
)
from common import (
    calculate_global_weighted_r2, calculate_sample_weights_01_normalized,
    get_image_data_transforms, load_data, setup_logging, set_seed, plot_fold_losses
)

set_seed(42)
logger = logging.getLogger('System Logger')

# ====================== UNIFIED DATASET ======================
class UnifiedDataset(Dataset):
    def __init__(self, df, transform=None, species_le=None, use_weights=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_weights = use_weights

        # --- Stage 1 Prep ---
        if species_le:
            self.df['species_label'] = self.df['Species'].fillna('Unknown').map(
                lambda x: species_le.transform([x])[0] if x in species_le.classes_ else -1
            )
        self.df['Height_Ave_cm_log'] = np.log1p(self.df['Height_Ave_cm'])
        
        # --- Stage 2 Prep ---
        self.y_real = self.df[TARGET_COLS].values.astype(np.float32)
        self.y_log = np.log1p(self.y_real)

        # Sample Weights
        if self.use_weights:
            self.df, _ = calculate_sample_weights_01_normalized(self.df, group_col='Species', logger=logger)
        else:
            self.df['sample_weight'] = 1.0

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

        # S1 Targets
        species_label = int(row['species_label'])
        ndvi_target = row['Pre_GSHH_NDVI'] 
        height_target_log = row['Height_Ave_cm_log'] 
        month_target = int(row['month'] - 1)
        
        # S2 Targets
        bio_log = torch.tensor(self.y_log[idx], dtype=torch.float32)
        bio_real = torch.tensor(self.y_real[idx], dtype=torch.float32)

        weight = torch.tensor(row['sample_weight'], dtype=torch.float32)
        
        # Mask for missing S1 targets
        mask = torch.tensor([
            species_label != -1,
            pd.notna(row['Pre_GSHH_NDVI']),
            pd.notna(row['Height_Ave_cm']),
            True 
        ], dtype=torch.bool)

        return (
            img, 
            torch.tensor(species_label, dtype=torch.long),
            torch.tensor(ndvi_target, dtype=torch.float32),
            torch.tensor(height_target_log, dtype=torch.float32),
            torch.tensor(month_target, dtype=torch.long),
            bio_log,
            bio_real,
            weight,
            mask
        )

# ====================== SHARED BACKBONE MODEL ======================
class UnifiedSharedModel(nn.Module):
    def __init__(self, num_species, backbone_name=BACKBONE_S1):
        super().__init__()
        # 1. SHARED BACKBONE (EfficientNet)
        # num_classes=0 returns the pooled feature vector (Batch, Num_Features)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        
        # 2. STAGE 1 HEADS (Attached to Shared Backbone)
        self.s1_species = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, num_species))
        self.s1_ndvi = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 1))
        self.s1_height = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 1))
        self.s1_month = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 12))

        # 3. STAGE 2 HEAD (Attached to Shared Backbone + Tabular + Species)
        # Species Embedding (for S2 conditioning)
        self.species_emb = nn.Embedding(num_species, 16)
        
        # Feature Fusion Adapters
        self.img_adapter = nn.Sequential(
            nn.Linear(feat_dim, FUSION_DIM),
            nn.BatchNorm1d(FUSION_DIM),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        self.tab_adapter = nn.Sequential(
            nn.Linear(4, FUSION_DIM),
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
            nn.Dropout(0.5), # Increased Dropout for Regularization
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3) # Softplus head for C, D, G
        )

    def forward(self, img, species_idx=None):
        # --- Shared Forward Pass ---
        feats = self.backbone(img) # Shape: (Batch, feat_dim)

        # --- Stage 1 Outputs ---
        sp_logits = self.s1_species(feats)
        ndvi_pred = self.s1_ndvi(feats).squeeze(1)
        h_pred = self.s1_height(feats).squeeze(1)
        month_logits = self.s1_month(feats)
        
        # --- Differentiable Feature Engineering ---
        # These gradients flow back through S1 heads AND Backbone
        ndvi_vec = ndvi_pred 
        h_vec = h_pred 
        ndvi_h_mul = ndvi_vec * h_vec
        
        # Robust Ratio: Ensure denominator is not too small
        # h_vec predicts log1p(height), so it should be >= 0. 
        # We use ReLU to zero out negative preds and add a safety margin.
        h_safe = F.relu(h_vec) + 0.1 
        ndvi_h_ratio = ndvi_vec / h_safe
        
        # Stack Tabular Features
        tab_features = torch.stack([ndvi_vec, h_vec, ndvi_h_mul, ndvi_h_ratio], dim=1)
        
        # --- Stage 2 Input Construction ---
        # Blend Image and Tabular embeddings
        img_emb = self.img_adapter(feats)
        tab_emb = self.tab_adapter(tab_features)
        
        # Weighted Blend
        s2_main = (IMG_FEAT_WEIGHT * img_emb) + (TAB_FEAT_WEIGHT * tab_emb)
        
        # Determine Species for Conditioning
        if species_idx is not None:
            # Training/Val: Use Ground Truth
            sp_emb = self.species_emb(species_idx)
        else:
            # Inference (if GT unknown): Use Prediction
            sp_pred_idx = torch.argmax(sp_logits, dim=1)
            sp_emb = self.species_emb(sp_pred_idx)
            
        # Concatenate: [Combined_Features, Species_Emb]
        s2_input = torch.cat([s2_main, sp_emb], dim=1)
        
        # --- Stage 2 Outputs ---
        log_components = F.softplus(self.s2_mlp(s2_input))
        log_components = torch.clamp(log_components, max=15.0)
        
        # --- Physics Reconstruction ---
        l_c, l_d, l_g = log_components[:, 0:1], log_components[:, 1:2], log_components[:, 2:3]
        r_c, r_d, r_g = torch.expm1(l_c), torch.expm1(l_d), torch.expm1(l_g)
        
        r_tot = r_c + r_d + r_g
        r_gdm = r_c + r_g
        l_tot = torch.log1p(r_tot)
        l_gdm = torch.log1p(r_gdm)
        
        pred_log = torch.cat([l_c, l_d, l_g, l_tot, l_gdm], dim=1)
        pred_real = torch.cat([r_c, r_d, r_g, r_tot, r_gdm], dim=1)

        return sp_logits, ndvi_pred, h_pred, month_logits, pred_log, pred_real

# ====================== UTILS ======================
def format_log(metrics_dict, prefix=""):
    s = []
    for k, v in metrics_dict.items():
        if 'MAE' in k: s.append(f"{k}:{v:.1f}")
        else: s.append(f"{k}:{v:.3f}")
    return f"{prefix} [" + " ".join(s) + "]"

# ====================== TRAIN LOOP ======================
def train_unified(df):
    logger.info("=== UNIFIED SHARED BACKBONE TRAINING ===")
    logger.info(f"Backbone: {BACKBONE_S1} (Shared)")
    logger.info(f"Regularization: Weight Decay 0.05, Dropout 0.5")
    
    # Create a grouping column representing unique site-visits
    # Grouping by Date + State ensures we don't leak site-specific conditions
    df['group_col'] = df['State'].astype(str) + "_" + df['Sampling_Date'].astype(str)
    
    logger.info(f"Using GroupKFold on 'State + Sampling_Date' ({df['group_col'].nunique()} groups) to prevent leakage.")
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    splitter = gkf.split(df, groups=df['group_col'])
        
    species_le = LabelEncoder()
    species_le.fit(df['Species'].fillna('Unknown'))
    
    # Save metadata (No S2 backbone anymore)
    os.makedirs("models_unified", exist_ok=True)
    metadata = {
        'species_encoder': species_le,
        'num_species': len(species_le.classes_),
        'backbone_s1': BACKBONE_S1,
        'backbone_s2': None, # Explicitly None to indicate shared
        'is_shared': True
    }
    torch.save(metadata, os.path.join("models_unified", 'unified_metadata.pth'))
    
    bio_comp_names = ['Clover', 'Dead', 'Green', 'Total', 'GDM']
    
    for fold, (train_idx, val_idx) in enumerate(splitter):
        logger.info(f"\n--- FOLD {fold+1}/{N_FOLDS} ---")
        
        tr_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        
        # Use Weighted Sampler for training
        tr_ds = UnifiedDataset(tr_df, get_image_data_transforms()[0], species_le, use_weights=USE_SAMPLE_WEIGHTS_S1)
        val_ds = UnifiedDataset(val_df, get_image_data_transforms()[1], species_le, use_weights=False)
        
        tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = UnifiedSharedModel(len(species_le.classes_)).to(DEVICE)
        
        # High weight decay for regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scaler = torch.amp.GradScaler("cuda")
        
        ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none',label_smoothing=0.1)
        mse_loss_none = nn.MSELoss(reduction='none')
        
        best_r2 = -float('inf')
        
        for epoch in range(STAGE1_EPOCHS):
            model.train()
            
            # Metrics Accumulators
            s1_run_loss = {'Tot': 0, 'Sp': 0, 'Ndvi': 0, 'H': 0, 'Mon': 0}
            s2_run_loss = {k: 0 for k in bio_comp_names}
            s2_run_loss['Tot'] = 0
            
            pbar = tqdm(tr_loader, desc=f"Ep {epoch+1}", leave=False)
            
            for batch in pbar:
                (img, sp_t, ndvi_t, h_t, mon_t, bio_log_t, bio_real_t, w, mask) = [x.to(DEVICE) for x in batch]
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    sp_p, ndvi_p, h_p, mon_p, bio_log_p, bio_real_p = model(img, sp_t)
                    
                    # --- S1 LOSS ---
                    l_sp = torch.zeros(img.size(0), device=DEVICE)
                    l_ndvi = torch.zeros(img.size(0), device=DEVICE)
                    l_h = torch.zeros(img.size(0), device=DEVICE)
                    
                    if mask[:,0].any(): l_sp = ce_loss(sp_p, sp_t) * mask[:,0].float()
                    if mask[:,1].any(): l_ndvi = mse_loss_none(ndvi_p, ndvi_t) * mask[:,1].float()
                    if mask[:,2].any(): l_h = mse_loss_none(h_p, h_t) * mask[:,2].float()
                    l_mon = ce_loss(mon_p, mon_t)
                    
                    loss_vec_s1 = (0.2 * l_sp) + (0.3 * l_ndvi) + (0.4 * l_h) + (0.1 * l_mon)
                    loss_s1 = (loss_vec_s1 * w).mean()
                    
                    # --- S2 LOSS ---
                    raw_s2 = mse_loss_none(bio_log_p, bio_log_t) 
                    weighted_s2_comps = raw_s2 * COL_WEIGHTS_TENSOR 
                    loss_vec_s2 = weighted_s2_comps.sum(dim=1)
                    loss_s2 = (loss_vec_s2 * w).mean()
                    
                    # --- TOTAL ---
                    total_loss = loss_s1 + loss_s2
                
                scaler.scale(total_loss).backward()
                
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # --- LOGGING ---
                s1_run_loss['Tot'] += loss_s1.item()
                s1_run_loss['Sp'] += (l_sp * w).mean().item()
                s1_run_loss['Ndvi'] += (l_ndvi * w).mean().item()
                s1_run_loss['H'] += (l_h * w).mean().item()
                s1_run_loss['Mon'] += (l_mon * w).mean().item()
                
                s2_run_loss['Tot'] += loss_s2.item()
                batch_comp_loss = (weighted_s2_comps * w.unsqueeze(1)).mean(dim=0).detach()
                for i, name in enumerate(bio_comp_names):
                    s2_run_loss[name] += batch_comp_loss[i].item()
                
                pbar.set_postfix({'L': f"{total_loss.item():.2f}", 'S1': f"{loss_s1.item():.2f}", 'S2': f"{loss_s2.item():.2f}"})

            # End Epoch Stats
            n_batches = len(tr_loader)
            train_s1_log = {k: v/n_batches for k, v in s1_run_loss.items()}
            train_s2_log = {k: v/n_batches for k, v in s2_run_loss.items()}
            
            # --- VALIDATION ---
            model.eval()
            all_true, all_pred = [], []
            val_s1_loss = {'Tot': 0, 'Sp': 0, 'Ndvi': 0, 'H': 0, 'Mon': 0}
            val_s2_mae = {k: 0 for k in bio_comp_names} 
            
            with torch.no_grad():
                for batch in val_loader:
                    (img, sp_t, ndvi_t, h_t, mon_t, bio_log_t, bio_real_t, _, mask) = [x.to(DEVICE) for x in batch]
                    
                    with torch.amp.autocast('cuda'):
                        sp_p, ndvi_p, h_p, mon_p, bio_log_p, bio_real_p = model(img, sp_t)
                        
                        # S1 Val
                        l_sp = F.cross_entropy(sp_p, sp_t, ignore_index=-1) if mask[:,0].any() else 0.0
                        l_ndvi = F.mse_loss(ndvi_p[mask[:,1]], ndvi_t[mask[:,1]]) if mask[:,1].any() else 0.0
                        l_h = F.mse_loss(h_p[mask[:,2]], h_t[mask[:,2]]) if mask[:,2].any() else 0.0
                        l_mon = F.cross_entropy(mon_p, mon_t)
                        
                        val_s1_loss['Tot'] += (0.2*l_sp + 0.3*l_ndvi + 0.4*l_h + 0.1*l_mon).item() if isinstance(l_sp, torch.Tensor) else 0
                        val_s1_loss['Sp'] += l_sp.item() if isinstance(l_sp, torch.Tensor) else l_sp
                        val_s1_loss['Ndvi'] += l_ndvi.item() if isinstance(l_ndvi, torch.Tensor) else l_ndvi
                        val_s1_loss['H'] += l_h.item() if isinstance(l_h, torch.Tensor) else l_h
                        val_s1_loss['Mon'] += l_mon.item() if isinstance(l_mon, torch.Tensor) else l_mon
                        
                        # S2 Val (MAE)
                        abs_diff = torch.abs(bio_real_p - bio_real_t)
                        mae_batch = abs_diff.mean(dim=0)
                        for i, name in enumerate(bio_comp_names):
                            val_s2_mae[name] += mae_batch[i].item()
                        
                        all_true.append(bio_real_t.cpu().numpy())
                        all_pred.append(bio_real_p.cpu().numpy())
            
            n_val = len(val_loader)
            val_s1_log = {k: v/n_val for k, v in val_s1_loss.items()}
            val_s2_mae_log = {k: v/n_val for k, v in val_s2_mae.items()}
            
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            r2 = calculate_global_weighted_r2(y_true, y_pred, OFFICIAL_WEIGHTS)
            
            # --- DETAILED LOGS ---
            logger.info(f"F{fold+1} E{epoch+1} [S1] | " + 
                        format_log(train_s1_log, "Train") + " | " + 
                        format_log(val_s1_log, "Val"))
            
            logger.info(f"F{fold+1} E{epoch+1} [S2] | " + 
                        format_log(train_s2_log, "Train Loss") + " | " + 
                        format_log(val_s2_mae_log, "Val MAE(g)"))
            
            logger.info(f"Global R2: {r2:.5f}")
            
            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), f"models_unified/fold{fold+1}.pth")
                logger.info(f">> Saved Best Model (R2: {best_r2:.5f})")

if __name__ == '__main__':
    setup_logging(logger_name="System Logger", log_dir='logs', file_name_part='Unified_Shared')
    os.makedirs("models_unified", exist_ok=True)
    df = load_data(logger)
    train_unified(df)