import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import os
from tqdm import tqdm

from configs import *
from common import load_data, get_transforms, setup_logging, set_seed,calculate_weighted_average_r2
from configs import BACKBONE, DROPOUT, EPOCHS, LOSS_WEIGHTS, NUM_WORKERS, SEED, STRATIFY_COL

# ====================== DATASET ======================
class BiomassDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Targets: Clover, Dead, Green, Total, GDM
        self.targets = self.df[TARGET_COLS].values.astype(np.float32)
        # We predict in Log(1+x) space for stability
        self.log_targets = np.log1p(self.targets)
        
        # Aux targets (NDVI, Height) to help backbone learn scale
        self.aux = self.df[['aux_ndvi', 'aux_height']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"train/{row['image_path'].split('/')[-1]}"
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            img = self.transform(img)
            
        return (
            img, 
            torch.tensor(self.log_targets[idx]), 
            torch.tensor(self.aux[idx])
        )

# ====================== UNIFIED PHYSICS MODEL ======================
class PhysicsInformedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        self.dropout = nn.Dropout(DROPOUT)
        
        # 1. Component Head: Predicts Clover, Dead, Green
        # We output 3 values. We will enforce physics on these.
        self.component_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 3) 
        )
        
        # 2. Auxiliary Head: Predicts NDVI and Log(Height)
        # This forces the backbone to learn "Greenness" and "Structure"
        self.aux_head = nn.Linear(in_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        
        # --- Physics Layer ---
        # 1. Predict Raw Logs for Components
        # Use Softplus to ensure outputs are non-negative (mass cannot be negative)
        # We treat the raw output as log-space, but let's ensure stability.
        # Simple Linear output is fine if we train on log targets, but softplus is safer.
        raw_components = self.component_head(features)
        
        # Force positivity on the log predictions (optional, but helps convergence)
        # log_c, log_d, log_g = raw_components[:, 0], raw_components[:, 1], raw_components[:, 2]
        # Let's assume raw outputs are log(1+x). 
        # Ideally, we want to allow 0.
        log_components = F.softplus(raw_components) 
        
        # 2. Convert to Real Space to apply Physics
        # real = exp(log) - 1
        real_components = torch.expm1(log_components)
        
        # Extract individual mass
        real_c = real_components[:, 0:1] # Clover
        real_d = real_components[:, 1:2] # Dead
        real_g = real_components[:, 2:3] # Green
        
        # 3. Apply Constraints (The "Perfect Physics")
        real_total = real_c + real_d + real_g
        real_gdm   = real_c + real_g
        
        # 4. Convert Derived values back to Log Space for Loss Calculation
        # (Calculating loss in Log space is more stable for biomass)
        log_total = torch.log1p(real_total)
        log_gdm   = torch.log1p(real_gdm)
        
        # Stack all 5 predictions: [C, D, G, Total, GDM]
        final_log_preds = torch.cat([log_components, log_total, log_gdm], dim=1)
        
        # Aux predictions
        aux_preds = self.aux_head(features)
        
        return final_log_preds, aux_preds

# ====================== METRICS & LOSS ======================
def weighted_mse_loss(input, target, weights):
    # input: (B, 5), target: (B, 5), weights: (5,)
    # Element-wise MSE
    mse = (input - target) ** 2
    # Apply weights to columns
    weighted_mse = mse * weights
    # Mean over batch
    return weighted_mse.mean()

def calculate_r2(y_true, y_pred, weights):
    # Flatten everything
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Repeat weights for flattened array
    n_samples = len(y_true) // 5
    w_flat = np.repeat([weights], n_samples, axis=0).flatten() # This might be wrong shape logic
    # Correct weighting for R2:
    w_flat = np.tile(weights, n_samples)
    
    # Weighted R2
    ss_res = np.sum(w_flat * (y_true - y_pred)**2)
    weighted_mean = np.average(y_true, weights=w_flat)
    ss_tot = np.sum(w_flat * (y_true - weighted_mean)**2)
    
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

# ====================== TRAINING LOOP ======================
def train_kfolds():
    logger = setup_logging()
    set_seed(SEED)
    
    df = load_data(logger)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(df, df[STRATIFY_COL]))
    
    train_tf, val_tf = get_transforms()
    
    # Move weights to device for Loss, keep as list for Metric
    loss_weights_tensor = torch.tensor(LOSS_WEIGHTS).to(DEVICE)
    
    oof_preds = []
    oof_targets = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
        
        train_ds = BiomassDataset(df.iloc[train_idx], train_tf)
        val_ds = BiomassDataset(df.iloc[val_idx], val_tf)
        
        # Drop_last=True is important for BatchNorm stability
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True)
        
        model = PhysicsInformedModel(BACKBONE).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler()
        
        best_r2 = -float('inf') # Track best R2, not just best Loss
        best_model_path = f"pinn_models/fold_{fold+1}_best.pth"
        os.makedirs("pinn_models", exist_ok=True)
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss_meter = 0
            train_preds_real = []
            train_targets_real = []
            
            pbar = tqdm(train_loader, leave=False, desc=f"Fold {fold+1} Ep {epoch+1}")
            for imgs, targets, aux in pbar:
                imgs, targets, aux = imgs.to(DEVICE), targets.to(DEVICE), aux.to(DEVICE)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    preds, aux_preds = model(imgs)
                    
                    # 1. Main Loss: Weighted MSE on Log Scale
                    #    This aligns with the objective: minimize squared error on important columns
                    mse = (preds - targets) ** 2
                    l_main = (mse * loss_weights_tensor).sum(dim=1).mean()
                    
                    # 2. Aux Loss
                    l_aux = F.mse_loss(aux_preds, aux)
                    
                    loss = l_main + 0.1 * l_aux # Reduced aux weight slightly
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_meter += loss.item()
                pbar.set_postfix({'L': f"{loss.item():.4f}"})
                
                # Collect training predictions for R² calculation
                with torch.no_grad():
                    pred_real = torch.expm1(preds).cpu().numpy()
                    target_real = torch.expm1(targets).cpu().numpy()
                    train_preds_real.append(pred_real)
                    train_targets_real.append(target_real)
            
            scheduler.step()
            
            # Calculate Training R²
            train_preds_concat = np.concatenate(train_preds_real)
            train_targets_concat = np.concatenate(train_targets_real)
            train_r2, train_component_r2s = calculate_weighted_average_r2(train_targets_concat, train_preds_concat, LOSS_WEIGHTS)
            
            # --- VALIDATION ---
            model.eval()
            val_preds_real = []
            val_targets_real = []
            val_loss_meter = 0
            
            with torch.no_grad():
                for imgs, targets, aux in val_loader:
                    imgs, targets, aux = imgs.to(DEVICE), targets.to(DEVICE), aux.to(DEVICE)
                    
                    with torch.amp.autocast('cuda'):
                        log_preds, aux_preds = model(imgs)
                        
                        # Calculate validation loss (same as training loss)
                        mse = (log_preds - targets) ** 2
                        l_main = (mse * loss_weights_tensor).sum(dim=1).mean()
                        l_aux = F.mse_loss(aux_preds, aux)
                        val_loss = l_main + 0.1 * l_aux
                        val_loss_meter += val_loss.item()
                    
                    # Convert Log(1+x) -> Real Mass for correct R2 calculation
                    # Physics constraints were applied inside the model in Real space,
                    # then converted to Log. Now we convert back.
                    pred_real = torch.expm1(log_preds).cpu().numpy()
                    target_real = torch.expm1(targets).cpu().numpy() # targets were loaded as log
                    
                    val_preds_real.append(pred_real)
                    val_targets_real.append(target_real)
            
            # Concatenate all batches
            vp = np.concatenate(val_preds_real)
            vt = np.concatenate(val_targets_real)
            
            # Calculate Validation Weighted Average R²
            val_r2, val_component_r2s = calculate_weighted_average_r2(vt, vp, LOSS_WEIGHTS)
            
            # Log detailed stats - Side by side comparison
            train_loss_avg = train_loss_meter / len(train_loader)
            val_loss_avg = val_loss_meter / len(val_loader)
            
            train_r2_str = f"Train R²: {train_r2:.4f} [C:{train_component_r2s[0]:.2f} D:{train_component_r2s[1]:.2f} G:{train_component_r2s[2]:.2f} Tot:{train_component_r2s[3]:.2f} GDM:{train_component_r2s[4]:.2f}]"
            val_r2_str = f"Val R²: {val_r2:.4f} [C:{val_component_r2s[0]:.2f} D:{val_component_r2s[1]:.2f} G:{val_component_r2s[2]:.2f} Tot:{val_component_r2s[3]:.2f} GDM:{val_component_r2s[4]:.2f}]"
            
            logger.info(f"Ep {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
            logger.info(f"       | {train_r2_str}")
            logger.info(f"       | {val_r2_str}")
            
            # Save Best Model based on Validation Weighted R²
            if val_r2 > best_r2:
                best_r2 = val_r2
                logger.info(f"New Best R2: {best_r2:.5f}")
                torch.save(model.state_dict(), best_model_path)
        
        # End of Fold
        logger.info(f"Fold {fold+1} Best Weighted R2: {best_r2:.5f}")
        fold_scores.append(best_r2)

        # Load best to store OOF
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            fold_preds = []
            fold_targets = []
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(DEVICE)
                p, _ = model(imgs)
                fold_preds.append(torch.expm1(p).cpu().numpy())
                fold_targets.append(torch.expm1(targets).cpu().numpy())
            
            oof_preds.append(np.concatenate(fold_preds))
            oof_targets.append(np.concatenate(fold_targets))

    # ====================== FINAL RESULTS ======================
    oof_preds = np.concatenate(oof_preds)
    oof_targets = np.concatenate(oof_targets)
    
    final_r2, final_comps = calculate_weighted_average_r2(oof_targets, oof_preds, LOSS_WEIGHTS)
    
    logger.info(f"\n==========================================")
    logger.info(f"CV TRAINING COMPLETE")
    logger.info(f"Average Fold R2: {np.mean(fold_scores):.5f}")
    logger.info(f"OOF Global Weighted R2: {final_r2:.5f}")
    logger.info(f"Component Breakdown:")
    names = ['Clover', 'Dead', 'Green', 'Total', 'GDM']
    for n, s in zip(names, final_comps):
        logger.info(f"  {n}: {s:.5f}")
    logger.info(f"==========================================")
    
    # Save OOF for stacking/analysis
    np.save('oof_preds.npy', oof_preds)
    np.save('oof_targets.npy', oof_targets)

if __name__ == "__main__":
    train_kfolds()