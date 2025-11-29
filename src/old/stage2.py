# stage2.py
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

# Assuming these exist in your common.py
from common import BATCH_SIZE, DEVICE, IMAGE_SIZE, LEARNING_RATE, NUM_EPOCHS, SeasonalCurriculumSampler, calculate_sample_weights, calculate_sample_weights_mean, get_image_data_transforms, get_season, print_stratification_stats, set_seed, setup_logging, calculate_count_frequency_features

# -------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------------------

def prepare_data(df_train, logger=None):
    """
    Pivots the long-format DataFrame to a wide format for joint training 
    and engineers date-based features.
    """
    if logger:
        logger.info("Starting data preparation...")
        logger.debug(f"Input data shape: {df_train.shape}")
    
    wide_df = df_train.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    wide_df['Sampling_Date'] = pd.to_datetime(wide_df['Sampling_Date'])
    wide_df['month'] = wide_df['Sampling_Date'].dt.month
    period = 12
    wide_df['month_sin'] = np.sin(2 * np.pi * wide_df['month'] / period)
    wide_df['month_cos'] = np.cos(2 * np.pi * wide_df['month'] / period)
    
    wide_df['season'] = wide_df['month'].apply(get_season)
    wide_df = wide_df.drop('Sampling_Date', axis=1)
    
    # Log transformation for Feature (Height)
    wide_df['Height_Ave_cm'] = np.log1p(wide_df['Height_Ave_cm'])
    
    # Feature interactions
    wide_df['NDVI_Height_MUL'] = wide_df['Pre_GSHH_NDVI'] * wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_ADD'] = wide_df['Pre_GSHH_NDVI'] + wide_df['Height_Ave_cm']
    ratio = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5) 
    wide_df['NDVI_Height_Ratio'] = ratio

    if logger:
        logger.info(f"Data preparation complete. Output shape: {wide_df.shape}")
    
    return wide_df


def conditional_target_impute(df_to_impute, train_df_for_fit=None, logger=None):
    """
    Imputes NaN target values using medians (performed on Real values).
    """
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    grouping_cols = ['Species', 'State', 'season'] 

    if logger:
        nan_counts_before = df_to_impute[target_cols].isnull().sum()
        if nan_counts_before.sum() > 0:
            logger.info(f"Imputing missing values. NaN counts before: {nan_counts_before.to_dict()}")

    if train_df_for_fit is not None:
        median_map = train_df_for_fit.groupby(grouping_cols)[target_cols].median()
        
        for col in target_cols:
            df_to_impute[col] = df_to_impute.apply(
                lambda row: median_map.loc[(row['Species'], row['State'], row['season']), col]
                if pd.isnull(row[col]) and (row['Species'], row['State'], row['season']) in median_map.index
                else row[col],
                axis=1
            )
        global_medians = train_df_for_fit[target_cols].median()
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(global_medians)
    else:
        df_to_impute[target_cols] = df_to_impute.groupby(grouping_cols)[target_cols].transform(
            lambda x: x.fillna(x.median())
        )
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(df_to_impute[target_cols].median())

    if logger:
        nan_counts_after = df_to_impute[target_cols].isnull().sum()
        if nan_counts_after.sum() > 0:
            logger.warning(f"NaN counts after imputation: {nan_counts_after.to_dict()}")
        else:
            logger.info("Target imputation complete.")

    return df_to_impute



def enforce_physical_constraints(predictions_real_scale):
    """
    OPTION B: Post-processing to enforce strict mass balance.
    Input: Numpy array of predictions in REAL GRAMS (not log).
    Order: [Clover, Dead, Green, Total, GDM]
    """
    # 1. Enforce Non-Negativity (Safety net)
    preds = np.maximum(predictions_real_scale, 0)
    
    # 2. Extract components
    clover = preds[:, 0]
    dead = preds[:, 1]
    green = preds[:, 2]
    
    # 3. Recalculate Aggregates based on components
    new_gdm = clover + green
    new_total = clover + dead + green
    
    # 4. Update the prediction array
    preds[:, 3] = new_total  # Total
    preds[:, 4] = new_gdm    # GDM
    
    return preds

# -------------------------------------------------------------------
# 2. DATASET & LOSS
# -------------------------------------------------------------------

class Stage2Dataset(Dataset):
    def __init__(self, df, tabular_features, target_cols, image_dir='train', transform=None, weight_col='sample_weight'):
        self.df = df
        self.image_dir = image_dir
        self.tabular_features = tabular_features
        self.target_cols = target_cols
        self.transform = transform
        self.weight_col = weight_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        img_path = '/'.join(row['image_path'].split('/')[1:])
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        tabular_data = torch.tensor(row[self.tabular_features].values.astype(np.float32))
        
        # Targets are already log-transformed in main block
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        sample_weight = torch.tensor(row[self.weight_col], dtype=torch.float32)
        
        return image, tabular_data, targets, sample_weight


class WeightedMassBalanceLoss(nn.Module):
    """
    Calculates MSE on Log-Space targets but enforces Mass Balance on Real-Space conversions.
    """
    def __init__(self, target_weights=None, mass_balance_alpha=0.8):
        super().__init__()
        self.clover_idx, self.dead_idx, self.green_idx, self.total_idx, self.gdm_idx = 0, 1, 2, 3, 4
        
        if target_weights is None:
            target_weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32).view(1, -1))
        self.alpha = mass_balance_alpha

    def forward(self, log_predictions, log_targets, sample_weights=None):
        # 1. Standard MSE on the LOG scale (Stabilizes training)
        squared_error = F.mse_loss(log_predictions, log_targets, reduction='none')
        weighted_error_targets = squared_error * self.target_weights
        per_sample_weighted_mse = weighted_error_targets.sum(dim=1) / self.target_weights.sum()

        # 2. Mass Balance Penalty (Convert log -> real to check A+B=C)
        # clamp max to prevent overflow during early training chaos
        pred_real = torch.expm1(log_predictions.clamp(max=15))
        
        pred_clover = pred_real[:, self.clover_idx]
        pred_dead = pred_real[:, self.dead_idx]
        pred_green = pred_real[:, self.green_idx]
        pred_total = pred_real[:, self.total_idx]
        pred_gdm = pred_real[:, self.gdm_idx]
        
        # Check Total Balance: Clover + Dead + Green = Total
        sum_total = pred_clover + pred_dead + pred_green
        # Log the error back down so it doesn't dominate the gradient
        total_balance_error = torch.log1p(F.l1_loss(pred_total, sum_total, reduction='none'))
        
        # Check GDM Balance: Clover + Green = GDM
        sum_gdm = pred_clover + pred_green
        gdm_balance_error = torch.log1p(F.l1_loss(pred_gdm, sum_gdm, reduction='none'))

        per_sample_loss = per_sample_weighted_mse + self.alpha * (total_balance_error + gdm_balance_error)
        
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights

        total_loss = per_sample_loss.mean()
        
        return total_loss


class MultiModalModel(nn.Module):
    def __init__(self, timm_model_name, tabular_feature_size, output_size=5, stage_index=3):
        super().__init__()
        assert stage_index in [0, 1, 2, 3], \
            f"Invalid stage_index={stage_index}. Swin models have stages [0,1,2,3]."
        self.stage_index = stage_index
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(stage_index,)   
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.image_feature_size = self.backbone.feature_info[stage_index]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        combined_input_size = self.image_feature_size + tabular_feature_size
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, image, tabular_data):
        img_features = self.backbone(image)[0]
        img_features = img_features.permute(0, 3, 1, 2)
        img_features = self.pool(img_features).squeeze(-1).squeeze(-1)
        combined = torch.cat([img_features, tabular_data], dim=1)
        
        raw_out = self.mlp(combined)
        # SOFTPLUS for Non-Negativity
        return F.softplus(raw_out)


if __name__ == '__main__':
    for stage_idx in [0, 1, 2, 3]:
        logger = setup_logging(file_name_part=f"stage2_training_PS_swin_stage{stage_idx}")
        
        logger.info("="*80)
        logger.info("STAGE 2: LOG-TRANSFORMED + PHYSICAL CONSTRAINTS")
        logger.info("="*80)
        logger.info(f"Image Size: {IMAGE_SIZE}")
        logger.info(f"Batch Size: {BATCH_SIZE}")
        logger.info(f"Number of Epochs: {NUM_EPOCHS}")
        logger.info(f"Learning Rate: {LEARNING_RATE}")
        logger.info(f"Device: {DEVICE}")
        logger.info("="*80)
        
        set_seed(logger=logger)
        
        # Load and Prepare Data
        logger.info("Loading training data from 'train.csv'...")
        df_train = pd.read_csv('./train.csv')
        logger.info(f"Loaded {len(df_train)} rows")
        
        df_wide = prepare_data(df_train, logger=logger)
        
        target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        
        X = df_wide.drop(columns=target_cols)
        y = df_wide[target_cols]
        
        logger.info("Performing stratified train-validation split (80/20)...")
        strat_col = 'season'
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=X[strat_col]
        )

        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)

        print_stratification_stats(df_wide, train_df, val_df, strat_col, logger=logger)

        # Perform Imputation (on REAL values)
        logger.info("Imputing missing target values...")
        train_df_imputed = conditional_target_impute(train_df, train_df_for_fit=None, logger=logger)
        val_df_imputed = conditional_target_impute(val_df, train_df_for_fit=train_df_imputed, logger=logger)

        # ---------------------------------------------------------
        # CRITICAL: APPLY LOG TRANSFORM TO TARGETS HERE
        # ---------------------------------------------------------
        logger.info("Applying np.log1p() to targets for training...")
        for col in target_cols:
            train_df_imputed[col] = np.log1p(train_df_imputed[col])
            val_df_imputed[col] = np.log1p(val_df_imputed[col])
        # ---------------------------------------------------------

        # Calculate sample weights
        prop_col = 'season'
        train_df_imputed, weight_col = calculate_sample_weights_mean(
            train_df_imputed, 
            group_col=prop_col, 
            weight_col='sample_weight', 
            logger=logger
        )
        
        val_df_imputed[weight_col] = 1.0 
        train_df = train_df_imputed
        val_df = val_df_imputed
        
        # ============================================================================
        # REFACTORED: ADD SPECIES COUNT FEATURES (GLOBAL AND SEASONAL)
        # ============================================================================
        train_df, val_df, count_freq_features = calculate_count_frequency_features(
            train_df=train_df,
            val_df=val_df,
            group_col='Species',
            local_group_col='season',
            logger=logger
        )
        
        base_numerical_features = [
            'Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos',
            'NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio'
        ]
        numerical_features = base_numerical_features + count_freq_features
        categorical_features = ['State', 'Species', 'season'] 

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ], remainder='passthrough'
        )

        logger.info("Fitting preprocessor on training data...")
        train_processed = preprocessor.fit_transform(train_df)
        joblib.dump(preprocessor, 'stage2_preprocessor.pkl')

        ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        tabular_feature_names = numerical_features + ohe_feature_names
        
        non_processed_cols = ['sample_id', 'image_path'] + target_cols + [weight_col] + ['season']
        num_processed_cols = len(numerical_features) + len(ohe_feature_names)
        
        train_df_processed = pd.DataFrame(train_processed[:, :num_processed_cols], columns=tabular_feature_names, index=train_df.index)
        train_df_processed = pd.concat([train_df_processed, train_df[non_processed_cols]], axis=1)

        val_processed = preprocessor.transform(val_df)
        val_df_processed = pd.DataFrame(val_processed[:, :num_processed_cols], columns=tabular_feature_names, index=val_df.index)
        val_df_processed = pd.concat([val_df_processed, val_df[non_processed_cols]], axis=1)

        train_transform, val_transform = get_image_data_transforms()
        
        # Dataset & Loader
        logger.info("Creating PyTorch datasets...")
        train_dataset = Stage2Dataset(train_df_processed, tabular_feature_names, target_cols, transform=train_transform, weight_col=weight_col)
        val_dataset = Stage2Dataset(val_df_processed, tabular_feature_names, target_cols, transform=val_transform, weight_col=weight_col)
        
        train_sampler = SeasonalCurriculumSampler(data_df=train_df_processed, shuffle_within_season=False, seed=42)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Model
        tabular_feature_size = len(tabular_feature_names)
        logger.info(f"Initializing MultiModalModel (Stage {stage_idx})...")
        model = MultiModalModel(
            timm_model_name='swin_base_patch4_window7_224',
            tabular_feature_size=tabular_feature_size,
            stage_index=stage_idx
        ).to(DEVICE)
        
        custom_target_weights = [1.0, 1.0, 1.0, 5.0, 2.0]
        criterion = WeightedMassBalanceLoss(target_weights=custom_target_weights, mass_balance_alpha=0.5).to(DEVICE) 

        optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LEARNING_RATE) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        
        best_val_r2 = -float('inf')

        logger.info("STARTING TRAINING - STAGE 2 (LOG DOMAIN)")

        for epoch in range(NUM_EPOCHS):
            # ==================== TRAINING PHASE ====================
            model.train()
            running_loss = 0.0
            train_preds_log = []
            train_targets_log = []
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", leave=False)
            
            for batch_idx, (images, tabular_data, targets, sample_weights) in enumerate(train_bar):
                images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
                sample_weights = sample_weights.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images, tabular_data) # Outputs are Log space (via softplus trained on log)
                
                loss = criterion(outputs, targets, sample_weights=sample_weights) 
                
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_loader))
                
                running_loss += loss.item() * images.size(0)
                
                train_preds_log.append(outputs.detach().cpu().numpy())
                train_targets_log.append(targets.cpu().numpy())
                
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = running_loss / len(train_loader.dataset)
            
            # Concatenate and CONVERT TO REAL for metrics
            train_preds_log = np.concatenate(train_preds_log, axis=0)
            train_targets_log = np.concatenate(train_targets_log, axis=0)
            train_preds_real = np.expm1(train_preds_log)
            train_targets_real = np.expm1(train_targets_log)

            # ==================== VALIDATION PHASE ====================
            model.eval()
            val_loss = 0.0
            val_preds_log = []
            val_targets_log = []
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
                for images, tabular_data, targets, sample_weights in val_bar:
                    images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
                    sample_weights = sample_weights.to(DEVICE)
                    
                    outputs = model(images, tabular_data)
                    loss = criterion(outputs, targets, sample_weights=sample_weights)
                    val_loss += loss.item() * images.size(0)
                    
                    val_preds_log.append(outputs.cpu().numpy())
                    val_targets_log.append(targets.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader.dataset)
            
            val_preds_log = np.concatenate(val_preds_log, axis=0)
            val_targets_log = np.concatenate(val_targets_log, axis=0)
            
            # Convert to Real
            val_preds_real = np.expm1(val_preds_log)
            val_targets_real = np.expm1(val_targets_log)

            # APPLY OPTION B: Enforce constraints on validation predictions
            val_preds_phys = enforce_physical_constraints(val_preds_real)
            
            # ==================== CALCULATE METRICS (ON REAL VALUES) ====================
            
            official_r2_weights = [0.1, 0.1, 0.1, 0.5, 0.2]
            
            def calculate_weighted_r2(y_true, y_pred, weights):
                y_true_flat = y_true.flatten()
                y_pred_flat = y_pred.flatten()
                num_samples = y_true.shape[0]
                num_targets = y_true.shape[1]
                weights_matrix = np.tile(np.array(weights).reshape(1, num_targets), (num_samples, 1))
                return r2_score(y_true_flat, y_pred_flat, sample_weight=weights_matrix.flatten())
            
            def calculate_per_target_metrics(y_true, y_pred, target_names):
                metrics = {}
                for i, name in enumerate(target_names):
                    true_vals, pred_vals = y_true[:, i], y_pred[:, i]
                    mask = true_vals != 0
                    mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100 if mask.sum() > 0 else 0.0
                    metrics[name] = {
                        'R²': r2_score(true_vals, pred_vals),
                        'MAE': np.mean(np.abs(true_vals - pred_vals)),
                        'RMSE': np.sqrt(np.mean((true_vals - pred_vals) ** 2)),
                        'MAPE': mape
                    }
                return metrics
            
            def calculate_mass_balance_metrics(y_pred):
                pred_clover, pred_dead, pred_green = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
                pred_total, pred_gdm = y_pred[:, 3], y_pred[:, 4]
                
                total_sum = pred_clover + pred_dead + pred_green
                gdm_sum = pred_clover + pred_green
                
                return {
                    'total_mae': np.mean(np.abs(pred_total - total_sum)),
                    'gdm_mae': np.mean(np.abs(pred_gdm - gdm_sum))
                }
            
            # Training Metrics (Raw Real Values)
            train_official_r2 = calculate_weighted_r2(train_targets_real, train_preds_real, official_r2_weights)
            train_per_target = calculate_per_target_metrics(train_targets_real, train_preds_real, target_cols)
            train_mass_balance = calculate_mass_balance_metrics(train_preds_real)
            
            # Validation Metrics (Using Physically Constrained Values)
            val_official_r2 = calculate_weighted_r2(val_targets_real, val_preds_phys, official_r2_weights)
            val_per_target = calculate_per_target_metrics(val_targets_real, val_preds_phys, target_cols)
            val_mass_balance = calculate_mass_balance_metrics(val_preds_phys)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # ==================== LOGGING ====================
            logger.info("="*80)
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
            logger.info("-"*80)
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Official Weighted R² (Train): {train_official_r2:.4f}")
            logger.info(f"Official Weighted R² (Val):   {val_official_r2:.4f}")
            logger.info("-"*80)
            
            logger.info("VALIDATION SET (Physically Constrained):")
            for target_name in target_cols:
                m = val_per_target[target_name]
                logger.info(f"  {target_name:12s} | R²: {m['R²']:>7.4f} | MAE: {m['MAE']:>7.3f} | MAPE: {m['MAPE']:>6.2f}%")
            
            logger.info("Mass Balance Error (MAE):")
            logger.info(f"  Total Balance: {val_mass_balance['total_mae']:.4f} (Should be 0.0000)")
            logger.info(f"  GDM Balance:   {val_mass_balance['gdm_mae']:.4f} (Should be 0.0000)")
            logger.info("="*80)

            if val_official_r2 > best_val_r2:
                best_val_r2 = val_official_r2
                torch.save(model.state_dict(), 'stage2_model_weighted_phys.pth')
                logger.info(f"✓ NEW BEST MODEL! R²: {best_val_r2:.4f}")

        logger.info("STAGE 2 COMPLETE")