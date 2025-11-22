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

from common import get_season, print_stratification_stats, set_seed, setup_logging


def prepare_data(df_train, logger=None):
    """
    Pivots the long-format DataFrame to a wide format for joint training 
    and engineers date-based features (month, season, interactions).
    """
    if logger:
        logger.info("Starting data preparation...")
        logger.debug(f"Input data shape: {df_train.shape}")
    
    # Pivot to wide format (one row per sample_id, 5 target columns)
    wide_df = df_train.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    # Engineer Date Features
    wide_df['Sampling_Date'] = pd.to_datetime(wide_df['Sampling_Date'])
    wide_df['month'] = wide_df['Sampling_Date'].dt.month
    period = 12
    wide_df['month_sin'] = np.sin(2 * np.pi * wide_df['month'] / period)
    wide_df['month_cos'] = np.cos(2 * np.pi * wide_df['month'] / period)
    
    wide_df['season'] = wide_df['month'].apply(get_season)
    
    wide_df = wide_df.drop('Sampling_Date', axis=1)
    
    # Log transformation
    wide_df['Height_Ave_cm'] = np.log1p(wide_df['Height_Ave_cm'])
    
    # Feature interactions
    wide_df['NDVI_Height_MUL'] = wide_df['Pre_GSHH_NDVI'] * wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_ADD'] = wide_df['Pre_GSHH_NDVI'] + wide_df['Height_Ave_cm']
    ratio = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5) 
    wide_df['NDVI_Height_Ratio'] = ratio

    if logger:
        logger.info(f"Data preparation complete. Output shape: {wide_df.shape}")
        logger.debug(f"Engineered features: month_sin, month_cos, season, NDVI_Height interactions")
    
    return wide_df


def conditional_target_impute(df_to_impute, train_df_for_fit=None, logger=None):
    """
    Imputes NaN target values using medians. 
    If train_df_for_fit is provided (i.e., for validation/test sets), 
    it uses medians calculated from the training data.
    """
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    grouping_cols = ['Species', 'State', 'season'] 

    if logger:
        nan_counts_before = df_to_impute[target_cols].isnull().sum()
        if nan_counts_before.sum() > 0:
            logger.info(f"Imputing missing values. NaN counts before: {nan_counts_before.to_dict()}")

    if train_df_for_fit is not None:
        # Use Medians from Training Data
        median_map = train_df_for_fit.groupby(grouping_cols)[target_cols].median()
        
        for col in target_cols:
            df_to_impute[col] = df_to_impute.apply(
                lambda row: median_map.loc[(row['Species'], row['State'], row['season']), col]
                if pd.isnull(row[col]) and (row['Species'], row['State'], row['season']) in median_map.index
                else row[col],
                axis=1
            )
        
        # Use Global Median from Training Data for remaining NaNs
        global_medians = train_df_for_fit[target_cols].median()
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(global_medians)
    else:
        # Calculate and apply group median (for training set)
        df_to_impute[target_cols] = df_to_impute.groupby(grouping_cols)[target_cols].transform(
            lambda x: x.fillna(x.median())
        )
        # Apply global median (for remaining NaNs in training set)
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(df_to_impute[target_cols].median())

    if logger:
        nan_counts_after = df_to_impute[target_cols].isnull().sum()
        if nan_counts_after.sum() > 0:
            logger.warning(f"NaN counts after imputation: {nan_counts_after.to_dict()}")
        else:
            logger.info("Target imputation complete. No missing values remain.")

    return df_to_impute


def calculate_sample_weights(df, proportions, prop_col, weight_col='sample_weight', logger=None):
    """
    Calculates inverse-frequency sample weights based on prop_col proportions distribution.
    The weights are normalized so the mean weight is 1.0.
    """
    # Calculate inverse proportions and normalize
    inverse_proportions = 1 / proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse
    
    # Create the weight map
    weight_map = normalized_weights.to_dict()

    # Apply the weight to the DataFrame
    df[weight_col] = df[prop_col].map(weight_map)
    
    if logger:
        logger.info(f"Sample weights calculated based on '{prop_col}'")
        logger.debug(f"Weight distribution: {weight_map}")
    
    return df, weight_col


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
    
        # Image Loading and Transformation
        img_path = '/'.join(row['image_path'].split('/')[1:])
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tabular Data
        tabular_data = torch.tensor(row[self.tabular_features].values.astype(np.float32))

        # Target Variables
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        # Sample Weight
        sample_weight = torch.tensor(row[self.weight_col], dtype=torch.float32)
        
        return image, tabular_data, targets, sample_weight


class WeightedMassBalanceLoss(nn.Module):
    def __init__(self, target_weights=None, mass_balance_alpha=0.8):
        super().__init__()
        self.clover_idx, self.dead_idx, self.green_idx, self.total_idx, self.gdm_idx = 0, 1, 2, 3, 4
        
        if target_weights is None:
            target_weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32).view(1, -1))
        self.alpha = mass_balance_alpha

    def forward(self, predictions, targets, sample_weights=None):
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        weighted_error_targets = squared_error * self.target_weights
        per_sample_weighted_mse = weighted_error_targets.sum(dim=1) / self.target_weights.sum()

        pred_clover = predictions[:, self.clover_idx]
        pred_dead = predictions[:, self.dead_idx]
        pred_green = predictions[:, self.green_idx]
        pred_total = predictions[:, self.total_idx]
        pred_gdm = predictions[:, self.gdm_idx]
        
        predicted_total_sum = pred_clover + pred_dead + pred_green
        total_balance_penalty = F.mse_loss(pred_total, predicted_total_sum, reduction='none')
        
        predicted_gdm_sum = pred_clover + pred_green
        gdm_balance_penalty = F.mse_loss(pred_gdm, predicted_gdm_sum, reduction='none')

        per_sample_loss = per_sample_weighted_mse + self.alpha * (total_balance_penalty + gdm_balance_penalty)
        
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
        return self.mlp(combined)


if __name__ == '__main__':

    logger = setup_logging(file_name_part="stage2_training")
    
    # Define parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*80)
    logger.info("STAGE 2: TRAINING CONFIGURATION")
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
    
    # Separate features and targets
    X = df_wide.drop(columns=target_cols)
    y = df_wide[target_cols]
    
    # Stratified Split
    logger.info("Performing stratified train-validation split (80/20)...")
    strat_col = 'season'
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=X[strat_col]
    )

    # Re-combine for preprocessing
    train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)

    print_stratification_stats(df_wide, train_df, val_df, strat_col, logger=logger)

    # Perform Imputation
    logger.info("Imputing missing target values...")
    train_df_imputed = conditional_target_impute(train_df, train_df_for_fit=None, logger=logger)
    val_df_imputed = conditional_target_impute(val_df, train_df_for_fit=train_df_imputed, logger=logger)

    # Calculate sample weights
    prop_col = 'season'
    proportions = train_df_imputed[prop_col].value_counts(normalize=True)
    train_df_imputed, weight_col = calculate_sample_weights(
        train_df_imputed, proportions, prop_col, weight_col='sample_weight', logger=logger
    )
    
    val_df_imputed[weight_col] = 1.0 

    train_df = train_df_imputed
    val_df = val_df_imputed
    
    # Add Species Count Features (calculated from training set only)
    logger.info("Adding species count features...")
    species_counts = train_df['Species'].value_counts()
    species_freq = species_counts / len(train_df)
    
    # Log transform counts for better scaling
    species_counts_log = np.log1p(species_counts)
    
    # Add to training dataframe
    train_df['species_count'] = train_df['Species'].map(species_counts_log)
    train_df['species_frequency'] = train_df['Species'].map(species_freq)
    
    # Add to validation dataframe (using training statistics)
    val_df['species_count'] = val_df['Species'].map(species_counts_log).fillna(0)
    val_df['species_frequency'] = val_df['Species'].map(species_freq).fillna(species_freq.mean())
    
    logger.info(f"Species count range: {train_df['species_count'].min():.4f} to {train_df['species_count'].max():.4f}")
    logger.info(f"Species frequency range: {train_df['species_frequency'].min():.4f} to {train_df['species_frequency'].max():.4f}")
    
    # Save species statistics for inference
    species_stats = {
        'counts_log': species_counts_log.to_dict(),
        'frequencies': species_freq.to_dict()
    }
    joblib.dump(species_stats, 'stage2_species_stats.pkl')
    logger.info("✓ Species statistics saved")
    
    # Tabular Feature Preprocessing
    numerical_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos',
                          'NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio',
                          'species_count', 'species_frequency']
    categorical_features = ['State', 'Species', 'season'] 

    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    logger.info("Fitting preprocessor on training data...")
    train_processed = preprocessor.fit_transform(train_df)
    
    try:
        joblib.dump(preprocessor, 'stage2_preprocessor.pkl')
        logger.info("✓ Preprocessor saved as 'stage2_preprocessor.pkl'")
    except Exception as e:
        logger.error(f"Failed to save preprocessor: {e}")

    ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    tabular_feature_names = numerical_features + ohe_feature_names
    logger.info(f"Total tabular features after preprocessing: {len(tabular_feature_names)}")
    
    non_processed_cols = ['sample_id', 'image_path'] + target_cols + [weight_col]
    num_processed_cols = len(numerical_features) + len(ohe_feature_names)
    
    train_df_processed = pd.DataFrame(
        train_processed[:, :num_processed_cols], 
        columns=tabular_feature_names, 
        index=train_df.index
    )
    train_df_processed = pd.concat([train_df_processed, train_df[non_processed_cols]], axis=1)

    val_processed = preprocessor.transform(val_df)
    val_df_processed = pd.DataFrame(
        val_processed[:, :num_processed_cols], 
        columns=tabular_feature_names, 
        index=val_df.index
    )
    val_df_processed = pd.concat([val_df_processed, val_df[non_processed_cols]], axis=1)

    # Image Transformations
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    logger.info("Setting up image transformations with augmentations...")
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),        
        transforms.RandomRotation(15),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),        
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    # Initialize Datasets
    logger.info("Creating PyTorch datasets and dataloaders...")
    train_dataset = Stage2Dataset(train_df_processed, tabular_feature_names, target_cols, 
                                   transform=train_transform, weight_col=weight_col)
    val_dataset = Stage2Dataset(val_df_processed, tabular_feature_names, target_cols, 
                                 transform=val_transform, weight_col=weight_col)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    logger.info(f"Training samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    logger.info(f"Validation samples: {len(val_dataset)}, Batches: {len(val_loader)}")

    # Initialize Model
    tabular_feature_size = len(tabular_feature_names)
    logger.info(f"Initializing MultiModalModel with tabular feature size: {tabular_feature_size}")
    model = MultiModalModel(
        timm_model_name='swin_base_patch4_window7_224',
        tabular_feature_size=tabular_feature_size,
        stage_index=3
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    custom_target_weights = [0.5, 1.0, 0.5, 3.0, 3.0]
    logger.info(f"Custom target weights: {custom_target_weights}")
    
    criterion = WeightedMassBalanceLoss(
        target_weights=custom_target_weights, 
        mass_balance_alpha=0.8
    ).to(DEVICE) 

    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LEARNING_RATE) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    logger.info("Optimizer: Adam")
    logger.info("Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)")
    
    # Training loop with detailed metrics
    best_val_r2 = -float('inf')

    logger.info("="*80)
    logger.info("STARTING TRAINING - STAGE 2")
    logger.info("="*80)

    for epoch in range(NUM_EPOCHS):
        # ==================== TRAINING PHASE ====================
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", leave=False)
        
        for batch_idx, (images, tabular_data, targets, sample_weights) in enumerate(train_bar):
            images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
            sample_weights = sample_weights.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, tabular_data)
            
            loss = criterion(outputs, targets, sample_weights=sample_weights) 
            
            loss.backward()
            optimizer.step()
            
            current_step = epoch + batch_idx / len(train_loader)
            scheduler.step(current_step)
            
            running_loss += loss.item() * images.size(0)
            
            # Store predictions and targets for metrics
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.cpu().numpy())
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = running_loss / len(train_loader.dataset)
        
        # Concatenate training predictions and targets
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
            
            for images, tabular_data, targets, sample_weights in val_bar:
                images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
                sample_weights = sample_weights.to(DEVICE)
                
                outputs = model(images, tabular_data)
                
                loss = criterion(outputs, targets, sample_weights=sample_weights)
                
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # ==================== CALCULATE DETAILED METRICS ====================
        
        # Official Competition Weighted R²
        official_r2_weights = [0.1, 0.1, 0.1, 0.5, 0.2]
        
        def calculate_weighted_r2(y_true, y_pred, weights):
            """Calculate official weighted R² score"""
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            num_samples = y_true.shape[0]
            num_targets = y_true.shape[1]
            
            weights_matrix = np.tile(
                np.array(weights).reshape(1, num_targets), 
                (num_samples, 1)
            )
            sample_weights_flat = weights_matrix.flatten()
            
            return r2_score(y_true_flat, y_pred_flat, sample_weight=sample_weights_flat)
        
        # Calculate per-target R² scores
        def calculate_per_target_metrics(y_true, y_pred, target_names):
            """Calculate R², MAE, and RMSE for each target"""
            metrics = {}
            for i, name in enumerate(target_names):
                true_vals = y_true[:, i]
                pred_vals = y_pred[:, i]
                
                # R² score
                r2 = r2_score(true_vals, pred_vals)
                
                # MAE (Mean Absolute Error)
                mae = np.mean(np.abs(true_vals - pred_vals))
                
                # RMSE (Root Mean Squared Error)
                rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
                
                # MAPE (Mean Absolute Percentage Error) - avoid division by zero
                mask = true_vals != 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100
                else:
                    mape = 0.0
                
                metrics[name] = {
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }
            
            return metrics
        
        # Calculate mass balance metrics
        def calculate_mass_balance_metrics(y_pred):
            """Calculate how well mass balance constraints are satisfied"""
            pred_clover = y_pred[:, 0]
            pred_dead = y_pred[:, 1]
            pred_green = y_pred[:, 2]
            pred_total = y_pred[:, 3]
            pred_gdm = y_pred[:, 4]
            
            # Total balance: Clover + Dead + Green = Total
            total_sum = pred_clover + pred_dead + pred_green
            total_balance_error = np.mean(np.abs(pred_total - total_sum))
            total_balance_rel_error = np.mean(np.abs((pred_total - total_sum) / (pred_total + 1e-6))) * 100
            
            # GDM balance: Clover + Green = GDM
            gdm_sum = pred_clover + pred_green
            gdm_balance_error = np.mean(np.abs(pred_gdm - gdm_sum))
            gdm_balance_rel_error = np.mean(np.abs((pred_gdm - gdm_sum) / (pred_gdm + 1e-6))) * 100
            
            return {
                'total_mae': total_balance_error,
                'total_mape': total_balance_rel_error,
                'gdm_mae': gdm_balance_error,
                'gdm_mape': gdm_balance_rel_error
            }
        
        # Training Metrics
        train_official_r2 = calculate_weighted_r2(train_targets, train_preds, official_r2_weights)
        train_per_target = calculate_per_target_metrics(train_targets, train_preds, target_cols)
        train_mass_balance = calculate_mass_balance_metrics(train_preds)
        
        # Validation Metrics
        val_official_r2 = calculate_weighted_r2(val_targets, val_preds, official_r2_weights)
        val_per_target = calculate_per_target_metrics(val_targets, val_preds, target_cols)
        val_mass_balance = calculate_mass_balance_metrics(val_preds)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # ==================== LOG DETAILED RESULTS ====================
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
        logger.info("-"*80)
        logger.info(f"Learning Rate: {current_lr:.6f}")
        logger.info("-"*80)
        
        # Loss Summary
        logger.info("Loss Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info("-"*80)
        
        # Official Weighted R²
        logger.info("Official Competition Metric (Weighted R²):")
        logger.info(f"  Train: {train_official_r2:.4f}")
        logger.info(f"  Val:   {val_official_r2:.4f}")
        logger.info("-"*80)
        
        # Per-Target Metrics
        logger.info("Per-Target Metrics:")
        logger.info("")
        logger.info("TRAINING SET:")
        for target_name in target_cols:
            metrics = train_per_target[target_name]
            logger.info(f"  {target_name}:")
            logger.info(f"    R²:   {metrics['R²']:>7.4f}  |  MAE:  {metrics['MAE']:>7.3f}  |  "
                       f"RMSE: {metrics['RMSE']:>7.3f}  |  MAPE: {metrics['MAPE']:>6.2f}%")
        
        logger.info("")
        logger.info("VALIDATION SET:")
        for target_name in target_cols:
            metrics = val_per_target[target_name]
            logger.info(f"  {target_name}:")
            logger.info(f"    R²:   {metrics['R²']:>7.4f}  |  MAE:  {metrics['MAE']:>7.3f}  |  "
                       f"RMSE: {metrics['RMSE']:>7.3f}  |  MAPE: {metrics['MAPE']:>6.2f}%")
        logger.info("-"*80)
        
        # Mass Balance Constraint Metrics
        logger.info("Mass Balance Constraint Adherence:")
        logger.info("")
        logger.info("TRAINING SET:")
        logger.info(f"  Total Balance (Clover+Dead+Green=Total):")
        logger.info(f"    MAE:  {train_mass_balance['total_mae']:.4f}  |  MAPE: {train_mass_balance['total_mape']:.2f}%")
        logger.info(f"  GDM Balance (Clover+Green=GDM):")
        logger.info(f"    MAE:  {train_mass_balance['gdm_mae']:.4f}  |  MAPE: {train_mass_balance['gdm_mape']:.2f}%")
        
        logger.info("")
        logger.info("VALIDATION SET:")
        logger.info(f"  Total Balance (Clover+Dead+Green=Total):")
        logger.info(f"    MAE:  {val_mass_balance['total_mae']:.4f}  |  MAPE: {val_mass_balance['total_mape']:.2f}%")
        logger.info(f"  GDM Balance (Clover+Green=GDM):")
        logger.info(f"    MAE:  {val_mass_balance['gdm_mae']:.4f}  |  MAPE: {val_mass_balance['gdm_mape']:.2f}%")
        logger.info("="*80)
        logger.info("")

        # ==================== SAVE BEST MODEL ====================
        if val_official_r2 > best_val_r2:
            improvement = val_official_r2 - best_val_r2
            best_val_r2 = val_official_r2
            
            # Save model checkpoint with metrics
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_r2': best_val_r2,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'val_per_target_metrics': val_per_target,
                'val_mass_balance': val_mass_balance
            }
            
            torch.save(checkpoint, 'stage2_model_checkpoint.pth')
            torch.save(model.state_dict(), 'stage2_model_weighted.pth')
            
            logger.info("✓" * 40)
            logger.info(f"✓ NEW BEST MODEL SAVED FOR STAGE 2!")
            logger.info(f"✓ Improved Official R² by {improvement:.4f}")
            logger.info(f"✓ New best Official R²: {best_val_r2:.4f}")
            logger.info(f"✓ Validation Loss: {avg_val_loss:.4f}")
            logger.info("✓" * 40)
            logger.info("")

    logger.info("="*80)
    logger.info("STAGE 2 TRAINING COMPLETE")
    logger.info(f"Best Validation Official R²: {best_val_r2:.4f}")
    logger.info("="*80)