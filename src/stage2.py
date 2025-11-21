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

    logger = setup_logging()
    
    # Define parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
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

    print_stratification_stats(df_wide, train_df, val_df,strat_col, logger=logger)


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
    
    # Tabular Feature Preprocessing
    numerical_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos',
                          'NDVI_Height_MUL', 'NDVI_Height_ADD','NDVI_Height_Ratio']
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
        logger.info("✓ Preprocessor saved as 'preprocessor.pkl'")
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
    
    # Training loop
    best_val_r2 = -float('inf')

    logger.info("="*80)
    logger.info("STARTING TRAINING - STAGE 2")
    logger.info("="*80)

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        
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
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
            
            for images, tabular_data, targets, sample_weights in val_bar:
                images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
                sample_weights = sample_weights.to(DEVICE)
                
                outputs = model(images, tabular_data)
                
                loss = criterion(outputs, targets, sample_weights=sample_weights)
                
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate Official Weighted R²
        official_r2_weights = [0.1, 0.1, 0.1, 0.5, 0.2] 
        
        y_true_flat = all_targets.flatten()
        y_pred_flat = all_preds.flatten()
        
        num_samples = all_targets.shape[0]
        num_targets = all_targets.shape[1]
        
        official_weights_matrix = np.tile(
            np.array(official_r2_weights).reshape(1, num_targets), 
            (num_samples, 1)
        )
        sample_weights_flat = official_weights_matrix.flatten()
        
        official_weighted_r2 = r2_score(
            y_true_flat, 
            y_pred_flat, 
            sample_weight=sample_weights_flat
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        epoch_summary = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"LR: {current_lr:.6f}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Official Weighted R²: {official_weighted_r2:.4f}"
        )
        logger.info(epoch_summary)

        # Save best model
        if official_weighted_r2 > best_val_r2:
            improvement = official_weighted_r2 - best_val_r2
            best_val_r2 = official_weighted_r2
            torch.save(model.state_dict(), 'best_multimodal_model_weighted.pth')
            logger.info(f"✓ Model saved! Improved R² by {improvement:.4f} to {best_val_r2:.4f}")

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Validation R²: {best_val_r2:.4f}")
    logger.info("="*80)