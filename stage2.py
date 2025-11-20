from typing import Optional
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

def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


# Data Preparation and Feature Engineering 
def prepare_data(df_train):
    """
    Pivots the long-format DataFrame to a wide format for joint training 
    and engineers date-based features (month, season, interactions).
    """
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
    
    # Simple Australian Seasons (approximate) - for categorical feature
    def get_season(month):
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        else: return 'Spring' # [9, 10, 11]

    wide_df['season'] = wide_df['month'].apply(get_season)
    
    wide_df = wide_df.drop('Sampling_Date', axis=1)
    
    # --- FEATURE INTERACTIONS ---  
    wide_df['NDVI_Height_MUL'] = wide_df['Pre_GSHH_NDVI'] * wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_ADD'] = wide_df['Pre_GSHH_NDVI'] + wide_df['Height_Ave_cm']
    ratio = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5) 
    wide_df['NDVI_Height_Ratio'] = ratio

    return wide_df

def conditional_target_impute(df_to_impute, train_df_for_fit=None):
    """
    Imputes NaN target values using medians. 
    If train_df_for_fit is provided (i.e., for validation/test sets), 
    it uses medians calculated from the training data.
    """
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    grouping_cols = ['Species', 'State', 'season'] 

    if train_df_for_fit is not None:
        # 1. Use Medians from Training Data
        median_map = train_df_for_fit.groupby(grouping_cols)[target_cols].median()
        
        for col in target_cols:
            df_to_impute[col] = df_to_impute.apply(
                lambda row: median_map.loc[(row['Species'], row['State'], row['season']), col]
                if pd.isnull(row[col]) and (row['Species'], row['State'], row['season']) in median_map.index
                else row[col],
                axis=1
            )
        
        # 2. Use Global Median from Training Data for remaining NaNs
        global_medians = train_df_for_fit[target_cols].median()
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(global_medians)
    else:
        # 1. Calculate and apply group median (for training set)
        df_to_impute[target_cols] = df_to_impute.groupby(grouping_cols)[target_cols].transform(
            lambda x: x.fillna(x.median())
        )
        # 2. Apply global median (for remaining NaNs in training set)
        df_to_impute[target_cols] = df_to_impute[target_cols].fillna(df_to_impute[target_cols].median())

    return df_to_impute

def calculate_sample_weights(df, season_proportions, weight_col='sample_weight'):
    """
    Calculates inverse-frequency sample weights based on season distribution.
    The weights are normalized so the mean weight is 1.0.
    """
    # 1. Calculate inverse proportions and normalize
    inverse_proportions = 1 / season_proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse
    
    # 2. Create the weight map
    weight_map = normalized_weights.to_dict()

    # 3. Apply the weight to the DataFrame
    df[weight_col] = df['season'].map(weight_map)
    
    return df, weight_col

# Custom PyTorch Dataset
class Stage2Dataset(Dataset):
    def __init__(self, df, tabular_features, target_cols, image_dir='train', transform=None, weight_col='sample_weight'):
        self.df = df
        self.image_dir = image_dir
        self.tabular_features = tabular_features
        self.target_cols = target_cols
        self.transform = transform
        self.weight_col = weight_col # New: store weight column name

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
        
        # New: Sample Weight
        sample_weight = torch.tensor(row[self.weight_col], dtype=torch.float32)
        
        return image, tabular_data, targets, sample_weight # Return the weight

class WeightedMassBalanceLoss(nn.Module):
    def __init__(self, target_weights=None, mass_balance_alpha=0.8):
        """
        Custom loss function combining Weighted MSE and Mass-Balance penalties.
        The loss supports an optional per-sample weight from the DataLoader.
        
        Args:
            target_weights (list or None): Weights for the 5 individual targets. 
                                            Order: [Clover, Dead, Green, Total, GDM]
            mass_balance_alpha (float): Weight for the mass balance penalty terms.
        """
        super().__init__()
        # Target column indices for slicing:
        self.clover_idx, self.dead_idx, self.green_idx, self.total_idx, self.gdm_idx = 0, 1, 2, 3, 4
        
        # Default weights: giving Total and GDM twice the importance of components
        if target_weights is None:
            target_weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32).view(1, -1))
        self.alpha = mass_balance_alpha

    def forward(self, predictions, targets, sample_weights=None):
        
        # 1. Weighted MSE Loss (Per-Target)
        
        # Calculate squared error for all 5 targets (shape: [B, 5])
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        
        # Apply the custom target weights (shape: [B, 5])
        weighted_error_targets = squared_error * self.target_weights
        
        # Calculate the mean weighted MSE for each sample (shape: [B])
        per_sample_weighted_mse = weighted_error_targets.sum(dim=1) / self.target_weights.sum()

        # --- 2. Mass Balance Penalties (Per-Target & Per-Sample) ---
        
        pred_clover = predictions[:, self.clover_idx]
        pred_dead = predictions[:, self.dead_idx]
        pred_green = predictions[:, self.green_idx]
        pred_total = predictions[:, self.total_idx]
        pred_gdm = predictions[:, self.gdm_idx]
        
        # Penalty A: Dry Total Mass Balance Violation
        predicted_total_sum = pred_clover + pred_dead + pred_green
        # MSE_loss(reduction='none') gives the squared error for each sample (shape: [B])
        total_balance_penalty = F.mse_loss(pred_total, predicted_total_sum, reduction='none')
        
        # Penalty B: GDM Mass Balance Violation
        predicted_gdm_sum = pred_clover + pred_green
        gdm_balance_penalty = F.mse_loss(pred_gdm, predicted_gdm_sum, reduction='none')

        # Total Per-Sample Loss (shape: [B])
        per_sample_loss = per_sample_weighted_mse + self.alpha * (total_balance_penalty + gdm_balance_penalty)
        
        # 3. Apply Per-Sample (Inverse-Frequency) Weight
        if sample_weights is not None:
            # sample_weights shape must be [B]
            per_sample_loss = per_sample_loss * sample_weights

        # 4. Final Reduction: Average across the batch
        total_loss = per_sample_loss.mean()
        
        return total_loss
    
# Multimodal Model Architecture 
class MultiModalModel(nn.Module):
    # (Model architecture remains the same as it is a standard design)
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
    # Define parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    # Load and Prepare Data
    df_train = pd.read_csv('train.csv')
    df_wide = prepare_data(df_train)
    
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Separate features and targets
    X = df_wide.drop(columns=target_cols)
    y = df_wide[target_cols]
    
    # Split data STRATIFIED by season
    # Using X['season'] as the stratification vector
    # We split X and y separately and re-join them temporarily to ensure stratification
    # for the entire sample_id row.
    
    # Stratified split to ensure all seasons are represented in train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=X['season'] # Stratify by the engineered 'season' feature
    )

    print("\n--- Season Counts Verification (Should show nearly identical proportions) ---")

    # 1. Original Dataset Counts
    original_counts = X['season'].value_counts()
    original_proportions = X['season'].value_counts(normalize=True).mul(100).round(2)
    print("Original Dataset Counts:")
    print(pd.DataFrame({'Count': original_counts, 'Proportion (%)': original_proportions}))

    print("\nTraining Split Counts (80%):")
    # 2. Training Split Counts
    train_counts = X_train['season'].value_counts()
    train_proportions = X_train['season'].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({'Count': train_counts, 'Proportion (%)': train_proportions}))

    print("\nValidation Split Counts (20%):")
    # 3. Validation Split Counts
    val_counts = X_val['season'].value_counts()
    val_proportions = X_val['season'].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({'Count': val_counts, 'Proportion (%)': val_proportions}))

    print("--------------------------------------------------------------------------------")


    # Re-combine for preprocessing and easy indexing
    train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)

    # Perform Imputation on the targets 
    train_df_imputed = conditional_target_impute(train_df, train_df_for_fit=None)
    val_df_imputed = conditional_target_impute(val_df, train_df_for_fit=train_df_imputed)

    # CALCULATE AND APPLY INVERSE-FREQUENCY SAMPLE WEIGHTS
    season_proportions = train_df_imputed['season'].value_counts(normalize=True)
    train_df_imputed, weight_col = calculate_sample_weights(train_df_imputed, season_proportions)
    
    # Validation set samples should have a weight of 1.0 for loss calculation 
    # (since the R^2 metric is *not* weighted by season/sample, only by target type).
    val_df_imputed[weight_col] = 1.0 

    # Re-combine again for subsequent code (targets are imputed/processed)
    train_df = train_df_imputed
    val_df = val_df_imputed
    
    # Tabular Feature Preprocessing
    numerical_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos','NDVI_Height_MUL', 'NDVI_Height_ADD','NDVI_Height_Ratio']
    categorical_features = ['State', 'Species', 'season'] 

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit and transform the training data
    train_processed = preprocessor.fit_transform(train_df)
    
    # Save the preprocessor object
    try:
        joblib.dump(preprocessor, 'preprocessor.pkl')
        print("✓ Fitted preprocessor saved as preprocessor.pkl")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")

    # Get the column names for the processed tabular features
    ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    tabular_feature_names = numerical_features + ohe_feature_names
    
    # Recreate the DataFrame for the processed training data
    non_processed_cols = ['sample_id', 'image_path'] + target_cols + [weight_col] # Keep sample_weight
    num_processed_cols = len(numerical_features) + len(ohe_feature_names)
    
    train_df_processed = pd.DataFrame(
        train_processed[:, :num_processed_cols], 
        columns=tabular_feature_names, 
        index=train_df.index
    )
    train_df_processed = pd.concat([train_df_processed, train_df[non_processed_cols]], axis=1)

    # Transform validation data
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

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),        
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1),     
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    # Initialize Datasets with the weight column
    train_dataset = Stage2Dataset(train_df_processed, tabular_feature_names, target_cols, transform=train_transform, weight_col=weight_col)
    val_dataset = Stage2Dataset(val_df_processed, tabular_feature_names, target_cols, transform=val_transform, weight_col=weight_col)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tabular_feature_size = len(tabular_feature_names)
    model = MultiModalModel(
        timm_model_name='swin_base_patch4_window7_224',
        tabular_feature_size=tabular_feature_size,
        stage_index=3
    ).to(DEVICE)
    
    # Loss function's per-target weights (User-defined for optimization)
    custom_target_weights = [
            0.5,  # Dry_Clover_g 
            1.0,  # Dry_Dead_g 
            0.5,  # Dry_Green_g 
            3.0,  # Dry_Total_g 
            3.0   # GDM_g 
        ]
    
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
    
    # --- FULL TRAINING AND VALIDATION LOOP ---

    best_val_r2 = -float('inf') # Track best R2 instead of loss for the competition metric

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs with scheduler and augmentations...")

    for epoch in range(NUM_EPOCHS):
        # ----------------------------------------------------
        # 1. Training Phase (Apply Per-Sample Weight)
        # ----------------------------------------------------
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", leave=False)
        
        # New: Unpack the sample_weights from the DataLoader
        for batch_idx, (images, tabular_data, targets, sample_weights) in enumerate(train_bar):
            images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
            sample_weights = sample_weights.to(DEVICE) # Move weights to device
            
            optimizer.zero_grad()
            outputs = model(images, tabular_data)
            
            # Pass sample_weights to the modified criterion
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

        # ----------------------------------------------------
        # 2. Validation Phase (Calculate Official Weighted R^2)
        # ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
            
            # New: Unpack the sample_weights (though they are all 1.0 for val set)
            for images, tabular_data, targets, sample_weights in val_bar:
                images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE)
                sample_weights = sample_weights.to(DEVICE)
                
                outputs = model(images, tabular_data)
                
                # Pass sample_weights to the criterion (it will use 1.0 for all)
                loss = criterion(outputs, targets, sample_weights=sample_weights)
                
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # --- CRITICAL  Official Weighted R^2 Calculation ---
        
        # Official competition weights in the order: 
        # ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        official_r2_weights = [0.1, 0.1, 0.1, 0.5, 0.2] 
        
        # 1. Flatten the target and prediction arrays
        y_true_flat = all_targets.flatten()
        y_pred_flat = all_preds.flatten()
        
        # 2. Create the sample_weight vector for the R^2 metric
        num_samples = all_targets.shape[0]
        num_targets = all_targets.shape[1]
        
        # Repeat the official weights for every sample
        official_weights_matrix = np.tile(
            np.array(official_r2_weights).reshape(1, num_targets), 
            (num_samples, 1)
        )
        sample_weights_flat = official_weights_matrix.flatten()
        
        # 3. Calculate the single, globally weighted R^2 (the competition metric)
        official_weighted_r2 = r2_score(
            y_true_flat, 
            y_pred_flat, 
            sample_weight=sample_weights_flat
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch+1} finished. LR: {current_lr:.6f}, "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Official Weighted R2: {official_weighted_r2:.4f}"
        )

        # Save Best Model based on Official Weighted R2
        if official_weighted_r2 > best_val_r2:
            best_val_r2 = official_weighted_r2
            torch.save(model.state_dict(), 'best_multimodal_model_weighted.pth')
            print("Model saved due to improved validation R2.")