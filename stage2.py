import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import joblib
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_NAME = 'swin_base_patch4_window7_224'  # Or try 'tf_efficientnet_b0_ns' if you have memory issues
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
OFFICIAL_R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.5, 0.2])

# --- 1. DATA PREPARATION & FEATURE ENGINEERING ---
def prepare_data(df_train):
    """
    Pivots long-format train data to wide-format and adds engineered features.
    """
    wide_df = df_train.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    wide_df['Sampling_Date'] = pd.to_datetime(wide_df['Sampling_Date'])
    wide_df['month'] = wide_df['Sampling_Date'].dt.month
    wide_df['month_sin'] = np.sin(2 * np.pi * wide_df['month'] / 12)
    wide_df['month_cos'] = np.cos(2 * np.pi * wide_df['month'] / 12)

    def get_season(month):
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        else: return 'Spring'
    wide_df['season'] = wide_df['month'].apply(get_season)

    wide_df['NDVI_Height_MUL'] = wide_df['Pre_GSHH_NDVI'] * wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_ADD'] = wide_df['Pre_GSHH_NDVI'] + wide_df['Height_Ave_cm']
    wide_df['NDVI_Height_Ratio'] = wide_df['Pre_GSHH_NDVI'] / (wide_df['Height_Ave_cm'] + 1e-5)

    for col in TARGET_COLS:
        if col in wide_df.columns:
            wide_df[col] = wide_df[col].fillna(wide_df[col].median())

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

def calculate_sample_weights(df, season_proportions=None, weight_col='sample_weight'):
    """
    Calculates inverse-frequency sample weights based on season distribution.
    The weights are normalized so the mean weight is 1.0.
    """

    if season_proportions is None:
        season_proportions = df['season'].value_counts(normalize=True)

    # 1. Calculate inverse proportions and normalize
    inverse_proportions = 1 / season_proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse

    # 2. Create the weight map
    weight_map = normalized_weights.to_dict()

    # 3. Apply the weight to the DataFrame
    df[weight_col] = df['season'].map(weight_map)

    return df, weight_col


# --- 2. DATASET CLASS ---
class Stage2Dataset(Dataset):
    def __init__(self, df, tabular_cols, target_cols, image_dir, transform=None, weight_col='sample_weight'):
        self.df = df
        self.tabular_data = df[tabular_cols].values.astype(np.float32)
        self.targets = df[target_cols].values.astype(np.float32)
        self.weights = df[weight_col].values.astype(np.float32)  # Sample weights
        self.image_paths = df['image_path'].values
        self.image_dir = image_dir
        self.transform = transform
        self.weight_col = weight_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path_raw = self.image_paths[idx]
        img_name = img_path_raw.split('/')[-1]
        full_img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(full_img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(self.tabular_data[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.weights[idx])  # Return sample weight
        )

# --- 3. LOSS FUNCTION ---
class WeightedMassBalanceLoss(nn.Module):
    def __init__(self, mass_balance_weight=0.8):
        super().__init__()
        self.target_weights = torch.tensor(OFFICIAL_R2_WEIGHTS, dtype=torch.float32).to(DEVICE)
        self.alpha = mass_balance_weight

    def forward(self, preds, targets, sample_weights):
        mse_per_col = F.mse_loss(preds, targets, reduction='none')
        weighted_mse = (mse_per_col * self.target_weights).mean(dim=1)

        p_clover, p_dead, p_green, p_total, p_gdm = preds.T
        bal_total = F.mse_loss(p_total, p_clover + p_dead + p_green, reduction='none')
        bal_gdm = F.mse_loss(p_gdm, p_clover + p_green, reduction='none')

        total_loss = weighted_mse + self.alpha * (bal_total + bal_gdm)

        return (total_loss * sample_weights).mean()  # Apply sample weights

# --- 4. MODEL ARCHITECTURE ---
class MultiModalModel(nn.Module):
    def __init__(self, tab_input_size, output_size=5, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        img_feature_dim = self.backbone.num_features

        self.mlp = nn.Sequential(
            nn.Linear(img_feature_dim + tab_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, img, tab):
        img_feat = self.backbone(img)
        if len(img_feat.shape) > 2:
            img_feat = img_feat.mean(dim=[2, 3])

        combined = torch.cat([img_feat, tab], dim=1)
        return self.mlp(combined)

# --- 5. VALIDATION AND STATS ---
def calculate_and_print_stats(model, val_loader, criterion, device, official_r2_weights, epoch, optimizer):
    """
    Calculates validation loss and the official competition R^2 metric, and prints the stats.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular_data, targets, sample_weights in val_loader:
            images, tabular_data, targets = images.to(device), tabular_data.to(device), targets.to(device)
            sample_weights = sample_weights.to(device)

            outputs = model(images, tabular_data)
            loss = criterion(outputs, targets, sample_weights=sample_weights)

            val_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # --- CRITICAL CORRECTION: Official Weighted R^2 Calculation ---
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

    print(
        f"Epoch {epoch + 1} finished. LR: {current_lr:.6f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Official Weighted R2: {official_weighted_r2:.4f}"
    )

    return official_weighted_r2

def print_stratification_stats(df, X_train, X_val):
    """Prints stratification statistics for the 'season' column."""
    print("\n--- Season Counts Verification (Should show nearly identical proportions) ---")

    # 1. Original Dataset Counts
    original_counts = df['season'].value_counts()
    original_proportions = df['season'].value_counts(normalize=True).mul(100).round(2)
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

# --- 6. MAIN TRAINING LOOP ---
if __name__ == '__main__':
    print(f"--- STAGE 2: Main Biomass Training on {DEVICE} ---")

    # Load Raw Data
    raw_df = pd.read_csv('train.csv')

    # Prepare Data (Pivot & Feature Engineering)
    df = prepare_data(raw_df)
    print(f"Data Loaded. Shape: {df.shape}")

    # Split Data (Stratified by Season)
    X = df.drop(columns=TARGET_COLS)
    y = df[TARGET_COLS]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=X['season']
    )
    train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)
    print_stratification_stats(df, X_train, X_val)

    # Impute Missing Values
    train_df = conditional_target_impute(train_df)
    val_df = conditional_target_impute(val_df, train_df_for_fit=train_df)

    # Calculate Sample Weights
    train_df, weight_col = calculate_sample_weights(train_df)
    val_df, _ = calculate_sample_weights(val_df, season_proportions=train_df['season'].value_counts(normalize=True))
    val_df[weight_col] = 1.0  # Validation samples have weight 1

    # Define Feature Columns
    numerical_features = [
        'Pre_GSHH_NDVI', 'Height_Ave_cm', 'month', 'month_sin', 'month_cos',
        'NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio'
    ]
    categorical_features = ['State', 'Species', 'season']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # Fit Preprocessor on Train, Apply to Val
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    # Save Preprocessor for Inference
    joblib.dump(preprocessor, 'stage2_preprocessor.pkl')
    print("Preprocessor saved.")

    # Reconstruct DataFrames with processed features for PyTorch Dataset
    feature_cols = [f'feat_{i}' for i in range(X_train.shape[1])]

    train_df_final = pd.concat([train_df.reset_index(drop=True), pd.DataFrame(X_train, columns=feature_cols)], axis=1)
    val_df_final = pd.concat([val_df.reset_index(drop=True), pd.DataFrame(X_val, columns=feature_cols)], axis=1)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_ds = Stage2Dataset(train_df_final, feature_cols, TARGET_COLS, 'train', transform=train_tf, weight_col=weight_col)
    val_ds = Stage2Dataset(val_df_final, feature_cols, TARGET_COLS, 'train', transform=val_tf, weight_col=weight_col)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    model = MultiModalModel(tab_input_size=len(feature_cols)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = WeightedMassBalanceLoss()

    best_r2 = -float('inf')

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, tabs, targets, weights in pbar:
            imgs, tabs = imgs.to(DEVICE), tabs.to(DEVICE)
            targets, weights = targets.to(DEVICE), weights.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs, tabs)
            loss = criterion(preds, targets, weights)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_ds)

        # --- VALIDATION WITH CORRECTED METRIC ---
        val_r2 = calculate_and_print_stats(model, val_loader, criterion, DEVICE, OFFICIAL_R2_WEIGHTS, epoch, optimizer)
        scheduler.step(val_r2)

        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), 'stage2_model.pth')
            print(f"  >>> New Best Model Saved! (R2: {best_r2:.4f})")