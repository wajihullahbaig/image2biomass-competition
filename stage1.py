import os
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import timm
import joblib
from tqdm import tqdm
import warnings
from sklearn.metrics import r2_score, accuracy_score, f1_score

# Suppress generic warnings
warnings.filterwarnings("ignore")


def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

# --- CONFIG ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backbone Selection
BACKBONE_SIZE = 'b3'
MODEL_NAME = f'tf_efficientnet_{BACKBONE_SIZE}_ns'


# --- DATASET ---
class Stage1Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, label_encoder=None, weight_col='sample_weight'):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.le = label_encoder
        self.weight_col = weight_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Handle image path
        img_path_raw = row['image_path']
        img_name = img_path_raw.split('/')[-1]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Targets
        species_idx = self.le.transform([row['Species']])[0]
        ndvi = row['Pre_GSHH_NDVI']
        # Log transform height for stability
        height = np.log1p(row['Height_Ave_cm'])

        # Get sample weight
        weight = row[self.weight_col]

        return (
            image,
            torch.tensor(species_idx, dtype=torch.long),
            torch.tensor([ndvi, height], dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32)
        )


# --- MODEL ---
class InputPredictorModel(nn.Module):
    def __init__(self, num_species, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features

        # Head 1: Species Classification
        self.species_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_species)
        )

        # Head 2: Regression (NDVI & Log-Height)
        self.reg_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.species_head(features), self.reg_head(features)


# --- HELPER FUNCTIONS ---
def print_stratification_stats(df, train_df, val_df):
    """Prints stratification statistics for the 'Species' column."""
    print("\n--- Species Counts Verification (Should show nearly identical proportions) ---")

    # 1. Original Dataset Counts
    original_counts = df['Species'].value_counts()
    original_proportions = df['Species'].value_counts(normalize=True).mul(100).round(2)
    print("Original Dataset Counts:")
    print(pd.DataFrame({'Count': original_counts, 'Proportion (%)': original_proportions}))

    print("\nTraining Split Counts (85%):")
    # 2. Training Split Counts
    train_counts = train_df['Species'].value_counts()
    train_proportions = train_df['Species'].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({'Count': train_counts, 'Proportion (%)': train_proportions}))

    print("\nValidation Split Counts (15%):")
    # 3. Validation Split Counts
    val_counts = val_df['Species'].value_counts()
    val_proportions = val_df['Species'].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({'Count': val_counts, 'Proportion (%)': val_proportions}))
    print("--------------------------------------------------------------------------------")


def impute_missing_features(train_df, val_df=None):
    """
    Imputes missing NDVI and Height values using medians.
    If val_df is provided, uses training set medians for validation imputation.
    """
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']

    # Impute training data
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())

    # If validation data provided, use training medians
    if val_df is not None:
        train_medians = train_df[feature_cols].median()
        val_df[feature_cols] = val_df[feature_cols].fillna(train_medians)

    return train_df, val_df
    return train_df


def calculate_sample_weights(df, species_proportions=None, weight_col='sample_weight'):
    """
    Calculates inverse-frequency sample weights based on species distribution.
    The weights are normalized so the mean weight is 1.0.
    """
    if species_proportions is None:
        species_proportions = df['Species'].value_counts(normalize=True)

    # 1. Calculate inverse proportions and normalize
    inverse_proportions = 1 / species_proportions
    mean_inverse = inverse_proportions.mean()
    normalized_weights = inverse_proportions / mean_inverse

    # 2. Create the weight map
    weight_map = normalized_weights.to_dict()

    # 3. Apply the weight to the DataFrame
    df[weight_col] = df['Species'].map(weight_map)

    return df, weight_col


# --- MAIN ---
if __name__ == '__main__':
    print("--- STAGE 1: Training Input Feature Predictor ---")
    print(f"Using Backbone: {MODEL_NAME}")

    # Load Data (Unique samples only)
    df = pd.read_csv('train.csv')
    df_unique = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    print(f"Original rows: {len(df)}")
    print(f"Unique images: {len(df_unique)}")

    # Engineer Date Features
    # Assuming 'Sampling_Date' column exists in your DataFrame
    df_unique['Sampling_Date'] = pd.to_datetime(df_unique['Sampling_Date'])
    df_unique['month'] = df_unique['Sampling_Date'].dt.month
    period = 12
    df_unique['month_sin'] = np.sin(2 * np.pi * df_unique['month'] / period)
    df_unique['month_cos'] = np.cos(2 * np.pi * df_unique['month'] / period)

    # Simple Australian Seasons (approximate) - for categorical feature
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'  # [9, 10, 11]

    df_unique['season'] = df_unique['month'].apply(get_season)
    df_unique = df_unique.drop('Sampling_Date', axis=1)

    # Encode Species
    le = LabelEncoder()
    le.fit(df_unique['Species'])
    joblib.dump(le, 'stage1_species_encoder.pkl')
    print(f"Encoded {len(le.classes_)} species.")

    # STRATIFIED Split by Species
    train_df, val_df = train_test_split(
        df_unique,
        test_size=0.15,
        random_state=42,
        stratify=df_unique['Species']  # Key stratification parameter
    )

    # Print stratification statistics
    print_stratification_stats(df_unique, train_df, val_df)

    # Impute missing features
    train_df, val_df = impute_missing_features(train_df, val_df)
    print("Missing feature values imputed using training set medians.")

    # Calculate Sample Weights (based on Species distribution)
    train_df, weight_col = calculate_sample_weights(train_df)
    val_df, _ = calculate_sample_weights(val_df, species_proportions=train_df['Species'].value_counts(normalize=True))
    val_df[weight_col] = 1.0  # Validation samples have weight 1
    print(f"Sample weights calculated. Weight column: '{weight_col}'")

    # Calculate Class Weights (for handling imbalance)
    print("\nCalculating Class Weights...")
    train_labels = train_df['Species'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class Weights range: {class_weights.min():.2f} to {class_weights.max():.2f}")

     # Image Transformations
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

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

    # Loaders
    train_loader = DataLoader(
        Stage1Dataset(train_df, 'train', train_transform, le, weight_col=weight_col),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        Stage1Dataset(val_df, 'train', val_transform, le, weight_col=weight_col),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Train model
    model = InputPredictorModel(len(le.classes_)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Loss functions
    crit_cls = nn.CrossEntropyLoss(weight=class_weights)
    crit_reg = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for imgs, spec_idx, reg_targets, weights in pbar:
            imgs, spec_idx, reg_targets = imgs.to(DEVICE), spec_idx.to(DEVICE), reg_targets.to(DEVICE)
            weights = weights.to(DEVICE)
            optimizer.zero_grad()
            pred_spec, pred_reg = model(imgs)

            # Calculate losses
            cls_loss = crit_cls(pred_spec, spec_idx)
            reg_loss = crit_reg(pred_reg, reg_targets)

            # Apply sample weights to total loss
            cls_loss = F.cross_entropy(pred_spec, spec_idx, reduction='none')
            reg_loss = F.mse_loss(pred_reg, reg_targets, reduction='none').mean(dim=1)
            loss = ((cls_loss + reg_loss) * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        # Storage for metrics
        all_species_preds = []
        all_species_true = []
        all_reg_preds = []
        all_reg_true = []

        with torch.no_grad():
            for imgs, spec_idx, reg_targets, weights in val_loader:
                imgs = imgs.to(DEVICE)
                spec_idx = spec_idx.to(DEVICE)
                reg_targets = reg_targets.to(DEVICE)

                # Forward pass
                pred_spec_logits, pred_reg = model(imgs)

                # Calculate Loss (no sample weights for validation)
                loss = crit_cls(pred_spec_logits, spec_idx) + crit_reg(pred_reg, reg_targets)
                val_loss += loss.item()

                # Store Predictions for Metrics
                species_probs = torch.softmax(pred_spec_logits, dim=1)
                species_preds = torch.argmax(species_probs, dim=1)
                all_species_preds.append(species_preds.cpu().numpy())
                all_species_true.append(spec_idx.cpu().numpy())
                all_reg_preds.append(pred_reg.cpu().numpy())
                all_reg_true.append(reg_targets.cpu().numpy())

        # --- METRIC CALCULATION ---
        avg_val_loss = val_loss / len(val_loader)

        # Step Scheduler
        scheduler.step(avg_val_loss)

        # Concatenate batches
        all_species_preds = np.concatenate(all_species_preds)
        all_species_true = np.concatenate(all_species_true)
        all_reg_preds = np.concatenate(all_reg_preds)
        all_reg_true = np.concatenate(all_reg_true)

        # 1. Classification Metrics
        val_acc = accuracy_score(all_species_true, all_species_preds)
        val_f1 = f1_score(all_species_true, all_species_preds, average='weighted')

        # 2. Regression Metrics (R2)
        ndvi_r2 = r2_score(all_reg_true[:, 0], all_reg_preds[:, 0])
        height_r2 = r2_score(all_reg_true[:, 1], all_reg_preds[:, 1])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print Detailed Report
        print(f"Epoch {epoch + 1} finished. LR: {current_lr:.6f}")
        print(f" Val Loss: {avg_val_loss:.4f}")
        print(f" Species Acc: {val_acc * 100:.2f}% | F1: {val_f1:.4f}")
        print(f" NDVI R²: {ndvi_r2:.4f} | Height R²: {height_r2:.4f} (Log Space)")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'stage1_model.pth')
            print(" >>> New Best Model Saved!")