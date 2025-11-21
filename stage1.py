import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
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

from common import calculate_sample_weights, get_season, print_stratification_stats, set_seed, setup_logging

# Suppress generic warnings
warnings.filterwarnings("ignore")

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


class InputPredictorModel(nn.Module):
    def __init__(self, num_species, model_name='tf_efficientnet_b3_ns'):
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



def impute_missing_features(train_df, val_df=None, logger=None):
    """
    Imputes missing NDVI and Height values using medians.
    If val_df is provided, uses training set medians for validation imputation.
    """
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    if logger:
        train_missing = train_df[feature_cols].isnull().sum()
        if train_missing.sum() > 0:
            logger.info(f"Missing values in training set: {train_missing.to_dict()}")

    # Impute training data
    train_medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)

    # If validation data provided, use training medians
    if val_df is not None:
        if logger:
            val_missing = val_df[feature_cols].isnull().sum()
            if val_missing.sum() > 0:
                logger.info(f"Missing values in validation set: {val_missing.to_dict()}")
        
        val_df[feature_cols] = val_df[feature_cols].fillna(train_medians)
        
        if logger:
            logger.info(f"Imputation complete using training medians: {train_medians.to_dict()}")

    return train_df, val_df



if __name__ == '__main__':
    # Setup logging first
    logger = setup_logging()
    
    # Configuration
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONE_SIZE = 'b3'
    MODEL_NAME = f'tf_efficientnet_{BACKBONE_SIZE}_ns'
    
    logger.info("="*80)
    logger.info("STAGE 1: TRAINING INPUT FEATURE PREDICTOR")
    logger.info("="*80)
    logger.info(f"Backbone Model: {MODEL_NAME}")
    logger.info(f"Image Size: {IMAGE_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning Rate: {LR}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*80)
    
    set_seed(logger=logger)

    # Load Data
    logger.info("Loading training data from 'train.csv'...")
    df = pd.read_csv('train.csv')
    df_unique = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    logger.info(f"Original rows: {len(df)}")
    logger.info(f"Unique images: {len(df_unique)}")

    # Engineer Date Features
    logger.info("Engineering date-based features...")
    df_unique['Sampling_Date'] = pd.to_datetime(df_unique['Sampling_Date'])
    df_unique['month'] = df_unique['Sampling_Date'].dt.month
    period = 12
    df_unique['month_sin'] = np.sin(2 * np.pi * df_unique['month'] / period)
    df_unique['month_cos'] = np.cos(2 * np.pi * df_unique['month'] / period)



    df_unique['season'] = df_unique['month'].apply(get_season)
    df_unique = df_unique.drop('Sampling_Date', axis=1)
    logger.info("Date features created: month_sin, month_cos, season")

    # Encode Species
    logger.info("Encoding species labels...")
    le = LabelEncoder()
    le.fit(df_unique['Species'])
    
    try:
        joblib.dump(le, 'stage1_species_encoder.pkl')
        logger.info(f"✓ Species encoder saved. Number of classes: {len(le.classes_)}")
        logger.debug(f"Species classes: {le.classes_.tolist()}")
    except Exception as e:
        logger.error(f"Failed to save species encoder: {e}")

    # Stratified Split
    logger.info("Performing stratified train-validation split (80/20)...")
    strat_col = 'season'
    train_df, val_df = train_test_split(
        df_unique,
        test_size=0.2,
        random_state=42,
        stratify=df_unique[strat_col]
    )

    print_stratification_stats(df_unique, train_df, val_df,strat_col, logger=logger)

    train_df, val_df = impute_missing_features(train_df, val_df, logger=logger)

    logger.info("Calculating sample weights for training...")
    prop_col = 'Species'
    train_df, weight_col = calculate_sample_weights(train_df, prop_col=prop_col, logger=logger)
    val_df[weight_col] = 1.0
    logger.info("Sample weights applied. Validation weights set to 1.0")

    # Calculate Class Weights
    logger.info("Calculating class weights for balanced training...")
    train_labels = train_df[prop_col].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    logger.info(f"Class weights range: {class_weights.min():.2f} to {class_weights.max():.2f}")
    logger.debug(f"Class weights: {class_weights.cpu().numpy()}")

    # Image Transformations
    logger.info("Setting up image transformations with augmentations...")
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # Create Dataloaders
    logger.info("Creating PyTorch datasets and dataloaders...")
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
    
    logger.info(f"Training samples: {len(train_df)}, Batches: {len(train_loader)}")
    logger.info(f"Validation samples: {len(val_df)}, Batches: {len(val_loader)}")

    # Initialize Model
    logger.info(f"Initializing {MODEL_NAME} model...")
    model = InputPredictorModel(len(le.classes_), MODEL_NAME).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    logger.info(f"Optimizer: AdamW with LR={LR}")
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    logger.info("Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)")

    # Loss functions
    crit_cls = nn.CrossEntropyLoss(weight=class_weights)
    crit_reg = nn.MSELoss()
    best_loss = float('inf')

    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        
        for imgs, spec_idx, reg_targets, weights in pbar:
            imgs, spec_idx, reg_targets = imgs.to(DEVICE), spec_idx.to(DEVICE), reg_targets.to(DEVICE)
            weights = weights.to(DEVICE)
            optimizer.zero_grad()
            pred_spec, pred_reg = model(imgs)

            # Calculate losses with sample weights
            cls_loss = F.cross_entropy(pred_spec, spec_idx, reduction='none')
            reg_loss = F.mse_loss(pred_reg, reg_targets, reduction='none').mean(dim=1)
            loss = ((cls_loss + reg_loss) * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
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

                # Store Predictions
                species_probs = torch.softmax(pred_spec_logits, dim=1)
                species_preds = torch.argmax(species_probs, dim=1)
                all_species_preds.append(species_preds.cpu().numpy())
                all_species_true.append(spec_idx.cpu().numpy())
                all_reg_preds.append(pred_reg.cpu().numpy())
                all_reg_true.append(reg_targets.cpu().numpy())

        # Calculate Metrics
        avg_val_loss = val_loss / len(val_loader)
        
        # Step Scheduler
        scheduler.step(avg_val_loss)

        # Concatenate batches
        all_species_preds = np.concatenate(all_species_preds)
        all_species_true = np.concatenate(all_species_true)
        all_reg_preds = np.concatenate(all_reg_preds)
        all_reg_true = np.concatenate(all_reg_true)

        # Classification Metrics
        val_acc = accuracy_score(all_species_true, all_species_preds)
        val_f1 = f1_score(all_species_true, all_species_preds, average='weighted')

        # Regression Metrics (R²)
        ndvi_r2 = r2_score(all_reg_true[:, 0], all_reg_preds[:, 0])
        height_r2 = r2_score(all_reg_true[:, 1], all_reg_preds[:, 1])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch results
        epoch_summary = (
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"LR: {current_lr:.6f}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        logger.info(epoch_summary)
        logger.info(f"  Species - Accuracy: {val_acc * 100:.2f}%, F1: {val_f1:.4f}")
        logger.info(f"  Regression - NDVI R²: {ndvi_r2:.4f}, Height R² (log): {height_r2:.4f}")

        # Save best model
        if avg_val_loss < best_loss:
            improvement = best_loss - avg_val_loss
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'stage1_model.pth')
            logger.info(f"✓ Model saved! Improved validation loss by {improvement:.4f} to {best_loss:.4f}")

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Validation Loss: {best_loss:.4f}")
    logger.info("="*80)