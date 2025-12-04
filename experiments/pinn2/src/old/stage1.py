# stage1.py
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import timm
import joblib
from tqdm import tqdm
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from common import BATCH_SIZE, DEVICE, NUM_EPOCHS, IMAGE_SIZE, LEARNING_RATE, SeasonalCurriculumSampler, calculate_sample_weights, get_image_data_transforms, get_season, print_stratification_stats, set_seed, setup_logging

# Suppress generic warnings
warnings.filterwarnings("ignore")

class Stage1Dataset(Dataset):
    def __init__(self, df, tabular_features, image_dir, transform=None, weight_col='sample_weight'):
        self.df = df
        self.tabular_features = tabular_features
        self.image_dir = image_dir
        self.transform = transform
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

        # Tabular features (includes species one-hot, count features, etc.)
        tabular_data = torch.tensor(row[self.tabular_features].values.astype(np.float32))

        # Regression Targets (NDVI & Log-Height)
        ndvi = row['Pre_GSHH_NDVI']
        height = np.log1p(row['Height_Ave_cm'])
        reg_targets = torch.tensor([ndvi, height], dtype=torch.float32)

        # Get sample weight
        weight = row[self.weight_col]

        return (
            image,
            tabular_data,
            reg_targets,
            torch.tensor(weight, dtype=torch.float32)
        )


class InputPredictorModel(nn.Module):
    def __init__(self, tabular_feature_size, model_name='tf_efficientnet_b3_ns'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        
        # Project tabular features (species one-hot + count features)
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Combined features dimension
        combined_features = in_features + 64

        # Regression Head (NDVI & Log-Height)
        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: [NDVI, log(Height)]
        )

    def forward(self, x, tabular_features):
        # Extract image features
        img_features = self.backbone(x)
        
        # Project tabular features
        tab_embed = self.tabular_projection(tabular_features)
        
        # Concatenate image and tabular features
        features = torch.cat([img_features, tab_embed], dim=1)
        
        return self.reg_head(features)


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

    for b in ['b0','b1','b2','b3','b4','b5','b6' ,'b7']:
        logger = setup_logging(file_name_part=f"stage1_training_{b}")
        
        BACKBONE_SIZE = b
        MODEL_NAME = f'tf_efficientnet_{BACKBONE_SIZE}_ns'
        
        logger.info("="*80)
        logger.info("STAGE 1: TRAINING INPUT FEATURE PREDICTOR (NDVI & HEIGHT)")
        logger.info("="*80)
        logger.info(f"Backbone Model: {MODEL_NAME}")
        logger.info(f"Image Size: {IMAGE_SIZE}")
        logger.info(f"Batch Size: {BATCH_SIZE}")
        logger.info(f"Epochs: {NUM_EPOCHS}")
        logger.info(f"Learning Rate: {LEARNING_RATE}")
        logger.info(f"Device: {DEVICE}")
        logger.info("="*80)
        
        set_seed(logger=logger)

        # Load Data
        logger.info("Loading training data from 'train.csv'...")
        df = pd.read_csv('./train.csv')
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

        # Stratified Split
        logger.info("Performing stratified train-validation split (80/20)...")
        strat_col = 'season'
        train_df, val_df = train_test_split(
            df_unique,
            test_size=0.2,
            random_state=42,
            stratify=df_unique[strat_col]
        )

        print_stratification_stats(df_unique, train_df, val_df, strat_col, logger=logger)

        # ============================================================================
        # ADD SPECIES COUNT FEATURES (GLOBAL AND SEASONAL) - NO LEAKAGE
        # ============================================================================
        logger.info("Adding species count features (global and seasonal)...")
        
        # GLOBAL SPECIES COUNTS (from training set only)
        species_counts_global = train_df['Species'].value_counts()
        species_freq_global = species_counts_global / len(train_df)
        species_counts_global_log = np.log1p(species_counts_global)
        
        logger.info(f"Total unique species in training: {len(species_counts_global)}")
        
        # SEASONAL SPECIES COUNTS (from training set only)
        season_species_counts = train_df.groupby(['season', 'Species']).size()
        season_species_counts_log = np.log1p(season_species_counts)
        
        # Calculate frequency within each season
        season_totals = train_df.groupby('season').size()
        season_species_freq = season_species_counts / season_species_counts.index.map(
            lambda x: season_totals[x[0]]
        )
        
        logger.info(f"Season distribution in training: {season_totals.to_dict()}")
        
        # ADD FEATURES TO TRAINING SET
        # Global counts
        train_df['species_count_global'] = train_df['Species'].map(species_counts_global_log)
        train_df['species_freq_global'] = train_df['Species'].map(species_freq_global)
        
        # Seasonal counts
        train_df['species_count_seasonal'] = train_df.apply(
            lambda row: season_species_counts_log.get((row['season'], row['Species']), 0),
            axis=1
        )
        train_df['species_freq_seasonal'] = train_df.apply(
            lambda row: season_species_freq.get((row['season'], row['Species']), 0),
            axis=1
        )
        
        logger.info("Training set feature statistics:")
        logger.info(f"  Global count range: {train_df['species_count_global'].min():.4f} to {train_df['species_count_global'].max():.4f}")
        logger.info(f"  Global freq range: {train_df['species_freq_global'].min():.4f} to {train_df['species_freq_global'].max():.4f}")
        logger.info(f"  Seasonal count range: {train_df['species_count_seasonal'].min():.4f} to {train_df['species_count_seasonal'].max():.4f}")
        logger.info(f"  Seasonal freq range: {train_df['species_freq_seasonal'].min():.4f} to {train_df['species_freq_seasonal'].max():.4f}")
        
        # ADD FEATURES TO VALIDATION SET (using training statistics only - NO LEAKAGE)
        train_mean_freq_global = species_freq_global.mean()
        train_mean_freq_seasonal = season_species_freq.mean()
        
        val_df['species_count_global'] = val_df['Species'].map(species_counts_global_log).fillna(0)
        val_df['species_freq_global'] = val_df['Species'].map(species_freq_global).fillna(train_mean_freq_global)
        
        val_df['species_count_seasonal'] = val_df.apply(
            lambda row: season_species_counts_log.get((row['season'], row['Species']), 0),
            axis=1
        )
        val_df['species_freq_seasonal'] = val_df.apply(
            lambda row: season_species_freq.get((row['season'], row['Species']), train_mean_freq_seasonal),
            axis=1
        )
        
        logger.info("Validation set feature statistics:")
        logger.info(f"  Global count range: {val_df['species_count_global'].min():.4f} to {val_df['species_count_global'].max():.4f}")
        logger.info(f"  Global freq range: {val_df['species_freq_global'].min():.4f} to {val_df['species_freq_global'].max():.4f}")
        logger.info(f"  Seasonal count range: {val_df['species_count_seasonal'].min():.4f} to {val_df['species_count_seasonal'].max():.4f}")
        logger.info(f"  Seasonal freq range: {val_df['species_freq_seasonal'].min():.4f} to {val_df['species_freq_seasonal'].max():.4f}")
        
        # Check for unseen combinations
        val_combinations = set(zip(val_df['season'], val_df['Species']))
        train_combinations = set(season_species_counts.index)
        unseen_combinations = val_combinations - train_combinations
        if unseen_combinations:
            logger.info(f"Warning: {len(unseen_combinations)} species-season combinations in validation not seen in training")
            logger.info(f"These will use default values (0 for count, {train_mean_freq_seasonal:.4f} for frequency)")
        else:
            logger.info("✓ All validation species-season combinations were seen in training")

        # Impute missing features
        train_df, val_df = impute_missing_features(train_df, val_df, logger=logger)

        # Calculate sample weights
        logger.info("Calculating sample weights for training...")
        prop_col = 'Species'
        train_df, weight_col = calculate_sample_weights(train_df, group_col=prop_col, logger=logger)
        val_df[weight_col] = 1.0
        logger.info("Sample weights applied. Validation weights set to 1.0")

        count_freq_features = [
            'species_count_global', 
            'species_freq_global',
            'species_count_seasonal',
            'species_freq_seasonal'
        ]
        base_numerical_features = [
            'month', 'month_sin', 'month_cos',
        ]

        numerical_features = base_numerical_features + count_freq_features
        categorical_features = ['Species']
        
        logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )

        # Prepare data for preprocessing (only tabular features)
        tabular_cols = numerical_features + categorical_features
        
        logger.info("Fitting preprocessor on training data...")
        train_tabular = preprocessor.fit_transform(train_df[tabular_cols])
        
        # Save preprocessor
        try:
            joblib.dump(preprocessor, 'stage1_preprocessor.pkl')
            logger.info("✓ Preprocessor saved as 'stage1_preprocessor.pkl'")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {e}")

        # Get feature names
        ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        tabular_feature_names = numerical_features + ohe_feature_names
        logger.info(f"Total tabular features after preprocessing: {len(tabular_feature_names)}")
        logger.debug(f"Tabular features: {tabular_feature_names}")
        
        # Create processed dataframes
        non_processed_cols = ['sample_id', 'image_path', 'Pre_GSHH_NDVI', 'Height_Ave_cm', weight_col,'season']
        
        train_df_processed = pd.DataFrame(
            train_tabular,
            columns=tabular_feature_names,
            index=train_df.index
        )
        train_df_processed = pd.concat([train_df_processed, train_df[non_processed_cols]], axis=1)

        val_tabular = preprocessor.transform(val_df[tabular_cols])
        val_df_processed = pd.DataFrame(
            val_tabular,
            columns=tabular_feature_names,
            index=val_df.index
        )
        val_df_processed = pd.concat([val_df_processed, val_df[non_processed_cols]], axis=1)

        # Use SeasonalCurriculumSampler for training
        train_sampler = SeasonalCurriculumSampler(
            data_df=train_df_processed,
            shuffle_within_season=False,
            seed=42
        )
        # Image Transformations
        logger.info("Setting up image transformations with augmentations...")
        train_transform, val_transform = get_image_data_transforms()
        # Create Dataloaders
        logger.info("Creating PyTorch datasets and dataloaders...")
        train_loader = DataLoader(
            Stage1Dataset(train_df_processed, tabular_feature_names, 'train', train_transform, weight_col=weight_col),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            sampler=train_sampler,
            drop_last=True          
        )

        val_loader = DataLoader(
            Stage1Dataset(val_df_processed, tabular_feature_names, 'train', val_transform, weight_col=weight_col),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        logger.info(f"Training samples: {len(train_df_processed)}, Batches: {len(train_loader)}")
        logger.info(f"Validation samples: {len(val_df_processed)}, Batches: {len(val_loader)}")

        # Initialize Model
        tabular_feature_size = len(tabular_feature_names)
        logger.info(f"Initializing {MODEL_NAME} model with {tabular_feature_size} tabular features...")
        model = InputPredictorModel(tabular_feature_size, MODEL_NAME).to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        logger.info(f"Optimizer: AdamW with LR={LEARNING_RATE}")
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        logger.info("Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

        # Loss function
        crit_reg = nn.MSELoss()
        best_loss = float('inf')

        logger.info("="*80)
        logger.info("STARTING TRAINING - STAGE 1")
        logger.info("="*80)

        for epoch in range(NUM_EPOCHS):
            # ==================== TRAINING PHASE ====================
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            for imgs, tabular_feats, reg_targets, weights in pbar:
                # Move to device
                imgs = imgs.to(DEVICE)
                tabular_feats = tabular_feats.to(DEVICE)
                reg_targets = reg_targets.to(DEVICE)
                weights = weights.to(DEVICE)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with tabular features
                pred_reg = model(imgs, tabular_feats)

                # Calculate loss with sample weights
                reg_loss = F.mse_loss(pred_reg, reg_targets, reduction='none').mean(dim=1)
                loss = (reg_loss * weights).mean()

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate
                train_loss += loss.item()
                train_preds.append(pred_reg.detach().cpu().numpy())
                train_targets.append(reg_targets.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            
            # Concatenate training predictions
            train_preds = np.concatenate(train_preds)
            train_targets = np.concatenate(train_targets)

            # ==================== VALIDATION PHASE ====================
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for imgs, tabular_feats, reg_targets, weights in val_loader:
                    # Move to device
                    imgs = imgs.to(DEVICE)
                    tabular_feats = tabular_feats.to(DEVICE)
                    reg_targets = reg_targets.to(DEVICE)

                    # Forward pass
                    pred_reg = model(imgs, tabular_feats)

                    # Calculate Loss (no sample weights for validation)
                    loss = crit_reg(pred_reg, reg_targets)
                    val_loss += loss.item()

                    # Store predictions
                    val_preds.append(pred_reg.cpu().numpy())
                    val_targets.append(reg_targets.cpu().numpy())

            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            
            # Step the scheduler
            scheduler.step(avg_val_loss)

            # Concatenate validation predictions
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)

            # ==================== CALCULATE METRICS ====================
            # Training Metrics
            train_ndvi_r2 = r2_score(train_targets[:, 0], train_preds[:, 0])
            train_height_r2 = r2_score(train_targets[:, 1], train_preds[:, 1])
            train_ndvi_mae = mean_absolute_error(train_targets[:, 0], train_preds[:, 0])
            train_height_mae = mean_absolute_error(train_targets[:, 1], train_preds[:, 1])
            train_ndvi_rmse = np.sqrt(mean_squared_error(train_targets[:, 0], train_preds[:, 0]))
            train_height_rmse = np.sqrt(mean_squared_error(train_targets[:, 1], train_preds[:, 1]))
            
            # Validation Metrics
            val_ndvi_r2 = r2_score(val_targets[:, 0], val_preds[:, 0])
            val_height_r2 = r2_score(val_targets[:, 1], val_preds[:, 1])
            val_ndvi_mae = mean_absolute_error(val_targets[:, 0], val_preds[:, 0])
            val_height_mae = mean_absolute_error(val_targets[:, 1], val_preds[:, 1])
            val_ndvi_rmse = np.sqrt(mean_squared_error(val_targets[:, 0], val_preds[:, 0]))
            val_height_rmse = np.sqrt(mean_squared_error(val_targets[:, 1], val_preds[:, 1]))

            # Average R² (equal weight)
            train_avg_r2 = (train_ndvi_r2 + train_height_r2) / 2
            val_avg_r2 = (val_ndvi_r2 + val_height_r2) / 2

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # ==================== LOG EPOCH RESULTS ====================
            logger.info("="*80)
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} Summary")
            logger.info("-"*80)
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info("-"*80)
            logger.info("Loss Summary:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
            logger.info("-"*80)
            logger.info("Average R² Score:")
            logger.info(f"  Train: {train_avg_r2:.4f}")
            logger.info(f"  Val:   {val_avg_r2:.4f}")
            logger.info("-"*80)
            logger.info("Detailed Metrics:")
            logger.info("")
            logger.info("TRAINING SET:")
            logger.info(f"  NDVI:")
            logger.info(f"    R²:   {train_ndvi_r2:>7.4f}  |  MAE: {train_ndvi_mae:>7.4f}  |  RMSE: {train_ndvi_rmse:>7.4f}")
            logger.info(f"  Height (log):")
            logger.info(f"    R²:   {train_height_r2:>7.4f}  |  MAE: {train_height_mae:>7.4f}  |  RMSE: {train_height_rmse:>7.4f}")
            logger.info("")
            logger.info("VALIDATION SET:")
            logger.info(f"  NDVI:")
            logger.info(f"    R²:   {val_ndvi_r2:>7.4f}  |  MAE: {val_ndvi_mae:>7.4f}  |  RMSE: {val_ndvi_rmse:>7.4f}")
            logger.info(f"  Height (log):")
            logger.info(f"    R²:   {val_height_r2:>7.4f}  |  MAE: {val_height_mae:>7.4f}  |  RMSE: {val_height_rmse:>7.4f}")
            logger.info("="*80)
            logger.info("")

            # ==================== SAVE BEST MODEL ====================
            if avg_val_loss < best_loss:
                improvement = best_loss - avg_val_loss
                best_loss = avg_val_loss
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'val_ndvi_r2': val_ndvi_r2,
                    'val_height_r2': val_height_r2,
                    'val_avg_r2': val_avg_r2,
                }
                
                torch.save(checkpoint, 'stage1_model_checkpoint.pth')
                torch.save(model.state_dict(), 'stage1_model.pth')
                
                logger.info("✓" * 40)
                logger.info(f"✓ NEW BEST MODEL SAVED FOR STAGE 1!")
                logger.info(f"✓ Improved validation loss by {improvement:.4f}")
                logger.info(f"✓ New best validation loss: {best_loss:.4f}")
                logger.info(f"✓ Validation Avg R²: {val_avg_r2:.4f}")
                logger.info("✓" * 40)
                logger.info("")

        logger.info("="*80)
        logger.info("STAGE 1 TRAINING COMPLETE")
        logger.info(f"Best Validation Loss: {best_loss:.4f}")
        logger.info("="*80)