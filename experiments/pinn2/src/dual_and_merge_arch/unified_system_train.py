# unified_model.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import timm
from tqdm import tqdm
import joblib

from common import (
    BATCH_SIZE, DEVICE, NUM_EPOCHS, IMAGE_SIZE, LEARNING_RATE, calculate_count_frequency_features,
    get_image_data_transforms, get_season, set_seed, setup_logging,
    print_stratification_stats, SeasonalCurriculumSampler
)

# ============================================================================
# DATASET
# ============================================================================

class UnifiedDataset(Dataset):
    """
    Dataset that uses images + tabular features.
    Tabular features include count/frequency and interaction features.
    """
    def __init__(self, df, image_dir, target_cols, tabular_feature_cols, 
                 transform=None, weight_col='sample_weight', is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.tabular_feature_cols = tabular_feature_cols
        self.transform = transform
        self.weight_col = weight_col
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['image_path'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tabular features (count, frequency, interactions)
        tabular_data = torch.tensor(row[self.tabular_feature_cols].values.astype(np.float32))
        
        # Targets (log-transformed biomass)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        # Sample weight
        weight = torch.tensor(row[self.weight_col], dtype=torch.float32)
        
        if self.is_train:
            # Auxiliary labels for multi-task learning
            species_label = torch.tensor(row['species_encoded'], dtype=torch.long)
            season_label = torch.tensor(row['season_encoded'], dtype=torch.long)
            state_label = torch.tensor(row['state_encoded'], dtype=torch.long)
            ndvi = torch.tensor(row['Pre_GSHH_NDVI'], dtype=torch.float32)
            height = torch.tensor(row['Height_Ave_cm_log'], dtype=torch.float32)
            
            return (image, tabular_data, targets, weight, species_label, season_label, 
                    state_label, ndvi, height)
        else:
            return image, tabular_data, targets, weight


# ============================================================================
# MODEL
# ============================================================================

class UnifiedBiomassPredictor(nn.Module):
    """
    Single end-to-end model: Image + Tabular Features → Biomass Targets
    
    Uses multi-task learning with auxiliary heads during training to improve
    feature learning, but only the biomass head is used at inference.
    """
    def __init__(self, backbone_name='swin_base_patch4_window7_224',
                 tabular_feature_size=0,
                 num_species=50, num_seasons=4, num_states=10,
                 use_auxiliary_heads=True):
        super().__init__()
        
        self.use_auxiliary_heads = use_auxiliary_heads
        
        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        in_features = self.backbone.num_features
        
        # Tabular feature projection (if we have tabular features)
        if tabular_feature_size > 0:
            self.tabular_projection = nn.Sequential(
                nn.Linear(tabular_feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64)
            )
            combined_features = in_features + 64
        else:
            self.tabular_projection = None
            combined_features = in_features
        
        # Main head: Biomass prediction (5 targets)
        self.biomass_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # [Clover, Dead, Green, Total, GDM]
        )
        
        if use_auxiliary_heads:
            # Auxiliary heads help learn better features during training
            # Species classification
            self.species_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_species)
            )
            
            # Season classification
            self.season_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_features, 128),
                nn.ReLU(),
                nn.Linear(128, num_seasons)
            )
            
            # State classification
            self.state_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_features, 128),
                nn.ReLU(),
                nn.Linear(128, num_states)
            )
            
            # NDVI regression
            self.ndvi_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # Height regression
            self.height_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, x, tabular_data=None, return_auxiliary=False):
        # Extract image features
        features = self.backbone(x)
        
        # Concatenate with tabular features if provided
        if tabular_data is not None and self.tabular_projection is not None:
            tab_features = self.tabular_projection(tabular_data)
            features = torch.cat([features, tab_features], dim=1)
        
        # Main prediction: Biomass (log-space, with softplus for non-negativity)
        biomass = F.softplus(self.biomass_head(features))
        
        if return_auxiliary and self.use_auxiliary_heads:
            species = self.species_head(features)
            season = self.season_head(features)
            state = self.state_head(features)
            ndvi = self.ndvi_head(features).squeeze(-1)
            height = self.height_head(features).squeeze(-1)
            
            return biomass, species, season, state, ndvi, height
        
        return biomass


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class UnifiedLoss(nn.Module):
    """
    Multi-task loss combining:
    1. Main biomass loss (weighted MSE + mass balance)
    2. Auxiliary losses (species, season, state, NDVI, height)
    """
    def __init__(self, biomass_weights=None, mass_balance_alpha=0.5,
                 aux_weight=0.1):
        super().__init__()
        
        if biomass_weights is None:
            biomass_weights = [1.0, 1.0, 1.0, 5.0, 2.0]  # Emphasize Total and GDM
        
        self.register_buffer('biomass_weights', 
                            torch.tensor(biomass_weights, dtype=torch.float32).view(1, -1))
        self.mass_balance_alpha = mass_balance_alpha
        self.aux_weight = aux_weight
        
        # Auxiliary loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, sample_weights=None, auxiliary_data=None):
        """
        Args:
            outputs: Tuple of (biomass, species, season, state, ndvi, height) or just biomass
            targets: Ground truth biomass (log-space)
            sample_weights: Per-sample weights
            auxiliary_data: Dict with keys: species_label, season_label, state_label, ndvi, height
        """
        # Unpack outputs
        if isinstance(outputs, tuple):
            biomass_pred, species_pred, season_pred, state_pred, ndvi_pred, height_pred = outputs
            use_aux = True
        else:
            biomass_pred = outputs
            use_aux = False
        
        # 1. Main biomass loss (weighted MSE in log-space)
        squared_error = F.mse_loss(biomass_pred, targets, reduction='none')
        weighted_error = squared_error * self.biomass_weights
        per_sample_biomass_loss = weighted_error.sum(dim=1) / self.biomass_weights.sum()
        
        # 2. Mass balance penalty (in real space)
        pred_real = torch.expm1(biomass_pred.clamp(max=15))
        clover, dead, green = pred_real[:, 0], pred_real[:, 1], pred_real[:, 2]
        total, gdm = pred_real[:, 3], pred_real[:, 4]
        
        total_balance = torch.log1p(F.l1_loss(total, clover + dead + green, reduction='none'))
        gdm_balance = torch.log1p(F.l1_loss(gdm, clover + green, reduction='none'))
        
        per_sample_loss = per_sample_biomass_loss + self.mass_balance_alpha * (total_balance + gdm_balance)
        
        # Apply sample weights
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        
        main_loss = per_sample_loss.mean()
        
        # 3. Auxiliary losses
        if use_aux and auxiliary_data is not None:
            species_loss = self.ce_loss(species_pred, auxiliary_data['species_label'])
            season_loss = self.ce_loss(season_pred, auxiliary_data['season_label'])
            state_loss = self.ce_loss(state_pred, auxiliary_data['state_label'])
            ndvi_loss = self.mse_loss(ndvi_pred, auxiliary_data['ndvi'])
            height_loss = self.mse_loss(height_pred, auxiliary_data['height'])
            
            aux_loss = (species_loss + season_loss + state_loss + ndvi_loss + height_loss) / 5
            total_loss = main_loss + self.aux_weight * aux_loss
            
            return total_loss, main_loss, aux_loss
        
        return main_loss, main_loss, torch.tensor(0.0).to(main_loss.device)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_data(df_long, logger=None):
    """Prepare data from long format to wide format with feature engineering."""
    if logger:
        logger.info("Preparing data...")
    
    # Pivot to wide format
    df_wide = df_long.pivot_table(
        index=['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()
    
    # Date features
    df_wide['Sampling_Date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['month'] = df_wide['Sampling_Date'].dt.month
    df_wide['season'] = df_wide['month'].apply(get_season)
    df_wide = df_wide.drop('Sampling_Date', axis=1)
    
    # Log-transform height for auxiliary task
    df_wide['Height_Ave_cm_log'] = np.log1p(df_wide['Height_Ave_cm'])
    
    # Impute missing values
    for col in ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Height_Ave_cm_log']:
        if df_wide[col].isnull().any():
            df_wide[col] = df_wide[col].fillna(df_wide[col].median())
    
    if logger:
        logger.info(f"Data prepared. Shape: {df_wide.shape}")
    
    return df_wide


def encode_categorical(train_df, val_df, logger=None):
    """Encode categorical variables and add count/frequency features WITHOUT LEAKAGE."""
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    for col in ['Species', 'season', 'State']:
        le = LabelEncoder()
        train_df[f'{col.lower()}_encoded'] = le.fit_transform(train_df[col])
        
        # Handle unseen categories in validation
        val_df[f'{col.lower()}_encoded'] = val_df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        # Replace -1 with most frequent class
        most_frequent = train_df[f'{col.lower()}_encoded'].mode()[0]
        val_df[f'{col.lower()}_encoded'] = val_df[f'{col.lower()}_encoded'].replace(-1, most_frequent)
        
        encoders[col] = le
        
        if logger:
            logger.info(f"Encoded {col}: {len(le.classes_)} classes")
    
    # ADD COUNT/FREQUENCY FEATURES (NO LEAKAGE)
    train_df, val_df, count_features = calculate_count_frequency_features(
        train_df=train_df,
        val_df=val_df,
        group_col='Species',
        local_group_col='season',
        logger=logger
    )
    
    # ADD NDVI/HEIGHT INTERACTION FEATURES
    train_df['NDVI_Height_MUL'] = train_df['Pre_GSHH_NDVI'] * train_df['Height_Ave_cm_log']
    train_df['NDVI_Height_ADD'] = train_df['Pre_GSHH_NDVI'] + train_df['Height_Ave_cm_log']
    train_df['NDVI_Height_Ratio'] = train_df['Pre_GSHH_NDVI'] / (train_df['Height_Ave_cm_log'] + 1e-5)
    
    val_df['NDVI_Height_MUL'] = val_df['Pre_GSHH_NDVI'] * val_df['Height_Ave_cm_log']
    val_df['NDVI_Height_ADD'] = val_df['Pre_GSHH_NDVI'] + val_df['Height_Ave_cm_log']
    val_df['NDVI_Height_Ratio'] = val_df['Pre_GSHH_NDVI'] / (val_df['Height_Ave_cm_log'] + 1e-5)
    
    interaction_features = ['NDVI_Height_MUL', 'NDVI_Height_ADD', 'NDVI_Height_Ratio']
    
    if logger:
        logger.info(f"Added interaction features: {interaction_features}")
    
    # Store feature names for later use
    encoders['count_features'] = count_features
    encoders['interaction_features'] = interaction_features
    
    return train_df, val_df, encoders


def enforce_physical_constraints(predictions_real_scale):
    """Post-process predictions to enforce mass balance."""
    preds = np.maximum(predictions_real_scale, 0)
    clover, dead, green = preds[:, 0], preds[:, 1], preds[:, 2]
    preds[:, 3] = clover + dead + green  # Total
    preds[:, 4] = clover + green  # GDM
    return preds


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_unified_model():
    logger = setup_logging(file_name_part="unified_biomass_training")
    
    logger.info("="*80)
    logger.info("UNIFIED END-TO-END BIOMASS PREDICTION MODEL")
    logger.info("="*80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Image Size: {IMAGE_SIZE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info("="*80)
    
    set_seed(logger=logger)
    
    # Load data
    logger.info("Loading train.csv...")
    df_long = pd.read_csv('./train.csv')
    df_wide = prepare_data(df_long, logger=logger)
    
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Impute targets
    logger.info("Imputing missing target values...")
    for col in target_cols:
        if df_wide[col].isnull().any():
            median_val = df_wide[col].median()
            df_wide[col] = df_wide[col].fillna(median_val)
            logger.info(f"  {col}: filled {df_wide[col].isnull().sum()} values with median {median_val:.2f}")
    
    # Log-transform targets
    logger.info("Applying log1p to targets...")
    for col in target_cols:
        df_wide[col] = np.log1p(df_wide[col])
    
    # Train-val split
    logger.info("Performing stratified split...")
    train_df, val_df = train_test_split(
        df_wide, test_size=0.2, random_state=42, stratify=df_wide['season']
    )
    print_stratification_stats(df_wide, train_df, val_df, 'season', logger=logger)
    
    # Encode categorical variables and add count/frequency features
    train_df, val_df, encoders = encode_categorical(train_df, val_df, logger=logger)
    
    # Save encoders for inference
    joblib.dump(encoders, 'unified_model_encoders.pkl')
    
    # Define tabular feature columns
    count_features = encoders['count_features']
    interaction_features = encoders['interaction_features']
    tabular_feature_cols = count_features + interaction_features
    
    logger.info(f"Tabular features ({len(tabular_feature_cols)}): {tabular_feature_cols}")
    
    # Calculate sample weights
    from common import calculate_sample_weights
    train_df, weight_col = calculate_sample_weights(
        train_df, group_col='season', weight_col='sample_weight', logger=logger
    )
    val_df[weight_col] = 1.0
    
    # Create datasets
    train_transform, val_transform = get_image_data_transforms()
    
    train_dataset = UnifiedDataset(
        train_df, 'train', target_cols, tabular_feature_cols, 
        train_transform, weight_col, is_train=True
    )
    val_dataset = UnifiedDataset(
        val_df, 'train', target_cols, tabular_feature_cols,
        val_transform, weight_col, is_train=True
    )
    
    # Create samplers and loaders
    train_sampler = SeasonalCurriculumSampler(train_df, shuffle_within_season=False, seed=42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
        num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    logger.info(f"Training samples: {len(train_df)}, batches: {len(train_loader)}")
    logger.info(f"Validation samples: {len(val_df)}, batches: {len(val_loader)}")
    
    # Initialize model
    num_species = len(encoders['Species'].classes_)
    num_seasons = len(encoders['season'].classes_)
    num_states = len(encoders['State'].classes_)
    
    model = UnifiedBiomassPredictor(
        backbone_name='swin_base_patch4_window7_224',
        tabular_feature_size=len(tabular_feature_cols),
        num_species=num_species,
        num_seasons=num_seasons,
        num_states=num_states,
        use_auxiliary_heads=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = UnifiedLoss(
        biomass_weights=[1.0, 1.0, 1.0, 5.0, 2.0],
        mass_balance_alpha=0.5,
        aux_weight=0.1
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_r2 = -float('inf')
    official_r2_weights = [0.1, 0.1, 0.1, 0.5, 0.2]
    
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    for epoch in range(NUM_EPOCHS):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_main_loss = 0.0
        train_aux_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_data in pbar:
            images = batch_data[0].to(DEVICE)
            tabular_data = batch_data[1].to(DEVICE)
            targets = batch_data[2].to(DEVICE)
            weights = batch_data[3].to(DEVICE)
            
            auxiliary_data = {
                'species_label': batch_data[4].to(DEVICE),
                'season_label': batch_data[5].to(DEVICE),
                'state_label': batch_data[6].to(DEVICE),
                'ndvi': batch_data[7].to(DEVICE),
                'height': batch_data[8].to(DEVICE)
            }
            
            optimizer.zero_grad()
            
            outputs = model(images, tabular_data, return_auxiliary=True)
            loss, main_loss, aux_loss = criterion(outputs, targets, weights, auxiliary_data)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_main_loss += main_loss.item()
            train_aux_loss += aux_loss.item()
            
            train_preds.append(outputs[0].detach().cpu().numpy())
            train_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_main = train_main_loss / len(train_loader)
        avg_train_aux = train_aux_loss / len(train_loader)
        
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_preds_real = np.expm1(train_preds)
        train_targets_real = np.expm1(train_targets)
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
            for batch_data in val_bar:
                images = batch_data[0].to(DEVICE)
                tabular_data = batch_data[1].to(DEVICE)
                targets = batch_data[2].to(DEVICE)
                weights = batch_data[3].to(DEVICE)
                
                auxiliary_data = {
                    'species_label': batch_data[4].to(DEVICE),
                    'season_label': batch_data[5].to(DEVICE),
                    'state_label': batch_data[6].to(DEVICE),
                    'ndvi': batch_data[7].to(DEVICE),
                    'height': batch_data[8].to(DEVICE)
                }
                
                outputs = model(images, tabular_data, return_auxiliary=True)
                loss, _, _ = criterion(outputs, targets, weights, auxiliary_data)
                
                val_loss += loss.item()
                val_preds.append(outputs[0].cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_preds_real = np.expm1(val_preds)
        val_targets_real = np.expm1(val_targets)
        
        # Enforce physical constraints
        val_preds_phys = enforce_physical_constraints(val_preds_real)
        
        # Calculate metrics
        def weighted_r2(y_true, y_pred, weights):
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            weights_matrix = np.tile(np.array(weights).reshape(1, -1), (y_true.shape[0], 1))
            return r2_score(y_true_flat, y_pred_flat, sample_weight=weights_matrix.flatten())
        
        train_r2 = weighted_r2(train_targets_real, train_preds_real, official_r2_weights)
        val_r2 = weighted_r2(val_targets_real, val_preds_phys, official_r2_weights)
        
        # Per-target R²
        val_r2_per_target = [r2_score(val_targets_real[:, i], val_preds_phys[:, i]) 
                             for i in range(5)]
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ========== LOGGING ==========
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        logger.info("-"*80)
        logger.info(f"LR: {current_lr:.6f}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} (Main: {avg_train_main:.4f}, Aux: {avg_train_aux:.4f})")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Train R² (weighted): {train_r2:.4f}")
        logger.info(f"Val R² (weighted): {val_r2:.4f}")
        logger.info("-"*80)
        logger.info("Per-Target Validation R²:")
        for i, col in enumerate(target_cols):
            logger.info(f"  {col:15s}: {val_r2_per_target[i]:.4f}")
        logger.info("="*80)
        
        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'val_r2_per_target': val_r2_per_target
            }, 'unified_model_best.pth')
            logger.info(f"✓ NEW BEST MODEL! R²: {best_val_r2:.4f}")
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Validation R²: {best_val_r2:.4f}")
    logger.info("="*80)


if __name__ == '__main__':
    train_unified_model()