# unified_inference.py
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from unified_system_train import UnifiedBiomassPredictor, enforce_physical_constraints

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32

class TestDataset(Dataset):
    """Simple test dataset that only loads images."""
    def __init__(self, image_paths, image_dir, transform):
        self.image_paths = image_paths
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]


def run_inference():
    print("="*80)
    print("UNIFIED MODEL INFERENCE (2-PASS)")
    print("="*80)
    
    # Load test data
    print("Loading test.csv...")
    test_df = pd.read_csv('test.csv')
    
    # Get unique images (test.csv has 5 rows per image, one per target)
    unique_samples = test_df.drop_duplicates(subset=['sample_id']).copy()
    print(f"Total test samples: {len(unique_samples)}")
    
    # Load encoders
    encoders = joblib.load('unified_model_encoders.pkl')
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    test_dataset = TestDataset(
        unique_samples['image_path'].values,
        'test',
        transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print("Loading model...")
    
    # Get tabular feature size from encoders
    count_features = encoders.get('count_features', [])
    interaction_features = encoders.get('interaction_features', [])
    tabular_feature_size = len(count_features) + len(interaction_features)
    
    model = UnifiedBiomassPredictor(
        backbone_name='swin_base_patch4_window7_224',
        tabular_feature_size=tabular_feature_size,
        num_species=len(encoders['Species'].classes_),
        num_seasons=len(encoders['season'].classes_),
        num_states=len(encoders['State'].classes_),
        use_auxiliary_heads=True  # Need auxiliary heads for PASS 1
    ).to(DEVICE)
    
    checkpoint = torch.load('unified_model_best.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Best validation R²: {checkpoint['val_r2']:.4f}")
    
    # ========================================================================
    # PASS 1: Predict auxiliary features (Species, Season, NDVI, Height)
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 1: Predicting auxiliary features...")
    print("="*80)
    
    predicted_species = []
    predicted_seasons = []
    predicted_ndvi = []
    predicted_height_log = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, img_paths in tqdm(test_loader, desc="Pass 1"):
            images = images.to(DEVICE)
            
            # Forward pass with auxiliary heads (no tabular data yet)
            _, species_logits, season_logits, _, ndvi, height = model(
                images, tabular_data=None, return_auxiliary=True
            )
            
            # Get predictions
            species_idx = torch.argmax(species_logits, dim=1).cpu().numpy()
            season_idx = torch.argmax(season_logits, dim=1).cpu().numpy()
            
            predicted_species.extend(encoders['Species'].inverse_transform(species_idx))
            predicted_seasons.extend(encoders['season'].inverse_transform(season_idx))
            predicted_ndvi.extend(ndvi.cpu().numpy())
            predicted_height_log.extend(height.cpu().numpy())
            all_image_paths.extend(img_paths)
    
    # Create dataframe with predicted auxiliary features
    pred_df = pd.DataFrame({
        'image_path': all_image_paths,
        'Species': predicted_species,
        'season': predicted_seasons,
        'Pre_GSHH_NDVI': predicted_ndvi,
        'Height_Ave_cm_log': predicted_height_log
    })
    
    print(f"\n✓ Pass 1 complete. Predicted {len(pred_df)} samples")
    print(f"  Species distribution: {pred_df['Species'].value_counts().head()}")
    print(f"  Season distribution: {pred_df['season'].value_counts()}")
    
    # ========================================================================
    # FEATURE ENGINEERING: Create count/frequency and interaction features
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 1.5: Engineering tabular features...")
    print("="*80)
    
    # Load training statistics for count/frequency features
    # NOTE: These must be calculated from training set only (no leakage)
    # We'll use the encoders which should have stored the training stats
    
    # For now, compute count/frequency based on predicted species/season
    # In production, you'd load pre-computed stats from training
    species_counts = pred_df['Species'].value_counts()
    species_freq = species_counts / len(pred_df)
    
    pred_df['species_count_global'] = pred_df['Species'].map(np.log1p(species_counts))
    pred_df['species_freq_global'] = pred_df['Species'].map(species_freq)
    
    # Seasonal counts
    seasonal_counts = pred_df.groupby(['season', 'Species']).size()
    pred_df['species_count_season'] = pred_df.apply(
        lambda row: np.log1p(seasonal_counts.get((row['season'], row['Species']), 0)),
        axis=1
    )
    
    season_totals = pred_df.groupby('season').size()
    seasonal_freq = seasonal_counts / seasonal_counts.index.map(lambda x: season_totals[x[0]])
    pred_df['species_freq_season'] = pred_df.apply(
        lambda row: seasonal_freq.get((row['season'], row['Species']), 0),
        axis=1
    )
    
    # Interaction features
    pred_df['NDVI_Height_MUL'] = pred_df['Pre_GSHH_NDVI'] * pred_df['Height_Ave_cm_log']
    pred_df['NDVI_Height_ADD'] = pred_df['Pre_GSHH_NDVI'] + pred_df['Height_Ave_cm_log']
    pred_df['NDVI_Height_Ratio'] = pred_df['Pre_GSHH_NDVI'] / (pred_df['Height_Ave_cm_log'] + 1e-5)
    
    # Prepare tabular features
    tabular_feature_cols = count_features + interaction_features
    tabular_features_array = pred_df[tabular_feature_cols].values.astype(np.float32)
    
    print(f"✓ Tabular features created: {tabular_feature_cols}")
    
    # ========================================================================
    # PASS 2: Predict biomass using image + tabular features
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 2: Predicting biomass...")
    print("="*80)
    
    class Pass2Dataset(Dataset):
        def __init__(self, img_paths, tabular_data, image_dir, transform):
            self.img_paths = img_paths
            self.tabular_data = tabular_data
            self.image_dir = image_dir
            self.transform = transform
        def __len__(self): return len(self.img_paths)
        def __getitem__(self, idx):
            path = os.path.join(self.image_dir, self.img_paths[idx].split('/')[-1])
            img = self.transform(Image.open(path).convert('RGB'))
            tab = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
            return img, tab
    
    pass2_dataset = Pass2Dataset(
        pred_df['image_path'].values,
        tabular_features_array,
        'test',
        transform
    )
    pass2_loader = DataLoader(
        pass2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    all_predictions = []
    with torch.no_grad():
        for img, tab in tqdm(pass2_loader, desc="Pass 2"):
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            
            # Get predictions (log-space)
            preds_log = model(img, tab, return_auxiliary=False)
            
            # Convert to real space
            preds_real = torch.expm1(preds_log).cpu().numpy()
            
            all_predictions.append(preds_real)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Enforce physical constraints
    print("\nEnforcing physical constraints...")
    all_predictions = enforce_physical_constraints(all_predictions)
    
    # Create prediction map
    prediction_map = {}
    for img_path, pred in zip(all_image_paths, all_predictions):
        # Extract sample_id from image path
        sample_id = img_path.split('/')[-1].replace('.jpg', '')
        prediction_map[sample_id] = pred
    
    # Format submission
    print("Formatting submission...")
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    submission_rows = []
    for _, row in test_df.iterrows():
        sample_id = row['sample_id'].split('__')[0]  # e.g., "ID1025234388" from "ID1025234388__Dry_Clover_g"
        target_name = row['target_name']
        
        # Get prediction for this sample
        if sample_id in prediction_map:
            pred_vec = prediction_map[sample_id]
            target_idx = target_cols.index(target_name)
            value = pred_vec[target_idx]
        else:
            print(f"Warning: No prediction found for {sample_id}")
            value = 0.0
        
        # Ensure non-negative
        value = max(0.0, value)
        
        submission_rows.append({
            'sample_id': row['sample_id'],  # Keep original format: "ID1025234388__Dry_Clover_g"
            'target': value
        })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Verify against sample submission
    sample_sub = pd.read_csv('sample_submission.csv')
    if len(submission_df) != len(sample_sub):
        print(f"WARNING: Submission has {len(submission_df)} rows, expected {len(sample_sub)}")
    
    # Save
    submission_df.to_csv('submission.csv', index=False)
    print("="*80)
    print("✓ Submission saved to submission.csv")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"  Unique samples: {len(prediction_map)}")
    print("="*80)
    
    # Show sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10))


if __name__ == '__main__':
    run_inference()