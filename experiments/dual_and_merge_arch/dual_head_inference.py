# dual_head_inference.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import joblib

from dual_head_unified_train import DualHeadBiomassPredictor, enforce_physical_constraints

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
    print("DUAL-HEAD MODEL INFERENCE (2-PASS)")
    print("="*80)
    
    # Load test data
    print("Loading test.csv...")
    test_df = pd.read_csv('test.csv')
    
    # Get unique images
    unique_samples = test_df.drop_duplicates(subset=['sample_id']).copy()
    print(f"Total test samples: {len(unique_samples)}")
    
    # Load encoders
    print("Loading encoders...")
    encoders = joblib.load('dual_head_encoders.pkl')
    
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
    
    # Load model checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load('dual_head_model_best.pth', map_location=DEVICE)
    model_config = checkpoint['config']
    
    print(f"\nModel Configuration:")
    print(f"  Auxiliary Backbone: {model_config['auxiliary_backbone']}")
    print(f"  Biomass Backbone: {model_config['biomass_backbone']}")
    print(f"  Best Validation R²: {checkpoint['val_r2']:.4f}")
    
    # Initialize model
    model = DualHeadBiomassPredictor(
        auxiliary_backbone=model_config['auxiliary_backbone'],
        biomass_backbone=model_config['biomass_backbone'],
        tabular_feature_size=model_config['tabular_feature_size'],
        num_species=model_config['num_species'],
        num_seasons=model_config['num_seasons'],
        num_states=model_config['num_states']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # ========================================================================
    # PASS 1: Predict auxiliary features using HEAD 1 (EfficientNet)
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 1: Predicting auxiliary features (HEAD 1: EfficientNet)...")
    print("="*80)
    
    predicted_species = []
    predicted_seasons = []
    predicted_states = []
    predicted_ndvi = []
    predicted_height_log = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, img_paths in tqdm(test_loader, desc="Pass 1 - Auxiliary"):
            images = images.to(DEVICE)
            
            # Forward pass through auxiliary head only
            # We pass tabular_data=None since we don't have it yet
            _, species_logits, season_logits, state_logits, ndvi, height = model(
                images, tabular_data=None, return_auxiliary=True
            )
            
            # Get predictions
            species_idx = torch.argmax(species_logits, dim=1).cpu().numpy()
            season_idx = torch.argmax(season_logits, dim=1).cpu().numpy()
            state_idx = torch.argmax(state_logits, dim=1).cpu().numpy()
            
            predicted_species.extend(encoders['Species'].inverse_transform(species_idx))
            predicted_seasons.extend(encoders['season'].inverse_transform(season_idx))
            predicted_states.extend(encoders['State'].inverse_transform(state_idx))
            predicted_ndvi.extend(ndvi.cpu().numpy())
            predicted_height_log.extend(height.cpu().numpy())
            all_image_paths.extend(img_paths)
    
    # Create dataframe with predicted auxiliary features
    pred_df = pd.DataFrame({
        'image_path': all_image_paths,
        'Species': predicted_species,
        'season': predicted_seasons,
        'State': predicted_states,
        'Pre_GSHH_NDVI': predicted_ndvi,
        'Height_Ave_cm_log': predicted_height_log
    })
    
    print(f"\n✓ Pass 1 complete. Predicted {len(pred_df)} samples")
    print(f"\nPredicted distributions:")
    print(f"  Species (top 5):")
    for species, count in pred_df['Species'].value_counts().head(5).items():
        print(f"    {species}: {count}")
    print(f"  Season:")
    for season, count in pred_df['season'].value_counts().items():
        print(f"    {season}: {count}")
    print(f"  State:")
    for state, count in pred_df['State'].value_counts().items():
        print(f"    {state}: {count}")
    print(f"  NDVI: {pred_df['Pre_GSHH_NDVI'].min():.3f} to {pred_df['Pre_GSHH_NDVI'].max():.3f}")
    print(f"  Height (log): {pred_df['Height_Ave_cm_log'].min():.3f} to {pred_df['Height_Ave_cm_log'].max():.3f}")
    
    # ========================================================================
    # PASS 1.5: Engineer tabular features from auxiliary predictions
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 1.5: Engineering tabular features...")
    print("="*80)
    
    # Count/Frequency features (computed from predictions)
    # NOTE: Ideally, use training statistics saved during training
    # For simplicity, we compute from test predictions here
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
    
    # Interaction features (NDVI × Height, etc.)
    pred_df['NDVI_Height_MUL'] = pred_df['Pre_GSHH_NDVI'] * pred_df['Height_Ave_cm_log']
    pred_df['NDVI_Height_ADD'] = pred_df['Pre_GSHH_NDVI'] + pred_df['Height_Ave_cm_log']
    pred_df['NDVI_Height_Ratio'] = pred_df['Pre_GSHH_NDVI'] / (pred_df['Height_Ave_cm_log'] + 1e-5)
    
    # Define tabular feature columns (must match training order)
    count_features = encoders.get('count_features', [])
    interaction_features = encoders.get('interaction_features', [])
    tabular_feature_cols = count_features + interaction_features
    
    print(f"✓ Tabular features engineered: {tabular_feature_cols}")
    
    # Prepare tabular features array
    tabular_features_array = pred_df[tabular_feature_cols].values.astype(np.float32)
    
    print(f"  Feature shape: {tabular_features_array.shape}")
    print(f"  Feature stats:")
    for i, col in enumerate(tabular_feature_cols):
        print(f"    {col}: {tabular_features_array[:, i].min():.3f} to {tabular_features_array[:, i].max():.3f}")
    
    # ========================================================================
    # PASS 2: Predict biomass using HEAD 2 (Swin Transformer)
    # ========================================================================
    print("\n" + "="*80)
    print("PASS 2: Predicting biomass (HEAD 2: Swin Transformer)...")
    print("="*80)
    
    class Pass2Dataset(Dataset):
        """Dataset that provides both image and tabular features."""
        def __init__(self, img_paths, tabular_data, image_dir, transform):
            self.img_paths = img_paths
            self.tabular_data = tabular_data
            self.image_dir = image_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.img_paths)
        
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
        for img, tab in tqdm(pass2_loader, desc="Pass 2 - Biomass"):
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            
            # Forward pass: HEAD 2 uses both image and tabular features
            preds_log = model(img, tab, return_auxiliary=False)
            
            # Convert from log-space to real space
            preds_real = torch.expm1(preds_log).cpu().numpy()
            
            all_predictions.append(preds_real)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"\n✓ Pass 2 complete. Generated {len(all_predictions)} predictions")
    
    # ========================================================================
    # POST-PROCESSING: Enforce physical constraints
    # ========================================================================
    print("\n" + "="*80)
    print("POST-PROCESSING: Enforcing physical constraints...")
    print("="*80)
    
    print("Before constraints:")
    print(f"  Clover: {all_predictions[:, 0].min():.2f} to {all_predictions[:, 0].max():.2f}")
    print(f"  Dead: {all_predictions[:, 1].min():.2f} to {all_predictions[:, 1].max():.2f}")
    print(f"  Green: {all_predictions[:, 2].min():.2f} to {all_predictions[:, 2].max():.2f}")
    print(f"  Total: {all_predictions[:, 3].min():.2f} to {all_predictions[:, 3].max():.2f}")
    print(f"  GDM: {all_predictions[:, 4].min():.2f} to {all_predictions[:, 4].max():.2f}")
    
    all_predictions = enforce_physical_constraints(all_predictions)
    
    print("\nAfter constraints:")
    print(f"  Clover: {all_predictions[:, 0].min():.2f} to {all_predictions[:, 0].max():.2f}")
    print(f"  Dead: {all_predictions[:, 1].min():.2f} to {all_predictions[:, 1].max():.2f}")
    print(f"  Green: {all_predictions[:, 2].min():.2f} to {all_predictions[:, 2].max():.2f}")
    print(f"  Total: {all_predictions[:, 3].min():.2f} to {all_predictions[:, 3].max():.2f}")
    print(f"  GDM: {all_predictions[:, 4].min():.2f} to {all_predictions[:, 4].max():.2f}")
    
    # Verify mass balance
    clover, dead, green = all_predictions[:, 0], all_predictions[:, 1], all_predictions[:, 2]
    total_calc = clover + dead + green
    gdm_calc = clover + green
    total_error = np.abs(all_predictions[:, 3] - total_calc).mean()
    gdm_error = np.abs(all_predictions[:, 4] - gdm_calc).mean()
    
    print(f"\nMass balance verification:")
    print(f"  Total balance error (MAE): {total_error:.6f} (should be ~0.0)")
    print(f"  GDM balance error (MAE): {gdm_error:.6f} (should be ~0.0)")
    
    # ========================================================================
    # FORMAT SUBMISSION
    # ========================================================================
    print("\n" + "="*80)
    print("FORMATTING SUBMISSION...")
    print("="*80)
    
    # Create prediction map: sample_id -> predictions
    prediction_map = {}
    for img_path, pred in zip(pred_df['image_path'].values, all_predictions):
        # Extract sample_id from image path
        sample_id = img_path.split('/')[-1].replace('.jpg', '')
        prediction_map[sample_id] = pred
    
    # Target columns in order
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Build submission rows
    submission_rows = []
    for _, row in test_df.iterrows():
        # Parse sample_id (format: "ID1025234388__Dry_Clover_g")
        sample_id_full = row['sample_id']
        sample_id = sample_id_full.split('__')[0]
        target_name = row['target_name']
        
        # Get prediction for this sample
        if sample_id in prediction_map:
            pred_vec = prediction_map[sample_id]
            target_idx = target_cols.index(target_name)
            value = pred_vec[target_idx]
        else:
            print(f"WARNING: No prediction found for {sample_id}")
            value = 0.0
        
        # Ensure non-negative
        value = max(0.0, value)
        
        submission_rows.append({
            'sample_id': sample_id_full,
            'target': value
        })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Verify row count
    sample_sub = pd.read_csv('sample_submission.csv')
    if len(submission_df) != len(sample_sub):
        print(f"WARNING: Submission has {len(submission_df)} rows, expected {len(sample_sub)}")
    else:
        print(f"✓ Row count matches sample submission: {len(submission_df)}")
    
    # Save
    submission_df.to_csv('submission.csv', index=False)
    
    print("="*80)
    print("✓ INFERENCE COMPLETE")
    print("="*80)
    print(f"  Submission saved to: submission.csv")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"  Unique samples: {len(prediction_map)}")
    print("="*80)
    
    # Show sample predictions
    print("\nSample predictions (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))
    
    # Summary statistics
    print("\nSubmission statistics:")
    print(submission_df['target'].describe())


if __name__ == '__main__':
    run_inference()