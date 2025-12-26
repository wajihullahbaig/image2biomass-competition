# dataset.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import logging
from torchvision import transforms

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None, target_cols=None, aux_cols=None, is_test=False, species_to_id=None):
        """
        Args:
            df: Dataframe containing image paths and targets
            transform: Albumentations or torchvision transforms
            target_cols: List of biomass target columns
            aux_cols: List of auxiliary features (NDVI, Height, etc.)
            is_test: If True, only return images and sample_ids
            species_to_id: Optional dict mapping species names to IDs
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols or ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        # Updated Aux Columns based on plan
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul']
        self.is_test = is_test
        
        # Species mapping
        if species_to_id:
            self.species_to_id = species_to_id
            self.species_list = sorted(list(species_to_id.keys()))
        else:
            self.species_list = sorted(self.df['Species'].unique().tolist())
            self.species_to_id = {s: i for i, s in enumerate(self.species_list)}
        self.n_species = len(self.species_to_id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # Fallback
            
        # Apply transforms
        # self.transform should include RandomResizedCrop(scale=(0.5, 1.0)) for training!
        # This acts as our "Random Tile Selector"
        if self.transform:
            image = self.transform(image)
        else:
            # Basic fallback
            to_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = to_tensor(image)
            
        if self.is_test:
            return {
                'image': image,
                'sample_id': row['sample_id']
            }
            
        # Biomass targets - Raw scale (grams)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        # Aux features (NDVI, Height)
        aux_values = row[self.aux_cols].values
        aux_values = np.nan_to_num(aux_values.astype(np.float32), nan=0.0)
        aux_feats = torch.tensor(aux_values)
        
        # Species One-Hot or Label
        species_id = self.species_to_id.get(row['Species'], 0)
        
        # Cyclical Month Encoding for Phenology Regularization
        try:
            if hasattr(row['Sampling_Date'], 'month'):
                m = row['Sampling_Date'].month
            else:
                m = pd.to_datetime(str(row['Sampling_Date'])).month
            
            # Convert to radians (1-12 range)
            month_rad = 2.0 * np.pi * (m - 1) / 12.0
            month_sin = np.sin(month_rad)
            month_cos = np.cos(month_rad)
        except:
            month_sin, month_cos = 0.0, 1.0 # Default to Jan (rad 0)
            
        return {
            'image': image,
            'targets': targets,
            'aux_feats': aux_feats,
            'species_id': species_id,
            'month_sin_cos': torch.tensor([month_sin, month_cos], dtype=torch.float32),
            'sample_id': row['sample_id'],
            'is_mosaic': False
        }


def get_species_mapping(df):
    species_list = sorted(df['Species'].unique().tolist())
    return {s: i for i, s in enumerate(species_list)}

class MosaicDataset(Dataset):
    """
    Wraps BiomassDataset to provide Mosaic Augmentation (4-image tiling).
    """
    def __init__(self, dataset, prob=0.75, image_size=384):
        self.dataset = dataset
        self.prob = prob
        self.image_size = image_size
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if np.random.rand() >= self.prob:
            sample = self.dataset[idx]
            sample['is_mosaic'] = False
            return sample

        # Select 3 other random indices
        indices = [idx] + np.random.choice(self.indices, 3).tolist()
        
        # Load 4 samples
        samples = [self.dataset[i] for i in indices]
        
        # Prepare Output Containers
        # Image: 4 tiles
        mosaic_img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        
        # Coordinates for 2x2 grid
        s = self.image_size // 2
        coords = [(0, 0), (0, s), (s, 0), (s, s)] # TL, TR, BL, BR
        
        # Targets & Features to Average
        targets_lin_list = []
        aux_feats_list = []
        month_sincos_list = []
        
        # Species Logic: Just pick the first one's species as "Dominant" for classification aux loss
        # Or mixed? For now, let's keep the main index species to avoid confusion in CrossEntropy
        species_id = samples[0]['species_id']
        sample_id = samples[0]['sample_id'] # Keep ID for reference

        for i, sample in enumerate(samples):
            img = sample['image']
            
            # Resize to half size
            transforms_resize = transforms.Resize((s, s))
            img_small = transforms_resize(img)
            
            # Place in grid
            y, x = coords[i]
            mosaic_img[:, y:y+s, x:x+s] = img_small
            
            # Collect Targets (Inverse Log -> Linear)
            # Assuming dataset returns Log1p targets, we expm1 them to average in linear space
            # But wait, looking at BiomassDataset below, it currently returns raw grams/1000 (kg).
            # So we average linear kg directly.
            targets_lin_list.append(sample['targets'])
            
            # Collect Aux Feats
            aux_feats_list.append(sample['aux_feats'])
            month_sincos_list.append(sample['month_sin_cos'])

        # Average Targets (Linear Space)
        mean_targets_lin = torch.stack(targets_lin_list).mean(dim=0)
        
        # Average Aux Feats
        mean_aux = torch.stack(aux_feats_list).mean(dim=0)
        
        # Average Month (Geometric Center on Unit Circle)
        mean_month = torch.stack(month_sincos_list).mean(dim=0)
        mean_month = mean_month / (mean_month.norm() + 1e-8) # Re-normalize
        
        return {
            'image': mosaic_img,
            'targets': mean_targets_lin,
            'aux_feats': mean_aux, 
            'species_id': species_id,
            'month_sin_cos': mean_month,
            'sample_id': sample_id,
            'is_mosaic': True
        }


