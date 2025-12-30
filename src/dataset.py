# dataset.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import logging
from torchvision import transforms

# Core species identified from dataset analysis
# We maintain unique categories for specific clover types as recorded.
CORE_SPECIES = [
    'Clover', 'WhiteClover', 'SubcloverDalkeith', 'SubcloverLosa',
    'Ryegrass', 'Phalaris', 'Fescue', 'Lucerne', 
    'Barleygrass', 'Silvergrass', 'Speargrass', 'Bromegrass', 
    'Capeweed', 'Crumbweed'
]

def parse_species_to_soft_labels(species_str, core_species):
    """
    Decomposes strings into a probability vector.
    Maintans unique categories for WhiteClover, SubcloverDalkeith, etc.
    """
    if not isinstance(species_str, str):
        return torch.zeros(len(core_species))
    
    # Normalize
    s = species_str.lower().replace(' ', '')
    
    # Found base components
    found_indices = []
    
    # Specific Mapping Logic for composite/variations
    # We only map truly identical things or spelling variants if needed.
    mapping = {
        'subclover': 'subclover', # Could be a general subclover if it exists
        'whiteclover': 'whiteclover',
        'subcloverdalkeith': 'subcloverdalkeith',
        'subcloverlosa': 'subcloverlosa',
        'barleygrass': 'barleygrass',
        'silvergrass': 'silvergrass',
        'speargrass': 'speargrass',
        'bromegrass': 'bromegrass',
        'capeweed': 'capeweed',
        'crumbweed': 'crumbweed'
    }

    # Split byproduct labels
    parts = s.split('_')
    for p in parts:
        # Check mapping or direct match
        target = mapping.get(p, p)
        # Find index in core_species (case-insensitive)
        for i, core in enumerate(core_species):
            if core.lower() == target:
                found_indices.append(i)
                break
                
    # Create vector
    vec = torch.zeros(len(core_species))
    if found_indices:
        val = 1.0 / len(set(found_indices)) # Avoid double counting same core species
        for idx in set(found_indices):
            vec[idx] = val
    else:
        # Fallback for "Mixed" or unknown
        if 'mixed' in s:
            vec = torch.full((len(core_species),), 1.0 / len(core_species))
            
    return vec

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None, target_cols=None, aux_cols=None, is_test=False):
        """
        Args:
            df: Dataframe containing image paths and targets
            transform: Albumentations or torchvision transforms
            target_cols: List of biomass target columns
            aux_cols: List of auxiliary features (NDVI, Height, etc.)
            is_test: If True, only return images and sample_ids
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols or ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        # Updated Aux Columns based on plan
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul']
        self.is_test = is_test
        
        # Species mapping
        self.core_species = CORE_SPECIES
        self.n_species = len(self.core_species)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Apply transforms
        # self.transform should include RandomResizedCrop(scale=(0.5, 1.0)) for training!
        # This acts as our "Random Tile Selector"
        if self.transform:
            image = self.transform(image)
        else:
            print("No transform provided. Using default transform.")
            raise ValueError("No transform provided. Please provide a transform.")
            
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
        
        # Species Soft-Label Vector
        species_vec = parse_species_to_soft_labels(row['Species'], self.core_species)
                
        return {
            'image': image,
            'targets': targets,
            'aux_feats': aux_feats,
            'species_id': species_vec, # Now a probability vector
            'sample_id': row['sample_id'],
            'is_mosaic': False
        }


def get_species_mapping(df):
    species_list = sorted(df['Species'].unique().tolist())
    return {s: i for i, s in enumerate(species_list)}


class MosaicDataset(Dataset):
    """
    Wraps BiomassDataset to provide Mosaic Augmentation (4-image tiling).
    Averages biomass, aux features, and soft species labels.
    """
    def __init__(self, dataset, prob=0.75):
        from configs import IMAGE_HEIGHT, IMAGE_WIDTH
        self.dataset = dataset
        self.prob = prob
        self.indices = list(range(len(dataset)))
        self.h = IMAGE_HEIGHT
        self.w = IMAGE_WIDTH

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
        mosaic_img = torch.zeros((3, self.h, self.w), dtype=torch.float32)
        
        # Coordinates for 2x2 grid
        h_half, w_half = self.h // 2, self.w // 2
        coords = [(0, 0), (0, w_half), (h_half, 0), (h_half, w_half)] # TL, TR, BL, BR
        
        targets_list = []
        aux_list = []
        species_list = []
        
        sample_id = samples[0]['sample_id'] # Use primary sample ID

        for i, sample in enumerate(samples):
            img = sample['image']
            
            # Resize tile to 1/4 area
            img_small = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(h_half, w_half), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            y, x = coords[i]
            mosaic_img[:, y:y+h_half, x:x+w_half] = img_small
            
            targets_list.append(sample['targets'])
            aux_list.append(sample['aux_feats'])
            species_list.append(sample['species_id'])
            
        # Average Continuous Targets
        mean_targets = torch.stack(targets_list).mean(dim=0)
        mean_aux = torch.stack(aux_list).mean(dim=0)
        
        # Average Soft Species labels
        mean_species = torch.stack(species_list).mean(dim=0)
        
        return {
            'image': mosaic_img,
            'targets': mean_targets,
            'aux_feats': mean_aux, 
            'species_id': mean_species,
            'sample_id': sample_id,
            'is_mosaic': True
        }

