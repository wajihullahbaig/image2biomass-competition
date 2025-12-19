# dataset.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import logging

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
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log']
        self.is_test = is_test
        
        # Species mapping
        self.species_list = sorted(self.df['Species'].unique().tolist())
        self.species_to_id = {s: i for i, s in enumerate(self.species_list)}
        self.n_species = len(self.species_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to black image if load fails (should not happen in prepared data)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return {
                'image': image,
                'sample_id': row['sample_id']
            }
            
        # Biomass targets - Apply log1p for stability
        targets_raw = row[self.target_cols].values.astype(np.float32)
        targets = torch.tensor(np.log1p(targets_raw))
        
        # Aux features (NDVI, Height)
        # Convert to numeric first to avoid object-dtype fillna warnings
        aux_values = row[self.aux_cols].values
        aux_values = np.nan_to_num(aux_values.astype(np.float32), nan=0.0)
        aux_feats = torch.tensor(aux_values)
        
        # Species One-Hot or Label
        species_id = self.species_to_id.get(row['Species'], 0)
        
        return {
            'image': image,
            'targets': targets,
                'aux_feats': aux_feats,
            'species_id': species_id,
            'sample_id': row['sample_id']
        }

def get_species_mapping(df):
    species_list = sorted(df['Species'].unique().tolist())
    return {s: i for i, s in enumerate(species_list)}
