# dataset.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import logging
from torchvision import transforms

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import logging
from torchvision import transforms

# Local Imports (Ensure common.py has CORE_SPECIES)
from common import CORE_SPECIES

class BiomassDataset(Dataset):
    def __init__(self, df, transform=None, target_cols=None, aux_cols=None, is_test=False):
        """
        Updated to read pre-calculated Species Probability Columns.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_cols = target_cols or ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        self.aux_cols = aux_cols or ['Pre_GSHH_NDVI', 'Height_Ave_cm_log', 'Interaction_Mul', 'Height_Clean', 'Height_Clean_Log']
        self.is_test = is_test
        
        self.core_species = CORE_SPECIES
        
        # Pre-calculate column names for species probabilities (from Step 1)
        self.species_cols = [f'Species_{sp}' for sp in self.core_species]
        
        # Validation check to ensure Step 1 was run
        if not all(c in self.df.columns for c in self.species_cols):
            # Fallback for inference or if Step 1 skipped (creates dummy cols)
            for c in self.species_cols:
                if c not in self.df.columns: self.df[c] = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # 1. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # 2. Transforms (Rotation/Crop/Zoom happens here)
        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("No transform provided.")
            
        if self.is_test:
            return {'image': image, 'sample_id': row['sample_id']}
            
        # 3. Targets (Log1p happens inside training loop, here we output Raw Grams)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        # 4. Aux Features
        aux_values = row[self.aux_cols].values
        aux_feats = torch.tensor(np.nan_to_num(aux_values.astype(np.float32)))
        
        # 5. Species Vector (READ DIRECTLY FROM DATAFRAME)
        # We no longer parse strings here. We use the global logic from Step 1.
        species_vec = torch.tensor(row[self.species_cols].values.astype(np.float32))
                
        return {
            'image': image,
            'targets': targets,
            'aux_feats': aux_feats,
            'species_id': species_vec,
            'sample_id': row['sample_id'],
            'is_mixup': False # Flag for logging/debugging
        }

class MixupDataset(Dataset):
    """
    Texture Blender.
    Mathematically constructs 'Composite' species from pure ones.
    Replaces Mosaic for texture-heavy tasks.
    """
    def __init__(self, dataset, prob=0.5, alpha=0.4):
        self.dataset = dataset
        self.prob = prob
        self.alpha = alpha # Beta distribution parameter
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Coin Flip: Do we apply MixUp?
        if np.random.rand() >= self.prob:
            return self.dataset[idx]

        # 2. Select Partner
        idx2 = np.random.choice(self.indices)
        
        # 3. Load Both
        sample1 = self.dataset[idx]
        sample2 = self.dataset[idx2]
        
        # 4. Sample Lambda (Mixing Ratio) from Beta Dist
        # Result is usually near 0 or 1, or 0.5 depending on alpha. 
        # alpha=0.4 encourages mixing but keeps one dominant.
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 5. Mix Images (Pixel-wise Linear Interpolation)
        # This creates the "Ghost Texture"
        img1 = sample1['image']
        img2 = sample2['image']
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # 6. Mix Targets (Linear Interpolation)
        # 100g + 50g -> 0.5*100 + 0.5*50 = 75g (Correct density logic)
        mixed_targets = lam * sample1['targets'] + (1 - lam) * sample2['targets']
        
        # 7. Mix Aux & Species
        mixed_aux = lam * sample1['aux_feats'] + (1 - lam) * sample2['aux_feats']
        mixed_species = lam * sample1['species_id'] + (1 - lam) * sample2['species_id']
        
        return {
            'image': mixed_img,
            'targets': mixed_targets,
            'aux_feats': mixed_aux,
            'species_id': mixed_species,
            'sample_id': f"{sample1['sample_id']}_MIX_{sample2['sample_id']}",
            'is_mixup': True
        }