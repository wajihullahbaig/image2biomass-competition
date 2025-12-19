# models.py
import torch
import torch.nn as nn
import timm
from configs import BACKBONE_S1, FUSION_DIM, IMAGE_SIZE

class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE_S1, num_targets=5, num_aux=2, num_species=11, pretrained=True):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            self.backbone_dim = self.backbone(dummy_input).shape[1]
            
        # 2. Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_aux)
        )
        
        # 3. Species Head (Categorical)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_species)
        )
        
        # 4. Biomass Head
        # It takes backbone features + predicted aux features + species features
        self.biomass_head = nn.Sequential(
            nn.Linear(self.backbone_dim + num_aux + num_species, FUSION_DIM),
            nn.BatchNorm1d(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets),
            nn.Softplus() # Ensures positive outputs for mass
        )

    def forward(self, x):
        # Extract features from image
        img_feats = self.backbone(x) # (B, backbone_dim)
        
        # Predict species (categorical logits)
        species_logits = self.species_head(img_feats) # (B, num_species)
        
        # Predict auxiliary features (NDVI, Height)
        aux_out = self.aux_head(img_feats) # (B, num_aux)
        
        # Stability Clamp: prevent wild aux predictions from breaking the biomass head 
        aux_out_clamped = torch.clamp(aux_out, -2.0, 10.0)
        
        # Concatenate image features with PREDICTED aux and species features
        combined_feats = torch.cat([img_feats, aux_out_clamped, species_logits], dim=1)
        
        # Predict biomass targets
        biomass_out = self.biomass_head(combined_feats) # (B, num_targets)
        
        return biomass_out, aux_out, species_logits

# Weight Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
