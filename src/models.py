# models.py
import torch
import torch.nn as nn
import timm
from configs import BACKBONE, FUSION_DIM, IMAGE_SIZE, BACKBONE_FREEZE_FRACTION, AUX_FEAT_WEIGHT, SPECIES_FEAT_WEIGHT, MONTH_FEAT_WEIGHT, BIOMASS_FEAT_WEIGHT

class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_targets=5, num_aux=2, num_species=11, num_months=12, pretrained=True):
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
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_aux)
        )
        
        # Multi-task heads 
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_species)
        )
        
        # 4. Month Head (Cyclical Regression)
        # Forces backbone to learn seasonal cycles (sin/cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        # 5. Biomass Head
        fusion_dim = FUSION_DIM
        
        self.biomass_head = nn.Sequential(
            nn.Linear(self.backbone_dim + num_aux + num_species + 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softplus()
        )

    def freeze_backbone(self, freeze_fraction=BACKBONE_FREEZE_FRACTION):
        """Freezes a fraction of the backbone layers."""
        params = list(self.backbone.parameters())
        num_to_freeze = int(len(params) * freeze_fraction)
        
        for i, param in enumerate(params):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        if freeze_fraction > 0.9:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()

    def forward(self, x):
        # Extract features from image
        img_feats = self.backbone(x) # (B, backbone_dim)
        
        # Predict species (categorical logits)
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        # Predict month (continous logits)
        month_logits = self.month_head(img_feats)
        
        # Predict auxiliary features (NDVI, Height)
        aux_out = self.aux_head(img_feats) 
        #aux_out = torch.clamp(aux_out, 0.0, 10.0)
            
        # --- FUSION OF ALL FEATURES ---
        combined_feats = torch.cat([
            img_feats, 
            aux_out, 
            species_probs,
            month_logits
        ], dim=1)
        
        # --- PHYSICS-INFORMED HEAD ---
        # 1. Predict raw KG Components (Clover, Dead, Green)
        # Softplus ensures non-negative mass
        raw_components_kg = self.biomass_head(combined_feats) # (B, 3)
        
        # 2. Extract Components 
        raw_c = raw_components_kg[:, 0:1]
        raw_d = raw_components_kg[:, 1:2]
        raw_g = raw_components_kg[:, 2:3]
        
        # 3. Reconstruct Aggregates (Linear Sum)
        raw_total = raw_c + raw_d + raw_g
        raw_gdm   = raw_c + raw_g
        
        # 4. Concatenate for Loss (Order: C, D, G, Total, GDM)
        # All in KG scale
        biomass_out = torch.cat([
            raw_c, 
            raw_d, 
            raw_g, 
            raw_total, 
            raw_gdm
        ], dim=1)
        
        return biomass_out, aux_out, species_logits, month_logits

# Weight Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
