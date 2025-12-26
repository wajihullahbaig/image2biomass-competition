# models.py
import torch
import torch.nn as nn
import timm
from configs import BACKBONE, FUSION_DIM, IMAGE_SIZE, BACKBONE_FREEZE_FRACTION


class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_aux=2, num_species=11, pretrained=True):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
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
        
        # Species Head
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_species)
        )
        
        # 4. Month Head (Cyclical Regression) -seasonal cycles (sin/cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Sin, Cos
        )
        
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux(2) + Species(Probabilities) + Month(2)
        input_dim = self.backbone_dim + num_aux + num_species + 2
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, FUSION_DIM), # +2 for Month Sin/Cos
            nn.LayerNorm(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # OUTPUT: [Clover, Dead, Green] ONLY
            nn.Softplus() # Ensures positive outputs
        )

        self._init_biomass_head()
        

    def _init_biomass_head(self):
        """
        FIX: Initialize the final regression layer to output very small values close to 0.
        This prevents massive loss at epoch 0.
        """
        # Initialize the last Linear layer of biomass_head
        last_layer = self.biomass_head[-2] 
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)
        # Bias = -3.0 ensures Softplus(-3.0) is approx 0.05, close to mean biomass
        nn.init.constant_(last_layer.bias, -3.0)

    def freeze_backbone(self, freeze_fraction=BACKBONE_FREEZE_FRACTION):
        params = list(self.backbone.parameters())
        num_to_freeze = int(len(params) * freeze_fraction)
        for i, param in enumerate(params):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x):
        img_feats = self.backbone(x)
        
        # Heads
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        month_logits = self.month_head(img_feats)
        aux_out = self.aux_head(img_feats) 
        
        # Fusion
        combined_feats = torch.cat([img_feats, aux_out, species_probs, month_logits], dim=1)
        
        # Physics Head
        # Output is KG.
        raw_components = self.biomass_head(combined_feats) # (B, 3)
        
        raw_c = raw_components[:, 0:1]
        raw_d = raw_components[:, 1:2]
        raw_g = raw_components[:, 2:3]
        
        # Linear aggregates
        raw_total = raw_c + raw_d + raw_g
        raw_gdm   = raw_c + raw_g
        
        # Stack: C, D, G, Total, GDM
        biomass_out = torch.cat([raw_c, raw_d, raw_g, raw_total, raw_gdm], dim=1)
        
        return biomass_out, aux_out, species_logits, month_logits

def initialize_weights(model):
    # General init for other layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Skip the specific initialization we did for biomass head
            pass 
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)