# models.py - FIXED VERSION
import torch
import torch.nn as nn
import timm
from configs import BACKBONE, FUSION_DIM, BACKBONE_FREEZE_FRACTION, IMAGE_HEIGHT, IMAGE_WIDTH


class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_aux=3, num_species=16, pretrained=True):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            feats = self.backbone(dummy_input)
            self.backbone_dim = feats.shape[1]
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
            
        # 2. Auxiliary Head (NDVI, LogHeight, Interaction)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_aux)
        )
        
        # Species Head
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_species)
        )
        
        # 4. Month Head (Cyclical Regression) -seasonal cycles (sin/cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2) # Sin, Cos
        )
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux(3) + Species(Probabilities) + Month(2)
        input_dim = self.backbone_dim + num_aux + num_species + 2
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 4), # OUTPUT: [Log_Clover, Log_Dead, Log_Green, Log_Total]
            # NO SOFTPLUS - we'll use raw outputs and rely on loss to keep them positive
        )

        self._init_biomass_head()
        
    def _init_biomass_head(self):
        """
        CRITICAL FIX: Initialize for TINY targets (mean ~0.01 kg)
        
        Target mean: 0.01 kg
        Log1p(0.01) = 0.00995 ≈ 0.01
        
        We want initial predictions around log(1+0.01) ≈ 0.01
        """
        last_layer = self.biomass_head[-1]  # The Linear layer (removed Softplus)
        
        # Very small weights to start with tiny predictions
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.0001)
        
        # CRITICAL: Bias for tiny targets
        # We want log1p(0.01) ≈ 0.01 as initial output
        # Set bias to -4.0 gives raw output around -4.0
        # But we need to think in terms of the scale...
        # Actually, let's set bias to predict log1p(0.02) ≈ 0.0198
        nn.init.constant_(last_layer.bias, -4.0)  # Start very small

    def freeze_backbone(self, freeze_fraction=BACKBONE_FREEZE_FRACTION):
        params = list(self.backbone.parameters())
        num_to_freeze = int(len(params) * freeze_fraction)
        for i, param in enumerate(params):
            if i < num_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x):
        feat_map = self.backbone(x) # (B, C, H_feat, W_feat)
        img_feats = self.global_pool(feat_map).flatten(1) # (B, C)
        
        # Heads
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        month_logits = self.month_head(img_feats)
        aux_out = self.aux_head(img_feats) 
        
        # Fusion
        combined_feats = torch.cat([img_feats, aux_out, species_probs, month_logits], dim=1)
        
        # Log-Space Predictions (raw outputs, no activation)
        log_preds_raw = self.biomass_head(combined_feats) # (B, 4)
        
        # Apply Softplus to ensure positive values (log1p output must be >= 0)
        # Softplus(x) = log(1 + exp(x))
        # For x=-4: Softplus(-4) ≈ 0.018
        log_preds = nn.functional.softplus(log_preds_raw)
        
        log_c = log_preds[:, 0:1]
        log_d = log_preds[:, 1:2]
        log_g = log_preds[:, 2:3]
        log_t = log_preds[:, 3:4]
        
        # Derive Log(GDM) = Log(1 + C + G)
        # Where C and G are in linear KG space
        c = torch.expm1(log_c)
        g = torch.expm1(log_g)
        log_gdm = torch.log1p(c + g + 1e-8)
        
        # Stack: C, D, G, Total, GDM
        biomass_out = torch.cat([log_c, log_d, log_g, log_t, log_gdm], dim=1)
        
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