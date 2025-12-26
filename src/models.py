# models.py
import torch
import torch.nn as nn
import timm
from configs import BACKBONE, FUSION_DIM, IMAGE_SIZE, BACKBONE_FREEZE_FRACTION


class BiomassUnifiedModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_aux=3, num_species=16, pretrained=True):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            self.backbone_dim = self.backbone(dummy_input).shape[1]
            
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
            nn.Softplus() # Ensures log-targets >= 0 (since 1+Mass >= 1)
        )

        self._init_biomass_head()
        
    def _init_biomass_head(self):
        """
        Init regression to output small positive values.
        """
        last_layer = self.biomass_head[-2] 
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)
        # Bias -1.0 gives Softplus(-1) approx 0.3, decent starting log-mass
        nn.init.constant_(last_layer.bias, -1.0)

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
        
        # Log-Space Predictions
        log_preds = self.biomass_head(combined_feats) # (B, 4)
        
        log_c = log_preds[:, 0:1]
        log_d = log_preds[:, 1:2]
        log_g = log_preds[:, 2:3]
        log_t = log_preds[:, 3:4]
        
        # Derive Log(GDM) = Log(C + G) = Log( exp(LogC) + exp(LogG) )
        # Actually since we predict Log(1+X), this is trickier.
        # Approximation: Log(GDM) approx Logaddexp(LogC, LogG) 
        # But strictly: (e^LogC - 1) + (e^LogG - 1) = GDM. 
        # So Log(1+GDM) = Log(e^LogC + e^LogG - 1). 
        # Let's use the explicit math for physics consistency.
        
        c = torch.expm1(log_c)
        g = torch.expm1(log_g)
        log_gdm = torch.log1p(c + g + 1e-6)
        
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