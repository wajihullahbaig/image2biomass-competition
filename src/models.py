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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_aux)
        )
        
        # Multi-task heads with amnesia (dropout) to prevent memorization
        self.species_dropout = nn.Dropout(0.5)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_species)
        )
        
        self.month_dropout = nn.Dropout(0.5)
        # 4. Month Head (Cyclical Regression - Regularizer)
        # Forces backbone to learn seasonal cycles (sin/cos)
        self.month_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # Sin, Cos
        )
        
        # 5. Biomass Head
        # Fusion Dim is now controlled by config (512)
        fusion_dim = FUSION_DIM
        
        self.biomass_head = nn.Sequential(
            nn.Linear(self.backbone_dim + num_aux + num_species + 2, fusion_dim), # +2 for Month Sin/Cos
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # OUTPUT: [Clover, Dead, Green] ONLY
            nn.Softplus() # Ensures positive outputs
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
        species_logits = self.species_head(self.species_dropout(img_feats)) 
        species_probs = torch.softmax(species_logits, dim=1)
        
        # Predict month (categorical logits)
        month_logits = self.month_head(self.month_dropout(img_feats))
        
        # Predict auxiliary features (NDVI, Height)
        aux_out = self.aux_head(img_feats) 
        aux_out_clamped = torch.clamp(aux_out, 0.0, 10.0)
            
        # --- FUSION OF ALL FEATURES ---
        combined_feats = torch.cat([
            img_feats, 
            aux_out_clamped, 
            species_probs,
            month_logits
        ], dim=1)
        
        # --- PHYSICS-INFORMED HEAD ---
        # 1. Predict ONLY Components (Clover, Dead, Green)
        # We use Softplus ensuring non-negative raw mass (0 to inf)
        components_pred = self.biomass_head(combined_feats) # (B, 3)
        components_pred = torch.clamp(components_pred, 0.0, 256.0)
        
        c = components_pred[:, 0:1] # Clover
        d = components_pred[:, 1:2] # Dead
        g = components_pred[:, 2:3] # Green
        
        # 2. Physics Constraints (Performed in Computational Graph)
        # Total = Clover + Dead + Green
        # GDM   = Clover + Green
        total = c + d + g
        gdm   = c + g
        
        # 3. Concatenate for Loss Calculation (Order: C, D, G, Total, GDM)
        # This allows gradients from 'Total' loss to flow back to C, D, G
        biomass_out = torch.cat([c, d, g, total, gdm], dim=1)
        
        return biomass_out, aux_out_clamped, species_logits, month_logits

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
