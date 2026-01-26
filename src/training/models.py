# models.py 
import torch
import torch.nn as nn
import timm
from config.loader import cfg

class BiomassUnifiedModel(nn.Module):
    def __init__(self, 
                 num_aux=5, 
                 num_hsv=5,
                 config=None,
                 backbone_name=None,
                 num_species=None,
                 fusion_dim=384,
                 img_h=256,
                 img_w=256,
                 biomass_clamp=2500.0):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        # Priority: explicit arguments -> config -> defaults
        if config:
            self.backbone_name = backbone_name or config.hyperparameters.backbone
            self.num_species = num_species or len(config.species_taxonomy.core_species)
            self.fusion_dim = fusion_dim if fusion_dim != 384 else config.training.fusion_dim
            self.img_h = img_h if img_h != 256 else config.preprocessing.image_height
            self.img_w = img_w if img_w != 256 else config.preprocessing.image_width
            self.num_aux = num_aux
            self.biomass_clamp_val = config.targets.biomass_clamp
        else:
            self.backbone_name = backbone_name or "resnet18"
            self.num_species = num_species or 14
            self.fusion_dim = fusion_dim
            self.img_h = img_h
            self.img_w = img_w
            self.num_aux = num_aux
            self.biomass_clamp_val = biomass_clamp
        
        self.backbone = timm.create_model(self.backbone_name, pretrained=True, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_h, self.img_w)
            feats = self.backbone(dummy_input)
            
            # Handle different backbone architectures (CNN vs ViT)
            if len(feats.shape) == 4:  # CNN: [B, C, H, W]
                self.backbone_dim = feats.shape[1]
            elif len(feats.shape) == 3:  # ViT: [B, seq_len, embed_dim]
                self.backbone_dim = feats.shape[2]
            else:  # Already pooled: [B, features]
                self.backbone_dim = feats.shape[1]
                
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_aux)
        )
        
        # 3. HSV Head (Visual Biomass Scores)
        self.num_hsv = num_hsv
        self.hsv_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_hsv)
        )
        
        # 4. Species Head (Fine-Grained)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.LayerNorm(32),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_species)
        )
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux + HSV + Species
        input_dim = self.backbone_dim + self.num_aux + self.num_hsv + self.num_species
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim, 128),
            nn.LayerNorm(128),  
            nn.ReLU(),
            nn.Linear(128, 5), # [Green, Dead, Clover, GDM, Total] predicted directly
        )

        self.log_clamp = torch.log1p(torch.tensor(float(self.biomass_clamp_val)))
        
        self._init_biomass_head()
        
    def _init_biomass_head(self):
        # 1. Global Initialization for all heads
        for m in [self.aux_head, self.hsv_head, self.species_head, self.biomass_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    # Use He initialization for ReLU activated layers
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

        # 2. Specific centering for the output layer to prevent "Berserk" logs
        last_layer = self.biomass_head[-1]
        # Use Xavier for the final layer which feeds into Sigmoid/Softplus
        nn.init.xavier_uniform_(last_layer.weight)
        
        with torch.no_grad():
            last_layer.bias[0] = 3.0 # Green (~20g)
            last_layer.bias[1] = 2.0 # Dead (~7g)
            last_layer.bias[2] = 2.5 # Clover (~12g)
            last_layer.bias[3] = 3.2 # GDM (~25g)
            last_layer.bias[4] = 3.5 # Total (~30g)

    def forward(self, x):
        feat_map = self.backbone(x)
        
        # Handle different backbone architectures
        if len(feat_map.shape) == 4:  # CNN: [B, C, H, W]
            img_feats = self.global_pool(feat_map).flatten(1)
        elif len(feat_map.shape) == 3:  # ViT: [B, seq_len, embed_dim]
            # For ViTs, typically take the [CLS] token (first token) or mean pool
            img_feats = feat_map.mean(dim=1)  # Mean pooling over sequence length
        else:  # Already pooled: [B, features]
            img_feats = feat_map
        
        # Heads
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        aux_out = self.aux_head(img_feats)
        hsv_out = self.hsv_head(img_feats)
        
        # Fusion
        combined_feats = torch.cat([img_feats, aux_out, hsv_out, species_probs], dim=1)
        
        # 5. Biomass Prediction (Green, Dead_Direct, Clover, GDM_Direct, Total_Direct)
        log_preds_raw = self.biomass_head(combined_feats)
        
        # Split and process raw outputs
        # Note: We use softplus + log1p for stability and to ensure positive values
        p_green        = torch.clamp(nn.functional.softplus(log_preds_raw[:, 0:1]), 0.0, self.log_clamp)
        p_dead_direct  = torch.clamp(nn.functional.softplus(log_preds_raw[:, 1:2]), 0.0, self.log_clamp)
        p_clover       = torch.clamp(nn.functional.softplus(log_preds_raw[:, 2:3]), 0.0, self.log_clamp)
        p_gdm_direct   = torch.clamp(nn.functional.softplus(log_preds_raw[:, 3:4]), 0.0, self.log_clamp)
        p_total_direct = torch.clamp(nn.functional.softplus(log_preds_raw[:, 4:5]), 0.0, self.log_clamp)
        
        # --- Physics-based derivation Path ---
        # 1. GDM Derived = Green + Clover
        lin_green = torch.expm1(p_green)
        lin_clover = torch.expm1(p_clover)
        lin_gdm_derived = torch.clamp(lin_green + lin_clover, min=1e-4)
        p_gdm_derived = torch.log1p(lin_gdm_derived)
        
        # Blend GDM: mostly direct but constrained by derivation
        p_gdm = 0.7 * p_gdm_direct + 0.3 * p_gdm_derived
        
        # 2. Dead Derived = Total - (Green + Clover)
        lin_total = torch.expm1(p_total_direct)
        lin_dead_derived = torch.clamp(lin_total - (lin_green + lin_clover), min=1e-4)
        p_dead_derived = torch.log1p(lin_dead_derived)
        
        # --- Visibility-Aware Blending for Dead ---
        # The dead_hsv score is index 3 of hsv_out (predicted visibility)
        if hsv_out.shape[1] > 3:
            # Predict visibility fraction. We use a sigmoid to create a smooth but decisive gate.
            vis_score = torch.sigmoid((hsv_out[:, 3:4] - 0.08) * 20.0) 
            p_dead = vis_score * p_dead_direct + (1 - vis_score) * p_dead_derived
        else:
            p_dead = 0.5 * p_dead_direct + 0.5 * p_dead_derived
            
        # 3. Final Total consistency
        # Total = GDM + Dead
        lin_gdm_final = torch.expm1(p_gdm)
        lin_dead_final = torch.expm1(p_dead)
        p_total_final = torch.log1p(torch.clamp(lin_gdm_final + lin_dead_final, min=1e-4))
        
        # Final output order for competition: [Green, Dead, Clover, GDM, Total]
        biomass_out = torch.cat([p_green, p_dead, p_clover, p_gdm, p_total_final], dim=1)
        
        return biomass_out, aux_out, hsv_out, species_logits

