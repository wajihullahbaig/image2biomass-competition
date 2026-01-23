# models.py - FIXED VERSION
import torch
import torch.nn as nn
import timm
from config.loader import cfg

class BiomassUnifiedModel(nn.Module):
    def __init__(self, 
                 num_aux=3, 
                 config=None):
        super(BiomassUnifiedModel, self).__init__()
        
        # 1. Image Backbone
        # Use config if provided, else use passed arguments or defaults
        if config:
            self.backbone_name = config.hyperparameters.backbone
            self.num_species = len(config.species_taxonomy.core_species)
            self.fusion_dim = config.training.fusion_dim
            self.img_h = config.preprocessing.image_height
            self.img_w = config.preprocessing.image_width
            self.num_aux = num_aux
        else:
            raise ValueError("Config must be provided")
        
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
        
        # 2. Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64,self.num_aux)
        )
        
        # 3. Species Head (Fine-Grained: 14 classes)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_species)
        )
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux + Specie
        input_dim = self.backbone_dim + self.num_aux + self.num_species
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(self.fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 5), # [Green, Dead, Clover, GDM, Total]
        )

        self.log_clamp = torch.log1p(torch.tensor(cfg.targets.biomass_clamp))
        
        self._init_biomass_head()
        
    def _init_biomass_head(self):
        last_layer = self.biomass_head[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        with torch.no_grad():
            last_layer.bias.fill_(0)
            last_layer.bias[0] = 3.0 # ~20g green
            last_layer.bias[1] = 2.0 # ~7g dead
            last_layer.bias[2] = 2.5 # ~12g clover
            last_layer.bias[3] = 3.2 # ~25g gdm (green + clover)
            last_layer.bias[4] = 3.5 # ~30g total (green + dead + clover)

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
        
        # Fusion
        combined_feats = torch.cat([img_feats, aux_out, species_probs], dim=1)
        
        # Biomass Prediction (Green, Dead, Clover)
        log_preds_raw = self.biomass_head(combined_feats)
        # softplus ensures positivity, clamp ensures we don't blow up expm1
        biomass_out = torch.clamp(nn.functional.softplus(log_preds_raw), 0.0, self.log_clamp)
        
        return biomass_out, aux_out, species_logits

