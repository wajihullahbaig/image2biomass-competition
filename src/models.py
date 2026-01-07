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
            self.backbone_dim = feats.shape[1]
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64,self.num_aux)
        )
        
        # 3. Species Head (Fine-Grained: 14 classes)
        self.species_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_species)
        )

        # 4. Taxonomy Head (Coarse-Grained: 3 classes - Legume, Grass, Weed)
        self.taxonomy_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(16, 3) 
        )
        
        # 5. Biomass Head
        # Inputs: Backbone + Aux(3) + Species(14) + Taxonomy(3)
        input_dim = self.backbone_dim + self.num_aux + self.num_species + 3
                
        self.biomass_head = nn.Sequential(
            nn.Linear(input_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4), # [Log_C, Log_D, Log_G, Log_T]
        )

        self.log_clamp = torch.log1p(torch.tensor(cfg.targets.biomass_clamp))
        
        self._init_biomass_head()
        
    def _init_biomass_head(self):
        last_layer = self.biomass_head[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        with torch.no_grad():
            last_layer.bias.fill_(0)
            last_layer.bias[0] = 3.0 # ~20g
            last_layer.bias[1] = 2.0 # ~7g
            last_layer.bias[2] = 3.0 # ~20g
            last_layer.bias[3] = 4.0 # ~54g

    def forward(self, x):
        feat_map = self.backbone(x)
        img_feats = self.global_pool(feat_map).flatten(1)
        
        # Heads
        species_logits = self.species_head(img_feats)
        species_probs = torch.softmax(species_logits, dim=1)
        
        taxonomy_logits = self.taxonomy_head(img_feats)
        taxonomy_probs = torch.softmax(taxonomy_logits, dim=1)
        
        aux_out = self.aux_head(img_feats) 
        
        # Fusion: Include Taxonomy Probs
        combined_feats = torch.cat([img_feats, aux_out, species_probs, taxonomy_probs], dim=1)
        
        # Biomass Prediction
        log_preds_raw = self.biomass_head(combined_feats)
        # softplus ensures positivity, clamp ensures we don't blow up expm1
        log_preds = torch.clamp(nn.functional.softplus(log_preds_raw), 0.0, self.log_clamp)
        
        log_c = log_preds[:, 0:1]
        log_d = log_preds[:, 1:2]
        log_g = log_preds[:, 2:3]
        log_t = log_preds[:, 3:4]
        
        # Derived GDM
        c = torch.expm1(log_c)
        g = torch.expm1(log_g)
        log_gdm = torch.log1p(c + g + 1e-8)
        
        biomass_out = torch.cat([log_c, log_d, log_g, log_t, log_gdm], dim=1)
        
        return biomass_out, aux_out, species_logits, taxonomy_logits

