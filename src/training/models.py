# models.py - Dual-Stream DINO Vision Model with Interval Classification
import torch
import torch.nn as nn
import timm


class DualStreamBiomassModel(nn.Module):
    """
    Dual-Stream Vision Transformer for Biomass Prediction.
    
    Architecture:
    1. Shared Vision Backbone (DINOv3 / DINOv2) extracting rich patch/class tokens from Left and Right views.
    2. Multi-Head Self-Attention Cross-View Interaction layer allowing spatial features across the plot seam to interact.
    3. Fusion MLP projecting concatenated cross-view tokens.
    4. 5 Independent Continuous Regression Heads (no hard-coded physical constraints during backprop).
    5. 5 Auxiliary Interval Classification Heads (7 bins each from UEPNet formulation) providing stabilizing gradients.
    """
    def __init__(self, 
                 backbone_name="vit_small_patch14_dinov2",
                 num_targets=3,
                 num_intervals=7,
                 fusion_dim=384,
                 dropout=0.3,
                 pretrained=True,
                 config=None):
        super().__init__()
        
        if config is not None:
            backbone_name = config.hyperparameters.backbone
            fusion_dim = getattr(config.training, 'fusion_dim', 384)
            dropout = getattr(config.training, 'dropout', 0.3)
            num_intervals = getattr(config.loss, 'num_intervals', 7)
            num_targets = len(config.targets.cols)
            
        self.backbone_name = backbone_name
        self.num_targets = num_targets
        self.num_intervals = num_intervals
        self.fusion_dim = fusion_dim
        
        # 1. Shared Vision Backbone
        kwargs = {}
        if 'dinov2' in self.backbone_name or 'patch14' in self.backbone_name:
            kwargs['dynamic_img_size'] = True
        self.backbone = timm.create_model(
            self.backbone_name, 
            pretrained=pretrained, 
            num_classes=0,
            **kwargs
        )
        self.backbone_dim = self.backbone.num_features
        
        # 2. Cross-View Interaction: Multi-Head Self-Attention on [B, 2, backbone_dim]
        num_heads = 8 if self.backbone_dim % 8 == 0 else 4
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=self.backbone_dim, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.backbone_dim)
        
        # 3. Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. 5 Independent Continuous Regression Heads (3-layer MLP each)
        # Order: [Green, Dead, Clover, GDM, Total]
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim // 2),
                nn.LayerNorm(self.fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(self.fusion_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(self.num_targets)
        ])
        
        # 5. 5 Auxiliary Interval Classification Heads (7 classes each)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, self.num_intervals)
            ) for _ in range(self.num_targets)
        ])
        
        self._init_heads()
        
    def _init_heads(self):
        for m in list(self.reg_heads) + list(self.cls_heads) + [self.fusion_mlp]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        
    def extract_features(self, x):
        feats = self.backbone(x)
        if len(feats.shape) == 3:  # ViT tokens [B, N, C]
            return feats.mean(dim=1)
        elif len(feats.shape) == 4:  # CNN feature maps [B, C, H, W]
            return feats.mean(dim=[2, 3])
        return feats

    def forward(self, img_left, img_right):
        # 1. Feature extraction through shared backbone
        feat_l = self.extract_features(img_left)   # [B, backbone_dim]
        feat_r = self.extract_features(img_right)  # [B, backbone_dim]
        
        # 2. Cross-view interaction via self-attention
        tokens = torch.stack([feat_l, feat_r], dim=1)  # [B, 2, backbone_dim]
        attn_out, _ = self.cross_view_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)  # Residual connection
        
        # 3. Concatenate and project through fusion MLP
        fused = torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)  # [B, backbone_dim * 2]
        fused = self.fusion_mlp(fused)                           # [B, fusion_dim]
        
        # 4. Continuous regression predictions (grams)
        # Softplus ensures positive predictions
        reg_preds = [
            nn.functional.softplus(head(fused)) for head in self.reg_heads
        ]
        
        # 5. Discrete interval classification logits
        cls_preds = [
            head(fused) for head in self.cls_heads
        ]
        
        return reg_preds, cls_preds


