"""
Anatomy-Aware Visual-to-LLM Connector (Diagnosis Enforcer Edition)

Core Update: Added a classification head to force pathology awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings

@dataclass
class ConnectorConfig:
    """Configuration for AnatomyAwareConnector"""
    input_dim: int = 768          # RAD-DINO output
    llm_dim: int = 2048           # Gemma embedding dim
    num_latents: int = 64         # Number of query tokens
    num_layers: int = 2           # Perceiver layers
    num_heads: int = 8            # Attention heads
    patch_size: int = 1369        # 37x37 grid
    max_views: int = 12           
    dropout: float = 0.1          
    ffn_expansion: int = 4        
    use_flash_attn: bool = False  
    layer_norm_eps: float = 1e-6  
    
    def validate(self):
        assert self.input_dim > 0
        assert self.llm_dim % self.num_heads == 0

# --- 辅助类保持不变 (PositionalEncoding 和 PerceiverBlock) ---

class AnatomyAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, patch_size=49, max_views=12, init_std=0.02):
        super().__init__()
        self.patch_size = patch_size
        self.max_views = max_views
        self.spatial_embed = nn.Embedding(patch_size, d_model)
        self.view_embed = nn.Embedding(max_views, d_model)
        self._init_weights(init_std)
    
    def _init_weights(self, std):
        nn.init.trunc_normal_(self.spatial_embed.weight, std=std)
        nn.init.trunc_normal_(self.view_embed.weight, std=std)
    
    def forward(self, x):
        B, Total_Len, D = x.shape
        device = x.device
        global_indices = torch.arange(Total_Len, device=device, dtype=torch.long)
        spatial_indices = global_indices % self.patch_size
        view_indices = torch.clamp(global_indices // self.patch_size, max=self.max_views - 1)
        pos = self.spatial_embed(spatial_indices) + self.view_embed(view_indices)
        return x + pos.unsqueeze(0)

class PerceiverResamplerBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1, ffn_expansion=4, layer_norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, latents, visual_features, key_padding_mask=None):
        latents = latents + self.cross_attn(self.norm1(latents), visual_features, visual_features, key_padding_mask=key_padding_mask, need_weights=False)[0]
        latents = latents + self.self_attn(self.norm2(latents), self.norm2(latents), self.norm2(latents), need_weights=False)[0]
        latents = latents + self.ffn(self.norm3(latents))
        return latents

# --- Connector 主类修改 ---

class AnatomyAwareConnector(nn.Module):
    """
    Anatomy-Aware Connector with Diagnosis Head
    """
    def __init__(self, config: Optional[ConnectorConfig] = None, num_classes=14, **kwargs):
        super().__init__()
        if config is None: config = ConnectorConfig(**kwargs)
        config.validate()
        self.config = config
        
        print(f"[AnatomyAwareConnector] Initializing with {num_classes} diagnosis classes")
        
        # 1. Input Proj
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.llm_dim),
            nn.LayerNorm(config.llm_dim, eps=config.layer_norm_eps),
            nn.GELU(), nn.Dropout(config.dropout)
        )
        
        # 2. Pos Enc
        self.pos_encoder = AnatomyAwarePositionalEncoding(
            d_model=config.llm_dim, patch_size=config.patch_size, max_views=config.max_views
        )
        
        # 3. Latents
        self.latents = nn.Parameter(torch.zeros(1, config.num_latents, config.llm_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        
        # 4. Perceiver Layers
        self.layers = nn.ModuleList([
            PerceiverResamplerBlock(
                d_model=config.llm_dim, num_heads=config.num_heads,
                dropout=config.dropout, ffn_expansion=config.ffn_expansion,
                layer_norm_eps=config.layer_norm_eps
            ) for _ in range(config.num_layers)
        ])
        
        # 5. Output Proj
        self.output_proj = nn.Linear(config.llm_dim, config.llm_dim)
        self.output_norm = nn.LayerNorm(config.llm_dim, eps=config.layer_norm_eps)
        
        # [NEW] 6. Diagnosis Head (The Enforcer)
        # 用来预测 14 种疾病，强制 Latents 包含病理信息
        self.diagnosis_head = nn.Sequential(
            nn.Linear(config.llm_dim, config.llm_dim // 2),
            nn.LayerNorm(config.llm_dim // 2),
            nn.GELU(),
            nn.Linear(config.llm_dim // 2, num_classes) # 输出未归一化的 Logits
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(
        self, 
        visual_features: torch.Tensor, 
        visual_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]: # [NEW] 返回两个值
        
        B = visual_features.shape[0]
        
        # 1. Project & Pos
        x = self.input_proj(visual_features)
        x = self.pos_encoder(x)
        
        # 2. Mask
        key_padding_mask = ~visual_mask if visual_mask is not None else None
        
        # 3. Perceiver Process
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, x, key_padding_mask)
            
        # [NEW] 4. Diagnosis Classification
        # 对 Latents 取平均，获得全局语义
        global_feat = torch.mean(latents, dim=1) # (B, D)
        diagnosis_logits = self.diagnosis_head(global_feat) # (B, 14)
        
        # 5. Output Tokens
        visual_tokens = self.output_proj(latents)
        visual_tokens = self.output_norm(visual_tokens)
        
        return visual_tokens, diagnosis_logits # 返回给训练 Loop

    def get_num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable