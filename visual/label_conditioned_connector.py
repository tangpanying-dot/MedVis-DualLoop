"""
Label-Conditioned Connector for Medical Report Generation (v23.0 Final)

核心思想:
- 用CheXpert标签引导patch选择
- 动态调整token数量 (Batch Max 策略)
- 标签条件化的注意力机制

修改日志:
- [Fixed] AnatomyAwarePE 标签索引与 Dataset LABEL_NAMES 严格对齐
- [Optimized] Token Budget 使用 Batch Max 策略，提升稳定性

Author: Information
Date: 2025-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==================== 配置 ====================

class ConnectorConfig:
    """Connector配置"""
    # 输入维度
    visual_dim = 768          # RAD-DINO特征维度
    num_labels = 14           # CheXpert标签数
    
    # 标签嵌入
    label_embed_dim = 256     # 标签嵌入维度
    label_context_dim = 128   # 标签上下文维度
    
    # 输出维度
    output_dim = 2048         # Gemma-2B输入维度
    
    # Token预算（根据异常数量动态调整）
    token_budget = {
        0: 64,    # 无异常 → 64 tokens
        1: 128,   # 1-2个异常 → 128 tokens
        2: 128,
        3: 192,   # 3+个异常 → 192 tokens (最大)
    }
    
    # 权重参数
    visual_weight = 0.4      # α: 视觉显著性权重
    label_weight = 0.6        # β: 标签引导权重
    
    # 训练参数
    dropout = 0.1
    temperature = 1.0         # 注意力温度


# ==================== 模块1: 标签嵌入 ====================

class LabelEmbedding(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.num_labels = config.num_labels
        self.embed_dim = config.label_embed_dim
        
        self.label_embeds = nn.Parameter(
            torch.randn(self.num_labels, self.embed_dim) * 0.02
        )
        
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B, 14) -> (B, 256)
        embedded = torch.matmul(labels, self.label_embeds)  
        embedded = self.proj(embedded)
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)
        return embedded


# ==================== 模块1.5: 解剖感知位置编码 (FIXED) ====================

class AnatomyAwarePE(nn.Module):
    """
    解剖感知位置编码
    [Fixed]: 索引顺序已修正，与 Dataset 对齐 (0=Atelectasis ... 8=No Finding)
    """
    
    ANATOMY_REGIONS = {
        0: "RUL", 1: "RML", 2: "RLL", 3: "LUL", 4: "LLL", 5: "Bilateral"
    }
    
    def __init__(self, output_dim: int, num_labels: int = 14):
        super().__init__()
        self.output_dim = output_dim
        self.num_regions = 6
        
        self.anatomy_embeddings = nn.Embedding(self.num_regions, output_dim)
        self.register_buffer('label_anatomy_map', self._build_label_anatomy_map())
        self.map_adjustment = nn.Parameter(torch.zeros(num_labels, self.num_regions))
    
    def _build_label_anatomy_map(self) -> torch.Tensor:
        """
        构建CheXpert 14类标签到6个解剖区域的映射
        顺序: Atelectasis, Cardiomegaly, Consolidation, Edema, Enl. Card., Fracture, 
              Lung Lesion, Lung Opacity, No Finding, Effusion, Pleural Other, Pneumonia, 
              Pneumothorax, Support Devices
        """
        mapping = torch.zeros(14, 6)
        
        # 0: Atelectasis (偏下叶)
        mapping[0] = torch.tensor([0.15, 0.15, 0.25, 0.15, 0.25, 0.05])
        # 1: Cardiomegaly (中央/双侧)
        mapping[1] = torch.tensor([0.05, 0.05, 0.10, 0.05, 0.10, 0.65])
        # 2: Consolidation (全肺)
        mapping[2] = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20, 0.00])
        # 3: Edema (双侧)
        mapping[3] = torch.tensor([0.15, 0.15, 0.20, 0.15, 0.20, 0.15])
        # 4: Enlarged Cardiomediastinum (中央)
        mapping[4] = torch.tensor([0.05, 0.05, 0.10, 0.05, 0.10, 0.65])
        # 5: Fracture (全肺/骨骼)
        mapping[5] = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20, 0.00])
        # 6: Lung Lesion (全肺)
        mapping[6] = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20, 0.00])
        # 7: Lung Opacity (全肺)
        mapping[7] = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20, 0.00])
        # 8: No Finding (均匀分布)
        mapping[8] = torch.tensor([0.16, 0.16, 0.17, 0.16, 0.17, 0.18])
        # 9: Pleural Effusion (下叶)
        mapping[9] = torch.tensor([0.05, 0.10, 0.35, 0.05, 0.35, 0.10])
        # 10: Pleural Other (全肺)
        mapping[10] = torch.tensor([0.15, 0.15, 0.25, 0.15, 0.25, 0.05])
        # 11: Pneumonia (全肺/下叶)
        mapping[11] = torch.tensor([0.15, 0.15, 0.25, 0.15, 0.25, 0.05])
        # 12: Pneumothorax (上叶)
        mapping[12] = torch.tensor([0.30, 0.15, 0.10, 0.30, 0.10, 0.05])
        # 13: Support Devices (中央/上部)
        mapping[13] = torch.tensor([0.10, 0.10, 0.15, 0.10, 0.15, 0.40])
        
        return mapping
    
    def forward(self, tokens, chexpert_labels):
        adjusted_map = self.label_anatomy_map + self.map_adjustment
        adjusted_map = F.softmax(adjusted_map, dim=-1)
        
        # 修复：自动匹配 adjusted_map 的精度 (BFloat16)
        anatomy_dist = torch.matmul(chexpert_labels.to(adjusted_map.dtype), adjusted_map)
        anatomy_dist_sum = anatomy_dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        anatomy_dist = anatomy_dist / anatomy_dist_sum
        
        region_ids = torch.arange(self.num_regions, device=tokens.device)
        anatomy_embeds = self.anatomy_embeddings(region_ids)
        
        anatomy_pe = torch.matmul(anatomy_dist, anatomy_embeds)
        anatomy_pe = anatomy_pe.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        
        return tokens + anatomy_pe


# ==================== 模块2: 标签编码器 ====================

class LabelEncoder(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.label_embed_dim, num_heads=4, dropout=config.dropout, batch_first=True
        )
        self.context_proj = nn.Sequential(
            nn.Linear(config.label_embed_dim, config.label_context_dim),
            nn.LayerNorm(config.label_context_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, label_embed):
        x = label_embed.unsqueeze(1)
        attn_out, _ = self.self_attn(x, x, x)
        return self.context_proj(attn_out.squeeze(1))


# ==================== 模块3: Patch选择器 ====================

class LabelGuidedPatchSelector(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.visual_scorer = nn.Sequential(
            nn.Linear(config.visual_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.label_visual_proj = nn.Linear(config.label_context_dim, config.visual_dim)
        self.visual_weight = config.visual_weight
        self.label_weight = config.label_weight
        self.temperature = config.temperature
    
    def select_patches(self, visual_feats, label_context, k, visual_mask=None):
        B, N, D = visual_feats.shape
        
        # Visual Saliency
        visual_scores = self.visual_scorer(visual_feats).squeeze(-1)
        
        # Label Similarity
        label_proj = self.label_visual_proj(label_context).unsqueeze(1)
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        label_norm = F.normalize(label_proj, p=2, dim=-1)
        label_scores = (visual_norm * label_norm).sum(dim=-1)
        
        # Combined
        combined_scores = (self.visual_weight * visual_scores + self.label_weight * label_scores) / self.temperature
        
        if visual_mask is not None:
            combined_scores = combined_scores.masked_fill(~visual_mask, float('-inf'))
        
        # Top-K
        k = min(k, N)
        topk_scores, topk_indices = torch.topk(combined_scores, k, dim=1)
        
        batch_indices = torch.arange(B, device=visual_feats.device).unsqueeze(1).expand(-1, k)
        selected_feats = visual_feats[batch_indices, topk_indices]
        
        return selected_feats, topk_indices


# ==================== 模块4: 主Connector ====================

class LabelConditionedConnector(nn.Module):
    def __init__(self, config: ConnectorConfig = None):
        super().__init__()
        if config is None: config = ConnectorConfig()
        self.config = config
        
        self.label_embedding = LabelEmbedding(config)
        self.label_encoder = LabelEncoder(config)
        self.patch_selector = LabelGuidedPatchSelector(config)
        
        self.visual_proj = nn.Sequential(
            nn.Linear(config.visual_dim, config.output_dim),
            nn.LayerNorm(config.output_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim), nn.Dropout(config.dropout)
        )
        
        self.anatomy_pe = AnatomyAwarePE(output_dim=config.output_dim, num_labels=config.num_labels)
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, config.output_dim) * 0.02)
    
    def _count_abnormalities(self, labels: torch.Tensor) -> torch.Tensor:
        # 排除 No Finding (index 8)
        mask = torch.ones(14, dtype=torch.bool, device=labels.device)
        mask[8] = False
        abnormal_labels = labels[:, mask]
        # 阈值 0.5 (兼容 Soft Labels)
        counts = (abnormal_labels > 0.5).sum(dim=1)
        return counts
    
    def decide_token_budget(self, labels: torch.Tensor) -> int:
        """
        [优化版] Batch-Max 策略
        为防止复杂样本被平均数"饿死"，我们计算Batch中最大的需求作为本轮的k。
        """
        # 1. 统计每个样本的异常数
        abnormality_counts = self._count_abnormalities(labels) # (B,)
        
        # 2. 计算每个样本应得的 budget
        budgets = torch.full_like(abnormality_counts, self.config.token_budget[0]) # Default 64
        
        mask_medium = (abnormality_counts >= 1) & (abnormality_counts <= 2)
        budgets[mask_medium] = self.config.token_budget[1] # 128
        
        mask_high = abnormality_counts > 2
        budgets[mask_high] = self.config.token_budget[3] # 192
        
        # 3. 取 Batch 最大值
        final_k = budgets.max().item()
        return int(final_k)
    
    def forward(self, visual_features, chexpert_labels, visual_mask=None):
        # 1. Embed Labels
        label_embed = self.label_embedding(chexpert_labels)
        label_context = self.label_encoder(label_embed)
        
        # 2. Decide Budget (Batch Max)
        k = self.decide_token_budget(chexpert_labels)
        
        # 3. Select Patches
        selected_feats, _ = self.patch_selector.select_patches(
            visual_features, label_context, k, visual_mask
        )
        
        # 4. Project
        output_tokens = self.visual_proj(selected_feats)
        
        # 5. Add PEs
        output_tokens = output_tokens + self.pos_encoding[:, :k, :]
        output_tokens = self.anatomy_pe(output_tokens, chexpert_labels)
        
        return output_tokens