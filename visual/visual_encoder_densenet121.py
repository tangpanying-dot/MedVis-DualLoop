# visual_encoder_improved.py
"""
改进版视觉编码器：保留空间信息，捕捉局部病变
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
from PIL import Image
from skimage.transform import resize
import os


class ImprovedVisualEncoder:
    """
    改进的视觉编码器：
    1. 保留多尺度空间特征
    2. 使用注意力加权而非简单平均
    3. 提取局部和全局特征
    """
    def __init__(self, feature_dim=1024, model_name='densenet121-res224-all', 
                 strategy='multi_scale'):
        """
        Args:
            feature_dim: 输出特征维度
            model_name: TorchXRayVision模型名称
            strategy: 特征提取策略
                - 'global_avg': 原始全局平均池化
                - 'multi_scale': 多尺度特征（推荐）
                - 'spatial_attention': 空间注意力加权
                - 'roi_pooling': ROI池化
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.feature_dim = feature_dim
        
        print(f"加载TorchXRayVision模型: {model_name}")
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # 根据策略初始化不同的模块
        if strategy == 'multi_scale':
            self._init_multi_scale()
        elif strategy == 'spatial_attention':
            self._init_spatial_attention()
        elif strategy == 'roi_pooling':
            self._init_roi_pooling()
        
        print(f"初始化完成 - 策略: {strategy}, 特征维度: {feature_dim}")
    
    def _init_multi_scale(self):
        """初始化多尺度特征提取"""
        # DenseNet121的特征维度是1024
        # 我们提取不同尺度的特征并拼接
        
        # 全局特征 + 局部特征
        # 7x7 -> 保留4个区域 + 全局 = 5个特征向量
        input_dim = 1024 * 5  # 每个位置1024维
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, self.feature_dim)
        ).to(self.device)
        
        # 初始化权重
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _init_spatial_attention(self):
        """初始化空间注意力模块"""
        # 学习空间位置的重要性权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        ).to(self.device)
        
        # 投影到目标维度
        if self.feature_dim != 1024:
            self.projection = nn.Linear(1024, self.feature_dim).to(self.device)
        else:
            self.projection = None
    
    def _init_roi_pooling(self):
        """初始化ROI池化（需要病变检测结果）"""
        # 简化版：将图像分成9个区域，分别池化
        self.num_regions = 9  # 3x3 grid
        input_dim = 1024 * self.num_regions
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, self.feature_dim)
        ).to(self.device)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')
        
        img_np = np.array(image)
        img_resized = resize(img_np, (224, 224), preserve_range=True)
        
        # 归一化到TorchXRayVision标准
        img_normalized = img_resized / 255.0
        img_normalized = (img_normalized * 2048) - 1024
        
        img_tensor = torch.from_numpy(img_normalized).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)

    def extract_features(self, image_path: str) -> torch.Tensor:
        """
        根据策略提取特征
        
        Returns:
            特征向量 (1, feature_dim)
        """
        try:
            image_tensor = self.preprocess_image(image_path)
            
            with torch.no_grad():
                # 提取特征图 (1, 1024, 7, 7)
                feature_map = self.model.features(image_tensor)
                
                # 根据策略处理
                if self.strategy == 'global_avg':
                    features = self._extract_global_avg(feature_map)
                elif self.strategy == 'multi_scale':
                    features = self._extract_multi_scale(feature_map)
                elif self.strategy == 'spatial_attention':
                    features = self._extract_spatial_attention(feature_map)
                elif self.strategy == 'roi_pooling':
                    features = self._extract_roi_pooling(feature_map)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                
                return features
                
        except Exception as e:
            print(f"提取特征错误 {image_path}: {e}")
            return None
    
    def _extract_global_avg(self, feature_map):
        """原始全局平均池化"""
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1))
        features = torch.flatten(pooled, 1)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def _extract_multi_scale(self, feature_map):
        """
        多尺度特征提取：
        1. 全局特征（整张图）
        2. 四象限特征（左上、右上、左下、右下）
        """
        B, C, H, W = feature_map.shape  # (1, 1024, 7, 7)
        
        # 1. 全局特征
        global_feat = F.adaptive_avg_pool2d(feature_map, (1, 1))  # (1, 1024, 1, 1)
        global_feat = global_feat.view(B, -1)  # (1, 1024)
        
        # 2. 四象限特征（模拟四个肺区域）
        h_mid, w_mid = H // 2, W // 2
        
        # 左上象限（左肺上叶）
        top_left = F.adaptive_avg_pool2d(
            feature_map[:, :, :h_mid, :w_mid], (1, 1)
        ).view(B, -1)
        
        # 右上象限（右肺上叶）
        top_right = F.adaptive_avg_pool2d(
            feature_map[:, :, :h_mid, w_mid:], (1, 1)
        ).view(B, -1)
        
        # 左下象限（左肺下叶）
        bottom_left = F.adaptive_avg_pool2d(
            feature_map[:, :, h_mid:, :w_mid], (1, 1)
        ).view(B, -1)
        
        # 右下象限（右肺下叶）
        bottom_right = F.adaptive_avg_pool2d(
            feature_map[:, :, h_mid:, w_mid:], (1, 1)
        ).view(B, -1)
        
        # 拼接所有特征
        multi_scale_feat = torch.cat([
            global_feat, 
            top_left, top_right, 
            bottom_left, bottom_right
        ], dim=1)  # (1, 1024*5)
        
        # 投影到目标维度
        features = self.projection(multi_scale_feat)
        
        return features
    
    def _extract_spatial_attention(self, feature_map):
        """
        空间注意力加权：
        学习哪些空间位置更重要（病变区域应该有更高权重）
        """
        # 生成注意力权重图 (1, 1, 7, 7)
        attention_weights = self.spatial_attention(feature_map)
        
        # 加权特征图
        weighted_features = feature_map * attention_weights  # (1, 1024, 7, 7)
        
        # 全局池化
        pooled = F.adaptive_avg_pool2d(weighted_features, (1, 1))
        features = torch.flatten(pooled, 1)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def _extract_roi_pooling(self, feature_map):
        """
        ROI池化：将图像分成3x3网格，每个区域独立池化
        """
        B, C, H, W = feature_map.shape
        
        # 计算每个网格的大小
        grid_h = H // 3
        grid_w = W // 3
        
        region_features = []
        
        for i in range(3):
            for j in range(3):
                # 提取区域
                h_start = i * grid_h
                h_end = (i + 1) * grid_h if i < 2 else H
                w_start = j * grid_w
                w_end = (j + 1) * grid_w if j < 2 else W
                
                region = feature_map[:, :, h_start:h_end, w_start:w_end]
                
                # 区域池化
                region_pooled = F.adaptive_avg_pool2d(region, (1, 1))
                region_features.append(region_pooled.view(B, -1))
        
        # 拼接所有区域特征
        roi_features = torch.cat(region_features, dim=1)  # (1, 1024*9)
        
        # 投影到目标维度
        features = self.projection(roi_features)
        
        return features


# ========== 对比实验工具 ==========

def compare_strategies(image_path: str):
    """
    对比不同策略提取的特征
    """
    print("\n=== 策略对比实验 ===")
    
    strategies = ['global_avg', 'multi_scale', 'spatial_attention', 'roi_pooling']
    results = {}
    
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        encoder = ImprovedVisualEncoder(
            feature_dim=1024, 
            strategy=strategy
        )
        
        features = encoder.extract_features(image_path)
        
        if features is not None:
            results[strategy] = features
            print(f"  特征形状: {features.shape}")
            print(f"  特征范数: {torch.norm(features).item():.4f}")
            print(f"  特征均值: {torch.mean(features).item():.4f}")
            print(f"  特征标准差: {torch.std(features).item():.4f}")
    
    # 计算策略间的特征相似度
    print("\n=== 策略间余弦相似度 ===")
    strategy_list = list(results.keys())
    for i in range(len(strategy_list)):
        for j in range(i+1, len(strategy_list)):
            s1, s2 = strategy_list[i], strategy_list[j]
            feat1 = results[s1]
            feat2 = results[s2]
            
            similarity = F.cosine_similarity(feat1, feat2)
            print(f"{s1} vs {s2}: {similarity.item():.4f}")
    
    return results


if __name__ == '__main__':
    # 测试示例
    test_image = "data/mimic-cxr/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    
    if os.path.exists(test_image):
        results = compare_strategies(test_image)
    else:
        print("请修改test_image路径为实际存在的图像")