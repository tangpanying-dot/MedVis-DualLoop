# visual/visual_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel
from PIL import Image
import numpy as np

class RadDinoVisualEncoder:
    """
    RAD-DINO 视觉编码器 (Microsoft) - 修复增强版
    
    核心修改：
    1. 启用 518x518 分辨率 (RAD-DINO 的原生最佳分辨率)
    2. 增加 Letterbox Padding (等比例缩放+黑边)，防止医学影像长宽比失真
    """
    def __init__(self, model_name="microsoft/rad-dino", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"正在加载 RAD-DINO: {model_name}...")
        
        # 加载 HuggingFace 模型
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # RAD-DINO (ViT-Base) 参数
        self.feature_dim = 768  
        self.patch_size = 14    
        # 注意：518 / 14 = 37, 所以 patch Grid 是 37x37
        self.num_patches = 1369 # 37*37 = 1369 (原先是 196)
        
        print(f"  - 特征维度: {self.feature_dim}")
        print(f"  - 视觉序列长度: {self.num_patches} (37x37 grid @ 518px)")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理: 等比例缩放 + 填充黑边 (Letterbox Padding) -> 518x518
        
        重要：防止把 '长方形肺部' 压成 '正方形'，导致心脏被拉宽误报 Cardiomegaly。
        """
        # ✅ [关键修改] 改为 RAD-DINO 最佳分辨率 518
        target_size = 518 
        
        try:
            image = Image.open(image_path).convert('RGB')
            w, h = image.size
            
            # --- 核心修改开始: 保持比例缩放 ---
            
            # 1. 计算缩放系数 (以长边为准，缩放到 target_size)
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 2. 缩放图片 (使用 BICUBIC 插值保持微小病灶细节)
            image_resized = image.resize((new_w, new_h), Image.BICUBIC)
            
            # 3. 创建全黑背景 (Canvas)
            new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            
            # 4. 将图片粘贴到中心位置 (居中)
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            new_image.paste(image_resized, (paste_x, paste_y))
            
            # --- 核心修改结束 ---

            # 转为 numpy 并归一化
            img_np = np.array(new_image)
            
            # 标准 ImageNet 归一化 (RAD-DINO 训练标准)
            img_normalized = img_np / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_normalized = (img_normalized - mean) / std
            
            # (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1)
            return img_tensor
            
        except Exception as e:
            print(f"⚠️ 图片处理失败 {image_path}: {e}")
            return None

    def extract_features_batch(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Args: image_tensors (B, 3, 518, 518)
        Returns: features (B, 1369, 768)
        """
        with torch.no_grad():
            image_tensors = image_tensors.to(self.device)
            
            # RAD-DINO forward
            outputs = self.model(pixel_values=image_tensors)
            
            # last_hidden_state: (B, 1370, 768) -> [CLS] + 1369 Patches
            last_hidden_state = outputs.last_hidden_state
            
            # 丢弃 [CLS] token，保留 1369 个空间 Patch
            patch_features = last_hidden_state[:, 1:, :] # (B, 1369, 768)
            
            return patch_features