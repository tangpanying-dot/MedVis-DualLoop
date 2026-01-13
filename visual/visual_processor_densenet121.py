# visual_processor_fixed.py
"""
完全修复版：解决所有维度和多进程问题
- 修复tensor维度错误
- 真正的并行预处理
- 高GPU利用率
"""
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import argparse
from PIL import Image
from skimage.transform import resize

from visual_encoder_densenet121 import ImprovedVisualEncoder


class MimicCxrDatasetOptimized(Dataset):
    """优化的数据集类 - 在worker进程中预处理"""
    def __init__(self, df, image_root):
        self.samples = []
        
        for idx, row in df.iterrows():
            study_id = row['study_id']
            image_paths = json.loads(row['image_paths'])
            views = json.loads(row['view_positions'])
            
            for img_path, view in zip(image_paths, views):
                path_segment = img_path.replace("files/", "", 1)
                jpg_path = os.path.splitext(path_segment)[0] + ".jpg"
                full_path = os.path.join(image_root, jpg_path)
                
                self.samples.append({
                    'study_id': study_id,
                    'image_path': full_path,
                    'view': view
                })
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess_image_static(self, image_path):
        """静态图像预处理方法"""
        try:
            # 读取图像
            image = Image.open(image_path)
            
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            # 转换为numpy数组
            img_np = np.array(image)
            
            # 调整大小为224x224
            img_resized = resize(img_np, (224, 224), preserve_range=True)
            
            # 归一化到[-1024, 1024]范围
            img_normalized = img_resized / 255.0
            img_normalized = (img_normalized * 2048) - 1024
            
            # 转换为tensor，保持2D: (224, 224)
            img_tensor = torch.from_numpy(img_normalized).float()
            
            return img_tensor, True
        except Exception as e:
            return None, False
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # ← 关键：在worker进程中预处理
        img_tensor, success = self.preprocess_image_static(sample['image_path'])
        
        return {
            'study_id': sample['study_id'],
            'image_tensor': img_tensor,  # (224, 224)
            'view': sample['view'],
            'success': success
        }


def collate_fn(batch):
    """批处理collate函数"""
    return batch


def process_batch_fixed(encoder, batch_samples, strategy):
    """
    修复版批处理函数
    """
    features_by_study = defaultdict(list)
    views_by_study = defaultdict(list)
    
    # 收集有效的tensor
    batch_tensors = []
    valid_samples = []
    
    for sample in batch_samples:
        if sample['success'] and sample['image_tensor'] is not None:
            batch_tensors.append(sample['image_tensor'])
            valid_samples.append(sample)
    
    if not batch_tensors:
        return features_by_study, views_by_study
    
    # ← 修复：正确处理维度
    # batch_tensors: List[(224, 224)]
    # stack: (B, 224, 224)
    # unsqueeze(1): (B, 1, 224, 224) ✓
    batch_input = torch.stack(batch_tensors, dim=0)  # (B, 224, 224)
    batch_input = batch_input.unsqueeze(1)  # (B, 1, 224, 224)
    batch_input = batch_input.to(encoder.device)
    
    with torch.no_grad():
        # 提取特征图
        feature_maps = encoder.model.features(batch_input)  # (B, 1024, 7, 7)
        
        # 根据策略处理每个样本的特征
        batch_features = []
        for i in range(len(feature_maps)):
            feature_map = feature_maps[i:i+1]  # (1, 1024, 7, 7)
            
            # 调用对应策略的提取方法
            if strategy == 'global_avg':
                feat = encoder._extract_global_avg(feature_map)
            elif strategy == 'multi_scale':
                feat = encoder._extract_multi_scale(feature_map)
            elif strategy == 'spatial_attention':
                feat = encoder._extract_spatial_attention(feature_map)
            elif strategy == 'roi_pooling':
                feat = encoder._extract_roi_pooling(feature_map)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            batch_features.append(feat)
    
    # 按study分组
    for i, sample in enumerate(valid_samples):
        study_id = sample['study_id']
        features_by_study[study_id].append(batch_features[i])
        views_by_study[study_id].append(sample['view'])
    
    return features_by_study, views_by_study


def fuse_study_features(feature_list, views, method='weighted'):
    """融合同一study的多个图像特征"""
    if not feature_list:
        return None
    
    device = feature_list[0].device
    
    if method == 'mean':
        stacked = torch.cat(feature_list, dim=0)
        fused = torch.mean(stacked, dim=0, keepdim=True)
    
    elif method == 'weighted':
        weight_map = {'PA': 1.2, 'AP': 1.2, 'LATERAL': 0.8, 'LL': 0.8}
        default_weight = 1.0
        
        weights = [weight_map.get(str(v).upper(), default_weight) for v in views]
        weights_tensor = torch.tensor(weights, device=device).float()
        weights_normalized = weights_tensor / torch.sum(weights_tensor)
        
        stacked = torch.cat(feature_list, dim=0)
        weighted = stacked * weights_normalized.unsqueeze(1)
        fused = torch.sum(weighted, dim=0, keepdim=True)
    
    elif method == 'attention':
        fused = fuse_study_features(feature_list, views, method='weighted')
    
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused


def main(args):
    """主函数"""
    print("=" * 70)
    print("完全修复版视觉特征提取")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.strategy)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n配置:")
    print(f"  策略: {args.strategy}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  融合方法: {args.fusion_method}")
    print(f"  批大小: {args.batch_size}")
    print(f"  工作线程: {args.num_workers}")
    print(f"  输出目录: {output_dir}")
    
    # 1. 初始化编码器
    print(f"\n1. 初始化编码器...")
    encoder = ImprovedVisualEncoder(
        feature_dim=args.feature_dim,
        model_name=args.model_name,
        strategy=args.strategy
    )
    
    # 2. 加载数据
    print(f"\n2. 加载数据集...")
    df = pd.read_csv(args.csv_file)
    dataset = MimicCxrDatasetOptimized(df, args.image_root)
    
    # 3. 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False
    )
    
    print(f"   总图像数: {len(dataset)}")
    print(f"   总batch数: {len(dataloader)}")
    
    # 4. 批量提取特征
    print(f"\n3. 批量提取特征...")
    print(f"   提示: 如果GPU利用率低，请检查num_workers设置")
    
    all_features = defaultdict(list)
    all_views = defaultdict(list)
    
    for batch_samples in tqdm(dataloader, desc="处理进度"):
        features_dict, views_dict = process_batch_fixed(
            encoder, batch_samples, args.strategy
        )
        
        for study_id in features_dict:
            all_features[study_id].extend(features_dict[study_id])
            all_views[study_id].extend(views_dict[study_id])
    
    # 5. 融合并保存
    print(f"\n4. 融合并保存特征...")
    
    saved_count = 0
    for study_id, feature_list in tqdm(all_features.items(), desc="融合保存"):
        views = all_views[study_id]
        
        # 融合
        fused = fuse_study_features(feature_list, views, method=args.fusion_method)
        
        if fused is not None:
            output_path = os.path.join(output_dir, f"{study_id}.npy")
            np.save(output_path, fused.cpu().squeeze().numpy())
            saved_count += 1
    
    # 6. 统计
    print(f"\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"处理图像: {len(dataset)} 张")
    print(f"保存study: {saved_count} 个")
    print(f"平均每study图像数: {len(dataset) / len(all_features):.2f}")
    print(f"特征维度: {args.feature_dim}")
    print(f"输出目录: {output_dir}")
    
    # 保存元数据
    metadata = {
        'strategy': args.strategy,
        'feature_dim': args.feature_dim,
        'fusion_method': args.fusion_method,
        'model_name': args.model_name,
        'num_studies': saved_count,
        'num_images': len(dataset)
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"元数据已保存: {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='完全修复版视觉特征提取')
    
    # 数据路径
    parser.add_argument('--csv_file', type=str, 
                        default='data/processed_dataset.csv',
                        help='数据集CSV文件')
    parser.add_argument('--image_root', type=str,
                        default='data/mimic-cxr/images',
                        help='图像根目录')
    parser.add_argument('--output_dir', type=str,
                        default='visual/visual_features',
                        help='输出目录')
    
    # 模型配置
    parser.add_argument('--model_name', type=str,
                        default='densenet121-res224-all',
                        help='TorchXRayVision模型名称')
    parser.add_argument('--feature_dim', type=int,
                        default=1024,
                        help='特征维度')
    
    # 特征提取策略
    parser.add_argument('--strategy', type=str,
                        default='multi_scale',
                        choices=['global_avg', 'multi_scale', 'spatial_attention', 'roi_pooling'],
                        help='特征提取策略')
    
    # 融合方法
    parser.add_argument('--fusion_method', type=str,
                        default='weighted',
                        choices=['mean', 'weighted', 'attention'],
                        help='Study特征融合方法')
    
    # 批处理配置
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='批大小')
    parser.add_argument('--num_workers', type=int,
                        default=16,
                        help='数据加载线程数')
    
    args = parser.parse_args()
    
    main(args)