# visual/visual_processor.py
"""
RAD-DINO 特征提取器 (最终优化版)

核心特性:
1. 使用 RadDinoVisualEncoder 提取 (1369, 768) 网格特征 (37x37 grid @ 518px)
2. 多视角拼接：同一 Study 的多张图片特征沿序列维度拼接 -> (N*1369, 768)
3. MAX_VIEWS=4 限制：覆盖99.93%样本，优化训练效率
4. 断点续传：自动跳过已提取的study
5. 内存优化：Buffer 机制流式处理，防止 OOM
6. 路径兼容：自动处理 files/和images/前缀

版本: v1.0 Final
日期: 2025-11-29
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

# HuggingFace镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入 RAD-DINO Encoder
from visual.visual_encoder_raddino import RadDinoVisualEncoder


class MimicCxrDatasetRadDino(Dataset):
    """
    MIMIC-CXR 数据集加载器 (RAD-DINO 适配版)
    
    功能:
    - 复用 Encoder 的预处理方法确保一致性
    - 支持断点续传（跳过已完成的study）
    - 兼容多种路径格式
    """
    def __init__(self, df, image_root, encoder_instance, existing_studies=None):
        self.samples = []
        self.encoder = encoder_instance 
        self.existing_studies = existing_studies or set()
        
        print("正在解析数据集路径...")
        skipped_existing = 0
        skipped_error = 0
        
        for idx, row in df.iterrows():
            study_id = str(row['study_id'])
            
            # ✅ 断点续传：跳过已存在的study
            if study_id in self.existing_studies:
                skipped_existing += 1
                continue
            
            try:
                # 兼容JSON字符串或列表
                image_paths = json.loads(row['image_paths']) if isinstance(row['image_paths'], str) else row['image_paths']
                views = json.loads(row['view_positions']) if isinstance(row['view_positions'], str) else row['view_positions']
            except Exception as e:
                skipped_error += 1
                continue 

            # 确保路径和视角数量对齐
            if len(image_paths) != len(views):
                skipped_error += 1
                continue
            
            for img_path, view in zip(image_paths, views):
                # ✅ 路径清洗：兼容多种格式
                # 支持: files/p10/.../*.dcm 或 images/p10/.../*.jpg
                
                # 1. 去掉前缀 (files/ 或 images/)
                if img_path.startswith("files/"):
                    path_segment = img_path.replace("files/", "", 1)
                elif img_path.startswith("images/"):
                    path_segment = img_path.replace("images/", "", 1)
                else:
                    path_segment = img_path
                
                # 2. 替换扩展名 .dcm -> .jpg
                if path_segment.endswith(".dcm"):
                    path_segment = os.path.splitext(path_segment)[0] + ".jpg"
                
                # 3. 构建完整路径
                # image_root 已经是 data/mimic-cxr/images
                full_path = os.path.join(image_root, path_segment)
                
                self.samples.append({
                    'study_id': study_id,
                    'image_path': full_path,
                    'view': view
                })
        
        # 统计信息
        if skipped_existing > 0:
            print(f"✅ 跳过已存在的study: {skipped_existing} 个")
        if skipped_error > 0:
            print(f"⚠️  跳过解析错误的行: {skipped_error} 个")
        print(f"📊 待处理图片总数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 使用 Encoder 的标准预处理
        # 返回: (3, 518, 518) tensor
        img_tensor = self.encoder.preprocess_image(sample['image_path'])
        
        success = img_tensor is not None
        
        return {
            'study_id': sample['study_id'],
            'image_tensor': img_tensor, 
            'view': sample['view'],
            'image_path': sample['image_path'],
            'success': success
        }


def collate_fn(batch):
    """过滤失败的样本"""
    return [b for b in batch if b is not None and b['success']]


def process_batch_raddino(encoder, batch_samples):
    """
    RAD-DINO 批处理提取
    
    Args:
        encoder: RadDinoVisualEncoder 实例
        batch_samples: 批次样本列表
        
    Returns:
        features_by_study: {study_id: [{'features': tensor, 'view': str}, ...]}
    """
    features_by_study = defaultdict(list)
    
    batch_tensors = []
    valid_samples = []
    
    # 1. 收集有效样本
    for sample in batch_samples:
        if sample['success'] and sample['image_tensor'] is not None:
            batch_tensors.append(sample['image_tensor'])
            valid_samples.append(sample)
    
    if not batch_tensors:
        return features_by_study
    
    # 2. 批量提取特征
    # Input:  (B, 3, 518, 518)
    # Output: (B, 1369, 768)
    batch_input = torch.stack(batch_tensors, dim=0)
    batch_features = encoder.extract_features_batch(batch_input)
    
    # 3. 分配到各study
    for i, sample in enumerate(valid_samples):
        study_id = sample['study_id']
        view = sample['view']
        
        features_by_study[study_id].append({
            'features': batch_features[i].cpu(),  # 转CPU释放显存
            'view': view
        })
    
    return features_by_study


def concat_study_features(feature_view_list):
    """
    拼接同一 Study 的多视角特征
    
    核心逻辑:
    1. 按优先级排序视角 (PA -> AP -> LATERAL -> LL -> 其他)
    2. 限制最大视角数=4 (覆盖99.93%样本)
    3. 沿序列维度拼接特征
    
    Args:
        feature_view_list: [{'features': (1369,768), 'view': str}, ...]
        
    Returns:
        concat_features: (N*1369, 768) tensor
        metadata: dict with num_views, total_tokens, etc.
    """
    if not feature_view_list:
        return None, None
    
    # 1. 视角优先级排序
    view_order = ['PA', 'AP', 'LATERAL', 'LL']
    
    def get_sort_key(x):
        view_val = x['view']
        view_str = str(view_val) if view_val is not None else ""
        view_upper = view_str.upper()
        if view_upper in view_order:
            return view_order.index(view_upper)
        return 999

    sorted_items = sorted(feature_view_list, key=get_sort_key)
    
    # 2. ✅ 限制最大视角数
    MAX_VIEWS = 4  # 覆盖99.93%样本，平衡效率与覆盖率
    
    original_num_views = len(sorted_items)
    if len(sorted_items) > MAX_VIEWS:
        # 保留优先级最高的前4个视角
        sorted_items = sorted_items[:MAX_VIEWS]
    
    # 3. 提取特征和视角列表
    features_list = [item['features'] for item in sorted_items]
    views_list = [str(item['view']) for item in sorted_items]
    
    # 4. 拼接特征
    # 单视角: (1369, 768)
    # 多视角: (N*1369, 768) 沿dim=0拼接
    concat_features = torch.cat(features_list, dim=0)
    
    # 5. 生成元数据
    metadata = {
        'num_views': len(views_list),              # 实际使用的视角数
        'original_num_views': original_num_views,  # 原始视角数
        'total_tokens': concat_features.shape[0],  # 总token数 = N*1369
        'views': views_list,                       # 视角列表
        'patch_size': 1369,                        # 每个视角的patch数
        'feature_dim': 768,                        # RAD-DINO特征维度
        'truncated': original_num_views > MAX_VIEWS  # 是否被截断
    }
    
    return concat_features, metadata


def scan_existing_features(output_dir):
    """
    扫描已存在的特征文件，支持断点续传
    
    Returns:
        existing_studies: set of study_ids (str)
    """
    if not os.path.exists(output_dir):
        return set()
    
    existing_files = os.listdir(output_dir)
    existing_studies = set([
        f.replace('.npy', '') 
        for f in existing_files 
        if f.endswith('.npy')
    ])
    return existing_studies


def main(args):
    print("=" * 80)
    print("RAD-DINO 特征提取器 (Final Optimized Version)")
    print("=" * 80)
    print(f"配置:")
    print(f"  分辨率: 518x518")
    print(f"  特征维度: (1369, 768) per view")
    print(f"  最大视角数: 4 (覆盖99.93%)")
    print(f"  存储格式: float16")
    print("=" * 80)
    
    # 输出目录
    output_dir = os.path.join(args.output_dir, "rad_dino")
    os.makedirs(output_dir, exist_ok=True)
    
    # ✅ 断点续传：扫描已存在的特征
    print(f"\n🔍 扫描已存在的特征文件...")
    existing_studies = scan_existing_features(output_dir)
    if existing_studies:
        print(f"✅ 找到 {len(existing_studies)} 个已完成的study (将自动跳过)")
    else:
        print(f"📝 未找到已存在特征，将从头开始")
    
    # 1. 初始化 Encoder
    print(f"\n1. 初始化 RAD-DINO Encoder...")
    print(f"   模型: {args.model_name}")
    encoder = RadDinoVisualEncoder(model_name=args.model_name)
    
    # 2. 加载数据集
    print(f"\n2. 加载数据集...")
    print(f"   CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # 按 study_id 排序（流式Buffer需要）
    print("   正在按 study_id 排序...")
    df = df.sort_values('study_id')
    
    # 创建数据集
    dataset = MimicCxrDatasetRadDino(
        df, 
        args.image_root, 
        encoder,
        existing_studies=existing_studies
    )
    
    if len(dataset) == 0:
        print("\n✅ 所有特征已提取完成，无需处理!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 必须False以配合流式保存
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 3. 特征提取主循环
    print(f"\n3. 开始提取特征...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num workers: {args.num_workers}")
    
    study_buffer = defaultdict(list)
    saved_count = 0
    error_count = 0
    truncated_count = 0
    
    pbar = tqdm(dataloader, desc="提取进度", total=len(dataloader))
    
    for batch_samples in pbar:
        if not batch_samples:
            continue
        
        # 3.1 提取当前batch特征
        try:
            features_dict = process_batch_raddino(encoder, batch_samples)
        except Exception as e:
            error_count += 1
            print(f"\n⚠️  Batch处理错误: {e}")
            continue
        
        # 3.2 加入Buffer
        for study_id, items in features_dict.items():
            study_buffer[study_id].extend(items)
        
        # 3.3 流式保存：检查哪些study已完成
        current_batch_ids = set(features_dict.keys())
        buffer_ids = list(study_buffer.keys())
        
        for sid in buffer_ids:
            if sid not in current_batch_ids:
                # 该study的所有图片已处理完
                views = study_buffer[sid]
                
                # 拼接特征
                concat_feat, meta = concat_study_features(views)
                
                if concat_feat is not None:
                    try:
                        # 统计截断
                        if meta.get('truncated', False):
                            truncated_count += 1
                        
                        # 保存特征 (float16)
                        save_path = os.path.join(output_dir, f"{sid}.npy")
                        np.save(save_path, concat_feat.numpy().astype(np.float16))
                        
                        # 保存元数据
                        meta_path = os.path.join(output_dir, f"{sid}_meta.json")
                        with open(meta_path, 'w') as f:
                            json.dump(meta, f)
                        
                        saved_count += 1
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'saved': saved_count, 
                            'errors': error_count,
                            'truncated': truncated_count
                        })
                        
                    except Exception as e:
                        error_count += 1
                        print(f"\n⚠️  保存错误 {sid}: {e}")
                
                # ✅ 从内存中删除，释放空间
                del study_buffer[sid]
    
    # 4. 保存剩余Buffer中的study
    print("\n正在保存剩余缓存...")
    for sid, views in study_buffer.items():
        concat_feat, meta = concat_study_features(views)
        if concat_feat is not None:
            try:
                if meta.get('truncated', False):
                    truncated_count += 1
                
                np.save(
                    os.path.join(output_dir, f"{sid}.npy"), 
                    concat_feat.numpy().astype(np.float16)
                )
                with open(os.path.join(output_dir, f"{sid}_meta.json"), 'w') as f:
                    json.dump(meta, f)
                saved_count += 1
            except Exception as e:
                error_count += 1
                print(f"⚠️  保存错误 {sid}: {e}")
    
    # 5. 完成统计
    print(f"\n" + "=" * 80)
    print(f"✅ 特征提取完成!")
    print("=" * 80)
    print(f"  成功保存: {saved_count:,} 个study")
    print(f"  处理错误: {error_count:,} 个")
    
    if truncated_count > 0:
        print(f"  视角截断: {truncated_count:,} 个 ({truncated_count/saved_count*100:.2f}%) - 超过4视角")
    
    total_completed = len(existing_studies) + saved_count
    print(f"  总计完成: {total_completed:,} 个")
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"📊 特征格式: [N*1369, 768] (N ≤ 4 views, 518px)")
    print(f"💾 存储精度: float16")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RAD-DINO 特征提取脚本 (Final Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python visual/visual_processor.py --batch_size 32 --num_workers 4
  
  # 断点续传（直接重新运行即可）
  python visual/visual_processor.py
  
配置说明:
  - batch_size: 建议16-32 (4090可用32)
  - num_workers: 建议4-8
  - MAX_VIEWS=4: 覆盖99.93%样本，优化训练效率
        """
    )
    
    # 路径配置
    parser.add_argument('--csv_file', type=str, 
                        default='data/processed_dataset.csv',
                        help='CSV文件路径')
    parser.add_argument('--image_root', type=str, 
                        default='data/mimic-cxr/images',
                        help='图片根目录')
    parser.add_argument('--output_dir', type=str, 
                        default='visual/visual_features',
                        help='特征保存目录')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, 
                        default='microsoft/rad-dino',
                        help='HuggingFace模型ID')
    
    # 运行配置
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批大小 (建议16-32)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载线程数 (建议4-8)')
    
    args = parser.parse_args()
    
    main(args)