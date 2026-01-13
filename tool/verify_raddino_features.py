#!/usr/bin/env python3
"""
verify_raddino_features.py
RAD-DINO 特征质量验证脚本 (518px / 1369 patches 适配版)

功能:
1. 维度与数值检查
2. 空间激活热力图 (37x37)
3. t-SNE语义聚类
4. 相似度分析
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class RadDinoValidator:
    def __init__(self, feature_dir, csv_path, output_dir="visual/validation_raddino"):
        self.feature_dir = feature_dir
        self.csv_path = csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在加载数据表: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df['study_id'] = self.df['study_id'].astype(str)
        
        # 获取特征文件
        self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        self.available_studies = set([f.replace('.npy', '') for f in self.feature_files])
        print(f"找到 {len(self.feature_files)} 个特征文件")
        
        # ✅ RAD-DINO @ 518px 标准参数
        self.TARGET_DIM = 768        # 特征维度
        self.PATCH_SIZE = 1369       # 每个视角的patch数 (37x37)
        self.GRID_SIZE = 37          # 空间网格大小
        
        print(f"\n配置:")
        print(f"  特征维度: {self.TARGET_DIM}")
        print(f"  Patch数/视角: {self.PATCH_SIZE}")
        print(f"  空间网格: {self.GRID_SIZE}x{self.GRID_SIZE}")

    def _get_diagnosis(self, row):
        """简单的诊断标签提取，用于t-SNE着色"""
        # 合并所有文本列
        text = ""
        for col in ['history', 'findings', 'impression', 'report']:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    text += str(val).lower() + " "
        
        if not text.strip(): 
            return 'Unknown'

        # 优先级匹配
        if 'pneumothorax' in text: 
            return 'Pneumothorax'
        if 'pleural effusion' in text or 'effusion' in text: 
            return 'Effusion'
        if 'cardiomegaly' in text or 'enlarged heart' in text: 
            return 'Cardiomegaly'
        if 'pneumonia' in text: 
            return 'Pneumonia'
        if 'edema' in text: 
            return 'Edema'
        if 'fracture' in text: 
            return 'Fracture'
        if 'consolidation' in text:
            return 'Consolidation'
        if 'atelectasis' in text:
            return 'Atelectasis'
        if 'no acute' in text or 'unremarkable' in text or 'normal' in text: 
            return 'Normal'
            
        return 'Other'

    def check_integrity(self, sample_size=200):
        """1. 维度与数值检查"""
        print("\n" + "="*80)
        print("1. RAD-DINO 特征完整性检查")
        print("="*80)
        
        valid_count = 0
        error_count = 0
        similarity_warnings = 0
        
        check_list = self.feature_files[:sample_size]
        
        stats = {
            'shapes': [],
            'num_views': [],
            'dtypes': [],
            'nan_count': 0,
            'inf_count': 0,
            'high_similarity': 0
        }
        
        for fname in tqdm(check_list, desc="检查特征"):
            try:
                study_id = fname.replace('.npy', '')
                npy_path = os.path.join(self.feature_dir, fname)
                json_path = os.path.join(self.feature_dir, f"{study_id}_meta.json")
                
                # 加载特征
                feat = np.load(npy_path)
                
                # 加载元数据
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                else:
                    print(f"⚠️  {study_id}: 缺少元数据文件")
                    error_count += 1
                    continue
                
                # 统计信息
                stats['shapes'].append(feat.shape)
                stats['num_views'].append(meta.get('num_views', 0))
                stats['dtypes'].append(str(feat.dtype))
                
                # ✅ 检查 NaN/Inf
                if np.isnan(feat).any():
                    print(f"❌ {study_id}: 包含NaN")
                    stats['nan_count'] += 1
                    error_count += 1
                    continue
                    
                if np.isinf(feat).any():
                    print(f"❌ {study_id}: 包含Inf")
                    stats['inf_count'] += 1
                    error_count += 1
                    continue
                
                # ✅ 检查维度: 应该是 (N * 1369, 768)
                N_views = meta.get('num_views', 0)
                expected_rows = N_views * self.PATCH_SIZE
                
                if feat.shape[0] != expected_rows:
                    print(f"❌ {study_id}: 行数错误! 预期 {expected_rows} (视角{N_views}×1369), 实际 {feat.shape[0]}")
                    error_count += 1
                    continue
                    
                if feat.shape[1] != self.TARGET_DIM:
                    print(f"❌ {study_id}: 列数错误! 预期 768, 实际 {feat.shape[1]}")
                    error_count += 1
                    continue
                
                # ✅ 检查多视角差异性 (防止两张图完全一样)
                if N_views > 1:
                    # 取第一张图和第二张图
                    v1 = feat[0:self.PATCH_SIZE]                    # (1369, 768)
                    v2 = feat[self.PATCH_SIZE:2*self.PATCH_SIZE]    # (1369, 768)
                    
                    # 计算余弦相似度
                    v1_flat = v1.flatten()
                    v2_flat = v2.flatten()
                    
                    sim = np.dot(v1_flat, v2_flat) / (
                        np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat) + 1e-8
                    )
                    
                    if sim > 0.95:
                        print(f"⚠️  {study_id}: 多视角特征过于相似 (Sim={sim:.4f}), 可能是重复图片")
                        similarity_warnings += 1
                        stats['high_similarity'] += 1
                
                valid_count += 1
                
            except Exception as e:
                print(f"❌ 读取失败 {fname}: {e}")
                error_count += 1
        
        # 统计报告
        print("\n" + "="*80)
        print("检查结果统计")
        print("="*80)
        print(f"✅ 正常: {valid_count}")
        print(f"❌ 异常: {error_count}")
        print(f"⚠️  高相似度警告: {similarity_warnings}")
        
        # 形状分布
        if stats['shapes']:
            from collections import Counter
            shape_dist = Counter([str(s) for s in stats['shapes']])
            print(f"\n特征形状分布 (前5):")
            for shape, count in shape_dist.most_common(5):
                print(f"  {shape:20s}: {count:>5} 个")
        
        # 视角分布
        if stats['num_views']:
            view_dist = Counter(stats['num_views'])
            print(f"\n视角数量分布:")
            for nv, count in sorted(view_dist.items()):
                print(f"  {nv} 视角: {count:>5} 个 ({count/len(stats['num_views'])*100:.1f}%)")
        
        # 数据类型
        if stats['dtypes']:
            dtype_dist = Counter(stats['dtypes'])
            print(f"\n数据类型分布:")
            for dt, count in dtype_dist.items():
                print(f"  {dt}: {count}")
        
        return valid_count, error_count

    def visualize_heatmaps(self, num_samples=5):
        """2. 生成 37x37 空间热力图"""
        print("\n" + "="*80)
        print("2. 生成 RAD-DINO 空间激活热力图 (37x37)")
        print("="*80)
        
        # 筛选有多视角的样本
        candidates = []
        for fname in self.feature_files:
            if len(candidates) >= num_samples: 
                break
            
            sid = fname.replace('.npy', '')
            meta_path = os.path.join(self.feature_dir, f"{sid}_meta.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    if meta.get('num_views', 0) >= 2:
                        candidates.append(sid)
            except: 
                continue
        
        if not candidates:
            print("⚠️  未找到多视角样本，尝试单视角...")
            candidates = [self.feature_files[i].replace('.npy', '') 
                         for i in range(min(num_samples, len(self.feature_files)))]
        
        print(f"选择 {len(candidates)} 个样本绘制热力图")
        
        for study_id in candidates:
            try:
                npy_path = os.path.join(self.feature_dir, f"{study_id}.npy")
                json_path = os.path.join(self.feature_dir, f"{study_id}_meta.json")
                
                feat = np.load(npy_path)
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                
                views = meta.get('views', ['Unknown'])
                num_views = meta.get('num_views', 1)
                
                # 绘图
                fig, axes = plt.subplots(1, num_views, figsize=(5 * num_views, 5))
                if num_views == 1: 
                    axes = [axes]
                
                fig.suptitle(f"RAD-DINO Spatial Activation (37×37) - Study {study_id}", 
                           fontsize=14, fontweight='bold')
                
                for i in range(num_views):
                    # 切片: 该视角对应的 1369 行
                    start_idx = i * self.PATCH_SIZE
                    end_idx = (i + 1) * self.PATCH_SIZE
                    
                    view_feat = feat[start_idx:end_idx, :]  # (1369, 768)
                    
                    # 计算激活强度 (L2 Norm)
                    activation = np.linalg.norm(view_feat, axis=1)  # (1369,)
                    
                    # 归一化
                    if activation.max() > activation.min():
                        activation = (activation - activation.min()) / (activation.max() - activation.min())
                    
                    # 重塑为 37x37
                    heatmap = activation.reshape(self.GRID_SIZE, self.GRID_SIZE)
                    
                    ax = axes[i]
                    im = ax.imshow(heatmap, cmap='inferno', interpolation='bicubic')
                    
                    view_name = views[i] if i < len(views) else f"View {i+1}"
                    ax.set_title(f"{view_name}", fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, f"heatmap_{study_id}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"✅ 已保存: {save_path}")
                
            except Exception as e:
                print(f"⚠️  热力图生成失败 {study_id}: {e}")
                continue

    def visualize_tsne(self, max_samples=800):
        """3. 全局语义聚类"""
        print("\n" + "="*80)
        print("3. 运行 t-SNE 语义聚类")
        print("="*80)
        
        # 随机采样
        sample_ids = list(self.available_studies)
        if len(sample_ids) > max_samples:
            import random
            random.seed(42)
            sample_ids = random.sample(sample_ids, max_samples)
        
        print(f"采样 {len(sample_ids)} 个study")
        
        features = []
        labels = []
        study_ids = []
        
        print("准备数据中...")
        for sid in tqdm(sample_ids, desc="加载特征"):
            # 找标签
            if sid not in self.df['study_id'].values: 
                continue
            row = self.df[self.df['study_id'] == sid].iloc[0]
            label = self._get_diagnosis(row)
            
            # 找特征
            try:
                feat = np.load(os.path.join(self.feature_dir, f"{sid}.npy"))
                # 平均池化: 把 (N*1369, 768) -> (768,)
                feat_pooled = np.mean(feat, axis=0)
                
                features.append(feat_pooled)
                labels.append(label)
                study_ids.append(sid)
            except Exception as e:
                # print(f"⚠️  加载失败 {sid}: {e}")
                continue
        
        if len(features) < 10:
            print("❌ 有效样本太少，无法进行t-SNE")
            return
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"\n开始 t-SNE 降维...")
        print(f"  样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}")
        
        # 统计标签分布
        from collections import Counter
        label_dist = Counter(y)
        print(f"\n标签分布:")
        for label, count in label_dist.most_common():
            print(f"  {label:20s}: {count:>4} ({count/len(y)*100:.1f}%)")
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', 
                   learning_rate='auto', perplexity=min(30, len(X)-1))
        X_2d = tsne.fit_transform(X)
        
        # 绘图
        plt.figure(figsize=(14, 10))
        
        # 使用seaborn绘制
        unique_labels = sorted(set(y))
        palette = sns.color_palette("bright", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       label=label, alpha=0.7, s=60, 
                       color=palette[i], edgecolors='black', linewidth=0.5)
        
        plt.title(f"t-SNE of RAD-DINO Features (Pooled, n={len(X)})", 
                 fontsize=16, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  title="Diagnosis", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "tsne_clusters.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"\n✅ 已保存聚类图: {save_path}")

    def analyze_similarity(self, num_samples=100):
        """4. 相似度分析 (检测数据泄漏)"""
        print("\n" + "="*80)
        print("4. 多视角相似度分析")
        print("="*80)
        
        similarities = []
        
        # 只检查多视角样本
        multi_view_samples = []
        for fname in self.feature_files[:num_samples]:
            sid = fname.replace('.npy', '')
            meta_path = os.path.join(self.feature_dir, f"{sid}_meta.json")
            try:
                with open(meta_path) as f:
                    if json.load(f).get('num_views', 0) > 1:
                        multi_view_samples.append(sid)
            except:
                continue
        
        if not multi_view_samples:
            print("⚠️  未找到多视角样本")
            return
        
        print(f"分析 {len(multi_view_samples)} 个多视角样本")
        
        for sid in tqdm(multi_view_samples, desc="计算相似度"):
            try:
                feat = np.load(os.path.join(self.feature_dir, f"{sid}.npy"))
                with open(os.path.join(self.feature_dir, f"{sid}_meta.json")) as f:
                    num_views = json.load(f)['num_views']
                
                # 计算第一个和第二个视角的相似度
                v1 = feat[0:self.PATCH_SIZE].flatten()
                v2 = feat[self.PATCH_SIZE:2*self.PATCH_SIZE].flatten()
                
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                similarities.append(sim)
                
            except Exception as e:
                continue
        
        if not similarities:
            print("❌ 无法计算相似度")
            return
        
        similarities = np.array(similarities)
        
        print(f"\n相似度统计:")
        print(f"  均值: {similarities.mean():.4f}")
        print(f"  中位数: {np.median(similarities):.4f}")
        print(f"  标准差: {similarities.std():.4f}")
        print(f"  最小值: {similarities.min():.4f}")
        print(f"  最大值: {similarities.max():.4f}")
        
        # 绘制分布
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(similarities.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {similarities.mean():.3f}')
        plt.axvline(np.median(similarities), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(similarities):.3f}')
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Multi-view Feature Similarity Distribution', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, "similarity_distribution.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ 已保存分布图: {save_path}")
        
        # 警告
        high_sim_count = np.sum(similarities > 0.95)
        if high_sim_count > 0:
            print(f"\n⚠️  警告: {high_sim_count} 个样本的多视角相似度 >0.95")
            print(f"   这可能表示:")
            print(f"   1. 相同图片被重复使用")
            print(f"   2. 视角差异很小")
            print(f"   建议检查这些样本的实际图片")

if __name__ == "__main__":
    print("="*80)
    print("RAD-DINO 特征质量验证")
    print("="*80)
    
    # 配置
    FEATURE_DIR = "visual/visual_features/rad_dino"
    CSV_FILE = "data/processed_dataset.csv"
    
    # 检查路径
    if not os.path.exists(FEATURE_DIR):
        print(f"❌ 特征目录不存在: {FEATURE_DIR}")
        exit(1)
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV文件不存在: {CSV_FILE}")
        exit(1)
    
    # 创建验证器
    validator = RadDinoValidator(FEATURE_DIR, CSV_FILE)
    
    # 运行所有检查
    print("\n开始验证流程...")
    
    # 1. 完整性检查
    valid, error = validator.check_integrity(sample_size=200)
    
    # 2. 热力图
    validator.visualize_heatmaps(num_samples=5)
    
    # 3. t-SNE聚类
    validator.visualize_tsne(max_samples=800)
    
    # 4. 相似度分析
    validator.analyze_similarity(num_samples=100)
    
    print("\n" + "="*80)
    print("✅ 验证完成！")
    print("="*80)
    print(f"输出目录: {validator.output_dir}")
    print("\n生成的文件:")
    print("  - heatmap_*.png : 空间激活热力图")
    print("  - tsne_clusters.png : 语义聚类可视化")
    print("  - similarity_distribution.png : 相似度分布")
    print("="*80)