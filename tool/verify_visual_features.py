import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class NewVisualValidator:
    def __init__(self, feature_dir, csv_path, output_dir="visual/validation_new"):
        self.feature_dir = feature_dir
        self.csv_path = csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载CSV
        print(f"正在加载数据表: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df['study_id'] = self.df['study_id'].astype(str)
        
        # 获取现有的特征文件列表
        self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        self.available_studies = set([f.replace('.npy', '') for f in self.feature_files])
        print(f"找到 {len(self.feature_files)} 个特征文件")

    def _get_diagnosis(self, row):
        """
        更详细的诊断提取逻辑 (扩展版)
        """
        # 获取文本并转小写
        history = str(row.get('history', '')).lower()
        findings = str(row.get('findings', '')).lower()
        # MIMIC-CXR CSV通常还有一列叫 'impression'，如果有的话加上更好
        impression = str(row.get('impression', '')).lower() 
        
        text = f"{history} {findings} {impression}"
        
        # 按优先级匹配关键词 (MIMIC-CXR 常见 14 类)
        if 'pneumothorax' in text: return 'Pneumothorax' # 气胸 (比较紧急，放前面)
        if 'pneumonia' in text: return 'Pneumonia'       # 肺炎
        if 'edema' in text: return 'Edema'               # 水肿
        if 'cardiomegaly' in text: return 'Cardiomegaly' # 心脏肥大
        if 'pleural effusion' in text: return 'Effusion' # 积液
        if 'atelectasis' in text: return 'Atelectasis'   # 肺不张
        if 'consolidation' in text: return 'Consolidation' # 实变
        if 'fracture' in text: return 'Fracture'         # 骨折
        if 'nodule' in text or 'mass' in text: return 'Nodule/Mass' # 结节/肿块
        
        # 正常
        if 'no acute' in text or 'no active disease' in text or 'normal' in text: 
            return 'Normal'
            
        return 'Other'

    def check_integrity(self, sample_size=100):
        """1. 基础完整性检查"""
        print("\n=== 1. 正在进行数值完整性检查 ===")
        valid_count = 0
        problem_count = 0
        
        check_list = self.feature_files[:sample_size]
        
        for fname in tqdm(check_list):
            try:
                # 加载特征和meta
                study_id = fname.replace('.npy', '')
                npy_path = os.path.join(self.feature_dir, fname)
                json_path = os.path.join(self.feature_dir, f"{study_id}_meta.json")
                
                feat = np.load(npy_path)
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                
                # 检查1: NaN/Inf
                if np.isnan(feat).any() or np.isinf(feat).any():
                    print(f"❌ {study_id}: 包含 NaN 或 Inf!")
                    problem_count += 1
                    continue
                
                # 检查2: 维度匹配
                expected_len = meta['num_positions']
                if feat.shape[0] != expected_len or feat.shape[1] != 1024:
                    print(f"❌ {study_id}: 维度不匹配! Json说 {expected_len}, 实际 {feat.shape}")
                    problem_count += 1
                    continue
                
                # 检查3: 视角重复性 (防止复制粘贴错误)
                ranges = meta['view_ranges']
                if len(ranges) > 1:
                    views = list(ranges.keys())
                    v1_range = ranges[views[0]]
                    v2_range = ranges[views[1]]
                    
                    feat1 = feat[v1_range[0]:v1_range[1]]
                    feat2 = feat[v2_range[0]:v2_range[1]]
                    
                    # 计算余弦相似度
                    sim = np.dot(feat1.flatten(), feat2.flatten()) / (
                        np.linalg.norm(feat1.flatten()) * np.linalg.norm(feat2.flatten())
                    )
                    
                    if sim > 0.999: # 过于相似
                        print(f"⚠️ {study_id}: 不同视角特征几乎完全相同 (Sim={sim:.4f})")
                        problem_count += 1
                
                valid_count += 1
                
            except Exception as e:
                print(f"❌ 读取错误 {fname}: {str(e)}")
                problem_count += 1

        print(f"检查完成: {valid_count} 个正常, {problem_count} 个异常")

    def visualize_spatial_heatmaps(self, num_samples=3):
        """2. 空间特征热力图可视化 (核心功能)"""
        print("\n=== 2. 正在生成空间激活热力图 ===")
        
        # 随机选取几个多视角的样本
        samples = []
        for fname in self.feature_files:
            study_id = fname.replace('.npy', '')
            json_path = os.path.join(self.feature_dir, f"{study_id}_meta.json")
            with open(json_path, 'r') as f:
                meta = json.load(f)
            if meta['num_views'] >= 2:
                samples.append(study_id)
            if len(samples) >= num_samples:
                break
        
        for study_id in samples:
            npy_path = os.path.join(self.feature_dir, f"{study_id}.npy")
            json_path = os.path.join(self.feature_dir, f"{study_id}_meta.json")
            
            feat = np.load(npy_path) # (N*49, 1024)
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            views = meta['views']
            ranges = meta['view_ranges']
            
            # 创建画布
            fig, axes = plt.subplots(1, len(views), figsize=(4 * len(views), 4))
            if len(views) == 1: axes = [axes]
            
            fig.suptitle(f"Study {study_id} - Spatial Activation Maps", fontsize=14)
            
            for i, view_name in enumerate(views):
                start, end = ranges[view_name]
                # 取出该视角的特征 (49, 1024)
                view_feat = feat[start:end, :]
                
                # 方法: 计算每个位置的 L2 范数 (即激活强度)
                # 这代表模型认为这个 patch 有多"重要"
                activation_map = np.linalg.norm(view_feat, axis=1) # (49,)
                
                # 重塑为 7x7
                heatmap = activation_map.reshape(7, 7)
                
                # 绘图
                im = axes[i].imshow(heatmap, cmap='viridis')
                axes[i].set_title(f"View: {view_name}")
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            save_path = os.path.join(self.output_dir, f"heatmap_{study_id}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"已保存热力图: {save_path}")
            plt.close()

    def visualize_global_clustering(self, max_samples=1000):
        """3. 全局特征聚类 (t-SNE)"""
        print("\n=== 3. 正在进行全局语义聚类 (t-SNE) ===")
        
        features_pool = []
        labels = []
        
        # 采样数据
        df_subset = self.df[self.df['study_id'].isin(self.available_studies)].sample(
            n=min(max_samples, len(self.available_studies)), random_state=42
        )
        
        print("准备特征中...")
        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
            study_id = row['study_id']
            npy_path = os.path.join(self.feature_dir, f"{study_id}.npy")
            
            try:
                feat = np.load(npy_path) # (98, 1024)
                
                # 【关键步骤】做平均池化，把 (98, 1024) 变成 (1024,)
                # 这样才能画点图
                feat_pooled = np.mean(feat, axis=0)
                
                features_pool.append(feat_pooled)
                labels.append(self._get_diagnosis(row))
            except:
                continue
        
        X = np.array(features_pool)
        y = np.array(labels)
        
        print(f"运行 t-SNE (样本数: {len(X)})...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette='tab10', s=60, alpha=0.7)
        plt.title(f"t-SNE of Spatial-Pooled Features (N={len(X)})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        save_path = os.path.join(self.output_dir, "global_cluster_tsne.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"已保存聚类图: {save_path}")
        plt.close()

if __name__ == "__main__":
    # 配置你的路径
    FEATURE_DIR = "visual/visual_features/spatial_preserve"  # 你的特征输出目录
    CSV_FILE = "data/processed_dataset.csv"                  # 你的CSV路径
    
    validator = NewVisualValidator(FEATURE_DIR, CSV_FILE)
    
    # 1. 检查数值
    validator.check_integrity(sample_size=200)
    
    # 2. 可视化 7x7 热力图 (检查空间结构)
    validator.visualize_spatial_heatmaps(num_samples=5)
    
    # 3. 可视化整体分布 (检查语义)
    validator.visualize_global_clustering(max_samples=1000)