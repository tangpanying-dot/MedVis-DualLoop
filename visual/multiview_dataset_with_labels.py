# visual/multiview_dataset_with_labels.py
"""
Multi-View Feature Dataset (v23.0 Final - Label Consistency Fix)

修改日志:
1. [Consistency] load_chexpert_labels: 将 -1.0 (Uncertain) 映射为 0.5。
   -> 原因: 配合 VisualClassifier 的 Sigmoid 输出 (0~1概率)，
      以及 Connector 对 Soft Labels 的支持。
2. [Split] 移除了所有 Hash 拆分逻辑，严格遵循外部传入的 csv_file。
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional

# CheXpert Labels (14 classes)
LABEL_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

def load_chexpert_labels(csv_path: str, normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    加载 CheXpert 标签
    
    关键修改:
    - -1.0 (Uncertain) -> 0.5
    - NaN -> 0.0
    - 1.0 -> 1.0
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"[Warning] CheXpert file not found: {csv_path}. Labels will be zero.")
        return {}

    df = pd.read_csv(csv_path)
    labels_dict = {}
    
    for _, row in df.iterrows():
        # 统一 study_id 格式 (去除 .0 后缀)
        raw_id = row['study_id']
        try:
            study_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            study_id = str(raw_id).replace('.0', '')
        
        labels = []
        for label_name in LABEL_NAMES:
            val = row.get(label_name, 0.0)
            
            if pd.isna(val):
                value = 0.0
            elif val == -1.0:
                # ✅ [关键修改] 统一为 0.5 (不确定 = 弱信号)
                # 配合 Classifier 的 Soft Label 输出 (0~1概率)
                value = 0.5 
            elif val == 1.0:
                value = 1.0
            else:
                value = 0.0 # 其他情况视为0
                
            labels.append(value)
        
        labels_dict[study_id] = np.array(labels, dtype=np.float32)
    
    return labels_dict

class MultiViewFeatureDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        csv_file: str,       # 必须是具体的 split 文件 (train.csv / val.csv / test.csv)
        split: str = 'train',
        dtype=torch.bfloat16,
        chexpert_csv: str = None,
        history_dir: str = None,
        max_history_length: int = 512,
        target_col: str = 'report'
    ):
        super().__init__()
        self.feature_dir = feature_dir
        self.split = split
        self.dtype = dtype
        self.target_col = target_col
        self.history_dir = history_dir
        self.max_history_length = max_history_length
        self.has_history = history_dir is not None and os.path.exists(history_dir)

        # 1. 加载 CheXpert 标签
        self.chexpert_labels = load_chexpert_labels(chexpert_csv)

        # 2. 加载主 CSV (完全信任外部拆分)
        if csv_file and os.path.exists(csv_file):
            print(f"[Dataset] Loading {split.upper()} set from: {csv_file}")
            self.csv_data = pd.read_csv(csv_file)
            
            # 创建索引加速查找
            self.csv_data['study_id_str'] = self.csv_data['study_id'].astype(str)
            self.data_index = self.csv_data.set_index('study_id_str').to_dict('index')
            self.study_ids = list(self.data_index.keys())
            
            print(f"          Loaded {len(self.study_ids):,} samples.")
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

    def load_history_text(self, study_id: str) -> str:
        if not self.has_history: return ""
        path = os.path.join(self.history_dir, f"{study_id}.txt")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read().strip()[:self.max_history_length * 6] 
            except:
                pass
        return ""

    def __len__(self):
        return len(self.study_ids)
    
    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        
        # 1. Visual Features (RAD-DINO)
        feature_path = os.path.join(self.feature_dir, f"{study_id}.npy")
        if not os.path.exists(feature_path): 
            return None # 丢弃缺失图片的样本
        
        try:
            visual_features = np.load(feature_path, mmap_mode='r')
            visual_features = torch.from_numpy(np.array(visual_features)).to(dtype=self.dtype)
        except: 
            return None

        # 2. Report Text (Ground Truth)
        row = self.data_index.get(study_id, {})
        raw_text = row.get(self.target_col, "")
        
        # 简单的回退逻辑
        if pd.isna(raw_text) or len(str(raw_text)) < 5:
            fallback = 'report' if self.target_col == 'findings' else 'findings'
            raw_text = row.get(fallback, "")
        
        report = str(raw_text) if not pd.isna(raw_text) else ""

        # 3. Labels (Soft/Hard GT)
        # 如果缺失，默认为全0
        chexpert_labels = torch.zeros(14, dtype=torch.float32)
        if study_id in self.chexpert_labels:
            chexpert_labels = torch.from_numpy(self.chexpert_labels[study_id])

        # 4. History
        history_text = self.load_history_text(study_id)

        return {
            'study_id': study_id,
            'visual_features': visual_features,
            'length': visual_features.shape[0],
            'report': report,
            'chexpert_labels': chexpert_labels, # 注意：这是来自CSV的标签(包含0.5)
            'history_text': history_text
        }

def collate_fn_with_padding(batch: List[Dict]) -> Dict:
    batch = [b for b in batch if b is not None]
    if not batch: return {}
    
    # Visual Padding
    max_len = max(b['length'] for b in batch)
    feat_dim = batch[0]['visual_features'].shape[-1]
    dtype = batch[0]['visual_features'].dtype
    
    padded_feats = torch.zeros(len(batch), max_len, feat_dim, dtype=dtype)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        l = b['length']
        padded_feats[i, :l] = b['visual_features']
        masks[i, :l] = True
        
    return {
        'visual_features': padded_feats,
        'visual_mask': masks,
        'study_ids': [b['study_id'] for b in batch],
        'reports': [b['report'] for b in batch],
        'chexpert_labels': torch.stack([b['chexpert_labels'] for b in batch]),
        'history_texts': [b['history_text'] for b in batch]
    }

def create_dataloader(
    feature_dir, csv_file, chexpert_csv, target_col='report', 
    batch_size=32, num_workers=4, shuffle=True, split='train', 
    dtype=torch.bfloat16, history_dir=None, max_history_length=512
):
    dataset = MultiViewFeatureDataset(
        feature_dir, csv_file, split, dtype, chexpert_csv, 
        history_dir, max_history_length, target_col
    )
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, collate_fn=collate_fn_with_padding, 
        pin_memory=True, drop_last=(split=='train')
    )