"""
CheXpert Label Processing Utilities

功能:
- 加载和处理CheXpert标签
- 标签归一化和转换
- 标签统计和可视化

Author: Information
Date: 2025-12
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ==================== 配置常量 ====================

# 14类CheXpert标签（按CSV列顺序）
LABEL_NAMES = [
    'Atelectasis',
    'Cardiomegaly', 
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices'
]

# 标签索引映射
LABEL_INDICES = {name: idx for idx, name in enumerate(LABEL_NAMES)}

# 标签权重（处理类别不平衡，可根据数据集调整）
LABEL_WEIGHTS = {
    'Atelectasis': 1.0,
    'Cardiomegaly': 1.2,
    'Consolidation': 1.5,  # 重要异常，加权
    'Edema': 1.3,
    'Enlarged Cardiomediastinum': 1.0,
    'Fracture': 1.5,
    'Lung Lesion': 1.8,  # 罕见但重要
    'Lung Opacity': 1.0,
    'No Finding': 0.5,   # 正常，降权
    'Pleural Effusion': 1.4,
    'Pleural Other': 1.0,
    'Pneumonia': 1.6,
    'Pneumothorax': 1.8,  # 严重，加权
    'Support Devices': 0.8
}

# 默认CheXpert文件路径
DEFAULT_CHEXPERT_PATH = 'data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv'


# ==================== 核心函数 ====================

def normalize_label_value(value) -> float:
    """
    归一化标签值
    
    CheXpert标签规则:
    - 1.0: Positive (确定有)
    - 0.0: Negative (确定无)
    - -1.0: Uncertain (不确定)
    - NaN: Not mentioned (未提及)
    
    Args:
        value: 原始标签值
        
    Returns:
        normalized_value: 归一化后的值
            - 1.0 → 1.0 (保持)
            - 0.0 → 0.0 (保持)
            - -1.0 → 0.5 (不确定视为弱阳性)
            - NaN → 0.0 (未提及视为阴性)
    """
    if pd.isna(value):
        return 0.0  # 未提及视为阴性
    elif value == -1.0:
        return 0.5  # 不确定视为弱阳性（0.5）
    else:
        return float(value)  # 保持0或1


def load_chexpert_labels(
    csv_path: str = DEFAULT_CHEXPERT_PATH,
    normalize: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    加载CheXpert标签文件
    
    Args:
        csv_path: CheXpert CSV文件路径
        normalize: 是否归一化标签值
        verbose: 是否打印统计信息
        
    Returns:
        labels_dict: {study_id: array[14]} 字典
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CheXpert文件不存在: {csv_path}")
    
    if verbose:
        print(f"[CheXpert] 加载标签文件: {csv_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"[CheXpert] 总样本数: {len(df):,}")
    
    # 转换为字典
    labels_dict = {}
    
    for _, row in df.iterrows():
        # 强制统一study_id格式（去除所有可能的.0后缀）
        raw_id = row['study_id']
        
        # 方法：先转int（去除小数），再转str
        try:
            if pd.isna(raw_id):
                study_id = '0'  # 处理NaN情况
            else:
                study_id = str(int(float(raw_id)))  # 确保去除.0
        except (ValueError, TypeError):
            study_id = str(raw_id).replace('.0', '')  # 备用方案
        
        # 提取14个标签
        labels = []
        for label_name in LABEL_NAMES:
            value = row[label_name]
            if normalize:
                value = normalize_label_value(value)
            else:
                value = float(value) if not pd.isna(value) else 0.0
            labels.append(value)
        
        labels_dict[study_id] = np.array(labels, dtype=np.float32)
    
    if verbose:
        print(f"[CheXpert] 加载完成: {len(labels_dict):,} 个study")
        
        # 统计标签分布
        print(f"\n[CheXpert] 标签分布统计:")
        all_labels = np.stack(list(labels_dict.values()))  # (N, 14)
        
        for i, name in enumerate(LABEL_NAMES):
            positive_count = (all_labels[:, i] > 0.5).sum()
            positive_ratio = positive_count / len(all_labels) * 100
            print(f"  {name:30s}: {positive_count:6,} ({positive_ratio:5.2f}%)")
    
    return labels_dict


def get_abnormality_count(labels: np.ndarray, threshold: float = 0.5) -> int:
    """
    统计阳性标签数量
    
    Args:
        labels: array[14] 标签向量
        threshold: 阳性阈值（默认0.5）
        
    Returns:
        count: 阳性标签数量
    """
    # 排除"No Finding"标签（索引8）
    mask = np.arange(14) != 8  # 不统计"No Finding"
    abnormal_labels = labels[mask]
    
    count = (abnormal_labels > threshold).sum()
    return int(count)


def get_label_mask(labels: np.ndarray) -> np.ndarray:
    """
    生成有效标签mask
    
    Args:
        labels: array[14] 标签向量
        
    Returns:
        mask: array[14] bool mask (True=有效, False=无效/NaN)
    """
    # 标签值为0可能是真实阴性或未标注，这里都视为有效
    # 只有当原始数据是NaN时才是无效
    mask = ~np.isnan(labels)
    return mask


def get_label_weights_tensor(
    labels_dict: Dict[str, np.ndarray],
    device: str = 'cuda'
) -> torch.Tensor:
    """
    计算标签权重tensor（用于处理类别不平衡）
    
    基于数据集中每个标签的频率计算权重:
    - 频率高的标签 → 权重低
    - 频率低的标签 → 权重高
    
    Args:
        labels_dict: {study_id: array[14]} 标签字典
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        weights: (14,) 权重tensor
    
    Example:
        >>> labels = load_chexpert_labels()
        >>> weights = get_label_weights_tensor(labels)
        >>> print(weights.shape)  # torch.Size([14])
    """
    # 收集所有标签
    all_labels = np.stack([labels_dict[sid] for sid in labels_dict])
    
    # 统计每类标签的阳性率
    positive_counts = (all_labels > 0.5).sum(axis=0)
    total = len(all_labels)
    
    # 计算权重 (inverse frequency)
    weights = np.zeros(14, dtype=np.float32)
    for i in range(14):
        if positive_counts[i] > 0:
            # 权重 = 总样本数 / (类别数 * 阳性样本数)
            weights[i] = total / (14 * positive_counts[i])
        else:
            weights[i] = 1.0  # 如果没有阳性样本，使用默认权重
    
    # 归一化权重（使平均值为1）
    weights = weights / weights.mean()
    
    # 限制权重范围（避免极端值）
    weights = np.clip(weights, 0.5, 3.0)
    
    return torch.from_numpy(weights).to(device)


def merge_labels_to_dataset(
    dataset_df: pd.DataFrame,
    chexpert_labels: Dict[str, np.ndarray],
    verbose: bool = True
) -> pd.DataFrame:
    """
    将CheXpert标签合并到数据集DataFrame
    
    Args:
        dataset_df: 数据集DataFrame (必须有study_id列)
        chexpert_labels: CheXpert标签字典
        verbose: 是否打印统计
        
    Returns:
        merged_df: 合并后的DataFrame
    """
    df = dataset_df.copy()
    
    # 添加14列标签
    for i, label_name in enumerate(LABEL_NAMES):
        df[label_name] = df['study_id'].astype(str).map(
            lambda sid: chexpert_labels.get(sid, np.zeros(14))[i]
        )
    
    if verbose:
        matched = df[LABEL_NAMES[0]].notna().sum()
        print(f"[CheXpert] 匹配到标签: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    
    return df


def visualize_label_distribution(
    labels_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
):
    """
    可视化标签分布（调试用）
    
    Args:
        labels_dict: 标签字典
        save_path: 保存图片路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  需要安装matplotlib: pip install matplotlib")
        return
    
    all_labels = np.stack(list(labels_dict.values()))  # (N, 14)
    
    # 统计每类标签的阳性率
    positive_counts = (all_labels > 0.5).sum(axis=0)
    positive_ratios = positive_counts / len(all_labels) * 100
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(LABEL_NAMES))
    bars = ax.bar(x, positive_ratios, color='steelblue', alpha=0.7)
    
    # 标注数值
    for i, (bar, count, ratio) in enumerate(zip(bars, positive_counts, positive_ratios)):
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 1,
            f'{count}\n({ratio:.1f}%)',
            ha='center', va='bottom', fontsize=8
        )
    
    ax.set_xlabel('Label Category', fontsize=12)
    ax.set_ylabel('Positive Rate (%)', fontsize=12)
    ax.set_title('CheXpert Label Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[CheXpert] 图表保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== 测试函数 ====================

def test_chexpert_utils():
    """测试工具函数"""
    print("=" * 80)
    print("Testing CheXpert Utils")
    print("=" * 80)
    
    # 测试加载
    try:
        labels = load_chexpert_labels(verbose=True)
        print(f"\n✅ 成功加载 {len(labels)} 个study的标签")
        
        # 测试单个样本
        example_id = list(labels.keys())[0]
        example_labels = labels[example_id]
        
        print(f"\n示例 study_id: {example_id}")
        print(f"标签向量: {example_labels}")
        print(f"阳性标签数: {get_abnormality_count(example_labels)}")
        
        # 统计异常分布
        abnormality_counts = [get_abnormality_count(labels[sid]) for sid in labels]
        print(f"\n异常数量分布:")
        for i in range(6):
            count = sum(1 for c in abnormality_counts if c == i)
            print(f"  {i}个异常: {count:,} ({count/len(abnormality_counts)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print("✅ 测试通过!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请确保CheXpert CSV文件存在于: data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv")


if __name__ == "__main__":
    test_chexpert_utils()
