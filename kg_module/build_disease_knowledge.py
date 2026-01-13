# tools/build_disease_knowledge.py
"""
构建疾病知识库
从ICD层级结构和数据挖掘构建疾病关系图谱
"""
import sys
sys.path.append('.')

import json
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

from kg_module.disease_knowledge_base import (
    DiseaseKnowledgeBase,
    DiseaseCooccurrenceMiner,
    parse_history_field  # 使用统一的工具函数
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_icd9_hierarchy():
    """
    加载ICD-9层级结构
    
    这里提供一个简化版本，实际应用中应该使用完整的ICD-9数据
    """
    # ICD-9主要类别
    icd_categories = {
        # 肝脏疾病
        '570-579': '消化系统疾病',
        '571': '慢性肝病和肝硬化',
        '5715': '肝硬化',
        '5716': '胆汁性肝硬化',
        '5723': '门静脉高压',
        
        # 病毒性肝炎
        '070': '病毒性肝炎',
        '07070': '丙型肝炎无肝昏迷',
        '07071': '丙型肝炎伴肝昏迷',
        '07044': '慢性丙型肝炎伴肝昏迷',
        
        # 呼吸系统
        '490-496': '慢性阻塞性肺疾病',
        '496': 'COPD',
        
        # 症状
        '789': '腹部症状',
        '78959': '腹水',
        
        # 血液系统
        '287': '紫癜和其他出血情况',
        '2875': '血小板减少',
        
        # 电解质紊乱
        '276': '电解质和体液失衡',
        '2761': '低钠血症',
        '2767': '高钾血症',
        
        # 精神疾病
        '296': '情感性精神病',
        '29680': '双相情感障碍',
        
        # 应激障碍
        '309': '适应障碍',
        '30981': 'PTSD',
        
        # 恶病质
        '799': '其他不明原因的疾病',
        '7994': '恶病质',
    }
    
    return icd_categories


def build_icd_relations():
    """
    构建基于医学知识的疾病关系
    """
    relations = [
        # 丙肝 → 肝硬化 → 并发症
        ('07070', '5715', 'causes', 0.9),  # 丙肝导致肝硬化
        ('07071', '5715', 'causes', 0.95),
        ('07044', '5715', 'causes', 0.95),
        ('5715', '5723', 'causes', 0.85),  # 肝硬化导致门静脉高压
        ('5723', '78959', 'causes', 0.8),  # 门静脉高压导致腹水
        ('5715', '78959', 'complication_of', 0.7),  # 腹水是肝硬化的并发症
        ('5715', '2875', 'complication_of', 0.6),  # 血小板减少
        ('5715', '2761', 'associated_with', 0.5),  # 低钠血症
        
        # 疾病恶化
        ('07070', '07071', 'progresses_to', 0.7),  # 无肝昏迷 → 伴肝昏迷
        ('07071', '07044', 'progresses_to', 0.8),  # 急性 → 慢性
        
        # 层级关系
        ('5715', '571', 'is_a', 1.0),
        ('571', '570-579', 'is_a', 1.0),
        ('07070', '070', 'is_a', 1.0),
        ('07071', '070', 'is_a', 1.0),
        ('496', '490-496', 'is_a', 1.0),
        ('78959', '789', 'is_a', 1.0),
        ('2875', '287', 'is_a', 1.0),
        ('2761', '276', 'is_a', 1.0),
        ('29680', '296', 'is_a', 1.0),
    ]
    
    return relations


def build_knowledge_base(dataset_path: str, 
                        output_path: str,
                        mine_cooccurrence: bool = True):
    """
    构建完整的疾病知识库
    
    Args:
        dataset_path: 数据集CSV路径
        output_path: 输出知识库路径
        mine_cooccurrence: 是否挖掘共现关系
    """
    logger.info("="*70)
    logger.info("开始构建疾病知识库")
    logger.info("="*70)
    
    # 1. 初始化知识库
    kb = DiseaseKnowledgeBase(knowledge_source='icd')
    
    # 2. 加载ICD层级
    logger.info("\n[步骤1] 加载ICD层级结构...")
    icd_categories = load_icd9_hierarchy()
    kb.disease_categories = icd_categories
    
    # 3. 从数据集收集所有疾病
    logger.info("\n[步骤2] 从数据集收集疾病实体...")
    df = pd.read_csv(dataset_path)
    
    disease_info = {}
    disease_counts = defaultdict(int)
    
    valid_count = 0
    empty_count = 0
    
    for idx, row in df.iterrows():
        # 使用统一的工具函数
        history = parse_history_field(row['history'])
        
        if not history:
            empty_count += 1
            continue
        
        valid_count += 1
        
        for admission in history:
            for diag in admission.get('diagnoses', []):
                icd = diag['icd_code']
                desc = diag['description']
                
                disease_counts[icd] += 1
                
                if icd not in disease_info:
                    # 确定疾病类别
                    category = 'Unknown'
                    for cat_code, cat_name in icd_categories.items():
                        if icd.startswith(cat_code.replace('-', '')[:3]):
                            category = cat_name
                            break
                    
                    disease_info[icd] = {
                        'name': desc,
                        'type': 'disease',
                        'category': category,
                        'frequency': 0
                    }
                
                disease_info[icd]['frequency'] = disease_counts[icd]
    
    kb.disease_graph['entities'] = disease_info
    
    logger.info(f"收集到 {len(disease_info)} 个疾病实体")
    logger.info(f"有效样本: {valid_count}, 空病史样本: {empty_count}")
    logger.info(f"前10个高频疾病:")
    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for icd, count in top_diseases:
        logger.info(f"  {icd}: {disease_info[icd]['name']} (出现{count}次)")
    
    # 4. 添加基于医学知识的关系
    logger.info("\n[步骤3] 添加医学知识关系...")
    base_relations = build_icd_relations()
    kb.disease_graph['relations'] = base_relations
    
    logger.info(f"添加了 {len(base_relations)} 个基础关系")
    
    # 5. 挖掘共现关系（可选）
    if mine_cooccurrence:
        logger.info("\n[步骤4] 挖掘疾病共现关系...")
        miner = DiseaseCooccurrenceMiner(dataset_path)
        cooccurrence_data = miner.mine_cooccurrence(
            min_support=10,
            min_confidence=0.3
        )
        
        kb.add_cooccurrence_relations(cooccurrence_data)
        
        logger.info(f"共现关系挖掘完成，新增 {len(cooccurrence_data)} 个关系")
    
    # 6. 统计信息
    logger.info("\n[步骤5] 知识库统计:")
    stats = kb.get_statistics()
    logger.info(f"  疾病实体: {stats['num_diseases']}")
    logger.info(f"  关系数量: {stats['num_relations']}")
    logger.info(f"  关系类型分布:")
    for rel_type, count in stats['relation_types'].items():
        logger.info(f"    {rel_type}: {count}")
    
    # 7. 保存知识库
    logger.info(f"\n[步骤6] 保存知识库到 {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    kb.save(output_path)
    
    logger.info("\n" + "="*70)
    logger.info("疾病知识库构建完成！")
    logger.info("="*70)
    
    return kb


if __name__ == "__main__":
    # 配置路径
    DATASET_PATH = "data/processed_dataset.csv"
    OUTPUT_PATH = "data/disease_knowledge_base.json"
    
    # 构建知识库
    kb = build_knowledge_base(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        mine_cooccurrence=True
    )
    
    print("\n下一步:")
    print("1. 查看生成的知识库文件:")
    print(f"   cat {OUTPUT_PATH}")
    print("2. 构建疾病词汇表:")
    print("   python tools/build_disease_vocabulary.py")