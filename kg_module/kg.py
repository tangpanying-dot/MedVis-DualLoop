# generate_kg_prompts.py
import sys
import os
import pandas as pd
import torch
import json
import logging

# 路径配置
DATA_DIR = 'data'
VISUAL_DIR = 'visual/visual_features/multi_scale' # 旧向量
RADGRAPH_JSON = os.path.join(DATA_DIR, 'graph/radgraph/MIMIC-CXR_graphs.json')
CSV_PATH = os.path.join(DATA_DIR, 'processed_dataset.csv')
KB_PATH = os.path.join(DATA_DIR, 'disease_knowledge_base.json')

# 导入模块
from kg_module.radgraph_loader import RadGraphLoader, CaseDatabase
from kg_module.dynamic_kg_module import DynamicKGModule
from kg_module.disease_kg_module import build_disease_kg_module

logging.basicConfig(level=logging.INFO)

def safe_parse_history(h):
    try: return json.loads(h) if isinstance(h, str) else []
    except: return eval(h.replace('null', 'None')) if isinstance(h, str) else []

def main():
    print("=== 初始化 KG 模块 (LLM-Centric) ===")
    
    # 1. 初始化 Layer 1
    print("Loading Layer 1 (RadGraph)...")
    rad_loader = RadGraphLoader(RADGRAPH_JSON)
    case_db = CaseDatabase(CSV_PATH, visual_feature_dir=VISUAL_DIR, radgraph_loader=rad_loader)
    
    kg1_module = DynamicKGModule(
        case_database=case_db.database,
        retriever_config={'top_k': 3}
    )
    
    # 2. 初始化 Layer 2
    print("Loading Layer 2 (DiseaseKG)...")
    # 确保知识库存在
    if not os.path.exists(KB_PATH):
        print(f"Error: {KB_PATH} not found. Run tools/build_disease_knowledge.py first.")
        return

    kg2_module = build_disease_kg_module(KB_PATH)
    
    # 3. 加载测试数据
    df = pd.read_csv(CSV_PATH).head(3) # 测试前3个
    
    print("\n=== 开始生成 Prompt ===")
    
    for idx, row in df.iterrows():
        study_id = row['study_id']
        print(f"\n--- Case: {study_id} ---")
        
        # --- Layer 1 生成 ---
        # 获取旧视觉向量 [1024]
        if study_id in case_db.database:
            visual_feat = case_db.database[study_id]['visual_feat'] # numpy array
            # 扩展为 batch [1, 1024]
            visual_tensor = torch.tensor(visual_feat).unsqueeze(0).float()
            
            prompts1 = kg1_module(visual_tensor, exclude_study_ids=[study_id])
            print(f"[Layer 1 Prompt]:\n{prompts1[0]}")
        else:
            print("[Layer 1]: Visual features not found.")
            
        # --- Layer 2 生成 ---
        history = safe_parse_history(row['history'])
        prompts2 = kg2_module(
            patient_histories=[history],
            study_datetimes=[row['study_datetime']]
        )
        print(f"[Layer 2 Prompt]:\n{prompts2[0]}")
        
        # --- 最终 Prompt 预览 --- 
        final_prompt = (
            f"System: ...\n"
            f"{prompts1[0] if study_id in case_db.database else ''}\n"
            f"{prompts2[0]}\n"
            f"Image: <image>\nReport:"
        )
        print(f"[Final LLM Input Logic]:\n{final_prompt[:200]}...")

if __name__ == "__main__":
    main()