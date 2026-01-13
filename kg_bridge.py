# kg_bridge.py (修复版 - 添加Layer开关)
import os
import torch
import numpy as np
import json
import pandas as pd
import logging

try:
    from kg_module.radgraph_loader import RadGraphLoader, CaseDatabase
    from kg_module.dynamic_kg_module import DynamicKGModule
    from kg_module.disease_kg_module import build_disease_kg_module
    from kg_module.disease_graph_builder import PatientHistoryParser
except ImportError as e:
    print(f"[KG Bridge] Warning: Failed to import KG modules: {e}")

logger = logging.getLogger(__name__)

class KGEngine:
    """
    单例模式的KG引擎
    支持动态调用Layer 1 (检索) 和 Layer 2 (推理)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KGEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, 
                   radgraph_path='data/graph/radgraph/MIMIC-CXR_graphs.json',
                   csv_path='data/processed_dataset_train.csv',  # 修改：默认用训练集
                   retrieval_visual_dir='visual/visual_features/multi_scale',
                   kb_path='data/disease_knowledge_base.json',
                   enable_layer1=True,  # ✅ 新增：Layer 1 开关
                   enable_layer2=True): # ✅ 新增：Layer 2 开关
        
        if self._initialized: 
            return
        
        print("=" * 80)
        print("🚀 [KG Engine] Initializing...")
        print(f"   Retrieval Database: {csv_path}")
        print(f"   Layer 1 (Retrieval): {'ON ✅' if enable_layer1 else 'OFF ❌'}")
        print(f"   Layer 2 (Inference): {'ON ✅' if enable_layer2 else 'OFF ❌'}")
        print("=" * 80)

        self.kg1_module = None
        self.kg2_module = None
        
        # --- Layer 1 (Visual Retrieval) ---
        if enable_layer1:
            if os.path.exists(radgraph_path) and os.path.exists(retrieval_visual_dir):
                try:
                    print("\n[Layer 1] Loading RadGraph & Retrieval Index...")
                    rad_loader = RadGraphLoader(radgraph_path)
                    
                    # 使用训练集构建检索库（避免数据泄露）
                    case_db = CaseDatabase(
                        csv_path, 
                        visual_feature_dir=retrieval_visual_dir, 
                        radgraph_loader=rad_loader
                    )
                    
                    self.kg1_module = DynamicKGModule(
                        case_database=case_db.database,
                        retriever_config={'top_k': 3}, 
                        aggregator_config={'aggregation_mode': 'diversity_weighted'}
                    )
                    print("   ✅ Layer 1 Ready.")
                except Exception as e:
                    print(f"   ⚠️ Layer 1 Init Failed: {e}")
            else:
                print("   ⚠️ Layer 1 Skipped: Paths not found")
        else:
            print("\n[Layer 1] Disabled by user.")

        # --- Layer 2 (Disease Reasoning) ---
        if enable_layer2:
            if os.path.exists(kb_path):
                try:
                    print("\n[Layer 2] Loading Disease Knowledge Base...")
                    self.kg2_module = build_disease_kg_module(kb_path)
                    print("   ✅ Layer 2 Ready.")
                except Exception as e:
                    print(f"   ⚠️ Layer 2 Init Failed: {e}")
            else:
                print("   ⚠️ Layer 2 Skipped: KB not found")
        else:
            print("\n[Layer 2] Disabled by user.")
        
        print("=" * 80)        
        self._initialized = True

    def get_prompts(self, 
                    study_id: int, 
                    history_json_str: str,  # CSV中的history字段（JSON字符串）
                    study_datetime: str,
                    old_feature_path: str) -> tuple:
        """
        动态生成KG Prompts
        
        Args:
            study_id: 研究ID
            history_json_str: CSV中的history字段（JSON字符串）
            study_datetime: 检查时间
            old_feature_path: 旧特征路径（用于Layer 1检索）
            
        Returns:
            (kg1_prompt, kg2_prompt): 两层KG的文本提示
        """
        p1, p2 = "", ""
        
        # === Layer 1: Visual Retrieval ===
        if self.kg1_module and old_feature_path and os.path.exists(old_feature_path):
            try:
                # 加载旧特征 [1024]
                feat_1024 = np.load(old_feature_path)
                if feat_1024.ndim == 2: 
                    feat_1024 = feat_1024.squeeze(0)
                
                v_tensor = torch.tensor(feat_1024).float().unsqueeze(0)
                
                # 检索（排除自己）
                res = self.kg1_module(v_tensor, exclude_study_ids=[int(study_id)])
                if res and len(res) > 0: 
                    p1 = res[0]
            except Exception as e: 
                logger.warning(f"Layer 1 error for study {study_id}: {e}")

        # === Layer 2: Disease Reasoning ===
        if self.kg2_module and history_json_str:
            try:
                # 解析JSON病史
                history = self._parse_history(history_json_str)
                
                if history:
                    # 调用Layer 2
                    res = self.kg2_module(
                        patient_histories=[history],
                        historical_reports=None,  # 可选
                        study_datetimes=[study_datetime]
                    )
                    if res and len(res) > 0:
                        p2 = res[0]
            except Exception as e:
                logger.warning(f"Layer 2 error for study {study_id}: {e}")

        return p1, p2

    def _parse_history(self, history_str: str) -> list:
        """安全解析history JSON字符串"""
        if pd.isna(history_str) or not history_str or history_str == '[]':
            return []
        
        try:
            return json.loads(history_str) if isinstance(history_str, str) else []
        except:
            try:
                return eval(history_str.replace('null', 'None'))
            except:
                return []

# 全局单例
kg_engine = KGEngine()