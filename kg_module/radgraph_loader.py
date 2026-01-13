# tools/radgraph_loader.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadGraphLoader:
    """RadGraph数据加载器 (保持不变)"""
    def __init__(self, radgraph_json_path: str):
        self.radgraph_json_path = Path(radgraph_json_path)
        self.raw_graphs = {}
        self.study_id_mapping = {}
        self._load_radgraph()
        self._build_study_mapping()
    
    def _load_radgraph(self):
        try:
            with open(self.radgraph_json_path, 'r') as f:
                self.raw_graphs = json.load(f)
        except Exception as e:
            logger.error(f"加载RadGraph失败: {e}")
            raise
    
    def _parse_study_id_from_key(self, key: str) -> Optional[int]:
        try:
            return int(key.split('/')[-1].replace('.txt', '').replace('s', ''))
        except:
            return None
    
    def _build_study_mapping(self):
        for key, graph in self.raw_graphs.items():
            sid = self._parse_study_id_from_key(key)
            if sid: self.study_id_mapping[sid] = graph

    def get_graph_by_study_id(self, study_id: int):
        return self.study_id_mapping.get(study_id)

class CaseDatabase:
    """
    病例数据库 (修改版)
    只加载：Visual Vectors (用于检索) + RadGraph (用于构建知识)
    不再加载：History Vectors (已弃用)
    """
    def __init__(self, 
                 csv_path: str,
                 # 指向您之前的融合向量目录
                 visual_feature_dir: str = 'visual/visual_features/multi_scale', 
                 radgraph_loader: RadGraphLoader = None):
        
        self.csv_path = Path(csv_path)
        self.visual_dir = Path(visual_feature_dir)
        self.radgraph_loader = radgraph_loader
        self.database = {}
        self.build_database()
    
    def build_database(self):
        logger.info("构建病例数据库 (Lite版 - 无历史向量)...")
        if not self.csv_path.exists():
            logger.error(f"CSV文件不存在: {self.csv_path}")
            return

        df = pd.read_csv(self.csv_path)
        success_count = 0
        
        for idx, row in df.iterrows():
            study_id = int(row['study_id'])
            
            # 1. 检查视觉特征 (用于 Layer 1 检索)
            visual_path = self.visual_dir / f"{study_id}.npy"
            
            # 2. 检查 RadGraph
            radgraph = self.radgraph_loader.get_graph_by_study_id(study_id)
            
            if visual_path.exists() and radgraph:
                try:
                    # 加载视觉特征 [1024]
                    visual_feat = np.load(visual_path)
                    if visual_feat.ndim == 2: visual_feat = visual_feat.squeeze(0)
                        
                    self.database[study_id] = {
                        'visual_feat': visual_feat,
                        'radgraph': radgraph,
                        'study_id': study_id
                    }
                    success_count += 1
                except Exception as e:
                    continue
        
        logger.info(f"数据库构建完成: {success_count}/{len(df)} 有效病例")

    def get_case(self, study_id: int):
        return self.database.get(study_id)