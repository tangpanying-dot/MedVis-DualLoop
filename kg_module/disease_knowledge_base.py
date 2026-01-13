# kg_module/disease_knowledge_base.py
"""
疾病知识库管理 (Fix: Stop Graph Explosion Inside Loop)
"""
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class DiseaseKnowledgeBase:
    def __init__(self, knowledge_source: str = 'icd'):
        self.knowledge_source = knowledge_source
        self.disease_graph = {'entities': {}, 'relations': []}
        self.icd_hierarchy = {} 
        self.disease_categories = {}
        logger.info(f"DiseaseKnowledgeBase初始化: source={knowledge_source}")
    
    def load_from_file(self, filepath: str):
        logger.info(f"加载疾病知识库: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.disease_graph = data['disease_graph']
        self.icd_hierarchy = data.get('icd_hierarchy', {})
        self.disease_categories = data.get('disease_categories', {})
        logger.info(f"知识库加载完成: {len(self.disease_graph['entities'])} 疾病")
    
    def get_disease_info(self, icd_code: str) -> Optional[Dict]:
        return self.disease_graph['entities'].get(icd_code)
    
    def get_disease_relations(self, icd_code: str, relation_types: Optional[List[str]] = None) -> List[Tuple]:
        relations = []
        for src, tgt, rel_type, weight in self.disease_graph['relations']:
            if src == icd_code:
                if relation_types is None or rel_type in relation_types:
                    relations.append((tgt, rel_type, weight))
        return relations
    
    def has_relation(self, src_icd: str, tgt_icd: str, relation_types: Optional[List[str]] = None) -> bool:
        for src, tgt, rel_type, weight in self.disease_graph['relations']:
            if src == src_icd and tgt == tgt_icd:
                if relation_types is None or rel_type in relation_types:
                    return True
        return False
    
    def extract_disease_subgraph(self, icd_codes: List[str], max_hop: int = 2, max_entities: int = 50) -> Dict:
        """提取疾病子图 (Inner Loop Break Fix)"""
        core_entities = set(icd_codes)
        all_entities = set(icd_codes)
        all_relations = []
        
        current_layer = core_entities
        
        # 初始截断
        if len(all_entities) > max_entities:
            sorted_init = sorted(list(all_entities))[:max_entities]
            all_entities = set(sorted_init)
            current_layer = all_entities

        for hop in range(max_hop):
            next_layer = set()
            
            # ✨ 检查是否已满 (大循环检查)
            if len(all_entities) >= max_entities:
                break

            for disease in current_layer:
                # ✨ 检查是否已满 (中循环检查)
                if len(all_entities) >= max_entities:
                    break

                relations = self.get_disease_relations(disease)
                
                for target, rel_type, weight in relations:
                    # ✨ 检查是否已满 (内循环检查)
                    if target not in all_entities and len(all_entities) >= max_entities:
                        continue 

                    all_relations.append((disease, target, rel_type, weight))
                    
                    if target not in all_entities:
                        next_layer.add(target)
                        all_entities.add(target)
                
                parent = self._get_parent_category(disease)
                if parent:
                    if len(all_entities) < max_entities or parent in all_entities:
                        if parent not in all_entities:
                            next_layer.add(parent)
                            all_entities.add(parent)
                        all_relations.append((disease, parent, 'is_a', 1.0))

            current_layer = next_layer
            
        subgraph_entities = {}
        for icd in all_entities:
            info = self.get_disease_info(icd)
            if info:
                subgraph_entities[icd] = info
        
        valid_relations = []
        for src, tgt, rel_type, weight in all_relations:
            if src in subgraph_entities and tgt in subgraph_entities:
                valid_relations.append((src, tgt, rel_type, weight))
        
        return {
            'entities': subgraph_entities,
            'relations': valid_relations
        }
    
    def _get_parent_category(self, icd_code: str) -> Optional[str]:
        if len(icd_code) <= 3: return None
        parent_code = icd_code[:3]
        if parent_code in self.disease_graph['entities']: return parent_code
        return None
    
    def get_disease_category(self, icd_code: str) -> str:
        return self.disease_categories.get(icd_code, 'Unknown')
    
    def add_cooccurrence_relations(self, cooccurrence_data: Dict[Tuple[str, str], float]):
        for (icd1, icd2), score in cooccurrence_data.items():
            self.disease_graph['relations'].append((icd1, icd2, 'co-occurs_with', score))
            self.disease_graph['relations'].append((icd2, icd1, 'co-occurs_with', score))
    
    def get_statistics(self) -> Dict:
        return {'num_diseases': len(self.disease_graph['entities']), 'num_relations': len(self.disease_graph['relations'])}
    
    def save(self, filepath: str):
        data = {
            'knowledge_source': self.knowledge_source,
            'disease_graph': self.disease_graph,
            'icd_hierarchy': self.icd_hierarchy,
            'disease_categories': self.disease_categories
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)