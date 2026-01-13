# kg_module/radgraph_textualizer.py
from typing import Dict, List

class RadGraphTextualizer:
    """
    Layer 1 文本化器：将聚合后的 RadGraph 转化为参考提示
    """
    def __init__(self, max_entities: int = 15):
        self.max_entities = max_entities

    def textualize(self, aggregated_graph: Dict) -> str:
        entities = aggregated_graph.get('entities', {})
        relations = aggregated_graph.get('relations', [])
        
        if not entities:
            return ""

        # 1. 按权重排序实体
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1].get('weight', 0),
            reverse=True
        )
        
        observations = []
        anatomies = []
        
        for label, data in sorted_entities:
            # 过滤低权重噪声
            if data.get('weight', 0) < 0.15: continue
            
            etype = data.get('type', '')
            if 'OBS' in etype:
                observations.append(label)
            elif 'ANAT' in etype:
                anatomies.append(label)
        
        # 2. 提取关键关系 (Top 6)
        sorted_relations = sorted(relations, key=lambda x: x[3], reverse=True)
        key_relations = []
        
        for src, tgt, rel_type, weight in sorted_relations[:6]:
            if weight < 0.15: continue
            readable_rel = rel_type.replace('_', ' ')
            key_relations.append(f"{src} {readable_rel} {tgt}")

        # 3. 生成 Prompt
        parts = []
        if key_relations:
            # 话术：强调这是"类似病例"的参考
            parts.append(f"Reference patterns from similar cases: {'; '.join(key_relations)}.")
        
        if observations:
            obs_str = ', '.join(observations[:self.max_entities])
            parts.append(f"Common findings in similar images: {obs_str}.")
            
        return " ".join(parts)