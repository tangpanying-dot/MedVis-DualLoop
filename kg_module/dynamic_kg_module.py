# kg_module/dynamic_kg_module.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import logging

# 确保这三个文件都在 kg_module 目录下
from .KnowledgeGraphRetriever import CaseRetriever
from .graph_aggregator_enhanced import EnhancedGraphAggregator
from .radgraph_textualizer import RadGraphTextualizer

logger = logging.getLogger(__name__)

class DynamicKGModule(nn.Module):
    """
    Layer 1: 动态知识图谱 (Layer 1 - RadGraph)
    输入: [B, 1024] 视觉向量
    输出: List[str] 提示文本
    """
    
    def __init__(self,
                 case_database: Dict,
                 retriever_config: Optional[Dict] = None,
                 aggregator_config: Optional[Dict] = None):
        super().__init__()
        
        retriever_config = retriever_config or {}
        aggregator_config = aggregator_config or {}
        
        # 1. 检索器
        self.retriever = CaseRetriever(
            case_database=case_database,
            **retriever_config
        )
        
        # 2. 聚合器
        self.aggregator = EnhancedGraphAggregator(**aggregator_config)
        
        # 3. 文本化器
        self.textualizer = RadGraphTextualizer(max_entities=15)
        
        self.device = torch.device('cpu')
        logger.info("DynamicKGModule (Layer 1) initialized.")

    def forward(self, 
                visual_features: torch.Tensor,
                exclude_study_ids: Optional[List[int]] = None) -> List[str]:
        
        if isinstance(visual_features, torch.Tensor):
            visual_np = visual_features.detach().cpu().numpy()
        else:
            visual_np = visual_features
            
        batch_size = visual_np.shape[0]
        kg_prompts = []
        
        for i in range(batch_size):
            query_vec = visual_np[i]
            exclude_id = exclude_study_ids[i] if exclude_study_ids else None
            
            try:
                retrieval_results = self.retriever.retrieve(
                    query_vec, exclude_study_id=exclude_id
                )
                
                if retrieval_results:
                    cases = self.retriever.get_retrieved_cases(retrieval_results)
                    radgraphs = [c['radgraph'] for c in cases]
                    sims = [c['similarity'] for c in cases]
                    agg_graph = self.aggregator.aggregate(radgraphs, sims)
                    text = self.textualizer.textualize(agg_graph)
                else:
                    text = ""
            except Exception as e:
                logger.warning(f"KG1 Error: {e}")
                text = ""
            
            kg_prompts.append(text)
            
        return kg_prompts

# =====================================================
# ⚠️ 关键点：这个函数必须顶格写，不能有缩进！
# 也就是说它不能在 class DynamicKGModule 里面
# =====================================================
def build_kg_module(case_database: Dict, config: Optional[Dict] = None):
    config = config or {}
    return DynamicKGModule(
        case_database, 
        config.get('retriever', {}), 
        config.get('aggregator', {})
    )