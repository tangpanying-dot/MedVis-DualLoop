# kg_module/disease_kg_module.py
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .disease_knowledge_base import DiseaseKnowledgeBase
from .disease_graph_builder import PatientHistoryParser, DiseaseGraphBuilder
from .disease_graph_textualizer import DiseaseGraphTextualizer

logger = logging.getLogger(__name__)

class DiseaseKGModule(nn.Module):
    """
    Layer 2: 疾病知识图谱模块 (Textualization Mode)
    
    功能:
    1. 解析患者病史 (History Parser)
    2. 构建个性化疾病图谱 (Graph Builder)
    3. 将图谱转化为自然语言提示 (Textualizer)
    
    输入: 患者病史列表
    输出: List[str] (自然语言文本列表，直接用于Prompt)
    """
    
    def __init__(self,
                 knowledge_base: DiseaseKnowledgeBase,
                 builder_config: Optional[Dict] = None):
        super().__init__()
        
        builder_config = builder_config or {}
        self.knowledge_base = knowledge_base
        
        # 1. 图谱构建器 (复用原有逻辑)
        self.graph_builder = DiseaseGraphBuilder(
            knowledge_base=knowledge_base,
            time_decay_lambda=builder_config.get('time_decay_lambda', 0.1),
            severity_weight=builder_config.get('severity_weight', 1.0)
        )
        
        # 2. 文本化器 (新模块，替代 Encoder)
        self.textualizer = DiseaseGraphTextualizer()

    def forward(self,
                patient_histories: List[List[Dict]],
                historical_reports: Optional[List[List[str]]] = None,
                study_datetimes: Optional[List[str]] = None) -> List[str]:
        """
        前向传播
        
        Args:
            patient_histories: 病史列表
            historical_reports: 历史报告列表 (可选)
            study_datetimes: 当前检查时间 (可选)
            
        Returns:
            prompts: List[str] 长度为B的字符串列表
        """
        prompts = []
        # 遍历批次中的每个样本
        for i in range(len(patient_histories)):
            history = patient_histories[i]
            study_dt = study_datetimes[i] if study_datetimes else None
            reports = historical_reports[i] if historical_reports else None
            
            try:
                # 1. 解析病史
                parser = PatientHistoryParser(current_study_datetime=study_dt)
                parsed = parser.parse(history, reports)
                
                # 2. 构建图谱
                # max_entities 控制图大小，防止 Prompt 过长
                graph = self.graph_builder.build_patient_graph(parsed, max_entities=25)
                
                # 3. 文本化 (生成 Prompt)
                text = self.textualizer.textualize(graph, parsed)
                
            except Exception as e:
                logger.warning(f"KG2 Error for sample {i}: {e}")
                text = ""
            
            prompts.append(text)
            
        return prompts

def build_disease_kg_module(knowledge_base_path: str, config: Optional[Dict] = None):
    """
    构建 DiseaseKGModule 的工厂函数
    """
    # 修复之前的 NoneType 报错
    config = config or {}
    
    kb = DiseaseKnowledgeBase()
    kb.load_from_file(knowledge_base_path)
    
    return DiseaseKGModule(kb, config.get('builder', {}))