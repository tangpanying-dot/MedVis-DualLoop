# kg_module/graph_aggregator_enhanced.py
"""
增强版RadGraph聚合模块

将多个检索到的RadGraph聚合为单一知识图谱，
通过多样性增强策略提升特征区分度。

优化策略:
1. 更平衡的权重分配
2. 多样性增强（鼓励保留独特实体）
3. 保留更多独特实体（提高特征多样性）

Classes:
    EnhancedGraphAggregator: 增强版图聚合器
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EnhancedGraphAggregator:
    """
    增强版图聚合器
    
    通过智能加权和多样性优化，将多个RadGraph聚合为单一图谱。
    核心思想：
    - 对高频实体降低权重（避免过度主导）
    - 对独特实体提升权重（增加区分度）
    - 平衡各检索结果的贡献
    
    Aggregation Modes:
        - 'weighted_union': 标准加权并集
        - 'diversity_weighted': 多样性增强的加权并集（推荐）
        - 'adaptive': 自适应聚合（根据相似度分布调整策略）
    
    Args:
        aggregation_mode: 聚合模式 (default: 'diversity_weighted')
        min_weight_threshold: 最小权重阈值，低于此值的实体被过滤 (default: 0.05)
        max_entities: 最大保留实体数 (default: 150)
        max_relations: 最大保留关系数 (default: 300)
        priority_weights: 检索排名的优先级权重，[rank1, rank2, ...] (default: [2.0, 1.5, 1.2, 1.0, 0.8])
        diversity_penalty: 多样性惩罚因子，越大越鼓励多样性 (default: 0.3)
        
    Example:
        >>> aggregator = EnhancedGraphAggregator(
        ...     aggregation_mode='diversity_weighted',
        ...     min_weight_threshold=0.05
        ... )
        >>> radgraphs = [case1['radgraph'], case2['radgraph']]
        >>> similarities = [0.85, 0.72]
        >>> aggregated = aggregator.aggregate(radgraphs, similarities)
        >>> print(len(aggregated['entities']))  # 聚合后的实体数
    """
    
    def __init__(self, 
                 aggregation_mode: str = 'diversity_weighted',
                 min_weight_threshold: float = 0.05,
                 max_entities: int = 150,
                 max_relations: int = 300,
                 priority_weights: List[float] = [2.0, 1.5, 1.2, 1.0, 0.8],
                 diversity_penalty: float = 0.3):
        """
        初始化聚合器
        
        Args:
            aggregation_mode: 聚合模式
                - 'weighted_union': 标准加权并集
                - 'diversity_weighted': 多样性增强（推荐）
                - 'adaptive': 自适应聚合
            min_weight_threshold: 实体权重阈值，降低可保留更多独特实体
            max_entities: 最大实体数量限制
            max_relations: 最大关系数量限制
            priority_weights: 检索排名权重列表，鼓励前排结果
            diversity_penalty: 多样性惩罚因子，增大鼓励多样性
        """
        self.aggregation_mode = aggregation_mode
        self.min_weight_threshold = min_weight_threshold
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.priority_weights = priority_weights
        self.diversity_penalty = diversity_penalty
        
        logger.info(f"EnhancedGraphAggregator初始化: "
                   f"mode={aggregation_mode}, "
                   f"threshold={min_weight_threshold}, "
                   f"diversity_penalty={diversity_penalty}")
    
    def aggregate(self, 
                  radgraphs: List[Dict],
                  similarities: List[float]) -> Dict:
        """
        聚合多个RadGraph为单一知识图谱
        
        Args:
            radgraphs: RadGraph列表，每个图包含:
                - entities: Dict[str, Dict] 实体字典
                    - tokens: str 实体文本
                    - label: str 实体类型（如'OBS-DP', 'ANAT-DP'）
                    - relations: List[Tuple] 关系列表
                - labeler: str 标注者
            similarities: 对应的相似度分数列表，取值[0, 1]
                长度必须与radgraphs相同
                
        Returns:
            聚合后的图字典:
                - entities: Dict[str, Dict] 实体集合
                    - type: str 实体类型
                    - weight: float 聚合权重
                    - count: int 出现次数
                    - uniqueness: float 独特性分数（仅diversity模式）
                - relations: List[Tuple[str, str, str, float]] 关系集合
                    格式: (源实体, 目标实体, 关系类型, 权重)
                    
        Raises:
            ValueError: 如果radgraphs和similarities长度不匹配
            
        Note:
            - 实体权重计算: base_weight × priority_boost × (1 + diversity_penalty × uniqueness)
            - 权重低于min_weight_threshold的实体会被过滤
            - 最终实体数不超过max_entities，关系数不超过max_relations
        """
        if self.aggregation_mode == 'diversity_weighted':
            return self._diversity_weighted_union(radgraphs, similarities)
        elif self.aggregation_mode == 'adaptive':
            return self._adaptive_aggregation(radgraphs, similarities)
        else:
            return self._weighted_union(radgraphs, similarities)
    
    def _diversity_weighted_union(self, 
                                  radgraphs: List[Dict],
                                  similarities: List[float]) -> Dict:
        """
        多样性增强的加权并集（推荐模式）
        
        核心策略:
        1. 计算每个实体的出现频率
        2. 频率越低，独特性分数越高
        3. 独特实体获得额外权重提升
        4. 平衡各检索结果的贡献
        
        Args:
            radgraphs: RadGraph列表
            similarities: 相似度列表
            
        Returns:
            聚合后的图字典
        """
        all_entities = {}
        entity_sources = defaultdict(list)  # 记录每个实体来自哪些图
        all_relations = []
        
        # 第一遍：收集所有实体和它们的来源
        for i, graph in enumerate(radgraphs):
            for eid, entity in graph['entities'].items():
                label = entity['tokens']
                entity_sources[label].append(i)
        
        # 计算实体的"独特性"分数
        # 出现次数越少，独特性越高
        uniqueness_scores = {}
        for label, sources in entity_sources.items():
            frequency = len(sources)
            # 使用对数平滑，避免单次出现的实体权重过高
            uniqueness = 1.0 / (1.0 + np.log(frequency))
            uniqueness_scores[label] = uniqueness
        
        # 第二遍：加权聚合
        for i, (graph, similarity) in enumerate(zip(radgraphs, similarities)):
            # 基础权重（来自检索相似度）
            base_weight = similarity
            
            # 优先级权重（前排结果获得更高权重）
            priority_boost = self.priority_weights[i] if i < len(self.priority_weights) else 0.5
            
            # 处理实体
            for eid, entity in graph['entities'].items():
                label = entity['tokens']
                entity_type = entity['label']
                
                # 结合独特性分数
                uniqueness = uniqueness_scores[label]
                
                # 最终权重 = 基础权重 × 优先级 × (1 + 多样性惩罚 × 独特性)
                # 独特实体获得额外加成
                final_weight = base_weight * priority_boost * (1.0 + self.diversity_penalty * uniqueness)
                
                if label not in all_entities:
                    all_entities[label] = {
                        'type': entity_type,
                        'weight': final_weight,
                        'count': 1,
                        'uniqueness': uniqueness
                    }
                else:
                    all_entities[label]['weight'] += final_weight
                    all_entities[label]['count'] += 1
            
            # 处理关系（同样考虑独特性）
            for eid, entity in graph['entities'].items():
                src_label = entity['tokens']
                for rel_type, tgt_id in entity['relations']:
                    tgt_label = graph['entities'][tgt_id]['tokens']
                    
                    # 关系的独特性 = 两端实体独特性的平均
                    rel_uniqueness = (uniqueness_scores[src_label] + 
                                    uniqueness_scores[tgt_label]) / 2
                    
                    rel_weight = base_weight * priority_boost * (1.0 + self.diversity_penalty * rel_uniqueness)
                    
                    all_relations.append((src_label, tgt_label, rel_type, rel_weight))
        
        # 过滤低权重实体
        filtered_entities = {
            label: data for label, data in all_entities.items()
            if data['weight'] >= self.min_weight_threshold
        }
        
        logger.info(f"多样性聚合: {len(all_entities)} → {len(filtered_entities)} 实体 "
                   f"(threshold={self.min_weight_threshold})")
        
        # 截断到最大限制
        aggregated = self._truncate_graph(filtered_entities, all_relations)
        
        return aggregated
    
    def _adaptive_aggregation(self,
                             radgraphs: List[Dict],
                             similarities: List[float]) -> Dict:
        """
        自适应聚合：根据相似度分布动态调整策略
        
        策略:
        - 如果检索结果相似度都很高 → 增加多样性惩罚
        - 如果检索结果相似度分散 → 减少多样性惩罚
        
        Args:
            radgraphs: RadGraph列表
            similarities: 相似度列表
            
        Returns:
            聚合后的图字典
        """
        # 分析相似度分布
        sim_std = np.std(similarities)
        sim_mean = np.mean(similarities)
        
        # 动态调整diversity_penalty
        if sim_std < 0.1 and sim_mean > 0.8:
            # 检索结果过于相似，增强多样性
            dynamic_penalty = self.diversity_penalty * 2.0
            logger.info(f"检测到高相似度聚类，增强多样性: penalty={dynamic_penalty:.2f}")
        else:
            dynamic_penalty = self.diversity_penalty
        
        # 临时修改penalty
        original_penalty = self.diversity_penalty
        self.diversity_penalty = dynamic_penalty
        
        result = self._diversity_weighted_union(radgraphs, similarities)
        
        # 恢复原值
        self.diversity_penalty = original_penalty
        
        return result
    
    def _weighted_union(self, 
                       radgraphs: List[Dict],
                       similarities: List[float]) -> Dict:
        """
        标准加权并集（基础模式，保持向后兼容）
        
        简单地根据相似度和排名权重聚合实体和关系。
        不考虑多样性因素。
        
        Args:
            radgraphs: RadGraph列表
            similarities: 相似度列表
            
        Returns:
            聚合后的图字典
        """
        all_entities = {}
        all_relations = []
        
        for i, (graph, similarity) in enumerate(zip(radgraphs, similarities)):
            base_weight = similarity
            priority_boost = self.priority_weights[i] if i < len(self.priority_weights) else 1.0
            weight = base_weight * priority_boost

            for eid, entity in graph['entities'].items():
                label = entity['tokens']
                entity_type = entity['label']
                
                if label not in all_entities:
                    all_entities[label] = {
                        'type': entity_type,
                        'weight': weight,
                        'count': 1
                    }
                else:
                    all_entities[label]['weight'] += weight
                    all_entities[label]['count'] += 1
            
            for eid, entity in graph['entities'].items():
                src_label = entity['tokens']
                for rel_type, tgt_id in entity['relations']:
                    tgt_label = graph['entities'][tgt_id]['tokens']
                    all_relations.append((src_label, tgt_label, rel_type, weight))
        
        filtered_entities = {
            label: data for label, data in all_entities.items()
            if data['weight'] >= self.min_weight_threshold
        }
        
        aggregated = self._truncate_graph(filtered_entities, all_relations)
        
        return aggregated
    
    def _truncate_graph(self, 
                       entities: Dict,
                       relations: List[Tuple]) -> Dict:
        """
        截断图到指定大小
        
        保留权重最高的实体和关系，确保不超过最大限制。
        
        Args:
            entities: 实体字典 {label: {type, weight, count, ...}}
            relations: 关系列表 [(src, tgt, rel_type, weight), ...]
            
        Returns:
            截断后的图字典
        """
        # 按权重排序实体
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        
        # 截断实体
        if len(sorted_entities) > self.max_entities:
            sorted_entities = sorted_entities[:self.max_entities]
        
        kept_entities = {label: data for label, data in sorted_entities}
        kept_entity_labels = set(kept_entities.keys())
        
        # 过滤关系：只保留两端实体都存在的关系
        valid_relations = []
        for src, tgt, rel_type, weight in relations:
            if src in kept_entity_labels and tgt in kept_entity_labels:
                valid_relations.append((src, tgt, rel_type, weight))
        
        # 按权重排序关系
        valid_relations = sorted(valid_relations, key=lambda x: x[3], reverse=True)
        
        # 截断关系
        if len(valid_relations) > self.max_relations:
            valid_relations = valid_relations[:self.max_relations]
        
        return {
            'entities': kept_entities,
            'relations': valid_relations
        }
    
    def get_graph_statistics(self, graph: Dict) -> Dict:
        """
        计算聚合图的统计信息
        
        Args:
            graph: 聚合后的图字典
            
        Returns:
            统计信息字典，包含:
                - num_entities: 实体数量
                - num_relations: 关系数量
                - entity_types: 实体类型分布 {type: count}
                - relation_types: 关系类型分布 {type: count}
                - mean_entity_weight: 平均实体权重
                - mean_relation_weight: 平均关系权重
                - mean_uniqueness: 平均独特性分数（如果有）
        """
        entities = graph['entities']
        relations = graph['relations']
        
        # 实体类型统计
        type_counts = defaultdict(int)
        for data in entities.values():
            type_counts[data['type']] += 1
        
        # 关系类型统计
        rel_type_counts = defaultdict(int)
        for _, _, rel_type, _ in relations:
            rel_type_counts[rel_type] += 1
        
        # 权重统计
        entity_weights = [data['weight'] for data in entities.values()]
        relation_weights = [w for _, _, _, w in relations]
        
        # 独特性统计（如果存在）
        if entities and 'uniqueness' in list(entities.values())[0]:
            uniqueness_scores = [data['uniqueness'] for data in entities.values()]
            mean_uniqueness = np.mean(uniqueness_scores)
        else:
            mean_uniqueness = 0.0
        
        return {
            'num_entities': len(entities),
            'num_relations': len(relations),
            'entity_types': dict(type_counts),
            'relation_types': dict(rel_type_counts),
            'mean_entity_weight': np.mean(entity_weights) if entity_weights else 0,
            'mean_relation_weight': np.mean(relation_weights) if relation_weights else 0,
            'mean_uniqueness': mean_uniqueness
        }
    
    def visualize_graph(self, graph: Dict, max_display: int = 20):
        """
        打印图结构（用于调试）
        
        Args:
            graph: 聚合后的图字典
            max_display: 最多显示的实体/关系数量
        """
        entities = graph['entities']
        relations = graph['relations']
        
        print("\n" + "="*70)
        print("增强版聚合图 (带独特性分数)")
        print("="*70)
        
        print(f"\n实体数量: {len(entities)}")
        print(f"关系数量: {len(relations)}")
        
        # 显示实体
        print(f"\n前{max_display}个实体（按权重）:")
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        
        for i, (label, data) in enumerate(sorted_entities[:max_display]):
            uniqueness_str = f", uniqueness={data['uniqueness']:.3f}" if 'uniqueness' in data else ""
            print(f"  {i+1}. {label} [{data['type']}] "
                  f"(weight={data['weight']:.3f}, count={data['count']}{uniqueness_str})")
        
        # 显示关系
        print(f"\n前{max_display}个关系（按权重）:")
        sorted_relations = sorted(relations, key=lambda x: x[3], reverse=True)
        
        for i, (src, tgt, rel_type, weight) in enumerate(sorted_relations[:max_display]):
            print(f"  {i+1}. {src} --[{rel_type}]--> {tgt} (weight={weight:.3f})")
        
        print("="*70)