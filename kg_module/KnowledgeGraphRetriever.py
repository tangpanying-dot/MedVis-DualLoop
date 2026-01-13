# kg_module/KnowledgeGraphRetriever.py (已修复性能瓶颈)
"""
疾病感知的病例检索模块
核心改进：
1. 疾病分类引导的检索
2. 多阶段检索策略
3. 质量过滤机制
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DiseaseClassifier(nn.Module):
    """
    疾病分类器：用于引导检索
    从视觉特征预测疾病类别
    """
    def __init__(self, visual_dim=1024, num_diseases=15, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_diseases)
        )
        
        # 疾病标签映射
        self.disease_labels = [
            'normal', 'pneumonia', 'effusion', 'atelectasis', 'cardiomegaly',
            'consolidation', 'edema', 'pneumothorax', 'mass', 'nodule',
            'fracture', 'metastases', 'emphysema', 'fibrosis', 'other'
        ]
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, 1024) or (1024,)
        Returns:
            disease_probs: (B, num_diseases) or (num_diseases,)
        """
        if visual_features.ndim == 1:
            visual_features = visual_features.unsqueeze(0)
        
        logits = self.classifier(visual_features)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def get_top_diseases(self, visual_features, top_k=3):
        """获取最可能的top-k疾病"""
        probs = self.forward(visual_features)
        if probs.ndim == 2:
            probs = probs.squeeze(0)
        
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            disease = self.disease_labels[idx.item()]
            results.append((disease, prob.item()))
        
        return results


class DiseaseAwareRetriever:
    """
    疾病感知的检索器 (CPU密集型, 适用于单个查询，不适用于批量)
    """
    
    def __init__(self, 
                 case_database: Dict,
                 disease_classifier: Optional[DiseaseClassifier] = None,
                 top_k: int = 5,
                 disease_weight: float = 0.4,
                 visual_weight: float = 0.6,
                 min_similarity_threshold: float = 0.3,
                 max_similarity_threshold: float = 0.92, 
                 diversity_factor: float = 0.2,
                 **kwargs): 
        """
        Args:
            case_database: 病例数据库
            disease_classifier: 疾病分类器（如果为None则只用视觉检索）
            top_k: 返回数量
            disease_weight: 疾病匹配的权重
            visual_weight: 视觉相似度的权重
            min_similarity_threshold: 最小相似度阈值
            diversity_factor: 多样性因子（0=无多样性，1=最大多样性）
        """
        self.case_database = case_database
        self.disease_classifier = disease_classifier
        self.top_k = top_k
        self.disease_weight = disease_weight
        self.visual_weight = visual_weight
        self.min_similarity_threshold = min_similarity_threshold
        self.max_similarity_threshold = max_similarity_threshold 
        self.diversity_factor = diversity_factor
        
        # 预处理数据库
        self._preprocess_database()
        
        # 构建疾病索引
        if disease_classifier:
            self._build_disease_index()
        
        logger.info(f"DiseaseAwareRetriever初始化: {len(self.case_database)} 病例, "
                    f"disease_weight={disease_weight}, diversity={diversity_factor}, "
                    f"similarity_range=[{min_similarity_threshold}, {max_similarity_threshold}]")
    
    def _preprocess_database(self):
        """预处理数据库"""
        self.study_ids = []
        self.visual_features = []
        
        for study_id, case_data in self.case_database.items():
            self.study_ids.append(study_id)
            self.visual_features.append(case_data['visual_feat'])
        
        self.visual_features = np.vstack(self.visual_features)
        logger.info(f"数据库索引: {self.visual_features.shape}")
    
    def _build_disease_index(self):
        """构建疾病索引：为每个病例预测疾病类别"""
        logger.info("构建疾病索引...")
        
        self.disease_predictions = {}
        self.disease_to_cases = defaultdict(list)
        
        device = next(self.disease_classifier.parameters()).device
        
        with torch.no_grad():
            for i, study_id in enumerate(self.study_ids):
                visual_feat = torch.from_numpy(self.visual_features[i]).float().to(device)
                top_diseases = self.disease_classifier.get_top_diseases(visual_feat, top_k=3)
                
                self.disease_predictions[study_id] = top_diseases
                
                # 构建反向索引
                for disease, prob in top_diseases:
                    self.disease_to_cases[disease].append((study_id, prob))
        
        logger.info(f"疾病索引构建完成: {len(self.disease_to_cases)} 个疾病类别")
    
    def retrieve(self,
                 query_visual: np.ndarray,
                 exclude_study_id: Optional[int] = None,
                 use_disease_guidance: bool = True) -> List[Tuple[int, float, Dict]]:
        """
        智能检索 (CPU密集型)
        """
        if query_visual.ndim == 1:
            query_visual = query_visual.reshape(1, -1)
        
        # 1. 【【【---- 瓶颈 1 ----】】】
        # Sklearn 在CPU上计算 1 vs 9374
        visual_similarities = cosine_similarity(query_visual, self.visual_features)[0]
        
        # 2. 如果使用疾病引导
        disease_scores = np.zeros_like(visual_similarities)
        query_diseases = None
        
        if use_disease_guidance and self.disease_classifier:
            device = next(self.disease_classifier.parameters()).device
            query_tensor = torch.from_numpy(query_visual.squeeze()).float().to(device)
            
            query_diseases = self.disease_classifier.get_top_diseases(query_tensor, top_k=3)
            
            for i, study_id in enumerate(self.study_ids):
                if study_id in self.disease_predictions:
                    db_diseases = self.disease_predictions[study_id]
                    disease_scores[i] = self._compute_disease_match_score(
                        query_diseases, db_diseases
                    )
        
        # 3. 融合分数
        if use_disease_guidance and self.disease_classifier:
            final_scores = (self.visual_weight * visual_similarities + 
                            self.disease_weight * disease_scores)
        else:
            final_scores = visual_similarities
        
        # 4. 排序并过滤
        candidates = []
        for i, study_id in enumerate(self.study_ids):
            if exclude_study_id and study_id == exclude_study_id:
                continue
            
            score = final_scores[i]
            visual_sim = visual_similarities[i]
            
            if score < self.min_similarity_threshold:
                continue
            if visual_sim > self.max_similarity_threshold: 
                continue
            
            metadata = {
                'visual_sim': float(visual_similarities[i]),
                'disease_score': float(disease_scores[i]),
                'predicted_diseases': query_diseases
            }
            
            candidates.append((study_id, float(score), metadata))
        
        # 5. 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 6. 应用多样性过滤（可选）
        if self.diversity_factor > 0:
            candidates = self._apply_diversity_filtering(candidates)
        
        # 7. 返回top-k
        results = candidates[:self.top_k]
        
        return results
    
    def _compute_disease_match_score(self, 
                                     query_diseases: List[Tuple],
                                     db_diseases: List[Tuple]) -> float:
        """
        计算两个疾病列表的匹配分数
        """
        query_dict = {disease: prob for disease, prob in query_diseases}
        db_dict = {disease: prob for disease, prob in db_diseases}
        
        common_diseases = set(query_dict.keys()) & set(db_dict.keys())
        
        if not common_diseases:
            return 0.0
        
        match_score = sum(
            query_dict[disease] * db_dict[disease] 
            for disease in common_diseases
        )
        
        max_possible = sum(prob for _, prob in query_diseases[:len(common_diseases)])
        if max_possible > 0:
            match_score = match_score / max_possible
        
        return match_score
    
    def _apply_diversity_filtering(self, 
                                   candidates: List[Tuple]) -> List[Tuple]:
        """
        应用多样性过滤：使用MMR (Maximal Marginal Relevance) 算法
        """
        if len(candidates) <= self.top_k:
            return candidates
        
        selected = [candidates[0]]  
        remaining = candidates[1:]
        
        while len(selected) < self.top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, (study_id, score, metadata) in enumerate(remaining):
                study_idx = self.study_ids.index(study_id)
                visual_feat = self.visual_features[study_idx:study_idx+1]
                
                max_sim_to_selected = 0
                for selected_id, _, _ in selected:
                    selected_idx = self.study_ids.index(selected_id)
                    selected_feat = self.visual_features[selected_idx:selected_idx+1]
                    
                    # 【【【---- 瓶颈 2 ----】】】
                    # Sklearn 在CPU上计算 1 vs 1
                    sim = cosine_similarity(visual_feat, selected_feat)[0, 0]
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                mmr_score = score - self.diversity_factor * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def get_retrieved_cases(self, 
                            retrieval_results: List[Tuple]) -> List[Dict]:
        """获取完整病例数据"""
        cases = []
        for study_id, score, metadata in retrieval_results: 
            case = self.case_database[study_id].copy()
            case['similarity'] = score
            case['study_id'] = study_id
            case['retrieval_metadata'] = metadata 
            cases.append(case)
        
        return cases
    
    def get_statistics(self, retrieval_results: List[Tuple]) -> Dict:
        """计算统计信息"""
        if not retrieval_results:
            return {}
        
        scores = [score for _, score, _ in retrieval_results]
        visual_sims = [meta['visual_sim'] for _, _, meta in retrieval_results]
        disease_scores = [meta['disease_score'] for _, _, meta in retrieval_results]
        
        return {
            'num_retrieved': len(retrieval_results),
            'mean_score': np.mean(scores),
            'mean_visual_sim': np.mean(visual_sims),
            'mean_disease_score': np.mean(disease_scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }


# ============================================================================
# 【【【---- 关键性能修复区 ----】】】
# ============================================================================

class CaseRetriever(DiseaseAwareRetriever):
    """
    向后兼容的简单检索器（不使用疾病引导）
    
    【【【---- 修复 ----】】】
    此类已重写，以使用高速NumPy进行批量检索，
    而不是继承父类DiseaseAwareRetriever的慢速sklearn实现。
    """
    
    def __init__(self, case_database: Dict, top_k: int = 5, 
                 similarity_metric: str = 'cosine', **kwargs): 
        
        # 正常调用父类__init__来设置self.case_database, self.visual_features, 
        # self.study_ids, self.min_similarity_threshold等
        super().__init__(
            case_database=case_database,
            disease_classifier=None,  # 不使用疾病分类器
            top_k=top_k,
            disease_weight=0.0,
            visual_weight=1.0,
            **kwargs  # 传递 diversity_factor, max_similarity_threshold 等
        )
        
        # 【【【---- 关键性能修复 1/2 ----】】】
        # 预先归一化所有数据库特征，以便使用np.dot进行快速cosine similarity
        logger.info(f"CaseRetriever (Fast Mode): 预归一化 {self.visual_features.shape[0]} 个视觉特征...")
        
        # 计算L2范数
        norms = np.linalg.norm(self.visual_features, axis=1, keepdims=True)
        # 替换0范数以避免除以0
        norms[norms == 0] = 1e-6 
        
        self.visual_features_normalized = self.visual_features / norms
        
        # 释放未归一化的副本以节省内存
        del self.visual_features
        logger.info("CaseRetriever (Fast Mode): 预归一化完成。")
        

    def retrieve(self, query_visual: np.ndarray, 
                    exclude_study_id: Optional[int] = None):
            """
            使用NumPy (np.dot) 进行高速检索
            【【【---- 新增: 实现MMR多样性过滤 ----】】】
            """
            
            if query_visual.ndim == 1:
                query_visual = query_visual.reshape(1, -1)

            # 1. 归一化查询向量
            query_norm = np.linalg.norm(query_visual)
            if query_norm == 0:
                return []
            query_visual_normalized = query_visual / query_norm

            # 2. 计算相似度 (NumPy点积)
            visual_similarities = np.dot(self.visual_features_normalized, query_visual_normalized.T).squeeze()

            # 3. 过滤并构建候选列表
            candidates = []
            for idx, score in enumerate(visual_similarities):
                study_id = self.study_ids[idx]
                
                # 过滤查询ID
                if exclude_study_id and study_id == exclude_study_id:
                    continue
                
                # 过滤阈值
                if score < self.min_similarity_threshold or score > self.max_similarity_threshold:
                    continue
                
                candidates.append((idx, score, study_id)) # (db_index, score, study_id)

            # 4. 按相关性排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            if not candidates:
                return []
            
            # 5. 【【【---- MMR 多样性过滤 ----】】】
            # 检查是否需要多样性且候选人足够多
            if self.diversity_factor > 0 and len(candidates) > self.top_k:
                
                selected = []
                selected_indices = []
                
                # 1. 首先添加相关性最高的
                best_candidate = candidates.pop(0)
                selected.append(best_candidate)
                selected_indices.append(best_candidate[0])

                # 准备剩余候选人的特征和分数
                remaining_indices = [c[0] for c in candidates]
                remaining_scores = {c[0]: c[1] for c in candidates}
                
                # 预计算剩余候选人的特征
                remaining_features = self.visual_features_normalized[remaining_indices]
                
                # 迭代选择
                lambda_ = 1.0 - self.diversity_factor
                
                while len(selected) < self.top_k and candidates:
                    
                    # 获取已选中的特征
                    selected_features = self.visual_features_normalized[selected_indices]
                    
                    # 计算剩余候选人与已选人群的最大相似度
                    # (len(remaining), len(selected))
                    sim_to_selected = np.dot(remaining_features, selected_features.T)
                    max_sim = np.max(sim_to_selected, axis=1) # (len(remaining),)
                    
                    best_mmr_score = -float('inf')
                    best_idx_in_remaining = -1
                    
                    for i in range(len(candidates)):
                        candidate_db_idx = remaining_indices[i]
                        relevance_score = remaining_scores[candidate_db_idx]
                        diversity_penalty = max_sim[i]
                        
                        mmr_score = lambda_ * relevance_score - (1.0 - lambda_) * diversity_penalty
                        
                        if mmr_score > best_mmr_score:
                            best_mmr_score = mmr_score
                            best_idx_in_remaining = i
                    
                    # 添加最佳MMR候选人
                    best_candidate = candidates.pop(best_idx_in_remaining)
                    selected.append(best_candidate)
                    selected_indices.append(best_candidate[0])
                    
                    # 更新剩余候选人
                    remaining_indices = [c[0] for c in candidates]
                    if remaining_indices: # 避免在最后一次迭代时出错
                        remaining_features = self.visual_features_normalized[remaining_indices]

                final_results = [(study_id, score) for _, score, study_id in selected]

            else:
                # 不需要多样性过滤，或候选人不足
                final_results = [(study_id, score) for _, score, study_id in candidates[:self.top_k]]
                
            return final_results
    # (这个方法在上一轮已修复，保持不变)
    def get_retrieved_cases(self, 
                            retrieval_results: List[Tuple[int, float]]) -> List[Dict]:
        """获取完整病例数据 (2-tuple compatible)"""
        cases = []
        for study_id, score in retrieval_results:
            case = self.case_database[study_id].copy()
            case['similarity'] = score
            case['study_id'] = study_id
            cases.append(case)
        
        return cases

    # (这个方法在上一轮已修复，保持不变)
    def get_statistics(self, retrieval_results: List[Tuple[int, float]]) -> Dict:
        """计算统计信息 (2-tuple compatible)"""
        if not retrieval_results:
            return {}
        
        similarities = [sim for _, sim in retrieval_results]
        
        return {
            'num_retrieved': len(retrieval_results),
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }


# ============================================================================
# 混合检索器 (保持不变)
# ============================================================================

class HybridRetriever(DiseaseAwareRetriever):
    """
    混合检索器：视觉 + 历史 + 疾病
    (注意：此类仍继承了DiseaseAwareRetriever的CPU密集型检索)
    """
    
    def __init__(self, 
                 case_database: Dict,
                 disease_classifier: Optional[DiseaseClassifier] = None,
                 top_k: int = 5,
                 visual_weight: float = 0.4,
                 history_weight: float = 0.2,
                 disease_weight: float = 0.4,
                 **kwargs): 
        
        super().__init__(
            case_database=case_database,
            disease_classifier=disease_classifier,
            top_k=top_k,
            disease_weight=disease_weight,
            visual_weight=visual_weight,
            **kwargs 
        )
        
        self.history_weight = history_weight
        
        # 提取历史特征
        self.history_features = []
        for study_id in self.study_ids:
            hist_feat = self.case_database[study_id]['history_feat']
            self.history_features.append(hist_feat)
        
        self.history_features = np.vstack(self.history_features)
        
        logger.info(f"HybridRetriever: visual={visual_weight}, "
                    f"history={history_weight}, disease={disease_weight}")
    
    def retrieve(self,
                 query_visual: np.ndarray,
                 query_history: Optional[np.ndarray] = None,
                 exclude_study_id: Optional[int] = None) -> List[Tuple]:
        """三模态检索 (CPU密集型)"""
        
        weights_sum = self.visual_weight + self.history_weight + self.disease_weight
        v_w = self.visual_weight / weights_sum
        h_w = self.history_weight / weights_sum
        d_w = self.disease_weight / weights_sum
        
        if query_visual.ndim == 1:
            query_visual = query_visual.reshape(1, -1)
        
        # 【瓶颈】
        visual_sim = cosine_similarity(query_visual, self.visual_features)[0]
        
        if query_history is not None:
            if query_history.ndim == 1:
                query_history = query_history.reshape(1, -1)
            # 【瓶颈】
            history_sim = cosine_similarity(query_history, self.history_features)[0]
        else:
            history_sim = np.zeros_like(visual_sim)
            h_w = 0
            v_w, d_w = v_w + h_w/2, d_w + h_w/2
        
        # 3. 疾病相似度
        disease_scores = np.zeros_like(visual_sim)
        if self.disease_classifier:
            device = next(self.disease_classifier.parameters()).device
            query_tensor = torch.from_numpy(query_visual.squeeze()).float().to(device)
            query_diseases = self.disease_classifier.get_top_diseases(query_tensor)
            
            for i, study_id in enumerate(self.study_ids):
                if study_id in self.disease_predictions:
                    db_diseases = self.disease_predictions[study_id]
                    disease_scores[i] = self._compute_disease_match_score(
                        query_diseases, db_diseases
                    )
        
        # 4. 融合
        final_scores = v_w * visual_sim + h_w * history_sim + d_w * disease_scores
        
        # 5. 排序和过滤
        candidates = []
        for i, study_id in enumerate(self.study_ids):
            if exclude_study_id and study_id == exclude_study_id:
                continue
            
            if final_scores[i] < self.min_similarity_threshold:
                continue
            
            metadata = {
                'visual_sim': float(visual_sim[i]),
                'history_sim': float(history_sim[i]) if query_history is not None else 0.0,
                'disease_score': float(disease_scores[i]),
                'final_score': float(final_scores[i])
            }
            
            candidates.append((study_id, float(final_scores[i]), metadata))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:self.top_k]