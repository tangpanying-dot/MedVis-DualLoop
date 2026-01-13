# kg_module/disease_graph_builder.py
"""
疾病图谱构建器（改进版）
包含：病史解析 + 历史报告解析 + 图谱构建
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging

from .disease_imaging_config import DISEASE_FINDING_MAP, FINDING_KEYWORDS

logger = logging.getLogger(__name__)


class PatientHistoryParser:
    """
    患者病史解析器（改进版）
    新增：历史报告解析
    """
    
    def __init__(self, 
                 current_study_datetime: Optional[str] = None,
                 text_encoder=None):
        self.current_study_datetime = current_study_datetime
        self.text_encoder = text_encoder
    
    def parse(self, 
              history: List[Dict],
              historical_reports: Optional[List[str]] = None) -> Dict:
        """
        解析患者病史 + 历史报告
        
        Returns:
            {
                'diagnoses': [...],
                'primary_disease': {...},
                'chronic_diseases': [...],
                'disease_timeline': [...],        # 新增
                'historical_findings': [...],      # 新增
                'imaging_evidence': {...}          # 新增
            }
        """
        if not history:
            return {
                'diagnoses': [],
                'primary_disease': None,
                'chronic_diseases': [],
                'disease_timeline': [],
                'historical_findings': [],
                'imaging_evidence': {}
            }
        
        # 收集所有诊断
        all_diagnoses = []
        disease_appearance_count = defaultdict(int)
        
        for admission in history:
            if admission is None or not isinstance(admission, dict):
                continue
            
            admittime = admission.get('admittime')
            drg_info = admission.get('drg')
            if drg_info is None or not isinstance(drg_info, dict):
                severity = 1.0
            else:
                severity = drg_info.get('severity', 1.0)
            
            diagnoses_list = admission.get('diagnoses', [])
            if diagnoses_list is None or not isinstance(diagnoses_list, list):
                continue
            
            for diag in diagnoses_list:
                if diag is None or not isinstance(diag, dict):
                    continue
                
                icd = diag.get('icd_code')
                desc = diag.get('description')
                if not icd or not desc:
                    continue
                
                months_ago = self._calculate_months_ago(admittime)
                
                diagnosis_info = {
                    'icd': icd,
                    'description': desc,
                    'severity': severity,
                    'admittime': admittime,
                    'months_ago': months_ago,
                    'is_primary': False
                }
                
                all_diagnoses.append(diagnosis_info)
                disease_appearance_count[icd] += 1
        
        # 识别主诊断
        if all_diagnoses:
            primary_disease = max(
                all_diagnoses,
                key=lambda x: (x['severity'], -x['months_ago'])
            )
            primary_disease['is_primary'] = True
        else:
            primary_disease = None
        
        # 识别慢性病
        chronic_diseases = [
            diag for diag in all_diagnoses
            if disease_appearance_count[diag['icd']] >= 2
        ]
        
        # 构建时间线
        disease_timeline = sorted(
            all_diagnoses,
            key=lambda x: x['months_ago'],
            reverse=True
        )
        
        # 解析历史报告
        historical_findings = []
        imaging_evidence = {}
        
        if historical_reports:
            historical_findings = self._parse_historical_reports(historical_reports)
            imaging_evidence = self._link_diseases_to_findings(
                all_diagnoses, 
                historical_findings
            )
        
        return {
            'diagnoses': all_diagnoses,
            'primary_disease': primary_disease,
            'chronic_diseases': chronic_diseases,
            'disease_timeline': disease_timeline,
            'historical_findings': historical_findings,
            'imaging_evidence': imaging_evidence
        }
    
    def _calculate_months_ago(self, admittime: str) -> float:
        """计算距今月数"""
        if not admittime or not self.current_study_datetime:
            return 0.0
        
        try:
            admit_dt = datetime.fromisoformat(admittime.replace('Z', '+00:00'))
            current_dt = datetime.fromisoformat(
                self.current_study_datetime.replace('Z', '+00:00')
            )
            months = (current_dt.year - admit_dt.year) * 12 + \
                    (current_dt.month - admit_dt.month)
            return max(0.0, float(months))
        except:
            return 0.0
    
    def _parse_historical_reports(self, reports: List[str]) -> List[Dict]:
        """从历史报告提取影像发现（规则based）"""
        findings = []
        
        for report in reports:
            if not report:
                continue
            
            report_lower = report.lower()
            
            for keyword, finding_name in FINDING_KEYWORDS.items():
                if keyword in report_lower:
                    findings.append({
                        'finding': finding_name,
                        'keyword': keyword
                    })
        
        # 去重
        unique_findings = {}
        for f in findings:
            fname = f['finding']
            if fname not in unique_findings:
                unique_findings[fname] = {
                    'finding': fname,
                    'keywords': [f['keyword']],
                }
            else:
                unique_findings[fname]['keywords'].append(f['keyword'])
        
        result = []
        for fname, data in unique_findings.items():
            num_keywords = len(set(data['keywords']))
            confidence = 'high' if num_keywords >= 2 else 'medium'
            
            result.append({
                'finding': fname,
                'confidence': confidence,
                'matched_keywords': list(set(data['keywords']))
            })
        
        logger.info(f"从历史报告提取到 {len(result)} 个影像发现")
        return result
    
    def _link_diseases_to_findings(self, 
                                   diagnoses: List[Dict],
                                   historical_findings: List[Dict]) -> Dict:
        """建立疾病-影像证据关联"""
        imaging_evidence = {}
        finding_names = [f['finding'] for f in historical_findings]
        
        for diag in diagnoses:
            icd = diag['icd']
            expected_findings = DISEASE_FINDING_MAP.get(icd, [])
            
            matched = [f for f in expected_findings if f in finding_names]
            
            if matched:
                imaging_evidence[icd] = matched
        
        logger.info(f"建立了 {len(imaging_evidence)} 个疾病-影像证据关联")
        return imaging_evidence


class DiseaseGraphBuilder:
    """
    疾病图谱构建器（改进版）
    """
    
    def __init__(self,
                 knowledge_base,
                 time_decay_lambda: float = 0.1,
                 severity_weight: float = 1.0):
        self.knowledge_base = knowledge_base
        self.time_decay_lambda = time_decay_lambda
        self.severity_weight = severity_weight
    
    def build_patient_graph(self,
                        parsed_history: Dict,
                        max_entities: int = 50) -> Dict:
        """构建患者个性化疾病图谱"""
        diagnoses = parsed_history['diagnoses']
        imaging_evidence = parsed_history.get('imaging_evidence', {})
        disease_timeline = parsed_history.get('disease_timeline', [])
        
        if not diagnoses:
            return {'entities': {}, 'relations': []}
        
        patient_icds = [d['icd'] for d in diagnoses]
        unique_icds = list(set(patient_icds))
        
        # 从知识库提取子图
        base_subgraph = self.knowledge_base.extract_disease_subgraph(
            unique_icds,
            max_hop=2,
            max_entities=max_entities
        )
        
        # 计算疾病权重（加入影像证据）
        disease_weights = self._calculate_disease_weights(
            diagnoses,
            imaging_evidence
        )
        
        # 为实体赋权重
        weighted_entities = {}
        for icd, entity_info in base_subgraph['entities'].items():
            if icd in disease_weights:
                weight = disease_weights[icd]
            else:
                weight = 0.3
            
            weighted_entities[icd] = {
                'type': entity_info.get('type', 'disease'),
                'weight': weight,
                'name': entity_info.get('name', icd),
                'category': entity_info.get('category', 'Unknown'),
                'has_imaging_evidence': icd in imaging_evidence
            }
        
        # 为关系赋权重
        weighted_relations = []
        for src, tgt, rel_type, base_weight in base_subgraph['relations']:
            src_weight = weighted_entities.get(src, {}).get('weight', 0.5)
            tgt_weight = weighted_entities.get(tgt, {}).get('weight', 0.5)
            relation_weight = base_weight * (src_weight + tgt_weight) / 2
            
            weighted_relations.append((src, tgt, rel_type, relation_weight))
        
        # 添加时序边或共现边
        temporal_edge_count = 0
        if disease_timeline and len(disease_timeline) > 1:
            # 多次就诊：添加时序边
            logger.info(f"病史时间线包含 {len(disease_timeline)} 个诊断")
            for i in range(len(disease_timeline) - 1):
                src_icd = disease_timeline[i]['icd']
                tgt_icd = disease_timeline[i+1]['icd']
                
                logger.debug(f"尝试连接: {src_icd} -> {tgt_icd}")
                
                if src_icd in weighted_entities and tgt_icd in weighted_entities and src_icd != tgt_icd:
                    weighted_relations.append((src_icd, tgt_icd, 'evolved_to', 0.9))
                    temporal_edge_count += 1
                    logger.debug(f"✓ 成功添加时序边")
                else:
                    logger.debug(f"✗ 跳过: src_in={src_icd in weighted_entities}, "
                            f"tgt_in={tgt_icd in weighted_entities}, "
                            f"different={src_icd != tgt_icd}")
        else:
            # 单次就诊或时间线为空：为所有诊断之间添加共现边
            logger.info(f"单次就诊或无时间线，病史中有 {len(diagnoses)} 个诊断")
            if len(diagnoses) > 1:
                for i, diag1 in enumerate(diagnoses):
                    for diag2 in diagnoses[i+1:]:
                        icd1, icd2 = diag1['icd'], diag2['icd']
                        if icd1 in weighted_entities and icd2 in weighted_entities and icd1 != icd2:
                            # 避免重复
                            if not any(r[0] == icd1 and r[1] == icd2 for r in weighted_relations):
                                weighted_relations.append((icd1, icd2, 'co-occurring', 0.8))
                                temporal_edge_count += 1
                                logger.debug(f"添加共现边: {icd1} <-> {icd2}")
        
        logger.info(f"构建患者图谱: {len(weighted_entities)} 实体, "
                f"{len(weighted_relations)} 关系 (其中推理边: {temporal_edge_count})")
        
        return {
            'entities': weighted_entities,
            'relations': weighted_relations
        }
    
    def _calculate_disease_weights(self, 
                                   diagnoses: List[Dict],
                                   imaging_evidence: Dict) -> Dict[str, float]:
        """计算疾病权重（加入影像证据加成）"""
        disease_weights = {}
        
        for diag in diagnoses:
            icd = diag['icd']
            severity = diag['severity']
            months_ago = diag['months_ago']
            is_primary = diag['is_primary']
            
            # 基础权重
            time_decay = np.exp(-self.time_decay_lambda * months_ago)
            weight = self.severity_weight * severity * time_decay
            
            if is_primary:
                weight *= 2.0
            
            # 影像证据加成
            if icd in imaging_evidence:
                num_evidence = len(imaging_evidence[icd])
                imaging_boost = 1.0 + 0.5 * num_evidence
                weight *= imaging_boost
                logger.debug(f"疾病 {icd} 有影像证据，权重增强 {imaging_boost:.2f}x")
            
            if icd in disease_weights:
                disease_weights[icd] = max(disease_weights[icd], weight)
            else:
                disease_weights[icd] = weight
        
        # 归一化
        if disease_weights:
            max_weight = max(disease_weights.values())
            disease_weights = {
                icd: w / max_weight for icd, w in disease_weights.items()
            }
        
        return disease_weights
    
    def _build_temporal_edges(self, 
                             disease_timeline: List[Dict],
                             entities: Dict) -> List[Tuple]:
        """构建时序边"""
        temporal_edges = []
        
        for i in range(len(disease_timeline) - 1):
            curr = disease_timeline[i]
            next_d = disease_timeline[i + 1]
            
            curr_icd = curr['icd']
            next_icd = next_d['icd']
            
            if curr_icd in entities and next_icd in entities:
                if self.knowledge_base.has_relation(
                    curr_icd, next_icd,
                    relation_types=['causes', 'progresses_to']
                ):
                    time_gap = abs(curr['months_ago'] - next_d['months_ago'])
                    time_weight = np.exp(-0.05 * time_gap)
                    
                    temporal_edges.append((
                        curr_icd,
                        next_icd,
                        'temporal_progression',
                        time_weight
                    ))
        
        return temporal_edges
    
    def get_graph_statistics(self, patient_graph: Dict) -> Dict:
        """计算图统计信息"""
        entities = patient_graph['entities']
        relations = patient_graph['relations']
        
        source_counts = defaultdict(int)
        for entity in entities.values():
            source_counts[entity.get('source', 'core')] += 1
        
        weights = [e.get('weight', 0) for e in entities.values()]
        
        return {
            'num_entities': len(entities),
            'num_relations': len(relations),
            'mean_weight': np.mean(weights) if weights else 0,
            'max_weight': np.max(weights) if weights else 0
        }