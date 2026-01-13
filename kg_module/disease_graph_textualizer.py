# kg_module/disease_graph_textualizer.py
"""
增强版 Layer 2 文本化器
目标：生成更详细、结构化的临床指导Prompt
改进点：
1. 更详细的临床背景信息（疾病严重程度、时间）
2. 明确的视觉检查重点
3. 预期影像表现
4. 结构化输出（多个section）
"""
from typing import Dict, List

# 确保同目录下有 disease_imaging_config.py
from .disease_imaging_config import DISEASE_FINDING_MAP, DISEASE_CATEGORY_MAP


class DiseaseGraphTextualizer:
    """
    增强版文本化器
    将疾病风险转化为结构化的临床指导
    """
    
    def textualize(self, patient_graph: Dict, parsed_history: Dict) -> str:
        """
        将患者疾病图谱转化为结构化的临床指导文本
        
        Args:
            patient_graph: 疾病图谱，包含entities和relations
            parsed_history: 解析后的病史，包含primary_disease、chronic_diseases等
            
        Returns:
            结构化的临床指导文本
        """
        entities = patient_graph.get('entities', {})
        if not entities:
            return ""  # 无病史时不生成
        
        # 1. 获取核心诊断信息
        primary = parsed_history.get('primary_disease')
        chronic = parsed_history.get('chronic_diseases', [])
        imaging_evidence = parsed_history.get('imaging_evidence', {})
        
        # 2. 映射视觉征象（带fallback）
        target_findings = set()
        findings_by_disease = {}  # {疾病名: [影像发现]}
        
        # 主诊断的视觉征象
        if primary:
            icd = primary.get('icd')
            findings = None
            
            # 尝试精确匹配
            if icd in DISEASE_FINDING_MAP:
                findings = DISEASE_FINDING_MAP[icd]
            # 🔥 Fallback: 尝试前3位ICD码
            elif icd and len(icd) >= 3:
                icd_prefix = icd[:3]
                if icd_prefix in DISEASE_CATEGORY_MAP:
                    findings = DISEASE_CATEGORY_MAP[icd_prefix]
            
            if findings:
                target_findings.update(findings)
                findings_by_disease[primary.get('description')] = findings
        
        # 慢性病的视觉征象（只取前3个）
        for d in chronic[:3]:
            icd = d.get('icd')
            findings = None
            
            # 尝试精确匹配
            if icd in DISEASE_FINDING_MAP:
                findings = DISEASE_FINDING_MAP[icd]
            # 🔥 Fallback: 尝试前3位ICD码
            elif icd and len(icd) >= 3:
                icd_prefix = icd[:3]
                if icd_prefix in DISEASE_CATEGORY_MAP:
                    findings = DISEASE_CATEGORY_MAP[icd_prefix]
            
            if findings:
                target_findings.update(findings)
                findings_by_disease[d.get('description')] = findings
        
        # 3. 组装增强版Prompt（结构化输出）
        sections = []
        
        # === Section 1: 临床背景 ===
        clinical_context = self._build_clinical_context(
            primary, chronic, imaging_evidence
        )
        if clinical_context:
            sections.append(f"CLINICAL CONTEXT:\n{clinical_context}")
        
        # === Section 2: 视觉检查重点 ===
        visual_focus = self._build_visual_focus(
            target_findings, findings_by_disease
        )
        if visual_focus:
            sections.append(f"ASSESSMENT FOCUS:\n{visual_focus}")
        
        # === Section 3: 预期影像表现 ===
        expected_findings = self._build_expected_findings(
            primary, chronic, imaging_evidence
        )
        if expected_findings:
            sections.append(f"EXPECTED:\n{expected_findings}")
        
        # 用双换行分隔各section，增强可读性
        return "\n\n".join(sections)
    
    def _build_clinical_context(self, primary, chronic, imaging_evidence):
        """构建临床背景信息"""
        lines = []
        
        # 主诊断（包含严重程度和时间）
        if primary:
            desc = primary.get('description', 'Unknown condition')
            severity = primary.get('severity', 1)
            months_ago = primary.get('months_ago', 0)
            
            severity_text = self._get_severity_text(severity)
            time_text = self._get_time_text(months_ago)
            
            lines.append(
                f"- Primary: {desc} ({severity_text}, {time_text})"
            )
        
        # 慢性病（只列前2个，避免太长）
        if chronic:
            chronic_names = [d.get('description', 'Unknown') for d in chronic[:2]]
            lines.append(
                f"- Chronic: {', '.join(chronic_names)}"
            )
        
        # 历史影像证据
        if imaging_evidence:
            evidence_list = []
            for icd, findings in imaging_evidence.items():
                evidence_list.extend(findings)
            
            if evidence_list:
                # 去重并限制数量
                unique_evidence = list(set(evidence_list))[:4]
                lines.append(
                    f"- Prior imaging: {', '.join(unique_evidence)}"
                )
        
        return "\n".join(lines)
    
    def _build_visual_focus(self, target_findings, findings_by_disease):
        """构建视觉检查重点"""
        if not target_findings:
            return ""
        
        lines = []
        
        # 主要关注点（不超过5个）
        key_findings = list(target_findings)[:5]
        lines.append(
            f"Key findings to assess: {', '.join(key_findings)}"
        )
        
        # 按疾病分类的关注点（更具体，只展示最主要的1个疾病）
        if findings_by_disease:
            # 取第一个疾病（通常是primary disease）
            disease, findings = list(findings_by_disease.items())[0]
            lines.append(
                f"For {disease}: examine {', '.join(findings[:3])}"
            )
        
        return "\n".join(lines)
    
    def _build_expected_findings(self, primary, chronic, imaging_evidence):
        """构建预期影像表现"""
        lines = []
        
        # 根据疾病严重程度给出预期
        if primary:
            severity = primary.get('severity', 1)
            
            if severity >= 3:
                lines.append(
                    "Moderate to severe findings likely present"
                )
            elif severity >= 2:
                lines.append(
                    "Mild to moderate changes may be seen"
                )
            else:
                lines.append(
                    "Subtle changes may be present"
                )
        
        # 如果有历史影像，提示对比
        if imaging_evidence:
            lines.append(
                "Compare with prior imaging to assess progression"
            )
        
        return "\n".join(lines) if lines else ""
    
    def _get_severity_text(self, severity):
        """将严重程度数值转换为文本"""
        if severity >= 4:
            return "severe"
        elif severity >= 3:
            return "moderate-severe"
        elif severity >= 2:
            return "moderate"
        else:
            return "mild"
    
    def _get_time_text(self, months_ago):
        """将时间（月数）转换为更易读的文本"""
        if months_ago < 1:
            return "recent"
        elif months_ago < 6:
            return f"{int(months_ago)}mo ago"
        elif months_ago < 12:
            return "this year"
        elif months_ago < 24:
            return "1-2y ago"
        else:
            return ">2y ago"


# ============ 输出示例 ============
"""
增强版输出示例：

CLINICAL CONTEXT:
- Primary: Congestive heart failure (moderate, 6mo ago)
- Chronic: Hypertension, Diabetes mellitus
- Prior imaging: Cardiomegaly, Pulmonary Edema

ASSESSMENT FOCUS:
Key findings to assess: Cardiomegaly, Pulmonary Edema, Pleural Effusion, Pulmonary Congestion
For Congestive heart failure: examine Cardiomegaly, Pulmonary Edema, Pleural Effusion

EXPECTED:
Mild to moderate changes may be seen
Compare with prior imaging to assess progression
"""