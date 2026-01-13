# -*- coding: utf-8 -*-
"""
V9 (适配新数据格式):
- 移除向量编码功能（不再依赖 TextEncoder）
- 适配 make_csv.py 输出的新格式：report 字段已是清洗后的纯文本
- 只生成历史文本文件供训练时使用
- 保留历史诊断和历史报告的两部分结构
"""
import pandas as pd
import json
from tqdm import tqdm
import sys
import os
import logging
from typing import Set

# --- 配置区 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置要加载的最近历史报告的数量
NUM_PAST_REPORTS_TO_LOAD = 2
# 设置最多保留的相关诊断数量
MAX_PAST_DIAGNOSES = 30

# V7: 心肺相关的关键词列表（用于过滤诊断）
RELEVANCE_KEYWORDS: Set[str] = {
    # 心脏 (Cardiac)
    'heart', 'cardiac', 'cardio', 'myocardial', 'pericardium', 'coronary', 'atrial', 'aortic', 'mitral', 'valve',
    'cardiomegaly', 'hypertension', 'aneurysm',
    
    # 肺/胸 (Pulmonary/Thoracic)
    'lung', 'pulmonary', 'pneumonia', 'thoracic', 'pleural', 'pneumothorax', 'bronchus', 'trachea',
    'edema', 'effusion', 'embolism', 'atelectasis', 'consolidation', 'opacity', 'hilar', 'mediastinal',
    'diaphragm', 'esophagus', 'rib', 'spine', 'clavicle', 'sternum',
    
    # 呼吸 (Respiratory)
    'respiratory', 'breath', 'airway', 'dyspnea', 'apnea', 'hypoxia', 'hypoxemia',
    
    # 慢性肺部疾病 (Chronic Lung Disease)
    'copd', 'emphysema', 'asthma', 'bronchiectasis', 'fibrosis', 'interstitial', 
    'tuberculosis', 'tb',
    
    # 肿瘤/占位 (Tumor/Mass)
    'cancer', 'carcinoma', 'tumor', 'mass', 'nodule', 'neoplasm', 'malignancy',
    'metastasis', 'metastatic', 'nodular',
    
    # 肾脏 (Renal - 经常与心衰相关)
    'renal', 'kidney',
    
    # 血管 (Vascular)
    'vascular', 'vessel', 'aorta', 'svc', 'venous', 'artery',
    'thrombus', 'thrombosis',
    
    # 常见影像学表现 (Common Imaging Findings)
    'infiltrate', 'infiltration', 'calcification', 'calcified',
    'cyst', 'cystic', 'cavity', 'reticular',
    'congestion', 'hemorrhage', 'bleeding',
    'infarction', 'ischemia', 'ischemic',
    'inflammation', 'inflammatory', 'infection',
    'fracture', 'enlarged', 'enlargement',
    
    # 淋巴系统 (Lymphatic)
    'lymph', 'lymphadenopathy',
    
    # 症状与诊断 (Symptoms & Diagnosis)
    'failure', 'disease', 'pain', 'shortness',
    'acute', 'chronic', 'abnormal', 'lesion'
}
# --- 配置区结束 ---


class HistoryRetriever:
    """
    V9: 简化版历史检索器
    只负责提取和组织历史文本，不再进行向量编码
    """
    def __init__(self, csv_path: str):
        self.required_columns = ['study_id', 'subject_id', 'study_datetime', 'history', 'report']
        try:
            logger.info("正在加载和预处理数据集...")
            df = pd.read_csv(csv_path)
            self._validate_dataframe(df)
            df['study_datetime'] = pd.to_datetime(df['study_datetime'], errors='coerce')
            df = df.sort_values(by=['subject_id', 'study_datetime']).reset_index(drop=True)
            self.df = df
            self.df['study_id'] = self.df['study_id'].astype('Int64')
            self.study_id_to_idx = {
                int(row['study_id']): idx 
                for idx, row in self.df.iterrows() 
                if pd.notna(row['study_id'])
            }
            logger.info(f"数据集加载完成，共 {len(self.df)} 条记录，有效 study_id: {len(self.study_id_to_idx)} 个")
        except Exception as e:
            logger.error(f"加载数据时发生错误: {str(e)}")
            sys.exit(1)
        
        logger.info(f"HistoryRetriever (V9 文本提取版) 初始化成功。")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """验证DataFrame是否包含必需的列"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必需的列: {missing_columns}")

    def _is_relevant(self, text: str) -> bool:
        """检查文本是否包含任何心肺相关关键词"""
        if not text:
            return False
        text_low = text.lower()
        for keyword in RELEVANCE_KEYWORDS:
            if keyword in text_low:
                return True
        return False

    def _parse_history_diagnoses(self, history_json_str: str) -> str:
        """
        解析并过滤历史诊断信息
        
        Args:
            history_json_str: history字段的JSON字符串
            
        Returns:
            格式化的诊断文本
        """
        if pd.isna(history_json_str) or not history_json_str:
            return ""
        
        try:
            history_list = json.loads(history_json_str)
            if not isinstance(history_list, list):
                return ""
            
            # 收集所有相关的诊断
            relevant_diagnoses = []
            
            for admission in history_list:
                if not isinstance(admission, dict):
                    continue
                
                diagnoses = admission.get('diagnoses', [])
                if not isinstance(diagnoses, list):
                    continue
                
                for diag in diagnoses:
                    if not isinstance(diag, dict):
                        continue
                    
                    description = diag.get('description', '')
                    if not description or description == "Unknown ICD Code":
                        continue
                    
                    # 过滤相关性
                    if self._is_relevant(description):
                        icd_code = diag.get('icd_code', '')
                        relevant_diagnoses.append({
                            'code': icd_code,
                            'description': description,
                            'seq': diag.get('seq_num', 999)
                        })
            
            # 去重（基于description）
            seen_descriptions = set()
            unique_diagnoses = []
            for diag in relevant_diagnoses:
                desc = diag['description']
                if desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    unique_diagnoses.append(diag)
            
            # 按seq_num排序，取前MAX_PAST_DIAGNOSES个
            unique_diagnoses.sort(key=lambda x: x['seq'])
            unique_diagnoses = unique_diagnoses[:MAX_PAST_DIAGNOSES]
            
            # 格式化输出
            if not unique_diagnoses:
                return ""
            
            lines = []
            for diag in unique_diagnoses:
                lines.append(f"- {diag['description']}")
            
            return "\n".join(lines)
            
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"解析history字段时出错: {str(e)}")
            return ""

    def _get_past_reports_text(self, current_idx: int) -> str:
        """
        获取历史影像报告文本（已清洗）
        
        Args:
            current_idx: 当前记录的索引
            
        Returns:
            格式化的历史报告文本
        """
        try:
            current_row = self.df.loc[current_idx]
            current_subject_id = current_row['subject_id']
            current_datetime = current_row['study_datetime']
            
            if pd.isna(current_datetime):
                return ""
            
            # 查找该患者的所有历史报告
            past_reports_df = self.df[
                (self.df['subject_id'] == current_subject_id) & 
                (self.df['study_datetime'] < current_datetime) &
                (pd.notna(self.df['study_datetime']))
            ]
            
            if past_reports_df.empty:
                return ""
            
            # 取最近的N份报告
            recent_past_reports_df = past_reports_df.tail(NUM_PAST_REPORTS_TO_LOAD)
            
            # 收集报告文本
            report_texts = []
            for _, row in recent_past_reports_df.iterrows():
                report_text = row['report']
                
                # 验证报告文本是否有效
                if pd.isna(report_text) or not report_text or len(str(report_text).strip()) < 10:
                    continue
                
                # 直接使用清洗后的文本（不再需要JSON解析）
                report_texts.append(f"Past Report: {str(report_text).strip()}")
            
            return "\n\n".join(report_texts)
            
        except Exception as e:
            logger.error(f"获取历史报告时出错: {str(e)}")
            return ""

    def generate_history_context(self, study_id: int) -> str:
        """
        为指定的study_id生成完整的历史上下文
        
        Args:
            study_id: 研究ID
            
        Returns:
            格式化的历史上下文文本
        """
        if study_id not in self.study_id_to_idx:
            logger.warning(f"Study ID {study_id} 不在数据集中")
            return ""
        
        idx = self.study_id_to_idx[study_id]
        current_row = self.df.loc[idx]
        
        # 1. 获取历史报告
        past_reports_text = self._get_past_reports_text(idx)
        
        # 2. 获取历史诊断
        history_diagnoses = self._parse_history_diagnoses(current_row['history'])
        
        # 3. 组合文本
        sections = []
        
        if past_reports_text:
            sections.append(f"[HISTORICAL IMAGING REPORTS]\n{past_reports_text}")
        
        if history_diagnoses:
            sections.append(f"[PATIENT MEDICAL HISTORY]\n{history_diagnoses}")
        
        return "\n\n".join(sections) if sections else ""

    def process_and_save_all(self, output_dir: str) -> bool:
        """
        处理所有study并保存历史文本
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        logger.info("=" * 70)
        logger.info("开始处理所有历史上下文...")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"历史报告数量: {NUM_PAST_REPORTS_TO_LOAD}")
        logger.info(f"最大诊断数量: {MAX_PAST_DIAGNOSES}")
        logger.info("=" * 70)
        
        # 创建输出目录
        output_txt_dir = os.path.join(output_dir, 'texts')
        os.makedirs(output_txt_dir, exist_ok=True)
        
        # 统计信息
        stats = {
            'total': len(self.study_id_to_idx),
            'with_reports': 0,
            'with_diagnoses': 0,
            'with_both': 0,
            'empty': 0,
            'errors': 0
        }
        
        # 处理所有study
        for study_id in tqdm(self.study_id_to_idx.keys(), desc="处理历史上下文"):
            try:
                # 生成历史文本
                history_text = self.generate_history_context(study_id)
                
                # 统计
                if not history_text:
                    stats['empty'] += 1
                    # 即使为空也创建文件（内容为空字符串）
                    history_text = ""
                else:
                    has_reports = '[HISTORICAL IMAGING REPORTS]' in history_text
                    has_diagnoses = '[PATIENT MEDICAL HISTORY]' in history_text
                    
                    if has_reports:
                        stats['with_reports'] += 1
                    if has_diagnoses:
                        stats['with_diagnoses'] += 1
                    if has_reports and has_diagnoses:
                        stats['with_both'] += 1
                
                # 保存文本文件
                txt_path = os.path.join(output_txt_dir, f"{study_id}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(history_text)
                    
            except Exception as e:
                logger.error(f"处理 study_id {study_id} 时出错: {e}")
                stats['errors'] += 1
                # 出错时也创建空文件
                txt_path = os.path.join(output_txt_dir, f"{study_id}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("")
        
        # 打印统计信息
        logger.info("=" * 70)
        logger.info("处理完成！统计信息：")
        logger.info(f"  总记录数:           {stats['total']:>6,}")
        logger.info(f"  包含历史报告:       {stats['with_reports']:>6,}  ({stats['with_reports']/stats['total']*100:>5.1f}%)")
        logger.info(f"  包含历史诊断:       {stats['with_diagnoses']:>6,}  ({stats['with_diagnoses']/stats['total']*100:>5.1f}%)")
        logger.info(f"  两者都有:           {stats['with_both']:>6,}  ({stats['with_both']/stats['total']*100:>5.1f}%)")
        logger.info(f"  无历史信息:         {stats['empty']:>6,}  ({stats['empty']/stats['total']*100:>5.1f}%)")
        logger.info(f"  处理错误:           {stats['errors']:>6,}")
        logger.info("=" * 70)
        logger.info(f"✅ 文本文件已保存至: {output_txt_dir}")
        logger.info("=" * 70)
        
        return stats['errors'] < stats['total']  # 只要不是全部失败就算成功


def main(csv_path: str, output_dir: str) -> bool:
    """
    主函数
    
    Args:
        csv_path: 数据集CSV路径
        output_dir: 输出目录
        
    Returns:
        是否成功
    """
    try:
        # 初始化检索器
        retriever = HistoryRetriever(csv_path)
        
        # 处理并保存所有历史文本
        success = retriever.process_and_save_all(output_dir)
        
        if success:
            logger.info("=" * 70)
            logger.info("🎉 历史上下文提取完成！")
            logger.info("=" * 70)
        else:
            logger.error("=" * 70)
            logger.error("❌ 历史上下文提取失败")
            logger.error("=" * 70)
        
        return success
        
    except Exception as e:
        logger.error(f"主流程发生严重错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 配置路径
    CSV_FILE_PATH = "data/processed_dataset.csv"
    OUTPUT_FEATURES_DIR = "retrieval/history_context"
    
    # 执行主函数
    success = main(CSV_FILE_PATH, OUTPUT_FEATURES_DIR)
    
    sys.exit(0 if success else 1)