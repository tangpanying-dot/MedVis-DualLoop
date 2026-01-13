#!/usr/bin/env python3
# ==========================================================
# make_csv_v16_smart_filter.py (智能分流最终版)
# 核心策略:
#   1. [智能拒收]: 仅对 "Communication/Dr" (约4000条) 执行一票否决
#   2. [强力清洗]: 对 "Technique/Comparison" (约40000条) 执行清洗并保留
#   3. [双轨输出]: Report(通顺) / Findings(高密度)
# ==========================================================

import os, glob, argparse, json
import pandas as pd
from tqdm import tqdm
import re
import sys

# 0 ──────────────────── 命令行参数解析 ────────────────────────────

def setup_arg_parser():
    argp = argparse.ArgumentParser(
        description="MIMIC-CXR 数据集构建 (v16.0 智能分流版)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argp.add_argument("--window", type=int, default=0,
                      help="历史时间窗口(天)。0=包含所有历史记录(推荐)")
    argp.add_argument("--skip-local-check", action='store_true',
                      help="跳过本地影像文件检查")
    return argp.parse_args()

args = setup_arg_parser()
WIN_DAYS = None if args.window <= 0 else args.window
SKIP_LOCAL_CHECK = args.skip_local_check

# 1 ──────────────────── 路径常量定义 ───────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")

CXR_DIR        = os.path.join(DATA_DIR, "mimic-cxr")
CXR_META_PATH  = os.path.join(CXR_DIR, "mimic-cxr-2.0.0-metadata.csv") 
CXR_REC_PATH   = os.path.join(CXR_DIR, "cxr-record-list.csv")
CXR_STUDY_PATH = os.path.join(CXR_DIR, "cxr-study-list.csv")

IV_DIR          = os.path.join(DATA_DIR, "mimic-iv")
ADMISSIONS_PATH = os.path.join(IV_DIR, "admissions.csv")
DIAGNOSES_PATH  = os.path.join(IV_DIR, "diagnoses_icd.csv")
ICD_DICT_PATH   = os.path.join(IV_DIR, "d_icd_diagnoses.csv")
DRGCODES_PATH   = os.path.join(IV_DIR, "drgcodes.csv")

stats = {
    'initial_studies': 0,
    'file_verified': 0,
    'reports_cleaned': 0,
    'rejected_toxic': 0,   # 因沟通/医生废话被拒收 (预计~4000)
    'rejected_quality': 0, # 因太短/空被拒收
    'findings_filled': 0,
    'final_records': 0
}

# 2 ──────────────────── 核心清洗逻辑 (智能分流) ──────────────────

def extract_sections(raw_text):
    findings = ""
    impression = ""
    headers = ['FINDINGS', 'IMPRESSION', 'CONCLUSION', 'RECOMMENDATION', 'OPINION']
    pattern = re.compile(r'^\s*(' + '|'.join(headers) + r')\s*:?', re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(raw_text))
    if not matches: return raw_text, "" 
    sections = {}
    for i, match in enumerate(matches):
        name = match.group(1).upper()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(raw_text)
        sections[name] = raw_text[start:end].strip()
    findings = sections.get('FINDINGS', '')
    impression = sections.get('IMPRESSION', sections.get('CONCLUSION', sections.get('RECOMMENDATION', '')))
    return findings, impression

def is_toxic_content(text):
    """
    【智能拒收过滤器】
    只针对 "沟通/通知/医生" 类无法清洗的硬伤进行拒收。
    注意：Technique 和 Comparison 不在这里，它们会被清洗掉，不拒收。
    """
    toxic_patterns = [
        # 沟通动词
        r"(?i)(?:communicated|paged|discussed|notified|contacted|telephoned|called\s+to)",
        r"(?i)read\s+back", 
        # 医生提及
        r"(?i)(?:Dr\.|physician|radiologist).{0,30}(?:aware|contact|note|discuss)",
        # 签字 (如果是正文里出现)
        r"(?i)(?:signed|dictated)\s+by"
    ]
    for pat in toxic_patterns:
        if re.search(pat, text): return True
    return False

def basic_clean(text):
    if not isinstance(text, str) or not text.strip(): return ""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text) 
    text = re.sub(r'__+', ' ', text)
    text = re.sub(r'\*\*+', ' ', text)
    return text.strip()

def remove_cleanable_noise(text):
    """
    L1: 移除 "良性噪音" (Technique, Comparison, Time)
    这些是可以安全删除而不影响病理描述的。
    """
    text = re.sub(r'\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM|am|pm)', '', text)
    patterns = [
        # Technique (良性，删掉即可)
        r"(?i)technique\s*:.*?(?:\.|$)", 
        r"(?i)differences\s+in\s+technique.*?(?:\.|$)",
        
        # Comparison (良性，删掉即可)
        r"(?i)comparison\s*(?:is)?\s*(?:made\s*)?(?:to|with).*?(?:\.|$)",
        r"(?i)(?:in\s+)?(?:direct\s+)?comparison\s+(?:is\s+)?(?:made\s+)?(?:to|with|from).*?(?:\.|$)",
        r"(?i)compared\s+(?:to|with|from).*?(?:\.|$)",
        r"(?i)intervals?\s+change(?:s)?\s+(?:since|from).*?(?:\.|$)",
        r"(?i)unchanged\s+compared\s+(?:to|with).*?(?:\.|$)",
        r"(?i)in\s+(?:with\s+)?the\s+study\s+of.*?(?:\.|$)",
        
        # 孤立词
        r"(?i)\bagain\b\s*", 
        r"(?i)\bprior\s+(?:study|exam|radiograph|film|images)\b",
        r"(?i)\bprevious\s+(?:study|exam|radiograph|film|images)\b",
        r"(?i)(?:on|dated)\s+\.\s",
        
        # 简单的签字行 (如果没被 Toxic 抓到，这里作为兜底删除)
        r"(?i)signed\s+by.*?(?:\.|$)",
        r"(?i)dictated\s+by.*?(?:\.|$)",
    ]
    for pat in patterns: text = re.sub(pat, " ", text)
    return text

def remove_views_strict(text):
    """L3: 移除视角描述 (Stage 1 用)"""
    patterns = [
        r"(?i)(?:frontal|PA|AP)\s+(?:and|&)\s+lateral\s+(?:views?|radiographs?|images?).*?(?:\.|$)",
        r"(?i)(?:single\s+)?(?:portable\s+)?(?:conventional\s+)?(?:AP|PA|frontal|lateral|upright|supine|semi-upright)\s+(?:portable\s+)?(?:view|radiograph|image|chest|film)s?\s+(?:of\s+the\s+chest)?.*?(?:\.|$)",
        r"(?i)^AP\s+chest\.?$",
        r"(?i)(?:on\s+)?(?:the\s+)?(?:frontal|lateral|PA|AP)\s+(?:view|radiograph|image)s?.*?(?:shows|demonstrates|is|pres)", 
        r"(?i)(?:frontal|lateral)\s+view.*?(?:\.|$)",
        r"(?i)(?:portable\s+)?(?:upright\s+)?views?\s+(?:of\s+the\s+chest\s+)?(?:are|were)\s+obtained.*?(?:\.|$)",
    ]
    for pat in patterns: text = re.sub(pat, " ", text)
    return text

def fix_punctuation(text):
    """修补标点和幽灵词"""
    ghosts = [
        r"As", r"In", r"Therefore", r"However", r"When", r"Thus", r"The",
        r"As\s+a\s+consequence", r"As\s+previously", r"In\s+the", r"In\s+addition"
    ]
    for g in ghosts:
        text = re.sub(rf"(?i)^{g}(?:,)?\s+", "", text)
        text = re.sub(rf"(?i):\s+{g}(?:,)?\s+", ": ", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"^[^a-zA-Z0-9]+", "", text)
    return text.strip()

def process_single_report(raw_text):
    findings_raw, impression_raw = extract_sections(raw_text)
    f_base = basic_clean(findings_raw)
    i_base = basic_clean(impression_raw)
    
    # 1. 智能拒收 (只针对 Toxic 4000)
    full_content = f"{f_base} {i_base}"
    if is_toxic_content(full_content):
        return None, None, 'rejected_toxic'

    # 2. 互补
    if not f_base and i_base:
        f_base = i_base
        stats['findings_filled'] += 1
    elif f_base and not i_base:
        i_base = f_base
    
    if not f_base: return None, None, 'rejected_quality'

    # 3. Report (Stage 2): 删良性噪音 (Technique/Comparison)，保留视角
    full_text = f"Findings: {f_base} Impression: {i_base}"
    r_s1 = remove_cleanable_noise(full_text)
    report_text = fix_punctuation(r_s1)

    # 4. Findings (Stage 1): 删良性噪音 + 删视角
    f_s1 = remove_cleanable_noise(f_base)
    f_s2 = remove_views_strict(f_s1)
    findings_clean = fix_punctuation(f_s2)
    
    # 质量检查
    if len(findings_clean) < 5: return None, None, 'rejected_quality'
    
    return report_text, findings_clean, 'valid'

# 3 ──────────────────── 数据加载 (保持强校验与去重) ──────────────────

def aggregate_paths_and_views(group):
    group = group.sort_values(by='path')
    return pd.Series({
        'image_paths': json.dumps(group['path'].tolist()),
        'view_positions': json.dumps(group['ViewPosition'].tolist())
    })

def load_and_process_cxr_data():
    print("\n" + "="*70)
    print("Step 1/5  载入并合并 MIMIC-CXR 数据")
    print("="*70)
    
    df_meta = pd.read_csv(CXR_META_PATH, 
                         usecols=["subject_id", "study_id", "dicom_id", "StudyDate", "StudyTime", "ViewPosition"])
    time_str = df_meta['StudyTime'].fillna(0).astype(int).astype(str).str.zfill(6)
    df_meta['study_dt'] = pd.to_datetime(df_meta['StudyDate'].astype(str) + ' ' + time_str, format='%Y%m%d %H%M%S', errors='coerce')
    df_meta = df_meta.dropna(subset=['study_dt']).sort_values('study_dt')
    
    df_rec = pd.read_csv(CXR_REC_PATH, usecols=["dicom_id", "path"])
    df_full = pd.merge(df_meta, df_rec, on='dicom_id')
    
    print("[聚合影像] ...")
    df_grouped = df_full.groupby(['subject_id', 'study_id']).apply(aggregate_paths_and_views).reset_index()
    
    df_stu = pd.read_csv(CXR_STUDY_PATH, usecols=["subject_id", "study_id", "path"])
    df_stu = df_stu.rename(columns={"path": "report_path"})

    df_final = pd.merge(df_grouped, df_stu, on=["subject_id", "study_id"])
    df_unique_time = df_meta[['subject_id', 'study_id', 'study_dt']].drop_duplicates(subset=['subject_id', 'study_id'], keep='first')
    df_final = pd.merge(df_final, df_unique_time, on=["subject_id", "study_id"])

    stats['initial_studies'] = len(df_final)
    print(f"✅ Metadata 唯一 Study 数: {len(df_final):,}")
    return df_final

def filter_for_downloaded_images(df_cxr, data_dir):
    print("\n" + "="*70)
    print("Step 2/5  过滤本地影像文件")
    print("="*70)
    
    img_files_root = os.path.join(data_dir, "mimic-cxr", "images")
    if not os.path.isdir(img_files_root): return pd.DataFrame()

    local_partitions = [d for d in os.listdir(img_files_root) if d.startswith('p') and d[1:].isdigit()]
    local_partitions.sort()
    print(f"🔍 本地分区: {', '.join(local_partitions)}")

    valid_prefixes = [p[1:] for p in local_partitions]
    df_cxr['pt_prefix'] = df_cxr['subject_id'].astype(str).str[:2]
    df_partition = df_cxr[df_cxr['pt_prefix'].isin(valid_prefixes)].copy()
    df_partition.drop(columns=['pt_prefix'], inplace=True)
    
    print(f"    - 分区过滤后: {len(df_partition):,}")
    if SKIP_LOCAL_CHECK: return df_partition

    print("🔍 核对文件路径...")
    def check_first_img(json_paths):
        try:
            paths = json.loads(json_paths)
            if not paths: return False
            rel_path = paths[0].replace('files/', 'images/').replace('.dcm', '.jpg')
            return os.path.exists(os.path.join(data_dir, "mimic-cxr", rel_path))
        except: return False

    mask = df_partition['image_paths'].apply(check_first_img)
    df_verified = df_partition[mask].copy()
    
    stats['file_verified'] = len(df_verified)
    print(f"✅ 有效本地数据: {stats['file_verified']:,}")
    return df_verified

def load_mimic_iv_aux_data():
    print("\n" + "="*70)
    print("Step 3/5  载入 MIMIC-IV 辅助表格")
    print("="*70)
    
    df_adm = pd.read_csv(ADMISSIONS_PATH, usecols=["subject_id", "hadm_id", "admittime", "dischtime"], parse_dates=["admittime", "dischtime"])
    df_diag = pd.read_csv(DIAGNOSES_PATH, usecols=["subject_id", "hadm_id", "seq_num", "icd_code"])
    df_icd = pd.read_csv(ICD_DICT_PATH, usecols=["icd_code", "long_title"])
    icd_map = dict(zip(df_icd.icd_code, df_icd.long_title))
    
    df_drg = pd.read_csv(DRGCODES_PATH, usecols=["hadm_id", "drg_type", "description", "drg_severity", "drg_mortality"])
    df_drg = df_drg[df_drg["drg_type"] == "APR"].drop(columns=["drg_type"])
    drg_map = df_drg.set_index("hadm_id").to_dict("index")
    
    diag_idx = {}
    for row in tqdm(df_diag.itertuples(index=False), total=len(df_diag), desc="Indexing"):
        if row.subject_id not in diag_idx: diag_idx[row.subject_id] = []
        diag_idx[row.subject_id].append((row.hadm_id, row.seq_num, row.icd_code))
    
    return df_adm, icd_map, drg_map, diag_idx

def generate_final_dataset(df_cxr, df_adm, diag_idx, ICD_MAP, DRG_MAP):
    print("\n" + "="*70)
    print("Step 4/5  生成最终数据集")
    print("="*70)
    
    hadm_full_info = {}
    for sid, records in diag_idx.items():
        for (hid, seq, code) in records:
            if hid not in hadm_full_info: hadm_full_info[hid] = []
            hadm_full_info[hid].append((seq, code))
    
    patient_timeline = {}
    for row in df_adm.itertuples(index=False):
        sid = row.subject_id
        if sid not in patient_timeline: patient_timeline[sid] = []
        diags_list = []
        raw_diags = hadm_full_info.get(row.hadm_id, [])
        raw_diags.sort(key=lambda x: x[0])
        for seq, code in raw_diags[:15]:
            diags_list.append({"seq_num": seq, "icd_code": code, "description": ICD_MAP.get(code, "Unknown")})
        drg = DRG_MAP.get(row.hadm_id, {})
        patient_timeline[sid].append({
            "hadm_id": row.hadm_id, "admittime": row.admittime, "dischtime": row.dischtime,
            "diagnoses": diags_list, "drg": drg if drg else None
        })
    for sid in patient_timeline: patient_timeline[sid].sort(key=lambda x: x["dischtime"])
    
    dataset = []
    reports_root = os.path.join(DATA_DIR, "mimic-cxr")
    
    for row in tqdm(df_cxr.itertuples(index=False), total=len(df_cxr), desc="生成数据集"):
        sid, stid, t0 = row.subject_id, row.study_id, row.study_dt
        if pd.isna(t0): continue
        
        history_list = []
        if sid in patient_timeline:
            for adm in patient_timeline[sid]:
                if adm["dischtime"] < t0:
                    if WIN_DAYS and (t0 - adm["dischtime"]).days > WIN_DAYS: continue
                    adm_copy = adm.copy()
                    adm_copy["admittime"] = adm_copy["admittime"].isoformat()
                    adm_copy["dischtime"] = adm_copy["dischtime"].isoformat()
                    history_list.append(adm_copy)
        
        try:
            with open(os.path.join(reports_root, row.report_path), 'r', encoding='utf-8') as f:
                raw_report = f.read()
            
            # 【调用智能分流逻辑】
            r_clean, f_clean, status = process_single_report(raw_report)
            
            if status == 'rejected_toxic':
                stats['rejected_toxic'] += 1
                continue
            elif status == 'rejected_quality':
                stats['rejected_quality'] += 1
                continue
                
            stats['reports_cleaned'] += 1
            
            dataset.append({
                "study_id": int(stid),
                "subject_id": int(sid),
                "study_datetime": t0.isoformat(), 
                "image_paths": row.image_paths,
                "view_positions": row.view_positions,
                "history": json.dumps(history_list, ensure_ascii=False),
                "report": r_clean,      
                "findings": f_clean,
                "report_raw": raw_report 
            })
        except Exception as e:
            continue
    
    stats['final_records'] = len(dataset)
    return dataset

def save_to_csv(dataset):
    print("\n" + "="*70)
    print("Step 5/5  保存CSV文件")
    print("="*70)
    if not dataset:
        print("❌ 数据集为空")
        return
    out_csv = os.path.join(DATA_DIR, "processed_dataset.csv")
    df_out = pd.DataFrame(dataset)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ 成功保存 {len(dataset):,} 条记录")
    print(f"📁 文件: {out_csv}")

def print_final_summary():
    print("\n" + "="*70)
    print("最终统计摘要 (v16.0 智能分流版)")
    print("="*70)
    print(f"本地文件有效数:       {stats['file_verified']:>10,}")
    print("-" * 40)
    print(f"清洗成功 (保留):      {stats['reports_cleaned']:>10,}")
    print(f"拒收-毒性 (沟通类):   {stats['rejected_toxic']:>10,} (约 5%)")
    print(f"拒收-质量 (太短):     {stats['rejected_quality']:>10,}")
    print(f"Findings 互补填充:    {stats['findings_filled']:>10,}")
    print("-" * 40)
    print(f"最终保存记录:         {stats['final_records']:>10,}")
    print("="*70)

if __name__ == '__main__':
    try:
        df_cxr = load_and_process_cxr_data()
        df_cxr = filter_for_downloaded_images(df_cxr, DATA_DIR)
        
        if df_cxr.empty:
            print("\n❌ 错误: 没有有效的本地数据")
            exit(1)
        
        df_adm, ICD_MAP, DRG_MAP, diag_idx = load_mimic_iv_aux_data()
        final_dataset = generate_final_dataset(df_cxr, df_adm, diag_idx, ICD_MAP, DRG_MAP)
        
        save_to_csv(final_dataset)
        print_final_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)