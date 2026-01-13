# kg_module/disease_imaging_config.py
"""
疾病-影像发现映射配置（扩充版）
目标：提升覆盖率从46%到70%+
"""

# 疾病 → 影像发现映射
DISEASE_FINDING_MAP = {
    # ============ 心脏疾病 ============
    '428': ['Cardiomegaly', 'Pulmonary Edema', 'Pleural Effusion'],  # CHF
    '4280': ['Cardiomegaly', 'Pulmonary Congestion'],  # CHF unspecified
    '4281': ['Cardiomegaly', 'Pulmonary Edema'],  # Acute systolic HF
    '42821': ['Cardiomegaly', 'Pulmonary Edema'],  # 🔥 Acute diastolic HF (新增)
    '42831': ['Cardiomegaly', 'Pulmonary Edema'],  # 🔥 Acute on chronic diastolic HF (新增)
    '4148': ['Cardiomegaly'],  # Other forms of chronic ischemic heart disease
    '414': ['Cardiomegaly'],  # Chronic ischemic heart disease
    '41401': ['Cardiomegaly', 'Tortuous Aorta'],  # 🔥 Coronary atherosclerosis (新增)
    
    # 高血压
    '401': ['Cardiomegaly', 'Tortuous Aorta'],  # 🔥 Essential hypertension (新增)
    '4011': ['Cardiomegaly', 'Tortuous Aorta'],  # Benign hypertension
    '4019': ['Cardiomegaly', 'Tortuous Aorta'],  # Unspecified hypertension
    '4031': ['Cardiomegaly', 'Tortuous Aorta'],  # 🔥 Hypertensive chronic kidney disease (新增)
    
    # 心律失常
    '4272': ['Cardiomegaly'],  # 🔥 Atrial fibrillation (新增)
    '42731': ['Cardiomegaly'],  # 🔥 Atrial flutter (新增)
    
    # ============ 肺部疾病 ============
    '486': ['Consolidation', 'Infiltrate', 'Lung Opacity'],  # Pneumonia
    '481': ['Consolidation', 'Lobar Opacity'],  # Pneumococcal pneumonia
    '482': ['Consolidation', 'Infiltrate'],  # Other bacterial pneumonia
    '48241': ['Consolidation', 'Infiltrate'],  # 🔥 MRSA pneumonia (新增)
    '485': ['Consolidation', 'Infiltrate'],  # Bronchopneumonia
    
    # 呼吸衰竭
    '518': ['Lung Opacity', 'Atelectasis'],  # 🔥 Respiratory failure (新增，从验证中看到)
    '51881': ['Lung Opacity', 'Atelectasis'],  # 🔥 Acute respiratory failure (新增)
    '5184': ['Atelectasis', 'Pulmonary Edema'],  # Acute edema of lung
    '5185': ['Pulmonary Edema'],  # Pulmonary insufficiency
    '51882': ['ARDS', 'Bilateral Opacities'],  # 🔥 ARDS (新增)
    
    # COPD/哮喘
    '496': ['Hyperinflation', 'Emphysema'],  # COPD
    '492': ['Emphysema', 'Hyperinflation'],  # Emphysema
    '49121': ['Hyperinflation', 'Lung Opacity'],  # 🔥 Obstructive chronic bronchitis with exacerbation (新增)
    '49322': ['Hyperinflation', 'Lung Opacity'],  # 🔥 Asthma with exacerbation (新增)
    
    # ============ 感染/败血症 ============
    '038': ['Consolidation', 'ARDS', 'Pleural Effusion'],  # Septicemia
    '0389': ['Consolidation', 'Infiltrate'],  # Unspecified septicemia
    '03811': ['Consolidation', 'Lung Opacity'],  # 🔥 Septicemia due to E. coli (新增)
    '99591': ['Consolidation', 'ARDS', 'Pleural Effusion'],  # 🔥 Sepsis (新增)
    '99592': ['Consolidation', 'ARDS', 'Pleural Effusion'],  # 🔥 Severe sepsis (新增)
    
    # ============ 肾脏疾病 ============
    '585': ['Pulmonary Edema', 'Pleural Effusion', 'Cardiomegaly'],  # CKD
    '5859': ['Pulmonary Edema', 'Pleural Effusion'],  # CKD stage unspecified
    '586': ['Pulmonary Edema'],  # Renal failure unspecified
    '584': ['Pulmonary Edema'],  # 🔥 Acute kidney failure (新增，从验证中看到)
    '5849': ['Pulmonary Edema'],  # 🔥 AKI unspecified (新增)
    
    # ============ 肝脏疾病 ============
    '5715': ['Pleural Effusion', 'Ascites', 'Hepatomegaly'],  # Cirrhosis
    '5716': ['Pleural Effusion', 'Ascites'],  # Biliary cirrhosis
    '5723': ['Pleural Effusion', 'Pulmonary Hypertension'],  # Hepatorenal syndrome
    '571': ['Pleural Effusion', 'Ascites'],  # Chronic liver disease
    '07070': ['Pleural Effusion'],  # Hepatitis C
    '07071': ['Pleural Effusion'],  # Hepatitis C with hepatic coma
    '07044': ['Pleural Effusion'],  # Hepatitis C
    '070': ['Pleural Effusion'],  # Viral hepatitis
    
    # ============ 胸腔疾病 ============
    '5119': ['Pleural Effusion'],  # Pleurisy with effusion
    '511': ['Pleural Effusion'],  # Pleurisy
    '512': ['Pneumothorax'],  # Pneumothorax
    '5121': ['Pneumothorax'],  # Iatrogenic pneumothorax
    
    # ============ 肿瘤 ============
    '197': ['Lung Mass', 'Nodule', 'Pleural Effusion'],  # Secondary malignant neoplasm of respiratory
    '162': ['Lung Mass', 'Nodule', 'Consolidation'],  # Malignant neoplasm of bronchus and lung
    '1622': ['Lung Mass', 'Nodule'],  # Upper lobe lung cancer
    '1625': ['Lung Mass', 'Nodule'],  # Lower lobe lung cancer
    
    # ============ 症状相关疾病 ============
    # 注意：这些是症状，但可能提示某些影像学异常
    '78650': ['Chest pain'],  # 🔥 Chest pain (新增，虽然不是影像发现，但常见)
    '7862': ['Cough'],  # Cough
    '7866': ['Shortness of breath'],  # 🔥 Shortness of breath/dyspnea (新增)
    '78605': ['Shortness of breath'],  # 🔥 Shortness of breath (新增)
    
    # ============ 其他常见疾病 ============
    '250': ['Infiltrate'],  # Diabetes
    '78959': ['Ascites'],  # Ascites
    '5990': [],  # 🔥 UTI (新增，但无特定影像表现)
    '59900': [],  # 🔥 UTI site not specified (新增)
    '682': [],  # 🔥 Cellulitis (新增，无特定胸部影像表现)
    
    # ============ 外伤 ============
    '8070': ['Rib Fracture'],  # Multiple fractures involving ribs
    '8072': ['Rib Fracture'],  # Closed fracture of one rib
}

# 🔥 新增：通用映射（针对未明确映射的疾病）
# 如果疾病不在DISEASE_FINDING_MAP中，根据疾病类别给出通用指导
DISEASE_CATEGORY_MAP = {
    # 心脏相关（ICD前3位）
    '428': ['Cardiomegaly', 'Pulmonary Edema'],  # Heart failure
    '427': ['Cardiomegaly'],  # Cardiac dysrhythmias
    '414': ['Cardiomegaly'],  # Ischemic heart disease
    '410': ['Cardiomegaly'],  # Acute myocardial infarction
    '401': ['Cardiomegaly', 'Tortuous Aorta'],  # Hypertension
    '403': ['Cardiomegaly', 'Tortuous Aorta'],  # Hypertensive kidney disease
    
    # 肺部相关
    '486': ['Consolidation', 'Lung Opacity'],  # Pneumonia
    '482': ['Consolidation', 'Lung Opacity'],  # Pneumonia
    '518': ['Lung Opacity', 'Atelectasis'],  # Respiratory failure
    '496': ['Hyperinflation'],  # COPD
    '493': ['Hyperinflation', 'Lung Opacity'],  # Asthma
    
    # 感染
    '038': ['Consolidation', 'ARDS'],  # Septicemia
    '995': ['Consolidation', 'ARDS'],  # Sepsis
    
    # 肾脏
    '584': ['Pulmonary Edema'],  # Acute kidney failure
    '585': ['Pulmonary Edema', 'Pleural Effusion'],  # CKD
    
    # 肝脏
    '571': ['Pleural Effusion', 'Ascites'],  # Liver disease
    
    # 胸腔
    '511': ['Pleural Effusion'],  # Pleurisy
    '512': ['Pneumothorax'],  # Pneumothorax
}

# 影像发现关键词库（保持不变）
FINDING_KEYWORDS = {
    'effusion': 'Pleural Effusion',
    'pleural effusion': 'Pleural Effusion',
    'pleural fluid': 'Pleural Effusion',
    'costophrenic angle': 'Pleural Effusion',
    
    'cardiomegaly': 'Cardiomegaly',
    'enlarged heart': 'Cardiomegaly',
    'cardiac silhouette': 'Cardiomegaly',
    
    'consolidation': 'Consolidation',
    'airspace consolidation': 'Consolidation',
    
    'opacity': 'Lung Opacity',
    'opacities': 'Lung Opacity',
    'airspace opacity': 'Lung Opacity',
    
    'infiltrate': 'Infiltrate',
    'infiltrates': 'Infiltrate',
    
    'edema': 'Pulmonary Edema',
    'pulmonary edema': 'Pulmonary Edema',
    'vascular congestion': 'Pulmonary Edema',
    
    'pneumothorax': 'Pneumothorax',
    'collapsed lung': 'Pneumothorax',
    
    'atelectasis': 'Atelectasis',
    'volume loss': 'Atelectasis',
    
    'nodule': 'Nodule',
    'nodules': 'Nodule',
    'mass': 'Mass',
    'lung mass': 'Mass',
    
    'emphysema': 'Emphysema',
    'hyperinflation': 'Hyperinflation',
    'ascites': 'Ascites',
    'ards': 'ARDS',
}

# 🔥 Tier分级（保持不变，用于Gating）
TIER_1_FINDINGS = [
    'Pneumothorax',
    'Pneumonia',
    'Pulmonary Edema',
    'Pleural Effusion',
    'Fracture',
    'ARDS'
]

TIER_2_FINDINGS = [
    'Cardiomegaly',
    'Atelectasis',
    'Consolidation',
    'Lung Opacity'
]