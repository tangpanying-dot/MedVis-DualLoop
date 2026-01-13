# generate_report_final_resume.py
"""
Generate Reports Best-of-4 Ensemble (With Resume Capability)
- Features: Best-of-4 generation, Dynamic Weighted Selection.
- New Feature: Resume from existing JSONL file (--resume_file).
- Optimization: Skips visual encoding/generation for already processed IDs.
"""

import os
import sys
import torch
import torch.nn as nn
import json
import pandas as pd
import argparse
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

# ==================== Imports from Project Modules ====================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'visual'))
from visual.label_conditioned_connector import LabelConditionedConnector, ConnectorConfig
from visual.multiview_dataset_with_labels import MultiViewFeatureDataset, collate_fn_with_padding
from kg_bridge import kg_engine

# Import the independent evaluator module
from report_evaluator import UnifiedEvaluator, clean_text_basic

# ==================== Config ====================

CHEXPERT_CLASS_MAP = {
    0: 'Enlarged Cardiomediastinum', 1: 'Cardiomegaly', 2: 'Lung Opacity', 3: 'Lung Lesion',
    4: 'Edema', 5: 'Consolidation', 6: 'Pneumonia', 7: 'Atelectasis', 8: 'Pneumothorax',
    9: 'Pleural Effusion', 10: 'Pleural Other', 11: 'Fracture', 12: 'Support Devices', 13: 'No Finding'
}
NAME_TO_INDEX = {v: k for k, v in CHEXPERT_CLASS_MAP.items()}

CONFIG = {
    'llm_name': '/home/tpy/TPY/llm/gemma-2b-it', 
    'dtype': torch.bfloat16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_path': 'checkpoints/stage2/best_checkpoint_metric.pt',
    'classifier_path': 'checkpoints/classifier_auc/final_best_model.pt',
    'feature_dir': 'visual/visual_features/rad_dino',
    'csv_file': 'data/processed_dataset_test.csv',
    'chexpert_csv': 'data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv',
    'history_dir': 'retrieval/history_context/texts',
    'old_visual_dir': 'visual/visual_features/multi_scale',
    'radgraph_path': 'data/graph/radgraph/MIMIC-CXR_graphs.json',
    'train_csv': 'data/processed_dataset_train.csv',
    'kb_path': 'data/disease_knowledge_base.json',
    'enable_kg1': True,
    'enable_kg2': True, 
    'use_history': True,
    'output_file': '', # Will be set dynamically
    'resume_file': None, # New config param
    'batch_size': 16,             
    'num_workers': 4,
    'max_new_tokens': 200, 
    'min_new_tokens': 15,       
    'num_beams': 4,               
    'repetition_penalty': 1.02,  
    'length_penalty': 1.0,
    'visual_gate_threshold': 0.15,
    'visual_high_conf_threshold': 0.85,
}

PROMPT_TEMPLATE = """You are a radiologist. Write a chest X-ray report.

[Context]
{kg_context}

[History]
{history}

[Report]
FINDINGS:""" 

# ==================== Visual & Model Classes ====================

class VisualClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=14, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, features, mask=None):
        B = features.shape[0]
        query = self.query.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attention(query, features, features, key_padding_mask=key_padding_mask)
        return self.classifier(attn_out.squeeze(1))

def load_models(config):
    print(f"🚀 Loading generation models...")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    
    base = AutoModelForCausalLM.from_pretrained(config['llm_name'], torch_dtype=config['dtype'], device_map=config['device'])
    ckpt = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=False)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=64, lora_alpha=128, lora_dropout=0.05, 
                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(base, peft_config)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    
    conn = LabelConditionedConnector(ConnectorConfig())
    conn.load_state_dict(ckpt.get('connector_state_dict', ckpt), strict=True)
    conn = conn.to(config['device'], dtype=config['dtype']).eval()
    
    clf = VisualClassifier().to(config['device'])
    cls_ckpt = torch.load(config['classifier_path'], map_location='cpu', weights_only=False)
    clf.load_state_dict(cls_ckpt['model_state_dict'] if 'model_state_dict' in cls_ckpt else cls_ckpt)
    clf.eval()
    return model, conn, tokenizer, clf

def get_dataloader(config):
    ds = MultiViewFeatureDataset(
        feature_dir=config['feature_dir'], csv_file=config['csv_file'], split='test',  
        chexpert_csv=config['chexpert_csv'], history_dir=config['history_dir'],
        max_history_length=512, target_col='report'
    )
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate_fn_with_padding)

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    return df.set_index('study_id').to_dict('index')

# ==================== KG Logic ====================

def parse_kg2_findings(raw):
    if not raw or not raw.strip(): return []
    findings = []
    if '\n' in raw:
        for line in raw.split('\n'):
            line = line.strip()
            if not line: continue
            for name in CHEXPERT_CLASS_MAP.values():
                if name.lower() in line.lower():
                    findings.append((name, line))
                    break
    else:
        for item in raw.split(','):
            item = item.strip()
            if item: findings.append((item.split('[')[0].strip() if '[' in item else item, item))
    return findings

def map_finding_to_chexpert(name):
    low = name.lower()
    for n, idx in NAME_TO_INDEX.items():
        if n.lower() in low: return idx, idx in [1, 4, 8, 9]
    fuzzy = {'edema': (4, True), 'effusion': (9, True), 'pneumonia': (6, False), 'consolidation': (5, False), 
             'opacity': (2, False), 'infiltrate': (2, False), 'atelectasis': (7, False)}
    for k, (idx, crit) in fuzzy.items():
        if k in low: return idx, crit
    return None, False

def apply_gating(kg2_raw, soft_labels, gate_th, conf_th):
    findings = parse_kg2_findings(kg2_raw)
    kept_texts, kept_names = [], set()
    for fname, raw_txt in findings:
        idx, crit = map_finding_to_chexpert(fname)
        keep = True
        if idx is not None:
            if not (crit or soft_labels[idx].item() > gate_th): keep = False
        if keep:
            kept_texts.append(raw_txt)
            kept_names.add(fname)
            
    vis_inj = []
    for idx, name in CHEXPERT_CLASS_MAP.items():
        th = 0.5 if name == 'Support Devices' else conf_th
        if soft_labels[idx].item() > th and not any(name.lower() in kn.lower() for kn in kept_names):
            vis_inj.append("Support Devices (Tubes/Lines)" if name == 'Support Devices' else name)
    return ("\n".join(kept_texts) if kept_texts else ""), ", ".join(vis_inj)

def format_prompts(batch_hist, batch_ids, csv_dict, config, template, batch_soft_labels, ovr_kg1, ovr_kg2):
    prompts = []
    use_kg1 = config['enable_kg1'] if ovr_kg1 is None else ovr_kg1
    use_kg2 = config['enable_kg2'] if ovr_kg2 is None else ovr_kg2
    
    for i, (hist, sid) in enumerate(zip(batch_hist, batch_ids)):
        raw_hist = hist if hist else ""
        has_hist = (len(raw_hist.strip()) > 0) and ("[PATIENT MEDICAL HISTORY]" in raw_hist)
        clean_hist = raw_hist.strip()[:1500] if has_hist and config['use_history'] else "No prior history."
        
        kg1_p, kg2_p = "", ""
        if use_kg1 or use_kg2:
            row = csv_dict.get(str(sid), {})
            try:
                k1_f, k2_f = kg_engine.get_prompts(sid, row.get('history', ''), row.get('study_datetime', ''), 
                                                 os.path.join(config['old_visual_dir'], f"{sid}.npy"))
                if use_kg1 and not has_hist: kg1_p = k1_f
                if use_kg2: kg2_p = k2_f
            except: pass
            
        vis_str = ""
        if use_kg2 and batch_soft_labels is not None:
            kg2_p, vis_str = apply_gating(kg2_p, batch_soft_labels[i], config['visual_gate_threshold'], config['visual_high_conf_threshold'])
            
        kg_ctx = (kg2_p + "\n" + vis_str + "\n" + kg1_p.replace("Reference patterns from similar cases:", "").strip()[:150]).strip()
        prompts.append(template.format(kg_context=kg_ctx, history=clean_hist))
    return prompts

# ==================== Main Generation Loop ====================

@torch.no_grad()
def generate(model, connector, tokenizer, classifier, loader, csv_dict, config):
    evaluator = UnifiedEvaluator(device=config['device'])
    
    # --- Resume Logic ---
    processed_ids = set()
    outfile = config['resume_file']
    file_mode = 'w'
    
    if outfile and os.path.exists(outfile):
        print(f"🔄 Resuming from: {outfile}")
        try:
            with open(outfile, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if 'study_id' in data:
                            processed_ids.add(str(data['study_id']))
                    except json.JSONDecodeError:
                        pass
            print(f"✅ Found {len(processed_ids)} already processed studies.")
            file_mode = 'a' # Append mode
        except Exception as e:
            print(f"⚠️ Error reading resume file: {e}. Starting fresh.")
            file_mode = 'w'
    else:
        # If no resume file provided or it doesn't exist, create new one
        if not outfile:
            outfile = f'report/generated_reports_gemma2b_{datetime.now().strftime("%Y%m%d_%H%M")}.jsonl'
        print(f"📝 Starting new output file: {outfile}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    f_out = open(outfile, file_mode, encoding='utf-8')
    # --------------------

    strategies = [
        {'name': 'Vis+Hist',           'kg1': False, 'kg2': False},
        {'name': 'Vis+Hist+KG1',       'kg1': True,  'kg2': False},
        {'name': 'Vis+Hist+KG2',       'kg1': False, 'kg2': True},
        {'name': 'Vis+Hist+KG1+KG2',   'kg1': True,  'kg2': True},
    ]
    
    # Iterate over dataloader
    for batch in tqdm(loader, desc="Best-of-4"):
        # Identify which samples in this batch are NOT yet processed
        batch_ids_str = [str(sid) for sid in batch['study_ids']]
        indices_to_run = [i for i, sid in enumerate(batch_ids_str) if sid not in processed_ids]
        
        # Optimization: If entire batch is processed, skip
        if len(indices_to_run) == 0:
            continue
            
        B_active = len(indices_to_run)
        device = config['device']
        
        # Slice batch tensors/lists to only include unprocessed items
        # Visual features [Batch, Seq, Dim] -> slice first dim
        vis_feats = batch['visual_features'][indices_to_run].to(device).to(config['dtype'])
        vis_mask = batch['visual_mask'][indices_to_run].to(device)
        
        # Sliced metadata
        active_ids = [batch['study_ids'][i] for i in indices_to_run]
        active_reports = [batch['reports'][i] for i in indices_to_run]
        active_histories = [batch['history_texts'][i] for i in indices_to_run]
        
        # 1. Visual Forward (Only for active items)
        logits = classifier(vis_feats.float(), vis_mask)
        preds = torch.sigmoid(logits) 
        vis_tokens = connector(vis_feats, preds.to(config['dtype']), vis_mask)
        
        # 2. Generate 4 Candidates
        candidates = [[] for _ in range(B_active)] 
        
        for strat in strategies:
            prompts = format_prompts(active_histories, active_ids, csv_dict, config, PROMPT_TEMPLATE, preds, strat['kg1'], strat['kg2'])
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            p_embeds = model.get_input_embeddings()(inputs.input_ids)
            in_embeds = torch.cat([vis_tokens, p_embeds], dim=1)
            att_mask = torch.cat([torch.ones((B_active, vis_tokens.shape[1]), dtype=torch.long, device=device), inputs.attention_mask.long()], dim=1)
            
            outputs = model.generate(
                inputs_embeds=in_embeds, attention_mask=att_mask, max_new_tokens=config['max_new_tokens'], 
                min_new_tokens=config['min_new_tokens'], num_beams=config['num_beams'], 
                repetition_penalty=config['repetition_penalty'], pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id, use_cache=True
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                clean_text = clean_text_basic(text)
                candidates[i].append({'text': "FINDINGS: " + clean_text, 'method': strat['name']})
        
        # 3. Evaluate & Save
        for i in range(B_active):
            real = " ".join(active_reports[i].split())
            best = evaluator.select_best(real, candidates[i])
            sid = str(active_ids[i])
            
            # Print (Optional details)
            rg = best['metrics'].get('RadGraph_F1', 0.0)
            cx = best['metrics'].get('CheXpert_F1', 0.0)
            b4 = best['metrics'].get('BLEU-4', 0.0)
            print(f"\n[Study {sid}] Selected: {best['method']} (Score: {best['weighted_score']:.4f})")
            print(f"   Metrics: RadGraph={rg:.4f}, CheXpert={cx:.4f}, BLEU-4={b4:.4f}")
            print("-" * 50)
            
            # Write Clean Output
            f_out.write(json.dumps({
                "study_id": sid, "real_report": real, 
                "generated_report": best['text'], "selected_method": best['method']
            }, ensure_ascii=False) + '\n')
            f_out.flush()
            
    f_out.close()
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gate_threshold', type=float, default=0.15)
    parser.add_argument('--visual_threshold', type=float, default=0.85)
    # Add Resume Argument   断点续跑    report/generated_reports_final_20260101_1315.jsonl
    parser.add_argument('--resume_file', type=str, default=None, help="Path to existing jsonl file to resume from")    # 需要断点续跑，在命令行里加这个
    
    args = parser.parse_args()
    
    CONFIG['batch_size'] = args.batch_size
    CONFIG['visual_gate_threshold'] = args.gate_threshold
    CONFIG['visual_high_conf_threshold'] = args.visual_threshold
    CONFIG['resume_file'] = args.resume_file
    
    kg_engine.initialize(CONFIG['radgraph_path'], CONFIG['train_csv'], CONFIG['old_visual_dir'], CONFIG['kb_path'], True, True)
    csv_dict = load_csv(CONFIG['csv_file'])
    model, conn, tok, clf = load_models(CONFIG)
    generate(model, conn, tok, clf, get_dataloader(CONFIG), csv_dict, CONFIG)