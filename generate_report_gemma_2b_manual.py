"""
Generate Reports: Gemma-2B Manual Edition (For Ablation)
Target Model: Gemma-2B (2048 dim)
Usage:
  python generate_report_gemma_2b_manual.py --kg1 off --kg2 off  (Base)
  python generate_report_gemma_2b_manual.py --kg1 on --kg2 on    (All)
"""

import os
import sys
import torch
import torch.nn as nn
import json
import pandas as pd
import argparse
import re
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'visual'))
from visual.label_conditioned_connector import LabelConditionedConnector, ConnectorConfig
from visual.multiview_dataset_with_labels import MultiViewFeatureDataset, collate_fn_with_padding
from kg_bridge import kg_engine

CONFIG = {
    'llm_name': '/home/tpy/TPY/llm/gemma-2b-it',
    'checkpoint_path': 'checkpoints/stage2/best_checkpoint_metric.pt', # ✅ 确认使用 2B 权重
    'classifier_path': 'checkpoints/classifier_auc/final_best_model.pt',
    
    'feature_dir': 'visual/visual_features/rad_dino',
    'csv_file': 'data/processed_dataset_test.csv',
    'chexpert_csv': 'data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv',
    'history_dir': 'retrieval/history_context/texts',
    'old_visual_dir': 'visual/visual_features/multi_scale',
    'radgraph_path': 'data/graph/radgraph/MIMIC-CXR_graphs.json',
    'train_csv': 'data/processed_dataset_train.csv', 
    'kb_path': 'data/disease_knowledge_base.json',
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': torch.bfloat16,
    'num_workers': 8,
    'use_flash_attn': True,
    
    'max_new_tokens': 200,
    'min_new_tokens': 5,
    'num_beams': 1, # 2B 常用 Greedy
    'repetition_penalty': 1.05,
    
    'visual_gate_threshold': 0.15,
    'visual_high_conf_threshold': 0.85,
}

# (CHEXPERT_CLASS_MAP 和 VisualClassifier 代码与上面相同，省略以节省空间，脚本中已包含)
CHEXPERT_CLASS_MAP = {0: 'Enlarged Cardiomediastinum', 1: 'Cardiomegaly', 2: 'Lung Opacity', 3: 'Lung Lesion', 4: 'Edema', 5: 'Consolidation', 6: 'Pneumonia', 7: 'Atelectasis', 8: 'Pneumothorax', 9: 'Pleural Effusion', 10: 'Pleural Other', 11: 'Fracture', 12: 'Support Devices', 13: 'No Finding'}
NAME_TO_INDEX = {v: k for k, v in CHEXPERT_CLASS_MAP.items()}

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

PROMPT_TEMPLATE = """<start_of_turn>user
You are an expert radiologist. Write a professional chest X-ray report.

[Context]
{kg_context}

[Patient History]
{history}

Write the findings section for this case.<end_of_turn>
<start_of_turn>model
FINDINGS:"""

def load_models(config):
    print(f"🚀 Loading Gemma-2B from: {config['llm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config['llm_name'], torch_dtype=config['dtype'], device_map=config['device'],
        attn_implementation="flash_attention_2"
    )
    llm_dim = 2048 # ✅ 2B
    
    ckpt = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=False)
    if 'lora_state_dict' in ckpt or 'model_state_dict' in ckpt:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=64, lora_alpha=128, lora_dropout=0.05,
                                 target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        model = get_peft_model(base_model, peft_config)
        state_dict = ckpt.get('lora_state_dict', ckpt.get('model_state_dict', None))
        if state_dict: model.load_state_dict(state_dict, strict=False)
    else:
        model = base_model
    model.eval()

    conn_config = ConnectorConfig()
    conn_config.output_dim = llm_dim
    conn_config.llm_dim = llm_dim
    conn = LabelConditionedConnector(conn_config)
    conn.load_state_dict(ckpt.get('connector_state_dict', ckpt), strict=True)
    conn = conn.to(config['device'], dtype=config['dtype']).eval()

    clf = VisualClassifier().to(config['device'])
    if os.path.exists(config['classifier_path']):
        c_ckpt = torch.load(config['classifier_path'], map_location='cpu', weights_only=False)
        clf.load_state_dict(c_ckpt.get('model_state_dict', c_ckpt))
        clf.eval()
    return model, conn, tokenizer, clf

def map_finding_to_chexpert(finding_name):
    finding_lower = finding_name.lower()
    for name, idx in NAME_TO_INDEX.items():
        if name.lower() in finding_lower: return idx
    return None

def apply_gating_and_injection(kg2_raw, soft_labels_probs, config):
    if not kg2_raw: return "", ""
    findings = kg2_raw.split('\n') if '\n' in kg2_raw else kg2_raw.split(',')
    visual_high_conf_findings = []
    filtered_kg = []

    for line in findings:
        line = line.strip()
        if not line: continue
        chexpert_idx = map_finding_to_chexpert(line)
        keep = True
        if chexpert_idx is not None:
            if soft_labels_probs[chexpert_idx].item() < config['visual_gate_threshold']:
                keep = False 
        if keep: filtered_kg.append(line)

    for idx, name in CHEXPERT_CLASS_MAP.items():
        if soft_labels_probs[idx].item() > config['visual_high_conf_threshold']:
            if not any(name.lower() in k.lower() for k in filtered_kg):
                if name != 'No Finding': visual_high_conf_findings.append(name)
    
    return ", ".join(filtered_kg), ", ".join(visual_high_conf_findings)

@torch.no_grad()
def generate(model, connector, tokenizer, classifier, dataloader, csv_dict, config):
    f_out = open(config['output_file'], 'w', encoding='utf-8')
    print(f"📝 Generating to: {config['output_file']}")
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]

    for batch in tqdm(dataloader, desc="2B Manual"):
        vis_feats = batch['visual_features'].to(config['device']).to(config['dtype'])
        vis_mask = batch['visual_mask'].to(config['device'])
        
        logits = classifier(vis_feats.float(), vis_mask)
        probs = torch.sigmoid(logits)
        visual_tokens = connector(vis_feats, probs.to(config['dtype']), vis_mask)
        
        prompts = []
        for i, study_id in enumerate(batch['study_ids']):
            hist = batch['history_texts'][i]
            clean_hist = hist[:1500] if hist else "No prior history."
            
            kg1_prompt, kg2_raw = "", ""
            if config['enable_kg1'] or config['enable_kg2']:
                row = csv_dict.get(str(study_id), {})
                try:
                    k1, k2 = kg_engine.get_prompts(study_id, row.get('history', ''), os.path.join(config['old_visual_dir'], f"{study_id}.npy"))
                    if config['enable_kg1']: kg1_prompt = k1
                    if config['enable_kg2']: kg2_raw = k2
                except: pass
            
            kg_str, vis_inj = apply_gating_and_injection(kg2_raw, probs[i], config)
            context = []
            if kg_str: context.append(f"Similar Cases: {kg_str}")
            if vis_inj: context.append(f"AI Detected: {vis_inj}")
            if config['enable_kg1'] and kg1_prompt: context.append(f"Reference: {kg1_prompt[:200]}")
            
            kg_context = "\n".join(context) if context else "No additional context."
            prompts.append(PROMPT_TEMPLATE.format(kg_context=kg_context, history=clean_hist))
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(config['device'])
        emb_layer = model.get_input_embeddings()
        prompt_emb = emb_layer(inputs.input_ids)
        inputs_embeds = torch.cat([visual_tokens, prompt_emb], dim=1)
        vis_attn = torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=config['device'])
        attn_mask = torch.cat([vis_attn, inputs.attention_mask.long()], dim=1)
        
        outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn_mask, max_new_tokens=config['max_new_tokens'],
                                 num_beams=config['num_beams'], repetition_penalty=config['repetition_penalty'],
                                 eos_token_id=terminators, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, sid in enumerate(batch['study_ids']):
            clean_gen = re.sub(r'[^\x00-\x7F]+', ' ', decoded[i]).replace("<end_of_turn>", "").strip()
            if "FINDINGS:" in clean_gen.upper(): clean_gen = clean_gen[clean_gen.upper().find("FINDINGS:"):].strip()
            f_out.write(json.dumps({"study_id": str(sid), "real_report": " ".join(batch['reports'][i].split()), 
                                    "generated_report": "FINDINGS: " + " ".join(clean_gen.split())}, ensure_ascii=False) + '\n')
            
    f_out.close()
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg1', type=str, default='off')
    parser.add_argument('--kg2', type=str, default='off')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    CONFIG['enable_kg1'] = (args.kg1 == 'on')
    CONFIG['enable_kg2'] = (args.kg2 == 'on')
    CONFIG['batch_size'] = args.batch_size
    
    modes = ["Visual", "History"]
    if CONFIG['enable_kg1']: modes.append("KG1")
    if CONFIG['enable_kg2']: modes.append("KG2")
    CONFIG['output_file'] = f'report/gen_2b_manual_{"_".join(modes)}_{datetime.now().strftime("%Y%m%d_%H%M")}.jsonl'
    
    if CONFIG['enable_kg1'] or CONFIG['enable_kg2']:
        kg_engine.initialize(CONFIG['radgraph_path'], CONFIG['train_csv'], CONFIG['old_visual_dir'], CONFIG['kb_path'], CONFIG['enable_kg1'], CONFIG['enable_kg2'])
    
    df = pd.read_csv(CONFIG['csv_file'])
    df['study_id'] = df['study_id'].astype(str)
    csv_dict = df.set_index('study_id').to_dict('index')
    
    model, conn, tokenizer, classifier = load_models(CONFIG)
    loader = DataLoader(
        MultiViewFeatureDataset(CONFIG['feature_dir'], CONFIG['csv_file'], split='test', 
                                chexpert_csv=CONFIG['chexpert_csv'], history_dir=CONFIG['history_dir'], target_col='report'),
        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=collate_fn_with_padding
    )
    generate(model, conn, tokenizer, classifier, loader, csv_dict, CONFIG)