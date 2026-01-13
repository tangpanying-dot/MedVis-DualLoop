"""
Stage 1: Visual-Language Alignment Training
Target: Clean Findings (High Density, No Noise)
Goal: Maximizing CheXbert F1 & RadGraph via strong visual grounding.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'visual'))
from visual.label_conditioned_connector import LabelConditionedConnector, ConnectorConfig
from visual.multiview_dataset_with_labels import create_dataloader
from visual.chexpert_utils import load_chexpert_labels, get_label_weights_tensor

# ==================== CONFIGURATION ====================

CONFIG = {
    # Models
    'llm_name': '/home/tpy/TPY/llm/gemma-2b-it',
    'connector_config': ConnectorConfig(),
    
    # Data Paths
    'feature_dir': 'visual/visual_features/rad_dino',
    'csv_file': 'data/processed_dataset_train.csv',
    'chexpert_csv': 'data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv',
    
    # Training Hyperparams
    'batch_size': 4,              # Findings are short, can increase batch
    'gradient_accumulation': 8,   # Effective batch = 32
    'num_epochs': 8,              # Increased slightly for better convergence
    'learning_rate': 1e-4,        # Standard for connector alignment
    'weight_decay': 0.05,
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    
    # --- METRIC BOOSTING SETTINGS ---
    'lambda_gen': 1.0,      # Generation Loss (Learn to speak)
    'lambda_abn': 0.5,      # Classification Loss (Learn to recognize diseases -> High CheXbert F1)
    'lambda_loc': 0.2,      # Localization Loss (Learn where things are)
    
    # Generation Config (for validation)
    'max_new_tokens': 120,
    'num_beams': 3,
    
    'output_dir': 'checkpoints/stage1_findings_aligned',
    'save_every': 1,
    'dtype': torch.bfloat16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Optimized Prompt for 'Findings' (No structure, just facts)
PROMPT_TEMPLATE = """You are an expert radiologist. Describe the pathological findings in the chest X-ray image concisely.
Focus on: lungs, pleura, heart, mediastinum, and support devices.
Findings:"""

# ==================== LOSS FUNCTION ====================

class MultiTaskLoss(nn.Module):
    def __init__(self, hidden_dim=2048, num_labels=14, num_locations=6, labels_dict=None, device='cuda'):
        super().__init__()
        # Heads attached to the visual embeddings
        self.abnormality_head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, num_labels)
        )
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_locations)
        )
        
        # Class weighting for imbalance
        if labels_dict is not None:
            print("[Loss] Calculating class weights...")
            self.label_weights = get_label_weights_tensor(labels_dict, device=device)
        else:
            self.label_weights = torch.ones(num_labels, device=device)

    def forward(self, hidden_states, chexpert_labels):
        # hidden_states: (B, Seq, Dim) - Output from Connector
        # Global pooling for classification
        global_repr = hidden_states.mean(dim=1) 
        
        # 1. Abnormality Classification (Crucial for CheXbert F1)
        abn_logits = self.abnormality_head(global_repr)
        weights = self.label_weights.to(abn_logits.device)
        abn_loss = F.binary_cross_entropy_with_logits(abn_logits, chexpert_labels, weight=weights.unsqueeze(0))
        
        # 2. Localization (Auxiliary)
        loc_logits = self.location_head(global_repr)
        # Heuristic target: if disease exists, assume it might be in common locations
        loc_targets = (chexpert_labels.max(dim=1)[0].unsqueeze(1).repeat(1, 6) * 0.3)
        loc_loss = F.binary_cross_entropy_with_logits(loc_logits, loc_targets)
        
        return abn_loss, loc_loss

# ==================== HELPERS ====================

def load_model_and_tokenizer(config):
    print(f"\n[Model] Loading LLM: {config['llm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['llm_name'], torch_dtype=config['dtype'], device_map=config['device']
    )
    for param in model.parameters(): param.requires_grad = False # Freeze LLM
    return model, tokenizer

def load_connector(config):
    print(f"\n[Connector] Initializing Label-Conditioned Connector")
    connector = LabelConditionedConnector(config['connector_config'])
    return connector.to(config['device'], dtype=torch.float32) # Connector FP32

# ==================== TRAINING LOOP ====================

def compute_loss(model, connector, multitask_loss_fn, batch, tokenizer, config):
    device = config['device']
    visual_features = batch['visual_features'].to(device)
    visual_mask = batch['visual_mask'].to(device)
    chexpert_labels = batch['chexpert_labels'].to(device)
    target_text = batch['reports'] # This contains 'Findings' now
    
    B = visual_features.shape[0]
    
    # 1. Connector Forward
    with torch.amp.autocast('cuda', dtype=config['dtype']):
        visual_tokens = connector(visual_features.float(), chexpert_labels, visual_mask)
    
    # 2. Prepare Text Inputs
    prompts = [PROMPT_TEMPLATE] * B
    
    # Tokenize Prompt & Target separately to create labels
    prompt_tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
    target_tokens = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    
    # 3. LLM Forward
    with torch.amp.autocast('cuda', dtype=config['dtype']):
        prompt_embeds = model.model.embed_tokens(prompt_tokens.input_ids)
        target_embeds = model.model.embed_tokens(target_tokens.input_ids)
        
        # Concat: [Visual, Prompt, Target]
        inputs_embeds = torch.cat([visual_tokens, prompt_embeds, target_embeds], dim=1)
        
        # Labels: -100 for Visual & Prompt, Real IDs for Target
        labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        start_target = visual_tokens.shape[1] + prompt_embeds.shape[1]
        labels[:, start_target:start_target+target_embeds.shape[1]] = target_tokens.input_ids
        
        outputs = model(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)
        gen_loss = outputs.loss
        
        # 4. Multi-Task Loss on Connector Output
        # Use connector output directly for classification, not LLM output
        abn_loss, loc_loss = multitask_loss_fn(visual_tokens, chexpert_labels)
        
    total_loss = (config['lambda_gen'] * gen_loss + 
                  config['lambda_abn'] * abn_loss + 
                  config['lambda_loc'] * loc_loss)
    
    return total_loss, gen_loss.item(), abn_loss.item(), loc_loss.item()

def train_one_epoch(model, connector, multitask_loss_fn, train_loader, optimizer, scheduler, config, tokenizer, epoch):
    connector.train()
    model.eval()
    stats = {'loss':0, 'gen':0, 'abn':0, 'loc':0}
    steps = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        loss, gen, abn, loc = compute_loss(model, connector, multitask_loss_fn, batch, tokenizer, config)
        loss = loss / config['gradient_accumulation']
        loss.backward()
        
        stats['loss'] += loss.item() * config['gradient_accumulation']
        stats['gen'] += gen
        stats['abn'] += abn
        stats['loc'] += loc
        steps += 1
        
        if (i + 1) % config['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(connector.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if i % 10 == 0:
            pbar.set_postfix({k: v/steps for k,v in stats.items()})
            
    return {k: v/steps for k,v in stats.items()}

@torch.no_grad()
def validate(model, connector, multitask_loss_fn, val_loader, config, tokenizer):
    connector.eval()
    stats = {'loss':0, 'gen':0, 'abn':0, 'loc':0}
    steps = 0
    for batch in tqdm(val_loader, desc="Validating"):
        loss, gen, abn, loc = compute_loss(model, connector, multitask_loss_fn, batch, tokenizer, config)
        stats['loss'] += loss.item()
        stats['gen'] += gen
        stats['abn'] += abn
        stats['loc'] += loc
        steps += 1
    return {k: v/steps for k,v in stats.items()}

# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("Stage 1 Training: Findings Alignment + Disease Classification")
    print("Goal: Maximize Visual-Semantic density for downstream CheXbert F1")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. Labels
    chexpert_labels = load_chexpert_labels(CONFIG['chexpert_csv'], verbose=True)
    
    # 2. DataLoaders (Target = FINDINGS)
    print(f"\n[Data] Loading Cleaned Findings Data (from '{CONFIG['csv_file']}')...")
    train_loader = create_dataloader(
        CONFIG['feature_dir'], CONFIG['csv_file'], CONFIG['chexpert_csv'],
        target_col='findings',  # <--- CRITICAL: Using Clean Findings
        batch_size=CONFIG['batch_size'], shuffle=True, split='train'
    )
    val_loader = create_dataloader(
        CONFIG['feature_dir'], CONFIG['csv_file'], CONFIG['chexpert_csv'],
        target_col='findings', 
        batch_size=CONFIG['batch_size'], shuffle=False, split='val'
    )
    
    # 3. Model
    model, tokenizer = load_model_and_tokenizer(CONFIG)
    connector = load_connector(CONFIG)
    multitask_loss = MultiTaskLoss(labels_dict=chexpert_labels, device=CONFIG['device']).to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(
        list(connector.parameters()) + list(multitask_loss.parameters()), 
        lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_loss = float('inf')
    
    # 4. Loop
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        train_stats = train_one_epoch(model, connector, multitask_loss, train_loader, optimizer, scheduler, CONFIG, tokenizer, epoch)
        print(f"Train: Loss={train_stats['loss']:.4f} (Gen={train_stats['gen']:.4f}, Abn={train_stats['abn']:.4f})")
        
        val_stats = validate(model, connector, multitask_loss, val_loader, CONFIG, tokenizer)
        print(f"Val:   Loss={val_stats['loss']:.4f} (Gen={val_stats['gen']:.4f}, Abn={val_stats['abn']:.4f})")
        
        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            save_path = os.path.join(CONFIG['output_dir'], 'best_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'connector_state_dict': connector.state_dict(),
                'multitask_loss_state_dict': multitask_loss.state_dict(),
                'val_loss': best_loss,
                'config': CONFIG
            }, save_path)
            print(f"✅ Saved Best Model: {save_path}")

if __name__ == "__main__":
    main()