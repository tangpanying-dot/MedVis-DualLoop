"""
Stage 2: Report Generation (v22.3 Perfect Math Edition)
Target: Best Generation Quality (ROUGE/Clinical F1)
Input:  Visual + History Context
Hardware: RTX 4090 (24GB)

v22.3 最终数学修正 (Based on Final Expert Review):
1. ✅ [Math Fix] 梯度累积余数 Scaling 修正 (整个 Window 统一分母，保证梯度无偏)
2. ✅ [Metric] 评估 Decode 后增加 .strip() (减少格式噪声)
3. ✅ [Inherited] 继承 v22.2 所有核心修复 (ROUGE DP Fix, Dynamic Padding, etc.)
"""

import os
import sys
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_cosine_schedule_with_warmup
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType
)

# 引入你的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'visual'))
from visual.label_conditioned_connector import LabelConditionedConnector, ConnectorConfig
from visual.multiview_dataset_with_labels import create_dataloader

# ==================== 配置 ====================

CONFIG = {
    'llm_name': '/home/tpy/TPY/llm/gemma-2b-it',
    'connector_config': ConnectorConfig(),
    # 请确保这里指向正确的 Stage 1 权重
    'stage1_checkpoint': 'checkpoints/stage1/best_checkpoint.pt',
    
    'feature_dir': 'visual/visual_features/rad_dino',
    'csv_file': 'data/processed_dataset_train.csv',
    'chexpert_csv': 'data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv',
    'history_dir': 'retrieval/history_context/texts', 
    
    # ========== 训练参数 ==========
    'batch_size': 2,              
    'gradient_accumulation': 16,   
    'num_epochs': 15,             
    'num_workers': 8,
    
    'lr_connector': 5e-5,
    'lr_llm': 2e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.05,
    'max_grad_norm': 1.0,
    
    'prompt_max_len': 512,
    'target_max_len': 512,        
    'max_history_len': 512, 
    
    # 验证参数
    'eval_samples': 200,          
    'max_new_tokens': 120,
    
    'dtype': torch.bfloat16,      
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'output_dir': 'checkpoints/stage2',
    'save_every': 1,
}

PROMPT_TEMPLATE = """You are an expert radiologist.
Patient History:
{history}

Task:
Write a professional chest X-ray report (Findings and Impression) based on the image and history above.
Report:
"""

# ==================== ROUGE-L (DP Bug Fixed) ====================

def calculate_rouge_l_score(references, hypotheses):
    """
    简易 ROUGE-L F1 (Space Optimized DP, Bug Fixed)
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        # 简单分词
        ref = ref.lower().split()
        hyp = hyp.lower().split()
        
        m, n = len(ref), len(hyp)
        if m == 0 or n == 0:
            scores.append(0.0)
            continue
            
        # 内存优化：只存两行
        prev = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr = [0] * (n + 1) # [Fix] 每行重置，确保 LCS 计算正确
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = list(curr)
            
        lcs = prev[n]
        p = lcs / n if n > 0 else 0
        r = lcs / m if m > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        scores.append(f1)
        
    return sum(scores) / len(scores)

# ==================== 模型加载 ====================

def load_models(config):
    print(f"\n[Model] Loading LLM (BF16): {config['llm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # [Config] 训练时默认 Right Padding
    tokenizer.padding_side = 'right' 
    
    model = AutoModelForCausalLM.from_pretrained(
        config['llm_name'], 
        torch_dtype=config['dtype'],
        device_map=config['device']
    )
    
    # [Stability] 开启梯度检查点 + 关闭 Cache
    model.gradient_checkpointing_enable() 
    model.config.use_cache = False 
    model.enable_input_require_grads()
    
    print("[Model] Applying High-Rank LoRA (r=64)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    model = get_peft_model(model, peft_config)
    
    print(f"\n[Connector] Loading Stage 1 Weights...")
    connector = LabelConditionedConnector(config['connector_config'])
    if os.path.exists(config['stage1_checkpoint']):
        checkpoint = torch.load(config['stage1_checkpoint'], map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('connector_state_dict', checkpoint)
        connector.load_state_dict(state_dict, strict=False)
    
    connector = connector.to(config['device'], dtype=config['dtype']) 
    for param in connector.parameters():
        param.requires_grad = True
        
    return model, connector, tokenizer

# ==================== 工具函数 ====================

def format_prompts(batch_history, template):
    prompts = []
    for hist in batch_history:
        clean_hist = hist.strip() if hist and len(hist) > 5 else "No prior history available."
        clean_hist = clean_hist[:2000] 
        prompts.append(template.format(history=clean_hist))
    return prompts

# ==================== 训练 Step ====================

def compute_loss(model, connector, batch, tokenizer, config):
    device = config['device']
    
    visual_features = batch['visual_features'].to(device).to(config['dtype'])
    raw_visual_mask = batch['visual_mask'].to(device) 
    chexpert_labels = batch['chexpert_labels'].to(device).to(config['dtype'])
    
    visual_tokens = connector(visual_features, chexpert_labels, raw_visual_mask)
    
    target_text = [t + tokenizer.eos_token for t in batch['reports']]
    history_text = batch['history_texts']
    prompts = format_prompts(history_text, PROMPT_TEMPLATE)
    
    # Tokenize (Right Padding for Training)
    prompt_tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=config['prompt_max_len']).to(device)
    target_tokens = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=config['target_max_len']).to(device)
    
    input_embeds_layer = model.get_input_embeddings()
    prompt_embeds = input_embeds_layer(prompt_tokens.input_ids)
    target_embeds = input_embeds_layer(target_tokens.input_ids)

    inputs_embeds = torch.cat([visual_tokens, prompt_embeds, target_embeds], dim=1)
    
    B, K = visual_tokens.shape[:2]
    visual_attn = torch.ones((B, K), dtype=torch.long, device=device) 
    attention_mask = torch.cat([visual_attn, prompt_tokens.attention_mask.long(), target_tokens.attention_mask.long()], dim=1)
    
    labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
    start_target = visual_tokens.shape[1] + prompt_embeds.shape[1]
    end_target = start_target + target_tokens.input_ids.shape[1]
    
    labels[:, start_target:end_target] = target_tokens.input_ids
    labels[:, start_target:end_target][target_tokens.attention_mask.long() == 0] = -100
    
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

# ==================== 评估 Step (生成) ====================

@torch.inference_mode()
def evaluate_generation(model, connector, dataloader, tokenizer, config):
    """
    [Refined] 动态切换 Left Padding + Clean Decode
    """
    model.eval()
    connector.eval()
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    preds = []
    refs = []
    limit = config['eval_samples']
    count = 0
    
    print("\n[Eval] Generating validation samples (Left Padding)...")
    for batch in tqdm(dataloader, desc="Generating"):
        if count >= limit: break
        
        device = config['device']
        B = len(batch['study_ids'])
        
        visual_features = batch['visual_features'].to(device).to(config['dtype'])
        raw_visual_mask = batch['visual_mask'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device).to(config['dtype'])
        visual_tokens = connector(visual_features, chexpert_labels, raw_visual_mask)
        
        prompts = format_prompts(batch['history_texts'], PROMPT_TEMPLATE)
        # Left Padding Tokenization
        prompt_inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=config['prompt_max_len']).to(device)
        
        input_embeds_layer = model.get_input_embeddings()
        prompt_embeds = input_embeds_layer(prompt_inputs.input_ids)
        inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)
        
        B, K = visual_tokens.shape[:2]
        visual_attn = torch.ones((B, K), dtype=torch.long, device=device)
        attention_mask = torch.cat([visual_attn, prompt_inputs.attention_mask.long()], dim=1)
        
        # [Debug Check]
        if count == 0:
            input_len = inputs_embeds.shape[1]
            print(f"  [Debug] Input Len: {input_len}")

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=config['max_new_tokens'],
            num_beams=2,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True 
        )
        
        # Slice only new tokens
        input_len = inputs_embeds.shape[1]
        generated_ids = outputs[:, input_len:] if outputs.shape[1] > input_len else outputs
        
        # [Fix] 增加 strip()
        decoded_preds = [t.strip() for t in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        
        preds.extend(decoded_preds)
        refs.extend(batch['reports'])
        count += B

    # 恢复 Right Padding
    tokenizer.padding_side = original_padding_side
    
    return calculate_rouge_l_score(refs, preds)

# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("Stage 2 v22.3: Perfect Math Edition (TRAINING SCRIPT)")
    print("Optimization: Consistent Gradient Scaling + Clean Decoding")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. DataLoaders
    train_loader = create_dataloader(
        CONFIG['feature_dir'], CONFIG['csv_file'], CONFIG['chexpert_csv'],
        target_col='report',
        history_dir=CONFIG['history_dir'],
        batch_size=CONFIG['batch_size'], shuffle=True, split='train', num_workers=CONFIG['num_workers'],
        max_history_length=CONFIG['max_history_len']
    )
    val_loader = create_dataloader(
        CONFIG['feature_dir'], CONFIG['csv_file'], CONFIG['chexpert_csv'],
        target_col='report',
        history_dir=CONFIG['history_dir'],
        batch_size=CONFIG['batch_size'], shuffle=False, split='val', num_workers=4,
        max_history_length=CONFIG['max_history_len']
    )
    
    # 2. Models
    model, connector, tokenizer = load_models(CONFIG)
    
    # 3. Optimizer
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": CONFIG['lr_llm']},
        {"params": [p for n, p in connector.named_parameters() if p.requires_grad], "lr": CONFIG['lr_connector']},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=CONFIG['weight_decay'])
    
    # Scheduler Ceil
    accum = CONFIG['gradient_accumulation']
    total_steps = math.ceil(len(train_loader) / accum) * CONFIG['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * CONFIG['warmup_ratio']), num_training_steps=total_steps)
    
    print(f"\n[Train] Starting Training for {CONFIG['num_epochs']} epochs...")
    best_rouge = 0.0
    
    # [Math Fix] 预计算梯度累积的余数逻辑
    total_batches = len(train_loader)
    remainder = total_batches % accum
    last_window_start = total_batches - remainder if remainder > 0 else total_batches
    print(f"[Grad Info] Total Batches: {total_batches}, Accum: {accum}")
    if remainder > 0:
        print(f"[Grad Info] Last Window (remainder={remainder}) starts at batch index: {last_window_start}")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*40} Epoch {epoch}/{CONFIG['num_epochs']} {'='*40}")
        
        # === Training ===
        model.train()
        connector.train()
        model.config.use_cache = False
        
        total_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            loss = compute_loss(model, connector, batch, tokenizer, CONFIG)
            
            # [Math Fix] 动态计算分母，保证整个 window 内梯度一致
            is_in_last_window = (remainder > 0) and (i >= last_window_start)
            denom = remainder if is_in_last_window else accum
            
            loss = loss / denom
            loss.backward()
            
            # 步进条件
            is_step_time = ((i + 1) % accum == 0) or ((i + 1) == total_batches)
            
            if is_step_time:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(connector.parameters(), CONFIG['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * denom
            steps += 1
            pbar.set_postfix({'loss': total_loss / steps, 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
        print(f"Epoch {epoch} Train Loss: {total_loss / steps:.4f}")
        torch.cuda.empty_cache()
        # === Metric Eval ===
        val_rouge = evaluate_generation(model, connector, val_loader, tokenizer, CONFIG)
        print(f"Epoch {epoch} Val ROUGE-L: {val_rouge:.4f}")
        
        if val_rouge > best_rouge:
            best_rouge = val_rouge
            save_path = os.path.join(CONFIG['output_dir'], 'best_checkpoint_metric.pt')
            torch.save({
                'epoch': epoch,
                'connector_state_dict': connector.state_dict(),
                'model_state_dict': model.state_dict(),
                'best_rouge': best_rouge,
                'config': CONFIG
            }, save_path)
            print(f"⭐ New Best Model (ROUGE-L)! Saved to {save_path}")
        last_save_path = os.path.join(CONFIG['output_dir'], 'last_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'connector_state_dict': connector.state_dict(),
            'model_state_dict': model.state_dict(),
            'best_rouge': best_rouge,
            'config': CONFIG
        }, last_save_path)
        print(f"💾 Backup saved: {last_save_path}")

if __name__ == "__main__":
    main()