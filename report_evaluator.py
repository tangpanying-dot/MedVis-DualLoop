# report_evaluator.py
"""
Unified Report Evaluator Module
Designed to be imported by generation scripts.
Handles:
- Text cleaning
- Metric calculation (RadGraph, CheXbert, BLEU-4)
- Dynamic Weighted Selection (Best-of-N)
"""

import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple

# Enable HF mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from sklearn.metrics import precision_recall_fscore_support
except ImportError:
    precision_recall_fscore_support = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# ==================== Text Cleaning Utils ====================

SECTION_RE = re.compile(r'\b(findings|impression|conclusion|history|comparison)\b\s*:\s*', flags=re.IGNORECASE)

def clean_text_basic(text: str) -> str:
    """Basic cleaning for display and simple storage."""
    if not text: return ""
    text = str(text).strip()
    filter_kws = ["CLINICAL CONTEXT", "[CHECK]", "[Context]", "[History]", "You are a radiologist", "Task:", "EXPECTED:"]
    lines = [l for l in text.split('\n') if not any(k in l for k in filter_kws)]
    text = " ".join(lines)
    if "Report:" in text: text = text.split("Report:")[-1]
    if text.upper().startswith("FINDINGS"):
        idx = text.find(":")
        if idx != -1 and idx < 15: text = text[idx+1:]
    return " ".join(text.split())

def clean_for_eval(text: str) -> str:
    """Rigorous cleaning for metric calculation."""
    if not text: return ""
    text = str(text).replace("<unk>", "unk").lower()
    text = SECTION_RE.sub(" ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r'([.,;?!:])', r' \1 ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'[{}]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ==================== Evaluator Classes ====================

class NLGEvaluator:
    def __init__(self):
        self.available = False
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            # Only load BLEU-4 for efficiency as requested
            self.scorers = [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])]
            self.available = True
        except ImportError:
            print("⚠️ [Eval] pycocoevalcap not found. BLEU disabled.")

    def compute(self, ref: str, hyp: str) -> Dict[str, float]:
        if not self.available: return {}
        gts = {"1": [clean_for_eval(ref)]}
        res = {"1": [clean_for_eval(hyp)]}
        scores = {}
        for scorer, method in self.scorers:
            try:
                sc, _ = scorer.compute_score(gts, res)
                if isinstance(method, list):
                    for m, s in zip(method, sc): scores[m] = s
                else:
                    scores[method] = sc
            except: pass
        return scores

class RadGraphEvaluator:
    def __init__(self, device='cuda'):
        self.available = False
        try:
            from radgraph import F1RadGraph
            self.labeler = F1RadGraph(reward_level="partial") 
            self.available = True
            print("✅ [Eval] RadGraph loaded.")
        except Exception as e:
            # Suppress massive stack trace, just warn
            print(f"⚠️ [Eval] RadGraph not available (Reason: {e}).")

    def compute(self, ref: str, hyp: str) -> float:
        if not self.available: return 0.0
        try:
            r, h = clean_for_eval(ref), clean_for_eval(hyp)
            if not r or not h: return 0.0
            score = self.labeler(hyps=[h], refs=[r])
            if isinstance(score, list): score = score[0]
            if isinstance(score, dict): return float(score.get("radgraph-f1", score.get("f1", 0.0)))
            return float(score)
        except: return 0.0

class CheXbertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(768, 4) for _ in range(13)] + [nn.Linear(768, 2)])
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return [head(out) for head in self.linear_heads]

class CheXbertEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.available = False
        try:
            from transformers import BertTokenizer
            if hf_hub_download:
                path = hf_hub_download(repo_id="StanfordAIMI/RRG_scorers", filename="chexbert.pth")
                ckpt = torch.load(path, map_location='cpu')
                sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                sd = {k.replace("module.", ""): v for k,v in sd.items()}
                self.model = CheXbertClassifier()
                self.model.load_state_dict(sd, strict=True)
                self.model.to(device).eval()
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.available = True
                print("✅ [Eval] CheXbert loaded.")
            else:
                print("⚠️ [Eval] hf_hub_download missing.")
        except Exception as e:
            print(f"⚠️ [Eval] CheXbert failed: {e}")

    def compute(self, ref: str, hyp: str) -> float:
        if not self.available: return 0.0
        try:
            r, h = clean_for_eval(ref), clean_for_eval(hyp)
            if not r or not h: return 0.0
            
            inputs = self.tokenizer([r, h], return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs)
            
            preds = []
            for i in range(2):
                row = []
                for j in range(14):
                    p = logits[j][i].argmax().item()
                    row.append(p - 2 if j < 13 else p)
                preds.append(row)
            
            y_true, y_pred = np.array(preds[0]), np.array(preds[1])
            p_mask, h_mask = (y_true == 1), (y_pred == 1)
            
            if precision_recall_fscore_support:
                _, _, f1, _ = precision_recall_fscore_support(p_mask, h_mask, average='binary', zero_division=0)
                return f1
            return 0.0
        except: return 0.0

# ==================== Unified Interface ====================

class UnifiedEvaluator:
    """
    Central evaluator that handles Best-of-N logic and Dynamic Weights.
    """
    def __init__(self, device='cuda', weights=None):
        print("🚀 Initializing Unified Report Evaluator...")
        self.nlg = NLGEvaluator()
        self.radgraph = RadGraphEvaluator(device)
        self.chexbert = CheXbertEvaluator(device)
        
        # Default Weights (can be overridden in select_best)
        self.default_weights = weights if weights else {
            'RadGraph_F1': 0.4,
            'CheXpert_F1': 0.4,
            'BLEU-4': 0.2
        }
        self.active_weights = self._normalize_weights(self.default_weights)
        print(f"⚖️  Default Active Weights: {self.active_weights}")

    def _normalize_weights(self, weights_dict):
        """
        Normalize weights based on availability of the modules.
        If a module is not available, its weight is redistributed.
        """
        active = {}
        total = 0.0
        
        # 1. Filter checks
        # If user passed a weight for RadGraph but RadGraph isn't installed, ignore it.
        w_rg = weights_dict.get('RadGraph_F1', 0.0)
        w_cx = weights_dict.get('CheXpert_F1', 0.0)
        w_b4 = weights_dict.get('BLEU-4', 0.0)
        w_bert = weights_dict.get('BERTScore', 0.0) # Placeholder

        if self.radgraph.available: total += w_rg
        if self.chexbert.available: total += w_cx
        if self.nlg.available:      total += w_b4
        # We don't have BERTScore class here, so we ignore w_bert for now to avoid divide by zero if only BERT is requested.
        
        if total == 0: 
            # Fallback if nothing selected or nothing available
            return {}
        
        # 2. Normalize
        if self.radgraph.available and w_rg > 0:
            active['RadGraph_F1'] = w_rg / total
        if self.chexbert.available and w_cx > 0:
            active['CheXpert_F1'] = w_cx / total
        if self.nlg.available and w_b4 > 0:
            active['BLEU-4'] = w_b4 / total
            
        return active

    def score_candidate(self, ref: str, hyp: str, weights: Dict = None) -> Dict:
        """Computes metrics and single weighted score for a candidate."""
        metrics = {}
        
        # 1. Compute Raw Metrics
        if self.radgraph.available:
            metrics['RadGraph_F1'] = self.radgraph.compute(ref, hyp)
        else:
            metrics['RadGraph_F1'] = 0.0

        if self.chexbert.available:
            metrics['CheXpert_F1'] = self.chexbert.compute(ref, hyp)
        else:
            metrics['CheXpert_F1'] = 0.0

        if self.nlg.available:
            nlg_scores = self.nlg.compute(ref, hyp)
            metrics.update(nlg_scores)
        
        # 2. Determine Weights to Use
        # If dynamic weights are passed, use them (normalized), else use defaults
        use_weights = self._normalize_weights(weights) if weights else self.active_weights

        # 3. Compute Weighted Score
        score = 0.0
        for k, w in use_weights.items():
            score += w * metrics.get(k, 0.0)
            
        return {'metrics': metrics, 'weighted_score': score}

    def select_best(self, ref: str, candidates: List[Dict], weights: Dict = None) -> Dict:
        """
        Input: List of dicts {'text': '...', 'method': '...'}
        Output: The best candidate dict with added 'metrics' and 'weighted_score'
        
        weights argument allows per-call override of metric importance.
        """
        rated = []
        for cand in candidates:
            res = self.score_candidate(ref, cand['text'], weights=weights)
            cand.update(res) # Add metrics/score to candidate dict
            rated.append(cand)
        
        # Sort descending by score
        rated.sort(key=lambda x: x['weighted_score'], reverse=True)
        return rated[0]