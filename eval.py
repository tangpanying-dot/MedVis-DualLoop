#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FI GT-only evaluation script
- Uses FI (Findings+Impression) ground-truth protocol
- For NLG metrics (BLEU/METEOR/ROUGE/CIDEr): merge (and optionally dedup) FI, then PTB-style tokenization
- For clinical metrics (RadGraph/CheXpert): merge (and optionally dedup) FI, then light cleaning

Output CSV columns (4 decimals):
eval_timestamp,report_file,BLEU-1,BLEU-2,BLEU-3,BLEU-4,METEOR,ROUGE-L,CIDEr,RadGraph_F1,CheXpert_F1,CheXpert_P,CheXpert_R
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# --- deps ---
try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("⚠️ 未安装 PyTorch 或 transformers。CheXbert 将无法使用。")
    torch = None

try:
    from sklearn.metrics import precision_recall_fscore_support
except ImportError:
    print("⚠️ 未安装 scikit-learn。F1 分数将无法计算。")
    precision_recall_fscore_support = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("⚠️ 未安装 huggingface_hub。无法自动下载权重。")
    hf_hub_download = None

# ==================== Config ====================

CONDITIONS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
]

# Headings used in many datasets
SECTION_RE = re.compile(r'\b(findings|impression|conclusion|history|comparison)\b\s*:\s*', flags=re.IGNORECASE)

# For FI parsing
FINDINGS_RE = re.compile(r'(?is)\bfindings\s*:\s*')
IMPRESSION_RE = re.compile(r'(?is)\bimpression\s*:\s*')

def parse_report_fi(text: str) -> Tuple[str, str]:
    """
    Parse Findings/Impression if present. Otherwise treat whole text as findings.
    Returns (findings, impression).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    low = text.lower()
    has_f = "findings:" in low
    has_i = "impression:" in low

    if not (has_f or has_i):
        # no explicit structure
        return text.strip(), ""

    # Strategy:
    # - if findings exists: take segment from findings to before impression (if any)
    # - if impression exists: take segment from impression to end
    f_txt = ""
    i_txt = ""

    # Impression segment
    mi = IMPRESSION_RE.search(text)
    if mi:
        i_txt = text[mi.end():].strip()

    # Findings segment
    mf = FINDINGS_RE.search(text)
    if mf:
        if mi and mi.start() > mf.end():
            f_txt = text[mf.end():mi.start()].strip()
        else:
            f_txt = text[mf.end():].strip()
    else:
        # only impression exists -> treat prefix as findings? usually not meaningful; leave findings empty
        f_txt = ""

    return f_txt, i_txt

def dedup_findings_impression(findings: str, impression: str) -> str:
    """
    If impression repeats findings (common in some exports), keep only one.
    Uses a conservative normalized comparison.
    """
    f = (findings or "").strip()
    i = (impression or "").strip()
    if not f and not i:
        return ""
    if not f:
        return i
    if not i:
        return f

    def norm(x: str) -> str:
        x = x.lower()
        x = SECTION_RE.sub(" ", x)
        x = re.sub(r'\s+', ' ', x).strip()
        return x

    if norm(f) == norm(i):
        return f
    # Some datasets: impression starts with findings verbatim
    if norm(i).startswith(norm(f)) and len(norm(i)) <= len(norm(f)) + 10:
        return f
    return (f + " " + i).strip()

def merge_fi(text_any: Any, dedup_fi: bool = True) -> str:
    """
    Accepts string/dict/JSON-string-dict.
    Produces FI merged text (dedup applied).
    """
    # dict case
    if isinstance(text_any, dict):
        f = text_any.get("findings", "") or ""
        i = text_any.get("impression", "") or ""
        return dedup_findings_impression(f, i) if dedup_fi else (f + " " + i).strip()

    # string-like
    text = "" if text_any is None else (text_any if isinstance(text_any, str) else str(text_any))

    # JSON-string dict
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            f = obj.get("findings", "") or ""
            i = obj.get("impression", "") or ""
            return dedup_findings_impression(f, i) if dedup_fi else (f + " " + i).strip()
    except Exception:
        pass

    findings, impression = parse_report_fi(text)
    if dedup_fi:
        merged = dedup_findings_impression(findings, impression)
    else:
        merged = (findings + " " + impression).strip()
    return merged if merged else text.strip()

# ==================== Cleaning ====================

def clean_for_nlg_fi(text: str) -> str:
    """
    NLG strong-clean aligned with your eval.py:
    - lower
    - remove headings
    - PTB-style punctuation separation for . , ; ? ! :
    - normalize whitespace
    Additionally:
    - normalize <unk> -> unk (avoid angle-brackets interfering)
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.replace("<unk>", "unk")
    text = text.lower()
    text = SECTION_RE.sub(" ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r'([.,;?!:])', r' \1 ', text)
    # remove latex-ish artifacts (kept from your code)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_clinical_fi(text: str) -> str:
    """
    Clinical light-clean aligned with your eval.py:
    - keep punctuation tight
    - remove headings
    - normalize whitespace
    - normalize <unk>
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.replace("<unk>", "unk")
    text = SECTION_RE.sub(" ", text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}]+', '', text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== Data loading (FI GT only) ====================

def load_reports_fi_only(file_path: str, dedup_fi: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str], List[str]]:
    """
    Load jsonl with fields: study_id, real_report, generated_report.
    Apply FI merge + optional dedup, then dual-stream cleaning.
    """
    gts_nlg: Dict[str, List[str]] = {}
    res_nlg: Dict[str, List[str]] = {}
    ref_clin: List[str] = []
    hyp_clin: List[str] = []
    skipped = 0
    total = 0

    print(f"✓ Loading: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                study_id = str(data.get("study_id"))

                real_raw = merge_fi(data.get("real_report", ""), dedup_fi=dedup_fi)
                gen_raw = merge_fi(data.get("generated_report", ""), dedup_fi=dedup_fi)

                nlg_real = clean_for_nlg_fi(real_raw)
                nlg_gen = clean_for_nlg_fi(gen_raw)

                clin_real = clean_for_clinical_fi(real_raw)
                clin_gen = clean_for_clinical_fi(gen_raw)

                if not nlg_real or not nlg_gen:
                    skipped += 1
                    continue

                gts_nlg[study_id] = [nlg_real]
                res_nlg[study_id] = [nlg_gen]
                ref_clin.append(clin_real)
                hyp_clin.append(clin_gen)
                total += 1
            except Exception:
                skipped += 1
                continue

    print(f"✓ Loaded {total} samples (skipped {skipped})")
    return gts_nlg, res_nlg, ref_clin, hyp_clin

# ==================== NLG Evaluator ====================

class NLGEvaluator:
    def __init__(self):
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            self.scorers = [
                (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE-L"),
                (Cider(), "CIDEr"),
            ]
        except Exception as e:
            print(f"❌ pycocoevalcap not available: {e}")
            self.scorers = []

    def evaluate(self, gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
        results: Dict[str, float] = {
            "BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
            "METEOR": 0.0, "ROUGE-L": 0.0, "CIDEr": 0.0,
        }
        for scorer, method in self.scorers:
            try:
                score, _ = scorer.compute_score(gts, res)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        results[m] = float(s)
                else:
                    results[method] = float(score)
            except Exception:
                # keep zeros
                pass
        return results

# ==================== RadGraph ====================

class RadGraphEvaluator:
    def __init__(self):
        try:
            from radgraph import F1RadGraph
            self.labeler = F1RadGraph(reward_level="partial")
            self.available = True
        except Exception as e:
            print(f"⚠️ RadGraph not available: {e}")
            self.available = False

    def evaluate(self, refs: List[str], hyps: List[str]) -> Dict[str, float]:
        if not self.available:
            return {"RadGraph_F1": 0.0}
        print(f"\n⏳ Running RadGraph (n={len(refs)}) ...")
        try:
            metrics = self.labeler(hyps=hyps, refs=refs)
            f1 = metrics[0] if isinstance(metrics, (list, tuple)) else metrics
            if isinstance(f1, dict):
                f1 = f1.get("radgraph-f1", f1.get("f1", 0.0))
            return {"RadGraph_F1": float(f1)}
        except Exception as e:
            print(f"❌ RadGraph error: {e}")
            return {"RadGraph_F1": 0.0}

# ==================== CheXbert ====================

class CheXbertClassifier(nn.Module):
    """
    Same structure as your eval.py:
    - 13 heads: 4-way (Blank, Neg, Unc, Pos)
    - 14th head (No Finding): 2-way (Absent, Present)
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(768, 4) for _ in range(13)] + [nn.Linear(768, 2)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        return [head(pooled_output) for head in self.linear_heads]

def download_and_load_chexbert(device):
    if hf_hub_download is None or torch is None:
        return None, False

    print("   [CheXbert] downloading/checking StanfordAIMI/RRG_scorers chexbert.pth ...")
    try:
        checkpoint_path = hf_hub_download(repo_id="StanfordAIMI/RRG_scorers", filename="chexbert.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model = CheXbertClassifier()
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        print(f"   [CheXbert] ready: {checkpoint_path}")
        return model, True
    except Exception as e:
        print(f"❌ CheXbert load failed: {e}")
        return None, False

class CheXbertEvaluatorWrapper:
    def __init__(self):
        if torch is None or precision_recall_fscore_support is None:
            self.available = False
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n⏳ Init CheXbert on {self.device} ...")
        self.model, self.available = download_and_load_chexbert(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if self.available else None

    def get_labels(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if not self.available:
            return np.zeros((len(texts), 14), dtype=np.int32)

        all_labels = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits_list = self.model(**inputs)

                batch_labels = []
                for idx in range(len(batch)):
                    sample_labels = []
                    for head_idx in range(14):
                        pred = int(torch.argmax(logits_list[head_idx][idx]).item())
                        if head_idx < 13:
                            # 4-way -> (-2,-1,0,1) with Pos=1
                            sample_labels.append(pred - 2)
                        else:
                            # No Finding: 2-way -> (0/1)
                            sample_labels.append(pred)
                    batch_labels.append(sample_labels)
                all_labels.append(np.asarray(batch_labels, dtype=np.int32))

        return np.vstack(all_labels) if all_labels else np.zeros((0, 14), dtype=np.int32)

    def evaluate(self, refs: List[str], hyps: List[str]) -> Dict[str, float]:
        if not self.available:
            return {"CheXpert_F1": 0.0, "CheXpert_P": 0.0, "CheXpert_R": 0.0}

        print(f"\n⏳ Running CheXbert (n={len(refs)}) ...")
        ref_labels = self.get_labels(refs)
        hyp_labels = self.get_labels(hyps)

        # micro-F1 on positive class (==1)
        ref_pos = (ref_labels == 1)
        hyp_pos = (hyp_labels == 1)

        p, r, f1, _ = precision_recall_fscore_support(ref_pos, hyp_pos, average="micro", zero_division=0)
        return {"CheXpert_F1": float(f1), "CheXpert_P": float(p), "CheXpert_R": float(r)}

# ==================== Main ====================

CSV_COLUMNS = [
    "eval_timestamp", "report_file",
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr",
    "RadGraph_F1",
    "CheXpert_F1", "CheXpert_P", "CheXpert_R",
]

def evaluate_fi_gt(report_file: str, output_dir: str, dedup_fi: bool = True) -> Dict[str, Any]:
    gts, res, ref_texts, hyp_texts = load_reports_fi_only(report_file, dedup_fi=dedup_fi)
    if not gts:
        raise RuntimeError("No valid samples loaded. Check your jsonl format/fields.")

    print("\n--- [1/3] NLG metrics ---")
    nlg_results = NLGEvaluator().evaluate(gts, res)

    print("\n--- [2/3] RadGraph ---")
    rg_results = RadGraphEvaluator().evaluate(ref_texts, hyp_texts)

    print("\n--- [3/3] CheXpert (CheXbert) ---")
    ce_results = CheXbertEvaluatorWrapper().evaluate(ref_texts, hyp_texts)

    all_results: Dict[str, Any] = {
        **nlg_results,
        **rg_results,
        **ce_results,
        "eval_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_file": Path(report_file).name,
    }

    # Print (4 decimals)
    print("\n" + "=" * 70)
    print("Results (FI GT protocol)")
    print("=" * 70)
    for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "RadGraph_F1", "CheXpert_F1", "CheXpert_P", "CheXpert_R"]:
        print(f"{k:<12}: {float(all_results.get(k, 0.0)):.4f}")

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gemma_2b_eval_summary.csv")

    # ensure all columns exist
    row = {c: all_results.get(c, 0.0) for c in CSV_COLUMNS}
    row["eval_timestamp"] = all_results["eval_timestamp"]
    row["report_file"] = all_results["report_file"]

    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    df_all = df_all.reindex(columns=CSV_COLUMNS)
    df_all.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n✓ Saved CSV: {csv_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="FI GT-only evaluation (BLEU/METEOR/ROUGE/CIDEr + RadGraph + CheXpert)")
    parser.add_argument("--report", type=str, required=True, help="jsonl report file path")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="output dir")
    parser.add_argument("--no-dedup-fi", action="store_true", help="do NOT deduplicate findings/impression when identical")
    args = parser.parse_args()

    evaluate_fi_gt(args.report, args.output_dir, dedup_fi=not args.no_dedup_fi)

if __name__ == "__main__":
    main()
