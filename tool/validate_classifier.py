import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from chexpert_utils import load_chexpert_labels, LABEL_NAMES  # 你已有的工具文件

MAX_TOKENS = 1370
NUM_CLASSES = 14

# ===== Dataset (val split study_id + npy feats + labels) =====
class SimpleVisualDataset(Dataset):
    def __init__(self, feature_dir: str, split_csv: str, chexpert_csv: str):
        self.feature_dir = feature_dir
        df = pd.read_csv(split_csv, usecols=["study_id"])
        self.study_ids = df["study_id"].astype(str).tolist()
        self.labels_dict = load_chexpert_labels(chexpert_csv, normalize=True, verbose=True)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        path = os.path.join(self.feature_dir, f"{sid}.npy")
        if not os.path.exists(path):
            return None

        feats = np.load(path, mmap_mode="r")  # (T,768), float16 typically
        if feats.shape[0] > MAX_TOKENS:
            feats = feats[:MAX_TOKENS, :]

        feats = torch.from_numpy(np.array(feats)).to(dtype=torch.bfloat16)
        labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if sid in self.labels_dict:
            labels = torch.from_numpy(self.labels_dict[sid])  # float32, (14,)
        return {"study_id": sid, "visual_features": feats, "length": feats.shape[0], "labels": labels}

def collate_minimal(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    max_len = max(b["length"] for b in batch)
    feat_dim = batch[0]["visual_features"].shape[-1]
    B = len(batch)

    feats = torch.zeros(B, max_len, feat_dim, dtype=batch[0]["visual_features"].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.stack([b["labels"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        l = b["length"]
        feats[i, :l] = b["visual_features"]
        mask[i, :l] = True

    return {"features": feats, "mask": mask, "labels": labels}

# ===== Model (与你 train_classifier.py 一致) =====
class VisualClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=NUM_CLASSES):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, features, mask=None):
        attn_scores = self.attention(features)  # (B,T,1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B,T,1)
        context = torch.sum(features * attn_weights, dim=1)  # (B,D)
        return self.head(context)  # (B,C)

def best_f1_threshold(y_true_bin, y_prob, num_steps=201):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, num_steps):
        y_pred = (y_prob >= t).astype(np.int32)
        f1 = f1_score(y_true_bin, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--chexpert_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = SimpleVisualDataset(args.feature_dir, args.split_csv, args.chexpert_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, collate_fn=collate_minimal)

    model = VisualClassifier().to(args.device).float()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    all_probs = []
    all_labels = []
    for batch in tqdm(dl, desc="Infer"):
        if not batch:
            continue
        feats = batch["features"].to(args.device, non_blocking=True)
        mask = batch["mask"].to(args.device, non_blocking=True)
        labels = batch["labels"].cpu().numpy()

        # 与训练一致：bf16 autocast
        use_amp = (args.device.startswith("cuda"))
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(feats, mask)
        else:
            logits = model(feats, mask)

        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    probs = np.concatenate(all_probs, axis=0)  # (N,14)
    labels = np.concatenate(all_labels, axis=0)  # (N,14)

    print("\n===== Per-label metrics (ignore uncertain=0.5) =====")
    rows = []
    for i, name in enumerate(LABEL_NAMES):
        y = labels[:, i]
        p = probs[:, i]

        known = (y != 0.5)
        yk = y[known]
        pk = p[known]

        # mean/var on ALL (including uncertain) & also on known
        mean_all, var_all = float(p.mean()), float(p.var())
        frac_mid_all = float(((p >= 0.4) & (p <= 0.6)).mean())

        if yk.size < 10 or len(np.unique(yk)) < 2:
            rows.append((name, np.nan, np.nan, np.nan, mean_all, var_all, frac_mid_all, yk.size))
            continue

        y_bin = (yk > 0.5).astype(np.int32)  # now binary 0/1
        try:
            auc = float(roc_auc_score(y_bin, pk))
        except Exception:
            auc = np.nan

        t_best, f1_best = best_f1_threshold(y_bin, pk)
        rows.append((name, auc, f1_best, t_best, mean_all, var_all, frac_mid_all, yk.size))

    df = pd.DataFrame(rows, columns=["label", "AUROC", "best_F1", "best_thr",
                                     "prob_mean", "prob_var", "frac_[0.4,0.6]", "n_known"])
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))

    # micro metrics over known labels
    known_mask = (labels != 0.5)
    y_flat = labels[known_mask]
    p_flat = probs[known_mask]
    y_flat_bin = (y_flat > 0.5).astype(np.int32)
    try:
        micro_auc = float(roc_auc_score(y_flat_bin, p_flat))
    except Exception:
        micro_auc = np.nan
    t_micro, f1_micro = best_f1_threshold(y_flat_bin, p_flat, num_steps=401)
    print("\n===== Micro (flatten over known labels) =====")
    print(f"micro_AUROC={micro_auc:.4f}  micro_bestF1={f1_micro:.4f}  micro_bestThr={t_micro:.3f}")

if __name__ == "__main__":
    main()
