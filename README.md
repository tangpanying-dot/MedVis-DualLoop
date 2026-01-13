# 🏥 Medical Report Generation with Knowledge Graphs

A multimodal deep learning system for automated chest X-ray radiology report generation.

---

## 📋 Quick Start

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

---

## 🚀 Running the Code

### Step 1: 📦 Install Dependencies

```bash
pip install torch torchvision transformers accelerate peft bitsandbytes
pip install numpy pandas pillow pydicom nltk rouge-score networkx
```

### Step 2: 💾 Prepare Dataset

Organize your MIMIC-CXR dataset in the `data/` directory:

```
data/
├── mimic_cxr/
│   ├── files/
│   ├── images/
│   └── metadata.csv
└── mimic_iv/
    └── .csv
```

### Step 3: 🎯 Training Pipeline

#### **Phase 1: Feature Alignment Training**

Train the multimodal connector to align visual features with language model embeddings:

```bash
python train_stage1.py
```

**What this does:**
- ✅ Loads RAD-DINO visual features
- ✅ Aligns image features with Gemma embeddings
- ✅ Saves connector weights to `checkpoints/`

**Expected output:** Stage 1 checkpoint in `checkpoints/stage1/best_checkpoint.pt`

---

#### **Phase 2: Report Generation Training**

Fine-tune the Gemma language model for report generation:

```bash
python train_stage2.py
```

**What this does:**
- ✅ Loads Stage 1 checkpoint
- ✅ Integrates knowledge graph features
- ✅ Fine-tunes Gemma with QLoRA
- ✅ Optimizes for BLEU-4 scores

**Expected output:** Final model in `checkpoints/stage2/best_checkpoint.pt`

---

### Step 4: 📝 Generate Reports

#### **Option A: Ensemble Mode (Recommended)** ⭐

Generate reports using ensemble predictions:

```bash
# For Gemma-2B model
python generate_report_gemma_2b_ensemble.py
```

#### **Option B: Manual Mode**

Generate reports with manual control:

```bash
# For Gemma-2B model
python generate_report_gemma_2b_manual.py
```

**Output location:** Generated reports are saved to `report/` or `report_final/`

---

### Step 5: 📊 Evaluate Results

Run evaluation to compute metrics:

```bash
python eval.py --report genrate_report.jsonl
```

**Metrics computed:**
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- CheXpert_F1,CheXpert_P,CheXpert_R

**Results location:** `eval_results/evaluation_summary.csv`

---

## 🔧 Additional Tools

### Knowledge Graph Visualization

Preprocess knowledge graph features for faster loading:

```bash
python run_visualization.py
```

### Report Quality Evaluation

Evaluate specific generated reports:

```bash
python report_evaluator.py --report_dir report_final/
```
---

## ⚙️ Configuration Tips

### 💡 For Limited GPU Memory (< 16GB)

Reduce batch size and use gradient accumulation:

```bash
python train_stage2.py \
  --batch_size 2 \
  --gradient_accumulation_steps 8
```

## 📁 Key Directories

```
├── checkpoints/       # 💾 Saved model weights
├── data/              # 📂 Training dataset
├── eval_results/      # 📊 Evaluation metrics
├── kg/                # 🧠 Knowledge graph data
├── report/            # 📝 Generated reports (intermediate)
├── report_final/      # ✅ Final curated reports
└── visual/            # 👁️ Visual feature processing
```
