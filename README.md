# Hugging Face Fine-tuning Project — General + Three Modules Outline

> This project demonstrates fine-tuning a Hugging Face pretrained BERT (`bert-base-chinese`) for:

- Binary sentiment classification (e.g., comments from online shopping platforms).

- Multi-class (6-way) sentiment classification on a Weibo COVID-19 emotion dataset (labels: `neutral`, `happy`, `angry`, `sad`, `fear`, `surprise`). 

- News topic classification with sliding-window chunking on THUNews dataset (labels: `sport`, `health`, `finance`, `entertainment`, `tech`, `game`).

---

## General Knowledge Outline

### 0. Project Code Topology & Roles
- **`dataset.py`**: define dataset item(s) and (if needed) token-based chunking for long texts; attach metadata (`doc_id`, `chunk_idx`, `num_chunks`, `label`).
- **`modeling.py`** (optional): encoder backbone + classification head ([CLS]/mean-pool/pooler) → logits.
- **`trainer.py`**: training loop/`Trainer` config; loss, optimizer, LR scheduler, early stopping, checkpointing; (for long texts) **document-level aggregation** logic.
- **`test.py`**: evaluation pipeline; metrics; (for long texts) **doc-level** aggregation before metrics.
- **`inference.py`**: single/batch inference; (for long texts) chunk → forward → aggregate → final label/probabilities.
- **`metrics.py`** (optional): metrics helpers (Macro-F1, ROC/PR-AUC, confusion matrix).
- **`collate_fn`**: padding to batch max length; keep metadata; return tensors.

```bash
.
├─ Dataset.py          
├─ trainer.py 
├─ test.py    
├─ inference.py 
├─ metrics.py   
└─ data/   
```

---

### 1. Tokenizer & Inputs
- **Tokenization**: WordPiece (Chinese often ≈ per character); hard limit is **tokens**, not characters.
- **Key fields**:  
  - `input_ids`: vocabulary indices.  
  - `attention_mask`: 1 for real tokens, 0 for padding (used to mask attention).  
  - `token_type_ids` (segment IDs): sentence-pair A/B (single-sentence tasks are all-zeros).  
- **`return_tensors`**: `"pt"` (PyTorch), `"tf"`, or `"np".
- **Truncation side** (`tokenizer.truncation_side`):  
  - `"right"` keeps the beginning and truncates the end (default).  
  - `"left"` keeps the end and truncates the beginning.  
  - **Short-text tasks** (sentiment/emotion) should **compare both**, as negation/turns often at the end.  
  - **Long-text with sliding window**: truncation side matters less; stride overlap is more critical.

---

### 2. Pooling Choices
- **`pooler_output`**: `tanh(W·[CLS] + b)` — a dense + Tanh projection historically used in BERT’s NSP; simple and stable.
- **raw `[CLS]`** (no Tanh): the first token’s contextual representation; sometimes retains more information.  
- **mean pooling** over `last_hidden_state` (mask-aware): more robust when info is distributed / noisy.
- **Recommendation**: expose a switch (pooled | cls | mean) and compare Macro-F1 on validation.

---

### 3. Handling Long Inputs
- **Plan A (recommended baseline)**: **sliding-window chunking** (≤ max_length; overlap by `stride`), model each chunk, then **aggregate logits** into a document-level prediction:  
  - Aggregation: `mean` (default), `max`, or `logsumexp`.
  - Train with **document-level loss**: aggregate per-doc logits first, then compute CE once.
- **Plan B**: **hierarchical model** (chunk encodings → aggregator like BiLSTM/Transformer/attention pooling → doc logits).
- **Plan C**: **long-sequence architectures** (Longformer/BigBird) with sparse attention.
- **Plan E (not typical)**: increasing `max_position_embeddings` for BERT (needs continued pretraining; still O(n²) attention; unstable).

---

### 4. Collation, DataLoader & Performance
- **Collate**: use `DataCollatorWithPadding` or `tokenizer.pad` to dynamic-pad to batch max length; keep metadata (`doc_id`, etc.).
- **`num_workers`**: number of subprocesses for data loading; increase until GPU no longer starves (4–16 typical on workstations).  
  - Use `persistent_workers=True` for long training; ensure `if __name__ == "__main__":` on Windows/macOS.  
- **`pin_memory=True` + `non_blocking=True` (when moving to GPU)**: faster H2D copies; useful only for GPU training/inference.

---

### 5. Class Imbalance Countermeasures
- **Loss-side**:  
  - Class-weighted **CrossEntropy** (first choice).  
  - **Focal loss** (focus on hard examples; γ≈1–2).  
  - **Logit adjustment** using prior `π_c` (train or inference).
- **Sampling-side**: `WeightedRandomSampler` (beware overfitting); downsampling major class (rarely first choice).
- **Metrics**: prefer **Macro-F1** + per-class F1; PR-AUC for rare-class sensitivity.

---

### 6. LR Schedules & Optimizer
- Optimizer: `AdamW/AdamWeightDecay`, LR in **1e-5 ~ 5e-5**, `weight_decay=0.01` (exclude bias/LayerNorm).
- Schedulers:  
  - **`linear` + warmup (3–10%)** — strong default.  
  - `cosine` (smoother tail), `cosine_with_restarts`, `polynomial`, `constant(_with_warmup)` for special cases.
- Gradient tricks: `max_grad_norm=1.0`, gradient accumulation for small GPUs; consider layer-wise LR decay (LLRD) on small data.

---

### 7. Metrics & Error Analysis
- **ROC-AUC vs PR-AUC**: PR-AUC is more informative on **imbalanced** datasets; ROC-AUC measures ranking quality overall.
- **F1 variants**:  
  - **Micro-F1** aggregates TP/FP/FN across classes (majority-dominated).  
  - **Macro-F1** averages per-class F1 equally (imbalance-robust).  
  - Weighted-F1 is frequency-weighted average.
- **Confusion matrix**: standardize by row/column and extract **Top-k confusion pairs** for targeted fixes.
- **Top-k class print**: show top-k logits/probs per sample for quick sanity checks.
- **Token/phrase importance**: attention maps (exploratory), and **gradient-based attributions** (Grad×Input / Integrated Gradients) for more causal signals.
- **Doc-level evaluation** (with chunking): **aggregate chunk logits by `doc_id`** before computing metrics.

---

### 8. Reproducibility & Logging
- Set random seeds (`python`, `numpy`, `torch`, `transformers`); log hyperparams and results (CSV/W&B/TensorBoard).
- Save artifacts: best checkpoint directory (config, tokenizer, model weights).

---

## Pipelines

The implementation follows the standard fine-tuning procedure for downstream tasks with BERT:

1. **Load Pretrained Model & Tokenizer**
   - Backbone: `BertModel.from_pretrained("bert-base-chinese")`
   - Tokenizer: `BertTokenizer.from_pretrained("bert-base-chinese")`

2. **Prepare Dataset**
   - Use Hugging Face `datasets` to load `ChnSentiCorp` and `Weibo-COVID-19-emotion` locally.
   - Wrap with a custom `torch.utils.data.Dataset` (`dataset.py`) to return `(text, label)` pairs.

3. **Tokenization & Collation**
   - Convert text into `input_ids`, `attention_mask`, `token_type_ids` via tokenizer.
   - Use a `collate_fn` in `DataLoader` to batch and pad sequences.

4. **Define Model Architecture**
   - Base encoder: pretrained BERT.
   - Classification head: `[CLS]` embedding → Dropout → Linear layer → logits over 2 classes (`modeling.py`).

5. **Training Loop (`trainer.py`)**
   - Loss: `nn.CrossEntropyLoss`
   - Optimizer: `AdamWeightDecay`
   - Training: forward pass → compute loss → `loss.backward()` → `optimizer.step()`
   - Mixed precision (`torch.GradScaler`) for efficiency (if GPU available).
   - Save checkpoints per epoch.

6. **Evaluation (`metrics.py` + `test.py`)**
   - Switch model to `eval()` mode with `torch.no_grad()`.
   - Compute accuracy, precision, recall, F1 on validation/test splits.

7. **Inference (`inference.py`)**
   - Load best checkpoint.
   - Encode user input text, run through model, output predicted label + confidence.
```bash
python inference.py
>> 这个电影太好看了！
Text: 这个电影太好看了！
Pred: happy (conf=0.92)
Top-3: happy=0.920, neutral=0.050, surprise=0.015
```

---


## Module 1 — BERT Sentiment Binary Classification (Short Reviews)

### 1. Task & Data
- Dataset: ChnSentiCorp (or your actual set); train/val/test splits; label map {0: neg, 1: pos}.  
- Cleaning strategy (dedup/empty/too-long). Class balance check.

### 2. Tokenization & Max Length
- Compare `truncation_side={"right","left"}` at `max_length ∈ {128, 256}`; keep configs identical across train/val/test.

### 3. Model & Head
- `bert-base-chinese` backbone. Head option: **pooler_output**.

### 4. Training
- Loss: CE (optionally class-weighted). Optimizer: AdamW, LR=2e-5, WD=0.01.  
- Scheduler: `linear` + warmup_ratio≈0.06.  
- Early stopping by Macro-F1; gradient clipping; AMP.

### 5. Evaluation
- Metrics: Accuracy + **Macro-F1** (primary); confusion matrix; error case sampling.  
- Optional: ROC/PR-AUC (treat as binary).

### 6. Inference
- Single/batch inputs; top-k classes; optional token importance (Grad×Input).

### 7. Findings Template
- Best head, best truncation side, best LR/max_length; typical failure modes (negation, sarcasm, domain slang).

---

## Module 2 — Weibo Six-Emotion Classification

### 1. Task & Data
- Labels & distribution; emoji/topic-hashtag prevalence; short-text characteristics.

### 2. Tokenization
- Try smaller `max_length` (e.g., 96/128) to reduce padding; compare truncation side given tail emojis/hashtags.

### 3. Model & Head
- Same backbone; head switch (pooler / cls / mean). Possibly stronger dropout.

### 4. Training
- Loss: CE (class-weighted if imbalance). LR grid around 2e-5; warmup 6%; AMP.  
- Optional: Focal loss if minority emotions suffer.

### 5. Evaluation
- Primary: **Macro-F1** + per-class F1; confusion matrix to inspect pairs like “悲伤↔愤怒/厌恶”。  
- Optional: macro PR-AUC (OvR).

### 6. Inference & Explainability
- Top-k print; token importance; handle emojis/hashtags normalization carefully.

### 7. Findings Template
- Which emotions are hardest; effect of truncation/mean pooling; common confusions and fixes.

---

## Module 3 — News Topic Classification with Sliding-Window Chunking

### 1. Long-Text Strategy
- Chunking: `max_length=512` (content_len=510) with `stride≈128~192`.  
- Dataset returns chunk-level samples with `doc_id`, `chunk_idx`, `num_chunks`.

### 2. Training with Doc-Level Loss
- Forward all chunks in batch → **group by `doc_id`** → aggregate chunk logits (mean/max/logsumexp) → CE once per doc.

### 3. Evaluation (Doc-Level)
- Predict all chunks → aggregate by `doc_id` → compute metrics (Accuracy, **Macro-F1**, confusion matrix).

### 4. Inference
- For a single article: chunk → forward → aggregate → final label/probs;  
- Print top-m contributing chunks by predicted class logit; optional Grad×Input within top chunks.

### 5. Ablations
- Vary `stride` / `max_length` / aggregation; possibly compare mean vs logsumexp.  
- Optional hierarchical or Longformer/BigBird baselines.

### 6. Findings Template
- Best aggregation; chunk-level contribution patterns; main confusion pairs (e.g., 科技↔财经).

---

## Appendix — Ready-to-Use Checklists

- **Tokenizer parity**: same `max_length`, `truncation_side`, and padding strategy across train/val/test/inference.  
- **Doc aggregation**: keep `doc_id`/`chunk_idx`/`num_chunks` through collate and trainer.  
- **Performance**: set `num_workers`, `pin_memory`, `persistent_workers`; use `non_blocking=True` on `.to(device)`.  
- **Imbalance**: class weights first; focal/logit adjustment if needed; monitor Macro-F1 and per-class F1.  
- **Schedulers**: start with `linear + warmup_ratio≈0.06`; try `cosine` if needed.  
- **Explainability**: top-k class print; Grad×Input or IG for token importance; attention maps only as exploratory.  
- **Reproducibility**: set seeds; log runs; save checkpoint dirs (config+model+tokenizer).

