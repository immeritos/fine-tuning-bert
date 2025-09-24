# Fine-tuning BERT for Sentiment Classification (Binary & 6-way)

This project demonstrates fine-tuning a Hugging Face pretrained BERT (`bert-base-chinese`) for:

- Binary sentiment classification (e.g., ChnSentiCorp).

- Multi-class (6-way) sentiment classification on a Weibo COVID-19 emotion dataset (labels: `neutral`, `happy`, `angry`, `sad`, `fear`, `surprise`).

---

## 🔑 Pipelines

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

## 📦 Project Structure

```bash
.
├─ Dataset.py                 # your custom torch Dataset -> returns (text, label)
├─ trainer.py                 # Trainer-based training (6-way)
├─ test.py                    # evaluation using Trainer.predict
├─ inference.py               # CLI inference (single/batch text)
├─ metrics.py                 # (optional) sklearn metrics helper
└─ data/                      # dataset storage (HF load_from_disk or raw)
```

---

## 📌 Notes
- Original Pipeline: Training loop is hand-written for learning purposes.
- Minimal HF Pipeline: For 6-way classification we use BertForSequenceClassification + Trainer to reduce boilerplate.
```bash
from transformers import BertForSequenceClassification, BertTokenizerFast
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=6)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
```
- Uses `[CLS]` token as the sentence-level representation.
- Supports both feature extraction (freeze BERT) and full fine-tuning.
- If OOM: reduce `max_length` or batch size; enable `fp16`.

