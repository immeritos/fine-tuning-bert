import os
from typing import List, Dict, Any, Tuple
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyDataset

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 512
STRIDE = 160
NUM_LABELS = 6
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCH = 5
GRAD_CLIP_NORM = 1.0
SEED = 42
OUTPUT_DIR = "outputs/bert-chinese-news"
BEST_PATH = os.path.join(OUTPUT_DIR, "best_model_pt")

# ---------- Collate: Pad sequences + Preserve document metadata ----------
def build_collate(tokenizer, max_length):
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    def collate_fc(batch):
        features = []
        meta_keys = ("doc_id", "chunk_idx", "num_chunks")
        for ex in batch:
            feat = {
                k: ex[k]
                for k in ("input_ids", "attention_mask", "token_type_ids")
                if k in ex
            }
            if "labels" in ex:
                feat["labels"] = int(ex["labels"])
            else:
                feat["labels"] = int(ex["label"])
            for k in meta_keys:
                if k in ex:
                    feat[k] = ex[k]
            features.append(feat)
            
        batch_tensor = collator(features)
        for k in ("doc_id", "chunk_idx", "num_chunks", "labels"):
            if k in batch_tensor and not torch.is_tensor(batch_tensor[k]):
                batch_tensor[k] = torch.tensor(batch_tensor[k], dtype=torch.long)
                
        return batch_tensor
    
    return collate_fc
    
def freeze_backbone(model):
    for p in model.bert.parameters():
        p.requires_grad=False
    model.bert.eval()
    return model

class TrainerForChunks(Trainer):
    def __init__(self, *args, agg: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        assert agg in {"mean", "logsumexp"}
        self.agg = agg
        self._ce = nn.CrossEntropyLoss()
    
    def _aggregate_logits(
        self, logits: torch.Tensor, doc_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return :
            doc_logits: [num_docs_in_batch, num_labels]
            doc_ids_unique: [num_docs_in_batch]
        """
        order = torch.argsort(doc_ids)
        doc_ids_sorted = doc_ids[order]
        logits_sorted = logits[order]
        
        uniques, counts = torch.unique_consecutive(doc_ids_sorted, return_counts=True)
        splits = torch.split(logits_sorted, tuple(counts.tolist()))
        
        doc_logits_list = []
        for chunk_logits in splits:
            if self.agg == "mean":
                doc_logits_list.append(chunk_logits.mean(dim=0))
            else:
                doc_logits_list.append(torch.logsumexp(chunk_logits, dim=0))
        doc_logits = torch.stack(doc_logits_list, dim=0)
        return doc_logits, uniques
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        doc_ids = inputs.pop("doc_id")
        outputs = model(**inputs)
        logits = outputs.logits
        
        doc_logits, uniques = self._aggregate_logits(logits, doc_ids)
        order = torch.argsort(doc_ids)
        labels_sorted = labels[order]
        _, counts = torch.unique_consecutive(doc_ids[order], return_counts=True)
        start_idx = torch.cumsum(torch.tensor([0] + counts[:-1].tolist()), dim=0)
        doc_labels = labels_sorted[start_idx]
        
        loss = self._ce(doc_logits, doc_labels)
        return (loss, outputs) if return_outputs else loss

    
def main():
    # 1) data & tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds = MyDataset(
        "train",
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        stride=STRIDE,
    )
    val_ds = MyDataset(
        "validation",
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        stride=STRIDE,
    )
    
    collate_fn = build_collate(tokenizer, max_length=MAX_LENGTH)

    
    # 2) model
    id2label = {i: f"label_{i}" for i in range(NUM_LABELS)}
    label2id = {v: k for k, v in id2label.items()}
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    
    freeze_backbone(model)
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCH,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        max_grad_norm=GRAD_CLIP_NORM,
    )
    
    eval_doc_ids = np.array([val_ds[i]["doc_id"] for i in range(len(val_ds))], dtype=np.int64)
    doc_first_index = {}
    for i, d in enumerate(eval_doc_ids):
        if d not in doc_first_index:
            doc_first_index[d] = i
    eval_doc_labels = np.array(
        [int(val_ds[idx]["labels"] if "labels" in val_ds[idx] else val_ds[idx]["label"])
         for d, idx in sorted(doc_first_index.items(), key=lambda x: x[0])],
        dtype=np.int64
    )
    eval_doc_ids_unique = np.array(sorted(doc_first_index.keys()), dtype=np.int64)
    
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        
        order = np.argsort(eval_doc_ids)
        doc_ids_sorted = eval_doc_ids[order]
        logits_sorted = logits[order]
        
        doc_logits = []
        start = 0
        for d in eval_doc_ids_unique:
            end = start
            while end < len(doc_ids_sorted) and doc_ids_sorted[end] == d:
                end += 1
            doc_logits.append(logits_sorted[start:end].mean(axis=0, keepdims=False))
            start = end
        doc_logits = np.stack(doc_logits, axis=0)
        
        preds = doc_logits.argmax(axis=-1)
        acc = (preds == eval_doc_labels).mean().item()
        return {"accuracy": acc}

    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    ]
    trainer = TrainerForChunks(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        agg="mean",
    )
    
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(BEST_PATH)
    tokenizer.save_pretrained(BEST_PATH)
    
if __name__ == "__main__":
    main()