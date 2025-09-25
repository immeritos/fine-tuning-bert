import os
import numpy as np
import torch
from typing import List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from dataset import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 6
MAX_LENGTH = 512
STRIDE = 160
BATCH_SIZE = 32
CKPT_PATH = "outputs/bert-chinese-news/best_model_pt"

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

            feat["labels"] = int(ex["labels"] if "labels" in ex else ex["label"])

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

def compute_metrics(logits: np.ndarray, labels_chunk: np.ndarray, doc_ids: np.ndarray):
    
    order = np.argsort(doc_ids)
    doc_ids_sorted = doc_ids[order]
    logits_sorted = logits[order]
    labels_sorted = labels_chunk[order]

    doc_first_index = {}
    for i, d in enumerate(doc_ids_sorted):
        if d not in doc_first_index:
            doc_first_index[d] = i
            
    doc_ids_unique = np.array(sorted(doc_first_index.keys()), dtype=np.int64)
    doc_labels = np.array([int(labels_sorted[doc_first_index[d]]) for d in doc_ids_unique], dtype=np.int64)    
    
    doc_logits = []
    start = 0
    for d in doc_ids_unique:
        end = start
        while end < len(doc_ids_sorted) and doc_ids_sorted[end] == d:
            end += 1
        doc_logits.append(logits_sorted[start:end].mean(axis=0))
        start = end
    doc_logits = np.stack(doc_logits, axis=0)
    
    preds = doc_logits.argmax(axis=-1)
    
    acc = accuracy_score(doc_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        doc_labels, preds, 
        average="macro", 
        zero_division=0
    )
    
    label_order = list(range(NUM_LABELS))
    cm = confusion_matrix(doc_labels, preds, labels=label_order)
    
    target_names = [f"label_{i}" for i in label_order]
    report = classification_report(
        doc_labels, preds, 
        target_names=target_names,
        digits=4, 
        zero_division=0
    )
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }    
    
def main():
    
    tok_from = CKPT_PATH if os.path.isdir(CKPT_PATH) else MODEL_NAME
    tokenizer = BertTokenizerFast.from_pretrained(tok_from)
    collate_fn = build_collate(tokenizer, max_length=MAX_LENGTH)
    
    test_ds = MyDataset(
        "test",
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        stride=STRIDE,
    )
    
    if not os.path.isdir(CKPT_PATH):
        raise FileNotFoundError(f"Not found model directory: {CKPT_PATH}")
    model = BertForSequenceClassification.from_pretrained(CKPT_PATH).to(DEVICE)
    
    args = TrainingArguments(
        output_dir="outputs/test_tmp",
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_drop_last=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
    )

    preds_output = trainer.predict(test_dataset=test_ds)
    logits = preds_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels_chunk = np.array(preds_output.label_ids)
    
    doc_ids = np.array([test_ds[i]["doc_id"] for i in range(len(test_ds))], dtype=np.int64)
    
    results = compute_metrics(logits, labels_chunk, doc_ids)
    
    print("== Test Results ==")
    if results["loss"] is not None:
        print(f"loss: {results['loss']:.4f}")
    print(f"accuracy: {results['accuracy']:.4f}")
    print(f"precision: {results['precision']:.4f}")
    print(f"recall: {results['recall']:.4f}")
    print(f"f1: {results['f1']:.4f}")
    print(f"confusion_matrix:\n", results["confusion_matrix"])
    print(f"classification_report:\n", results["report"])
    
if __name__ == "__main__":
    main()
    