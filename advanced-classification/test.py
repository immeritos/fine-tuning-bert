import os
import numpy as np
import torch
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from dataset import MyDataset

# 定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 6
MAX_LENGTH = 256
BATCH_SIZE = 64
CKPT_PATH = "outputs/bert-chinese-sentiment/best_model.pt"

def build_collate(tokenizer, max_length):
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    def collate_fc(batch):
        texts = [x[0] for x in batch]
        labels = [int(x[1]) for x in batch]
        
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        features = []
        n = len(texts)
        for i in range(n):
            feat = {k: enc[k][i] for k in enc}
            feat["labels"] = labels[i]
            features.append(feat)
            
        batch_tensor = collator(features)
        return collate_fc

def compute_metrics(logits, labels):
    
    preds = logits.argmax(axis=-1)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, 
        average="macro", 
        zero_division=0
    )
    
    label_order = list(range(NUM_LABELS))
    cm = confusion_matrix(labels, preds, labels=label_order)
    
    target_names = [f"label_{i}" for i in label_order]
    report = classification_report(
        labels, preds, 
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
    test_ds = MyDataset("test")
    
    tok_from = CKPT_PATH if os.path.isdir(CKPT_PATH) else MODEL_NAME
    tokenizer = BertTokenizerFast.from_pretrained(tok_from)
    collate_fn = build_collate(tokenizer, max_length=MAX_LENGTH)
    
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
        processing_class=tokenizer,
        data_collator=collate_fn,
    )

    preds_output = trainer.predict(test_dataset=test_ds)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    
    results = compute_metrics(logits, labels)
    
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
    