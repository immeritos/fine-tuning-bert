from __future__ import annotations
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


@torch.no_grad()
def evaluate_loader(model, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    for batch in dataloader:
        if len(batch) == 4:
            input_ids, attention_mask, token_type_ids, labels = batch
        else:
            raise ValueError("Batch should be (input_ids, attention_mask, token_type_ids, labels).")
        
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(input_ids, attention_mask, token_type_ids)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
            
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    
    avg_loss = total_loss / len(dataloader.dataset) if criterion is not None else None
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }