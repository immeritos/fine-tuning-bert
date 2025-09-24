import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

from dataset import MyDataset
from modeling import Model
from metrics import evaluate_loader

# ----------------------
# Define hyperparameter
# ----------------------
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCH = 5
GRAD_CLIP_NORM = 1.0
SEED = 42
OUTPUT_DIR = "outputs/bert-chinese-sentiment"
BEST_PATH = os.path.join(OUTPUT_DIR, "best_model_pt")

# ----------------------
# device & random seed
# ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# ensure the reproductibility of the experiment
set_seed(SEED)

# ----------------------
# Tokenizer & collator
# ----------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) # Convert natural language text into numerical input
# Align the samples in a batch to a uniform length, and then package them into a tensor
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# ----------------------
# Customize collate_fn: Encode in batch
# return (input_ids, attention_mask, token_type_ids, labels)
# ----------------------
def collate_fn(batch):
    sentences = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    # 编码处理
    enc = tokenizer.batch_encode_plus(
        sentences,
        truncation=True,
        padding=True,           # We have used DataCollator
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_length=True
    )
    
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    
    return input_ids, attention_mask, token_type_ids, labels
    
    
# --------------------
# Dataset & DataLoader
# --------------------
train_ds = MyDataset("train")
val_ds = MyDataset("validation")

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# --------------------
# Model, Optimizer, AMP
# --------------------
model = Model().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# Automatically scale gradients during FP16 training.
scaler = torch.GradScaler(device='cuda', enabled=torch.cuda.is_available())
criterion = torch.nn.CrossEntropyLoss()

os.makedirs(OUTPUT_DIR, exist_ok=True)
best_f1 = -1.0

# ----------------------
# Training loop
# ----------------------   
if __name__ == "__main__":

    print("Device:", DEVICE)
    step = 0
    for epoch in range(1, NUM_EPOCH + 1):
        model.train()
        for input_ids, attention_mask, token_type_ids, labels in train_loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            token_type_ids = token_type_ids.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # Clear the previous gradients
            optimizer.zero_grad(set_to_none=True) 
            # Mixed Precision Computing           
            with torch.autocast(device_type='cuda',enabled=torch.cuda.is_available()):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
            
            # Scale the loss before performing backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            # Update parameters using the optimizer
            scaler.step(optimizer)
            # Dynamically adjust the scaling factor
            scaler.update()
            
            if step % 50 == 0:
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                print(f"[Epoch {epoch}] step {step} | loss={loss.item():.4f} | acc={acc:.4f}")
            step += 1
        
        # evalute each epoch
        criterion = torch.nn.CrossEntropyLoss()
        metrics = evaluate_loader(model, val_loader, DEVICE, criterion=criterion)
        print(f"==> Eval @ Epoch {epoch}:"
              f"loss={metrics['loss']:.4f}"
              f"acc={metrics['accuracy']:.4f}"
              f"f1={metrics['f1']:.4f}")
        
        # save the best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), BEST_PATH)
            print(f"Saved best model to {BEST_PATH} (val_f1={best_f1:.4f})")

    print("Training done. Best F1:", best_f1)
    