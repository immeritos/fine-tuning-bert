import os
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import numpy as np
import torch

from dataset import MyDataset

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 256
NUM_LABELS = 6
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCH = 5
GRAD_CLIP_NORM = 1.0
SEED = 42
OUTPUT_DIR = "outputs/bert-chinese-sentiment"
BEST_PATH = os.path.join(OUTPUT_DIR, "best_model_pt")

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
        return batch_tensor
    
    return collate_fc
    
def freeze_backbone(model):
    for p in model.bert.parameters():
        p.requires_grad=False
    model.bert.eval()
    return model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}
    
def main():
    # 1) data & tokenizer
    train_ds = MyDataset("train")
    val_ds = MyDataset("validation")
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
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
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )
    
    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    ]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(BEST_PATH)
    tokenizer.save_pretrained(BEST_PATH)
    
if __name__ == "__main__":
    main()