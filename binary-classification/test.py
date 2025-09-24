import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import MyDataset
from modeling import Model
from metrics import evaluate_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 256
BATCH_SIZE = 64
CKPT_PATH = "outputs/bert-chinese-sentiment/best_model.pt"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    sentences = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    
    enc = tokenizer.batch_encode_plus(
        sentences,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensor="pt",
        return_attention_mask=True
    )
    
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    return input_ids, attention_mask, token_type_ids

def main():
    test_ds = MyDataset("test")
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    
    model = Model().to(DEVICE)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"could not find weights: {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    results = evaluate_loader(model, test_loader, DEVICE, criterion=criterion)
    print("== Test Results ==")
    if results["loss"] is not None:
        print(f"loss: {results['loss']:.4f}")
    print(f"accuracy: {results['accuracy']:.4f}")
    print(f"precision: {results['precision']:.4f}")
    print(f"recall: {results['recall']:.4f}")
    print(f"f1: {results['f1']:.4f}")
    print(f"confusion_matrix:\n", results["confusion_matrix"])
    print(f"classification_report:\n", results["report"])