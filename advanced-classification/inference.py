import os
import torch
from typing import List, Tuple
import numpy as np
from modeling import Model
from transformers import BertTokenizerFast, BertForTokenClassification

# 定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 256
CKPT_PATH = "outputs/bert6/best_model"

ID2LABEL =  ["neutral", "happy", "angry", "sad", "fear", "surprise"]


def load_model():
    src = CKPT_PATH if os.path.isdir(CKPT_PATH) else MODEL_NAME
    tokenizer = BertTokenizerFast.from_pretrained(src)
    model = BertForTokenClassification.from_pretrained(src).to(DEVICE)
    model.eval()
    return model, tokenizer
    

def encode_texts(texts: List[str] | str, tokenizer: BertTokenizerFast):
    """
    tests: List[str] or single str
    """
    if isinstance(texts, str):
        texts = [texts]
        
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensor="pt",
    )

    return {k: v.to(DEVICE) for k, v in enc.items()}

@torch.no_grad()
def predict(texts, model, tokenizer):
    batch = encode_texts(texts, tokenizer)
    
    logits = model(**batch).logits
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=-1)
    return preds.cpu().tolist(), probs.cpu().tolist()

def pretty_print(texts: List[str] | str, preds: List[int], probs: List[List[float]], topk: int = 3):
    if isinstance(texts, str):
        texts = [texts]
    for i, t in enumerate(texts):
        p = preds[i]
        prob = probs[i]
        
        top_idx = np.argsort(prob)[::-1][:topk]
        top_str = ", ".join([f"{ID2LABEL[j]}={prob[j]:.3f}" for j in top_idx])
        print(f"\nText: {t}\nPred: {ID2LABEL[p]} (conf={prob[p]:.3f})")
        print(f"Top-{topk}: {top_str}")
        
def main():
    model, tokenizer = load_model()   
    print("Enter your review text, then press Enter to get a prediction. Enter 'q' to exit.")
    while True:
        text = input(">> ").strip()
        if text.strip().lower() == "q":
            break
        preds, probs = predict(text, model, tokenizer)
        pretty_print(text, preds, probs, topk=3)
    
if __name__ == "__main__":
    main()
    