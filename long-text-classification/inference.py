import os
import torch
from typing import List, Tuple, Dict, Any
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 512
STRIDE = 160
CKPT_PATH = "outputs/bert-news/best_model"

ID2LABEL =  ["体育", "娱乐", "科技", "教育", "游戏", "财经"]


def load_model():
    src = CKPT_PATH if os.path.isdir(CKPT_PATH) else MODEL_NAME
    tokenizer = BertTokenizerFast.from_pretrained(src)
    model = BertForTokenClassification.from_pretrained(src).to(DEVICE)
    model.eval()
    return model, tokenizer
    

def chunk_encode(
    texts: List[str] | str, 
    tokenizer: BertTokenizerFast,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
) -> Dict[str, torch.Tensor]:
    """
    tests: List[str] or single str
    """

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    
    content_len = max_length - 2
    if content_len <= 0:
        raise ValueError("max_length must be >=2 to fit CLS/SEP. ")
    
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    chunks = []
    step = max(1, content_len - stride)
    if len(ids) == 0:
        piece = [cls_id, sep_id]
        chunks.append({"input_ids": piece, "attention_mask": [1, 1], "token_type_ids": [0, 0]})
    else:
        for start in range(0, len(ids), step):
            piece = ids[start:start + content_len]
            piece = [cls_id] + piece + [sep_id]
            chunks.append({
                "input_ids": piece,
                "attention_mask": [1] * len(piece),
                "token_type_ids": [0] * len(piece),
            })
            if start + content_len >= len(ids):
                break
    
    batch = tokenizer.pad(
        chunks,
        padding=True,
        return_tensors="pt",
    )

    return {k: v.to(DEVICE) for k, v in batch.items()}

@torch.no_grad()
def predict_long_text(text, model, tokenizer, agg: str = "mean"):
    batch = chunk_encode(text, tokenizer, MAX_LENGTH, STRIDE)
    
    logits = model(**batch).logits
    if agg == "mean":
        doc_logits = logits.mean(dim=0)
    else:
        doc_logits = torch.logsumexp(logits, dim=0)
        
    prob = torch.softmax(doc_logits, dim=1)
    pred = int(torch.argmax(prob, dim=-1).item())
    return pred, prob.cpu().tolist()

@torch.no_grad()
def predict(texts, model, tokenizer, agg: str = "mean"):
    if isinstance(texts, str):
        texts = [texts]
        
    preds, probs = [], []
    for t in texts:
        p, pr = predict_long_text(t, model, tokenizer, agg=agg)
        preds.append(p)
        probs.append(pr)
    return preds, probs

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
        preds, probs = predict(text, model, tokenizer, agg="mean")
        pretty_print(text, preds, probs, topk=3)
    
if __name__ == "__main__":
    main()
    