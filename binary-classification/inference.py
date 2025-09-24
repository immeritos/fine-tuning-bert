import os
import torch
from modeling import Model
from transformers import BertTokenizer

# 定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 256
CKPT_PATH = "outputs/bert-chinese-sentiment/best_model.pt"
ID2LABEL = {0: "negative review", 1: "positive review"}


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(texts):
    """
    tests: List[str] or single str
    """
    if isinstance(texts, str):
        texts = [texts]
        
    enc = tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensor="pt",
        return_attention_masks=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    return input_ids, attention_mask, token_type_ids

@torch.no_grad()
def predict(texts, model):
    model.eval()        
    input_ids, attention_mask, token_type_ids = encode_texts(texts)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    
    logits = model(input_ids, attention_mask, token_type_ids)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=-1).tolist()
    return preds, probs.tolist()

def main():
    model = Model().to(DEVICE)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"could not find weights: {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    
    print("Enter your review text, then press Enter to get a prediction. Enter 'q' to exit.")
    while True:
        text = input(">> ")
        if text.strip().lower() == "q":
            break
        preds, probs = predict(text, model)
        pred = preds[0]
        print(f"Model prediction: {ID2LABEL[pred]} (Confidence level={probs[0][pred]:.3f})\n")
    
if __name__ == "__main__":
    main()
    