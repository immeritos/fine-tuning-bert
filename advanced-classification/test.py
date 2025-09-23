import torch
from mydataset import Mydataset
from torch.utils.data import DataLoader
from network import Model
from transformers import AdamWeightDecay, BertTokenizer

# 定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained("bert-base-chinese")

# 自定义函数对数据进行编码处理
def collate_fn(data):
    sentence = [idx[0] for idx in data]
    label = [idx[1] for idx in data]
    # 编码处理
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentence,
        truncation=True,
        padding="max_length",
        max_length=350,
        return_tensors="pt",
        return_length=True
    )
    
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)
    
    return input_ids, attention_mask, token_type_ids, labels
    
# 创建数据集 
test_dataset = Mydataset("test")

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == "__main__":
    # 开始训练
    acc = 0
    total = 0
    
    print(DEVICE)
    model = Model.to(DEVICE)
    model.load_state_dict(torch.load("params/2bert.pt"))
    model.eval()
    
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据放到DEVICE上
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)
        
        acc = (out == labels).sum().item()
        total += len(labels)
    
    print(acc/total)
    