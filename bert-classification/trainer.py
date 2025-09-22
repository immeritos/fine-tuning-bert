import torch
from mydataset import Mydataset
from torch.utils.data import DataLoader
from network import Model
from transformers import BertTokenizer, AdamWeightDecay

# 定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

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
train_dataset = Mydataset("train")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == "__main__":
    # 开始训练
    print(DEVICE)
    model = Model.to(DEVICE)
    optimizer = AdamWeightDecay(model.parameters(), lr=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据放到DEVICE上
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            
            out = model(input_ids, attention_mask, token_type_ids)
            
            loss = loss_func(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i%5 == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(epoch, i, loss.item(), acc)
        
        # 保存模型参数
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
        print(epoch, "参数保存成功！")
    