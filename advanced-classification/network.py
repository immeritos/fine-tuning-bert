from transformers import BertModel

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
print(pretrained)

# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 6)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )
            
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out