from transformers import BertModel, BertConfig

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_pretrained("bert-base-chinese")
config.max_position_embeddings = 1500
print(config)

pretrained = BertModel(config).to(DEVICE)




# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 10)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        attention_mask = attention_mask.to(torch.float)
        embeddings_output = pretrained.embeddings(input_ids=input_ids)
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [N, 1, 1, sequence_length]
        attention_mask = attention_mask.to(embeddings_output.dtype)
        
        # 冻结 encoder, pooler
        with torch.no_grad():
            encoder_output = pretrained.encoder(embeddings_output, attention_mask=attention_mask)
        
        # 分类任务只使用 encoder_output的[CLS]token
        out = self.fc(encoder_output.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out