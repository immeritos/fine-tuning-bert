from __future__ import annotations
from typing import Optional
from transformers import BertModel
import torch
from torch import nn

class Model(torch.nn.Module):
    """
    BERT + Classification head
    - forward(input_ids, attention_mask, token_type_ids) -> logits [B, num_labels]
    - choose pooling: 'cls' | 'pooler' | 'mean'
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 6,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        # Load pre-trained model
        self.backbone = BertModel.from_pretrained(model_name)
        # Set the input dimension for the classification layer
        self.hidden = self.backbone.config.hidden_size
        
        if freeze_base:            
            # Return all trainable parameters in the model.
            for p in self.backbone.parameters():
                # Freeze the parameters and only train the downstream classification layer.
                p.requires_grad = False
                
        # prevent from overfitting         
        self.dropout = nn.Dropout(dropout)
        # FC layer
        self.classifier = nn.Linear(self.hidden, num_labels)
        
        # initialization
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
                
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids = input_ids,           # the index of the token after tokenization
            attention_mask = attention_mask, # 0/1 vector indicating which positions contain actual tokens and which are padding
            token_type_ids = token_type_ids, # distinguish between sentence A and sentence B.
            return_dict=True,
        )
        
        # [B, L, H] -> get [CLS] tensor
        last_hidden_state = outputs.last_hidden_state
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / lengths
        else:
            pooled = last_hidden_state.mean(dim=1)
            
                 
        logits = self.classifier(self.dropout(pooled))
        return logits
            


