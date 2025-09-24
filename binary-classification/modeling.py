from __future__ import annotations
from transformers import BertModel
import torch
from torch import nn

class Model(torch.nn.Module):
    """
    BERT + Linear classification head
    - compatible with call: forward(input_ids, attention_mask, token_type_ids)
    - return logits (shape: [B, num_labels])
    - can be used to freeze/unfreeze BERT by freeze_base
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        # Load a pre-trained BERT model provided by Hugging Face.
        self.backbone = BertModel.from_pretrained(model_name)
        # Set the input dimension for the classification layer
        self.hidden = self.backbone.config.hidden_size

        # initialization
 
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)        
        if freeze_base:            
            # Return all trainable parameters in the model.
            for p in self.backbone.parameters():
                # Freeze the parameters and only train the downstream classification layer.
                p.requires_grad = False
                
        # prevent from overfitting         
        self.dropout = nn.Dropout(dropout)
        # FC layer
        self.classifier = nn.Linear(self.hidden, num_labels)
                
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None, 
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids = input_ids,           # the index of the token after tokenization
            attention_mask = attention_mask, # 0/1 vector indicating which positions contain actual tokens and which are padding
            token_type_ids = token_type_ids, # distinguish between sentence A and sentence B.
            return_dict=True,
        )
        
        # [B, L, H] -> get [CLS] tensor
        logits = self.classifier(self.dropout(outputs.pooler_output))
        return logits
            


