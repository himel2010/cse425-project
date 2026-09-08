"""DistilBERT text encoder + classifier (Task 1 / spec Alg. 1)."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    """DistilBERT; exposes CLS (first-token) and full token sequence."""

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden = self.bert.config.hidden_size  # 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state          # [B, L, 768]
        cls = H[:, 0]                      # [B, 768] (DistilBERT: first token)
        return cls, H


class BertClassifier(nn.Module):
    """Task 1: DistilBERT + linear head -> multi-label logits."""

    def __init__(self, model_name: str, n_labels: int):
        super().__init__()
        self.encoder = BertTextEncoder(model_name)
        self.head = nn.Linear(self.encoder.hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        cls, _ = self.encoder(input_ids, attention_mask)
        return self.head(cls)


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
