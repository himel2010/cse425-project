"""Task 3 fusion model + ablation variants (spec §4.3, PLAN.md P5).

variants: cross_attn (main), concat, bert_only, gnn_only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bert_encoder import BertTextEncoder
from gnn_model import GraphSAGEEncoder


class FusionModel(nn.Module):
    def __init__(self, cfg: dict, n_labels: int, variant: str = "cross_attn"):
        super().__init__()
        self.variant = variant
        hidden = cfg["model"]["hidden_dim"]        # 256
        node_dim = cfg["audio"]["node_feat_dim"]   # 64
        bert_name = cfg["text"]["bert_model"]
        drop = cfg["model"]["dropout"]

        self.gnn = GraphSAGEEncoder(node_dim, hidden, drop)
        self.bert = BertTextEncoder(bert_name)
        bdim = self.bert.hidden                     # 768

        if variant == "gnn_only":
            self.head = nn.Linear(hidden, n_labels)
        elif variant == "bert_only":
            self.head = nn.Linear(bdim, n_labels)
        elif variant == "concat":
            self.head = nn.Sequential(
                nn.Linear(hidden + bdim, hidden), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(hidden, n_labels))
        elif variant == "cross_attn":
            # graph readout = query; BERT tokens -> keys/values (projected)
            self.q_proj = nn.Linear(hidden, hidden)
            self.k_proj = nn.Linear(bdim, hidden)
            self.v_proj = nn.Linear(bdim, hidden)
            self.scale = hidden ** 0.5
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(hidden, n_labels))
        else:
            raise ValueError(variant)

    def _cross_attn(self, g, H, mask):
        # g:[B,H]  H:[B,L,768]  mask:[B,L]
        q = self.q_proj(g).unsqueeze(1)            # [B,1,H]
        k = self.k_proj(H)                         # [B,L,H]
        v = self.v_proj(H)                         # [B,L,H]
        scores = (q @ k.transpose(1, 2)) / self.scale   # [B,1,L]
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v).squeeze(1)                # [B,H]
        return ctx

    def forward(self, data, input_ids, attention_mask):
        if self.variant == "bert_only":
            cls, _ = self.bert(input_ids, attention_mask)
            return self.head(cls)

        g = self.gnn(data.x, data.edge_index, data.batch)
        if self.variant == "gnn_only":
            return self.head(g)

        cls, H = self.bert(input_ids, attention_mask)
        if self.variant == "concat":
            return self.head(torch.cat([g, cls], dim=-1))
        # cross_attn
        ctx = self._cross_attn(g, H, attention_mask)
        z = torch.cat([g, ctx], dim=-1)            # [B, 2H]
        return self.head(z)
