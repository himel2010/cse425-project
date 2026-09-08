"""Task 4 dual-encoder + symmetric InfoNCE (spec §4.4, PLAN.md P6)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GraphSAGEEncoder


class DualEncoder(nn.Module):
    """GNN(graph)->proj->L2  ||  BERT(CLS)->proj->L2, shared embed space."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg["model"]["hidden_dim"]
        proj = cfg["model"]["proj_dim"]
        node_dim = cfg["audio"]["node_feat_dim"]
        drop = cfg["model"]["dropout"]

        self.gnn = GraphSAGEEncoder(node_dim, hidden, drop)
        self.gnn_proj = nn.Linear(hidden, proj)
        self.bert = BertTextEncoder(cfg["text"]["bert_model"])
        self.bert_proj = nn.Linear(self.bert.hidden, proj)

    def encode_graph(self, data):
        g = self.gnn(data.x, data.edge_index, data.batch)
        return F.normalize(self.gnn_proj(g), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        cls, _ = self.bert(input_ids, attention_mask)
        return F.normalize(self.bert_proj(cls), dim=-1)

    def forward(self, data, input_ids, attention_mask):
        za = self.encode_graph(data)
        zt = self.encode_text(input_ids, attention_mask)
        return za, zt


def info_nce(za, zt, temperature: float):
    """Symmetric InfoNCE; za,zt are L2-normalized [B,D]. Returns scalar loss."""
    logits = za @ zt.t() / temperature      # [B,B]
    targets = torch.arange(za.size(0), device=za.device)
    loss_a = F.cross_entropy(logits, targets)      # audio->text
    loss_t = F.cross_entropy(logits.t(), targets)  # text->audio
    return 0.5 * (loss_a + loss_t)


@torch.no_grad()
def recall_at_k(sim: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """sim[i,j] = score of query i vs candidate j; diagonal = correct match.
    Returns {'R@1':..,...} averaged over queries (rows)."""
    n = sim.size(0)
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(n, device=sim.device).unsqueeze(1)
    hit_pos = (ranks == gt).nonzero()[:, 1]  # rank index of correct item per row
    out = {}
    for k in ks:
        out[f"R@{k}"] = round(((hit_pos < k).float().mean().item()) * 100, 2)
    return out
