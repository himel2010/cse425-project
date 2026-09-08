"""GraphSAGE encoder + classifier (D9, spec §4.2)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE, mean readout -> graph embedding [B, hidden]."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.dropout(h, self.dropout, self.training)
        h = self.conv2(h, edge_index)
        return global_mean_pool(h, batch)  # [B, hidden]


class GNNClassifier(nn.Module):
    """Task 2: GraphSAGE encoder + linear head -> multi-label logits."""

    def __init__(self, in_dim: int, hidden: int, n_labels: int,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden, dropout)
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, data) -> torch.Tensor:
        g = self.encoder(data.x, data.edge_index, data.batch)
        return self.head(g)  # logits [B, n_labels]
