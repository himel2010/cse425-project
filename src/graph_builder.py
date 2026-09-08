"""P2.2 — Build a PyG graph per clip from its node-feature matrix.

Edges: (1) temporal bidirectional (i,i+1); (2) similarity cosine>tau, non-adjacent,
max 4 extra per node (highest sim), edge weight = sim.
"""
from __future__ import annotations

import numpy as np


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    xn = x / n
    return xn @ xn.T


def build_edges(nodes: np.ndarray, tau: float, max_extra: int):
    """Return (edge_index [2,E], edge_attr [E]) as numpy arrays."""
    n = nodes.shape[0]
    src, dst, w = [], [], []

    # temporal bidirectional
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        w += [1.0, 1.0]

    if n > 2:
        sim = _cosine_matrix(nodes)
        for i in range(n):
            cands = []
            for j in range(n):
                if j == i or abs(i - j) == 1:
                    continue
                if sim[i, j] > tau:
                    cands.append((sim[i, j], j))
            cands.sort(reverse=True)
            for s, j in cands[:max_extra]:
                src.append(i)
                dst.append(j)
                w.append(float(s))

    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = np.asarray(w, dtype=np.float32)
    return edge_index, edge_attr


def build_graph(nodes: np.ndarray, y, cfg: dict):
    """Create a torch_geometric Data object. y may be None (MusicCaps)."""
    import torch
    from torch_geometric.data import Data

    g = cfg["graph"]
    edge_index, edge_attr = build_edges(nodes, g["sim_threshold"],
                                        g["max_extra_edges_per_node"])
    data = Data(
        x=torch.from_numpy(nodes),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    if y is not None:
        data.y = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(0)  # [1,50]
    # sanity: no isolated nodes (temporal edges guarantee for n>1)
    if nodes.shape[0] > 1:
        assert edge_index.shape[1] > 0
    return data
