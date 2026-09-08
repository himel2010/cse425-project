"""Dataset + collate helpers for all tasks.

Graphs are cached .pt (PyG Data). Text tokenized in collate (short strings).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from utils import load_config


def _load_split(cfg, name: str) -> list:
    p = Path(cfg["paths"]["splits"]) / f"{name}.json"
    return json.loads(p.read_text())


class MTATDataset(Dataset):
    """Returns (graph Data with .y, text str, clip_id) for a split."""

    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.graph_dir = Path(cfg["paths"]["data_processed"]) / "graphs" / "mtat"
        ids = _load_split(cfg, f"mtat_{split}")
        self.text = json.loads(
            (Path(cfg["paths"]["data_processed"]) / "mtat_text.json").read_text())
        # keep only ids whose graph exists
        self.ids = [c for c in ids if (self.graph_dir / f"{c}.pt").exists()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        g = torch.load(self.graph_dir / f"{cid}.pt", weights_only=False)
        txt = self.text.get(str(cid), "")
        return g, txt, cid


class MusicCapsDataset(Dataset):
    """Returns (graph Data, caption str, ytid). Captions from musiccaps csv."""

    def __init__(self, cfg, split: str):
        import pandas as pd
        self.graph_dir = Path(cfg["paths"]["data_processed"]) / "graphs" / "musiccaps"
        ids = _load_split(cfg, f"musiccaps_{split}")
        df = pd.read_csv(cfg["datasets"]["musiccaps"]["csv"]).set_index("ytid")
        self.caption = df["caption"].to_dict()
        self.ids = [y for y in ids if (self.graph_dir / f"{y}.pt").exists()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ytid = self.ids[i]
        g = torch.load(self.graph_dir / f"{ytid}.pt", weights_only=False)
        return g, self.caption.get(ytid, ""), ytid


def make_collate(tokenizer, max_len: int):
    """Collate (graph, text, id) -> (Batch, input_ids, attn_mask, y, ids)."""
    def collate(items):
        graphs, texts, ids = zip(*items)
        batch = Batch.from_data_list(list(graphs))
        tok = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        y = batch.y if hasattr(batch, "y") and batch.y is not None else None
        return batch, tok["input_ids"], tok["attention_mask"], y, ids
    return collate
