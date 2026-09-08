"""Mel-spectrogram dataset for the CNN baseline (B2). Pads/crops T to a fixed len."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FIXED_T = 256  # ~ covers 29 s at hop 512 / sr 22050 is ~1250; we crop for memory


class MelDataset(Dataset):
    def __init__(self, cfg, split: str):
        self.mel_dir = Path(cfg["paths"]["data_processed"]) / "mel"
        ids = json.loads((Path(cfg["paths"]["splits"]) / f"mtat_{split}.json").read_text())
        lab = json.loads(
            (Path(cfg["paths"]["data_processed"]) / "mtat_labels.json").read_text())["clips"]
        self.items = [(c, lab[str(c)]["y"]) for c in ids
                      if str(c) in lab and (self.mel_dir / f"{c}.npy").exists()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, y = self.items[i]
        mel = np.load(self.mel_dir / f"{cid}.npy")  # [n_mels, T]
        T = mel.shape[1]
        if T >= FIXED_T:
            mel = mel[:, :FIXED_T]
        else:
            mel = np.pad(mel, ((0, 0), (0, FIXED_T - T)))
        x = torch.from_numpy(mel).unsqueeze(0).float()  # [1, n_mels, T]
        return x, torch.tensor(y, dtype=torch.float32)


def mel_collate(items):
    xs, ys = zip(*items)
    return torch.stack(xs), torch.stack(ys)
