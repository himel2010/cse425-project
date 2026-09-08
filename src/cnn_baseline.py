"""B2 CNN baseline on log-mel (PLAN.md P4.1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """4 conv blocks, global mean pool, linear -> multi-label logits."""

    def __init__(self, n_labels: int, n_mels: int = 128):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(256, n_labels)

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        h = self.features(x)
        h = h.mean(dim=(2, 3))  # global mean pool over freq+time -> [B, 256]
        return self.head(h)
