"""P2.1 — Per-clip audio features: log-mel (CNN) + per-segment node features (GNN).

log-mel: [128, T] normalized -> data/processed/mel/{id}.npy
nodes:   [n_windows, 64]     -> data/processed/nodes/{id}.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str, sr: int, mono: bool = True) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=mono)
    return y


def log_mel(y: np.ndarray, cfg: dict) -> np.ndarray:
    import librosa
    a = cfg["audio"]
    S = librosa.feature.melspectrogram(
        y=y, sr=a["sample_rate"], n_fft=a["n_fft"],
        hop_length=a["hop_length"], n_mels=a["n_mels"],
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu, sd = S_db.mean(), S_db.std()
    if sd < 1e-6:
        sd = 1.0
    return ((S_db - mu) / sd).astype(np.float32)  # [n_mels, T]


def _segment_bounds(n_samples: int, sr: int, cfg: dict) -> list[tuple[int, int]]:
    a = cfg["audio"]
    win = int(a["segment_seconds"] * sr)
    hop = int(a["segment_hop_seconds"] * sr)
    min_partial = int(a["min_partial_window_seconds"] * sr)
    bounds = []
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        if end - start >= min_partial:
            bounds.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return bounds


def node_features(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Per segment: MFCC-20 (mean+std) + chroma-12 (mean+std) = 64 dims."""
    import librosa
    a = cfg["audio"]
    sr = a["sample_rate"]
    feats = []
    for s, e in _segment_bounds(len(y), sr, cfg):
        seg = y[s:e]
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=a["n_mfcc"])
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, n_chroma=a["n_chroma"])
        vec = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
        ]).astype(np.float32)
        feats.append(vec)
    if not feats:
        return np.zeros((1, a["node_feat_dim"]), dtype=np.float32)
    arr = np.stack(feats)  # [n_windows, 64]
    assert arr.shape[1] == a["node_feat_dim"], arr.shape
    return arr


def process_clip(audio_path: str, mel_out: Path, nodes_out: Path,
                 cfg: dict, want_mel: bool = True) -> bool:
    """Compute + cache features for one clip. Idempotent (skips if cached)."""
    if nodes_out.exists() and (not want_mel or mel_out.exists()):
        return True
    try:
        y = load_audio(audio_path, cfg["audio"]["sample_rate"],
                       cfg["audio"]["mono"])
        if len(y) < cfg["audio"]["sample_rate"]:  # < 1 s -> junk
            return False
        nodes = node_features(y, cfg)
        nodes_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(nodes_out, nodes)
        if want_mel:
            mel = log_mel(y, cfg)
            mel_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_out, mel)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  feature fail {audio_path}: {e}")
        return False
