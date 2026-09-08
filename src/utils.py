"""Shared helpers: config loading, seeding, paths, metrics store."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike = None) -> dict:
    # Anchor all relative data/results paths to the repo root no matter where
    # a script is launched from (scripts use plain relative paths from config).
    os.chdir(REPO_ROOT)
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- metrics.json single store (PLAN.md) ---------------------------------
def metrics_path() -> Path:
    return REPO_ROOT / "results" / "metrics.json"


def load_metrics() -> dict:
    p = metrics_path()
    if p.exists() and p.stat().st_size > 0:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}  # empty/corrupt (e.g. concurrent write) -> start fresh
    return {}


def save_metrics(metrics: dict) -> None:
    p = metrics_path()
    ensure_dir(p.parent)
    # write-and-replace so a reader never sees a half-written file
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    tmp.replace(p)


def update_metrics(dotted_key: str, value) -> dict:
    """Set nested key like 'task1.test' = value; persist; return full dict."""
    m = load_metrics()
    node = m
    parts = dotted_key.split(".")
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value
    save_metrics(m)
    return m
