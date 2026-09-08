"""MTAT label/text/split helpers (D4, D5, D6).

annotations_final.csv: TAB-separated. Columns = [clip_id, <188 tag 0/1 cols>, mp3_path].
clip_info_final.csv:   TAB-separated. Columns include [clip_id, TITLE, ARTIST, ALBUM, ...].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, ensure_dir


def load_annotations(cfg: dict) -> pd.DataFrame:
    path = cfg["datasets"]["mtat"]["annotations"]
    df = pd.read_csv(path, sep="\t")
    return df


def top_k_tags(ann: pd.DataFrame, k: int) -> list[str]:
    tag_cols = [c for c in ann.columns if c not in ("clip_id", "mp3_path")]
    counts = ann[tag_cols].sum().sort_values(ascending=False)
    return counts.head(k).index.tolist()


def clip_folder(mp3_path: str) -> str:
    """First path component, e.g. 'f/foo-bar.mp3' -> 'f'."""
    return str(mp3_path).split("/")[0]


def build_labels(cfg: dict) -> dict:
    """Returns dict with tags list, and per clip_id multi-hot + mp3 rel path.

    Also drops clips whose all top-50 tags are zero (uninformative), per common
    MTAT practice — keeps the label space meaningful. Kept clips still guarantee
    the folder split works.
    """
    ann = load_annotations(cfg)
    k = cfg["labels"]["top_k_tags"]
    tags = top_k_tags(ann, k)

    mat = ann[tags].to_numpy(dtype=np.float32)
    keep_mask = mat.sum(axis=1) > 0

    labels = {}
    for i, row in enumerate(ann.itertuples(index=False)):
        if not keep_mask[i]:
            continue
        d = row._asdict()
        cid = int(d["clip_id"])
        labels[cid] = {
            "y": mat[i].tolist(),
            "mp3": d["mp3_path"],
            "folder": clip_folder(d["mp3_path"]),
        }
    return {"tags": tags, "clips": labels}


def build_text(cfg: dict, clip_ids: list[int]) -> dict:
    """D4: 'artist - title (album)' from clip_info_final.csv. Missing -> ''."""
    info = pd.read_csv(cfg["datasets"]["mtat"]["clip_info"], sep="\t")
    info = info.set_index("clip_id")
    # column names in MTAT clip_info are uppercase
    cols = {c.upper(): c for c in info.columns}
    a_col = cols.get("ARTIST")
    t_col = cols.get("TITLE")
    al_col = cols.get("ALBUM")

    out = {}
    for cid in clip_ids:
        if cid not in info.index:
            out[cid] = ""
            continue
        r = info.loc[cid]
        artist = str(r[a_col]) if a_col and pd.notna(r[a_col]) else ""
        title = str(r[t_col]) if t_col and pd.notna(r[t_col]) else ""
        album = str(r[al_col]) if al_col and pd.notna(r[al_col]) else ""
        s = f"{artist} - {title}".strip(" -")
        if album:
            s = f"{s} ({album})"
        out[cid] = s.strip()
    return out


def make_splits(cfg: dict, clips: dict) -> dict:
    m = cfg["datasets"]["mtat"]
    train_f = set(m["train_folders"])
    val_f = set(m["val_folders"])
    test_f = set(m["test_folders"])
    split = {"train": [], "val": [], "test": []}
    for cid, meta in clips.items():
        f = meta["folder"]
        if f in train_f:
            split["train"].append(cid)
        elif f in val_f:
            split["val"].append(cid)
        elif f in test_f:
            split["test"].append(cid)
    for k in split:
        split[k].sort()
    return split


def prepare_all(cfg: dict = None) -> dict:
    """Build labels, splits, text; write to data/processed + data/splits."""
    cfg = cfg or load_config()
    proc = ensure_dir(cfg["paths"]["data_processed"])
    splits_dir = ensure_dir(cfg["paths"]["splits"])

    lab = build_labels(cfg)
    with open(proc / "mtat_labels.json", "w", encoding="utf-8") as f:
        json.dump(lab, f)

    split = make_splits(cfg, lab["clips"])
    for name in ("train", "val", "test"):
        with open(splits_dir / f"mtat_{name}.json", "w", encoding="utf-8") as f:
            json.dump(split[name], f)

    text = build_text(cfg, list(lab["clips"].keys()))
    with open(proc / "mtat_text.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in text.items()}, f)

    print(f"tags={len(lab['tags'])} clips={len(lab['clips'])} "
          f"train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}")
    return {"labels": lab, "splits": split, "text": text}


if __name__ == "__main__":
    prepare_all()
