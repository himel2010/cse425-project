"""P7 plots: F1-vs-epoch curves, AUC-PR curves, t-SNE of Task-3 fused z.

  python src/make_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config, load_metrics, ensure_dir

DPI = 200

# top-50 tag -> 8 genre umbrellas for t-SNE coloring (P7.4)
GENRE_MAP = {
    "classical": "classical", "guitar": "acoustic", "slow": "ambient",
    "techno": "electronic", "strings": "classical", "drums": "rock",
    "electronic": "electronic", "rock": "rock", "fast": "rock",
    "piano": "classical", "ambient": "ambient", "beat": "electronic",
    "violin": "classical", "vocal": "vocal", "synth": "electronic",
    "female": "vocal", "indian": "world", "opera": "vocal", "male": "vocal",
    "singing": "vocal", "vocals": "vocal", "no vocals": "instrumental",
    "harpsichord": "classical", "loud": "rock", "quiet": "ambient",
    "flute": "classical", "woman": "vocal", "male vocal": "vocal",
    "no vocal": "instrumental", "pop": "pop", "soft": "ambient",
    "sitar": "world", "solo": "acoustic", "man": "vocal", "classic": "classical",
    "choir": "vocal", "voice": "vocal", "new age": "ambient", "dance": "electronic",
    "male voice": "vocal", "female vocal": "vocal", "beats": "electronic",
    "harp": "classical", "cello": "classical", "no voice": "instrumental",
    "weird": "electronic", "country": "acoustic", "metal": "rock",
    "female voice": "vocal", "choral": "vocal",
}

# top-50 tag -> mood umbrella for the mood t-SNE (spec §4.3: genre AND mood)
MOOD_MAP = {
    "slow": "calm", "soft": "calm", "quiet": "calm", "ambient": "calm",
    "new age": "calm", "classical": "calm", "classic": "calm", "harp": "calm",
    "harpsichord": "calm", "cello": "calm", "flute": "calm", "piano": "calm",
    "strings": "calm", "violin": "calm", "choir": "calm", "choral": "calm",
    "fast": "energetic", "loud": "energetic", "techno": "energetic",
    "electronic": "energetic", "dance": "energetic", "beat": "energetic",
    "beats": "energetic", "metal": "energetic", "rock": "energetic",
    "drums": "energetic", "guitar": "energetic",
    "weird": "dark", "opera": "dark", "sitar": "dark", "indian": "dark",
    "pop": "upbeat", "country": "upbeat",
}


def f1_curves(cfg):
    m = load_metrics()
    plt.figure(figsize=(8, 5))
    plotted = False
    for key, label in [("task1", "Task1 BERT"), ("task2", "Task2 GNN")]:
        h = m.get(key, {}).get("history")
        if h:
            ep = [r["epoch"] for r in h]
            plt.plot(ep, [r["macro_f1"] for r in h], marker="o",
                     label=f"{label} macro-F1")
            plt.plot(ep, [r["micro_f1"] for r in h], linestyle="--",
                     alpha=0.6, label=f"{label} micro-F1")
            plotted = True
    for var, h in m.get("task3", {}).get("variants", {}).items():
        hist = h.get("history")
        if hist:
            ep = [r["epoch"] for r in hist]
            plt.plot(ep, [r["macro_f1"] for r in hist], marker=".",
                     label=f"Task3 {var} macro-F1")
            plt.plot(ep, [r["micro_f1"] for r in hist], linestyle="--",
                     alpha=0.6, label=f"Task3 {var} micro-F1")
            plotted = True
    if not plotted:
        print("no history yet; skip f1_curves")
        return
    plt.xlabel("epoch"); plt.ylabel("val F1"); plt.legend(fontsize=7)
    plt.title("Macro/Micro-F1 vs epoch"); plt.grid(alpha=0.3)
    out = Path(cfg["paths"]["plots"]) / "f1_curves.png"
    ensure_dir(out.parent); plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(); print(f"wrote {out}")


def _load_task3_test_probs(cfg):
    """Recompute cross_attn test probs + labels for AUC-PR / t-SNE."""
    import torch
    from torch.utils.data import DataLoader
    from fusion_model import FusionModel
    from bert_encoder import get_tokenizer
    from data_loaders import MTATDataset, make_collate

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags = json.loads(
        (Path(cfg["paths"]["data_processed"]) / "mtat_labels.json").read_text())["tags"]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    ds = MTATDataset(cfg, "test")
    dl = DataLoader(ds, batch_size=16,
                    collate_fn=make_collate(tok, cfg["text"]["max_len_metadata"]))
    model = FusionModel(cfg, len(tags), "cross_attn").to(dev)
    ckpt = Path(cfg["paths"]["checkpoints"]) / "task3_cross_attn.pt"
    if not ckpt.exists():
        return None
    model.load_state_dict(torch.load(ckpt)); model.eval()

    # hook the pre-head fused z for t-SNE
    feats, probs, ys = [], [], []
    z_store = {}
    h = model.head[0].register_forward_hook(
        lambda mod, inp, out: z_store.__setitem__("z", inp[0].detach().cpu()))
    with torch.no_grad():
        for g, ids, mask, y, _ in dl:
            logit = model(g.to(dev), ids.to(dev), mask.to(dev))
            probs.append(torch.sigmoid(logit).cpu().numpy())
            feats.append(z_store["z"].numpy())
            ys.append(y.numpy())
    h.remove()
    return (np.concatenate(ys), np.concatenate(probs),
            np.concatenate(feats), tags)


def auc_pr_task3(cfg, data):
    from sklearn.metrics import precision_recall_curve
    y, p, _, tags = data
    order = np.argsort(-y.sum(axis=0))[:10]  # top-10 frequent tags
    plt.figure(figsize=(8, 6))
    for j in order:
        if y[:, j].sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(y[:, j], p[:, j])
        plt.plot(rc, pr, label=tags[j])
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("Task3 cross-attn: PR curves (top-10 tags)")
    plt.legend(fontsize=7); plt.grid(alpha=0.3)
    out = Path(cfg["paths"]["plots"]) / "auc_pr_task3.png"
    ensure_dir(out.parent); plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(); print(f"wrote {out}")


def _tsne_plot(cfg, data, mapping, title: str, out_name: str):
    from sklearn.manifold import TSNE
    y, _, z, tags = data
    # dominant tag per clip -> umbrella label
    dom = y.argmax(axis=1)
    labels = [mapping.get(tags[d], "other") for d in dom]
    uniq = sorted(set(labels))
    cmap = {g: i for i, g in enumerate(uniq)}
    colors = [cmap[g] for g in labels]

    n = min(len(z), 1500)
    idx = np.random.default_rng(cfg["seed"]).choice(len(z), n, replace=False)
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=cfg["seed"]).fit_transform(z[idx])
    # one palette for both the points and the legend: indexing the colormap by
    # category directly (not via a normalised scalar) keeps the two in sync.
    palette = plt.get_cmap("tab10")
    pt_colors = [palette(c % 10) for c in np.array(colors)[idx]]
    plt.figure(figsize=(9, 7))
    plt.scatter(emb[:, 0], emb[:, 1], c=pt_colors, s=8, alpha=0.7)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=palette(cmap[g] % 10),
                          label=f"{g} ({labels.count(g)})") for g in uniq]
    plt.legend(handles=handles, fontsize=8)
    plt.title(title)
    out = Path(cfg["paths"]["plots"]) / out_name
    ensure_dir(out.parent); plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(); print(f"wrote {out}")


def tsne_task3(cfg, data):
    _tsne_plot(cfg, data, GENRE_MAP,
               "t-SNE of Task-3 fused embedding z (MTAT test)", "tsne_task3.png")


def tsne_task3_mood(cfg, data):
    _tsne_plot(cfg, data, MOOD_MAP,
               "t-SNE of Task-3 fused embedding z (mood umbrella)",
               "tsne_task3_mood.png")


def main():
    cfg = load_config()
    f1_curves(cfg)
    data = _load_task3_test_probs(cfg)
    if data is not None:
        auc_pr_task3(cfg, data)
        tsne_task3(cfg, data)
        tsne_task3_mood(cfg, data)
    else:
        print("task3 checkpoint missing; skip AUC-PR + t-SNE")


if __name__ == "__main__":
    main()
