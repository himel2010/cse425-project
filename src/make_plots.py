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
    """Macro and micro F1 in two panels.

    One explicit colour per model, shared across both panels. The old
    single-axes version relied on the default colour cycle, which wraps after
    ten entries: with six models times two metrics it handed two different
    curves the same colour and the legend became unreadable.
    """
    m = load_metrics()
    series = []
    for key, label in [("task1", "Task 1 BERT"), ("task2", "Task 2 GraphSAGE"),
                       ("baseline_cnn", "B2 CNN (log-mel)")]:
        h = m.get(key, {}).get("history")
        if h:
            series.append((label, h))
    pretty = {"cross_attn": "cross-attention", "concat": "early concat",
              "gnn_only": "graph-only", "bert_only": "text-only"}
    for var, h in sorted(m.get("task3", {}).get("variants", {}).items()):
        if h.get("history"):
            series.append((f"Task 3 {pretty.get(var, var)}", h["history"]))
    if not series:
        print("no history yet; skip f1_curves")
        return

    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for metric, ax, title in [("macro_f1", axes[0], "Validation Macro-F1"),
                              ("micro_f1", axes[1], "Validation Micro-F1")]:
        for i, (label, h) in enumerate(series):
            ax.plot([r["epoch"] for r in h], [r[metric] for r in h],
                    marker="o", ms=3, lw=1.4, color=palette(i % 10), label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 (validation split)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path(cfg["paths"]["plots"]) / "f1_curves.png"
    ensure_dir(out.parent); fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"wrote {out}")


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
    plt.title("Task 3 cross-attention: precision-recall curves, ten most frequent tags")
    plt.legend(fontsize=7); plt.grid(alpha=0.3)
    out = Path(cfg["paths"]["plots"]) / "auc_pr_task3.png"
    ensure_dir(out.parent); plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(); print(f"wrote {out}")


def tag_profile(cfg, data):
    """Per-tag F1 and average precision against test-set tag prevalence.

    This is what makes the Macro/Micro-F1 gap legible: Micro-F1 is dominated by
    the handful of tags that appear on thousands of clips, while Macro-F1 gives
    a tag on 30 clips the same weight as a tag on 900, so the rare end of this
    plot is what holds the headline number down.
    """
    from sklearn.metrics import average_precision_score, f1_score
    y, p, _, tags = data
    thr = cfg["eval"]["threshold"]
    prev = y.sum(axis=0)
    keep = prev > 0
    f1 = np.array([f1_score(y[:, j], p[:, j] >= thr, zero_division=0)
                   for j in range(y.shape[1])])
    ap = np.array([average_precision_score(y[:, j], p[:, j]) if prev[j] > 0 else np.nan
                   for j in range(y.shape[1])])
    print(f"[check] cross_attn test macro-F1 {f1[keep].mean():.4f} "
          f"micro-F1 {f1_score(y, p >= thr, average='micro', zero_division=0):.4f} "
          f"mean AP {np.nanmean(ap):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(prev[keep], f1[keep], s=26, color="#1f77b4", label="F1 at threshold 0.5")
    ax.scatter(prev[keep], ap[keep], s=26, color="#d62728", marker="^",
               alpha=0.75, label="average precision")
    for j in np.argsort(-prev)[:6]:
        ax.annotate(tags[j], (prev[j], f1[j]), fontsize=7.5,
                    xytext=(3, 4), textcoords="offset points")
    for j in np.argsort(prev[keep])[:4]:
        idx = np.flatnonzero(keep)[j]
        ax.annotate(tags[idx], (prev[idx], f1[idx]), fontsize=7.5,
                    xytext=(3, 4), textcoords="offset points")
    ax.axhline(f1[keep].mean(), ls="--", lw=1, color="#444")
    ax.text(prev[keep].max(), f1[keep].mean() + 0.012,
            "unweighted mean over tags", ha="right", fontsize=8, color="#444")
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    dead = int(((f1 == 0) & keep).sum())
    print(f"[check] tags never predicted above threshold: {dead}/{int(keep.sum())}; "
          f"their share of test positives "
          f"{prev[(f1 == 0) & keep].sum() / prev[keep].sum():.3f}")
    ax.set_xlabel("tag prevalence in MTAT test split (clips, log scale)")
    ax.set_ylabel("per-tag score")
    ax.set_title("Task 3 cross-attention: per-tag score vs. tag prevalence")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    out = Path(cfg["paths"]["plots"]) / "tag_profile.png"
    ensure_dir(out.parent); fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"wrote {out}")


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
    # legend counts are over the whole test split, but t-SNE runs on a
    # subsample for tractability; say so on the figure so the two cannot be
    # confused for each other.
    plt.title(f"{title}\n{n} of {len(z)} test clips plotted; "
              "legend counts cover the full test split", fontsize=10)
    out = Path(cfg["paths"]["plots"]) / out_name
    ensure_dir(out.parent); plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(); print(f"wrote {out}")


def tsne_task3(cfg, data):
    _tsne_plot(cfg, data, GENRE_MAP,
               "t-SNE of the Task-3 fused embedding (MTAT test), genre umbrella", "tsne_task3.png")


def tsne_task3_mood(cfg, data):
    _tsne_plot(cfg, data, MOOD_MAP,
               "t-SNE of the Task-3 fused embedding (MTAT test), mood umbrella",
               "tsne_task3_mood.png")


def main():
    cfg = load_config()
    f1_curves(cfg)
    data = _load_task3_test_probs(cfg)
    if data is not None:
        auc_pr_task3(cfg, data)
        tag_profile(cfg, data)
        tsne_task3(cfg, data)
        tsne_task3_mood(cfg, data)
    else:
        print("task3 checkpoint missing; skip AUC-PR + t-SNE")


if __name__ == "__main__":
    main()
