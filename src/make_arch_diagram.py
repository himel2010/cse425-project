"""Generate a simple architecture diagram PNG for the report (P8).
Recreated schematically from gnn_bert_music_pipeline.excalidraw (no excalidraw
renderer available offline)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from utils import load_config, ensure_dir


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=14, lw=1.2, color="#333"))


def main():
    cfg = load_config()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")

    A = "#AEC7E8"; T = "#FFBB78"; F = "#98DF8A"; H = "#FF9896"
    # audio branch
    box(ax, 0.2, 4.2, 1.7, 0.8, "Audio clip\n(22.05 kHz)", A)
    box(ax, 0.2, 2.9, 1.7, 0.8, "5s/2.5s segments\nMFCC+chroma (64d)", A)
    box(ax, 0.2, 1.6, 1.7, 0.8, "Segment graph\ntemporal+similarity", A)
    box(ax, 2.4, 1.6, 1.9, 0.8, "GraphSAGE\n2 layers, mean pool", F)
    for y in (4.2, 2.9):
        arrow(ax, 1.05, y, 1.05, y - 0.5)
    arrow(ax, 1.9, 2.0, 2.4, 2.0)

    # text branch
    box(ax, 0.2, 0.2, 1.7, 0.8, "Text\n(metadata / caption)", T)
    box(ax, 2.4, 0.2, 1.9, 0.8, "DistilBERT\n(fine-tuned)", T)
    arrow(ax, 1.9, 0.6, 2.4, 0.6)

    # fusion
    box(ax, 4.9, 0.9, 2.2, 1.6, "Fusion\ncross-attention\n(graph q, text k/v)\nz = [g ; ctx]", H)
    arrow(ax, 4.3, 2.0, 4.9, 1.9)
    arrow(ax, 4.3, 0.6, 4.9, 1.1)

    # heads
    box(ax, 7.6, 2.6, 2.2, 0.9, "Task 1-3 head\n50-tag sigmoid (BCE)", "#C5B0D5")
    box(ax, 7.6, 0.6, 2.2, 0.9, "Task 4\nInfoNCE retrieval\n(dual encoder)", "#C5B0D5")
    arrow(ax, 7.1, 1.9, 7.6, 3.0)
    arrow(ax, 4.3, 2.0, 7.6, 3.1)   # gnn -> tag head (task2)
    arrow(ax, 7.1, 1.5, 7.6, 1.1)

    ax.set_title("GNN-BERT music context pipeline", fontsize=12)
    out = ensure_dir(Path(cfg["paths"]["results"]) / "plots") / "architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
