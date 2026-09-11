"""Generate the pipeline architecture figure for the report (P8).

Two vertical lanes, audio on the left and text on the right, meeting in a
fusion block that feeds one head to each side. Every connector is either
vertical inside a lane, a short symmetric diagonal into the fusion block, or a
horizontal run out to a head, so no arrow crosses a box. The four Task-3
variants are listed inside the fusion block instead of being drawn as bypass
arrows, which also matches the code: they are settings of one `FusionModel`,
not separate paths.

The aspect ratio is deliberately close to 3:2. An earlier wide version scaled
down to about a third of its size inside a NeurIPS column and the box text
became unreadable.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from utils import load_config, ensure_dir

AUDIO = "#AEC7E8"
TEXT = "#FFBB78"
ENC = "#98DF8A"
FUSE = "#FF9896"
HEAD = "#C5B0D5"

G_DIM = r"$g \in \mathbb{R}^{256}$"
T_CLS = r"$t_{\mathrm{CLS}} \in \mathbb{R}^{768}$"
H_DIM = r"$H \in \mathbb{R}^{L \times 768}$"
CTX = r"$\mathrm{ctx}=\mathrm{softmax}(qK^{\top}\!/\sqrt{d})\,V$"
QKV = r"$q=gW_Q$,   $K,V=HW$,   $z=[\,g\,;\,\mathrm{ctx}\,]$"


def box(ax, x, y, w, h, text, color, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=color, ec="#222", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color="#333",
                                 shrinkA=0, shrinkB=0))


def main():
    cfg = load_config()
    W, H = 9.5, 6.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    bw, bh = 3.6, 1.15
    lx, rx = 0.3, 5.6
    rows = [5.30, 3.70, 2.10]

    audio = [
        "Audio clip\nmono, 22.05 kHz\n29 s (MTAT), 10 s (MusicCaps)",
        "Segment windows\n5 s window, 2.5 s hop\nMFCC-20 + chroma-12 (64-d)",
        "Segment graph + GraphSAGE\ntemporal and similarity edges\n"
        "2 layers, mean readout, " + G_DIM,
    ]
    text = [
        "Text input\nMTAT: artist - title (album)\nMusicCaps: free-text caption",
        "DistilBERT\n6 layers, fully fine-tuned\nmax 64 (metadata), 128 (caption)",
        "Text states\n" + T_CLS + ",   " + H_DIM,
    ]
    for i, y in enumerate(rows):
        box(ax, lx, y, bw, bh, audio[i], ENC if i == 2 else AUDIO)
        box(ax, rx, y, bw, bh, text[i], ENC if i == 2 else TEXT)
        if i < 2:
            arrow(ax, lx + bw / 2, y, lx + bw / 2, rows[i + 1] + bh)
            arrow(ax, rx + bw / 2, y, rx + bw / 2, rows[i + 1] + bh)

    fx, fy, fw, fh = 2.60, 0.45, 4.30, 1.45
    box(ax, fx, fy, fw, fh,
        "Fusion block (Task 3)\n" + CTX + "\n" + QKV +
        "\nvariants: cross-attention, early concat,\ngraph-only, text-only",
        FUSE, fontsize=9)
    arrow(ax, lx + bw / 2, rows[2], fx + 0.55, fy + fh)
    arrow(ax, rx + bw / 2, rows[2], fx + fw - 0.55, fy + fh)

    hw = 2.00
    box(ax, 0.30, fy, hw, fh,
        "Tagging head\n50 sigmoids, BCE\nTasks 1-3", HEAD, fontsize=9)
    box(ax, 7.20, fy, hw, fh,
        "Projection head\n256-d, L2-normalised\nInfoNCE, Task 4", HEAD, fontsize=9)
    arrow(ax, fx, fy + fh / 2, 0.30 + hw, fy + fh / 2)
    arrow(ax, fx + fw, fy + fh / 2, 7.20, fy + fh / 2)

    ax.set_title("GNN-BERT music context pipeline", fontsize=13, pad=4)
    out = ensure_dir(Path(cfg["paths"]["results"]) / "plots") / "architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
