"""P7 analysis artifacts: B1 random baseline, example predictions, retrieval
examples, case studies, human-eval sheet, comparison table.

  python src/evaluate.py --what all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config, set_seed, ensure_dir, load_metrics, update_metrics
from metrics import random_baseline_metrics
from data_loaders import MTATDataset, MusicCapsDataset, make_collate
from bert_encoder import get_tokenizer


def _mtat_test_labels(cfg):
    lab = json.loads(
        (Path(cfg["paths"]["data_processed"]) / "mtat_labels.json").read_text())
    ids = json.loads((Path(cfg["paths"]["splits"]) / "mtat_test.json").read_text())
    ids = [c for c in ids if str(c) in lab["clips"]]
    Y = np.array([lab["clips"][str(c)]["y"] for c in ids], dtype=np.float32)
    return ids, Y, lab["tags"]


def b1_random(cfg):
    _, Y, _ = _mtat_test_labels(cfg)
    m = random_baseline_metrics(Y)
    update_metrics("baseline_random.test", m)
    print(f"B1 random: {m}")


def task1_examples(cfg, dev, n=5):
    from bert_encoder import BertClassifier
    tags = _mtat_test_labels(cfg)[2]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    ds = MTATDataset(cfg, "test")
    # seeded random pick of n distinct test clips (avoids 5x duplicate prefixes)
    import random
    rng = random.Random(cfg["seed"])
    idxs = rng.sample(range(len(ds)), k=min(n, len(ds)))
    rows = [ds[i] for i in idxs]
    collate = make_collate(tok, cfg["text"]["max_len_metadata"])
    g, ids, mask, y, cids = collate(rows)
    model = BertClassifier(cfg["text"]["bert_model"], len(tags)).to(dev)
    model.load_state_dict(torch.load(Path(cfg["paths"]["checkpoints"]) / "task1.pt"))
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(ids.to(dev), mask.to(dev))).cpu().numpy()
    lines = ["# Task 1 — 5 example predictions\n"]
    for k in range(min(n, len(cids))):
        txt = ds.text.get(str(cids[k]), "")
        true = [tags[j] for j in np.where(y[k].numpy() > 0.5)[0]]
        pred = [tags[j] for j in np.argsort(-prob[k])[:5]]
        lines += [f"## clip {cids[k]}",
                  f"- **text:** `{txt}`",
                  f"- **true tags:** {', '.join(true) or '(none)'}",
                  f"- **top-5 predicted:** {', '.join(pred)}\n"]
    out = Path(cfg["paths"]["results"]) / "task1_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def _load_dual(cfg, dev):
    from contrastive import DualEncoder
    model = DualEncoder(cfg).to(dev)
    model.load_state_dict(torch.load(Path(cfg["paths"]["checkpoints"]) / "task4.pt"))
    model.eval()
    return model


def retrieval_examples(cfg, dev, n=10):
    model = _load_dual(cfg, dev)
    tok = get_tokenizer(cfg["text"]["bert_model"])
    ds = MusicCapsDataset(cfg, "test")
    dl = DataLoader(ds, batch_size=32,
                    collate_fn=make_collate(tok, cfg["text"]["max_len_caption"]))
    za, zt, ids = [], [], []
    with torch.no_grad():
        for g, i_ids, mask, _, bids in dl:
            za.append(model.encode_graph(g.to(dev)).cpu())
            zt.append(model.encode_text(i_ids.to(dev), mask.to(dev)).cpu())
            ids += list(bids)
    za, zt = torch.cat(za), torch.cat(zt)
    cap = ds.caption
    sim = zt @ za.t()  # caption -> audio
    top = sim.argsort(dim=1, descending=True)[:, :3]

    rng = np.random.default_rng(cfg["seed"])
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    lines = ["# 10 retrieval examples (caption query -> top-3 audio clips)\n"]
    rows = []
    for qi in pick:
        q = ids[qi]
        lines.append(f"## query: `{cap.get(q,'')[:120]}`  (ytid={q})")
        for rank, ai in enumerate(top[qi].tolist(), 1):
            a = ids[ai]
            mark = " ✅(correct)" if a == q else ""
            lines.append(f"  {rank}. ytid={a} sim={sim[qi,ai]:.3f}{mark} — "
                         f"`{cap.get(a,'')[:100]}`")
        lines.append("")
        rows.append((q, cap.get(q, ""), ids[top[qi][0].item()]))
    outdir = ensure_dir(Path(cfg["paths"]["results"]) / "retrieval_examples")
    (outdir / "retrieval_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {outdir/'retrieval_examples.md'}")
    return rows


def human_eval_sheet(cfg, rows):
    import csv
    outdir = Path(cfg["paths"]["results"])
    with open(outdir / "human_eval_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "caption", "top1_ytid", "rating_1to5"])
        for i, (q, capt, top1) in enumerate(rows, 1):
            w.writerow([i, capt, top1, ""])
    (outdir / "human_eval_instructions.md").write_text(
        "# Human eval (self-rating, 15 min)\n\n"
        "Open `human_eval_sheet.csv`. For each row, listen to the top-1 retrieved "
        "clip (YouTube: https://www.youtube.com/watch?v=<top1_ytid>) and rate how "
        "well it matches the caption on a **1-5** scale (1=unrelated, 5=excellent). "
        "Fill the `rating_1to5` column, save, and tell the agent.\n",
        encoding="utf-8")
    print("wrote human_eval_sheet.csv + instructions (HANDOFF TO USER)")


def human_eval_ingest(cfg):
    """Read filled human_eval_sheet.csv -> task4.human_eval in metrics.json."""
    import csv
    import statistics
    sheet = Path(cfg["paths"]["results"]) / "human_eval_sheet.csv"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    scores = [float(r["rating_1to5"]) for r in rows
              if str(r.get("rating_1to5", "")).strip()]
    if len(scores) < 10:
        print(f"SKIP human_eval: only {len(scores)}/10 ratings filled in {sheet}")
        return None
    m = {"n": len(scores), "mean": round(statistics.mean(scores), 2),
         "sd": round(statistics.stdev(scores), 2), "scores": scores}
    update_metrics("task4.human_eval", m)
    print(f"human eval: {m}")
    return m


def case_studies(cfg, dev, n=3):
    model = _load_dual(cfg, dev)
    tok = get_tokenizer(cfg["text"]["bert_model"])
    ds = MusicCapsDataset(cfg, "test")
    dl = DataLoader(ds, batch_size=32,
                    collate_fn=make_collate(tok, cfg["text"]["max_len_caption"]))
    za, zt, ids, graphs = [], [], [], {}
    with torch.no_grad():
        for g, i_ids, mask, _, bids in dl:
            za.append(model.encode_graph(g.to(dev)).cpu())
            zt.append(model.encode_text(i_ids.to(dev), mask.to(dev)).cpu())
            ids += list(bids)
    za, zt = torch.cat(za), torch.cat(zt)
    sim = zt @ za.t()
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(len(ids)).unsqueeze(1)
    rank_pos = (ranks == gt).nonzero()[:, 1]
    good = [i for i in range(len(ids)) if rank_pos[i] < 5][:n]

    gdir = Path(cfg["paths"]["data_processed"]) / "graphs" / "musiccaps"
    lines = ["# 3 case studies (well-retrieved clips)\n"]
    for qi in good:
        ytid = ids[qi]
        gd = torch.load(gdir / f"{ytid}.pt", weights_only=False)
        n_nodes = gd.x.size(0)
        n_edges = gd.edge_index.size(1)
        temporal = 2 * (n_nodes - 1)
        sim_edges = n_edges - temporal
        lines += [f"## ytid={ytid} (retrieval rank {rank_pos[qi].item()+1})",
                  f"- caption: `{ds.caption.get(ytid,'')}`",
                  f"- graph: {n_nodes} nodes, {n_edges} edges "
                  f"({temporal} temporal + {sim_edges} similarity/repeated-segment)",
                  ""]
    (Path(cfg["paths"]["results"]) / "case_studies.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("wrote case_studies.md")


def comparison_table(cfg):
    m = load_metrics()
    def row(name, d):
        if not d:
            return f"| {name} | - | - | - |"
        return (f"| {name} | {d.get('macro_f1','-')} | "
                f"{d.get('micro_f1','-')} | {d.get('mean_auc_pr','-')} |")
    lines = ["# Model comparison (MTAT test)\n",
             "| Model | Macro-F1 | Micro-F1 | mean AUC-PR |",
             "|---|---|---|---|",
             row("B1 random", m.get("baseline_random", {}).get("test")),
             row("B2 CNN", m.get("baseline_cnn", {}).get("test")),
             row("B3/Task1 BERT", m.get("task1", {}).get("test")),
             row("Task2 GNN", m.get("task2", {}).get("test"))]
    v = m.get("task3", {}).get("variants", {})
    for name in ("bert_only", "gnn_only", "concat", "cross_attn"):
        lines.append(row(f"Task3 {name}", v.get(name, {}).get("test")))
    ret = m.get("task4", {}).get("retrieval", {})
    if ret:
        lines += ["", "## Task 4 retrieval (MusicCaps test)",
                  f"- Caption->Audio: {ret.get('caption_to_audio')}",
                  f"- Audio->Caption: {ret.get('audio_to_caption')}",
                  f"- Zero-shot tag probe: {m.get('task4',{}).get('zero_shot')}"]
    out = Path(cfg["paths"]["results"]) / "comparison_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "b1", "task1_examples", "retrieval",
                             "case_studies", "table", "human_eval"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.what in ("all", "b1"):
        b1_random(cfg)
    if args.what in ("all", "task1_examples"):
        task1_examples(cfg, dev)
    rows = None
    if args.what in ("all", "retrieval"):
        rows = retrieval_examples(cfg, dev)
        human_eval_sheet(cfg, rows)
    if args.what in ("all", "case_studies"):
        case_studies(cfg, dev)
    if args.what in ("all", "table"):
        comparison_table(cfg)
    if args.what in ("all", "human_eval"):
        human_eval_ingest(cfg)  # skips gracefully until sheet is filled


if __name__ == "__main__":
    main()
