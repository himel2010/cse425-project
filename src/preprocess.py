"""P2 orchestrator — features + graphs, cached & idempotent.

Usage:
  python src/preprocess.py --dataset mtat
  python src/preprocess.py --dataset musiccaps

Design: node-feature extraction (numpy/librosa, no torch) runs in a process pool;
graph assembly + .pt save (torch) runs in the main process. Keeps torch out of
worker imports and keeps graph build cheap.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import audio_features as af
from utils import load_config, ensure_dir, update_metrics


# --- worker: pure numpy/librosa feature extraction ------------------------
def _extract(job) -> tuple[str, bool]:
    audio_path, mel_out, nodes_out, cfg, want_mel = job
    ok = af.process_clip(audio_path, Path(mel_out), Path(nodes_out),
                         cfg, want_mel=want_mel)
    return nodes_out, ok


def _run_serial(jobs) -> set[str]:
    """Single-process extraction. Reliable on a RAM-starved box (one ~400 MB
    python vs several librosa workers that OOM-kill)."""
    done = set()
    total = len(jobs)
    for i, j in enumerate(jobs, 1):
        nodes_out, ok = _extract(j)
        if ok:
            done.add(nodes_out)
        if i % 100 == 0 or i == total:
            print(f"\r  features {i}/{total} ok={len(done)}", end="", flush=True)
    print()
    return done


def _run_pool(jobs, workers: int) -> set[str]:
    """Bounded-memory pool: keep at most ~2*workers futures in flight so the
    16 GB / ~2.5 GB-free box doesn't OOM-kill on librosa worker imports."""
    if workers <= 1:
        return _run_serial(jobs)
    done = set()
    total = len(jobs)
    n_ok = 0
    max_inflight = max(workers * 2, 8)
    # Recycle workers periodically: librosa/numpy leak across many clips and the
    # box has only ~1.5 GB free, so long-lived workers OOM-kill the run.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=40) as ex:
        it = iter(jobs)
        inflight = {}
        # prime
        for _ in range(max_inflight):
            try:
                j = next(it)
            except StopIteration:
                break
            inflight[ex.submit(_extract, j)] = True
        seen = 0
        while inflight:
            for fut in as_completed(list(inflight)):
                del inflight[fut]
                nodes_out, ok = fut.result()
                seen += 1
                if ok:
                    done.add(nodes_out)
                    n_ok += 1
                if seen % 100 == 0 or seen == total:
                    print(f"\r  features {seen}/{total} ok={n_ok}",
                          end="", flush=True)
                try:
                    inflight[ex.submit(_extract, next(it))] = True
                except StopIteration:
                    pass
                break  # re-enter as_completed with refreshed set
    print()
    return done


def _build_graphs(items, cfg, out_dir: Path):
    """items: list of (id, nodes_path, y_or_None). Build+save .pt in main proc."""
    import graph_builder as gb
    import torch

    ensure_dir(out_dir)
    built = 0
    for cid, nodes_path, y in items:
        gpath = out_dir / f"{cid}.pt"
        if gpath.exists():
            built += 1
            continue
        if not Path(nodes_path).exists():
            continue
        nodes = np.load(nodes_path)
        data = gb.build_graph(nodes, y, cfg)
        torch.save(data, gpath)
        built += 1
    return built


def preprocess_mtat(cfg, workers):
    import mtat_data
    prep = mtat_data.prepare_all(cfg)
    labels = prep["labels"]["clips"]

    audio_dir = Path(cfg["datasets"]["mtat"]["audio_dir"])
    mel_dir = ensure_dir(Path(cfg["paths"]["data_processed"]) / "mel")
    nodes_dir = ensure_dir(Path(cfg["paths"]["data_processed"]) / "nodes" / "mtat")
    graph_dir = ensure_dir(Path(cfg["paths"]["data_processed"]) / "graphs" / "mtat")

    jobs, items = [], []
    for cid, meta in labels.items():
        apath = audio_dir / meta["mp3"]
        if not apath.exists():
            continue
        nodes_out = str(nodes_dir / f"{cid}.npy")
        mel_out = str(mel_dir / f"{cid}.npy")
        items.append((cid, nodes_out, meta["y"]))
        # only queue feature extraction for clips not already cached, so a
        # short-lived run spends its time on new work, not re-stat'ing caches
        if not (Path(nodes_out).exists() and Path(mel_out).exists()):
            jobs.append((str(apath), mel_out, nodes_out, cfg, True))

    print(f"MTAT clips with audio: {len(items)} | to extract: {len(jobs)}")
    if jobs:
        _run_pool(jobs, workers)
    built = _build_graphs(items, cfg, graph_dir)

    # copy 20 test-set graphs to results/sample_graphs
    test_ids = json.loads((Path(cfg["paths"]["splits"]) / "mtat_test.json").read_text())
    sample_dir = ensure_dir(Path(cfg["paths"]["results"]) / "sample_graphs")
    copied = 0
    for cid in test_ids:
        src = graph_dir / f"{cid}.pt"
        if src.exists():
            shutil.copy(src, sample_dir / f"{cid}.pt")
            copied += 1
        if copied >= 20:
            break

    rate = built / max(len(items), 1)
    update_metrics("preprocess.mtat", {
        "clips_with_audio": len(jobs),
        "graphs_built": built,
        "success_rate": round(rate, 4),
        "sample_graphs": copied,
    })
    print(f"graphs_built={built}/{len(items)} rate={rate:.3f} samples={copied}")
    if rate < 0.95:
        print("WARNING: <95% processed (PLAN.md P2.2 threshold).")


def preprocess_musiccaps(cfg, workers):
    wav_dir = Path(cfg["datasets"]["musiccaps"]["wav_dir"])
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        print("No MusicCaps wavs found — run download_musiccaps.py first.")
        return

    nodes_dir = ensure_dir(Path(cfg["paths"]["data_processed"]) / "nodes" / "musiccaps")
    graph_dir = ensure_dir(Path(cfg["paths"]["data_processed"]) / "graphs" / "musiccaps")

    jobs, items = [], []
    for w in wavs:
        ytid = w.stem
        nodes_out = str(nodes_dir / f"{ytid}.npy")
        items.append((ytid, nodes_out, None))
        if not Path(nodes_out).exists():  # no mel for musiccaps
            jobs.append((str(w), "", nodes_out, cfg, False))

    print(f"MusicCaps wavs: {len(items)} | to extract: {len(jobs)}")
    if jobs:
        _run_pool(jobs, workers)
    built = _build_graphs(items, cfg, graph_dir)

    # deterministic 90/10 split by sorted ytid (D7)
    ytids = sorted(it[0] for it in items if (graph_dir / f"{it[0]}.pt").exists())
    import random
    rng = random.Random(cfg["datasets"]["musiccaps"]["split_seed"])
    shuffled = ytids[:]
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * cfg["datasets"]["musiccaps"]["test_frac"])
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    splits_dir = ensure_dir(cfg["paths"]["splits"])
    (splits_dir / "musiccaps_train.json").write_text(json.dumps(train))
    (splits_dir / "musiccaps_test.json").write_text(json.dumps(test))

    update_metrics("preprocess.musiccaps", {
        "wavs": len(jobs),
        "graphs_built": built,
        "train": len(train),
        "test": len(test),
    })
    print(f"graphs_built={built} train={len(train)} test={len(test)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mtat", "musiccaps"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    if args.dataset == "mtat":
        preprocess_mtat(cfg, args.workers)
    else:
        preprocess_musiccaps(cfg, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
