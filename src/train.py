"""Unified training entry point for all tasks (PLAN.md P3-P6).

  python src/train.py --task 1
  python src/train.py --task 2
  python src/train.py --task cnn
  python src/train.py --task 3 --variant cross_attn
  python src/train.py --task 4

Uses AMP + gradient accumulation (config micro_batch * grad_accum) to fit 4 GB GPU.
Best-val checkpoint saved; per-epoch history + test metrics -> results/metrics.json.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config, set_seed, ensure_dir, update_metrics
from metrics import multilabel_metrics
from data_loaders import MTATDataset, MusicCapsDataset, make_collate
from bert_encoder import BertClassifier, get_tokenizer


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_labels(cfg):
    return cfg["labels"]["top_k_tags"]


# ---- generic multi-label train loop (tasks 1,2,cnn,3) --------------------
def _predict_probs(model, loader, dev, forward):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            logits, y = forward(model, batch, dev)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _fit(model, train_loader, val_loader, test_loader, dev, forward,
         param_groups, epochs, grad_accum, ckpt_path, thr, sched_kind="cosine"):
    opt = torch.optim.AdamW(param_groups)
    steps = math.ceil(len(train_loader) / grad_accum) * epochs
    if sched_kind == "cosine":
        warmup = int(0.05 * steps)
        def lr_lambda(s):
            if s < warmup:
                return s / max(1, warmup)
            prog = (s - warmup) / max(1, steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sched = None
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss()

    # per-epoch resume: the environment kills long CPU/GPU runs, so persist full
    # training state each epoch and continue from it on relaunch.
    resume_path = Path(str(ckpt_path) + ".resume")
    best_macro, history, start_ep = -1.0, [], 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if sched and st.get("sched"):
            sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best_macro = st["best_macro"]
        history = st["history"]
        start_ep = st["epoch"]
        print(f"  resumed from epoch {start_ep}/{epochs} (best_macro={best_macro})")

    for ep in range(start_ep, epochs):
        model.train()
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                logits, y = forward(model, batch, dev)
                loss = bce(logits, y) / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if sched:
                    sched.step()
        yv, pv = _predict_probs(model, val_loader, dev, forward)
        vm = multilabel_metrics(yv, pv, thr)
        history.append({"epoch": ep + 1, **vm})
        print(f"  ep{ep+1}/{epochs} val macroF1={vm['macro_f1']} "
              f"microF1={vm['micro_f1']} aucpr={vm['mean_auc_pr']}")
        if vm["macro_f1"] > best_macro:
            best_macro = vm["macro_f1"]
            ensure_dir(Path(ckpt_path).parent)
            torch.save(model.state_dict(), ckpt_path)
        # checkpoint full state so a kill mid-training resumes at next epoch
        torch.save({
            "epoch": ep + 1, "best_macro": best_macro, "history": history,
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched else None,
            "scaler": scaler.state_dict(),
        }, resume_path)

    model.load_state_dict(torch.load(ckpt_path))
    yt, pt = _predict_probs(model, test_loader, dev, forward)
    test_m = multilabel_metrics(yt, pt, thr)
    print(f"  TEST macroF1={test_m['macro_f1']} microF1={test_m['micro_f1']} "
          f"aucpr={test_m['mean_auc_pr']}")
    resume_path.unlink(missing_ok=True)  # done cleanly; don't resume next time
    return history, test_m


def _loaders(cfg, tok, max_len, batch, dataset_cls=MTATDataset):
    collate = make_collate(tok, max_len)
    def mk(split, shuffle):
        return DataLoader(dataset_cls(cfg, split), batch_size=batch,
                          shuffle=shuffle, collate_fn=collate, num_workers=0)
    return mk


# ---- task-specific forward fns -------------------------------------------
def task1(cfg, dev):
    tc = cfg["train"]["task1"]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    mk = _loaders(cfg, tok, cfg["text"]["max_len_metadata"], tc["micro_batch"])
    model = BertClassifier(cfg["text"]["bert_model"], n_labels(cfg)).to(dev)

    def fwd(m, batch, dev):
        _, ids, mask, y, _ = batch
        return m(ids.to(dev), mask.to(dev)), y.to(dev)

    pg = [{"params": model.parameters(), "lr": tc["lr_bert"]}]
    ck = str(Path(cfg["paths"]["checkpoints"]) / "task1.pt")
    return _fit(model, mk("train", True), mk("val", False), mk("test", False),
                dev, fwd, pg, tc["epochs"], tc["grad_accum"], ck,
                cfg["eval"]["threshold"], "cosine"), "task1"


def task2(cfg, dev):
    from gnn_model import GNNClassifier
    tc = cfg["train"]["task2"]
    tok = get_tokenizer(cfg["text"]["bert_model"])  # unused text, fine
    mk = _loaders(cfg, tok, 8, tc["batch"])
    model = GNNClassifier(cfg["audio"]["node_feat_dim"], cfg["model"]["hidden_dim"],
                          n_labels(cfg), cfg["model"]["dropout"]).to(dev)

    def fwd(m, batch, dev):
        g, _, _, y, _ = batch
        return m(g.to(dev)), y.to(dev)

    pg = [{"params": model.parameters(), "lr": tc["lr"],
           "weight_decay": tc["weight_decay"]}]
    ck = str(Path(cfg["paths"]["checkpoints"]) / "task2.pt")
    return _fit(model, mk("train", True), mk("val", False), mk("test", False),
                dev, fwd, pg, tc["epochs"], 1, ck,
                cfg["eval"]["threshold"], "constant"), "task2"


def task_cnn(cfg, dev):
    from cnn_baseline import CNNBaseline
    from cnn_data import MelDataset, mel_collate
    tc = cfg["train"]["cnn_baseline"]
    def mk(split, shuffle):
        return DataLoader(MelDataset(cfg, split), batch_size=tc["micro_batch"],
                          shuffle=shuffle, collate_fn=mel_collate, num_workers=0)
    model = CNNBaseline(n_labels(cfg), cfg["audio"]["n_mels"]).to(dev)

    def fwd(m, batch, dev):
        x, y = batch
        return m(x.to(dev)), y.to(dev)

    pg = [{"params": model.parameters(), "lr": tc["lr"]}]
    ck = str(Path(cfg["paths"]["checkpoints"]) / "cnn_baseline.pt")
    return _fit(model, mk("train", True), mk("val", False), mk("test", False),
                dev, fwd, pg, tc["epochs"], tc["grad_accum"], ck,
                cfg["eval"]["threshold"], "constant"), "baseline_cnn"


def task3(cfg, dev, variant):
    from fusion_model import FusionModel
    tc = cfg["train"]["task3"]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    mk = _loaders(cfg, tok, cfg["text"]["max_len_metadata"], tc["micro_batch"])
    model = FusionModel(cfg, n_labels(cfg), variant).to(dev)

    def fwd(m, batch, dev):
        g, ids, mask, y, _ = batch
        return m(g.to(dev), ids.to(dev), mask.to(dev)), y.to(dev)

    bert_ids = set(id(p) for p in model.bert.parameters())
    pg = [
        {"params": [p for p in model.parameters() if id(p) in bert_ids],
         "lr": tc["lr_bert"]},
        {"params": [p for p in model.parameters() if id(p) not in bert_ids],
         "lr": tc["lr_other"]},
    ]
    ck = str(Path(cfg["paths"]["checkpoints"]) / f"task3_{variant}.pt")
    hist, test = _fit(model, mk("train", True), mk("val", False),
                      mk("test", False), dev, fwd, pg, tc["epochs"],
                      tc["grad_accum"], ck, cfg["eval"]["threshold"], "cosine")
    return (hist, test), f"task3.variants.{variant}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["1", "2", "cnn", "3", "4"])
    ap.add_argument("--variant", default="cross_attn",
                    choices=["cross_attn", "concat", "bert_only", "gnn_only"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    dev = device()
    print(f"device={dev}")

    if args.task == "4":
        from train_task4 import run_task4
        run_task4(cfg, dev)
        return

    if args.task == "1":
        (hist, test), key = task1(cfg, dev)
    elif args.task == "2":
        (hist, test), key = task2(cfg, dev)
    elif args.task == "cnn":
        (hist, test), key = task_cnn(cfg, dev)
    else:
        (hist, test), key = task3(cfg, dev, args.variant)

    update_metrics(f"{key}.history", hist)
    update_metrics(f"{key}.test", test)
    print(f"saved metrics under {key}")


if __name__ == "__main__":
    main()
