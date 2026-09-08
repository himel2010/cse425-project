"""Task 4 training: contrastive dual-encoder on MusicCaps + eval (PLAN.md P6)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import ensure_dir, update_metrics
from data_loaders import MusicCapsDataset, MTATDataset, make_collate
from bert_encoder import get_tokenizer
from contrastive import DualEncoder, info_nce, recall_at_k
from metrics import multilabel_metrics


def _init_from_pretrained(model: DualEncoder, cfg):
    """Warm-start GNN from task2.pt, BERT from task1.pt (document in report)."""
    ckdir = Path(cfg["paths"]["checkpoints"])
    t2 = ckdir / "task2.pt"
    t1 = ckdir / "task1.pt"
    if t2.exists():
        sd = torch.load(t2, map_location="cpu")
        # GNNClassifier: encoder.* -> model.gnn.*
        remap = {k[len("encoder."):]: v for k, v in sd.items()
                 if k.startswith("encoder.")}
        model.gnn.load_state_dict(remap, strict=False)
        print("  warm-started GNN from task2.pt")
    if t1.exists():
        sd = torch.load(t1, map_location="cpu")
        # BertClassifier: encoder.bert.* -> model.bert.bert.*
        remap = {k[len("encoder."):]: v for k, v in sd.items()
                 if k.startswith("encoder.")}
        model.bert.load_state_dict(remap, strict=False)
        print("  warm-started BERT from task1.pt")


@torch.no_grad()
def _encode_all(model, loader, dev):
    model.eval()
    za, zt, ids = [], [], []
    for g, i_ids, mask, _, batch_ids in loader:
        a = model.encode_graph(g.to(dev))
        t = model.encode_text(i_ids.to(dev), mask.to(dev))
        za.append(a.cpu()); zt.append(t.cpu()); ids += list(batch_ids)
    return torch.cat(za), torch.cat(zt), ids


def _zero_shot_probe(model, cfg, dev):
    """Encode MTAT-test-subset graphs + tag prompts; predict tags by cosine."""
    tags = json.loads(
        (Path(cfg["paths"]["data_processed"]) / "mtat_labels.json").read_text())["tags"]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    prompts = [f"This music sounds like {t}" for t in tags]
    ptok = tok(prompts, padding=True, truncation=True, max_length=16,
               return_tensors="pt")
    with torch.no_grad():
        tag_emb = model.encode_text(ptok["input_ids"].to(dev),
                                    ptok["attention_mask"].to(dev)).cpu()  # [50,D]

    def encode_split(split, limit=None):
        ds = MTATDataset(cfg, split)
        if limit:
            ds.ids = ds.ids[:limit]
        dl = DataLoader(ds, batch_size=32, collate_fn=make_collate(tok, 8))
        embs, ys = [], []
        with torch.no_grad():
            for g, _, _, y, _ in dl:
                embs.append(model.encode_graph(g.to(dev)).cpu())
                ys.append(y.numpy())
        return torch.cat(embs), np.concatenate(ys)

    # tune threshold on val, apply to test subset
    ve, vy = encode_split("val", limit=500)
    vscore = (ve @ tag_emb.t()).numpy()
    best_thr, best_f1 = 0.0, -1
    for thr in np.linspace(vscore.min(), vscore.max(), 25):
        f1 = multilabel_metrics(vy, vscore, thr)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    te, ty = encode_split("test", limit=cfg["eval"]["zero_shot_subset"])
    tscore = (te @ tag_emb.t()).numpy()
    m = multilabel_metrics(ty, tscore, best_thr)
    m["threshold"] = round(best_thr, 4)
    return m


def run_task4(cfg, dev):
    tc = cfg["train"]["task4"]
    tok = get_tokenizer(cfg["text"]["bert_model"])
    collate = make_collate(tok, cfg["text"]["max_len_caption"])

    train_dl = DataLoader(MusicCapsDataset(cfg, "train"),
                          batch_size=tc["micro_batch"], shuffle=True,
                          collate_fn=collate)
    test_dl = DataLoader(MusicCapsDataset(cfg, "test"),
                         batch_size=tc["micro_batch"], collate_fn=collate)

    model = DualEncoder(cfg).to(dev)
    _init_from_pretrained(model, cfg)

    bert_ids = set(id(p) for p in model.bert.parameters())
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in bert_ids],
         "lr": tc["lr_bert"]},
        {"params": [p for p in model.parameters() if id(p) not in bert_ids],
         "lr": tc["lr_gnn"]},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    ga = tc["grad_accum"]

    for ep in range(tc["epochs"]):
        model.train()
        opt.zero_grad()
        tot = 0.0
        for i, (g, ids, mask, _, _) in enumerate(train_dl):
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                za, zt = model(g.to(dev), ids.to(dev), mask.to(dev))
                loss = info_nce(za, zt, tc["temperature"]) / ga
            scaler.scale(loss).backward()
            if (i + 1) % ga == 0 or (i + 1) == len(train_dl):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * ga
        print(f"  ep{ep+1}/{tc['epochs']} loss={tot/len(train_dl):.4f}")

    ensure_dir(Path(cfg["paths"]["checkpoints"]))
    torch.save(model.state_dict(),
               Path(cfg["paths"]["checkpoints"]) / "task4.pt")

    # retrieval eval
    za, zt, ids = _encode_all(model, test_dl, dev)
    sim_t2a = zt @ za.t()   # caption(query) -> audio
    sim_a2t = za @ zt.t()   # audio(query) -> caption
    retrieval = {
        "caption_to_audio": recall_at_k(sim_t2a),
        "audio_to_caption": recall_at_k(sim_a2t),
        "n_test": len(ids),
    }
    update_metrics("task4.retrieval", retrieval)
    print(f"  retrieval C->A {retrieval['caption_to_audio']} "
          f"A->C {retrieval['audio_to_caption']}")

    zs = _zero_shot_probe(model, cfg, dev)
    update_metrics("task4.zero_shot", zs)
    print(f"  zero-shot {zs}")
