# Implementation Plan — GNN-BERT Music Context Understanding (CSE425 Project)

**Status:** Planning complete. Execution-ready. No decisions left to executor.
**Spec source:** `CSE425_Project_GNN_BERT_Music_Context.md`
**Deadline:** 2 October 2026

---

## 0. Locked Decisions (DO NOT RE-DECIDE)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Scope | **All 4 tasks** (Task 4 = 18 bonus marks) | Max marks; Task 4 reuses Task 2/1 encoders |
| D2 | Audio dataset | **MagnaTagATune (MTAT)** — 25,863 clips, 29 s, mp3 | Spec-recommended; audio + 188 tags in one download; ~1.5 GB |
| D3 | Text/caption dataset | **MusicCaps** (Google, 5,521 captions) | Only true caption dataset in Table 1; needed for Task 4 |
| D4 | Task 1 text input | MTAT clip metadata string: `"{artist} - {title} ({album})"` from `clip_info_final.csv` | MTAT has no lyrics/captions; spec allows "MagnaTagATune tag subset (top-50 tags)"; metadata→tag prediction is the standard proxy |
| D5 | Label set (Tasks 1–3) | **Top-50 tags** by frequency from `annotations_final.csv` | Spec explicitly suggests top-50 |
| D6 | MTAT split | Folder-based: MTAT ships 16 folders `0–9, a–f`. Train = first 13 (sorted), Val = folder `d`, Test = folders `e,f` | MTAT has no single official split; folder split is the community standard and guarantees no clip-level leakage. Document this in report |
| D7 | MusicCaps split | Deterministic 90/10 train/test by sorted `ytid`, seed 42 | MusicCaps has no official split |
| D8 | BERT model | `distilbert-base-uncased` (66M params) | Spec allows it; fits GPU budget; fine-tuned fully |
| D9 | GNN | **GraphSAGE** (PyTorch Geometric `SAGEConv`), 2 layers | Spec names GraphSAGE first; simpler than GAT |
| D10 | Graph type | **Segment graphs** (not chord graphs) | Chord extraction adds fragility; spec allows either; segment graphs spec'd in §3 step 3 |
| D11 | Emotion regression | **Skipped** (DEAM optional aux) | Spec marks optional; keeps pipeline lean |
| D12 | Human eval (Task 4) | **Self-rating by project owner**: 10 retrieval examples, 1–5 scale, auto-generated rating sheet | User chose skip/self-rate |
| D13 | Report | NeurIPS 2024 LaTeX template, 6–10 pages, compiled locally with pdflatex | Spec option 1 |
| D14 | Hardware assumption | NVIDIA GPU + CUDA torch | Confirmed by user |
| D15 | Seed | 42 everywhere (torch, numpy, random) | Reproducibility rubric |
| D16 | Python | 3.10 or 3.11 in `.venv` at repo root | torch/pyg compatibility |

---

## 1. Deliverables Map (spec §10 → our files)

| Spec deliverable | Produced by phase |
|---|---|
| GitHub repo / full source | Existing skeleton filled in (P0–P6) |
| ≥20 preprocessed graph samples (`.pt`) | P2 exports 20 test-set graphs to `results/` + full cache in `data/processed/graphs/` |
| Evaluation tables + plots | P7: `results/metrics.json`, `results/plots/*.png`, `results/retrieval_examples/` |
| Final report PDF (NeurIPS, 6–10 pp) | P8: `report/final_report.pdf` |
| Demo notebook | P9: `notebooks/demo_context.ipynb` (end-to-end inference on 1 clip) |
| ≥2 baselines (B1 random, B2 CNN, B3 BERT-only) | B1 in P7 (trivial), B2 in P4, B3 = Task 1 (P3) |
| Task 3 ablations (BERT-only / GNN-only / concat / cross-attn) | P5 |
| t-SNE of fused embedding `z` | P7 |
| 3 case studies (graph paths + caption alignment) | P7 (from Task 4 model on MusicCaps test set) |
| 10 qualitative retrieval examples | P7 |
| Task 1: 5 example predictions | P7 |
| Macro-F1 / Micro-F1 vs epoch curves | P7 (from logged history in `results/metrics.json`) |

---

## 2. Environment & Tooling

**Requirements to put in `requirements.txt` (exact pins decided here):**
```
torch>=2.2        # CUDA build, installed via pytorch index (see P0 step)
torch-geometric>=2.5
transformers>=4.40
librosa>=0.10
soundfile
scikit-learn
pandas
numpy
matplotlib
tqdm
pyyaml
yt-dlp
jupyter
```
Note in README: `torch` must be the CUDA wheel; `torch-scatter`/`torch-sparse` NOT needed (PyG ≥2.5 pure-torch works).

**User environment preconditions (verify in P0, escalate to user if missing):**
- `pdflatex` on PATH (MiKTeX/TeXLive). If missing → STOP, ask user: install MiKTeX or compile report on Overleaf.
- `ffmpeg` on PATH (required by yt-dlp for audio extraction). If missing → install via `winget install ffmpeg` (no user decision needed, just do it).

---

## 3. Phase 0 — Setup (est. 30 min)

1. `python -m venv .venv` in repo root; activate.
2. `pip install torch --index-url https://download.pytorch.org/whl/cu121` then `pip install -r requirements.txt`.
3. Sanity script `python -c "import torch, torch_geometric, transformers, librosa; print(torch.cuda.is_available())"` — must print `True`. If `False`, try `cu118`; if still False → escalate to user.
4. Verify `pdflatex --version` and `ffmpeg -version`.
5. Write `config.yaml` with ALL hyperparameters from §5 below (single source of truth; `src/*.py` reads it).
6. `.gitignore` already exists — append `data/raw/*`, `data/processed/*`, `*.pt`, `.venv/` if not present.

**Done when:** sanity prints pass, `config.yaml` complete.

---

## 4. Phase 1 — Data Acquisition (est. 2–6 h, mostly unattended download)

### P1.1 MagnaTagATune
- Download from official mirrors. Files needed:
  - Audio: the three zips from `http://mi.soi.city.ac.uk/datasets/magnatagatune/` (or the mirg.city.ac.uk link in spec) — parts containing folders `0–f`.
  - Annotations: `annotations_final.csv`
  - Clip info: `clip_info_final.csv`
- If the official site is down, fallback (in order): (1) `https://mirg.city.ac.uk/datasets/magnatagatune/`, (2) HuggingFace `seungheondoh/mtg` mirrors, (3) Kaggle MTAT mirrors. Executor tries in order; only escalate if all fail.
- Extract into `data/raw/mtat/audio/` (folders `0..f` inside) and CSVs into `data/raw/mtat/`.
- **Verify:** count mp3 files ≈ 25,863; both CSVs parse; every `clip_id` in annotations has an mp3. Log counts to `results/metrics.json` under `dataset.mtat`.

### P1.2 MusicCaps
- Get `musiccaps-public.csv` (columns: `ytid, start_s, end_s, caption, ...`) — from Google's research GitHub (`google-research-datasets/musiccaps`) or HuggingFace `google/musiccaps`. Try GitHub raw first, HF second.
- Write `src/download_musiccaps.py`: for each row, `yt-dlp -x --audio-format wav --postprocessor-args "-ar 22050 -ac 1" -o "data/raw/musiccaps/{ytid}.%(ext)s" "https://www.youtube.com/watch?v={ytid}"`, then crop `[start_s, end_s]` with ffmpeg into `data/raw/musiccaps/wav/{ytid}.wav`. Parallelize with 4 worker processes. Skip-and-log failures to `data/raw/musiccaps/failed.txt`.
- **Verify:** ≥ 4,000 successful clips (expect ~80–90%). If < 4,000 → still proceed (note in report); only escalate if < 2,000.

**Done when:** verification counts logged.

---

## 5. Phase 2 — Preprocessing (est. 4–8 h compute)

Single entry point: `python src/preprocess.py --dataset mtat|musiccaps` (new file; orchestrates `audio_features.py` + `graph_builder.py`). Multiprocess with `concurrent.futures.ProcessPoolExecutor(max_workers=8)`. All outputs cached; re-run must be idempotent (skip existing).

### P2.1 Audio features (per clip; `src/audio_features.py`)
- Load with librosa, **resample 22050 Hz mono** (spec §3.1).
- **log-mel:** n_fft=1024, hop=512, n_mels=128, `power_to_db`, per-track mean/std normalize → save `.npy` to `data/processed/mel/{clip_id}.npy`. (For CNN baseline B2.)
- **Segments:** fixed windows **5 s, hop 2.5 s** → 11 windows per 29 s MTAT clip; MusicCaps 10 s clips → 4 windows (last partial window dropped if < 2 s).
- **Node features per window:** MFCC-20 (mean+std over time) + chroma-12 (mean+std) = **64-dim vector**. Save to `data/processed/nodes/{clip_id}.npy` (shape `[n_windows, 64]`).

### P2.2 Graph construction (`src/graph_builder.py`)
Per clip, from node-feature matrix:
- Nodes = windows. Edge set E:
  1. Temporal: bidirectional (i, i+1).
  2. Similarity: cosine(node_i, node_j) > **τ = 0.6**, non-adjacent pairs, **max 4 extra edges per node** (keep highest sim), weight = sim.
- Build `torch_geometric.data.Data(x, edge_index, edge_attr, y)` → save `data/processed/graphs/{clip_id}.pt`.
- MTAT labels: multi-hot 50-dim from top-50 tags. MusicCaps graphs have no `y` (used by Task 4 only).
- **Submission samples:** copy 20 test-set `.pt` graphs to `results/sample_graphs/`.
- **Verify (script asserts):** no isolated nodes (temporal edges guarantee), feature dims correct, ≥ 95% of clips processed; failures logged and excluded.

### P2.3 Splits (`data/splits/`)
- `mtat_train.json`, `mtat_val.json`, `mtat_test.json` — lists of clip_ids per D6.
- `musiccaps_train.json`, `musiccaps_test.json` — ytid lists per D7 (only successfully downloaded).

### P2.4 Text prep
- MTAT: build `data/processed/mtat_text.json` mapping clip_id → metadata string per D4. Missing artist/album fields → empty string (never drop clip).
- MusicCaps: captions used raw; tokenize at train time (max_len 128, pad/truncate — spec §3.4).

**Done when:** all caches exist, verification script passes, split JSONs written.

---

## 6. Phase 3 — Task 1: BERT Baseline (est. 1–2 h)

`src/bert_encoder.py` + `src/train.py --task 1`.

- Model: `distilbert-base-uncased` + linear head (768→50), sigmoid. Loss: BCE (spec Alg. 1).
- Text: MTAT metadata strings (D4). max_len=64 (short strings), batch=32, lr=2e-5, AdamW, **10 epochs**, cosine schedule, warmup 5%.
- Eval on val each epoch: Macro-F1, Micro-F1, mean AUC-PR @ threshold 0.5. Save best-val checkpoint to `results/checkpoints/task1.pt`.
- Log per-epoch metrics to history dict in `results/metrics.json` → `task1.history`.
- **Acceptance:** test Macro-F1 meaningfully above random baseline (≥ 0.15; random ≈ 0.05). If not: first retry with lr 3e-5 / 15 epochs; second retry unfreezing nothing different — DistilBERT is fully trained already. Only escalate after 2 retries.

**Done when:** `task1.test` metrics in `metrics.json`.

---

## 7. Phase 4 — Task 2 + CNN Baseline (est. 2–4 h)

### P4.1 CNN baseline B2 (`src/cnn_baseline.py` — new file)
- Input: log-mel `[128, T]`. Architecture: 4 blocks × [Conv2d(3×3) → BN → ReLU → MaxPool(2×2)], channels 32→64→128→256; global mean-pool over freq+time; linear 256→50 sigmoid.
- Train: batch 32, lr 1e-3 Adam, 15 epochs, BCE. Same splits/seed.

### P4.2 GNN (`src/gnn_model.py`, `src/train.py --task 2`)
- GraphSAGE: `SAGEConv(64→256) → ReLU → Dropout(0.3) → SAGEConv(256→256)`; **mean readout** (spec §4.2); head 256→50 sigmoid.
- Train: PyG `DataLoader`, batch 32 graphs, lr 1e-3 Adam, weight_decay 1e-4, 20 epochs, BCE. Best-val checkpoint `results/checkpoints/task2.pt`.

**Acceptance:** both beat B1; log `task2.test` and `baseline_cnn.test` metrics.

---

## 8. Phase 5 — Task 3: Fusion + Ablations (est. 3–5 h)

`src/fusion_model.py`, `src/train.py --task 3 --variant {cross_attn,concat,bert_only,gnn_only}`.

- **Cross-attention (main):** graph readout g (256) → query; BERT token outputs H_text `[L,768]` → project to 256 → keys/values; single-head attention per spec §4.3; `z = concat(g, A·H_text)` (512) → MLP(512→256→50) sigmoid.
- BERT **fully fine-tuned**: lr 2e-5 (BERT params) / 1e-3 (GNN+fusion+head), 15 epochs, batch 16 (memory), BCE.
- Ablations (same data/epochs/seed):
  1. `bert_only` — should ≈ Task 1 numbers (sanity).
  2. `gnn_only` — should ≈ Task 2 numbers (sanity).
  3. `concat` — `z = concat(g, t_CLS)`.
  4. `cross_attn` — main.
- All four logged to `metrics.json` under `task3.variants.*`. Main model checkpoint → `task3_cross_attn.pt`.
- **Acceptance:** cross_attn Macro-F1 ≥ max(bert_only, gnn_only). If not after one retry (swap lr schedule to constant, 20 epochs) → still report honestly; ablation comparison is itself the deliverable.

---

## 9. Phase 6 — Task 4: Contrastive MusicCaps (est. 3–6 h)

`src/contrastive.py`, `src/train.py --task 4`.

- Dual encoder: GNN branch = Task-2 architecture (64→256→256, mean pool) → L2-normalize; text branch = DistilBERT CLS → linear 768→256 → L2-normalize.
- InfoNCE, **τ = 0.07** (spec §4.4), symmetric (both directions, averaged). batch=64, lr 1e-4 (GNN) / 2e-5 (BERT), 20 epochs, AdamW.
- Init: GNN branch from `task2.pt` weights, BERT from `task1.pt` (faster convergence; document).
- Eval on `musiccaps_test.json`: build full similarity matrix, report **Caption→Audio and Audio→Caption R@1/R@5/R@10**.
- Zero-shot tag probe (deliverable): on MTAT test subset (500 clips, seed 42), encode graphs with the contrastive GNN; encode prompts `"This music sounds like {tag}"` for each of top-50 tags with contrastive BERT; predict tags by cosine > tuned-on-val threshold; compare Macro-F1 vs Task 3 model. Log under `task4.zero_shot`.

**Done when:** `task4.retrieval` + `task4.zero_shot` in `metrics.json`.

---

## 10. Phase 7 — Analysis & Artifacts (est. 2–3 h)

`src/evaluate.py` (extend) + `src/make_plots.py` (new). All plots → `results/plots/`, matplotlib, 200 dpi.

1. **B1 random baseline:** compute expected random Macro-F1/AUC-PR analytically from tag priors — no training needed.
2. **F1-vs-epoch curves** for Tasks 1–3 from history in `metrics.json` → `f1_curves.png`.
3. **AUC-PR curves** (top-10 tags) for Task 3 → `auc_pr_task3.png`.
4. **t-SNE** of Task 3 fused `z` on MTAT test, colored by dominant genre tag (map top-50 tags → 8 genre umbrellas via hardcoded dict in `make_plots.py`) → `tsne_task3.png`.
5. **5 Task-1 example predictions** (text, true tags, predicted tags) → `results/task1_examples.md`.
6. **10 retrieval examples**: query caption → top-3 clips (ytid + caption similarity) → `results/retrieval_examples/retrieval_examples.md`.
7. **3 case studies**: pick 3 MusicCaps test items where graph retrieval rank ≤ 5; describe graph structure (n_nodes, n_edges, repeated-segment edges) + caption → `results/case_studies.md`.
8. **Rating sheet**: auto-generate `results/human_eval_sheet.csv` (10 rows: caption, top-1 ytid, blank 1–5 column) + `results/human_eval_instructions.md`. → **HANDOFF TO USER: rate sheet (15 min), return scores.** Then compute mean/SD into `metrics.json` → `task4.human_eval`.
9. Consolidate **Table 3-style comparison** (all models × Macro-F1/AUC-PR/R@5) → `results/comparison_table.md` + `.tex` fragment for report.

---

## 11. Phase 8 — Report (est. 4–6 h, LAST)

- NeurIPS 2024 template into `report/`. Structure (6–10 pp): Abstract, Intro (motivation from spec §1), Related Work (short), Method (eq. from spec §2/§4, our instantiation per D1–D15), Experiments (datasets, splits D6/D7, Table 3 replacement, ablation table, retrieval table), Analysis (t-SNE, case studies, human eval, limitations), Conclusion, References (bibtex: MTAT, MusicCaps, BERT, DistilBERT, GraphSAGE, PyG, CLIP/InfoNCE, librosa).
- Architecture diagram: recreate from `gnn_bert_music_pipeline.excalidraw` → export PNG → include.
- Compile with pdflatex → `report/final_report.pdf`. Missing citations/overfull boxes acceptable; build must succeed.

---

## 12. Phase 9 — Packaging (est. 1 h)

1. `notebooks/eda.ipynb`: tag distribution, clip-length hist, sample graph viz (PyG → networkx plot), mel example. Run end-to-end.
2. `notebooks/demo_context.ipynb`: load `task3_cross_attn.pt` + `task4.pt`; run inference on ONE MTAT test clip (tags out) and ONE caption retrieval; display results. Must run top-to-bottom.
3. `README.md`: setup, download steps, train commands per task, results summary table, repo map.
4. Final rubric self-check against spec §9/§10 — every line item ticked or explicitly noted.
5. `git status` clean summary handed to user (NO commits without user approval).

---

## 13. User Touchpoints (complete list — nothing else)

| When | What |
|---|---|
| P0 | Only if GPU missing or pdflatex missing |
| P7 → P8 | Rate 10 retrieval examples (15 min, self-rate per D12) |
| End | Review report PDF + approve git commit |

## 14. Risk Register (pre-decided fallbacks)

| Risk | Fallback |
|---|---|
| MTAT official site down | Mirror order in P1.1; escalate only if all fail |
| MusicCaps yield < 4k | Proceed, note in report; < 2k → escalate |
| Fusion underperforms baselines | One retry (P8 acceptance), then report honestly — ablation is the deliverable |
| GPU OOM in Task 3/4 | Halve batch, enable gradient accumulation (equiv. batch preserved) |
| librosa mp3 decode fails | `pip install soundfile audioread`; decode via ffmpeg → wav first |
| pdflatex missing | Escalate: user installs MiKTeX or compiles on Overleaf |

**Total estimate:** ~20–30 h wall-clock (mostly unattended downloads/training), ~3 working days of agent execution.
