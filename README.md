# GNN-BERT Music Context Understanding (CSE425)

Multi-task music understanding combining a **GraphSAGE** audio-segment graph encoder
with a **DistilBERT** text encoder. Four tasks: (1) text→tag baseline, (2) graph→tag,
(3) graph+text fusion with ablations, (4) contrastive audio↔caption retrieval.

See `PLAN.md` for the full design and locked decisions, and
`CSE425_Project_GNN_BERT_Music_Context.md` for the course spec.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Unix
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 wheel
pip install -r requirements.txt
```

Requires an NVIDIA GPU (tested on RTX 3050 Ti, 4 GB), `ffmpeg` on PATH, and (for the
report) `pdflatex`. `config.yaml` holds every hyperparameter — scripts read it and
resolve all paths against the repo root, so run them from anywhere.

## Data

```bash
python src/download_mtat.py          # MagnaTagATune: 25,863 mp3s + tag CSVs (~3 GB)
python src/download_musiccaps.py     # MusicCaps audio via HF mirror CLAPv2/MusicCaps
python src/preprocess.py --dataset mtat --workers 1        # features + graphs (idempotent)
python src/preprocess.py --dataset musiccaps --workers 1
```

- **MTAT** (tasks 1–3): top-50 tags, folder-based split (train `0–c`, val `d`, test `e,f`).
- **MusicCaps** (task 4): fetched from the HuggingFace `CLAPv2/MusicCaps` mirror because
  YouTube blocks yt-dlp with bot-detection; 5,352 clips, deterministic 90/10 split (seed 42).
- Each clip → 5 s / 2.5 s-hop segment graph; nodes = MFCC-20 + chroma-12 (mean+std, 64-dim);
  edges = temporal + cosine-similarity (τ=0.6, ≤4 extra/node).

Preprocessing runs **serial** and only processes uncached clips (idempotent); re-run until
counts stabilise. Sample graphs for submission are in `results/sample_graphs/`.

## Train

```bash
python src/train.py --task 1                       # BERT text baseline (B3)
python src/train.py --task 2                       # GraphSAGE
python src/train.py --task cnn                     # CNN mel baseline (B2)
python src/train.py --task 3 --variant cross_attn  # fusion (also: concat, bert_only, gnn_only)
python src/train.py --task 4                       # contrastive dual-encoder
```

Training checkpoints full state each epoch to `results/checkpoints/*.resume` and resumes on
relaunch (the environment kills long runs), deleting the resume file on clean completion.

## Analysis & artifacts

```bash
python src/evaluate.py --what all   # B1 baseline, example preds, retrieval, case studies, table
python src/make_plots.py            # F1 curves, PR curves, t-SNE of fused z
```

Outputs land in `results/`: `metrics.json` (single metrics store), `plots/*.png`,
`comparison_table.md`, `retrieval_examples/`, `case_studies.md`, `task1_examples.md`,
`human_eval_sheet.csv`.

## Results (MTAT test, top-50 tags)

| Model | Macro-F1 | Micro-F1 | mean AUC-PR |
|---|---|---|---|
| B1 random | 0.066 | 0.098 | 0.066 |
| B3 / Task 1 BERT (metadata) | 0.195 | 0.319 | 0.300 |
| Task 2 GraphSAGE | 0.235 | 0.360 | 0.346 |
| B2 CNN (log-mel) | 0.299 | 0.432 | 0.400 |
| Task 3 bert_only | 0.224 | 0.340 | 0.309 |
| Task 3 gnn_only | 0.224 | 0.377 | 0.332 |
| Task 3 concat | **0.276** | 0.403 | 0.389 |
| Task 3 cross-attn | 0.257 | 0.382 | 0.385 |

**Task 4 (MusicCaps test, 535 clips):** Caption→Audio R@1/5/10 = 3.0/14.9/23.4;
Audio→Caption = 3.7/13.6/22.1 (random R@10 ≈ 1.9%). Zero-shot tag probe Macro-F1 = 0.172.

Fusion (concat & cross-attn) beats both unimodal ablations, confirming the modalities are
complementary. All models beat the random baseline. Numbers are reproducible with seed 42.

## Repo map

```
src/            pipeline (download, preprocess, models, train, evaluate, plots)
config.yaml     all hyperparameters
data/           raw + processed (gitignored)
results/        metrics.json, checkpoints, plots, sample_graphs, artifacts
notebooks/      eda.ipynb, demo_context.ipynb
report/         NeurIPS report (P8)
```
