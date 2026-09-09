# Handoff — CSE425 GNN-BERT Music Context Project

**Date:** 2026-09-09
**Workspace:** `F:\user5\425\cse52-project` (git repo, branch `main`)
**Stage reached:** PLAN_V2.md executed end-to-end, Steps 1–9 complete. All verification gates pass. **Nothing committed** — 29 modified files in the working tree awaiting the user's own commit.

> Supersedes the previous handoff (which described the pre-execution state on the old machine, workspace `E:\Study\425\project`).

## What to read first

1. **`PLAN_V2.md`** — the rebuild plan I just executed. Authoritative for *what* was done and why; do not re-litigate its locked decisions.
2. **`PLAN.md` §0** — the 16 locked design decisions (D1–D16) behind the pipeline.
3. **`results/metrics.json`** — single source of truth for every number. `results/comparison_table.md` is its rendered form. Do not restate numbers from memory; read these.
4. **`README.md`** — pipeline commands and repo map.

## Current state

Full rebuild done on this machine: data downloaded, everything preprocessed, all 8 training runs completed, all analysis artifacts and plots regenerated, demo notebook executed clean, `report/final_report.md` + `README.md` synced to the new metrics.

All PLAN_V2 §9 gates pass: every model beats the random baseline; all Task-3 histories are 15 epochs; both retrieval directions clear R@10 ≥ 15%; 20 sample graphs; `task1_examples.md` has 5 distinct clips; no stale "8 epochs" text in the markdown report; no leftover `.resume` files (all runs finished cleanly).

**The one interpretive change worth knowing:** with Task 3 restored to the planned 15 epochs (the old run was cut to 8), `cross_attn` rose 0.257 → 0.272. It and `concat` (0.278) are now effectively tied — concat leads Macro-F1 by 0.006, cross-attention leads Micro-F1 and matches AUC-PR. The report's analysis was rewritten to claim **no advantage for either**, replacing the old "concat edges out cross-attention" narrative. Do not reintroduce the old claim.

## Open items (in priority order)

1. **`report/final_report.tex` is STALE and is the actual graded deliverable.** PLAN_V2 §7 scoped the sync to markdown only, so I deliberately did not touch it. It still contains the old run's numbers (0.276 / 0.257 / 0.224 / 0.299 / 0.195 / 0.235 / 23.36 / 0.172) and the "8 epochs" wording. `report/final_report.pdf` is a 0-byte placeholder. Per PLAN.md D13 the submission is a NeurIPS-template PDF, so someone must port the synced content from `final_report.md` into the `.tex` and compile. `final_report.md` is the corrected reference — port *from* it, not from the `.tex`.
2. **pdflatex is not installed** on this machine (PLAN_V2 §1.4 explicitly said not to install it; the user handles the PDF outside the plan). Compiling will need MiKTeX/TeXLive or Overleaf.
3. **Commit.** The user commits themselves — this was an explicit constraint all session. Do not commit without being asked.
4. **Human evaluation was declined** by the user this session. `report/final_report.md` Limitations correctly states it was not conducted and points at `results/human_eval_sheet.csv`. `task4.human_eval` is absent from metrics.json. **Never fill the sheet yourself or claim ratings that do not exist.** If the user later fills it: `python src/evaluate.py --what human_eval`, then update that Limitations sentence with the mean/SD.

## Environment (non-obvious, needed to reproduce)

- `.venv` at repo root, **Python 3.12.3** — PLAN.md D16 assumed 3.10/3.11, but only 3.12 exists here. This pulled newer library majors: torch 2.5.1+cu121, transformers 5.16.1, librosa 1.0.0, pandas 3.0.5, numpy 2.5.3, torch-geometric 2.8.0. All five model classes and mp3 decoding were smoke-tested on this stack before the long runs; everything works. The `torch.cuda.amp` FutureWarnings in training logs are cosmetic.
- GPU: **RTX 4070 Ti SUPER, 16 GB** (config.yaml's batch/grad-accum values were tuned for a 4 GB card; they were left unchanged and have plenty of headroom).
- ffmpeg installed via `winget install Gyan.FFmpeg`; not strictly needed since soundfile decodes MTAT mp3s directly.
- **Jupyter gotcha:** the notebook's kernelspec is `python3`, which resolves to a *user-level* kernel pointing at system Python, not the venv. To execute it you must force the venv kernel:
  ```bash
  export JUPYTER_PATH="F:/user5/425/cse52-project/.venv/share/jupyter"
  .venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/demo_context.ipynb
  ```
  (A venv kernel was already registered with `python -m ipykernel install --sys-prefix --name python3`.)

## Changes I made beyond the plan, and why

PLAN_V2 said no redesign. These are the only source/content deviations, all correctness fixes rather than redesign:

- **`src/make_plots.py` (the only source file changed).** `_tsne_plot` passed integer category codes as `c=` with `cmap="tab10"` — matplotlib normalizes those across the colormap — while building legend handles with `tab10(i/10)`. The legends therefore **did not match the plotted points** in both t-SNE figures. Fixed to one shared palette for points and legend, and added class counts to legend labels. Both figures regenerated.
- **Report analysis rewritten against the corrected figures.** The old text claimed genre umbrellas "classical, electronic, world, and acoustic form clearly separated regions" — `world` is 38 of 2,993 test clips. Also corrected: the case studies are 3-node graphs with 2 similarity edges (not "several"), and the "soft female-vocal pop" retrieval query returns clips sharing the vocal character but *not* the pop genre.
- **`results/plots/architecture.png` regenerated.** PLAN_V2 §5 claimed it was tracked in git; it is not (`.gitignore` excludes `results/plots/*`). Produced with `python src/make_arch_diagram.py`.
- **README GPU line** changed from "RTX 3050 Ti, 4 GB" to the actual machine, since it no longer described the run that produced the numbers.

## Gotchas

- `data/` is ~30 GB on local disk and gitignored; it does **not** travel through git. A fresh machine repeats the whole rebuild.
- Preprocessing and training are idempotent/resumable. Do **not** re-run a completed training task — a clean finish deletes its `.resume` file, and re-running overwrites good checkpoints.
- `results/metrics.json` is the single metrics store, written by many scripts via read-modify-write. Avoid running two metric-writing scripts concurrently (I serialised them for this reason).
- Actual wall-clock here: MTAT download ~40 min, MusicCaps ~12 min, MTAT preprocessing ~3 h (12 workers, CPU-saturated), all 8 training runs ~2.7 h, evaluate + plots ~15 min.
- MTAT mirror ran ~1.8 MB/s; the download script is idempotent and safe to re-run.

## Suggested skills

Call the Skill tool for:

- **`code-review`** — the highest-value next step before the user commits. There are 29 modified files including a source change to `src/make_plots.py` and substantial rewrites in `report/final_report.md`. Run it on the working-tree diff.
- **`dataviz`** — only if you touch the figures again (`f1_curves.png`, `auc_pr_task3.png`, the two t-SNEs). They are graded deliverables and the plotting code just had a real colour-mapping bug, so any further edit deserves the checklist.
- **`graphify`** — optional, if you need to orient in the codebase quickly rather than reading `src/*.py` directly.

No skill is needed for the LaTeX port — it is a mechanical sync from `report/final_report.md` into `report/final_report.tex`.

## Sensitive info

None in the workspace or this session. No API keys, credentials, or PII. Datasets are public (MagnaTagATune mirrors; MusicCaps audio via the public `CLAPv2/MusicCaps` HuggingFace mirror — no YouTube downloads, no authentication used). HuggingFace requests were unauthenticated, which only triggers a rate-limit warning.
