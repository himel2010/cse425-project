# PLAN-V2 — Full rebuild + rubric-gap closure on fresh machine (executor handoff)

**For:** fresh agent on a new machine (≥8 GB VRAM GPU, OS unknown — commands given for both Windows PowerShell and Linux bash).
**Repo:** obtained via `git clone` / `git pull` of this project (user pushes from the old machine).
**Authority order:** this document > PLAN.md > `CSE425_Project_GNN_BERT_Music_Context.md` (course spec). Read PLAN.md §0 (locked decisions D1–D16) once for rationale; do not re-decide anything.
**Mode:** execute steps in order, exactly. No redesign, no new features, no extra datasets/models. Seed stays 42 (config.yaml already enforces). Never commit to git — user commits themselves.

---

## 0. Why this plan exists (context, do not re-litigate)

The project (4-task GNN+BERT music-context pipeline) is fully implemented. The old machine (4 GB GPU) trained everything once; those artifacts do NOT travel through git (`.gitignore` excludes `data/`, `*.pt`, `results/plots/`). This machine rebuilds all data + training from scratch, with these improvements already baked into the repo code:

1. `config.yaml` `train.task3.epochs: 15` (old run was cut to 8 on the 4 GB GPU).
2. `src/make_plots.py` plots Macro **and** Micro-F1 curves (spec §4.1), plus a second t-SNE coloured by mood umbrella `tsne_task3_mood.png` (spec §4.3: "genre and mood").
3. `src/evaluate.py` Task-1 example predictions pick 5 distinct seeded-random test clips (old file had 5 copies of one Bach clip), and has a `human_eval` ingester (`--what human_eval`) that writes `task4.human_eval` to `results/metrics.json` once the rating sheet is filled.

Your job: rebuild data → retrain everything → regenerate artifacts → sync report/README numbers → one optional user touchpoint (human-eval rating) → final verification. Nothing else.

**Reference numbers from the old (seed-42, 4 GB) run** — use as sanity bounds, not hard gates. Fresh hardware/library versions may shift values slightly; ±0.02 Macro-F1 is normal:

| key in results/metrics.json | old Macro-F1 (test) |
|---|---|
| baseline_random | 0.066 |
| baseline_cnn | 0.299 |
| task1 | 0.195 |
| task2 | 0.235 |
| task3.variants.bert_only | 0.224 |
| task3.variants.gnn_only | 0.224 |
| task3.variants.concat | 0.276 |
| task3.variants.cross_attn | 0.257 |
| task4.retrieval caption_to_audio R@10 | 23.36 (%) |
| task4.zero_shot macro_f1 | 0.172 |

---

## 1. Bootstrap (≈30 min)

1.1. `git clone <repo-url>` (or `git pull` if already cloned). Work at repo root. Confirm `PLAN_V2.md` (this file), `config.yaml`, `src/`, `requirements.txt` exist.

1.2. Python 3.10 or 3.11 venv:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
If the cu121 wheel mismatches the driver, retry with `cu124`, then `cu118`. Verify:
```bash
python -c "import torch, torch_geometric, transformers, librosa, datasets; assert torch.cuda.is_available(); print('OK', torch.cuda.get_device_name(0))"
```
Must print OK. CPU-only after 3 index attempts → STOP, tell user GPU not visible.

1.3. ffmpeg (librosa mp3 decode for MTAT):
- Windows: `winget install ffmpeg` (then open a NEW shell so PATH refreshes).
- Linux: `sudo apt-get install -y ffmpeg`.
- Verify: `ffmpeg -version` prints. If install fails, MTAT decode may still work via `audioread` (already in requirements); continue and watch Step 2.2 error rate.

1.4. Do NOT install pdflatex. Report PDF compilation is handled by the user later, outside this plan.

## 2. Data download (≈1–3 h, unattended)

2.1. MTAT (audio + CSVs, ~3 GB):
```bash
python src/download_mtat.py
```
Idempotent; safe to re-run. It tries mirrors in PLAN.md P1.1 order. **Verify at end:** `results/metrics.json` → `dataset.mtat.mp3_count` = 25863. If mirrors all fail → STOP, tell user MTAT unreachable.
Count check (either shell): Windows `(Get-ChildItem data\raw\mtat\audio -Recurse -Filter *.mp3).Count` / Linux `find data/raw/mtat/audio -name '*.mp3' | wc -l` → 25863.

2.2. MusicCaps (HF mirror CLAPv2/MusicCaps, no YouTube/ffmpeg needed):
```bash
python src/download_musiccaps.py
```
**Verify:** `dataset.musiccaps.downloaded` ≥ 4000 (old run: 5352, failed 0). Script itself warns/errors on low yield per PLAN.md; <2000 → STOP, tell user.

## 3. Preprocessing (≈3–6 h, unattended)

```bash
python src/preprocess.py --dataset mtat --workers 8
python src/preprocess.py --dataset musiccaps --workers 8
```
Idempotent/cached; if the machine kills a run, re-run the same command (it resumes from cache). If RAM < 16 GB, drop `--workers 4`.

**Verify:**
- `preprocess.mtat.graphs_built` ≥ 21000 (old: 21108 of 21111 eligible clips; success_rate ≥ 0.99) and `preprocess.mtat.sample_graphs` = 20 (files in `results/sample_graphs/`).
- `preprocess.musiccaps.graphs_built` = wav count (old: 5352); train/test ≈ 4817/535.
- `data/splits/` has 5 JSONs: mtat_train/val/test (≈16779/1339/2993), musiccaps_train/test.

## 4. Training (≈6–10 h total, unattended; each run checkpoints per epoch and resumes if killed)

Run in this exact order (Task 4 warm-starts from task1/task2 checkpoints):
```bash
python src/train.py --task 1
python src/train.py --task cnn
python src/train.py --task 2
python src/train.py --task 3 --variant cross_attn
python src/train.py --task 3 --variant concat
python src/train.py --task 3 --variant bert_only
python src/train.py --task 3 --variant gnn_only
python src/train.py --task 4
```
Each ends with a `TEST ...` line and writes `results/metrics.json`. Task-3 variants must show **15-epoch** histories. If a run dies mid-epoch, re-run the same command (per-epoch resume). If a run completes, its `.resume` file is auto-deleted — do not re-run completed tasks.

**Acceptance (check metrics.json after all 8 runs):**
- Every model's test Macro-F1 > `baseline_random.test.macro_f1` (computed in Step 5; old value 0.066).
- task3.variants.* Macro-F1 ≥ old value − 0.03 each. If one variant misses after one full re-run (delete its `results/checkpoints/task3_<variant>.pt`, rerun command), keep the better result and note it in the report Limitations. Do NOT tune hyperparameters.
- task4.retrieval R@10 (both directions) ≥ 15%.

## 5. Analysis artifacts (≈30–60 min GPU)

```bash
python src/evaluate.py --what all
python src/make_plots.py
```
`--what all` = B1 random baseline, 5 Task-1 examples, 10 retrieval examples + human-eval sheet, 3 case studies, comparison table; it also attempts `human_eval` ingestion, which prints `SKIP human_eval` (sheet not yet filled) — expected, not an error.

**Verify these exist and are non-empty:**
- `results/plots/`: f1_curves.png (dashed micro-F1 lines present), auc_pr_task3.png, tsne_task3.png, tsne_task3_mood.png, architecture.png (architecture.png is tracked in git and already present; if missing, run `python src/make_arch_diagram.py`).
- `results/`: metrics.json, comparison_table.md, task1_examples.md (5 DISTINCT clips — check the 5 `## clip` headers differ), case_studies.md (3 entries), retrieval_examples/retrieval_examples.md (10 queries), human_eval_sheet.csv + human_eval_instructions.md, sample_graphs/ (20 .pt).
- `results/metrics.json` keys: baseline_random, baseline_cnn, task1, task2, task3.variants.{cross_attn,concat,bert_only,gnn_only}, task4.{retrieval,zero_shot}, dataset, preprocess.

## 6. Demo notebook (≈5 min)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/demo_context.ipynb
```
Must complete without error (loads task3_cross_attn.pt + task4.pt, runs one tag prediction + one retrieval). If it errors, fix the cause (usually a missing checkpoint → the Step 4/5 run that produces it did not finish) and re-execute.

## 7. Report + README sync (markdown only; NO LaTeX compile)

Numbers in `report/final_report.md` and `README.md` must equal `results/metrics.json` exactly (round to 3 decimals as in the existing tables). Sync these spots in `report/final_report.md`:
1. **Abstract** — the four Task-3/fusion numbers, R@10, zero-shot Macro-F1, random baseline.
2. **Table 1** — all 8 rows from `results/comparison_table.md` (bold stays on the best fusion row, whichever it is).
3. **§3 Training** — the epoch list must read "Task 1: 10 epochs; Task 2: 20; CNN baseline: 15; Task 3: 15; Task 4: 20." (remove any "8 epochs / plateaus early" wording).
4. **Table 2** — retrieval R@K both directions + zero-shot rows.
5. **Analysis** — add one sentence citing the new mood t-SNE and insert figure: `![t-SNE mood](../results/plots/tsne_task3_mood.png)` with caption *"Figure 4: t-SNE of the Task-3 fused embedding z (MTAT test), coloured by mood umbrella."* (renumber nothing else; md has no auto-numbering).
6. **Limitations** — delete the sentence "Task 3 was trained for 8 epochs under a hardware/runtime constraint." Handle the human-eval sentence per Step 8 outcome (two variants given there).
7. If cross_attn no longer beats / now beats concat vs. the old text, adjust the one Analysis sentence about concat vs cross-attention to match the new table honestly.

In `README.md`: sync the results table (8 rows) and the Task-4 line with new numbers; change the setup line "tested on RTX 3050 Ti, 4 GB" to reflect this machine only if user told you its name — otherwise leave it.

## 8. Human evaluation — THE ONLY USER TOUCHPOINT (optional, 15 min)

Ask the user ONCE, at the very end:

> "Human eval (spec §6): open `results/human_eval_sheet.csv`, listen to each `top1_ytid` at https://www.youtube.com/watch?v=<top1_ytid>, fill `rating_1to5` (1–5) for all 10 rows, save, tell me. Skip if you don't want to."

- **If user fills it:** run `python src/evaluate.py --what human_eval` (writes `task4.human_eval` {n, mean, sd, scores}), then in `report/final_report.md` Limitations write: "Human evaluation (single rater, 10 caption–clip pairs, 1–5 scale) scored mean {mean} / SD {sd} (metrics `task4.human_eval`)."
- **If user declines/no response:** replace the Limitations human-eval sentence with: "Human evaluation of retrieval was not conducted; the rating sheet (`results/human_eval_sheet.csv`) is provided for future assessment." Never claim ratings that do not exist. Never fill the sheet yourself.

## 9. Final verification gate — report all of this to the user, then stop

- [ ] metrics.json complete per Step 5 checklist; all models beat `baseline_random`; Task-3 histories are 15 epochs.
- [ ] All Step 5 artifacts present; task1_examples.md has 5 distinct clips; 20 sample graphs.
- [ ] demo_context.ipynb executed clean, outputs present.
- [ ] report/final_report.md + README.md numbers == metrics.json; no stale "8 epochs" text; human-eval sentence matches Step 8 outcome.
- [ ] `git status` + `git diff --stat` summary handed to user. **No commits — user commits.**

## Failure escalation (only reasons to STOP and ask user)

1. GPU invisible after 3 CUDA index attempts (Step 1.2).
2. MTAT mirrors all fail (Step 2.1) or MusicCaps yield < 2000 (Step 2.2).
3. preprocess success_rate < 0.95 after re-runs (Step 3).
4. A Task-3 variant still < old − 0.03 after one clean re-run, or retrieval R@10 < 15% after re-run (Step 4) — actually: note honestly in Limitations and CONTINUE; only stop if metrics.json keys are missing entirely.

## Suggested skills

- None required. If user later asks for a diff review: `cavecrew` (cavecrew-reviewer). Commits: user's own job.

**Estimated wall-clock:** ~12–20 h, almost all unattended (download + preprocess + training). Agent active time ~1 h.
