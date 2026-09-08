# Handoff — CSE425 GNN-BERT Music Context Project

**Date:** 2026-09-08
**Workspace:** `E:\Study\425\project` (git repo)
**Stage reached:** Planning complete. Zero code written. Next step = execute Phase 0.

## What to read first (in order)

1. **`E:\Study\425\project\PLAN.md`** — the full implementation plan. AUTHORITATIVE. 16 locked decisions (D1–D16), 10 phases (P0–P9) with done-criteria, acceptance thresholds, and pre-decided fallbacks. Executor must NOT re-decide anything; follow it literally.
2. **`E:\Study\425\project\CSE425_Project_GNN_BERT_Music_Context.md`** — course spec (source of truth for requirements/rubric). Plan maps every deliverable to it.
3. Repo skeleton already exists matching spec §10 structure (`src/*.py` empty, `data/`, `results/`, `report/`, `notebooks/` placeholders). Fill in place; do not restructure.

## Current state

- Only non-empty file: `src/pdf_to_markdown.py` (unrelated utility). `requirements.txt` has only `pymupdf4llm` — PLAN.md §2 specifies full replacement contents.
- User answered setup questions: **NVIDIA GPU available**, **~25 GB disk OK**, **yt-dlp YouTube downloads allowed for MusicCaps**, **Task 4 human eval = user self-rates** (plan D12).
- No datasets downloaded yet. No venv yet. pdflatex/ffmpeg presence unverified (P0 checks).

## User constraints / preferences

- Minimal user involvement. Only 3 sanctioned touchpoints (PLAN.md §13): GPU/pdflatex missing → escalate; 15-min self-rating of 10 retrieval examples after P7; final report review + commit approval.
- No git mutations without explicit user approval.
- Report (LaTeX, NeurIPS 2024 template) comes LAST (P8).
- Communication style: terse/caveman mode active for chat; code/commits/docs written normally.

## Execution instructions for next agent

1. Start at PLAN.md Phase 0 exactly as written; work sequentially P0 → P9.
2. Every phase has "Done when" criteria — verify before advancing.
3. Acceptance failures: follow the retry/fallback already specified in PLAN.md §14 and each phase; escalate to user ONLY per §13.
4. Long unattended steps (P1 downloads, P2 preprocessing, training runs): launch, monitor, verify counts/metrics as specified.
5. Do not duplicate spec/plan content into new docs; keep `results/metrics.json` as the single metrics store.

## Suggested skills (call Skill tool for these)

- **`handoff`** — at session end or before context exhaustion, to write the next handoff.
- **`pdf`** — in P8 if report PDF manipulation/extraction needed beyond pdflatex compile.
- **`academic-paper`** — in P8 for NeurIPS-style report drafting (structure, citations, quality pass).
- **`caveman-commit`** — when user approves final commit (P9).
- No skill needed for P0–P7 core ML work; follow PLAN.md directly.

## Sensitive info

None present. No API keys, credentials, or PII in workspace or conversation. MusicCaps download uses public YouTube URLs only.
