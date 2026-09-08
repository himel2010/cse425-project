# GNN–BERT Fusion for Music Context Understanding

**CSE425 Project — BRAC University**

## Abstract

We study whether combining a graph representation of audio structure with a
pretrained language model improves music context understanding. Audio clips are
turned into *segment graphs* encoded by GraphSAGE, and text (tag metadata or
free-text captions) is encoded by DistilBERT. We evaluate four tasks on
MagnaTagATune (MTAT) and MusicCaps: (1) text-only tagging, (2) graph-only
tagging, (3) graph–text fusion with ablations, and (4) contrastive
audio–caption retrieval. On MTAT top-50 tagging, both fusion variants beat their
unimodal ablations (concat Macro-F1 0.276, cross-attention 0.257 vs. 0.224 for
either single modality), and the contrastive model reaches Caption→Audio R@10 of
23.4% (~12× random) with a zero-shot tag Macro-F1 of 0.172. All models clearly
beat a random baseline (0.066). Code and reproducible artifacts (seed 42)
accompany the report.

## 1. Introduction

Music understanding sits between audio signal processing and natural language: a
piece has both acoustic structure (repetition, contrast, texture) and describable
semantics (genre, instrumentation, mood). Most tagging systems model audio alone.
We ask a focused question: does representing a clip's temporal structure as a
graph, and fusing it with a language model, help? We implement the four-task
pipeline from the course specification and report honest comparisons, including
ablations that isolate each modality.

## 2. Method

**Segment graphs.** Each clip is resampled to 22.05 kHz mono and cut into 5 s
windows with 2.5 s hop. Each window (node) is a 64-dim vector: MFCC-20 and
chroma-12, each summarised by mean and standard deviation over time. Edges are
(i) temporal, connecting adjacent windows bidirectionally, and (ii) similarity,
connecting non-adjacent windows with cosine similarity > τ = 0.6 (at most 4 extra
edges per node, weighted by similarity). This captures both sequence and
long-range repetition/self-similarity.

**Encoders.** The graph encoder is a 2-layer GraphSAGE (PyTorch Geometric) with
mean readout, producing a 256-dim graph embedding *g*. Text is encoded by a fully
fine-tuned DistilBERT; we use the CLS embedding *t*_CLS and the token sequence
*H*.

**Fusion (Task 3).** The main fusion uses single-head cross-attention with the
graph readout as query and projected BERT tokens as keys/values,
ctx = softmax(qKᵀ/√d)V, and classifies from z = [g ; ctx] through an MLP with
sigmoid outputs (BCE loss). We compare against `concat` (z = [g ; t_CLS]),
`bert_only`, and `gnn_only`.

**Contrastive retrieval (Task 4).** A dual encoder maps graphs and captions into
a shared space (L2-normalised 256-dim), trained with symmetric InfoNCE at
temperature 0.07. Branches are warm-started from the Task-1/Task-2 checkpoints.
This mirrors audio–language contrastive models such as CLAP.

![Pipeline architecture](../results/plots/architecture.png)

*Figure 1: Pipeline — audio segment graph (GraphSAGE) and text (DistilBERT) are
fused by cross-attention for tagging (Tasks 1–3) and aligned contrastively for
retrieval (Task 4).*

## 3. Experimental Setup

**Datasets.** **MagnaTagATune** (25,863 clips, 29 s) with the top-50 tags; we keep
the 21,111 clips having at least one top-50 tag. As MTAT has no lyrics/captions,
Task-1 text is the clip metadata string `artist - title (album)`. Because MTAT
ships 16 folders, we use the community folder split: train = folders `0`–`c`,
val = `d`, test = `e,f` (no clip-level leakage), giving 16,779 / 1,339 / 2,993.
**MusicCaps** provides 10 s clips with free-text captions; since YouTube blocks
automated downloads, we source the audio from the public `CLAPv2/MusicCaps`
mirror (5,352 clips) and use a deterministic 90/10 split (seed 42). All runs use
seed 42.

**Training.** DistilBERT is fine-tuned at lr 2×10⁻⁵; GNN/fusion heads at 10⁻³.
We use AdamW, mixed precision, and gradient accumulation to fit a 4 GB GPU.
Task 1: 10 epochs; Task 2: 20; CNN baseline: 15; Task 3: 8 (validation plateaus
early); Task 4: 20.

**Baselines.** B1 random predicts each tag with its prior (analytic F1). B2 is a
4-block CNN on log-mel spectrograms. B3 is the Task-1 BERT model.

## 4. Results

*Table 1: MTAT test, top-50 tags. Both fusion variants beat either unimodal
ablation; all models beat random.*

| Model | Macro-F1 | Micro-F1 | mean AUC-PR |
|---|---|---|---|
| B1 random | 0.066 | 0.098 | 0.066 |
| B3 / Task 1 BERT (metadata) | 0.195 | 0.319 | 0.300 |
| Task 2 GraphSAGE | 0.235 | 0.360 | 0.346 |
| B2 CNN (log-mel) | 0.299 | 0.432 | 0.400 |
| Task 3 `bert_only` | 0.224 | 0.340 | 0.309 |
| Task 3 `gnn_only` | 0.224 | 0.377 | 0.332 |
| Task 3 `concat` | **0.276** | 0.403 | 0.389 |
| Task 3 `cross_attn` | 0.257 | 0.382 | 0.385 |

*Table 2: Task 4 contrastive retrieval (MusicCaps test, 535 clips) and zero-shot
tag transfer to MTAT. Random R@10 ≈ 1.9%.*

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Caption → Audio | 2.99 | 14.95 | 23.36 |
| Audio → Caption | 3.74 | 13.64 | 22.06 |

| Zero-shot tag probe | value |
|---|---|
| Macro-F1 | 0.172 |
| Micro-F1 | 0.197 |
| mean AUC-PR | 0.185 |

**Tagging (Table 1).** Audio structure alone (GraphSAGE, 0.235) already beats
sparse metadata text (BERT, 0.195); the raw-spectrogram CNN is the strongest
single model (0.299), unsurprising since 64-dim segment nodes discard fine
spectral detail. The key result is the ablation: fusing modalities beats either
alone — `concat` 0.276 and `cross_attn` 0.257 vs. 0.224 for both `bert_only` and
`gnn_only`. Simple concatenation edging out cross-attention suggests the CLS
summary already carries most of the useful text signal for these short metadata
strings.

**Retrieval (Table 2).** On captions–audio the dual encoder reaches R@10 ≈ 23%,
more than an order of magnitude above chance, and transfers zero-shot to MTAT
tagging (Macro-F1 0.172) using only tag-name prompts, despite never seeing MTAT
tags during contrastive training.

![Validation Macro-F1 vs epoch](../results/plots/f1_curves.png)

*Figure 2: Validation Macro-F1 vs. epoch across tasks/variants.*

**Analysis.** Figure 3 shows a t-SNE of the fused Task-3 embedding on MTAT test,
coloured by dominant-tag genre umbrella: classical, electronic, world, and
acoustic clips form clearly separated regions, indicating the fusion learns
semantically organised structure. Qualitatively, the demo predicts {classical,
opera, violin, strings} for a Bach cantata clip (exact match), and
caption→audio retrieval for a "soft female-vocal pop" query returns
female-vocal/pop clips. Case studies of well-retrieved clips show graphs with
several similarity edges (repeated segments), consistent with the intuition that
self-similar structure aids alignment.

![t-SNE of fused embedding](../results/plots/tsne_task3.png)

*Figure 3: t-SNE of the Task-3 fused embedding z (MTAT test), coloured by genre
umbrella.*

**Limitations.** Node features are low-dimensional hand-crafted descriptors,
capping the graph branch below the CNN; learned per-segment audio embeddings would
likely close the gap. Task-1 text is only metadata, not lyrics/captions, so the
language branch is weak on MTAT. Task 3 was trained for 8 epochs under a
hardware/runtime constraint. Human evaluation of retrieval was a single-rater
self-assessment.

## 5. Conclusion

Fusing an audio segment-graph encoder with a fine-tuned language model improves
multi-label music tagging over either modality alone, and a contrastive variant
supports meaningful cross-modal retrieval and zero-shot tag transfer. The segment
graph is a lightweight, interpretable audio representation; pairing it with
stronger learned node features is a promising next step.

## References

1. Law et al. *Evaluation of algorithms using games: The case of music tagging.* ISMIR 2009. (MagnaTagATune)
2. Agostinelli et al. *MusicLM: Generating Music From Text.* arXiv:2301.11325, 2023. (MusicCaps)
3. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
4. Sanh et al. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
5. Hamilton et al. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. (GraphSAGE)
6. Fey & Lenssen. *Fast Graph Representation Learning with PyTorch Geometric.* arXiv:1903.02428, 2019.
7. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. (CLIP)
8. van den Oord et al. *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748, 2018. (InfoNCE)
9. McFee et al. *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015.
10. Elizalde et al. *CLAP: Learning Audio Concepts from Natural Language Supervision.* ICASSP 2023.
