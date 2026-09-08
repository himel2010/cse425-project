# Model comparison (MTAT test)

| Model | Macro-F1 | Micro-F1 | mean AUC-PR |
|---|---|---|---|
| B1 random | 0.066 | 0.0981 | 0.066 |
| B2 CNN | 0.2991 | 0.4317 | 0.3998 |
| B3/Task1 BERT | 0.1951 | 0.3194 | 0.3003 |
| Task2 GNN | 0.2349 | 0.3596 | 0.3464 |
| Task3 bert_only | 0.2245 | 0.3404 | 0.3087 |
| Task3 gnn_only | 0.224 | 0.3773 | 0.3317 |
| Task3 concat | 0.2756 | 0.4027 | 0.3887 |
| Task3 cross_attn | 0.2569 | 0.3823 | 0.3847 |

## Task 4 retrieval (MusicCaps test)
- Caption->Audio: {'R@1': 2.99, 'R@10': 23.36, 'R@5': 14.95}
- Audio->Caption: {'R@1': 3.74, 'R@10': 22.06, 'R@5': 13.64}
- Zero-shot tag probe: {'macro_f1': 0.1723, 'mean_auc_pr': 0.1854, 'micro_f1': 0.1968, 'threshold': 0.3216}