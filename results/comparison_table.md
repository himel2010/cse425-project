# Model comparison (MTAT test)

| Model | Macro-F1 | Micro-F1 | mean AUC-PR |
|---|---|---|---|
| B1 random | 0.066 | 0.0981 | 0.066 |
| B2 CNN | 0.2902 | 0.4274 | 0.3996 |
| B3/Task1 BERT | 0.1886 | 0.3184 | 0.302 |
| Task2 GNN | 0.2319 | 0.353 | 0.3502 |
| Task3 bert_only | 0.2168 | 0.3333 | 0.2936 |
| Task3 gnn_only | 0.2422 | 0.3576 | 0.3525 |
| Task3 concat | 0.2784 | 0.4003 | 0.3776 |
| Task3 cross_attn | 0.2721 | 0.4039 | 0.3777 |

## Task 4 retrieval (MusicCaps test)
- Caption->Audio: {'R@1': 5.23, 'R@10': 21.87, 'R@5': 14.58}
- Audio->Caption: {'R@1': 4.11, 'R@10': 20.75, 'R@5': 13.27}
- Zero-shot tag probe: {'macro_f1': 0.1639, 'mean_auc_pr': 0.1616, 'micro_f1': 0.1789, 'threshold': 0.1508}