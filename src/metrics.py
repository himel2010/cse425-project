"""Multi-label metrics: Macro/Micro-F1, mean AUC-PR."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # mean AUC-PR over labels that have at least one positive
    aps = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0:
            aps.append(average_precision_score(y_true[:, j], y_prob[:, j]))
    mean_ap = float(np.mean(aps)) if aps else 0.0

    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "mean_auc_pr": round(mean_ap, 4),
    }


def random_baseline_metrics(y_true: np.ndarray) -> dict:
    """B1: analytic expected metrics from tag priors (PLAN.md P7.1)."""
    y_true = np.asarray(y_true)
    p = y_true.mean(axis=0)  # per-label positive rate = predict-positive prob
    # Expected micro over all labels: predicting positive w.p. p_j independently.
    # Precision_j = p_j (of predicted-positive, fraction truly positive = p_j).
    # Recall_j = p_j. F1_j = p_j. Macro = mean(p_j); AUC-PR baseline = p_j.
    macro_f1 = float(np.mean(p))
    mean_ap = float(np.mean(p))
    tp = np.sum(p * p) * y_true.shape[0]
    pred_pos = np.sum(p) * y_true.shape[0]
    actual_pos = y_true.sum()
    micro_f1 = float(2 * tp / (pred_pos + actual_pos)) if (pred_pos + actual_pos) else 0.0
    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "mean_auc_pr": round(mean_ap, 4),
    }
