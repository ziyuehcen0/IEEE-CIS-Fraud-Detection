"""Evaluation and threshold choice against analyst capacity."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_for_capacity(scores, n_days: float, daily_capacity: int) -> float:
    """Lowest threshold whose expected alert volume stays within capacity."""
    scores = np.asarray(scores, dtype=float)
    budget = int(np.floor(daily_capacity * n_days))
    if budget <= 0:
        return float("inf")
    if budget >= len(scores):
        return float(scores.min())
    ranked = np.sort(scores)[::-1]
    t = ranked[budget - 1]
    # Ties at the cut could push volume over budget: step just above them if needed.
    if (scores >= t).sum() > budget:
        t = np.nextafter(t, np.inf)
    return float(t)


def evaluate(y, scores, threshold: float, n_days: float, alert=None) -> dict:
    """`alert` overrides `scores >= threshold` (e.g. the flags the endpoint actually returned)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    alert = s >= threshold if alert is None else np.asarray(alert, dtype=bool)
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    pos = int(y.sum())
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if 0 < pos < len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < pos < len(y) else float("nan"),
        "threshold": float(threshold),
        "alerts_per_day": float(alert.sum() / n_days) if n_days else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / pos if pos else float("nan"),
        "false_positives": fp,
    }
