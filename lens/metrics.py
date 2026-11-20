"""Binary metrics for the balanced, per-generator protocol."""
from __future__ import annotations

import numpy as np


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    neg = np.sort(neg)
    lower = np.searchsorted(neg, pos, side="left")
    upper = np.searchsorted(neg, pos, side="right")
    return float((lower + upper).sum() / (2 * len(pos) * len(neg)))


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    true_positives = np.cumsum(labels[order] == 1)
    threshold_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    tp_at_threshold = true_positives[threshold_ends]
    precision = tp_at_threshold / (threshold_ends + 1)
    new_positives = np.diff(tp_at_threshold, prepend=0)
    return float(np.dot(precision, new_positives) / true_positives[-1])
