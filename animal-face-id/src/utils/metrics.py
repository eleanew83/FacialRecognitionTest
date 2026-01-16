"""Evaluation metrics helpers for final evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def topk_accuracies(logits: np.ndarray, labels: np.ndarray, ks: Tuple[int, ...] = (1, 3, 5)) -> Dict[int, float]:
    """Compute top-k accuracies for provided logits and integer labels."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2D (N x C).")
    ks = tuple(sorted(set(ks)))
    max_k = min(max(ks), logits.shape[1])
    top_preds = np.argsort(-logits, axis=1)[:, :max_k]
    results: Dict[int, float] = {}
    for k in ks:
        k = min(k, logits.shape[1])
        correct = (top_preds[:, :k] == labels[:, None]).any(axis=1)
        results[k] = float(np.mean(correct))
    return results


def compute_basic_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return overall metrics including top-1, macro/weighted precision/recall/F1."""
    preds = logits.argmax(axis=1)
    return {
        "top1": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def per_class_report(logits: np.ndarray, labels: np.ndarray, class_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Return per-class metrics dict keyed by class name."""
    preds = logits.argmax(axis=1)
    report = classification_report(labels, preds, target_names=list(class_names), output_dict=True, zero_division=0)
    return {k: v for k, v in report.items() if k not in {"accuracy", "macro avg", "weighted avg"}}


def compute_confusion(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return confusion matrix (C x C)."""
    preds = logits.argmax(axis=1)
    return confusion_matrix(labels, preds)
