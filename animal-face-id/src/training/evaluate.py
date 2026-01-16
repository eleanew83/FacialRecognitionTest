"""Evaluation helpers for closed/open-set metrics."""

from typing import Any

def evaluate_closed_set(checkpoint_path: str, config: dict[str, Any]) -> dict[str, float]:
    """Compute closed-set metrics for a trained model."""
    ...


def evaluate_open_set(embeddings_path: str, thresholds: dict[str, float]) -> dict[str, float]:
    """Compute open-set performance numbers (FAR, FRR, AUROC)."""
    ...
