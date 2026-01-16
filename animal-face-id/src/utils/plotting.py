"""Plotting helpers for final evaluation outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Iterable


def plot_confusion_matrix(cm: np.ndarray, class_labels: Iterable[str] | None, out_path: str) -> None:
    """Save a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # If too many classes, skip tick labels to keep it readable.
    num_classes = cm.shape[0]
    if class_labels and num_classes <= 30:
        ticks = np.arange(num_classes)
        plt.xticks(ticks, class_labels, rotation=90)
        plt.yticks(ticks, class_labels)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk(topk_acc: Dict[int, float], out_path: str) -> None:
    """Save a bar plot for top-k accuracies."""
    ks = sorted(topk_acc.keys())
    vals = [topk_acc[k] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in ks], vals, color="#4c72b0")
    plt.ylim(0, 1)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy")
    for x, v in zip(ks, vals):
        plt.text(str(x), v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_class_accuracy(sorted_items: list[tuple[str, float]], out_path: str, top_n: int = 15) -> None:
    """Plot best and worst per-class accuracies."""
    if not sorted_items:
        return
    worst = sorted_items[:top_n]
    best = sorted_items[-top_n:] if len(sorted_items) > top_n else []

    fig, axes = plt.subplots(1, 2 if best else 1, figsize=(12, 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].barh([w[0] for w in worst], [w[1] for w in worst], color="#c44e52")
    axes[0].set_title("Worst classes (acc)")
    axes[0].set_xlabel("Accuracy")

    if best:
        axes[1].barh([b[0] for b in best], [b[1] for b in best], color="#55a868")
        axes[1].set_title("Best classes (acc)")
        axes[1].set_xlabel("Accuracy")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
