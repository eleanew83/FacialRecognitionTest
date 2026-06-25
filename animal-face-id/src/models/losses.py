"""Metric-learning loss builders and classifier heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin head for face recognition."""

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)
        cosine = F.linear(embeddings, weights)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return logits

    def logits_eval(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Inference-time logits WITHOUT the angular margin.

        The margin is a *training* device that needs the ground-truth label;
        applying it at test time both leaks the label and penalises the true
        class. At eval we score with plain scaled cosine similarity.
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.scale


def build_classifier_head(head_type: str, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 30.0) -> nn.Module:
    """Return a classifier head module."""
    if head_type == "linear":
        return nn.Linear(embedding_dim, num_classes)
    if head_type == "arcface":
        return ArcFaceHead(embedding_dim=embedding_dim, num_classes=num_classes, scale=scale, margin=margin)
    msg = f"Unknown head_type '{head_type}'. Supported: linear, arcface."
    raise ValueError(msg)


def class_balanced_weights(
    class_counts: "list[int] | torch.Tensor",
    scheme: str = "effective_number",
    beta: float = 0.999,
) -> torch.Tensor:
    """Per-class loss weights for long-tailed training.

    - ``effective_number`` (Cui et al. 2019): w_c ∝ (1 - beta) / (1 - beta^{n_c}).
      Softer than raw inverse frequency; ``beta`` near 1 => stronger re-balancing.
    - ``inverse``: w_c ∝ 1 / n_c.
    Weights are normalised to mean 1 so the overall loss scale is unchanged.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
    if scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "effective_number":
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / eff
    else:
        msg = f"Unknown weight scheme '{scheme}'. Supported: effective_number, inverse."
        raise ValueError(msg)
    return w * (len(w) / w.sum())  # mean ≈ 1


def build_loss(name: str, *, weight: torch.Tensor | None = None) -> nn.Module:
    """Return a configured loss object for training.

    ``class_balanced`` / ``weighted_ce`` apply per-class ``weight`` to the
    cross-entropy on the (ArcFace) logits — the long-tail recognition loss.
    """
    if name in {"cross_entropy", "ce", "arcface"}:
        return nn.CrossEntropyLoss()
    if name in {"class_balanced", "weighted_ce", "cb"}:
        return nn.CrossEntropyLoss(weight=weight)
    msg = f"Unknown loss '{name}'. Supported: cross_entropy, class_balanced."
    raise ValueError(msg)
