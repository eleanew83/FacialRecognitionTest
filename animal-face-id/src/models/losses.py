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


def build_classifier_head(head_type: str, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 30.0) -> nn.Module:
    """Return a classifier head module."""
    if head_type == "linear":
        return nn.Linear(embedding_dim, num_classes)
    if head_type == "arcface":
        return ArcFaceHead(embedding_dim=embedding_dim, num_classes=num_classes, scale=scale, margin=margin)
    msg = f"Unknown head_type '{head_type}'. Supported: linear, arcface."
    raise ValueError(msg)


def build_loss(name: str) -> nn.Module:
    """Return a configured loss object for training."""
    if name in {"cross_entropy", "ce", "arcface"}:
        return nn.CrossEntropyLoss()
    msg = f"Unknown loss '{name}'. Supported: cross_entropy."
    raise ValueError(msg)
