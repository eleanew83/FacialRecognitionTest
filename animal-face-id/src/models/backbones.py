"""Backbone network definitions for animal face embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedding(nn.Module):
    """Thin wrapper that exposes a ResNet backbone with a projection head."""

    def __init__(self, variant: str, embedding_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        if variant == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif variant == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            msg = f"Unsupported ResNet variant '{variant}'."
            raise ValueError(msg)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features
        self.projection = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return L2-normalized embeddings."""
        feats = self.feature_extractor(x)
        feats = feats.flatten(1)
        embeddings = self.projection(feats)
        return nn.functional.normalize(embeddings, dim=-1)


def build_backbone(model_name: str, embedding_dim: int, pretrained: bool = True) -> Any:
    """Return a backbone model configured for the requested embedding size."""
    name = model_name.lower()
    if name in {"resnet18", "resnet50"}:
        return ResNetEmbedding(variant=name, embedding_dim=embedding_dim, pretrained=pretrained)
    msg = f"Unsupported backbone '{model_name}'. Supported: resnet18, resnet50."
    raise ValueError(msg)
