"""Pool class probabilities across deterministic rotated views."""

from __future__ import annotations

import torch

ROTATION_ANGLES = (-30, -20, -10, 0, 10, 20, 30)


def pool_view_probabilities(probabilities: torch.Tensor, mode: str) -> torch.Tensor:
    """Input [batch, views, classes]; return normalized class scores.

    Summing and averaging give identical class rankings with a fixed view count.
    Max means the maximum probability per class across views, then renormalize.
    """
    if probabilities.ndim != 3 or probabilities.shape[1] == 0:
        raise ValueError("Expected nonempty [batch, views, classes] probabilities")
    if mode == "mean":
        pooled = probabilities.mean(dim=1)
    elif mode == "max":
        pooled = probabilities.amax(dim=1)
    else:
        raise ValueError(f"Unknown TTA pooling mode: {mode}")
    return pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
