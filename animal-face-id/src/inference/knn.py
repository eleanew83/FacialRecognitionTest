"""Lightweight k-NN utilities for embedding retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class EmbeddingIndex:
    """Wrapper around sklearn's NearestNeighbors for future re-use."""

    def __init__(self, metric: str = "cosine", top_k: int = 5) -> None:
        self.metric = metric
        self.top_k = top_k
        self.model = NearestNeighbors(metric=metric, n_neighbors=top_k)
        self.gallery_ids: list[str] = []

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        """Fit the index using pre-computed embeddings."""
        self.model.fit(embeddings)
        self.gallery_ids = labels

    def query(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Return the nearest ids and distances for a single embedding."""
        distances, indices = self.model.kneighbors([embedding], return_distance=True)
        hits: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            label = self.gallery_ids[idx]
            hits.append((label, float(dist)))
        return hits

    def save(self, path: str | Path) -> None:
        """Persist the fitted index to disk."""
        payload = {"model": self.model, "gallery_ids": self.gallery_ids, "metric": self.metric, "top_k": self.top_k}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        """Load a previously saved index."""
        payload: dict[str, Any] = joblib.load(path)
        instance = cls(metric=payload["metric"], top_k=payload["top_k"])
        instance.model = payload["model"]
        instance.gallery_ids = payload["gallery_ids"]
        return instance
