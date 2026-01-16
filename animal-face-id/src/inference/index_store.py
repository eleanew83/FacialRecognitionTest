"""Lightweight embedding index utilities for gallery management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.inference.inference_core import infer_single_image, load_model_from_config


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(1, -1)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


class SimpleIndex:
    """Minimal cosine-similarity index for embeddings."""

    def __init__(self, embeddings: np.ndarray | None = None, labels: Sequence[str] | None = None, meta: List[Dict[str, Any]] | None = None) -> None:
        self.embeddings = embeddings if embeddings is not None else np.zeros((0, 0), dtype=np.float32)
        self.labels = list(labels) if labels is not None else []
        self.meta = meta or []

    @property
    def size(self) -> int:
        return len(self.labels)

    def add(self, embeddings: np.ndarray, labels: Sequence[str], meta: List[Dict[str, Any]] | None = None) -> None:
        embeddings = np.asarray(embeddings)
        if self.embeddings.size == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.labels.extend(labels)
        if meta:
            self.meta.extend(meta)
        else:
            self.meta.extend([{} for _ in labels])

    def query(self, embedding: np.ndarray, topk: int = 5) -> List[Dict[str, Any]]:
        if self.size == 0:
            return []
        sims = _cosine_similarity(self.embeddings, embedding.reshape(1, -1))
        sims = sims[:, 0]
        topk = min(topk, self.size)
        idxs = np.argsort(-sims)[:topk]
        results = []
        for idx in idxs:
            results.append({"label": self.labels[idx], "score": float(sims[idx]), "meta": self.meta[idx] if idx < len(self.meta) else {}})
        return results

    def save(self, path_prefix: str) -> None:
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix) + "_embeddings.npy", self.embeddings)
        payload = {"labels": self.labels, "meta": self.meta}
        with open(str(prefix) + "_meta.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path_prefix: str) -> "SimpleIndex":
        prefix = Path(path_prefix)
        embeddings = np.load(str(prefix) + "_embeddings.npy")
        with open(str(prefix) + "_meta.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(embeddings=embeddings, labels=payload.get("labels", []), meta=payload.get("meta", []))


def build_index_from_folder(model_bundle: Dict[str, Any], root_dir: str, glob_pattern: str = "**/*.png") -> SimpleIndex:
    """Build index from a folder of images; label = parent folder name."""
    root = Path(root_dir)
    paths = list(root.glob(glob_pattern))
    if not paths:
        return SimpleIndex()
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    meta: List[Dict[str, Any]] = []
    for path in paths:
        label = path.parent.name
        result = infer_single_image(model_bundle, str(path), topk=5)
        embeddings.append(result["embedding"])
        labels.append(label)
        meta.append({"image_path": str(path)})
    index = SimpleIndex()
    if embeddings:
        index.add(np.stack(embeddings, axis=0), labels, meta)
    return index


def add_individual(model_bundle: Dict[str, Any], index: SimpleIndex, name: str, image_paths: Sequence[str], aggregate: bool = False) -> None:
    """Add a new individual by embedding provided images."""
    embeddings: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    for path in image_paths:
        result = infer_single_image(model_bundle, str(path), topk=5)
        embeddings.append(result["embedding"])
        meta.append({"image_path": str(path)})

    if not embeddings:
        return

    if aggregate:
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0, keepdims=True)
        index.add(mean_emb, [name], [{"image_paths": list(image_paths)}])
    else:
        labels = [name] * len(embeddings)
        index.add(np.stack(embeddings, axis=0), labels, meta)
