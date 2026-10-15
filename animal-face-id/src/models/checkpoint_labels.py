"""Keep classifier output order bound to the checkpoint that learned it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def checkpoint_class_names(checkpoint: dict, path: str | Path, count: int) -> list[str]:
    names = checkpoint.get("class_names")
    if names is None:
        sidecar = Path(str(path) + ".labels.json")
        if not sidecar.is_file():
            raise ValueError(f"Checkpoint has no saved class order: {path}; supply its original label sidecar")
        payload = json.loads(sidecar.read_text())
        # Bind legacy sidecars to the actual checkpoint, not just its filename.
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != payload["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint label sidecar belongs to different weights: {path}")
        names = payload["class_names"]
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"Checkpoint class order does not match its {count} classifier outputs")
    return list(names)


def output_groups(names: list[str], registry=None, target_names=None):
    resolved = [registry.resolve(n) for n in names] if registry is not None else names
    targets = list(target_names) if target_names is not None else sorted(set(resolved))
    if set(targets) != set(resolved):
        raise ValueError("Dataset classes and checkpoint classes differ; cannot score this evaluation")
    return targets, [[i for i, name in enumerate(resolved) if name == target] for target in targets]


def aggregate_logits(logits, groups):
    """Sum old aliases' softmax mass using logsumexp, without changing weights."""
    import torch
    if len(groups) == logits.shape[1] and all(g == [i] for i, g in enumerate(groups)):
        return logits
    return torch.stack([torch.logsumexp(logits[:, group], dim=1) for group in groups], dim=1)


def canonical_evaluation_tag(tag, version):
    """Keep canonical-taxonomy reports separate from historical 156-ID scores."""
    suffix = "canonical_" + version.replace("-", "")
    return tag if tag and suffix in tag else f"{tag}_{suffix}" if tag else suffix
