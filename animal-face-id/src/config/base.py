"""Configuration dataclasses and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Structured config for training runs."""

    name: str
    data: dict[str, Any]
    model: dict[str, Any]
    trainer: dict[str, Any]
    inference: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a nested dictionary representation."""
        return {
            "name": self.name,
            "data": self.data,
            "model": self.model,
            "trainer": self.trainer,
            "inference": self.inference or {},
        }


def load_config(path: str) -> TrainingConfig:
    """Load a structured config object from disk."""
    payload: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(
        name=payload.get("name", "animal-face-id"),
        data=payload.get("data", {}),
        model=payload.get("model", {}),
        trainer=payload.get("trainer", {}),
        inference=payload.get("inference"),
    )
