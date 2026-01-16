"""Dataset registry definitions and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from .animal_faces import AnimalFaceDataset
from .chimpanzee_faces import ChimpanzeeFacesDataset
from .transforms import build_transforms


def load_dataset_metadata(registry_path: str) -> dict[str, Any]:
    """Load dataset metadata (splits, augmentations) from disk."""
    with Path(registry_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _instantiate_dataset(name: str, split: str, config: dict[str, Any]) -> Any:
    transforms = build_transforms(stage=split, config=config)

    if name == "animal_faces":
        return AnimalFaceDataset(
            root=config["root"],
            split=split,
            transform=transforms,
        )
    if name == "chimpanzee_faces":
        return ChimpanzeeFacesDataset(
            raw_root=config["raw_root"],
            splits_path=config["splits_path"],
            split=split,
            transform=transforms,
        )

    msg = f"Unknown dataset '{name}'. Supported: animal_faces, chimpanzee_faces."
    raise ValueError(msg)


def build_dataloader(
    name: str,
    split: str,
    config: dict[str, Any],
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """Return a dataloader for the requested dataset split."""
    dataset = _instantiate_dataset(name=name, split=split, config=config)
    shuffle = split == "train" if shuffle is None else shuffle
    drop_last = config.get("drop_last", False) if drop_last is None else drop_last
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=shuffle,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=drop_last,
    )
