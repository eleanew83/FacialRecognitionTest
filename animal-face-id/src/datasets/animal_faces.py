"""Dataset for cropped animal faces stored in class-labelled folders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


class AnimalFaceDataset(Dataset):
    """Minimal dataset wrapper around animal face crops."""

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self) -> list[tuple[Path, int]]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            msg = f"Dataset split '{self.split}' not found under {self.root}."
            raise FileNotFoundError(msg)

        samples: list[tuple[Path, int]] = []
        class_to_idx: dict[str, int] = {}
        for idx, class_dir in enumerate(sorted(p for p in split_dir.iterdir() if p.is_dir())):
            class_to_idx[class_dir.name] = idx
            for image_path in class_dir.glob("*.jpg"):
                samples.append((image_path, idx))
            for image_path in class_dir.glob("*.png"):
                samples.append((image_path, idx))

        if not samples:
            msg = f"No image files discovered in {split_dir}."
            raise RuntimeError(msg)

        self.class_to_idx = class_to_idx
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label, "path": str(image_path)}

    @property
    def classes(self) -> Sequence[str]:
        """List of class labels in canonical order."""
        return tuple(self.class_to_idx.keys())

