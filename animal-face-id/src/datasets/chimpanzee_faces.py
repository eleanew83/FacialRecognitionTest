"""Dataset loader for chimpanzee faces using annotation + split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset


class ChimpanzeeFacesDataset(Dataset):
    """Dataset that reads images referenced in split manifests."""

    def __init__(
        self,
        raw_root: str | Path,
        splits_path: str | Path,
        split: str,
        transform: Any | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.splits_path = Path(splits_path)
        self.split = split
        self.transform = transform

        payload = json.loads(self.splits_path.read_text(encoding="utf-8"))
        if self.split not in payload:
            msg = f"Split '{self.split}' not found in {self.splits_path}."
            raise ValueError(msg)

        self.samples: List[Dict[str, str]] = payload[self.split]
        all_labels = {item["id"] for entries in payload.values() for item in entries}
        self.class_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(sorted(all_labels))}

        # Sanity check paths
        missing = []
        for item in self.samples:
            full_path = self.raw_root / item["path"]
            if not full_path.exists():
                missing.append(item["path"])
                if len(missing) > 10:
                    break
        if missing:
            msg = f"Missing files detected (showing up to 10): {missing[:10]}"
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.raw_root / sample["path"]
        label_name = sample["id"]
        label_idx = self.class_to_idx[label_name]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": label_idx,
            "id": label_name,
            "path": str(image_path),
        }

    @property
    def classes(self) -> Sequence[str]:
        """Return class labels in index order."""
        return tuple(sorted(self.class_to_idx, key=lambda k: self.class_to_idx[k]))
