"""Augmentation and preprocessing stubs."""

from __future__ import annotations

from typing import Any

from torchvision import transforms as T


def build_transforms(stage: str, config: dict[str, Any]) -> Any:
    """Return composed transforms for the specified pipeline stage."""
    image_size = config.get("image_size", 224)
    aug_list: list[Any] = []
    if stage == "train":
        aug_list.extend(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.RandomHorizontalFlip(),
            ],
        )
    else:
        aug_list.append(T.Resize((image_size, image_size)))

    aug_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=config.get("mean", [0.485, 0.456, 0.406]), std=config.get("std", [0.229, 0.224, 0.225])),
        ],
    )
    return T.Compose(aug_list)
