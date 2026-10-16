"""Augmentation and preprocessing pipelines.

Augmentation profiles, selected via ``config['augment']``:

- ``"basic"`` (default, back-compatible): the original light pipeline.
- ``"strong"``: RandAugment-based auto policy + rotation / shift / blur /
  occlusion, with **no horizontal flip** (macaque faces never appear mirrored).
  RandAugment is restricted to identity-preserving ops (no solarise / invert /
  posterise / equalise) because those destroy the fur-colour and face-mark cues
  that carry individual identity in fine-grained ReID.
- ``"aggressive"``: configurable affine rotation / zoom / translation with
  colour jitter, blur, tensor Gaussian noise and erasing; no horizontal flip.
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode, functional as TF

# Ops that can erase identity cues for fine-grained face ID — excluded from
# the restricted RandAugment policy.
_DESTRUCTIVE_OPS = {"Solarize", "Posterize", "Invert", "Equalize", "AutoContrast"}


class RestrictedRandAugment(T.RandAugment):
    """RandAugment with colour-destroying operations removed."""

    def _augmentation_space(self, num_bins: int, image_size):  # type: ignore[override]
        space = super()._augmentation_space(num_bins, image_size)
        return {k: v for k, v in space.items() if k not in _DESTRUCTIVE_OPS}


class GaussianNoise(torch.nn.Module):
    """Float [0, 1] fallback for torchvision versions without v2.GaussianNoise."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image + torch.randn_like(image) * self.sigma).clamp(0, 1)


def _gaussian_noise(sigma: float):
    from torchvision.transforms import v2

    if hasattr(v2, "GaussianNoise"):
        return v2.GaussianNoise(mean=0.0, sigma=sigma, clip=True)
    return GaussianNoise(sigma=sigma)


class RotationViews:
    """Deterministic views, rotated before normalization; output [V, C, H, W]."""

    def __init__(self, image_size, mean, std, angles) -> None:
        self.resize = T.Resize((image_size, image_size))
        self.finish = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.angles = tuple(float(angle) for angle in angles)
        if not self.angles:
            raise ValueError("Rotation TTA requires at least one angle")
        self.fill = tuple(round(255 * value) for value in mean)

    def __call__(self, image):
        image = self.resize(image)
        return torch.stack([
            self.finish(TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR,
                                  fill=self.fill) if angle else image)
            for angle in self.angles
        ])


def build_transforms(stage: str, config: dict[str, Any]) -> Any:
    """Return composed transforms for the specified pipeline stage."""
    image_size = config.get("image_size", 224)
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    profile = config.get("augment", "basic")

    if stage != "train":
        if config.get("tta_angles") is not None:
            return RotationViews(image_size, mean, std, config["tta_angles"])
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    if profile == "aggressive":
        aug = config["aggressive"]
        # A single geometry operation keeps the configured rotation range exact.
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomAffine(
                degrees=aug["degrees"], translate=tuple(aug["translate"]),
                scale=tuple(aug["scale"]), interpolation=InterpolationMode.BILINEAR,
                fill=tuple(round(255 * value) for value in mean),
            ),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=tuple(aug["blur_sigma"]))],
                          p=aug["blur_p"]),
            T.ToTensor(),
            T.RandomApply([_gaussian_noise(aug["noise_sigma"])], p=aug["noise_p"]),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=aug["erase_p"], scale=tuple(aug["erase_scale"]), value="random"),
        ])

    if profile == "strong":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.85, 1.18)),
            # geometry: rotation + shift (translate) + mild zoom — NO horizontal flip
            T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            # auto photometric policy (restricted to identity-preserving ops)
            RestrictedRandAugment(num_ops=2, magnitude=config.get("randaug_magnitude", 5)),
            # blur
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.35),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # occlusion (fur patches / obstructions)
            T.RandomErasing(p=0.30, scale=(0.02, 0.12), value="random"),
        ])

    # "basic" — original light pipeline (kept for reproducibility of the baseline)
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])
