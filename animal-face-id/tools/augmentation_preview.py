"""Visualise the proposed training augmentation pipeline on real macaque crops.

Shows original vs several augmented versions for a few head and tail IDs, so we
can sanity-check the augmentations BEFORE committing to a retrain.

Implements the supervisor's requests:
  - auto policy (RandAugment) instead of hand-tuning each op
  - rotation, lighting/colour, blur, shift, occlusion (RandomErasing)
  - NO horizontal flip (macaques don't appear mirrored)

Output: artifacts/longtail_analysis/augmentation_preview.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.transforms import RestrictedRandAugment

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "macaque_faces" / "splits.json"
RAW_ROOT = Path(
    "/rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/"
    "yolo_detection_code/output/macaque_crops"
)
OUT = ROOT / "artifacts" / "longtail_analysis" / "augmentation_preview.png"
IMAGE_SIZE = 224
N_AUG = 5  # augmented variants per image
torch.manual_seed(0)


# ----------------------------------------------------------------------
# Proposed augmentation pipeline (viewable form: ends at ToTensor, no Normalize)
# The real training transform would append T.Normalize(mean,std) after this.
# ----------------------------------------------------------------------
def build_preview_transform():
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.85, 1.18)),
        # rotation + shift (translate) + mild zoom — NO horizontal flip
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        # auto policy (restricted: no solarise/invert/posterise/equalise) at
        # moderate magnitude — fine-grained ID is sensitive to heavy colour ops
        # that can erase identity cues (fur colour, face marks).
        RestrictedRandAugment(num_ops=2, magnitude=5),
        # blur
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.35),
        T.ToTensor(),
        # occlusion (fur patches / obstructions)
        T.RandomErasing(p=0.30, scale=(0.02, 0.12), value="random"),
    ])


def to_view(t: torch.Tensor):
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    splits = json.loads(SPLITS.read_text())
    from collections import Counter
    counts = Counter(it["id"] for it in splits["train"])
    ranked = sorted(counts, key=lambda k: counts[k])
    tail_ids = ranked[:2]              # fewest images
    head_ids = ranked[-2:]            # most images
    chosen = head_ids + tail_ids

    # one representative image per chosen ID
    first_path = {}
    for it in splits["train"]:
        if it["id"] in chosen and it["id"] not in first_path:
            first_path[it["id"]] = RAW_ROOT / it["path"]

    tfm = build_preview_transform()
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))

    n_rows = len(chosen)
    n_cols = N_AUG + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))

    for r, cid in enumerate(chosen):
        img = Image.open(first_path[cid]).convert("RGB")
        grp = "HEAD" if cid in head_ids else "TAIL"
        # original
        ax = axes[r, 0]
        ax.imshow(resize(img))
        ax.set_title("original", fontsize=9)
        ax.set_ylabel(f"{cid}\n[{grp}] {counts[cid]} imgs", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        # augmented
        for c in range(1, n_cols):
            ax = axes[r, c]
            ax.imshow(to_view(tfm(img)))
            if r == 0:
                ax.set_title(f"aug {c}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        "Proposed augmentation — RandAugment + affine/blur/erasing, NO h-flip",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT, dpi=140)
    print("Saved:", OUT)
    print("Head IDs:", head_ids, "| Tail IDs:", tail_ids)


if __name__ == "__main__":
    main()
