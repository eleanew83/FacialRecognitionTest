#!/usr/bin/env python
"""
SAM 3 macaque detection test — 3 scenarios, ~3 sample images each.

NOTE: This is a validation test, NOT a full analysis.
      It samples ~9 images to confirm SAM 3 works correctly before
      running on all 9,896 images.

Scenarios:
  1. Single macaque face   — images from individual folders, no "+" in filename
  2. Multiple macaque faces — images whose filenames contain "+" (multiple named animals)
  3. Macaque + human face  — images from tourist locations (Cable Car, Prince Philip Arch)

Results saved to:
  .../sam3_experiments/results/
"""

import random
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

PHOTOS_DIR  = Path("/rds/user/ylj20/hpc-work/Gibraltar_Macaques_Photos")
RESULTS_DIR = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_SCENARIO = 3
random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_images(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not p.name.startswith("._")
    ]


def draw_results(image, masks, boxes, scores, prompt, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=10, wrap=True)
    ax1.imshow(image); ax1.set_title("Original"); ax1.axis("off")
    ax2.imshow(image)
    ax2.set_title(f"SAM 3: '{prompt}' → {len(boxes)} detected")
    ax2.axis("off")

    colours = plt.cm.Set1(np.linspace(0, 1, max(len(boxes), 1)))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        c = colours[i % len(colours)]
        ax2.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=c, facecolor="none"
        ))
        ax2.text(x1, max(y1 - 4, 0), f"{score:.2f}", color=c, fontsize=8, fontweight="bold")
        if masks is not None and i < len(masks):
            mask = np.array(masks[i]).squeeze()
            if mask.ndim == 2:
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask > 0.5] = [*c[:3], 0.35]
                ax2.imshow(overlay)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {save_path.name}")


def best_prompt_result(processor, image, prompts):
    """Try each prompt, return the one with the most detections."""
    best_out, best_prompt, best_n = None, None, -1
    for prompt in prompts:
        try:
            state = processor.set_image(image)
            out = processor.set_text_prompt(state=state, prompt=prompt)
            n = len(out["boxes"])
            print(f"     '{prompt}': {n} detections")
            if n > best_n:
                best_out, best_prompt, best_n = out, prompt, n
        except Exception as e:
            print(f"     '{prompt}': ERROR — {e}")
    return best_out, best_prompt


# ── Model setup ───────────────────────────────────────────────────────────────

print("=" * 65)
print("SAM 3 Macaque Detection Test  (sample: ~9 images)")
print("=" * 65)
print(f"\nPhotos dir : {PHOTOS_DIR}")
print(f"Results dir: {RESULTS_DIR}")

print("\n[Setup] Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device}")
if device == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
else:
    print("  WARNING: No GPU — inference will be very slow")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model     = build_sam3_image_model(device=device)
processor = Sam3Processor(model)
print("  Model loaded!\n")

# ── Scenario 1: Single macaque face ──────────────────────────────────────────
# Individual folders contain photos of one named macaque.
# Prefer filenames without "+" (no other named animal present).

print("─" * 65)
print("SCENARIO 1: Single macaque face")
print("  Source: individual named folders, no '+' in filename")
print("─" * 65)

individual_folders = [
    p for p in PHOTOS_DIR.rglob("*")
    if p.is_dir()
    and p.parent.name in ("females", "males")
    and not p.name.startswith(".")
]

candidates = []
for folder in individual_folders:
    imgs = [p for p in all_images(folder) if "+" not in p.name]
    candidates.extend(imgs)

random.shuffle(candidates)

for i, img_path in enumerate(candidates[:SAMPLES_PER_SCENARIO]):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}")
    print(f"       {img_path.name}")
    try:
        image = Image.open(img_path).convert("RGB")
        out, prompt = best_prompt_result(
            processor, image,
            ["macaque face", "Barbary macaque", "monkey face", "primate face"]
        )
        if out:
            draw_results(
                image, out["masks"], out["boxes"], out["scores"], prompt,
                f"S1 — {individual} ({location})",
                RESULTS_DIR / f"s1_{i+1}_{individual.replace(' ','_')}.jpg"
            )
    except Exception:
        traceback.print_exc()

# ── Scenario 2: Multiple macaque faces ───────────────────────────────────────
# Filenames with "+" name two or more animals, e.g.:
#   "020123 AD floater AM 1442 + AF V-A77 Sylv_1.jpg"

print("\n" + "─" * 65)
print("SCENARIO 2: Multiple macaque faces")
print("  Source: any image whose filename contains '+' (multiple named animals)")
print("─" * 65)

multi_candidates = [
    p for p in all_images(PHOTOS_DIR) if "+" in p.name
]
print(f"  Found {len(multi_candidates)} images with '+' in filename")
random.shuffle(multi_candidates)

for i, img_path in enumerate(multi_candidates[:SAMPLES_PER_SCENARIO]):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}")
    print(f"       {img_path.name}")
    try:
        image = Image.open(img_path).convert("RGB")
        out, prompt = best_prompt_result(
            processor, image,
            ["macaque face", "Barbary macaque", "monkey", "multiple macaques"]
        )
        if out:
            draw_results(
                image, out["masks"], out["boxes"], out["scores"], prompt,
                f"S2 — {individual} + others ({location})",
                RESULTS_DIR / f"s2_{i+1}_{individual.replace(' ','_')}.jpg"
            )
    except Exception:
        traceback.print_exc()

# ── Scenario 3: Macaque + human face ─────────────────────────────────────────
# Tourist locations (Cable Car, Prince Philip Arch) are most likely to have
# researchers or tourists in frame. We run both "macaque face" and "human face"
# prompts and overlay all detections.

print("\n" + "─" * 65)
print("SCENARIO 3: Macaque + human face")
print("  Source: tourist locations — Cable Car, Prince Philip Arch")
print("─" * 65)

tourist_imgs = []
for loc_name in ["Cable Car", "Prince Philip Arch"]:
    loc_path = PHOTOS_DIR / loc_name
    if loc_path.exists():
        found = all_images(loc_path)
        print(f"  {loc_name}: {len(found)} images")
        tourist_imgs.extend(found)

if not tourist_imgs:
    print("  No tourist location images found — falling back to all locations")
    tourist_imgs = all_images(PHOTOS_DIR)

random.shuffle(tourist_imgs)

for i, img_path in enumerate(tourist_imgs[:SAMPLES_PER_SCENARIO]):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}")
    print(f"       {img_path.name}")
    try:
        image = Image.open(img_path).convert("RGB")

        # Run both macaque and human prompts, collect all boxes
        all_boxes, all_scores, all_masks = [], [], []
        for prompt in ["macaque face", "human face", "person"]:
            state = processor.set_image(image)
            out = processor.set_text_prompt(state=state, prompt=prompt)
            n = len(out["boxes"])
            print(f"     '{prompt}': {n} detections")
            if n > 0:
                all_boxes.extend(out["boxes"])
                all_scores.extend(out["scores"])
                if out["masks"] is not None:
                    all_masks.extend(out["masks"])

        draw_results(
            image, all_masks or None, all_boxes, all_scores,
            "macaque face + human face + person",
            f"S3 — {individual} ({location}) — {len(all_boxes)} total",
            RESULTS_DIR / f"s3_{i+1}_{individual.replace(' ','_')}.jpg"
        )
    except Exception:
        traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
result_files = sorted(RESULTS_DIR.glob("s*.jpg"))
print("\n" + "=" * 65)
print(f"DONE — {len(result_files)} result images saved to:")
print(f"  {RESULTS_DIR}")
for f in result_files:
    print(f"  {f.name}")
print("=" * 65)
print("\nNext: if results look good, run full analysis on all 9,896 images")
