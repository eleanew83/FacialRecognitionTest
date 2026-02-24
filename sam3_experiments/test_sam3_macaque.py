#!/usr/bin/env python
"""
SAM 3 macaque detection test — 3 scenarios, ~3 sample images each.

NOTE: This is a validation test, NOT a full analysis.
      It samples ~9 images to confirm SAM 3 works correctly before
      running on all 9,896 images.

Scenarios:
  1. Single macaque face   — images from individual folders, no "+" in filename
  2. Multiple macaque faces — images whose filenames contain "+" (multiple named animals)
  3. Macaque + human face  — uncropped tourist location images (Cable Car, Prince Philip Arch)

Prompt selection:
  We evaluated 13 candidate prompts on 50 manually-labelled val images (AP50 @ IoU 0.5).
  Results ranked by AP50:
    1. "monkey face"             AP50=0.621  ← SELECTED
    2. "monkey head"             AP50=0.585  (better precision, lower recall)
    3. "macaque head"            AP50=0.477
    4. "face"                    AP50=0.452
    5. "close-up of a monkey face" AP50=0.422
    6. "macaque face"            AP50=0.346  (surprisingly worse than generic "monkey face")
    7+ All others                AP50<0.27
  Species-specific prompts ("Barbary macaque face", "macaque") underperformed the
  generic "monkey face" — CLIP's training data contains far more "monkey" than "macaque".
  YOLO (trained) reference: AP50=0.979

Results saved to:
  .../sam3_experiments/results/
"""

import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

PHOTOS_DIR   = Path("/rds/user/ylj20/hpc-work/Gibraltar_Macaques_Photos")
_BASE_RESULTS = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/results")
RESULTS_DIR  = _BASE_RESULTS / datetime.now().strftime("scenario_%Y%m%d_%H%M%S")
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


def _draw_boxes_on_ax(ax, boxes, scores, masks, colour, label=None):
    """Draw one group of boxes/masks on an axes with a single colour."""
    legend_patch = None
    colours = plt.cm.Set1(np.linspace(0, 1, max(len(boxes), 1)))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [v.cpu().item() if hasattr(v, 'cpu') else float(v) for v in box]
        score_val = score.cpu().item() if hasattr(score, 'cpu') else float(score)
        # When a fixed colour is given use it; otherwise cycle through the palette
        c = colour if colour is not None else colours[i % len(colours)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=c, facecolor="none",
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)
        if legend_patch is None and label:
            legend_patch = rect
        ax.text(x1, max(y1 - 4, 0), f"{score_val:.2f}",
                color=c, fontsize=8, fontweight="bold")
        if masks is not None and i < len(masks):
            m    = masks[i]
            mask = np.array(m.cpu() if hasattr(m, 'cpu') else m).squeeze()
            if mask.ndim == 2:
                rgba = matplotlib.colors.to_rgba(c)
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask > 0.5] = [*rgba[:3], 0.35]
                ax.imshow(overlay)
    return legend_patch


def draw_results(image, masks, boxes, scores, prompt, title, save_path,
                 yolo_boxes=None, prompt_groups=None):
    """
    Panels: Original | [YOLO] | SAM3

    prompt_groups — list of dicts for multi-prompt SAM3 panels (used in S3):
        [{"label": str, "boxes": [...], "scores": [...], "masks": [...], "colour": str}, ...]
    When prompt_groups is given, each group is drawn in its own colour with a legend.
    """
    ncols  = 3 if yolo_boxes is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    fig.suptitle(title, fontsize=9, wrap=True)

    ax_orig = axes[0]
    ax_sam3 = axes[-1]
    ax_orig.imshow(image); ax_orig.set_title("Original"); ax_orig.axis("off")

    # Middle panel — YOLO (macaque only, red boxes)
    if yolo_boxes is not None:
        ax_yolo = axes[1]
        ax_yolo.imshow(image)
        ax_yolo.set_title(f"YOLO → {len(yolo_boxes)} detected (macaque)")
        ax_yolo.axis("off")
        for box in yolo_boxes:
            x1, y1, x2, y2 = box
            ax_yolo.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none"
            ))

    # Right panel — SAM3
    ax_sam3.imshow(image)
    ax_sam3.axis("off")

    if prompt_groups:
        # Multi-prompt mode: each group drawn in its assigned colour + legend
        legend_patches = []
        count_parts    = []
        for grp in prompt_groups:
            p = _draw_boxes_on_ax(
                ax_sam3,
                grp["boxes"], grp["scores"], grp.get("masks"),
                colour=grp["colour"], label=grp["label"],
            )
            if p is not None:
                legend_patches.append(p)
            count_parts.append(f"{grp['label']}: {len(grp['boxes'])}")
        ax_sam3.set_title("SAM3 — " + "  |  ".join(count_parts))
        if legend_patches:
            ax_sam3.legend(handles=legend_patches, loc="upper right",
                           fontsize=8, framealpha=0.7)
    else:
        # Single-prompt mode: cycle through colour palette
        ax_sam3.set_title(f"SAM3 '{prompt}' → {len(boxes)} detected")
        _draw_boxes_on_ax(ax_sam3, boxes, scores, masks, colour=None)

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
print("  Model loaded!")

# YOLO comparison model (optional — for side-by-side visualisations)
YOLO_MODEL_PATH = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest"
                       "/yolo_detection/yolo_detection_code/models/runs"
                       "/macaque_face_detector_20260120_v1/weights/best.pt")
try:
    from ultralytics import YOLO as UltralyticsYOLO
    yolo_det      = UltralyticsYOLO(str(YOLO_MODEL_PATH))
    YOLO_AVAILABLE = True
    print("  YOLO model loaded!")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"  YOLO not available ({e}) — SAM3-only visualisations")
print()


def run_yolo(img_path) -> list:
    """Return list of [x1,y1,x2,y2] boxes from the trained YOLO detector."""
    if not YOLO_AVAILABLE:
        return []
    try:
        results = yolo_det(img_path, conf=0.25, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                boxes.append(box.xyxy[0].tolist())
        return boxes
    except Exception as e:
        print(f"     YOLO error: {e}")
        return []


# Best SAM3 prompt (selected from benchmark — see module docstring for full ranking)
SAM3_PROMPT = "monkey face"

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
        image      = Image.open(img_path).convert("RGB")
        yolo_boxes = run_yolo(img_path)
        print(f"     YOLO: {len(yolo_boxes)} detected")
        out, prompt = best_prompt_result(processor, image, [SAM3_PROMPT])
        if out:
            print(f"     SAM3: {len(out['boxes'])} detected")
            draw_results(
                image, out["masks"], out["boxes"], out["scores"], prompt,
                f"S1 — {individual} ({location})",
                RESULTS_DIR / f"s1_{i+1}_{individual.replace(' ','_')}.jpg",
                yolo_boxes=yolo_boxes,
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
        image      = Image.open(img_path).convert("RGB")
        yolo_boxes = run_yolo(img_path)
        print(f"     YOLO: {len(yolo_boxes)} detected")
        out, prompt = best_prompt_result(processor, image, [SAM3_PROMPT])
        if out:
            print(f"     SAM3: {len(out['boxes'])} detected  "
                  f"{'← SAM3 finds MORE' if len(out['boxes']) > len(yolo_boxes) else ''}")
            draw_results(
                image, out["masks"], out["boxes"], out["scores"], prompt,
                f"S2 — {individual} + others ({location})",
                RESULTS_DIR / f"s2_{i+1}_{individual.replace(' ','_')}.jpg",
                yolo_boxes=yolo_boxes,
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
        # Prefer uncropped images — those are the original scene shots most
        # likely to contain both macaques and humans in the same frame.
        uncropped = [p for p in all_images(loc_path) if "cropped" not in p.name.lower()]
        all_loc   = all_images(loc_path)
        found     = uncropped if uncropped else all_loc
        print(f"  {loc_name}: {len(all_loc)} total, {len(found)} uncropped used")
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
        image      = Image.open(img_path).convert("RGB")
        yolo_boxes = run_yolo(img_path)
        print(f"     YOLO: {len(yolo_boxes)} detected (macaque only)")

        # SAM3: run each prompt separately and keep results in labelled groups
        # so monkey and human boxes can be drawn in distinct colours.
        # Colour scheme: monkey face → dodgerblue, human face → darkorange
        s3_prompts = [
            {"label": "monkey face", "colour": "dodgerblue"},
            {"label": "human face",  "colour": "darkorange"},
        ]
        prompt_groups = []
        total_sam3 = 0
        for grp in s3_prompts:
            state = processor.set_image(image)
            out   = processor.set_text_prompt(state=state, prompt=grp["label"])
            n     = len(out["boxes"])
            print(f"     SAM3 '{grp['label']}': {n} detected")
            prompt_groups.append({
                "label":   grp["label"],
                "colour":  grp["colour"],
                "boxes":   out["boxes"],
                "scores":  out["scores"],
                "masks":   out.get("masks"),
            })
            total_sam3 += n

        print(f"     SAM3 total: {total_sam3}  "
              f"{'← SAM3 finds MORE' if total_sam3 > len(yolo_boxes) else ''}")
        draw_results(
            image, None, [], [], "",
            f"S3 — {individual} ({location}) — SAM3: {total_sam3}  YOLO: {len(yolo_boxes)}",
            RESULTS_DIR / f"s3_{i+1}_{individual.replace(' ','_')}.jpg",
            yolo_boxes=yolo_boxes,
            prompt_groups=prompt_groups,
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
print("\nInterpretation guide:")
print("  S1 (single face) : both YOLO and SAM3 should find 1 box")
print("  S2 (multi-face)  : SAM3 should find MORE than YOLO (2+ vs 1)")
print("  S3 (human+monkey): SAM3 boxes should EXCEED YOLO (adds human faces)")
print("\nEach result image has 3 panels: Original | YOLO | SAM3")
print("(If YOLO was unavailable, output is 2 panels: Original | SAM3)")
