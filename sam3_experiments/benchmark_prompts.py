#!/usr/bin/env python
"""
Prompt benchmark for SAM 3 macaque detection.

Tests every candidate prompt on the same ~9 sample images used by the main
test script (identical seed + sampling logic) and prints a ranked summary
table.  Results are also saved as a CSV.

Scenarios
---------
  S1 — Single macaque face    (individual folders, no '+' in filename)
  S2 — Multiple macaque faces ('+' in filename)
  S3 — Macaque + human        (Cable Car / Prince Philip Arch)

Metric columns
--------------
  det_rate  : fraction of images where ≥ 1 box was returned
  avg_count : mean number of boxes across all images (including zeros)
  avg_conf  : mean confidence of *returned* boxes (NaN if none)
  rank_score: det_rate × avg_conf  (higher is better)
"""

import csv
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
from torchvision.ops import nms as torchvision_nms

sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────────

PHOTOS_DIR   = Path("/rds/user/ylj20/hpc-work/Gibraltar_Macaques_Photos")
RESULTS_DIR  = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/results/prompt_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_SCENARIO = 3
random.seed(42)

PROMPTS = [
    "macaque face",
    "Barbary macaque face",
    "Barbary macaque",
    "monkey face",
    "primate face",
    "animal face",
    "macaque",
    "monkey",
    "face",
]

# Boxes scoring at or below this are treated as noise
SCORE_THRESHOLD = 0.05
# Boxes overlapping more than this IoU are considered duplicates of the same face
NMS_IOU_THRESHOLD = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_images(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not p.name.startswith("._")
    ]


def apply_nms(boxes, scores) -> tuple:
    """
    Deduplicate overlapping boxes with torchvision NMS.

    Boxes that overlap more than NMS_IOU_THRESHOLD are treated as duplicate
    detections of the same face; only the highest-scoring one is kept.

    Returns (kept_boxes, kept_scores) as plain Python lists.
    """
    if len(boxes) == 0:
        return [], []

    def to_tensor(v):
        if hasattr(v, "cpu"):
            return v.cpu().float()
        return torch.tensor(float(v))

    boxes_t  = torch.stack([
        torch.tensor([
            (v.cpu().item() if hasattr(v, "cpu") else float(v))
            for v in box
        ], dtype=torch.float32)
        for box in boxes
    ])
    scores_t = torch.tensor(
        [(s.cpu().item() if hasattr(s, "cpu") else float(s)) for s in scores],
        dtype=torch.float32,
    )

    keep = torchvision_nms(boxes_t, scores_t, NMS_IOU_THRESHOLD)
    kept_boxes  = [boxes[i]  for i in keep.tolist()]
    kept_scores = [scores_t[i].item() for i in keep.tolist()]
    return kept_boxes, kept_scores


def run_prompt(processor, image, prompt: str) -> dict:
    """
    Return {"count": int, "scores": list[float]} for one prompt on one image.

    Pipeline:
      1. Run SAM3
      2. Drop low-confidence boxes (< SCORE_THRESHOLD)
      3. NMS to remove duplicate boxes circling the same face
    """
    try:
        state  = processor.set_image(image)
        out    = processor.set_text_prompt(state=state, prompt=prompt)
        boxes  = out["boxes"]
        scores = out["scores"]

        # 1. confidence filter
        pairs = [
            (b, s)
            for b, s in zip(boxes, scores)
            if (s.cpu().item() if hasattr(s, "cpu") else float(s)) > SCORE_THRESHOLD
        ]
        if not pairs:
            return {"count": 0, "scores": []}
        boxes, scores = zip(*pairs)

        # 2. NMS deduplication
        boxes, scores = apply_nms(boxes, scores)

        return {"count": len(scores), "scores": list(scores)}
    except Exception as e:
        print(f"       ERROR [{prompt}]: {e}")
        return {"count": 0, "scores": []}


def draw_best(image, processor, best_prompt: str, title: str, save_path: Path):
    """Save a visualisation for the winning prompt on one image."""
    try:
        state  = processor.set_image(image)
        out    = processor.set_text_prompt(state=state, prompt=best_prompt)
        boxes  = out["boxes"]
        scores = out["scores"]
        masks  = out["masks"]

        # Mirror the same filtering applied in run_prompt so the picture
        # matches the recorded counts exactly.
        pairs = [
            (b, s, (masks[i] if masks is not None and i < len(masks) else None))
            for i, (b, s) in enumerate(zip(boxes, scores))
            if (s.cpu().item() if hasattr(s, "cpu") else float(s)) > SCORE_THRESHOLD
        ]
        if pairs:
            raw_boxes, raw_scores, raw_masks = zip(*pairs)
            kept_boxes, kept_scores = apply_nms(raw_boxes, raw_scores)
            # Rebuild mask list aligned to the kept boxes
            score_to_mask = {
                id(raw_boxes[i]): raw_masks[i] for i in range(len(raw_boxes))
            }
            kept_masks = [score_to_mask.get(id(b)) for b in kept_boxes]
        else:
            kept_boxes, kept_scores, kept_masks = [], [], []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=9, wrap=True)
        ax1.imshow(image); ax1.set_title("Original"); ax1.axis("off")
        ax2.imshow(image)
        ax2.set_title(f"'{best_prompt}' → {len(kept_boxes)} det. (after NMS)")
        ax2.axis("off")

        colours = plt.cm.Set1(np.linspace(0, 1, max(len(kept_boxes), 1)))
        for i, (box, score) in enumerate(zip(kept_boxes, kept_scores)):
            x1, y1, x2, y2 = [v.cpu().item() if hasattr(v, "cpu") else float(v) for v in box]
            c = colours[i % len(colours)]
            ax2.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=c, facecolor="none"
            ))
            ax2.text(x1, max(y1 - 4, 0), f"{score:.2f}", color=c, fontsize=8, fontweight="bold")
            m = kept_masks[i]
            if m is not None:
                m = np.array(m.cpu() if hasattr(m, "cpu") else m).squeeze()
                if m.ndim == 2:
                    ov = np.zeros((*m.shape, 4))
                    ov[m > 0.5] = [*c[:3], 0.35]
                    ax2.imshow(ov)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception:
        traceback.print_exc()


# ── Stats accumulator ─────────────────────────────────────────────────────────

class PromptStats:
    def __init__(self):
        # prompt → scenario → list of per-image dicts
        self.data: dict[str, dict[str, list[dict]]] = {}

    def record(self, prompt: str, scenario: str, result: dict):
        self.data.setdefault(prompt, {}).setdefault(scenario, []).append(result)

    def summary(self) -> list[dict]:
        rows = []
        for prompt in PROMPTS:
            for scenario in ["S1", "S2", "S3", "ALL"]:
                if scenario == "ALL":
                    results = [
                        r
                        for sc_results in self.data.get(prompt, {}).values()
                        for r in sc_results
                    ]
                else:
                    results = self.data.get(prompt, {}).get(scenario, [])
                if not results:
                    continue
                n         = len(results)
                det_rate  = sum(1 for r in results if r["count"] > 0) / n
                avg_count = sum(r["count"] for r in results) / n
                all_scores = [s for r in results for s in r["scores"]]
                avg_conf  = float(np.mean(all_scores)) if all_scores else float("nan")
                rank      = det_rate * avg_conf if all_scores else 0.0
                rows.append({
                    "prompt":     prompt,
                    "scenario":   scenario,
                    "n_images":   n,
                    "det_rate":   det_rate,
                    "avg_count":  avg_count,
                    "avg_conf":   avg_conf,
                    "rank_score": rank,
                })
        return rows


# ── Model setup ───────────────────────────────────────────────────────────────

print("=" * 70)
print("SAM 3 Prompt Benchmark")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model     = build_sam3_image_model(device=device)
processor = Sam3Processor(model)
print("Model loaded.\n")

stats = PromptStats()


# ── Scenario 1 ────────────────────────────────────────────────────────────────

print("─" * 70)
print("SCENARIO 1: Single macaque face")
print("─" * 70)

individual_folders = [
    p for p in PHOTOS_DIR.rglob("*")
    if p.is_dir()
    and p.parent.name in ("females", "males")
    and not p.name.startswith(".")
]
s1_candidates = []
for folder in individual_folders:
    s1_candidates.extend(p for p in all_images(folder) if "+" not in p.name)
random.shuffle(s1_candidates)
s1_images = s1_candidates[:SAMPLES_PER_SCENARIO]

for i, img_path in enumerate(s1_images):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}  —  {img_path.name}")
    image      = Image.open(img_path).convert("RGB")
    best_count, best_prompt = -1, None
    for prompt in PROMPTS:
        result = run_prompt(processor, image, prompt)
        stats.record(prompt, "S1", result)
        flag = " ←" if result["count"] > best_count else ""
        cf   = f"{np.mean(result['scores']):.3f}" if result["scores"] else "  n/a"
        print(f"     {prompt:<30s}  count={result['count']}  conf={cf}{flag}")
        if result["count"] > best_count:
            best_count, best_prompt = result["count"], prompt
    if best_prompt and best_count > 0:
        draw_best(
            image, processor, best_prompt,
            f"S1[{i+1}] {individual} ({location}) — best: '{best_prompt}'",
            RESULTS_DIR / f"s1_{i+1}_{individual.replace(' ','_')}.jpg",
        )


# ── Scenario 2 ────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("SCENARIO 2: Multiple macaque faces")
print("─" * 70)

s2_candidates = [p for p in all_images(PHOTOS_DIR) if "+" in p.name]
random.shuffle(s2_candidates)
s2_images = s2_candidates[:SAMPLES_PER_SCENARIO]
print(f"  Pool: {len(s2_candidates)} images with '+' in filename")

for i, img_path in enumerate(s2_images):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}  —  {img_path.name}")
    image      = Image.open(img_path).convert("RGB")
    best_count, best_prompt = -1, None
    for prompt in PROMPTS:
        result = run_prompt(processor, image, prompt)
        stats.record(prompt, "S2", result)
        flag = " ←" if result["count"] > best_count else ""
        cf   = f"{np.mean(result['scores']):.3f}" if result["scores"] else "  n/a"
        print(f"     {prompt:<30s}  count={result['count']}  conf={cf}{flag}")
        if result["count"] > best_count:
            best_count, best_prompt = result["count"], prompt
    if best_prompt and best_count > 0:
        draw_best(
            image, processor, best_prompt,
            f"S2[{i+1}] {individual} ({location}) — best: '{best_prompt}'",
            RESULTS_DIR / f"s2_{i+1}_{individual.replace(' ','_')}.jpg",
        )


# ── Scenario 3 ────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("SCENARIO 3: Macaque + human")
print("─" * 70)

s3_images = []
for loc_name in ["Cable Car", "Prince Philip Arch"]:
    loc_path = PHOTOS_DIR / loc_name
    if loc_path.exists():
        found = all_images(loc_path)
        print(f"  {loc_name}: {len(found)} images")
        s3_images.extend(found)
if not s3_images:
    print("  No tourist locations found — falling back to all images")
    s3_images = all_images(PHOTOS_DIR)
random.shuffle(s3_images)
s3_images = s3_images[:SAMPLES_PER_SCENARIO]

for i, img_path in enumerate(s3_images):
    individual = img_path.parent.name
    location   = img_path.parent.parent.parent.name
    print(f"\n  [{i+1}] {location} / {individual}  —  {img_path.name}")
    image      = Image.open(img_path).convert("RGB")
    best_count, best_prompt = -1, None
    for prompt in PROMPTS:
        result = run_prompt(processor, image, prompt)
        stats.record(prompt, "S3", result)
        flag = " ←" if result["count"] > best_count else ""
        cf   = f"{np.mean(result['scores']):.3f}" if result["scores"] else "  n/a"
        print(f"     {prompt:<30s}  count={result['count']}  conf={cf}{flag}")
        if result["count"] > best_count:
            best_count, best_prompt = result["count"], prompt
    if best_prompt and best_count > 0:
        draw_best(
            image, processor, best_prompt,
            f"S3[{i+1}] {individual} ({location}) — best: '{best_prompt}'",
            RESULTS_DIR / f"s3_{i+1}_{individual.replace(' ','_')}.jpg",
        )


# ── Summary table ─────────────────────────────────────────────────────────────

rows = stats.summary()
rows.sort(key=lambda r: (-r["rank_score"], r["prompt"]))

print("\n" + "=" * 70)
print("SUMMARY  (sorted by rank_score = det_rate × avg_conf, higher is better)")
print("=" * 70)

header = f"{'prompt':<30s}  {'scen':>4}  {'n':>2}  {'det%':>5}  {'avg_n':>5}  {'avg_c':>6}  {'rank':>6}"
print(header)
print("-" * len(header))

last_scenario = None
for row in rows:
    scen = row["scenario"]
    if scen != last_scenario:
        if last_scenario is not None:
            print()
        last_scenario = scen
    conf_str = f"{row['avg_conf']:6.3f}" if not np.isnan(row["avg_conf"]) else "   nan"
    print(
        f"{row['prompt']:<30s}  {scen:>4}  {row['n_images']:>2}  "
        f"{row['det_rate']*100:5.0f}%  {row['avg_count']:5.1f}  "
        f"{conf_str}  {row['rank_score']:6.3f}"
    )

# ── CSV export ────────────────────────────────────────────────────────────────

csv_path = RESULTS_DIR / "prompt_benchmark.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV saved → {csv_path}")
print(f"Visualisations saved → {RESULTS_DIR}/")
print("=" * 70)
