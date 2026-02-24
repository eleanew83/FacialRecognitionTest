#!/usr/bin/env python
"""
Prompt benchmark + YOLO comparison for SAM 3 macaque detection.

Modes
-----
  (default)  Proxy benchmark on raw Gibraltar photos (~9 images, 3 scenarios).
             Metric: rank_score = det_rate × avg_conf

  --eval     Ground-truth evaluation on the manually labelled YOLO val split.
             Metric: Precision / Recall / F1 at IoU 0.5, vs trained YOLO detector.

Usage
-----
  python benchmark_prompts.py          # proxy benchmark
  python benchmark_prompts.py --eval   # GT evaluation vs YOLO
"""

import csv
import random
import sys
import traceback
from pathlib import Path

EVAL_MODE = "--eval" in sys.argv

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
    # Species-specific
    "macaque face",
    "Barbary macaque face",
    "Barbary macaque",
    # Taxon-level
    "monkey face",
    "monkey head",
    "macaque head",
    "primate face",
    # Visual / descriptive
    "furry animal face",
    "close-up of a monkey face",
    "animal face",
    # Body-level (no face constraint)
    "macaque",
    "monkey",
    # Zero-shot ceiling
    "face",
]

# Boxes scoring at or below this are treated as noise
SCORE_THRESHOLD = 0.05
# Boxes overlapping more than this IoU are considered duplicates of the same face
NMS_IOU_THRESHOLD = 0.5

# ── Eval-mode config ──────────────────────────────────────────────────────────

YOLO_DATA_DIR   = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_data")
YOLO_MODEL_PATH = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_code/models/runs/macaque_face_detector_20260120_v1/weights/best.pt")
EVAL_RESULTS_DIR = Path("/rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/results/eval_vs_yolo")

N_EVAL_IMAGES   = 50    # images sampled from val split (seeded)
EVAL_IOU_THRESH = 0.5   # IoU threshold for a true positive
EVAL_CONF_THRESH = 0.25  # YOLO confidence threshold


# ── Helpers ───────────────────────────────────────────────────────────────────

# ·· Shared ····································································

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


# ·· GT evaluation helpers ·····················································

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Read a YOLO label file and return absolute [x1,y1,x2,y2] boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([
                (xc - w / 2) * img_w,
                (yc - h / 2) * img_h,
                (xc + w / 2) * img_w,
                (yc + h / 2) * img_h,
            ])
    return boxes


def compute_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes: list, gt_boxes: list) -> tuple[int, int, int]:
    """
    Greedy highest-IoU matching of predicted boxes to GT boxes.
    Returns (tp, fp, fn).
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    matched_gt: set[int] = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= EVAL_IOU_THRESH:
            tp += 1
            matched_gt.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def run_sam3_boxes(processor, image, prompt: str) -> list[list[float]]:
    """
    Run SAM3 and return NMS-filtered absolute [x1,y1,x2,y2] boxes.
    Also used by the eval loop which attaches scores separately.
    """
    try:
        state  = processor.set_image(image)
        out    = processor.set_text_prompt(state=state, prompt=prompt)
        boxes  = out["boxes"]
        scores = out["scores"]
        pairs  = [
            (b, s) for b, s in zip(boxes, scores)
            if (s.cpu().item() if hasattr(s, "cpu") else float(s)) > SCORE_THRESHOLD
        ]
        if not pairs:
            return []
        boxes, scores = zip(*pairs)
        boxes, scores = apply_nms(boxes, scores)
        return [
            [(v.cpu().item() if hasattr(v, "cpu") else float(v)) for v in b]
            for b in boxes
        ]
    except Exception as e:
        print(f"       SAM3 ERROR [{prompt}]: {e}")
        return []


def run_sam3_boxes_scored(processor, image, prompt: str) -> list[tuple[float, list[float]]]:
    """
    Same as run_sam3_boxes but returns (confidence, [x1,y1,x2,y2]) pairs,
    needed for proper AP50 computation via confidence-score sweep.
    """
    try:
        state  = processor.set_image(image)
        out    = processor.set_text_prompt(state=state, prompt=prompt)
        boxes  = out["boxes"]
        scores = out["scores"]
        pairs  = [
            (b, s) for b, s in zip(boxes, scores)
            if (s.cpu().item() if hasattr(s, "cpu") else float(s)) > SCORE_THRESHOLD
        ]
        if not pairs:
            return []
        boxes, scores = zip(*pairs)
        boxes, scores = apply_nms(boxes, scores)
        return [
            (float(s), [(v.cpu().item() if hasattr(v, "cpu") else float(v)) for v in b])
            for b, s in zip(boxes, scores)
        ]
    except Exception as e:
        print(f"       SAM3 ERROR [{prompt}]: {e}")
        return []


def draw_comparison(image, gt_boxes, yolo_boxes, sam3_boxes, best_prompt, save_path):
    """Side-by-side: GT | YOLO | best SAM3 prompt."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Ground Truth", f"YOLO ({len(yolo_boxes)} det.)",
              f"SAM3 '{best_prompt}' ({len(sam3_boxes)} det.)"]
    box_sets = [gt_boxes, yolo_boxes, sam3_boxes]
    colours  = ["lime", "red", "cyan"]

    for ax, title, bxs, col in zip(axes, titles, box_sets, colours):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        for box in bxs:
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=col, facecolor="none"
            ))

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


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
print("  Model loaded!\n")

# ── GT evaluation mode (--eval) ───────────────────────────────────────────────

if EVAL_MODE:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO detector ────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        yolo_det     = UltralyticsYOLO(str(YOLO_MODEL_PATH))
        YOLO_AVAILABLE = True
        print(f"[Eval] YOLO model loaded: {YOLO_MODEL_PATH.name}")
    except ImportError:
        YOLO_AVAILABLE = False
        print("[Eval] WARNING: ultralytics not installed — YOLO comparison skipped")

    # ── YOLO official score on full val set (historical reference) ────────────
    yolo_ref_ap50    = None
    yolo_ref_ap50_95 = None
    yolo_ref_n       = None
    if YOLO_AVAILABLE:
        print("[Eval] Running YOLO model.val() on full val set for historical reference ...")
        try:
            ref = yolo_det.val(
                data=str(YOLO_DATA_DIR / "dataset.yaml"),
                split="val", verbose=False,
            )
            yolo_ref_ap50    = float(ref.box.ap50.mean())
            yolo_ref_ap50_95 = float(ref.box.map)
            yolo_ref_n       = 1333
            print(f"   AP50={yolo_ref_ap50:.3f}   AP50-95={yolo_ref_ap50_95:.3f}\n")
        except Exception as e:
            print(f"   model.val() failed: {e}\n")

    # ── AP50 helper ───────────────────────────────────────────────────────────
    def compute_ap50(preds: list[tuple[float, bool]], n_gt: int) -> float:
        """Area under the precision-recall curve at IoU ≥ 0.5."""
        if not preds or n_gt == 0:
            return 0.0
        preds_s = sorted(preds, key=lambda x: -x[0])
        tp_c = fp_c = 0
        prec = [1.0]; rec = [0.0]
        for _, is_tp in preds_s:
            if is_tp: tp_c += 1
            else:     fp_c += 1
            prec.append(tp_c / (tp_c + fp_c))
            rec.append(tp_c / n_gt)
        # Monotone envelope (standard)
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        return float(sum((rec[i] - rec[i-1]) * prec[i] for i in range(1, len(rec))))

    # ── Sample val images ─────────────────────────────────────────────────────
    val_img_dir   = YOLO_DATA_DIR / "images" / "val"
    val_label_dir = YOLO_DATA_DIR / "labels" / "val"
    val_imgs = sorted([
        p for p in val_img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    random.Random(42).shuffle(val_imgs)
    val_imgs = val_imgs[:N_EVAL_IMAGES]
    print(f"[Eval] {len(val_imgs)}-image sample  (IoU ≥ {EVAL_IOU_THRESH})\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    # AP:  key → list of (confidence, is_tp)
    # F1:  key → [tp, fp, fn]
    keys     = ["YOLO"] + [f"SAM3:{p}" for p in PROMPTS]
    ap_data:  dict[str, list] = {k: [] for k in keys}
    f1_data:  dict[str, list] = {k: [0, 0, 0] for k in keys}
    total_gt  = 0

    # ── Per-image loop ────────────────────────────────────────────────────────
    for idx, img_path in enumerate(val_imgs):
        label_path = val_label_dir / (img_path.stem + ".txt")
        image      = Image.open(img_path).convert("RGB")
        W, H       = image.size
        gt_boxes   = load_gt_boxes(label_path, W, H)
        total_gt  += len(gt_boxes)

        print(f"  [{idx+1:>2}/{len(val_imgs)}] {img_path.name}  GT={len(gt_boxes)}")

        def _accumulate(key, pred_scored, gt):
            """Match scored predictions to GT, update AP and F1 accumulators."""
            pred_sorted = sorted(pred_scored, key=lambda x: -x[0])
            matched: set[int] = set()
            for conf, box in pred_sorted:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt):
                    if j in matched: continue
                    iou = compute_iou(box, g)
                    if iou > best_iou: best_iou, best_j = iou, j
                is_tp = best_iou >= EVAL_IOU_THRESH
                if is_tp: matched.add(best_j)
                ap_data[key].append((conf, is_tp))
            tp = len(matched)
            fp = len(pred_sorted) - tp
            fn = len(gt) - tp
            f1_data[key][0] += tp
            f1_data[key][1] += fp
            f1_data[key][2] += fn

        # YOLO
        yolo_boxes_vis = []
        if YOLO_AVAILABLE:
            try:
                results = yolo_det(img_path, conf=EVAL_CONF_THRESH, verbose=False)
                yolo_scored = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        yolo_scored.append((conf, [x1, y1, x2, y2]))
                        yolo_boxes_vis.append([x1, y1, x2, y2])
                _accumulate("YOLO", yolo_scored, gt_boxes)
            except Exception as e:
                print(f"       YOLO ERROR: {e}")

        # SAM3 — all prompts
        best_f1_img, best_prompt_img, best_boxes_img = -1.0, PROMPTS[0], []
        for prompt in PROMPTS:
            key         = f"SAM3:{prompt}"
            pred_scored = run_sam3_boxes_scored(processor, image, prompt)
            _accumulate(key, pred_scored, gt_boxes)
            pred        = [b for _, b in pred_scored]
            tp, fp, fn = f1_data[key][0], f1_data[key][1], f1_data[key][2]
            _, _, f1 = prf1(tp, fp, fn)
            if f1 > best_f1_img:
                best_f1_img, best_prompt_img, best_boxes_img = f1, prompt, pred
            print(f"       {prompt:<30s}  pred={len(pred):>2}  "
                  f"tp={f1_data[key][0]}  fp={f1_data[key][1]}  fn={f1_data[key][2]}")

        print(f"       yolo={len(yolo_boxes_vis)}  best_sam3='{best_prompt_img}'")

        draw_comparison(
            image, gt_boxes, yolo_boxes_vis, best_boxes_img, best_prompt_img,
            EVAL_RESULTS_DIR / f"eval_{idx+1:02d}_{img_path.stem[:28]}.jpg",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 34
    SEP = "─" * 72

    print("\n" + "═" * 72)
    print("EVALUATION RESULTS")
    print("═" * 72)

    # Historical YOLO reference
    if yolo_ref_ap50 is not None:
        print(f"\n  YOLO — official score on full val set ({yolo_ref_n} images):")
        print(f"    AP50 = {yolo_ref_ap50:.3f}   AP50-95 = {yolo_ref_ap50_95:.3f}")
        print(f"  These are the historical numbers to compare against.\n")

    print(SEP)
    print(f"  {len(val_imgs)}-image sample comparison  (IoU ≥ {EVAL_IOU_THRESH})")
    print(f"  AP50  = area under precision-recall curve (threshold-independent)")
    print(f"  Prec / Recall / F1  = at confidence threshold used during inference")
    print(SEP)

    hdr = f"  {'Rank':<5} {'Detector / Prompt':<{W}}  {'AP50':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}"
    print(hdr)
    print(f"  {'─'*5} {'─'*W}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

    # Build result rows
    eval_rows = []
    for key in keys:
        tp, fp, fn = f1_data[key]
        p, r, f1   = prf1(tp, fp, fn)
        ap         = compute_ap50(ap_data[key], total_gt)
        label      = "YOLO (trained)" if key == "YOLO" else key.replace("SAM3:", "SAM3: ")
        eval_rows.append({"detector": label, "ap50": ap,
                          "precision": p, "recall": r, "f1": f1,
                          "tp": tp, "fp": fp, "fn": fn})

    yolo_rows = [r for r in eval_rows if r["detector"] == "YOLO (trained)"]
    sam3_rows = sorted([r for r in eval_rows if r["detector"] != "YOLO (trained)"],
                       key=lambda r: -r["ap50"])

    def _fmt(rank, row, suffix=""):
        return (f"  {rank:<5} {row['detector']:<{W}}  "
                f"{row['ap50']:6.3f}  {row['precision']:6.3f}  "
                f"{row['recall']:6.3f}  {row['f1']:6.3f}{suffix}")

    if yolo_rows:
        print(_fmt("REF", yolo_rows[0]))
        print(f"  {'─'*5} {'─'*W}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

    for i, row in enumerate(sam3_rows, 1):
        tag = "  ← best SAM3 prompt" if i == 1 else ""
        print(_fmt(i, row, tag))

    # What to take away
    print(SEP)
    best_sam3 = sam3_rows[0] if sam3_rows else None
    if yolo_rows and best_sam3:
        gap = yolo_rows[0]["ap50"] - best_sam3["ap50"]
        direction = "below" if gap > 0 else "above"
        print(f"  Best SAM3 prompt : '{best_sam3['detector'].replace('SAM3: ', '')}'")
        print(f"  AP50 gap vs YOLO : {abs(gap):.3f} {direction} YOLO on this sample")
        if yolo_ref_ap50:
            print(f"  YOLO reference   : {yolo_ref_ap50:.3f} (full val)  vs  "
                  f"{yolo_rows[0]['ap50']:.3f} ({len(val_imgs)}-img sample)")
    print("═" * 72)

    # CSV
    all_rows = yolo_rows + sam3_rows
    csv_path = EVAL_RESULTS_DIR / "eval_vs_yolo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  CSV    → {csv_path}")
    print(f"  Images → {EVAL_RESULTS_DIR}/  ({len(val_imgs)} side-by-side comparisons)")
    sys.exit(0)

# ── Proxy benchmark (default mode) ────────────────────────────────────────────

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
