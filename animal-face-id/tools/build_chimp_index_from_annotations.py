#!/usr/bin/env python3
"""Build a chimpanzee gallery index from annotations (min10) using the trained model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.base import TrainingConfig, load_config
from src.datasets.transforms import build_transforms
from src.inference.inference_core import load_model_from_config
from src.inference.index_store import SimpleIndex


def _pick_annotation_file(annotations_dir: Path) -> Path:
    """Choose an annotation file, preferring names containing 'min10'."""
    candidates = sorted(p for p in annotations_dir.glob("annotations_*.txt") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No annotation files found under {annotations_dir}")
    for c in candidates:
        if "min10" in c.name.lower():
            return c
    return candidates[0]


def _pick_splits_file(annotations_dir: Path) -> Path | None:
    """Choose a splits file if present, preferring names containing 'min10' and 'split'."""
    candidates = sorted(p for p in annotations_dir.glob("*split*.json") if p.is_file())
    if not candidates:
        return None
    for c in candidates:
        if "min10" in c.name.lower():
            return c
    return candidates[0]


def _load_annotation(path: Path) -> List[tuple[str, str]]:
    """Load (relative_path, id) pairs from annotation txt."""
    samples: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path, identity = parts[0], parts[1]
            samples.append((rel_path, identity))
    return samples


def _load_splits(path: Path) -> Dict[str, List[Dict[str, str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_gallery_entries(
    annotations: List[tuple[str, str]],
    splits: Dict[str, List[Dict[str, str]]] | None,
    gallery_splits: Sequence[str],
    max_per_id: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Select gallery entries with per-id cap; uses provided splits when available."""
    rng = random.Random(seed)
    entries: List[Dict[str, str]] = []

    if splits:
        split_set = set(gallery_splits)
        for split_name, items in splits.items():
            if split_name not in split_set:
                continue
            for item in items:
                entries.append({"path": item["path"], "id": item["id"], "split": split_name})
    else:
        for rel_path, identity in annotations:
            entries.append({"path": rel_path, "id": identity, "split": "unknown"})

    # group and sample per id
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for item in entries:
        grouped[item["id"]].append(item)

    sampled: List[Dict[str, str]] = []
    for identity, items in grouped.items():
        rng.shuffle(items)
        take = items[:max_per_id] if max_per_id > 0 else items
        sampled.extend(take)
    return sampled


class AnnotationDataset(Dataset):
    """Dataset over selected annotation entries using provided transform."""

    def __init__(self, root: Path, entries: List[Dict[str, str]], transform) -> None:
        self.root = root
        self.entries = entries
        self.transform = transform
        ids = sorted({e["id"] for e in entries})
        self.class_to_idx = {c: i for i, c in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.entries[idx]
        path = self.root / item["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_idx = self.class_to_idx[item["id"]]
        return {"image": img, "label": label_idx, "id": item["id"], "path": str(path), "split": item["split"]}


def _compute_embeddings(
    model_bundle: Dict[str, Any],
    entries: List[Dict[str, str]],
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Compute embeddings in batches for selected entries."""
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    model = model_bundle["model"]

    dataset = AnnotationDataset(Path(model_bundle["config"]["data"]["raw_root"]), entries, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_embs: List[np.ndarray] = []
    all_labels: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            embs = model(images).detach().cpu().numpy()
            all_embs.append(embs)
            all_labels.extend(batch["id"])
            # keep meta with path + split
            for pth, sp in zip(batch["path"], batch["split"]):
                all_meta.append({"image_path": pth, "split": sp})

    return np.concatenate(all_embs, axis=0), all_labels, all_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chimpanzee gallery index from annotations.")
    parser.add_argument("--annotations-dir", default="data/chimpanzee_faces/annotations", help="Directory containing annotation txt/json files.")
    parser.add_argument("--config", default="configs/train_chimp_min10_resnet50_arc_full.yaml", help="Training config path.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (defaults to inference.checkpoint in config).")
    parser.add_argument("--gallery-splits", nargs="+", default=["train", "val"], help="Splits to include in gallery when splits file exists.")
    parser.add_argument("--max-per-id", type=int, default=10, help="Max images per ID (0 for all).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding extraction.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--prefix", default="artifacts/index/chimp_min10_auto", help="Prefix for saved index files.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    annot_path = _pick_annotation_file(annotations_dir)
    splits_path = _pick_splits_file(annotations_dir)

    annotations = _load_annotation(annot_path)
    splits = _load_splits(splits_path) if splits_path else None

    cfg_obj = load_config(args.config)
    cfg = cfg_obj.as_dict() if isinstance(cfg_obj, TrainingConfig) else cfg_obj
    ckpt_path = args.ckpt or cfg.get("inference", {}).get("checkpoint", "artifacts/chimp-min10-resnet50-arcface-full_best.pt")

    model_bundle = load_model_from_config(args.config, ckpt_path, device=args.device)
    if model_bundle["device"].type == "cpu" and args.device == "cuda":
        print("CUDA not available; using CPU.", file=sys.stderr)

    entries = _select_gallery_entries(
        annotations=annotations,
        splits=splits,
        gallery_splits=args.gallery_splits,
        max_per_id=args.max_per_id,
        seed=args.seed,
    )
    if not entries:
        raise RuntimeError("No entries selected for index.")

    embeddings, labels, meta = _compute_embeddings(
        model_bundle=model_bundle,
        entries=entries,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    index = SimpleIndex()
    index.add(embeddings, labels, meta)
    index.save(args.prefix)

    # save entries csv
    csv_path = Path(str(args.prefix) + "_entries.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "individual_id", "image_path", "split"])
        writer.writeheader()
        for i, (lbl, m) in enumerate(zip(labels, meta)):
            writer.writerow({"row_id": i, "individual_id": lbl, "image_path": m.get("image_path", ""), "split": m.get("split", "")})

    # summary stats
    counts = defaultdict(int)
    for lbl in labels:
        counts[lbl] += 1
    num_ids = len(counts)
    num_images = len(labels)
    min_ct = min(counts.values())
    max_ct = max(counts.values())
    mean_ct = sum(counts.values()) / num_ids

    print(f"Annotation file: {annot_path}")
    if splits_path:
        print(f"Splits file: {splits_path} (gallery splits: {args.gallery_splits})")
    else:
        print("Splits file: None (using all annotations)")
    print(f"Selected images: {num_images} across {num_ids} IDs (min {min_ct}, max {max_ct}, mean {mean_ct:.2f})")
    print(f"Index saved to prefix: {args.prefix}_embeddings.npy / _meta.json")
    print(f"Entries CSV: {csv_path}")


if __name__ == "__main__":
    main()
