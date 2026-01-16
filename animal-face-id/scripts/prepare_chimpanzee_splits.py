#!/usr/bin/env python3
"""Generate stratified train/val/test splits for the chimpanzee_faces min10 subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_ANNOTATION = Path("data/chimpanzee_faces/annotations/annotations_merged_min10.txt")
DEFAULT_OUTPUT = Path("data/chimpanzee_faces/annotations/splits_min10.json")
# Base directory that contains the datasets_cropped_chimpanzee_faces folder.
# Annotation paths already include the "datasets_cropped_chimpanzee_faces/..." prefix.
RAW_PREFIX = Path("data/chimpanzee_faces/raw")


def parse_annotation(annotation_path: Path) -> Dict[str, List[str]]:
    """Return a mapping from identity -> list of relative image paths."""
    samples: Dict[str, List[str]] = {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path, identity = parts[0], parts[1]
            samples.setdefault(identity, []).append(rel_path)
    return samples


def split_per_identity(
    items: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    """Split a single identity's items into train/val/test respecting minimum presence."""
    items = items.copy()
    rng.shuffle(items)
    n = len(items)
    val_count = max(1, round(n * val_ratio))
    test_count = max(1, round(n * test_ratio))
    train_count = n - val_count - test_count
    if train_count <= 0:
        # Rebalance to ensure train gets at least one sample.
        deficit = 1 - train_count
        # Steal equally from val/test where possible.
        while deficit > 0 and val_count > 1:
            val_count -= 1
            deficit -= 1
        while deficit > 0 and test_count > 1:
            test_count -= 1
            deficit -= 1
        train_count = n - val_count - test_count
        if train_count <= 0:
            msg = f"Cannot allocate at least one train sample for identity with {n} items."
            raise ValueError(msg)
    train_split = items[:train_count]
    val_split = items[train_count : train_count + val_count]
    test_split = items[train_count + val_count :]
    return train_split, val_split, test_split


def build_splits(
    samples: Dict[str, List[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Perform stratified splitting across all identities."""
    rng = random.Random(seed)
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    for identity, paths in samples.items():
        train_paths, val_paths, test_paths = split_per_identity(paths, train_ratio, val_ratio, test_ratio, rng)
        for path in train_paths:
            splits["train"].append({"path": path, "id": identity})
        for path in val_paths:
            splits["val"].append({"path": path, "id": identity})
        for path in test_paths:
            splits["test"].append({"path": path, "id": identity})

    for key in splits:
        splits[key] = sorted(splits[key], key=lambda x: (x["id"], x["path"]))
    return splits


def summarize_splits(splits: Dict[str, List[Dict[str, str]]]) -> str:
    """Return a human-readable summary string."""
    def count_ids(entries: List[Dict[str, str]]) -> int:
        return len({item["id"] for item in entries})

    summary = [
        f"Train: {len(splits['train'])} images, {count_ids(splits['train'])} ids",
        f"Val:   {len(splits['val'])} images, {count_ids(splits['val'])} ids",
        f"Test:  {len(splits['test'])} images, {count_ids(splits['test'])} ids",
    ]
    return "\n".join(summary)


def validate_paths_exist(splits: Dict[str, List[Dict[str, str]]]) -> None:
    """Sanity check that all referenced files exist under raw prefix."""
    missing = []
    for split_name, entries in splits.items():
        for entry in entries:
            full_path = RAW_PREFIX / entry["path"]
            if not full_path.exists():
                missing.append((split_name, entry["path"]))
                if len(missing) > 10:
                    break
        if missing:
            break
    if missing:
        msg = f"Missing files detected under raw prefix ({RAW_PREFIX}): {missing[:10]}"
        raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stratified splits for chimpanzee_faces min10 subset.")
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION, help="Path to annotations_merged_min10.txt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path for splits.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    samples = parse_annotation(args.annotation_file)
    splits = build_splits(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    validate_paths_exist(splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    print(f"Wrote splits to: {args.output}")
    print(summarize_splits(splits))


if __name__ == "__main__":
    main()
