#!/usr/bin/env python3
"""Split images deterministically, grouping only exact duplicate content."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Share exact-content grouping with the active macaque training pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "animal-face-id"))
from src.datasets.split_integrity import (
    IMAGE_EXTENSIONS, SPLIT_ORDER, file_fingerprint, fresh_output_directory, group_photos,
    partition_photos, validate_manifest,
)


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15,
                  test_ratio=0.15, random_seed=42):
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Source and target must be separate, non-nested directories.")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Split output is not empty; choose a fresh target: {target}")

    records = []
    for individual in sorted(source.iterdir()):
        if not individual.is_dir() or individual.name.startswith("."):
            continue
        for image in sorted(individual.iterdir()):
            if image.is_file() and not image.name.startswith(".") and image.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "id": individual.name, "path": str(image.relative_to(source)),
                    **file_fingerprint(image),
                })
    if not records:
        raise ValueError(f"No source images found under {source}")
    groups = group_photos(records)
    splits = partition_photos(groups, (train_ratio, val_ratio, test_ratio), random_seed)
    report = {"seed": random_seed, "source_images": len(records),
              "exact_content_groups": len(groups), "duplicate_policy": "exact bytes or decoded RGB only", "splits": {}}
    manifest = {split: [] for split in SPLIT_ORDER}
    with fresh_output_directory(target) as staging:
        for split, entries in splits.items():
            report["splits"][split] = len(entries)
            for record in entries:
                image = source / record["path"]
                relative = Path(split) / record["id"] / image.name
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, destination)
                manifest[split].append({**record, "path": str(relative)})
        validate_manifest(manifest, staging, verify_content=True)
        (staging / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for split in SPLIT_ORDER:
        print(f"{split}: {len(splits[split])} images")
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True, help="New or empty output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.source, args.target, args.train_ratio, args.val_ratio,
                  args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
