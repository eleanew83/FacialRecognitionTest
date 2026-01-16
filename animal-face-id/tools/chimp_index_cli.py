#!/usr/bin/env python3
"""CLI helper for building and updating chimpanzee embedding indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.index_store import SimpleIndex, add_individual, build_index_from_folder
from src.inference.inference_core import load_model_from_config


def build_cmd(args: argparse.Namespace) -> None:
    model_bundle = load_model_from_config(args.config, args.ckpt, device=args.device)
    index = build_index_from_folder(model_bundle, args.folder, glob_pattern=args.pattern)
    index.save(args.prefix)
    print(f"Built index with {index.size} entries from pattern '{args.pattern}' and saved to prefix: {args.prefix}")


def add_cmd(args: argparse.Namespace) -> None:
    model_bundle = load_model_from_config(args.config, args.ckpt, device=args.device)
    index = SimpleIndex.load(args.prefix)
    add_individual(model_bundle, index, args.name, args.images, aggregate=args.aggregate)
    index.save(args.prefix)
    print(f"Added {len(args.images)} images under '{args.name}' and saved to prefix: {args.prefix}")


def stats_cmd(args: argparse.Namespace) -> None:
    index = SimpleIndex.load(args.prefix)
    print(f"Index size: {index.size}")
    counts = {}
    for label in index.labels:
        counts[label] = counts.get(label, 0) + 1
    for label, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"{label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimpanzee index management CLI.")
    parser.add_argument("--prefix", default="artifacts/index/chimp_index", help="Path prefix for index files.")
    parser.add_argument("--config", default="configs/train_chimp_min10_resnet50_arc_full.yaml", help="Training config path.")
    parser.add_argument("--ckpt", default="artifacts/chimp-min10-resnet50-arcface-full_best.pt", help="Checkpoint path for embeddings.")
    parser.add_argument("--device", default="cuda", help="Device for embedding extraction.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    build_p = subparsers.add_parser("build", help="Build index from a folder of images (label = parent folder name).")
    build_p.add_argument("--folder", required=True, help="Root folder containing images in subfolders.")
    build_p.add_argument("--pattern", default="**/*.png", help="Glob pattern for images (default: **/*.png).")

    add_p = subparsers.add_parser("add", help="Add a new individual to an existing index.")
    add_p.add_argument("--name", required=True, help="Individual name/ID.")
    add_p.add_argument("--images", nargs="+", required=True, help="Images for the new individual.")
    add_p.add_argument("--aggregate", action="store_true", help="Average embeddings into a single vector.")

    subparsers.add_parser("stats", help="List counts per individual in the index.")

    args = parser.parse_args()
    if args.command == "build":
        build_cmd(args)
    elif args.command == "add":
        add_cmd(args)
    elif args.command == "stats":
        stats_cmd(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
