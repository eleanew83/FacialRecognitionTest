#!/usr/bin/env python3
"""Rename files with double spaces, log changes, and optionally regenerate/sync.

Usage:
  python3 clean_double_spaces.py \
    --cleaned /home/ylj20/Gibraltar_Macaques_Photos_Cleaned \
    --flatten-script /home/ylj20/FacialRecognitionTest/scripts/flatten_macaque_dirs.py \
    --flattened /home/ylj20/macaque_flattened \
    --split /home/ylj20/FacialRecognitionTest/macaque_split_data

Set --no-flatten or --no-split-sync to skip those steps.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def iter_files(root: Path):
    for r, _, files in os.walk(root):
        for f in files:
            if f.startswith('._'):
                continue
            yield Path(r) / f


def list_double_space_files(cleaned_root: Path):
    return [p for p in iter_files(cleaned_root) if '  ' in p.name]


def rename_double_spaces(cleaned_root: Path, rename_log: Path, conflict_log: Path):
    renamed = []
    conflicts = []
    for p in sorted(list_double_space_files(cleaned_root)):
        new_name = re.sub(r' {2,}', ' ', p.name)
        if new_name == p.name:
            continue
        new_path = p.with_name(new_name)
        if new_path.exists():
            conflicts.append(f"{p} -> {new_path} (exists)")
            continue
        p.rename(new_path)
        renamed.append(f"{p} -> {new_path}")
    rename_log.write_text("\n".join(renamed) + ("\n" if renamed else ""), encoding="utf-8")
    conflict_log.write_text("\n".join(conflicts) + ("\n" if conflicts else ""), encoding="utf-8")
    return len(renamed), len(conflicts)


def regenerate_flattened(flatten_script: Path):
    return os.system(f"python3 {flatten_script}")


def sync_split_from_flattened(flat_root: Path, split_root: Path):
    # Build flattened index: individual -> {filename_lower: Path}
    flat_index = {}
    for p in iter_files(flat_root):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        ind = p.parent.name
        flat_index.setdefault(ind.lower(), {})[p.name.lower()] = p

    updated = 0
    missing = []
    for split in ["train", "val", "test"]:
        base = split_root / split
        if not base.exists():
            continue
        for p in iter_files(base):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if "  " not in p.name:
                continue
            ind = p.parent.name
            new_name = re.sub(r" {2,}", " ", p.name)
            src = flat_index.get(ind.lower(), {}).get(new_name.lower())
            if src is None:
                missing.append(str(p))
                continue
            p.unlink()
            dst = p.with_name(src.name)
            shutil.copy2(src, dst)
            updated += 1
    return updated, missing


def main():
    parser = argparse.ArgumentParser(description="Clean double-space filenames.")
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--flattened", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--flatten-script", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--no-flatten", action="store_true")
    parser.add_argument("--no-split-sync", action="store_true")
    args = parser.parse_args()

    cleaned = Path(args.cleaned)
    flattened = Path(args.flattened)
    split = Path(args.split)
    flatten_script = Path(args.flatten_script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_dir / "double_space_files_cleaned.txt"
    rename_log = out_dir / "double_space_renames_cleaned.txt"
    conflict_log = out_dir / "double_space_conflicts_cleaned.txt"

    doubles = list_double_space_files(cleaned)
    list_path.write_text("\n".join(str(p) for p in sorted(doubles)) + ("\n" if doubles else ""), encoding="utf-8")

    renamed_count, conflict_count = rename_double_spaces(cleaned, rename_log, conflict_log)
    print(f"double_space_count={len(doubles)} renamed={renamed_count} conflicts={conflict_count}")
    print(f"list={list_path} renames={rename_log} conflicts={conflict_log}")

    if not args.no_flatten:
        ret = regenerate_flattened(flatten_script)
        print(f"flatten_exit={ret}")

    if not args.no_split_sync:
        updated, missing = sync_split_from_flattened(flattened, split)
        print(f"split_updated={updated} split_missing={len(missing)}")
        if missing:
            miss_path = out_dir / "double_space_split_missing.txt"
            miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
            print(f"split_missing_list={miss_path}")


if __name__ == "__main__":
    main()
