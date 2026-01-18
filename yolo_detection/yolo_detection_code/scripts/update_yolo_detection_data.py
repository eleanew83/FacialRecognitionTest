#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


VALID_EXTS = {".jpg", ".jpeg", ".png"}
FLAG_SET = {"o", "r", "d"}


def build_source_index(source_dir: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in VALID_EXTS:
                index[filename.lower()].append(os.path.join(root, filename))
    return index


def parse_image_field(field: str) -> Tuple[str, List[str]]:
    value = field.rstrip()
    flags: List[str] = []
    while True:
        match = re.search(r"\s+([ord])$", value)
        if not match:
            break
        flags.insert(0, match.group(1))
        value = value[: match.start()].rstrip()
    filename = value
    return filename, flags


def strip_hash_suffix(filename: str) -> Optional[str]:
    match = re.match(r"^(.*)_[0-9a-fA-F]{8}(\.[^.]+)$", filename)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"


def corrected_filename_for_r(base_filename: str) -> str:
    root, ext = os.path.splitext(base_filename)
    root = root.rstrip()
    return f"{root}{ext}"


def hash_for_path(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]


def find_targets(yolo_data_dir: str, filename: str) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    for split in ("train", "val"):
        path = os.path.join(yolo_data_dir, "images", split, filename)
        if os.path.exists(path):
            targets.append((split, path))
    return targets


def find_source_in_structure(
    source_dir: str,
    group: str,
    individual: str,
    filename: str,
    index: Dict[str, List[str]],
) -> Optional[str]:
    def normalize_filename(name: str) -> str:
        root, ext = os.path.splitext(name)
        root = root.rstrip()
        root = re.sub(r"\s+_", "_", root)
        return f"{root}{ext}".lower()

    def get_dir_match(directory: str, name: str) -> Optional[str]:
        if not os.path.isdir(directory):
            return None
        try:
            for entry in os.listdir(directory):
                if entry.lower() == name.lower():
                    return os.path.join(directory, entry)
        except OSError:
            return None
        return None

    def get_file_match(directory: str, name: str) -> Optional[str]:
        if not os.path.isdir(directory):
            return None
        target = normalize_filename(name)
        try:
            for entry in os.listdir(directory):
                if normalize_filename(entry) == target:
                    return os.path.join(directory, entry)
        except OSError:
            return None
        return None

    for gender in ("female", "females", "male", "males"):
        candidate_dir = os.path.join(source_dir, group, gender, individual)
        match = get_file_match(candidate_dir, filename)
        if match:
            return match

        # Case-insensitive group/individual resolution
        group_dir = get_dir_match(source_dir, group)
        if group_dir:
            gender_dir = get_dir_match(group_dir, gender)
            if gender_dir:
                individual_dir = get_dir_match(gender_dir, individual)
                if individual_dir:
                    match = get_file_match(individual_dir, filename)
                    if match:
                        return match

    # Fall back to filename index across the tree (case-insensitive).
    candidates = index.get(filename.lower(), [])
    if candidates:
        return candidates[0]
    return None


def label_path_for_image(yolo_data_dir: str, split: str, image_filename: str) -> str:
    base = os.path.splitext(image_filename)[0] + ".txt"
    return os.path.join(yolo_data_dir, "labels", split, base)


def safe_copy(src: str, dst: str, dry_run: bool) -> None:
    if dry_run:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def safe_move(src: str, dst: str, dry_run: bool) -> None:
    if dry_run:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def safe_remove(path: str, dry_run: bool) -> None:
    if dry_run:
        return
    if os.path.exists(path):
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply multiface_report_grouped_lines.tsv updates to yolo_detection_data."
    )
    parser.add_argument("--tsv", required=True, help="Path to multiface_report_grouped_lines.tsv")
    parser.add_argument("--source-dir", required=True, help="Path to Gibraltar_Macaques_Photos_Cleaned")
    parser.add_argument(
        "--yolo-data-dir",
        required=True,
        help="Path to yolo_detection_data directory",
    )
    parser.add_argument(
        "--only",
        choices=["o", "r", "d"],
        help="Only apply rows that contain this flag.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    args = parser.parse_args()

    source_index = build_source_index(args.source_dir)

    stats = {
        "rows": 0,
        "skip_no_flags": 0,
        "skip_bad_filename": 0,
        "missing_source": 0,
        "missing_target": 0,
        "deleted": 0,
        "replaced": 0,
        "renamed": 0,
    }

    with open(args.tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows"] += 1
            raw_field = (row.get("image") or "").strip()
            if not raw_field:
                continue

            filename, flags = parse_image_field(raw_field)
            if not flags:
                stats["skip_no_flags"] += 1
                continue
            if args.only and args.only not in flags:
                stats["skip_no_flags"] += 1
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTS:
                stats["skip_bad_filename"] += 1
                print(f"[WARN] Skipping invalid filename: {filename}")
                continue

            targets = find_targets(args.yolo_data_dir, filename)
            if not targets:
                stats["missing_target"] += 1
                print(f"[WARN] Target not found in yolo_detection_data: {filename}")
                continue

            if "d" in flags:
                for split, img_path in targets:
                    label_path = label_path_for_image(args.yolo_data_dir, split, filename)
                    print(f"[DEL] {img_path}")
                    print(f"[DEL] {label_path}")
                    safe_remove(img_path, args.dry_run)
                    safe_remove(label_path, args.dry_run)
                stats["deleted"] += len(targets)
                continue

            base_no_hash = strip_hash_suffix(filename)
            if base_no_hash is None:
                stats["skip_bad_filename"] += 1
                print(f"[WARN] Missing hash suffix: {filename}")
                continue

            base_for_source = base_no_hash
            if "r" in flags:
                base_for_source = corrected_filename_for_r(base_no_hash)
                # Also try removing extra spaces before underscores for known rename fixes.
                base_for_source = re.sub(r"\s+_", "_", base_for_source)

            group = row.get("group", "").strip()
            individual = row.get("individual", "").strip()
            source_path = find_source_in_structure(
                args.source_dir,
                group,
                individual,
                base_for_source,
                source_index,
            )
            if not source_path:
                stats["missing_source"] += 1
                print(f"[WARN] Source not found for: {base_for_source}")

            if "r" in flags:
                if not source_path:
                    continue
                new_hash = hash_for_path(source_path)
                root, ext = os.path.splitext(base_for_source)
                new_filename = f"{root}_{new_hash}{ext}"

                for split, img_path in targets:
                    new_img_path = os.path.join(args.yolo_data_dir, "images", split, new_filename)
                    new_label_path = label_path_for_image(args.yolo_data_dir, split, new_filename)
                    old_label_path = label_path_for_image(args.yolo_data_dir, split, filename)

                    if img_path != new_img_path:
                        print(f"[REN] {img_path} -> {new_img_path}")
                        safe_move(img_path, new_img_path, args.dry_run)
                    if os.path.exists(old_label_path):
                        print(f"[REN] {old_label_path} -> {new_label_path}")
                        safe_move(old_label_path, new_label_path, args.dry_run)

                    if "o" in flags:
                        print(f"[REP] {new_img_path} <= {source_path}")
                        safe_copy(source_path, new_img_path, args.dry_run)
                        stats["replaced"] += 1

                stats["renamed"] += len(targets)
                continue

            if "o" in flags:
                if not source_path:
                    continue
                for _, img_path in targets:
                    print(f"[REP] {img_path} <= {source_path}")
                    safe_copy(source_path, img_path, args.dry_run)
                    stats["replaced"] += 1

    print("\n=== Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
