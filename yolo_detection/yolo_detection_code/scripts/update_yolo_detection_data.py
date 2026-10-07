#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'animal-face-id'))
from src.datasets.detection_integrity import apply_dataset_changes
from src.datasets.split_integrity import file_fingerprint


VALID_EXTS = {".jpg", ".jpeg", ".png"}
FLAG_SET = {"o", "r", "d"}


def build_source_index(source_dir: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
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
            matches = sorted(entry for entry in os.listdir(directory) if normalize_filename(entry) == target)
            if len(matches) > 1:
                raise ValueError(f"Ambiguous source filename in {directory}: {name}")
            if matches:
                return os.path.join(directory, matches[0])
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
        fingerprints = {file_fingerprint(path)["pixels_sha256"] for path in candidates}
        if len(fingerprints) != 1:
            raise ValueError(f"Ambiguous source filename has distinct crops: {filename}; use source_relative_path")
        return sorted(candidates)[0]
    return None


def label_path_for_image(yolo_data_dir: str, split: str, image_filename: str) -> str:
    base = os.path.splitext(image_filename)[0] + ".txt"
    return os.path.join(yolo_data_dir, "labels", split, base)


def build_update_plan(rows, source_dir, yolo_data_dir, *, only=None, replacement_labels_dir=None):
    """Plan changes without modifying a split; refuse ambiguous names/collisions.

    Optional TSV columns: split (train/val), source_relative_path (disambiguates
    same-named source crops). Replacement labels mirror split/original-stem.txt.
    """
    source_root, data = Path(source_dir).resolve(), Path(yolo_data_dir).resolve()
    index = build_source_index(str(source_root))
    image_changes, label_changes = {}, {}
    stats = {key: 0 for key in ('rows', 'selected', 'deleted', 'replaced', 'renamed')}
    for row in rows:
        stats['rows'] += 1
        filename, flags = parse_image_field((row.get('image') or '').strip())
        if not flags or (only and only not in flags):
            continue
        if Path(filename).name != filename or Path(filename).suffix.lower() not in VALID_EXTS:
            raise ValueError(f'Invalid image filename: {filename}')
        targets = find_targets(str(data), filename)
        specified_split = (row.get('split') or '').strip()
        if specified_split:
            if specified_split not in ('train', 'val'):
                raise ValueError(f'Invalid split: {specified_split}')
            targets = [(s, p) for s, p in targets if s == specified_split]
        if not targets:
            raise FileNotFoundError(f'Target not found: {specified_split}/{filename}')
        if len(targets) != 1:
            raise ValueError(f'Ambiguous target filename in multiple splits: {filename}; add a split column')
        split, image_path = targets[0]
        image_relative = str(Path(image_path).relative_to(data))
        label_relative = str(Path(label_path_for_image(str(data), split, filename)).relative_to(data))
        if image_relative in image_changes:
            raise ValueError(f'Multiple updates target the same image: {filename}')
        old_labels = (data / label_relative).read_bytes()
        stats['selected'] += 1
        if 'd' in flags:
            image_changes[image_relative] = None
            label_changes[label_relative] = None
            stats['deleted'] += 1
            continue
        base = strip_hash_suffix(filename)
        if base is None:
            raise ValueError(f'Missing hash suffix: {filename}')
        if 'r' in flags:
            base = re.sub(r'\s+_', '_', corrected_filename_for_r(base))
        source_path = None
        if 'o' in flags:
            explicit_source = (row.get('source_relative_path') or '').strip()
            if explicit_source:
                source_path = (source_root / explicit_source).resolve()
                if not source_path.is_relative_to(source_root) or not source_path.is_file():
                    raise ValueError(f'Invalid source_relative_path: {explicit_source}')
            else:
                source_path = find_source_in_structure(str(source_root), (row.get('group') or '').strip(),
                                                       (row.get('individual') or '').strip(), base, index)
            if source_path is None:
                raise FileNotFoundError(f'Source not found: {base}')
            before, after = file_fingerprint(image_path), file_fingerprint(source_path)
            if before['pixels_sha256'] != after['pixels_sha256']:
                if replacement_labels_dir is None:
                    raise ValueError(f'Replacement changes image pixels: {filename}; provide --replacement-labels-dir')
                replacement = Path(replacement_labels_dir) / split / Path(filename).with_suffix('.txt')
                old_labels = replacement.read_bytes()
            stats['replaced'] += 1
        new_filename = filename
        if 'r' in flags:
            fingerprint = file_fingerprint(source_path or image_path)
            new_filename = f'{Path(base).stem}_{fingerprint["pixels_sha256"][:8]}{Path(base).suffix}'
            stats['renamed'] += 1
        new_image = str(Path('images') / split / new_filename)
        new_label = str(Path('labels') / split / Path(new_filename).with_suffix('.txt'))
        if new_image != image_relative:
            if (data / new_image).exists() or (data / new_label).exists() or new_image in image_changes or new_label in label_changes:
                raise FileExistsError(f'Rename would overwrite an existing image or annotation: {new_filename}')
            image_changes[image_relative] = None
            label_changes[label_relative] = None
        image_changes[new_image] = Path(source_path or image_path)
        label_changes[new_label] = old_labels
    return image_changes, label_changes, stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply reviewed TSV edits with exact-duplicate checks and rollback.')
    parser.add_argument('--tsv', required=True, help='TSV with image flags o (replace), r (rename), d (delete)')
    parser.add_argument('--source-dir', required=True, help='Source photo directory')
    parser.add_argument('--yolo-data-dir', required=True, help='Existing detector dataset directory')
    parser.add_argument('--replacement-labels-dir', help='Reviewed labels for changed pixels: split/original-stem.txt')
    parser.add_argument('--only', choices=['o', 'r', 'd'], help='Only select rows containing this flag')
    parser.add_argument('--dry-run', action='store_true', help='Validate the proposed dataset without changing files')
    args = parser.parse_args()
    with open(args.tsv, newline='') as handle:
        image_changes, label_changes, stats = build_update_plan(
            csv.DictReader(handle, delimiter='\t'), args.source_dir, args.yolo_data_dir,
            only=args.only, replacement_labels_dir=args.replacement_labels_dir)
    result = apply_dataset_changes(args.yolo_data_dir, image_changes, label_changes, dry_run=args.dry_run)
    for relative, source in image_changes.items():
        print(f'{"REMOVE" if source is None else "WRITE"}: {relative}')
    print(f'Validated splits: {result["image_counts"]}; no exact duplicates. Counts: {stats}')
    if result.get('backup_directory'):
        print(f'Original files and update journal: {result["backup_directory"]}')


if __name__ == '__main__':
    main()
