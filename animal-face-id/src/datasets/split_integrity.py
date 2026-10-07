"""Photo identity and content checks shared by splitting and dataset loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def source_photo_key(path: str | Path) -> str:
    """Match a source photo's renamed, detector-cropped and edited copies.

    Camera frame numbers remain significant; different frames are not merged.
    The key spans identity folders: two faces from one source photo belong
    in the same split.
    """
    name = Path(path).name.casefold()
    while True:
        previous = name
        if Path(name).suffix in IMAGE_EXTENSIONS:
            name = Path(name).stem
        name = re.sub(r"(?:_crop\d+|_cropped|_(?=[0-9a-f]{8}$)(?=[0-9a-f]*[a-f])[0-9a-f]{8})$", "", name)
        if name == previous:
            break
    return re.sub(r"\s+", " ", name).strip()


def file_fingerprint(path: str | Path) -> dict:
    """Identify exact files and decoded RGB duplicates."""
    from PIL import Image

    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        image = image.convert("RGB")
        pixels = hashlib.sha256()
        pixels.update(f"{image.width},{image.height}:RGB:".encode("ascii"))
        pixels.update(image.tobytes())
        size = [image.width, image.height]
    stat = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
        raise RuntimeError(f"Image changed while hashing: {path}")
    return {
        "sha256": digest.hexdigest(), "pixels_sha256": pixels.hexdigest(),
        "size": size, "file_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def image_inventory(raw_root: str | Path) -> set[str]:
    """Include every visible image, including stale files in unexpected subfolders."""
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {root}")
    return {str(path.relative_to(root)) for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.startswith(".") for part in path.relative_to(root).parts)}


@lru_cache(maxsize=20000)
def _cached_fingerprint(path: str, device: int, inode: int, size: int, mtime: int, ctime: int) -> dict:
    fingerprint = file_fingerprint(path)
    stat = Path(path).stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
            device, inode, size, mtime, ctime):
        raise RuntimeError(f"Image changed while verifying: {path}")
    return fingerprint


def verified_fingerprint(path: str | Path) -> dict:
    """Reuse hashes within one process only while inode and change time agree."""
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_fingerprint(str(path), stat.st_dev, stat.st_ino, stat.st_size,
                               stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def fresh_output_directory(output: str | Path):
    """Lock a destination and publish a complete generation without merging.

    Failed generations stay in a separate hidden directory for review.
    The caller must validate the yielded directory before leaving the context.
    """
    import fcntl

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / ("." + output.name + ".build.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is building this output: {output}") from error
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Output is not empty; choose a fresh directory: {output}")
        stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".build-", dir=output.parent))
        try:
            yield stage
            if output.exists():
                if not output.is_dir() or any(output.iterdir()):
                    raise FileExistsError(f"Output changed during generation: {output}")
                output.rmdir()
            os.rename(stage, output)
        except BaseException as error:
            if stage.exists():
                (stage / ".generation_failed.json").write_text(json.dumps({
                    "target": str(output), "error": str(error), "published": False,
                }, indent=2) + "\n")
                print(f"Generation was not published; files retained for review at {stage}", file=sys.stderr)
            raise


def validate_manifest(payload: dict, raw_root: str | Path | None = None, *,
                      verify_content: bool = False, expected_labels: set[str] | None = None) -> None:
    """Reject exact duplicates, stale folders, changed images, and missing identities.

    With verify_content, compare actual byte/RGB hashes with the manifest.
    Legacy records without both hashes are fingerprinted when a root is supplied.
    """
    seen_paths = set()
    owners = {}
    content_labels = {}
    root = Path(raw_root).resolve() if raw_root is not None else None
    for split in SPLIT_ORDER:
        if split not in payload or not isinstance(payload[split], list):
            raise ValueError(f"Missing or invalid split: {split}")
        for sample in payload[split]:
            path = sample["path"]
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe sample path: {path}")
            if len(relative.parts) != 3 or relative.parts[:2] != (split, sample["id"]):
                raise ValueError(f"Sample identity/split does not match its folder: {path}")
            if path in seen_paths:
                raise ValueError(f"Repeated sample path: {path}")
            seen_paths.add(path)
            hashes = {field: sample.get(field) for field in ("sha256", "pixels_sha256")}
            if root is not None:
                full_path = root / path
                if not full_path.is_file():
                    raise FileNotFoundError(f"Missing sample: {path}")
                if not full_path.resolve().is_relative_to(root):
                    raise ValueError(f"Image points outside the dataset: {path}")
                if "file_bytes" in sample and "mtime_ns" in sample:
                    stat = full_path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (sample["file_bytes"], sample["mtime_ns"]):
                        raise ValueError(f"Image changed after split audit; re-audit before training: {path}")
                if verify_content or not all(hashes.values()):
                    actual = verified_fingerprint(full_path)
                    for field in hashes:
                        if hashes[field] and hashes[field] != actual[field]:
                            raise ValueError(f"Image hash disagrees with manifest ({field}): {path}")
                        hashes[field] = actual[field]
            for field, digest in hashes.items():
                if not digest:
                    continue
                key = (field, digest)
                previous_label = content_labels.setdefault(key, sample["id"])
                if previous_label != sample["id"]:
                    raise ValueError(f"Duplicate image has conflicting identities: {owners[key][1]} and {path}")
                previous = owners.setdefault(key, (split, path))
                if previous[1] != path:
                    raise ValueError(f"Exact duplicate content occurs in {previous[0]} and {split}: {previous[1]} and {path}")
    if root is not None:
        actual = image_inventory(root)
        unexpected, missing = sorted(actual - seen_paths), sorted(seen_paths - actual)
        if unexpected or missing:
            raise ValueError(f"Image folders disagree with manifest; unlisted={unexpected[:5]}, missing={missing[:5]}")
    labels = {sample["id"] for split in SPLIT_ORDER for sample in payload[split]}
    if not labels:
        raise ValueError("Manifest contains no images.")
    if expected_labels is not None and labels != expected_labels:
        raise ValueError(f"Identity coverage changed; missing={sorted(expected_labels - labels)}, unexpected={sorted(labels - expected_labels)}")
    for split in SPLIT_ORDER:
        missing = labels - {sample["id"] for sample in payload[split]}
        if missing:
            raise ValueError(f"Closed-set identities missing from {split}: {sorted(missing)}")


def group_photos(records: list[dict], *, include_source_names: bool = False) -> list[list[dict]]:
    """Group exact content across labels; filename grouping is optional only.

    Different crops with identical names remain independent by default.
    ``include_source_names`` supports historical source-photo audits, not deduplication.
    """
    records = sorted(records, key=lambda record: record["path"])
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, record in enumerate(records):
        keys = [("source", source_photo_key(record["path"]))] if include_source_names else []
        for field in ("sha256", "pixels_sha256"):
            if record.get(field):
                keys.append((field, record[field]))
        for key in keys:
            previous = owners.setdefault(key, index)
            a, b = find(index), find(previous)
            if a != b:
                parents[max(a, b)] = min(a, b)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return list(groups.values())


def choose_representative(records: list[dict]) -> dict:
    """Prefer an existing train image, then an unedited source filename."""
    return min(records, key=lambda record: (
        SPLIT_ORDER.get(record.get("split"), 0),
        "_cropped" in Path(record["path"]).stem.casefold(),
        record["path"],
    ))


def partition_photos(groups: list[list[dict]], ratios: tuple[float, ...], seed: int) -> dict:
    """Assign whole photos, balancing identities without splitting a photo."""
    import random
    from collections import Counter

    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("Train/val/test ratios must all be positive.")
    total = sum(ratios)
    ratios = tuple(value / total for value in ratios)
    for group in groups:
        content_labels = {}
        for record in group:
            for field in ("sha256", "pixels_sha256"):
                if record.get(field):
                    labels = content_labels.setdefault((field, record[field]), set())
                    labels.add(record["id"])
                    if len(labels) > 1:
                        raise ValueError("Exact duplicate image has conflicting identities; review labels before splitting.")
    counts = Counter(record["id"] for group in groups
                     for record in choose_per_identity(group))
    small = sorted(label for label, count in counts.items() if count < 3)
    if small:
        raise ValueError(f"Fewer than three independent photos for: {small}")
    assigned = {split: Counter() for split in SPLIT_ORDER}
    remaining = counts.copy()
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda group: sum(1 / counts[label]
                                       for label in sorted({r["id"] for r in group})), reverse=True)
    result = {split: [] for split in SPLIT_ORDER}
    for group in shuffled:
        representatives = choose_per_identity(group)
        labels = [record["id"] for record in representatives]
        candidates = set(SPLIT_ORDER)
        for label in labels:
            missing = {split for split in SPLIT_ORDER if not assigned[split][label]}
            if remaining[label] == len(missing):
                candidates &= missing
        if not candidates:
            raise ValueError("Shared-photo constraints prevent a complete closed-set split.")
        split = max(sorted(candidates, key=SPLIT_ORDER.get), key=lambda name: sum(
            (counts[label] * ratios[SPLIT_ORDER[name]] - assigned[name][label])
            / max(1, counts[label] * ratios[SPLIT_ORDER[name]]) for label in labels))
        result[split].extend(representatives)
        for label in labels:
            assigned[split][label] += 1
            remaining[label] -= 1
    for split in SPLIT_ORDER:
        missing = sorted(label for label in counts if not assigned[split][label])
        if missing:
            raise ValueError(f"Identities missing from {split}: {missing}")
    return result


def choose_per_identity(group: list[dict]) -> list[dict]:
    from collections import defaultdict

    by_label = defaultdict(list)
    for record in group:
        by_label[record["id"]].append(record)
    return [choose_representative(by_label[label]) for label in sorted(by_label)]
