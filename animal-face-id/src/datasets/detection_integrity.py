"""Exact-content and annotation checks shared by detector preparation, updates and training.

Names never determine duplication: only file bytes or identical decoded RGB do.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .split_integrity import IMAGE_EXTENSIONS, image_inventory, verified_fingerprint


def parse_labels(content: bytes, *, description: str = 'label') -> list[tuple]:
    """Validate class-0 YOLO boxes; empty files explicitly represent negatives."""
    boxes = []
    for line_number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5 or fields[0] != '0':
                raise ValueError('expected class 0 and four coordinates')
            x, y, width, height = map(float, fields[1:])
            if not all(math.isfinite(v) for v in (x, y, width, height)):
                raise ValueError('non-finite coordinates')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError('coordinates outside [0, 1] or non-positive box size')
            if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
                raise ValueError('box extends outside the image')
            box = (0, x, y, width, height)
            if box in boxes:
                raise ValueError('repeated annotation')
            boxes.append(box)
        except ValueError as error:
            raise ValueError(f'Invalid YOLO label {description}:{line_number}: {error}') from error
    return boxes


def _safe_relative(relative: str, folder: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or len(path.parts) < 3 or path.parts[0] != folder:
        raise ValueError(f'Invalid dataset path: {relative}')
    if any(part.startswith('.') for part in path.parts):
        raise ValueError(f'Hidden dataset path: {relative}')
    return path


def _fingerprint(path: Path, trusted: dict | None) -> dict:
    # An explicitly supplied, verified audit can spare decoding unchanged images.
    # Always reread the bytes; size and mtime alone never justify reuse.
    if trusted:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, f) != getattr(after, f) for f in fields):
            raise RuntimeError(f'Image changed while hashing: {path}')
        if digest.hexdigest() == trusted['sha256']:
            return {key: trusted[key] for key in ('sha256', 'pixels_sha256', 'size')} | {
                'file_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
            }
    return verified_fingerprint(path)


def validate_detection_dataset(root: str | Path, *, image_changes: dict | None = None,
                               label_changes: dict | None = None,
                               check_config_root: bool = True,
                               trusted_fingerprints: dict | None = None,
                               allow_pending_update: bool = False) -> dict:
    """Check configured splits, every image/label pair, and all exact duplicates.

    Proposed image changes map relative paths to replacement files, labels to
    bytes, and None means removal. No files are changed by validation. Exact
    duplicates within and across splits are rejected even if boxes agree.
    """
    import yaml

    root = Path(root).resolve()
    if not allow_pending_update:
        for journal_path in root.parent.glob('.' + root.name + '.update-backup-*/operation.json'):
            try:
                journal = json.loads(journal_path.read_text())
            except (ValueError, OSError) as error:
                raise RuntimeError(f'Unreadable update journal; check recovery before using the dataset: {journal_path}') from error
            if journal.get('status') == 'applying':
                raise RuntimeError(f'Interrupted dataset update; recover originals from this journal before using it: {journal_path}')
    config = yaml.safe_load((root / 'dataset.yaml').read_text())
    if not isinstance(config, dict) or config.get('names') not in ({0: 'macaque_face'}, ['macaque_face']):
        raise ValueError('dataset.yaml must define only class 0: macaque_face')
    configured_root = Path(config.get('path') or '.').expanduser()
    if not configured_root.is_absolute():
        configured_root = root / configured_root
    if check_config_root and configured_root.resolve() != root:
        raise ValueError(f'dataset.yaml points at a different dataset: {configured_root}')
    splits = [split for split in ('train', 'val', 'test') if config.get(split)]
    if 'train' not in splits or 'val' not in splits:
        raise ValueError('Both train and val must be configured')
    for split in splits:
        if config[split] != f'images/{split}':
            raise ValueError(f'Expected {split}: images/{split}; refusing to audit different training inputs')
    images = {f'images/{p}' for p in image_inventory(root / 'images')}
    labels = {str(p.relative_to(root)) for p in (root / 'labels').rglob('*.txt')
              if p.is_file() and p.name != 'classes.txt'
              and not any(part.startswith('.') for part in p.relative_to(root).parts)}
    image_changes, label_changes = image_changes or {}, label_changes or {}
    for folder, inventory, changes in (('images', images, image_changes), ('labels', labels, label_changes)):
        for relative, replacement in changes.items():
            path = _safe_relative(relative, folder)
            if path.parts[1] not in splits or (folder == 'images' and path.suffix.lower() not in IMAGE_EXTENSIONS):
                raise ValueError(f'Unsupported image split/path: {relative}')
            if folder == 'labels' and (path.suffix != '.txt' or path.name == 'classes.txt'):
                raise ValueError(f'Unsupported annotation path: {relative}')
            if replacement is None:
                inventory.discard(relative)
            else:
                inventory.add(relative)
    for relative in images | labels:
        path = _safe_relative(relative, relative.split('/')[0])
        if path.parts[1] not in splits:
            raise ValueError(f'Unconfigured split contains data: {relative}')
    for class_file in (root / 'labels').rglob('classes.txt'):
        if not any(part.startswith('.') for part in class_file.relative_to(root).parts):
            if class_file.read_text().splitlines() != ['macaque_face']:
                raise ValueError(f'Unexpected classes.txt: {class_file}')
    expected_labels, records, owners = set(), [], {}
    for relative in sorted(images):
        path = Path(relative)
        label_relative = str(Path('labels') / Path(*path.parts[1:]).with_suffix('.txt'))
        if label_relative in expected_labels:
            raise ValueError(f'Multiple image extensions share one label: {label_relative}')
        expected_labels.add(label_relative)
        if label_relative not in labels:
            raise ValueError(f'Missing label for {relative}: {label_relative}')
        content = label_changes[label_relative] if label_relative in label_changes else (root / label_relative).read_bytes()
        boxes = parse_labels(content, description=label_relative)
        source = Path(image_changes.get(relative) or root / relative)
        fingerprint = _fingerprint(source, (trusted_fingerprints or {}).get(relative))
        for field in ('sha256', 'pixels_sha256'):
            key = (field, fingerprint[field])
            if key in owners:
                previous, previous_boxes = owners[key]
                detail = ' with conflicting annotations' if sorted(previous_boxes) != sorted(boxes) else ''
                raise ValueError(f'Exact duplicate images{detail}: {previous} and {relative}')
            owners[key] = (relative, boxes)
        records.append({'path': relative, 'label_path': label_relative, 'split': path.parts[1],
                        'boxes': boxes, 'label_sha256': hashlib.sha256(content).hexdigest(), **fingerprint})
    if labels != expected_labels:
        raise ValueError(f'Orphan labels: {sorted(labels - expected_labels)[:10]}')
    counts = dict(Counter(r['split'] for r in records))
    if any(not counts.get(split) for split in splits):
        raise ValueError('Configured splits must each contain at least one image')
    return {'image_counts': counts, 'exact_duplicate_groups': 0, 'records': records,
            'bounding_box_counts': {s: sum(len(r['boxes']) for r in records if r['split'] == s) for s in splits}}


def apply_dataset_changes(root: str | Path, image_changes: dict, label_changes: dict,
                          *, dry_run: bool = False) -> dict:
    """Validate an entire update, then apply with backups and rollback.

    Uses the preparation lock. Each destination is replaced atomically. A journal
    outside the dataset preserves originals if a process stops between files.
    """
    import fcntl

    root = Path(root).resolve()
    with (root.parent / ('.' + root.name + '.build.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is updating this dataset: {root}') from error
        validate_detection_dataset(root)
        proposed = validate_detection_dataset(root, image_changes=image_changes, label_changes=label_changes)
        if dry_run or not (image_changes or label_changes):
            return proposed
        backup = Path(tempfile.mkdtemp(prefix='.' + root.name + '.update-backup-', dir=root.parent))
        (backup / '.gitignore').write_text('*\n!.gitignore\n')
        changes = image_changes | label_changes
        caches = [str(p.relative_to(root)) for p in (root / 'labels').rglob('*.cache')]
        originals = {}
        for relative in list(changes) + caches:
            original = root / relative
            originals[relative] = original.exists()
            if original.exists():
                destination = backup / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
        journal = {'dataset': str(root), 'status': 'prepared', 'originals': originals}
        def save_journal():
            (backup / 'operation.json').write_text(json.dumps(journal, indent=2) + '\n')
        save_journal()
        with tempfile.TemporaryDirectory(prefix='.' + root.name + '.update-stage-', dir=root.parent) as temporary:
            stage = Path(temporary)
            staged_images = {}
            for relative, content in changes.items():
                if content is None:
                    if relative in image_changes:
                        staged_images[relative] = None
                    continue
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative in image_changes:
                    shutil.copy2(content, path)
                    staged_images[relative] = path
                else:
                    path.write_bytes(content)
            staged = validate_detection_dataset(root, image_changes=staged_images, label_changes=label_changes)
            if [(r['path'], r['sha256'], r['label_sha256']) for r in staged['records']] != [
                    (r['path'], r['sha256'], r['label_sha256']) for r in proposed['records']]:
                raise RuntimeError('Dataset or replacements changed while preparing the update')
            journal['status'] = 'applying'
            save_journal()
            try:
                for relative, content in changes.items():
                    destination = root / relative
                    if content is None:
                        destination.unlink(missing_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(stage / relative, destination)
                for relative in caches:
                    (root / relative).unlink(missing_ok=True)
                final = validate_detection_dataset(root, allow_pending_update=True)
                if [(r['path'], r['sha256'], r['label_sha256']) for r in final['records']] != [
                        (r['path'], r['sha256'], r['label_sha256']) for r in proposed['records']]:
                    raise RuntimeError('Committed data differ from the validated update')
            except BaseException:
                for relative, existed in originals.items():
                    destination = root / relative
                    if existed:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup / relative, destination)
                    else:
                        destination.unlink(missing_ok=True)
                journal['status'] = 'rolled_back'
                save_journal()
                raise
            journal['status'] = 'applied'
            save_journal()
            final['backup_directory'] = str(backup)
            return final
