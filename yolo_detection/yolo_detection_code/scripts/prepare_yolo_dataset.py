#!/usr/bin/env python3
"""Build a new detector dataset from exact-content groups and checked face labels."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'animal-face-id'))
from src.datasets.detection_integrity import parse_labels, validate_detection_dataset
from src.datasets.split_integrity import file_fingerprint, fresh_output_directory, group_photos, image_inventory

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / 'macaque_split_data' / 'train'
DATA_DIR = REPO_ROOT / 'yolo_detection' / 'yolo_detection_data'


def detection_labels(model, image_path: Path, confidence: float) -> bytes:
    """Use every detected macaque face; never fabricate a centre-box fallback."""
    names = model.names
    if names not in ({0: 'macaque_face'}, ['macaque_face']):
        raise ValueError('Annotation proposals require a macaque_face checkpoint; COCO class 0 is person')
    lines = []
    for result in model(str(image_path), verbose=False, conf=confidence):
        for box in result.boxes:
            if int(box.cls.item()) == 0 and box.conf.item() >= confidence:
                x, y, width, height = box.xywhn[0].tolist()
                lines.append(f'0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}')
    if not lines:
        raise ValueError(f'No face detected: {image_path}; annotate manually instead of creating a placeholder')
    content = ('\n'.join(lines) + '\n').encode()
    parse_labels(content, description=str(image_path))
    return content


def prepare_dataset(source_dir, output_dir, *, labels_dir=None, model=None,
                    train_ratio=0.8, seed=42, confidence=0.3):
    """Preserve original image bytes; collapse only exact content before splitting.

    Manual labels mirror the source image directory. Conflicting annotations for
    identical images must be resolved manually. Model outputs are proposals that
    still require visual review; format checks cannot establish label accuracy.
    Existing datasets are never cleared or merged.
    """
    import yaml

    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    labels = Path(labels_dir).resolve() if labels_dir is not None else None
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Source and output directories must be disjoint')
    if labels is not None and (labels == output or labels.is_relative_to(output) or output.is_relative_to(labels)):
        raise ValueError('Labels and output directories must be disjoint')
    if not 0 < train_ratio < 1 or not 0 <= confidence <= 1:
        raise ValueError('Invalid split ratio or confidence threshold')
    if (labels is None) == (model is None):
        raise ValueError('Provide either reviewed --labels-dir or a macaque-face --model')
    if model is not None and model.names not in ({0: 'macaque_face'}, ['macaque_face']):
        raise ValueError('Annotation proposals require a macaque_face checkpoint; COCO class 0 is person')
    with fresh_output_directory(output) as stage:
        records = []
        for relative in sorted(image_inventory(source)):
            path = source / relative
            record = {'path': relative, **file_fingerprint(path)}
            if labels is not None:
                label_path = labels / Path(relative).with_suffix('.txt')
                record['label_content'] = label_path.read_bytes()
                record['boxes'] = parse_labels(record['label_content'], description=str(label_path))
            records.append(record)
        groups = group_photos(records)
        if len(groups) < 2:
            raise ValueError('At least two distinct images are required for train and val')
        representatives = []
        for group in groups:
            if labels is not None and len({tuple(sorted(r['boxes'])) for r in group}) > 1:
                raise ValueError(f'Exact duplicate images have conflicting annotations: {[r["path"] for r in group]}')
            representatives.append(group[0])
        random.Random(seed).shuffle(representatives)
        train_count = max(1, min(len(representatives) - 1, int(len(representatives) * train_ratio)))
        provenance, destinations = [], set()
        for index, record in enumerate(representatives):
            split = 'train' if index < train_count else 'val'
            original = source / record['path']
            # Content suffixes remain stable after moves. The extension is preserved.
            filename = f'{original.stem}_{record["pixels_sha256"][:16]}{original.suffix.lower()}'
            image_relative = Path('images') / split / filename
            label_relative = Path('labels') / split / Path(filename).with_suffix('.txt')
            if image_relative in destinations or label_relative in destinations:
                raise ValueError(f'Output filename collision: {filename}')
            destinations.update((image_relative, label_relative))
            image_path = stage / image_relative
            label_path = stage / label_relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, image_path)
            if file_fingerprint(image_path)['sha256'] != record['sha256']:
                raise RuntimeError(f'Source image changed during preparation: {original}')
            content = record['label_content'] if labels is not None else detection_labels(model, image_path, confidence)
            label_path.write_bytes(content)
            provenance.append({'source': record['path'], 'image': str(image_relative), 'label': str(label_relative),
                               'sha256': record['sha256'], 'pixels_sha256': record['pixels_sha256']})
        for split in ('train', 'val'):
            (stage / 'labels' / split / 'classes.txt').write_text('macaque_face\n')
        config = {'path': str(output), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'macaque_face'}}
        (stage / 'dataset.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        result = validate_detection_dataset(stage, check_config_root=False)
        (stage / 'preparation_manifest.json').write_text(json.dumps({
            'seed': seed, 'train_ratio': train_ratio, 'source_image_count': len(records),
            'exact_copies_omitted': len(records) - len(groups), 'image_counts': result['image_counts'],
            'annotation_source': 'reviewed_labels' if labels is not None else 'model_proposals_require_visual_review',
            'records': provenance,
        }, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR, help='Source images, including nested folders')
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty dataset directory')
    annotation = parser.add_mutually_exclusive_group(required=True)
    annotation.add_argument('--labels-dir', type=Path, help='Reviewed YOLO labels mirroring source image paths')
    annotation.add_argument('--model', type=Path, help='Local macaque_face checkpoint for annotation proposals')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', type=float, default=0.3)
    args = parser.parse_args()
    model = None
    if args.model is not None:
        if not args.model.is_file():
            parser.error('--model must be an existing local macaque-face checkpoint')
        from ultralytics import YOLO
        model = YOLO(str(args.model))
    result = prepare_dataset(args.source_dir, args.output_dir, labels_dir=args.labels_dir, model=model,
                             train_ratio=args.train_ratio, seed=args.seed, confidence=args.confidence)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
    if model is not None:
        print('Review every proposed face box, especially validation labels, before training.')


if __name__ == '__main__':
    main()
