#!/usr/bin/env python3
"""Flatten the authoritative master into one folder per canonical individual."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'animal-face-id'))
from src.datasets.identity_registry import IdentityRegistry, unique_destination
from src.datasets.split_integrity import file_fingerprint, fresh_output_directory, group_photos, image_inventory


def flatten_directory_structure(source_dir, target_dir, identity_registry=None):
    source, target = Path(source_dir).resolve(), Path(target_dir).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Source and output must be separate directories')
    registry = IdentityRegistry(identity_registry or REPO / 'animal-face-id/data/macaque_faces/identity_registry.json')
    records = []
    for relative in sorted(image_inventory(source)):
        parts = Path(relative).parts
        if len(parts) != 4:
            raise ValueError(f'Expected group/gender/individual/photo: {relative}')
        records.append({'path': relative, **registry.metadata(parts[2]), **file_fingerprint(source / relative)})
    with fresh_output_directory(target) as stage:
        occupied, references = set(), {}
        for group in group_photos(records):
            record = group[0]
            relative = unique_destination(Path(record['id']) / Path(record['path']).name, record['sha256'], occupied)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / record['path'], destination)
            references[str(relative)] = {key: record[key] for key in ('id', 'current_group', 'gender', 'source_directory', 'identity_registry_version')}
            references[str(relative)]['master_source_path'] = record['path']
        (stage / 'identity_registry.json').write_text(json.dumps(registry.payload, indent=2))
        (stage / 'image_references.json').write_text(json.dumps(references, indent=2))
        assert len(image_inventory(stage)) == len(references)
    return references


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO.parent / 'Gibraltar_Macaques_Photos_Cleaned_Updated')
    parser.add_argument('--target', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--identity-registry', type=Path)
    args = parser.parse_args()
    flatten_directory_structure(args.source, args.target, args.identity_registry)
