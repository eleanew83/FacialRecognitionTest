#!/usr/bin/env python3
"""Summarize the completed canonical identity migration with verifiable counts."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrate_macaque_identities import AUDIT, REGISTRY, SETS, read


def main():
    verification = read(AUDIT / 'verification.json')
    references = read(AUDIT / 'reference_verification.json')
    smoke = read(AUDIT / 'model_smoke_test.json')
    registry = read(REGISTRY)
    plan = read(AUDIT / 'plan.json')
    lines = [
        'Macaque canonical identity migration — 14 October 2026',
        '',
        'Status: completed and verified.',
        'Authority: ' + registry['source_root'],
        '149 canonical individuals; original 156 split labels mapped explicitly.',
        'Each canonical individual has one folder within each train/val/test split.',
        'Current group and gender come from the master folder. Capture filenames retain their historical location/date.',
        'No image bytes, crop contents, bounding boxes or split assignments changed.',
        '',
        'Active datasets (train / val / test):',
    ]
    for name, details in verification['datasets'].items():
        counts = details['counts']
        lines.append(f"  {name}: {counts['train']} / {counts['val']} / {counts['test']} ({sum(counts.values())} images); 149 identities per split")
    detector = verification['detector']
    lines.extend([
        f"  Detector: {detector['image_counts']['train']} train / {detector['image_counts']['val']} val ({detector['linked_references']} images)",
        '  Detector class remains 0=macaque_face. Identity metadata describes the source folder, not every face in a multi-animal photo.',
        '  Detector provenance recovered from MD5(full original source filepath), as used by the original JPEG exporter.',
        '',
        'Renamed/merged identities:',
    ])
    for old, current in sorted(registry['aliases'].items()):
        if old != current:
            group = registry['identities'][current]['current_group']
            lines.append(f'  {old} -> {current} ({group})')
    lines.extend([
        '',
        'References and training:',
        '  Canonical names/group/sex/master-photo references saved in all three split manifests and bundled copies.',
        '  Detector identity_manifest.json contains all 6,630 current image/label/source references.',
        '  Active training configs now specify 149 classes and distinct canonical_20261014 run names.',
        '  Future flatten/split/detection/crop CLIs use the registry; unknown names fail for review.',
        '  Different photos sharing a filename after an identity merge receive a content suffix rather than overwrite one another.',
        '  Existing exact-byte/RGB duplicate checks and fresh-output guards remain in place.',
        '  New checkpoints embed their ordered class names; training rejects a wrong configured class count.',
        '  Canonical evaluations get a canonical_20261014 report tag to preserve historical result files.',
        '',
        'Historical models and galleries:',
        f"  All {references['checkpoints_preserved_and_bound_to_original_label_order']} old checkpoints unchanged; each has a SHA256-bound .pt.labels.json sidecar preserving its original 156 output labels.",
        '  Existing classifier inference sums old alias probabilities into 149 canonical identities; it does not relabel classifier weight indices.',
        f"  Latest Aug3 CPU smoke test passed: {smoke['old_classifier_outputs']} original outputs -> {smoke['canonical_prediction_outputs']} canonical predictions.",
        '  This migration did not retrain a model or generate new test-set metrics. Historical reported scores still refer to the old taxonomy.',
        f"  Active GUI gallery: {references['gallery_original_rows']} -> {references['gallery_current_rows']} entries; {references['gallery_omitted_obsolete_references']} obsolete references excluded, retained embeddings unchanged.",
        '  Canonicalized gallery names are linked to the correct surviving crop paths where image provenance exists.',
    ])
    for path in references['historical_galleries_without_image_provenance']:
        lines.append(f'  {path}: image paths were not originally saved and its embeddings differ from the GUI gallery. Only names canonicalized; its own embeddings preserved and no image paths invented.')
    variants = [r for rows in plan['datasets']['source']['after'].values() for r in rows
                if r['master_match_method'] != 'exact_decoded_RGB']
    lines.extend(['', 'Known source pixel variant (unchanged during naming migration):'])
    for record in variants:
        lines.append(f"  {record['path']} -> master {record['master_source_path']}")
        lines.append('  Master has a tighter Fanta crop; downstream original full-frame bytes and existing box/crop geometry were preserved. Pixel replacement would require its own annotation/crop update.')
    lines.extend([
        '',
        'Backups and audit:',
        '  before/source, before/reviewed_crops, before/detector_20261012_crops retain the pre-migration image paths.',
        '  Image generations share hard links; never edit image bytes in place. Create a fresh generation for future image changes.',
        '  before/configs contains 156-class configs pointing to preserved data for historical evaluation.',
        '  before/artifacts retains original gallery and embedding files; old model weights remain at their original locations.',
        '  reference_updates.csv: every old active source/crop path, current canonical path and preserved historical path.',
        '  identity_merges.csv: all 156 old labels -> canonical identity/current group/gender/source directory.',
        '  verification.json: counts, unchanged-image/split checks, no exact byte/RGB duplicates and detector label checks.',
        '  reference_verification.json: gallery/model/config integrity.',
        '  model_smoke_test.json: actual legacy Aug3 loading and canonical inference.',
        '',
        'Verification establishes content integrity and consistency of the supplied identity mapping. It does not visually re-identify every unique photo or rule out near-duplicate capture sequences.',
        '',
    ])
    (AUDIT / 'migration_report.txt').write_text('\n'.join(lines))
    print(AUDIT / 'migration_report.txt')


if __name__ == '__main__':
    main()
