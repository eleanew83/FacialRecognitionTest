#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset validation script for chimpanzee_faces dataset.
Validates folder structure, annotation paths, and ID consistency.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Base paths
BASE_DIR = Path("data/chimpanzee_faces")
RAW_DIR = BASE_DIR / "raw"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

def validate_annotation_file(annotation_file, base_raw_dir):
    """
    Validate that all image paths in annotation file exist.

    Returns:
        dict with validation results
    """
    print(f"\n{'='*80}")
    print(f"Validating: {annotation_file.name}")
    print(f"{'='*80}")

    results = {
        'total_lines': 0,
        'valid_paths': 0,
        'missing_paths': [],
        'ctai_count': 0,
        'czoo_count': 0,
        'unique_ids': set(),
        'id_image_counts': defaultdict(int)
    }

    with open(annotation_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: Line {line_num} has invalid format: {line}")
                continue

            results['total_lines'] += 1
            relative_path = parts[0]
            identity_id = parts[1]

            # Track unique IDs and their image counts
            results['unique_ids'].add(identity_id)
            results['id_image_counts'][identity_id] += 1

            # Check if image exists
            # The path should be relative to raw directory
            full_path = base_raw_dir / relative_path

            if full_path.exists():
                results['valid_paths'] += 1

                # Count CTai vs CZoo
                if 'data_CTai' in relative_path:
                    results['ctai_count'] += 1
                elif 'data_CZoo' in relative_path:
                    results['czoo_count'] += 1
            else:
                results['missing_paths'].append({
                    'line': line_num,
                    'path': relative_path,
                    'full_path': str(full_path)
                })

    # Print results
    print(f"\nTotal lines: {results['total_lines']}")
    print(f"Valid image paths: {results['valid_paths']}")
    print(f"Missing image paths: {len(results['missing_paths'])}")
    print(f"\nBreakdown by dataset:")
    print(f"  CTai images: {results['ctai_count']}")
    print(f"  CZoo images: {results['czoo_count']}")
    print(f"\nUnique IDs: {len(results['unique_ids'])}")

    if results['missing_paths']:
        print(f"\n⚠️  MISSING FILES ({len(results['missing_paths'])} total):")
        for item in results['missing_paths'][:10]:  # Show first 10
            print(f"  Line {item['line']}: {item['path']}")
        if len(results['missing_paths']) > 10:
            print(f"  ... and {len(results['missing_paths']) - 10} more")
    else:
        print(f"\n✓ All image paths are valid!")

    return results

def validate_id_consistency(kept_ids_file, annotation_results):
    """
    Validate ID consistency between kept_ids_min10.txt and annotation file.
    """
    print(f"\n{'='*80}")
    print(f"Validating ID Consistency")
    print(f"{'='*80}")

    # Read kept IDs
    with open(kept_ids_file, 'r') as f:
        kept_ids = set(line.strip() for line in f if line.strip())

    print(f"\nKept IDs (from kept_ids_min10.txt): {len(kept_ids)}")

    # Get IDs from annotation
    annotation_ids = annotation_results['unique_ids']
    id_counts = annotation_results['id_image_counts']

    print(f"IDs in annotation file: {len(annotation_ids)}")

    # Find discrepancies
    ids_not_in_kept = annotation_ids - kept_ids
    kept_ids_not_in_annotation = kept_ids - annotation_ids

    # Find IDs with < 10 images
    ids_below_threshold = {id_: count for id_, count in id_counts.items() if count < 10}

    # Print results
    if ids_not_in_kept:
        print(f"\n⚠️  IDs in annotation but NOT in kept_ids_min10.txt ({len(ids_not_in_kept)}):")
        for id_ in sorted(list(ids_not_in_kept)[:20]):
            print(f"  {id_} ({id_counts[id_]} images)")
        if len(ids_not_in_kept) > 20:
            print(f"  ... and {len(ids_not_in_kept) - 20} more")
    else:
        print(f"\n✓ All annotation IDs are in kept_ids_min10.txt")

    if kept_ids_not_in_annotation:
        print(f"\n⚠️  IDs in kept_ids_min10.txt but NOT in annotation ({len(kept_ids_not_in_annotation)}):")
        for id_ in sorted(list(kept_ids_not_in_annotation)[:20]):
            print(f"  {id_}")
        if len(kept_ids_not_in_annotation) > 20:
            print(f"  ... and {len(kept_ids_not_in_annotation) - 20} more")
    else:
        print(f"\n✓ All kept IDs appear in annotation file")

    if ids_below_threshold:
        print(f"\n⚠️  IDs with < 10 images ({len(ids_below_threshold)}):")
        for id_, count in sorted(ids_below_threshold.items(), key=lambda x: x[1])[:20]:
            print(f"  {id_}: {count} images")
        if len(ids_below_threshold) > 20:
            print(f"  ... and {len(ids_below_threshold) - 20} more")
    else:
        print(f"\n✓ All IDs have >= 10 images")

    return {
        'kept_ids_count': len(kept_ids),
        'annotation_ids_count': len(annotation_ids),
        'ids_not_in_kept': list(ids_not_in_kept),
        'kept_ids_not_in_annotation': list(kept_ids_not_in_annotation),
        'ids_below_threshold': ids_below_threshold
    }

def main():
    print("CHIMPANZEE FACES DATASET VALIDATION")
    print("="*80)

    # Validate annotations_merged_all.txt
    all_results = validate_annotation_file(
        ANNOTATIONS_DIR / "annotations_merged_all.txt",
        RAW_DIR
    )

    # Validate annotations_merged_min10.txt
    min10_results = validate_annotation_file(
        ANNOTATIONS_DIR / "annotations_merged_min10.txt",
        RAW_DIR
    )

    # Validate ID consistency for min10
    consistency_results = validate_id_consistency(
        ANNOTATIONS_DIR / "kept_ids_min10.txt",
        min10_results
    )

    # Generate summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    all_passed = True

    print(f"\n1. Folder Structure: ", end="")
    if (BASE_DIR / "raw").exists() and (BASE_DIR / "annotations").exists() and (BASE_DIR / "processed").exists():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False

    print(f"\n2. annotations_merged_all.txt:")
    print(f"   - Total images: {all_results['total_lines']}")
    print(f"   - Valid paths: {all_results['valid_paths']}")
    print(f"   - Missing paths: {len(all_results['missing_paths'])}")
    print(f"   - Status: ", end="")
    if len(all_results['missing_paths']) == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False

    print(f"\n3. annotations_merged_min10.txt:")
    print(f"   - Total images: {min10_results['total_lines']}")
    print(f"   - Valid paths: {min10_results['valid_paths']}")
    print(f"   - Missing paths: {len(min10_results['missing_paths'])}")
    print(f"   - Unique IDs: {len(min10_results['unique_ids'])}")
    print(f"   - Status: ", end="")
    if len(min10_results['missing_paths']) == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False

    print(f"\n4. ID Consistency (min10):")
    print(f"   - IDs in kept_ids_min10.txt: {consistency_results['kept_ids_count']}")
    print(f"   - IDs with < 10 images: {len(consistency_results['ids_below_threshold'])}")
    print(f"   - IDs in annotation but not in kept_ids: {len(consistency_results['ids_not_in_kept'])}")
    print(f"   - Status: ", end="")
    if len(consistency_results['ids_below_threshold']) == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("✓✓✓ Dataset structure verified — ready for model training.")
    else:
        print("⚠️  Dataset has issues that need to be resolved.")
    print(f"{'='*80}\n")

    # Save detailed results to JSON
    detailed_results = {
        'annotations_merged_all': {
            'total_lines': all_results['total_lines'],
            'valid_paths': all_results['valid_paths'],
            'missing_paths_count': len(all_results['missing_paths']),
            'missing_paths': all_results['missing_paths'][:100],  # First 100
            'ctai_count': all_results['ctai_count'],
            'czoo_count': all_results['czoo_count'],
            'unique_ids_count': len(all_results['unique_ids'])
        },
        'annotations_merged_min10': {
            'total_lines': min10_results['total_lines'],
            'valid_paths': min10_results['valid_paths'],
            'missing_paths_count': len(min10_results['missing_paths']),
            'missing_paths': min10_results['missing_paths'][:100],
            'ctai_count': min10_results['ctai_count'],
            'czoo_count': min10_results['czoo_count'],
            'unique_ids_count': len(min10_results['unique_ids'])
        },
        'id_consistency': consistency_results
    }

    with open('validation_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"Detailed results saved to: validation_results.json\n")

    return all_passed

if __name__ == "__main__":
    main()
