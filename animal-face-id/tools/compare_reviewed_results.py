"""Freeze common held-out photos and compare saved per-image predictions.

The primary reference is the October canonical 149-identity evaluation. Saved
predictions reproduce its original score before scoring the unchanged shared
cohort. Earlier 156-identity scores remain historical, not interchangeable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_reviewed_dataset import camera_key, read, sha, write

REFERENCE_DIRECTORY = ROOT / 'artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval'
REFERENCE_STEM = ('train_macaque_arcface_aggressive_rot45_macaque-resnet50-arcface_aggressive_rot45_'
    'canonical_20261014_best_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_mean_test_tta_mean')


def frame_tokens(row):
    names = set(row.get('camera_aliases', []))
    if row.get('source_group'):
        names.add('source_group:' + row['source_group'])
    for field in ('camera_key', 'master_source_path', 'source_path', 'input_path'):
        if row.get(field):
            names.add(camera_key(row[field]))
    return names


def common_cohort(previous, current, split):
    old = {(r['id'], r['sha256'], r['pixels_sha256']): r for r in previous[split]}
    excluded_splits = ['train'] if split == 'val' else ['train', 'val']
    forbidden_hashes, forbidden_frames = set(), set()
    for manifest in (previous, current):
        for excluded in excluded_splits:
            for row in manifest[excluded]:
                forbidden_hashes.update([row['sha256'], row['pixels_sha256']])
                forbidden_frames.update(frame_tokens(row))
    selected, excluded = [], []
    for row in current[split]:
        token = (row['id'], row['sha256'], row['pixels_sha256'])
        if token not in old:
            excluded.append({'path': row['path'], 'reason': 'Not identical content/label in both held-out splits.'})
        elif ({row['sha256'], row['pixels_sha256']} & forbidden_hashes or
              (frame_tokens(row) | frame_tokens(old[token])) & forbidden_frames):
            excluded.append({'path': row['path'], 'reason': 'Same content or camera frame appears in a training/tuning split.'})
        else:
            selected.append((row, old[token]))
    return selected, excluded


def metrics(rows, class_names):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    truth = [r['true_id'] for r in rows]
    predicted = [r['pred_id'] for r in rows]
    return {'num_samples': len(rows), 'num_classes': len(class_names),
            'top1': float(accuracy_score(truth, predicted)),
            'f1_macro': float(f1_score(truth, predicted, labels=class_names, average='macro', zero_division=0)),
            'precision_macro': float(precision_score(truth, predicted, labels=class_names, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(truth, predicted, labels=class_names, average='macro', zero_division=0))}


def prediction_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'image_path', 'true_id', 'pred_id'} <= set(rows[0]):
        raise ValueError('Expected the evaluator per-image prediction CSV.')
    seen = set()
    for row in rows:
        if row['image_path'] in seen:
            raise ValueError('Repeated prediction image: ' + row['image_path'])
        seen.add(row['image_path'])
        if 'is_correct' in row and int(row['is_correct']) != int(row['true_id'] == row['pred_id']):
            raise ValueError('Prediction correctness field disagrees with labels.')
    return rows


def align_predictions(rows, samples, class_names, verify_files=True):
    known = {}
    for sample in samples:
        for field in ('historical_path', 'current_path'):
            known[sample[field]] = sample
    selected = {}
    for row in rows:
        sample = known.get(str(Path(row['image_path'])))
        if sample is None:
            continue
        token = sample['sha256']
        if token in selected:
            raise ValueError('Two predictions refer to the same frozen sample.')
        if row['true_id'] != sample['id'] or row['pred_id'] not in class_names:
            raise ValueError('Prediction labels do not match the frozen canonical taxonomy.')
        if verify_files and sha(row['image_path']) != sample['sha256']:
            raise ValueError('Evaluation image changed: ' + row['image_path'])
        selected[token] = {**row, 'source_group': sample['source_group'],
                           'sha256': token, 'benchmark_id': sample['id']}
    if set(selected) != {s['sha256'] for s in samples}:
        raise ValueError('Predictions do not cover the complete frozen cohort.')
    return [selected[s['sha256']] for s in samples]


def paired_accuracy(reference, candidate, repeats=2000, seed=42):
    import numpy as np
    groups = {}
    for old, new in zip(reference, candidate):
        if old['sha256'] != new['sha256']:
            raise ValueError('Paired predictions must be aligned by exact content.')
        group = groups.setdefault(old['source_group'], [0, 0])
        group[0] += int(new['true_id'] == new['pred_id']) - int(old['true_id'] == old['pred_id'])
        group[1] += 1
    values = np.asarray(list(groups.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    changes = values[draws, 0].sum(1) / values[draws, 1].sum(1)
    return {'accuracy_delta': float(values[:, 0].sum() / values[:, 1].sum()),
            'accuracy_delta_ci95': np.quantile(changes, [.025, .975]).tolist(),
            'bootstrap_unit': 'camera-source group', 'bootstrap_repeats': repeats, 'seed': seed}


def save_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(generation, output):
    import yaml
    generation, output = Path(generation).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh comparison directory; frozen benchmarks are not overwritten.')
    previous_path = ROOT / 'data/macaque_faces/splits.json'
    current_path = generation / 'faces/splits.json'
    previous, current = read(previous_path), read(current_path)
    classes = sorted({r['id'] for rows in previous.values() for r in rows})
    summary_path = REFERENCE_DIRECTORY / (REFERENCE_STEM + '_metrics_summary.json')
    predictions_path = REFERENCE_DIRECTORY / (REFERENCE_STEM + '_hard_examples.csv')
    historical_summary = read(summary_path)
    original_predictions = prediction_rows(predictions_path)
    historical_root = Path(read(ROOT / 'reports/crop_review_2026-11-10/inventory.json')['crop_root'])
    expected = {str(historical_root / r['path']): r for r in previous['test']}
    if {r['image_path'] for r in original_predictions} != set(expected):
        raise ValueError('Saved baseline predictions do not match the preserved historical test manifest.')
    for row in original_predictions:
        if row['true_id'] != expected[row['image_path']]['id']:
            raise ValueError('Baseline prediction identity disagrees with historical manifest.')
    reproduced = metrics(original_predictions, classes)
    for field in ('top1', 'f1_macro', 'precision_macro', 'recall_macro'):
        if abs(reproduced[field] - historical_summary[field]) > 1e-12:
            raise ValueError('Baseline predictions do not reproduce historical metric: ' + field)
    benchmark = {'schema_version': 1, 'name': 'common_holdout_20261125', 'class_names': classes,
                 'generation': str(generation), 'splits': {},
                 'protocol': 'Identical bytes, pixels and canonical identity in both corresponding held-out splits; exclude camera/content exposure in train and, for test, validation. Match the validation-selected reference TTA/angles and logit adjustment.',
                 'reference_evaluation': {k: historical_summary[k] for k in ('tta_mode','tta_angles','logit_adjust_tau')},
                 'input_sha256': {str(p): sha(p) for p in [previous_path, current_path, summary_path, predictions_path]},
                 'reference_checkpoint': historical_summary['checkpoint'],
                 'reference_checkpoint_sha256': sha(ROOT / historical_summary['checkpoint']),
                 'historical_training_scope': 'October canonical 149-ID ±45-degree augmentation checkpoint, chosen using validation, with preserved training manifest. Pre-repair July training provenance is not fully available; its headline scores are historical only.'}
    exclusions = {}
    for split in ('val', 'test'):
        pairs, exclusions[split] = common_cohort(previous, current, split)
        samples = []
        for row, old in pairs:
            current_file, previous_file = generation / 'faces' / row['path'], historical_root / old['path']
            if sha(current_file) != row['sha256'] or sha(previous_file) != row['sha256']:
                raise ValueError('Common evaluation photo differs from audited bytes.')
            samples.append({'id': row['id'], 'sha256': row['sha256'], 'pixels_sha256': row['pixels_sha256'],
                            'source_group': row['source_group'], 'camera_key': row['camera_key'],
                            'current_path': str(current_file), 'historical_path': str(previous_file)})
        if {r['id'] for r in samples} != set(classes):
            raise ValueError('Shared cohort does not cover all identities: ' + split)
        benchmark['splits'][split] = sorted(samples, key=lambda r: (r['id'], r['sha256']))
    baseline = align_predictions(original_predictions, benchmark['splits']['test'], classes, verify_files=False)
    baseline_metrics = metrics(baseline, classes)
    old_by_path = {r['path']: (s, r) for s, rows in previous.items() for r in rows}
    transition = Counter()
    transitions = []
    for split, rows in current.items():
        for row in rows:
            if row['provenance'] != 'reviewed_historical_crop':
                continue
            relative = row['input_path'].split('/macaque_crops/', 1)[1]
            old_split, _ = old_by_path[relative]
            transition[old_split, split] += 1
            if old_split != split:
                transitions.append({'historical_path': relative, 'previous_split': old_split,
                                    'current_split': split, 'source_group': row['source_group']})
    output.mkdir(parents=True)
    write(output / 'benchmark.json', benchmark)
    benchmark_sha = sha(output / 'benchmark.json')
    save_csv(output / 'baseline_predictions.csv', baseline)
    save_csv(output / 'split_changes.csv', transitions)
    write(output / 'excluded_from_common_cohort.json', exclusions)
    result = {'benchmark': str(output / 'benchmark.json'), 'benchmark_sha256': benchmark_sha,
              'shared_validation_count': len(benchmark['splits']['val']),
              'shared_test_count': len(benchmark['splits']['test']), 'identities': len(classes),
              'previous_full_test': historical_summary, 'previous_score_on_common_test': baseline_metrics,
              'unchanged_crop_split_transitions': {f'{a}_to_{b}': n for (a,b),n in transition.items()},
              'unchanged_crops_staying_in_split': sum(n for (a,b),n in transition.items() if a==b),
              'unchanged_crops_moving_split': len(transitions),
              'physical_splits_modified': False, 'all_core_images_verified_identical': True,
              'scores_recovered_from_complete_saved_predictions': True}
    write(output / 'summary.json', result)
    configs = {}
    for path in (ROOT / 'configs').glob('*.yaml'):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get('data', {}).get('dataset_name') != 'macaque_faces':
            continue
        configs[path] = path.read_bytes()
    plans = {}
    for path, original in configs.items():
        cfg = yaml.safe_load(original)
        cfg['data'].update(evaluation_subset=str(output / 'benchmark.json'),
                           evaluation_subset_sha256=benchmark_sha)
        backup = output / 'configs_before_common_cohort' / path.name
        backup.parent.mkdir(exist_ok=True)
        backup.write_bytes(original)
        plans[path] = ('# Reviewed training data; fixed common validation and test photos.\n' + yaml.safe_dump(cfg, sort_keys=False)).encode()
    reviewed_config = yaml.safe_load(plans[ROOT / 'configs/train_macaque_arcface_aggressive_rot45.yaml'])
    reviewed_config['data']['evaluation_tta'] = 'mean'
    for name, legacy in [('compare_before_review', True), ('compare_after_review', False)]:
        cfg = copy.deepcopy(reviewed_config)
        cfg['name'] = name + '_common_holdout_20261125'
        cfg['data']['comparison_role'] = 'historical_baseline' if legacy else 'reviewed_candidate'
        if legacy:
            cfg['data'].update(raw_root=str(historical_root), splits_path=str(previous_path),
                               identity_registry=str(ROOT / 'data/macaque_faces/identity_registry.json'))
        cfg['inference'].update(checkpoint='artifacts/' + cfg['name'] + '_best.pt',
                                gallery_path='artifacts/' + cfg['name'] + '_gallery.pkl')
        path = ROOT / 'configs' / (name + '.yaml')
        if path.exists():
            raise FileExistsError('Comparison config already exists: ' + str(path))
        plans[path] = yaml.safe_dump(cfg, sort_keys=False).encode()
    full_evaluation = copy.deepcopy(reviewed_config)
    full_evaluation['data'].pop('evaluation_subset')
    full_evaluation['data'].pop('evaluation_subset_sha256')
    (output / 'eval_full_reviewed.yaml').write_text(yaml.safe_dump(full_evaluation, sort_keys=False))
    pointer_path = ROOT / 'data/macaque_faces/current_dataset.json'
    pointer_before = pointer_path.read_bytes()
    pointer = read(pointer_path)
    pointer.update(comparison_protocol=str(output / 'benchmark.json'),
                   comparison_summary=str(output / 'summary.json'),
                   primary_evaluation_counts={'val': result['shared_validation_count'], 'test': result['shared_test_count']})
    def replace(path, raw):
        temporary = path.with_name('.' + path.name + '.comparison.tmp')
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    try:
        for path, encoded in plans.items():
            if path in configs and path.read_bytes() != configs[path]:
                raise ValueError('Config changed during benchmark activation: ' + str(path))
            replace(path, encoded)
        replace(pointer_path, (json.dumps(pointer, indent=2)+'\n').encode())
    except BaseException:
        for path in plans:
            if path in configs:
                replace(path, configs[path])
            else:
                path.unlink(missing_ok=True)
        replace(pointer_path, pointer_before)
        raise
    print(json.dumps(result, indent=2), flush=True)


def compare(benchmark_path, predictions_path, summary_path, output):
    benchmark, evaluation = read(benchmark_path), read(summary_path)
    expected = benchmark.get('reference_evaluation', {'tta_mode':'none','tta_angles':[], 'logit_adjust_tau':0})
    if (evaluation.get('tta_mode', 'none') != expected['tta_mode'] or
            evaluation.get('logit_adjust_tau', 0) != expected['logit_adjust_tau'] or
            evaluation.get('tta_angles', []) != expected['tta_angles']):
        raise ValueError('Primary comparison requires the reference TTA mode, angles and logit adjustment.')
    if evaluation.get('num_classes') != len(benchmark['class_names']) or evaluation.get('split', 'test') != 'test':
        raise ValueError('Candidate evaluation must use the canonical test taxonomy.')
    folder = Path(benchmark_path).parent
    reference = prediction_rows(folder / 'baseline_predictions.csv')
    candidate_rows = prediction_rows(predictions_path)
    reproduced = metrics(candidate_rows, benchmark['class_names'])
    if len(candidate_rows) != evaluation['num_samples'] or any(
            abs(reproduced[field]-evaluation[field]) > 1e-12 for field in ('top1','f1_macro')):
        raise ValueError('Candidate prediction CSV and metric summary belong to different evaluations.')
    candidate = align_predictions(candidate_rows, benchmark['splits']['test'], benchmark['class_names'])
    reference = align_predictions(reference, benchmark['splits']['test'], benchmark['class_names'], verify_files=False)
    old, new = metrics(reference, benchmark['class_names']), metrics(candidate, benchmark['class_names'])
    result = {'benchmark': str(Path(benchmark_path).resolve()), 'benchmark_sha256': sha(benchmark_path),
              'previous': old, 'candidate': new, 'candidate_checkpoint': evaluation['checkpoint'],
              'candidate_predictions_sha256': sha(predictions_path),
              'candidate_metrics_sha256': sha(summary_path),
              'f1_macro_delta': new['f1_macro']-old['f1_macro'], **paired_accuracy(reference, candidate)}
    write(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('build')
    make.add_argument('--generation', type=Path, default=ROOT / 'data/macaque_faces/reviewed_2026-11-25')
    make.add_argument('--output', type=Path, default=ROOT / 'data/macaque_faces/comparison_2026-11-25')
    score = commands.add_parser('compare')
    score.add_argument('--benchmark', type=Path, required=True)
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--metrics', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.generation, args.output)
    else:
        compare(args.benchmark, args.predictions, args.metrics, args.output)


if __name__ == '__main__':
    main()
