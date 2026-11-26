"""Train the reviewed ±45-degree model and compare it with the saved winner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.evaluation_subset import select_evaluation_subset
from src.datasets.split_integrity import validate_manifest

WINNING_CONFIG = ROOT / 'artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/configs/train_macaque_arcface_aggressive_rot45.yaml'
DEFAULT_CONFIG = ROOT / 'configs/compare_after_review.yaml'
DEFAULT_RUN = ROOT / 'artifacts/reviewed_training/rot45_mean_20261125'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(config_path, run_dir):
    cfg = yaml.safe_load(config_path.read_text())
    winner = yaml.safe_load(WINNING_CONFIG.read_text())
    if cfg['model'] != winner['model']:
        raise ValueError('Model settings differ from the previous ±45-degree winner.')
    for key in ('batch_size', 'num_workers', 'image_size', 'num_classes', 'mean', 'std', 'augment', 'aggressive'):
        if cfg['data'][key] != winner['data'][key]:
            raise ValueError('Winning training setting changed: data.' + key)
    parameters = lambda c: {k: v for k, v in c['trainer'].items() if k != 'checkpoint_dir'}
    if parameters(cfg) != parameters(winner) or cfg['data'].get('evaluation_tta') != 'mean':
        raise ValueError('Use the original training parameters and mean TTA.')
    data = cfg['data']
    if not Path(data['raw_root']).is_dir():
        raise ValueError('Reviewed face folder is missing.')
    paths = {key: Path(data[key]) for key in ('splits_path', 'identity_registry', 'evaluation_subset')}
    fingerprints = {key: sha(path) for key, path in paths.items()}
    if fingerprints['evaluation_subset'] != data['evaluation_subset_sha256']:
        raise ValueError('Frozen benchmark changed.')
    benchmark = json.loads(paths['evaluation_subset'].read_text())
    if fingerprints['splits_path'] != benchmark['input_sha256'][str(paths['splits_path'])]:
        raise ValueError('Reviewed splits changed after the comparison was frozen.')
    registry = json.loads(paths['identity_registry'].read_text())
    payload = json.loads(paths['splits_path'].read_text())
    validate_manifest(payload, expected_labels=set(registry['identities']))
    effective = {'train': len(payload['train'])}
    for split in ('val', 'test'):
        selected, _ = select_evaluation_subset(payload[split], split, paths['evaluation_subset'],
            sorted(registry['identities']), fingerprints['evaluation_subset'])
        effective[split] = len(selected)
    checkpoint_dir = Path(cfg['trainer']['checkpoint_dir'])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    outputs = {suffix: checkpoint_dir / (cfg['name'] + '_' + suffix)
               for suffix in ('best.pt', 'last.pt', 'metrics.csv')}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing or run_dir.exists():
        raise FileExistsError('A run already exists; preserve it and select a new run name: ' + ', '.join(existing or [str(run_dir)]))
    plan = {'config': str(config_path), 'run_name': cfg['name'], 'run_directory': str(run_dir),
            'previous_winning_config': str(WINNING_CONFIG), 'winning_hyperparameters_match': True,
            'physical_splits': {s: len(rows) for s, rows in payload.items()},
            'effective_train_validation_test': effective, 'identities': len(registry['identities']),
            'evaluation_tta': 'mean', 'evaluation_angles': [-30, -20, -10, 0, 10, 20, 30],
            'input_sha256': {**{str(paths[k]): v for k, v in fingerprints.items()},
                             str(WINNING_CONFIG): sha(WINNING_CONFIG)},
            'config_sha256': sha(config_path), 'outputs': {k: str(p) for k, p in outputs.items()}}
    return cfg, plan


def run(config_path, run_dir, check_only=False):
    cfg, plan = preflight(config_path, run_dir)
    print(json.dumps(plan, indent=2), flush=True)
    if check_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this training job through Slurm on a GPU node.')
    checkpoint_dir = Path(plan['outputs']['best.pt']).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / (cfg['name'] + '.training.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, plan = preflight(config_path, run_dir)
        run_dir.mkdir(parents=True)
        snapshot = run_dir / 'config.yaml'
        shutil.copy2(WINNING_CONFIG, run_dir / 'previous_winning_config.yaml')
        comparison_dir = run_dir / 'comparison'
        comparison_dir.mkdir()
        source_comparison = Path(cfg['data']['evaluation_subset']).parent
        for name in ('benchmark.json', 'baseline_predictions.csv', 'summary.json'):
            shutil.copy2(source_comparison / name, comparison_dir / name)
        cfg['data']['evaluation_subset'] = str(comparison_dir / 'benchmark.json')
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
        plan['runtime_config'] = str(snapshot)
        plan['runtime_config_sha256'] = sha(snapshot)
        for path, expected in plan['input_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Training input changed during setup: ' + path)
        (run_dir / 'setup.json').write_text(json.dumps(plan, indent=2) + '\n')
        os.chdir(ROOT)
        from src.training.train import train_closed_set
        from src.training.evaluate import run_final_evaluation
        from compare_reviewed_results import compare
        print('Training reviewed data with the original winning ±45-degree settings.', flush=True)
        train_closed_set(cfg)
        result = run_final_evaluation(str(snapshot), plan['outputs']['best.pt'], device='cuda',
            tta_mode='mean', tag='reviewed_rot45_mean', out_dir=run_dir / 'eval',
            make_plots=False, write_report=False)
        prefix = str(result['_per_class_csv']).removesuffix('_per_class_metrics.csv')
        compare(cfg['data']['evaluation_subset'], Path(prefix + '_hard_examples.csv'),
            Path(prefix + '_metrics_summary.json'), run_dir / 'paired_comparison.json')
        print('Completed training, mean-TTA evaluation and paired comparison: ' + str(run_dir), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.config.resolve(), args.run_dir.resolve(), args.check_only)


if __name__ == '__main__':
    main()
