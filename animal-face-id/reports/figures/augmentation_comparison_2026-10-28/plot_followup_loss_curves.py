"""Reproduce the ±45°/±60° loss figures from saved epoch CSVs (no model execution)."""

from pathlib import Path
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIGURES = Path(__file__).resolve().parent
ROOT = FIGURES.parents[2]
SUITE = ROOT / 'artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_epochs(path):
    with path.open(newline='') as handle:
        rows = [{key: int(value) if key == 'epoch' else float(value)
                 for key, value in row.items()} for row in csv.DictReader(handle)]
    assert [row['epoch'] for row in rows] == list(range(1, 31)), path
    return rows


def main():
    original = json.loads((FIGURES / 'plot_data.json').read_text())
    source = ROOT / original['curve_sources']['baseline']['path']
    assert digest(source) == original['curve_sources']['baseline']['sha256']
    baseline = read_epochs(source)
    curves = {}
    sources = {'baseline': {'path': str(source.relative_to(ROOT)), 'sha256': digest(source)}}
    best_epochs = {}
    for angle, task in ((45, 0), (60, 1)):
        path = SUITE / f'checkpoints/macaque-resnet50-arcface_aggressive_rot{angle}_canonical_20261014_metrics.csv'
        rows = read_epochs(path)
        # Cross-check every CSV epoch against the rounded stdout record.
        log = SUITE / f'logs/39484293_{task}.out'
        pattern = r'\[Epoch (\d+)/30\].*train_loss=([\d.]+) top1=([\d.]+) \| val_loss=([\d.]+) top1=([\d.]+)'
        logged = re.findall(pattern, log.read_text())
        assert len(logged) == 30
        for row, values in zip(rows, logged):
            assert row['epoch'] == int(values[0])
            for key, value in zip(('train_loss', 'train_top1', 'val_loss', 'val_top1'), values[1:]):
                assert abs(row[key] - float(value)) < 0.000051, (angle, row['epoch'], key)
        curves[f'rot{angle}'] = rows
        best_epochs[f'rot{angle}'] = max(rows, key=lambda row: row['val_top1'])['epoch']
        sources[f'rot{angle}'] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                                'stdout_path': str(log.relative_to(ROOT)), 'stdout_sha256': digest(log)}
    assert best_epochs == {'rot45': 27, 'rot60': 30}
    plt.rcParams.update({'font.family': 'Carlito', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False})

    def panel(ax, angle):
        rows = curves[f'rot{angle}']
        epochs = [row['epoch'] for row in rows]
        ax.plot(epochs, [row['train_loss'] for row in rows], color='#0072B2', lw=2, label='Training')
        ax.plot(epochs, [row['val_loss'] for row in rows], color='#D55E00', lw=2, label='Validation')
        ax.plot(epochs, [row['val_loss'] for row in baseline], color='#777777', ls=':', lw=1.7, label='Baseline validation')
        ax.axvline(best_epochs[f'rot{angle}'], color='#777777', ls='--', lw=1)
        ax.set(title=f'±{angle}° policy · selected epoch {best_epochs[f"rot{angle}"]}',
               xlabel='Epoch', ylabel='ArcFace loss', xlim=(1, 30.5), ylim=(0, 20))
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.grid(alpha=.18)
        ax.legend(loc='upper right', frameon=True, facecolor='white',
                  edgecolor='none', framealpha=1, fontsize=9)

    for angle in (45, 60):
        fig, ax = plt.subplots(figsize=(6, 2.4), layout='constrained')
        panel(ax, angle)
        for extension in ('png', 'pdf'):
            fig.savefig(FIGURES / f'rot{angle}_loss.{extension}', dpi=240)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), layout='constrained')
    for ax, angle in zip(axes, (45, 60)):
        panel(ax, angle)
    for extension in ('png', 'pdf'):
        fig.savefig(FIGURES / f'followup_loss_by_policy.{extension}', dpi=240)
    plt.close(fig)
    payload = {'report_date': '2026-10-28', 'revision_date': '2026-10-30',
               'suite': str(SUITE.relative_to(ROOT)),
               'metric_notes': original['metric_notes'],
               'selection_rule': 'Best epoch is the first highest single-image validation top-1; it is not selected by minimum loss.',
               'display': 'All 30 raw epochs, no smoothing; common loss scale 0–20; dotted baseline validation; dashed selected epoch.',
               'curves': {'baseline': baseline, **curves}, 'best_epochs': best_epochs,
               'curve_sources': sources,
               'final_metrics': {key: rows[-1] for key, rows in curves.items()},
               'output_sha256': {p.name: digest(p) for p in FIGURES.iterdir()
                                 if p.stem in ('rot45_loss', 'rot60_loss', 'followup_loss_by_policy') and p.suffix in ('.png', '.pdf')}}
    (FIGURES / 'followup_loss_data.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('Verified 30 epochs against stdout for each run; plotted ±45°/±60° loss curves.')
    print('Selected epochs:', best_epochs)


if __name__ == '__main__':
    main()
