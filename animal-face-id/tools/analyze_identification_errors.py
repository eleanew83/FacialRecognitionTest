"""Reproduce the 31 October 2026 error analysis from saved rot45/mean outputs.

Run with the project Python environment (Python 3.9+):
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python animal-face-id/tools/analyze_identification_errors.py
Requires numpy, matplotlib and Pillow. No inference, training, or dataset edits.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import textwrap
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-10-31'
SUITE = ROOT / 'artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV'
EVAL = SUITE / 'eval'
OUT = ROOT / 'reports/figures' / ('error_analysis_' + DATE)
REPORT = ROOT / 'reports/markdown' / ('ERROR_ANALYSIS_' + DATE + '.md')
PDF = ROOT / 'reports/pdf' / ('Macaque_ReID_Error_Analysis_' + DATE + '.pdf')
CROPS = ROOT.parent / 'yolo_detection/yolo_detection_code/output/macaque_crops'
COLORS = {'val': '#0072B2', 'test': '#D55E00'}
TITLES = {'val': 'Validation', 'test': 'Test'}
plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_file_record(path, page, example_id, split, category, role, manifest, source_root):
    item = manifest[path]
    source = source_root / item['master_source_path'] if item.get('master_source_path') else None
    return {'pdf_page': page, 'example_id': example_id, 'example_split': split,
            'category': category, 'role': role, 'labelled_id': item['id'],
            'filename': Path(path).name, 'crop_relative_path': path,
            'crop_absolute_path': str(CROPS / path),
            'source_photo_filename': source.name if source else '',
            'source_photo_path': str(source) if source else ''}


def one(pattern):
    matches = list(EVAL.glob(pattern))
    assert len(matches) == 1, (pattern, matches)
    return matches[0]


def relcrop(p):
    return str(p).split('/macaque_crops/', 1)[1]


def pct(x):
    return f'{100*x:.2f}%'


def sex(x):
    return {'males': 'male', 'females': 'female'}[x]


def image(p):
    with Image.open(p) as im:
        return ImageOps.contain(im.convert('RGB'), (224, 224), method=Image.Resampling.LANCZOS)


def thumbnail(p):
    f = io.BytesIO()
    image(p).save(f, format='JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(f.getvalue()).decode('ascii')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows])


def compare(rows, old, field):
    assert set(old) == {r['image_path'] for r in rows}
    counts = Counter()
    for r in rows:
        p = old[r['image_path']]
        assert p['true_id'] == r['true_id']
        before = p['true_id'] == p['pred_id']
        r[field] = int(before)
        category = ('both_correct' if r['correct'] else 'introduced') if before else ('fixed' if r['correct'] else 'both_wrong')
        counts[category] += 1
    assert sum(counts.values()) == len(rows)
    return dict(counts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {}
    def record(p):
        evidence[str(p.relative_to(ROOT))] = sha(p)
    manifest_path = ROOT / 'data/macaque_faces/splits.json'
    registry_path = ROOT / 'data/macaque_faces/identity_registry.json'
    inputs = json.loads((SUITE / 'data_inputs.json').read_text())
    for key, p in [('splits_path', manifest_path), ('identity_registry', registry_path)]:
        assert sha(p) == inputs[key]['sha256'], f'Input changed since evaluation: {p}'
        record(p)
    for p in [SUITE / 'data_inputs.json', EVAL / 'selection.json', Path(__file__), Path(__file__).with_suffix('.html')]:
        record(p)
    selected = json.loads((EVAL / 'selection.json').read_text())
    assert selected['selection_split'] == 'val' and selected['selected']['tta_mode'] == 'mean'
    assert '_rot45_' in selected['selected']['run']
    manifest = json.loads(manifest_path.read_text())
    reg = json.loads(registry_path.read_text())
    identities = reg['identities']
    source_root = Path(reg['source_root'])
    index = {r['path']: r for entries in manifest.values() for r in entries}
    train = Counter(r['id'] for r in manifest['train'])
    refs = {}
    for name in sorted(identities):
        candidates = sorted(r['path'] for r in manifest['train'] if r['id'] == name)
        refs[name] = candidates[len(candidates)//2]
    data, summaries, matrices = {}, {}, {}
    pcs, pairs, slices, dimensions, confidence = [], [], [], [], []
    changes = {}
    classes = None
    verified = 0
    for split in ['val', 'test']:
        print(f'Checking {split} probabilities, metrics and crop hashes...', flush=True)
        summary_path = one(f'*rot45*mean_{split}*metrics_summary.json')
        summary = json.loads(summary_path.read_text())
        paths_by_kind = {kind: summary_path.with_name(summary_path.name.replace('_metrics_summary.json', suffix))
                         for kind, suffix in [('csv', '_hard_examples.csv'), ('npz', '_view_probabilities.npz'),
                                              ('cm', '_confusion_matrix.npy'), ('pc', '_per_class_metrics.csv')]}
        for p in [summary_path] + list(paths_by_kind.values()):
            record(p)
        recorded = {relcrop(r['image_path']): r for r in read_csv(paths_by_kind['csv'])}
        with np.load(paths_by_kind['npz'], allow_pickle=False) as z:
            names = z['class_names'].astype(str).tolist()
            labels = z['labels'].copy()
            paths = [relcrop(p) for p in z['paths']]
            probs = z['pooled_probabilities'].copy()
            assert z['angles'].tolist() == [-30, -20, -10, 0, 10, 20, 30]
            assert np.allclose(probs, z['probabilities'].mean(axis=1), atol=2e-6)
        classes = names if classes is None else classes
        assert classes == names == sorted(identities)
        assert set(paths) == set(recorded) == {r['path'] for r in manifest[split]}
        assert len(paths) == len(set(paths)) == summary['num_samples']
        assert np.isfinite(probs).all() and (probs >= 0).all()
        assert np.allclose(probs.sum(axis=1), 1, atol=2e-6)
        predicted = probs.argmax(axis=1)
        order = np.argsort(-probs, axis=1, kind='stable')
        # Competition rank: tied, underflowed probabilities do not receive arbitrary ranks.
        ranks = 1 + (probs > probs[np.arange(len(labels)), labels, None]).sum(axis=1)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (labels, predicted), 1)
        assert np.array_equal(cm, np.load(paths_by_kind['cm'], allow_pickle=False))
        for k in [1, 3, 5]:
            assert abs(float(np.any(order[:, :k] == labels[:, None], axis=1).mean()) - summary['topk'][str(k)]) < 1e-12
        rows = []
        for i, rel in enumerate(paths):
            item, old = index[rel], recorded[rel]
            true, pred = classes[labels[i]], classes[predicted[i]]
            assert item['id'] == old['true_id'] == true and old['pred_id'] == pred
            assert int(old['is_correct']) == int(true == pred)
            assert abs(float(old['top1_prob']) - float(probs[i].max())) < 2e-5
            assert sha(CROPS / rel) == item['sha256'], f'Crop changed: {rel}'
            verified += 1
            with Image.open(CROPS / rel) as im:
                width, height = im.size
            ti, pi = identities[true], identities[pred]
            source = source_root / item['master_source_path'] if item.get('master_source_path') else None
            rows.append({'split': split, 'image_path': rel, 'source_photo': str(source) if source else '',
                         'source_photo_exists': bool(source and source.exists()),
                         'true_id': true, 'pred_id': pred, 'correct': int(true == pred),
                         'confidence': float(probs[i].max()), 'true_probability': float(probs[i, labels[i]]),
                         'true_rank': int(ranks[i]), 'top3_hit': int(labels[i] in order[i, :3]),
                         'top5_hit': int(labels[i] in order[i, :5]),
                         'top2_margin': float(probs[i, order[i, 0]] - probs[i, order[i, 1]]),
                         'top3_ids': '; '.join(classes[j] for j in order[i, :3]),
                         'true_sex': sex(ti['gender']), 'pred_sex': sex(pi['gender']),
                         'true_group': ti['current_group'], 'pred_group': pi['current_group'],
                         'same_sex': int(ti['gender'] == pi['gender']),
                         'same_group': int(ti['current_group'] == pi['current_group']),
                         'train_images': train[true], 'width': width, 'height': height,
                         'min_dimension': min(width, height), 'single_correct': '', 'baseline_correct': ''})
        data[split], matrices[split] = rows, cm
        assert abs(sum(r['correct'] for r in rows)/len(rows) - summary['top1']) < 1e-12
        saved_pc = {r['class_id']: r for r in read_csv(paths_by_kind['pc'])}
        for i, name in enumerate(classes):
            n, tp, fp = int(cm[i].sum()), int(cm[i, i]), int(cm[:, i].sum() - cm[i, i])
            recall, precision = tp/n if n else 0, tp/(tp+fp) if tp+fp else 0
            f1 = 2*recall*precision/(recall+precision) if recall+precision else 0
            for k, v in [('recall', recall), ('precision', precision), ('f1', f1)]:
                assert abs(float(saved_pc[name][k]) - v) < 1e-12
            pcs.append({'split': split, 'id': name, 'sex': sex(identities[name]['gender']),
                        'group': identities[name]['current_group'], 'train_images': train[name],
                        'support': n, 'correct': tp, 'errors': n-tp, 'false_positives': fp,
                        'recall': recall, 'precision': precision, 'f1': f1})
        errors = [r for r in rows if not r['correct']]
        directed = Counter((r['true_id'], r['pred_id']) for r in errors)
        unordered = Counter(tuple(sorted((r['true_id'], r['pred_id']))) for r in errors)
        for (a, b), count in sorted(unordered.items(), key=lambda x: (-x[1], x[0])):
            pairs.append({'split': split, 'id_a': a, 'id_b': b, 'errors_both_directions': count,
                          'a_to_b': directed[(a,b)], 'b_to_a': directed[(b,a)],
                          'same_sex': int(identities[a]['gender'] == identities[b]['gender']),
                          'same_group': int(identities[a]['current_group'] == identities[b]['current_group'])})
        for attr in ['true_sex', 'true_group']:
            for value in sorted({r[attr] for r in rows}):
                subset = [r for r in rows if r[attr] == value]
                present = {r['true_id'] for r in subset}
                psub = [r for r in pcs if r['split'] == split and r['id'] in present]
                slices.append({'split': split, 'attribute': attr, 'value': value, 'identities': len(psub),
                               'images': len(subset), 'correct': sum(r['correct'] for r in subset),
                               'accuracy': sum(r['correct'] for r in subset)/len(subset),
                               'macro_recall': float(np.mean([r['recall'] for r in psub]))})
        for label, low, high in [('<64 px', 0, 64), ('64-127 px', 64, 128), ('128-223 px', 128, 224), ('>=224 px', 224, math.inf)]:
            subset = [r for r in rows if low <= r['min_dimension'] < high]
            dimensions.append({'split': split, 'size_bin': label, 'images': len(subset),
                               'errors': sum(not r['correct'] for r in subset),
                               'accuracy': sum(r['correct'] for r in subset)/len(subset) if subset else None})
        for low, high in [(0, .5), (.5, .8), (.8, .9), (.9, 1.000001)]:
            subset = [r for r in rows if low <= r['confidence'] < high]
            confidence.append({'split': split, 'score_bin': f'{low:.0%}-{min(high,1):.0%}', 'images': len(subset),
                               'errors': sum(not r['correct'] for r in subset),
                               'accuracy': sum(r['correct'] for r in subset)/len(subset) if subset else None})
        small = [r for r in rows if r['min_dimension'] < 128]
        summary.update({'small_crop_images': len(small), 'small_crop_correct': sum(r['correct'] for r in small),
                        'small_crop_accuracy': sum(r['correct'] for r in small)/len(small) if small else None,
                        'correct': sum(r['correct'] for r in rows), 'errors': len(errors),
                        'cross_sex_errors': sum(not r['same_sex'] for r in errors),
                        'same_sex_errors': sum(r['same_sex'] for r in errors),
                        'cross_group_errors': sum(not r['same_group'] for r in errors),
                        'errors_true_in_top3': sum(r['top3_hit'] for r in errors),
                        'errors_true_in_top5': sum(r['top5_hit'] for r in errors),
                        'errors_confidence_ge_90': sum(r['confidence'] >= .9 for r in errors),
                        'distinct_directed_pairs': len(directed), 'max_directed_pair_count': max(directed.values()),
                        'identities_with_errors': sum(r['errors'] > 0 for r in pcs if r['split'] == split),
                        'identities_below_50': sum(r['recall'] < .5 for r in pcs if r['split'] == split)})
        summaries[split] = summary
        p = one('*rot45*none_val*hard_examples.csv') if split == 'val' else one('*rot45*_canonical_20261014_none_hard_examples.csv')
        record(p)
        changes[split + '_single_to_mean'] = compare(rows, {relcrop(r['image_path']): r for r in read_csv(p)}, 'single_correct')
    p = next((ROOT/'artifacts/final_eval').glob('*aug3_canonical_20261014_hard_examples.csv'))
    record(p)
    changes['test_baseline_to_selected'] = compare(data['test'], {relcrop(r['image_path']): r for r in read_csv(p)}, 'baseline_correct')
    assert changes['test_baseline_to_selected']['fixed'] == 72
    assert changes['test_baseline_to_selected']['introduced'] == 33
    predictions = data['val'] + data['test']
    errors = [r for r in predictions if not r['correct']]
    for rel in refs.values():
        assert sha(CROPS/rel) == index[rel]['sha256'], f'Training reference changed: {rel}'
    print(f'Verified {verified} evaluation crops and {len(refs)} training references.', flush=True)
    for name, rows in [('per_individual_metrics.csv', pcs), ('all_predictions.csv', predictions),
                       ('confused_pairs.csv', pairs), ('metadata_slices.csv', slices),
                       ('resolution_slices.csv', dimensions), ('confidence_slices.csv', confidence)]:
        write_csv(name, rows)
    write_csv('error_review_queue.csv', [dict(r, crop_correct='', viewpoint='', occlusion='', crop_truncation='',
                                            blur_or_low_detail='', suspected_label_issue='', reviewer_notes='')
              for r in sorted(errors, key=lambda r: (r['split'] != 'val', -r['confidence'], r['image_path']))])
    note_path = OUT / 'visual_observations.json'
    notes = json.loads(note_path.read_text()) if note_path.exists() else []
    refnote_path = OUT / 'training_reference_observations.json'
    refnotes = json.loads(refnote_path.read_text()) if refnote_path.exists() else []
    if refnote_path.exists():
        record(refnote_path)
    assert all(n['image_path'] == refs[n['id']] for n in refnotes)
    if note_path.exists():
        record(note_path)
    assert all(n['image_path'] in {r['image_path'] for r in errors} for n in notes)
    analysis = {'analysis_date': DATE, 'selected_model': selected['selected'], 'summaries': summaries,
                'transitions': changes, 'source_sha256': evidence,
                'image_hash_checks': {'evaluation_images': verified, 'training_references': len(refs), 'all_match': True},
                'reference_selection': 'Middle training path in alphabetical order; illustrative, not nearest neighbour.',
                'visual_observations': notes, 'training_reference_observations': refnotes}
    (OUT/'analysis_data.json').write_text(json.dumps(analysis, indent=2)+'\n')
    pdf_images = []
    with PdfPages(PDF) as pdf:
        summary_page(pdf, summaries, changes, notes)
        figures(pdf, summaries, data, pcs, pairs, slices, dimensions)
        observation_pages(pdf, notes, refnotes, pdf_images, index, source_root)
        example_sheets(pdf, data, refs, pdf_images, index, source_root)
    write_csv('pdf_image_filenames.csv', pdf_images)
    write_csv('all_error_filenames.csv', [dict(split=r['split'], labelled_id=r['true_id'],
              predicted_id=r['pred_id'], filename=Path(r['image_path']).name,
              crop_relative_path=r['image_path'], crop_absolute_path=str(CROPS/r['image_path']),
              source_photo_filename=Path(r['source_photo']).name, source_photo_path=r['source_photo'])
              for r in errors])
    markdown_report(summaries, changes, pcs, pairs, slices, confidence, dimensions, notes, refnotes)
    html_report(analysis, pcs, errors, refs)
    print(json.dumps({'markdown': str(REPORT), 'pdf': str(PDF), 'html': str(OUT/'error_analysis.html'),
                      'errors': {s: summaries[s]['errors'] for s in summaries}, 'visual_notes': len(notes)}, indent=2))


def savefig(fig, stem, pdf):
    fig.savefig(OUT/(stem+'.png'), bbox_inches='tight')
    fig.savefig(OUT/(stem+'.pdf'), bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def summary_page(pdf, summaries, changes, notes):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.06, .93, 'Macaque identification: error analysis', fontsize=22, weight='bold')
    fig.text(.06, .88, '31 October 2026 | Selected model: training ±45°, seven-angle probability mean', fontsize=12)
    lines = []
    for split in ['val', 'test']:
        s = summaries[split]
        lines += [f"{TITLES[split]}: {s['correct']}/{s['num_samples']} correct ({pct(s['top1'])}); {s['errors']} errors.",
                  f"    {s['same_sex_errors']} same-sex / {s['cross_sex_errors']} cross-sex errors; {s['errors_confidence_ge_90']} wrong with score ≥90%.",
                  f"    True identity remains in the top 5 for {s['errors_true_in_top5']}/{s['errors']} errors.",
                  f"    Original crop shorter side <128px: {s['small_crop_correct']}/{s['small_crop_images']} correct ({pct(s['small_crop_accuracy'])}).", '']
    change = changes['test_baseline_to_selected']
    lines += [f"Versus original baseline on matched test images: {change['fixed']} errors fixed; {change['introduced']} introduced.",
              'Descriptive analysis of one recorded fit per policy. No new model was trained.', '',
              'Read per-individual recall with its denominator and training support.',
              'Confidence is a model score, not calibrated correctness probability.',
              'Group means current registry membership, not necessarily group at capture.', '',
              'PDF photos: 32 selected error queries (16 validation, 16 test), plus training references.',
              'An error means disagreement with a dataset label; labels are not independently verified.',
              'Every photo has its crop path; pdf_image_filenames.csv also lists original source photos.',
              f'Visual inspection notes: {len(notes)} images; causes remain hypotheses.',
              'Use validation for development; test-driven redesign needs a fresh final holdout.', '',
              'Companion HTML: searchable table for all 149 individuals and every error.',
              'Companion CSVs: all predictions, metadata slices, pairs and a blank review queue.']
    fig.text(.07, .81, '\n'.join(lines), va='top', fontsize=12, linespacing=1.45)
    pdf.savefig(fig)
    plt.close(fig)


def figures(pdf, summaries, data, pcs, pairs, slices, dimensions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, split in enumerate(['val', 'test']):
        subset = sorted([r for r in pcs if r['split'] == split], key=lambda r: (-r['errors'], r['id']))[:15][::-1]
        ax = axes[0, col]
        ax.barh([r['id'] for r in subset], [r['errors'] for r in subset], color=COLORS[split])
        for i, r in enumerate(subset):
            ax.text(r['errors']+.05, i, f"{r['errors']}/{r['support']} wrong", va='center', fontsize=8)
        ax.set_xlim(0, max(r['errors'] for r in subset)+3.5)
        ax.set_title(TITLES[split]+': individuals with most errors')
        ax.set_xlabel('Errors (labels show errors / evaluation images)')
        ax = axes[1, col]
        for correct, color, label in [(1, '#009E73', 'Correct'), (0, '#CC3311', 'Incorrect')]:
            values = [r['confidence'] for r in data[split] if r['correct'] == correct]
            ax.hist(values, bins=np.linspace(0, 1, 21), alpha=.6, color=color, label=f'{label} (n={len(values)})')
        ax.set_title(TITLES[split]+': confidence of correct/incorrect predictions')
        ax.set_xlabel('Maximum mean class probability (uncalibrated)')
        ax.set_ylabel('Images')
        ax.legend()
    fig.suptitle('Where errors concentrate, and whether the model knows it is wrong', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96))
    savefig(fig, 'error_overview', pdf)
    val = {r['id']: r for r in pcs if r['split'] == 'val'}
    test = {r['id']: r for r in pcs if r['split'] == 'test'}
    order = sorted(val, key=lambda n: (val[n]['recall'], -val[n]['support'], n))
    for page in range(3):
        names = order[page*50:(page+1)*50]
        y = np.arange(len(names))
        fig, axes = plt.subplots(1, 3, figsize=(18, 15), gridspec_kw={'width_ratios': [1.5, 1.5, 1]})
        for col, (split, lookup) in enumerate([('val', val), ('test', test)]):
            ax = axes[col]
            ax.barh(y, [lookup[n]['recall']*100 for n in names], color=COLORS[split], alpha=.85)
            for i, name in enumerate(names):
                r = lookup[name]
                ax.text(102, i, f"{r['correct']}/{r['support']}", va='center', fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(names if col == 0 else [])
            ax.set_xlim(0, 123)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(len(names)-.4, -.6)
            ax.set_title(TITLES[split]+' recall; correct / total')
            ax.set_xlabel('Recall (%) | dotted line: overall image accuracy')
            ax.axvline(summaries[split]['top1']*100, color='black', linestyle=':', linewidth=1, label='Overall image accuracy')
        ax = axes[2]
        ax.barh(y, [val[n]['train_images'] for n in names], color='#777777')
        for i, name in enumerate(names):
            ax.text(val[name]['train_images']+1, i, str(val[name]['train_images']), va='center', fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(len(names)-.4, -.6)
        ax.set_xlim(0, max(val[n]['train_images'] for n in names)*1.18)
        ax.set_title('Training images')
        ax.set_xlabel('Source crops; before online augmentation')
        fig.suptitle(f'All 149 individuals: panel {page+1}/3 | ordered by validation recall\nSame individual across panels; small denominators make rates unstable.', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .95))
        savefig(fig, f'per_individual_recall_{page+1}', pdf)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, split in zip(axes, ['val', 'test']):
        subset = [r for r in pairs if r['split'] == split][:15][::-1]
        values = [r['errors_both_directions'] for r in subset]
        ax.barh([r['id_a']+' / '+r['id_b'] for r in subset], values, color=COLORS[split])
        for i, r in enumerate(subset):
            ax.text(values[i]+.03, i, f"{r['a_to_b']} / {r['b_to_a']}", va='center', fontsize=9)
        ax.set_xlim(0, max(values)+1)
        ax.set_title(TITLES[split])
        ax.set_xlabel('Errors in both directions; labels show A→B / B→A')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle('Most frequent confused pairs (ties alphabetical; counts are small)', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .93))
    savefig(fig, 'confused_pairs', pdf)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, split in zip(axes, ['val', 'test']):
        subset = [r for r in pcs if r['split'] == split]
        ax.scatter([r['train_images'] for r in subset], [r['recall']*100 for r in subset],
                   s=[15+5*r['support'] for r in subset], color=COLORS[split], alpha=.55, edgecolor='white')
        for r in sorted(subset, key=lambda r: (r['recall'], -r['support'], r['id']))[:5]:
            ax.annotate(r['id'], (r['train_images'], r['recall']*100), xytext=(3, 4), textcoords='offset points', fontsize=8)
        ax.set_xscale('log')
        ax.set_ylim(-5, 105)
        ax.set_title(TITLES[split])
        ax.set_xlabel('Training source crops per individual (log scale)')
        ax.set_ylabel('Recall (%)')
    fig.suptitle('Training support versus recall | marker area reflects evaluation support\nDescriptive association, not evidence of a causal improvement from collecting images.')
    fig.tight_layout(rect=(0, 0, 1, .88))
    savefig(fig, 'training_support_vs_recall', pdf)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for split, offset in [('val', -.18), ('test', .18)]:
        subset = [r for r in slices if r['split'] == split and r['attribute'] == 'true_sex']
        for i, r in enumerate(subset):
            axes[0].bar(i+offset, 100*r['accuracy'], width=.35, color=COLORS[split], label=TITLES[split] if i == 0 else None)
            axes[0].text(i+offset, 100*r['accuracy']+1, f"n={r['images']}", ha='center', fontsize=8)
        s = summaries[split]
        axes[1].bar(offset, s['same_sex_errors'], width=.35, color=COLORS[split])
        axes[1].bar(1+offset, s['cross_sex_errors'], width=.35, color=COLORS[split])
        for i, r in enumerate([r for r in dimensions if r['split'] == split]):
            if r['accuracy'] is not None:
                axes[2].bar(i+offset, 100*r['accuracy'], width=.35, color=COLORS[split])
                axes[2].text(i+offset, 100*r['accuracy']+1, f"n={r['images']}", ha='center', fontsize=7, rotation=45)
    axes[0].set_xticks([0, 1], ['Female', 'Male'])
    axes[0].set_title('Accuracy by recorded sex')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].legend()
    axes[1].set_xticks([0, 1], ['Same sex', 'Cross sex'])
    axes[1].set_title('Identification error composition')
    axes[1].set_ylabel('Raw errors (not chance-adjusted)')
    axes[2].set_xticks(range(4), ['<64', '64–127', '128–223', '≥224'])
    axes[2].set_title('Accuracy by original crop size')
    axes[2].set_xlabel('Shorter side before resizing (pixels)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim(0, 112)
    axes[2].set_yticks([0, 20, 40, 60, 80, 100])
    fig.suptitle('Descriptive slices: populations and difficulty differ; these are not causal effects.')
    fig.tight_layout(rect=(0, 0, 1, .92))
    savefig(fig, 'sex_and_resolution', pdf)


def observation_pages(pdf, notes, refnotes, pdf_images, manifest, source_root):
    for split in ['val', 'test']:
        subset = [n for n in notes if n['split'] == split]
        if not subset:
            continue
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, TITLES[split]+': visual observations', fontsize=20, weight='bold')
        fig.text(.06, .89, 'Selected highest-confidence errors; these are observations, not established causes.\n'
                 'No full crop audit or identity adjudication. References and source paths are in the companion files.', fontsize=11)
        y = .81
        for n in subset:
            label = n['true_id']+' → '+n['pred_id']+': '
            wrapped = textwrap.fill(label+n['observation'], width=115)
            fig.text(.07, y, wrapped, va='top', fontsize=10, linespacing=1.25)
            y -= .021*(wrapped.count('\n')+1)+.019
        pdf.savefig(fig)
        plt.close(fig)


    if refnotes:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .94, 'A training crop to prioritise for manual review', fontsize=20, weight='bold')
        n = refnotes[0]
        fig.text(.06, .85, textwrap.fill(n['id']+': '+n['observation'], width=65), va='top', fontsize=12, linespacing=1.5)
        fig.text(.06, .62, textwrap.fill(n['image_path'], width=65), va='top', fontsize=10)
        ax = fig.add_axes([.6, .08, .33, .78])
        ax.imshow(image(CROPS/n['image_path']))
        ax.axis('off')
        fig.text(.06, .14, 'An incidental finding in an illustrative reference.\nThis is not an estimate of the dataset-wide crop error rate.', fontsize=11)
        pdf.savefig(fig)
        pdf_images.append(image_file_record(n['image_path'], pdf.get_pagecount(), 'F01',
                          'train', 'training_crop_flag', 'flagged_training_reference', manifest, source_root))
        plt.close(fig)


def example_sheets(pdf, data, refs, pdf_images, manifest, source_root):
    selection = []
    for split in ['val', 'test']:
        errors = [r for r in data[split] if not r['correct']]
        high = sorted(errors, key=lambda r: (-r['confidence'], r['image_path']))[:8]
        used = {r['image_path'] for r in high}
        ambiguous = sorted([r for r in errors if r['image_path'] not in used], key=lambda r: (r['top2_margin'], r['image_path']))[:8]
        for category, examples in [('high_confidence', high), ('ambiguous', ambiguous)]:
            title = 'Highest-confidence errors' if category == 'high_confidence' else 'Smallest top-two probability margins'
            for start in range(0, len(examples), 2):
                page = start // 2 + 1
                pdf_page = pdf.get_pagecount() + 1
                fig = plt.figure(figsize=(11.7, 8.3))
                fig.text(.04, .96, f'{TITLES[split]}: {title} — page {page}/4', fontsize=16, weight='bold')
                fig.text(.04, .92, 'Selected disagreements with dataset labels; neither query nor reference identities are independently verified.', fontsize=10)
                fig.text(.04, .89, 'Captions: split / individual / complete filename. Training references are illustrative, not evaluated errors.', fontsize=9)
                for row_on_page, r in enumerate(examples[start:start+2]):
                    rank = start + row_on_page + 1
                    example_id = f"{split[0].upper()}-{'H' if category == 'high_confidence' else 'A'}{rank:02d}"
                    paths = [r['image_path'], refs[r['true_id']], refs[r['pred_id']]]
                    titles = [f"{example_id}  QUERY label: {r['true_id']} → {r['pred_id']}\nscore={r['confidence']:.3f}; true rank={r['true_rank']}; {r['width']}×{r['height']}",
                              'Labelled-ID TRAIN reference\n'+r['true_id'], 'Predicted-ID TRAIN reference\n'+r['pred_id']]
                    roles = ['error_query', 'labelled_id_training_reference', 'predicted_id_training_reference']
                    top = .84 - .415*row_on_page
                    for col, path in enumerate(paths):
                        left = .045 + .325*col
                        fig.text(left, top, titles[col], fontsize=9, va='top')
                        ax = fig.add_axes([left, top-.265, .28, .225])
                        ax.imshow(image(CROPS/path))
                        ax.axis('off')
                        caption = '\n'.join(textwrap.wrap(path, width=48, replace_whitespace=False,
                                            drop_whitespace=False, break_on_hyphens=False))
                        fig.text(left, top-.278, caption, fontsize=7.5, family='monospace', va='top',
                                 linespacing=1.15, parse_math=False)
                        pdf_images.append(image_file_record(path, pdf_page, example_id, split,
                                          category, roles[col], manifest, source_root))
                    selection.append({'split': split, 'category': category, 'row': rank,
                                      'image_path': paths[0], 'true_reference': paths[1], 'pred_reference': paths[2],
                                      'example_id': example_id, 'pdf_page': pdf_page, 'page_within_category': page,
                                      'row_on_page': row_on_page+1})
                fig.text(.04, .02, f'PDF page {pdf_page} | Full crop and original-photo filenames: pdf_image_filenames.csv', fontsize=8)
                suffix = '' if page == 1 else f'_page_{page}'
                savefig(fig, f'examples_{split}_{category}{suffix}', pdf)
    write_csv('example_manifest.csv', selection)


def markdown_report(summaries, changes, pcs, pairs, slices, confidence, dimensions, notes, refnotes):
    fd = '../figures/error_analysis_'+DATE
    v, t = summaries['val'], summaries['test']
    lines = ['# Macaque identification error analysis — '+DATE, '',
             '**Selected model: ±45° training + seven-angle mean**, chosen using validation. Saved outputs only; no model or dataset changes.', '',
             f'[PDF report](../pdf/{PDF.name}) · [Searchable report and all error images]({fd}/error_analysis.html) · [Per-individual CSV]({fd}/per_individual_metrics.csv) · [Review queue]({fd}/error_review_queue.csv)', '',
             '**Photo coverage:** 32 selected error queries (16 validation, 16 test), with two illustrative training references per query; one flagged training crop is shown separately. These are not all 437 errors. Labels have not been independently verified.', '',
             f'[Every displayed photo: filenames, PDF pages and original sources]({fd}/pdf_image_filenames.csv) · [All 437 error filenames]({fd}/all_error_filenames.csv)', '',
             '## 1. Measured findings', '',
             table(['Measure', 'Validation', 'Test'], [
                 ['Correct / images', f"{v['correct']}/{v['num_samples']}", f"{t['correct']}/{t['num_samples']}"],
                 ['Top-1 accuracy', pct(v['top1']), pct(t['top1'])],
                 ['Macro recall', pct(v['recall_macro']), pct(t['recall_macro'])],
                 ['Macro F1', pct(v['f1_macro']), pct(t['f1_macro'])],
                 ['Errors', v['errors'], t['errors']],
                 ['Individuals with errors', v['identities_with_errors'], t['identities_with_errors']],
                 ['Individuals with recall below 50%', v['identities_below_50'], t['identities_below_50']],
                 ['Same-sex / cross-sex errors', f"{v['same_sex_errors']} / {v['cross_sex_errors']}", f"{t['same_sex_errors']} / {t['cross_sex_errors']}"],
                 ['Wrong with model score ≥90%', v['errors_confidence_ge_90'], t['errors_confidence_ge_90']],
                 ['Original crops <128px: correct / images', f"{v['small_crop_correct']}/{v['small_crop_images']}", f"{t['small_crop_correct']}/{t['small_crop_images']}"],
                 ['Errors with true ID still in top 5', f"{v['errors_true_in_top5']}/{v['errors']}", f"{t['errors_true_in_top5']}/{t['errors']}"],
                 ['Distinct directed error pairs', v['distinct_directed_pairs'], t['distinct_directed_pairs']],
                 ['Maximum directed pair count', v['max_directed_pair_count'], t['max_directed_pair_count']]]), '',
             'Most errors are between individuals of the same sex. Counts are not chance-adjusted: there are 97 female and 52 male identities. This does not predict the benefit of sex supervision. High-confidence mistakes also occur. No directed pair occurs more than twice in either split; errors are distributed over many pairs. **Small crops are a measurable concern:** accuracy for original crops under 128px on the shorter side is 47.22% (17/36) on validation and 45.83% (11/24) on test, compared with about 87% for crops at least 224px on both sides. These small samples differ in other ways too, so this is association rather than a controlled resolution effect.', '',
             f'![Error overview]({fd}/error_overview.png)', '', '## 2. Per-individual results', '',
             'Per-class accuracy here means **recall**: correctly identified images / evaluation images of that individual. The CSV also includes precision, F1 and false positives. All 149 individuals appear below in validation-recall order, with test results and training support alongside. The dotted line is overall image accuracy, not macro recall. One error among three images changes recall by 33.3 percentage points.', '']
    for page in range(1, 4):
        lines += [f'![All individuals, panel {page}/3]({fd}/per_individual_recall_{page}.png)', '']
    for split in ['val', 'test']:
        subset = sorted([r for r in pcs if r['split'] == split], key=lambda r: (r['recall'], -r['support'], r['id']))[:15]
        lines += ['### Lowest recall: '+TITLES[split].lower(), '',
                  table(['Individual', 'Correct / images', 'Recall', 'Training crops', 'False positives'],
                        [[r['id'], f"{r['correct']}/{r['support']}", pct(r['recall']), r['train_images'], r['false_positives']] for r in subset]), '']
    va = {r['id']: r for r in pcs if r['split'] == 'val'}
    te = {r['id']: r for r in pcs if r['split'] == 'test'}
    review = [{'id': n, 'train_images': va[n]['train_images'],
               'val_correct': va[n]['correct'], 'val_images': va[n]['support'], 'val_recall': va[n]['recall'],
               'test_correct': te[n]['correct'], 'test_images': te[n]['support'], 'test_recall': te[n]['recall']}
              for n in sorted(va, key=lambda n: (va[n]['recall']+te[n]['recall'], n))
              if va[n]['recall'] < .75 and te[n]['recall'] < .75]
    write_csv('individual_review_priorities.csv', review)
    lines += ['### Individuals below 75% recall in both splits', '',
              'The 75% cutoff is a descriptive review filter, not a statistical test or model-selection rule. '
              'Caro and Thief each have only one validation image. Individuals such as Bijou and Goblin also struggle despite more training images.', '',
              table(['Individual', 'Validation correct / images', 'Test correct / images', 'Training images'],
                    [[r['id'], str(r['val_correct'])+'/'+str(r['val_images']),
                      str(r['test_correct'])+'/'+str(r['test_images']), r['train_images']] for r in review]), '',
              f'[Review-priority CSV]({fd}/individual_review_priorities.csv)', '']
    lines += [f'![Training support]({fd}/training_support_vs_recall.png)', '',
              'Training support is descriptive, not a controlled experiment. Image diversity, pose, crop quality and identity difficulty may differ.', '',
              '## 3. Confused identity pairs', '', f'![Confused pairs]({fd}/confused_pairs.png)', '']
    for split in ['val', 'test']:
        subset = [r for r in pairs if r['split'] == split][:10]
        lines += ['### '+TITLES[split], '', table(['A / B', 'A→B', 'B→A', 'Same sex?', 'Same current group?'],
                  [[r['id_a']+' / '+r['id_b'], r['a_to_b'], r['b_to_a'], 'Yes' if r['same_sex'] else 'No', 'Yes' if r['same_group'] else 'No'] for r in subset]), '']
    lines += ['Ties are alphabetical, not ranked by importance. Group is current registry membership, not necessarily group or location at capture.', '',
              '## 4. Metadata, confidence and original crop size', '', f'![Sex and resolution]({fd}/sex_and_resolution.png)', '',
              table(['Split', 'Attribute', 'Value', 'Correct / images', 'Accuracy', 'Macro recall'],
                    [[TITLES[r['split']], r['attribute'], r['value'], f"{r['correct']}/{r['images']}", pct(r['accuracy']), pct(r['macro_recall'])] for r in slices]), '',
              table(['Split', 'Model score', 'Images', 'Errors', 'Observed accuracy'],
                    [[TITLES[r['split']], r['score_bin'], r['images'], r['errors'], pct(r['accuracy']) if r['accuracy'] is not None else '—'] for r in confidence]), '',
              'Scores are maximum mean class probabilities without calibration; these bins do not establish a safe operational threshold.', '',
              table(['Split', 'Original shorter side', 'Images', 'Errors', 'Accuracy'],
                    [[TITLES[r['split']], r['size_bin'], r['images'], r['errors'], pct(r['accuracy']) if r['accuracy'] is not None else '—'] for r in dimensions]), '',
              'Crop dimensions are measured before resizing. Size is not a blur or crop-correctness label; image populations differ between bins.', '',
              '## 5. Visual observations: selected examples, not prevalence estimates', '',
              'Sixteen pages show 32 selected errors: eight highest-confidence and eight smallest-top-two-margin errors per split. Each query is paired with training references for the labelled and predicted identities. Each displayed crop has a complete filename caption and example ID; the filename CSV maps it to its PDF page and original source photograph. References are the middle file in alphabetical training-path order, not nearest neighbours or proof of similarity. Crops are contained without stretching; no training augmentation is applied.', '']
    if notes:
        for n in notes:
            lines += [f"- **{n['split']} — {n['true_id']} → {n['pred_id']}**: {n['observation']} Source: {n['image_path']}"]
        lines += ['']
    else:
        lines += ['Visual causes have not yet been assigned. Automated outputs do not label pose, blur, occlusion or crop mistakes.', '']
    lines += ['These examples are deliberately selected, without matched correct-image controls. They cannot quantify how common or causal each problem is. Identity labels were not adjudicated. The complete manual YOLO crop audit remains separate.', '']
    for split in ['val', 'test']:
        for category in ['high_confidence', 'ambiguous']:
            for page in range(1, 5):
                suffix = '' if page == 1 else f'_page_{page}'
                lines += [f'![{split} {category}, page {page}/4]({fd}/examples_{split}_{category}{suffix}.png)', '']
    if refnotes:
        lines += ['### A concrete training-crop issue to inspect', '']
        for n in refnotes:
            link = quote(os.path.relpath(CROPS/n['image_path'], REPORT.parent), safe='/')
            lines += [f"- **{n['id']}**: {n['observation']} [Open the crop]({link})."]
        lines += ['', f'[Recorded training-reference observations]({fd}/training_reference_observations.json)', '']
    lines += ['## 6. What changed relative to earlier predictions', '',
              table(['Comparison', 'Errors fixed', 'Errors introduced', 'Both wrong', 'Both correct'],
                    [[label]+[changes[key].get(k,0) for k in ['fixed','introduced','both_wrong','both_correct']]
                     for key,label in [('val_single_to_mean','Validation: single → mean'), ('test_single_to_mean','Test: single → mean'),
                                       ('test_baseline_to_selected','Test: original baseline → selected policy + mean')]]), '',
              'Single→mean uses the same checkpoint and isolates inference averaging. The baseline comparison changes both training policy and inference. All comparisons are matched by image and true identity; one fit per policy.', '',
              '## 7. Review priorities and limits', '',
              '1. Review validation errors for truncation, occlusion, viewpoint, limited detail and possible label issues. Inspect the source photograph before deciding a crop or identity is incorrect.',
              '2. Compare the same conditions among correct predictions before assigning causes. The review CSV contains all errors and blank annotation fields; no guessed cause is pre-filled.',
              '3. Prioritise individuals with poor results across both splits and repeated confident errors, keeping support counts visible. Select experiments using validation.',
              '4. If test-error observations guide redesign, keep this cohort as a historical benchmark and reserve a fresh, source-photo-separated final holdout.', '',
              'No existing pose, age, visibility, blur or crop-quality labels support automatic causal attribution. Sex/group slices cannot establish mechanisms. This analysis does not establish statistical significance or performance on unseen identities; related photographs may be correlated.', '',
              '## 8. Reproduction and source integrity', '',
              'Run from the repository root using the existing project environment: /rds/user/ylj20/hpc-work/venvs/macaque/bin/python animal-face-id/tools/analyze_identification_errors.py', '',
              'Requires NumPy, Matplotlib and Pillow. No GPU, model loading or internet. Checks: current manifest/registry hashes match the saved run; probabilities reproduce predictions, top-k results, per-class metrics and saved confusion matrices; all 2,800 evaluation crop hashes plus 149 illustrative training-reference hashes match the manifest.', '',
              f'[Source hashes and analysis data]({fd}/analysis_data.json) · [All predictions]({fd}/all_predictions.csv) · [All confused pairs]({fd}/confused_pairs.csv) · [Example manifest]({fd}/example_manifest.csv)', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def html_report(analysis, pcs, errors, refs):
    print('Building searchable report with embedded error and reference thumbnails...', flush=True)
    packed = []
    for r in errors:
        row = dict(r)
        row['thumbnail'] = thumbnail(CROPS/r['image_path'])
        row['crop_link'] = quote(os.path.relpath(CROPS/r['image_path'], OUT), safe='/')
        row['source_link'] = quote(os.path.relpath(r['source_photo'], OUT), safe='/') if r['source_photo_exists'] else ''
        packed.append(row)
    payload = json.dumps({'summaries': analysis['summaries'], 'classes': pcs, 'errors': packed,
                          'references': {n: thumbnail(CROPS/p) for n,p in refs.items()},
                          'observations': {n['image_path']:n['observation'] for n in analysis['visual_observations']},
                          'referenceNotes': {n['id']:n['observation'] for n in analysis['training_reference_observations']},
                          'figures': {name: 'data:image/png;base64,' + base64.b64encode((OUT/(name+'.png')).read_bytes()).decode('ascii') for name in ['error_overview', 'confused_pairs']}}).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text()
    (OUT/'error_analysis.html').write_text(template.replace('__DATA__', payload), encoding='utf-8')


if __name__ == '__main__':
    main()
