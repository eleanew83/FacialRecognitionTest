"""Read-only correction planning from saved crop reviews; never edits datasets.

The proposed actions are text-based triage, not automatic correction decisions.
Filename uncertainty markers are candidates for expert review, not proven errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = {
    'unsure_or_not_sure': r'\bunsure\b|\bnot\s+sure\b',
    'alternative_names_or': r'\bor\b',
    'likely': r'\blikely\b',
    'maybe_or_unclear': r'\b(?:uncertain|maybe|possibly|possible|probably|probable|unknown|unidentified|unclear)\b|\?',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def filename_text(record):
    name = Path(record['crop_path'] or record['source_path']).stem
    return re.sub(r'[_-]+', ' ', re.sub(r'_crop\d+$', '', name, flags=re.I))


def markers(record):
    return [name for name, pattern in MARKERS.items() if re.search(pattern, filename_text(record), re.I)]


def removal_instruction(note):
    if re.search(r"\b(?:do not|don't|not)\s+remove\b", note, re.I):
        return False
    return bool(re.search(r'\bremove\b|shouldn.t be used', note, re.I))


def manual_instruction(note):
    return bool(re.search(r'\bmanual(?:ly)?\b|recrop', note, re.I))


def proposed_action(record, annotation, canonical_names):
    note = annotation.get('notes', '')
    status = annotation.get('status', 'unreviewed')
    suggested = annotation.get('suggested_identity', '')
    if removal_instruction(note):
        return 'exclude_from_new_generation'
    if record['record_type'] == 'source_only':
        return 'draw_new_face_box' if manual_instruction(note) else 'inspect_original_for_new_crop'
    if status == 'uncertain_identity':
        return 'expert_identity_review'
    if suggested and suggested not in canonical_names:
        return 'resolve_suggested_identity_then_check_target'
    if status in ('wrong_label', 'wrong_animal') or suggested or re.search(
        r'cropped\s+(?:one is|Richard)|instead of|label/crop swapped', note, re.I
    ):
        return 'verify_target_then_recrop_or_relabel'
    if manual_instruction(note) or re.search(r'cropped (?:more|better)|crop better', note, re.I):
        return 'redraw_face_box'
    if re.search(r'wrong crop|bad crop|crop(?:ped)? (?:the )?face (?:not|proper)|face not cropped|did not crop|needs to inspect original|recheck original|not cropped properly', note, re.I):
        return 'inspect_target_and_redraw_face_box'
    if status == 'ok':
        return 'keep_as_reviewed'
    return 'quality_review_keep_unless_unusable'


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['review_key']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Keep free-text notes and filenames safe when opened as a spreadsheet.
            writer.writerow({k: "'" + str(v) if str(v).startswith(('=', '+', '-', '@', '\t', '\r')) else v
                             for k, v in row.items()})


def summarize(review_dir, output):
    state_bytes = (review_dir / 'review_state.json').read_bytes()
    inventory_bytes = (review_dir / 'inventory.json').read_bytes()
    state, inventory = json.loads(state_bytes), json.loads(inventory_bytes)
    manifest_bytes = Path(inventory['manifest_path']).read_bytes()
    registry_bytes = Path(inventory['registry_path']).read_bytes()
    if state['dataset_id'] != inventory['dataset_id']:
        raise ValueError('Review state and inventory differ.')
    if sha(manifest_bytes) != inventory['manifest_sha256'] or sha(registry_bytes) != inventory['registry_sha256']:
        raise ValueError('The active crop manifest or registry changed since review.')
    if inventory.get('source_manifest_path'):
        if sha(Path(inventory['source_manifest_path']).read_bytes()) != inventory['source_manifest_sha256']:
            raise ValueError('The source manifest changed since review.')
    records = {r['key']: r for r in inventory['records']}
    if not set(state['annotations']) <= set(records):
        raise ValueError('Unknown review keys.')
    if any(a['status'] not in inventory['statuses'] for a in state['annotations'].values()):
        raise ValueError('Unknown review status.')
    registry = json.loads(registry_bytes)
    canonical = set(registry['identities'])
    rows = []
    for record in inventory['records']:
        a = state['annotations'].get(record['key'], {'status': 'unreviewed'})
        suggested = a.get('suggested_identity', '')
        rows.append({
            'review_key': record['key'], 'record_type': record['record_type'],
            'identity': record['identity'], 'split': record['split'], 'review_status': a['status'],
            'notes': a.get('notes', ''), 'suggested_identity': suggested,
            'suggested_identity_is_canonical': not suggested or suggested in canonical,
            'proposed_action': proposed_action(record, a, canonical),
            'uncertainty_markers': ';'.join(markers(record)),
            'crop_path': record['crop_path'], 'source_path': record.get('source_path', ''),
            'crop_absolute': record.get('crop_absolute', ''), 'source_photo': record['source_photo'],
            'displayed_image_sha256': record['sha256'],
            'final_action': '', 'confirmed_target_identity': '',
            'bbox_xmin': '', 'bbox_ymin': '', 'bbox_xmax': '', 'bbox_ymax': '',
            'identity_evidence': '', 'decision_by': '',
        })
    crops = [r for r in rows if r['record_type'] == 'crop']
    originals = [r for r in rows if r['record_type'] == 'source_only']
    uncertain = [r for r in rows if r['uncertainty_markers']]
    uncertain_crops = [r for r in uncertain if r['record_type'] == 'crop']
    exclusions = [r for r in rows if r['proposed_action'] == 'exclude_from_new_generation']
    attention = [r for r in rows if r['review_status'] != 'ok']
    new_crops = [r for r in originals if r['proposed_action'] != 'exclude_from_new_generation']
    target_review = [r for r in rows if r['proposed_action'] in (
        'expert_identity_review', 'resolve_suggested_identity_then_check_target',
        'verify_target_then_recrop_or_relabel')]

    impact = []
    for identity in sorted({r['identity'] for r in uncertain}):
        original = [r for r in crops if r['identity'] == identity]
        candidates = [r for r in uncertain_crops if r['identity'] == identity]
        row = {'identity': identity, 'filename_candidates': len(candidates)}
        for split in ('train', 'val', 'test'):
            before = sum(r['split'] == split for r in original)
            excluded = sum(r['split'] == split for r in candidates)
            row[split + '_before'] = before
            row[split + '_filename_candidates'] = excluded
            row[split + '_remaining_if_all_excluded'] = before - excluded
        impact.append(row)
    summary = {
        'analysis_date': '2026-11-24', 'review_revision': state['revision'],
        'last_review_saved_utc': max(a.get('updated_at', '') for a in state['annotations'].values()),
        'dataset_id': state['dataset_id'], 'review_state_sha256': sha(state_bytes),
        'inventory_sha256': sha(inventory_bytes), 'manifest_sha256': sha(manifest_bytes),
        'source_manifest_sha256': inventory.get('source_manifest_sha256'),
        'review_items': len(rows), 'active_crops': len(crops), 'originals_without_crops': len(originals),
        'reviewed': sum(r['review_status'] != 'unreviewed' for r in rows),
        'review_statuses': dict(Counter(r['review_status'] for r in rows)),
        'crop_statuses': dict(Counter(r['review_status'] for r in crops)),
        'source_only_statuses': dict(Counter(r['review_status'] for r in originals)),
        'proposed_actions': dict(Counter(r['proposed_action'] for r in rows)),
        'explicit_exclusion_instructions': dict(Counter(r['record_type'] for r in exclusions)),
        'new_crop_candidates': len(new_crops),
        'new_crop_candidates_with_manual_instruction': sum(manual_instruction(r['notes']) for r in new_crops),
        'uncertainty_filename_counts': dict(Counter(t for r in uncertain for t in r['uncertainty_markers'].split(';'))),
        'uncertainty_filename_union': len(uncertain), 'uncertainty_active_crops': len(uncertain_crops),
        'uncertainty_originals': len(uncertain) - len(uncertain_crops),
        'uncertainty_active_splits': dict(Counter(r['split'] for r in uncertain_crops)),
        'uncertainty_currently_marked_ok': sum(r['review_status'] == 'ok' for r in uncertain),
        'uncertainty_individuals': len(impact),
        'zero_validation_if_all_filename_candidates_excluded': [r['identity'] for r in impact if r['val_remaining_if_all_excluded'] == 0],
        'noncanonical_suggested_identities': [{'identity': r['identity'], 'suggestion': r['suggested_identity'], 'crop_path': r['crop_path']}
                                             for r in rows if not r['suggested_identity_is_canonical']],
        'limitations': [
            'Filename markers are candidate metadata, not expert identity adjudication.',
            'Proposed actions are text-based triage; no dataset changes have been applied.',
            'Excluding all uncertainty candidates is a sensitivity scenario, not a recommendation.',
            'Original-photo review categories can contradict manual-crop instructions in notes; notes are preserved.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'completed_review_state.json'
    if snapshot.exists() and snapshot.read_bytes() != state_bytes:
        raise ValueError('Output already contains another review snapshot. Choose a new --output.')
    snapshot.write_bytes(state_bytes)
    for name, items in [
        ('all_review_actions.csv', rows), ('attention_queue.csv', attention),
        ('explicit_exclusions.csv', exclusions), ('new_crop_candidates.csv', new_crops),
        ('label_and_target_review.csv', target_review), ('uncertain_filename_candidates.csv', uncertain),
    ]:
        write_csv(output / name, items, list(rows[0]))
    write_csv(output / 'uncertainty_impact_by_individual.csv', impact)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert sha((review_dir / 'review_state.json').read_bytes()) == sha(state_bytes), 'Review changed during planning; rerun on a frozen snapshot.'
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=ROOT / 'reports/crop_review_2026-11-10')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.review_dir / 'correction_plan_2026-11-24'
    summarize(args.review_dir.resolve(), output.resolve())


if __name__ == '__main__':
    main()
