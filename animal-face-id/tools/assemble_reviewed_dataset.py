"""Publish a self-contained dataset from completed manual review workspaces.

Historical crops and review manifests remain unchanged. Original-folder edits
supersede faces tied to those particular files; distinct retained face crops
from the same camera frame are preserved and assigned to one split together.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.datasets.identity_registry import IdentityRegistry
from src.datasets.split_integrity import (
    SPLIT_ORDER, file_fingerprint, fresh_output_directory, source_photo_key,
    validate_manifest,
)
from review_macaque_crops import ReviewStore
from crop_corrections import CorrectionStore
from original_photo_editor import OriginalPhotoStore
from organized_face_editor import OrganizedFaceStore


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def camera_key(path):
    path = Path(path)
    if path.suffix.lower() in {'.cr2', '.tif', '.tiff'}:
        path = path.with_suffix('.jpg')
    return source_photo_key(path)


def photo_paths(root):
    result = []
    for directory, folders, names in os.walk(root):
        folders[:] = [name for name in folders if not name.startswith('.')]
        result.extend(Path(directory) / name for name in names
                      if not name.startswith('.') and Path(name).suffix.lower() in {'.jpg', '.jpeg', '.png'})
    return sorted(result)


def merge_candidates(records, annotations, corrections, changed_paths, original_decisions,
                     organised_faces):
    """Merge only approved pixels, respecting per-file supersession/removal."""
    candidates, exclusions = [], []
    for record in records:
        key = record['key']
        annotation = annotations.get(key, {})
        effect = corrections.get(key, {})
        source = record['source_photo']
        reason = None
        if source in changed_paths:
            reason = 'Original-folder file replaced or retired; consume current Stage 2 faces instead.'
        elif effect.get('decision') == 'exclude':
            reason = 'Explicit exclusion in the completed face-correction editor.'
        elif effect.get('decision') == 'defer' and original_decisions.get(source) == 'edit':
            reason = 'Earlier face was deferred for source repair; consume approved faces from the completed original-photo batch, not the deferred old crop.'
        elif effect.get('decision') == 'defer' and original_decisions.get(source) != 'keep':
            raise ValueError('Unresolved deferred face: ' + key)
        elif effect.get('face_pending'):
            raise ValueError('Face still needs verification: ' + key)
        elif annotation.get('status', 'unreviewed') == 'unreviewed':
            raise ValueError('Unreviewed image: ' + key)
        elif annotation.get('status') != 'ok' and not effect.get('decision'):
            raise ValueError('Flagged record has no applied disposition: ' + key)
        if reason:
            exclusions.append({'review_key': key, 'source': source, 'reason': reason,
                               'historical_crop': record.get('crop_absolute')})
            continue
        common = {'review_key': key, 'camera_key': camera_key(source),
                  'original_path': source, 'split_hint': record['split'],
                  'review_status': annotation['status']}
        if record.get('crop_absolute') and not effect.get('drop_original'):
            candidates.append({**common, 'id': record['identity'],
                               'input_path': record['crop_absolute'], 'sha256': record['sha256'],
                               'origin': 'reviewed_historical_crop', 'priority': 2})
        for face in effect.get('faces', {}).values():
            candidates.append({**common, 'id': face['identity'], 'input_path': face['path'],
                               'sha256': face['sha256'], 'origin': 'earlier_manual_face_correction',
                               'face_provenance': face, 'priority': 1})
    for face in organised_faces:
        candidates.append({'id': face['identity'], 'input_path': face['path'],
                           'sha256': face['sha256'], 'original_path': face['organised_original'],
                           'split_hint': face['split_hint'], 'origin': 'organised_original_face',
                           'priority': 0, 'camera_key': face['camera_key'],
                           'review_key': face['source_group'], 'face_provenance': face,
                           'review_status': 'approved_face_from_organised_original'})
    return candidates, exclusions


def deduplicate_and_group(candidates):
    """Deduplicate exact bytes/RGB, never distinct crops sharing a camera name."""
    parent = list(range(len(candidates)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        parent[find(b)] = find(a)

    content = {}
    for i, candidate in enumerate(candidates):
        for field in ('sha256', 'pixels_sha256'):
            token = (field, candidate[field])
            previous = content.setdefault(token, i)
            if candidates[previous]['id'] != candidate['id']:
                raise ValueError('Exact face content has conflicting identities: '
                                 + candidates[previous]['input_path'] + ' and ' + candidate['input_path'])
            union(previous, i)
    duplicates = defaultdict(list)
    for i, candidate in enumerate(candidates):
        duplicates[find(i)].append(candidate)
    retained, omitted = [], []
    for group in duplicates.values():
        representative = min(group, key=lambda c: (c['priority'],
                             SPLIT_ORDER.get(c['split_hint'], 3), c['input_path']))
        representative = dict(representative)
        representative['camera_aliases'] = sorted({c['camera_key'] for c in group})
        representative['source_content_hashes'] = sorted({h for c in group
                                                        for h in c.get('source_content_hashes', [])})
        retained.append(representative)
        omitted.extend({'input_path': c['input_path'], 'representative': representative['input_path'],
                        'reason': 'Exact bytes or decoded RGB pixels.'} for c in group
                       if c['input_path'] != representative['input_path'])

    parent = list(range(len(retained)))
    tokens = {}
    for i, candidate in enumerate(retained):
        keys = [('camera', k) for k in candidate['camera_aliases']]
        keys += [('source_content', h) for h in candidate['source_content_hashes']]
        for token in keys:
            union(tokens.setdefault(token, i), i)
    components = defaultdict(list)
    for i, candidate in enumerate(retained):
        components[find(i)].append(candidate)
    groups = []
    for group in components.values():
        names = sorted({name for c in group for name in c['camera_aliases']})
        group_id = hashlib.sha256(json.dumps(names).encode()).hexdigest()[:24]
        for c in group:
            c['source_group'] = group_id
        groups.append(sorted(group, key=lambda c: (c['id'], c['input_path'])))
    return sorted(groups, key=lambda g: g[0]['source_group']), omitted


def assign_splits(groups, historical_hints):
    """Retain old membership where possible; resolve shared frames as a unit."""
    assignments, counts, sizes = {}, {s: Counter() for s in SPLIT_ORDER}, {}
    transfers = []
    for group in groups:
        gid = group[0]['source_group']
        hints = {hint for c in group for name in c['camera_aliases']
                 for hint in historical_hints.get(name, []) if hint in SPLIT_ORDER}
        hints |= {c['split_hint'] for c in group if c['split_hint'] in SPLIT_ORDER}
        split = min(hints, key=SPLIT_ORDER.get) if hints else 'train'
        assignments[gid] = split
        sizes[gid] = Counter(c['id'] for c in group)
        counts[split].update(sizes[gid])
    identities = {c['id'] for group in groups for c in group}
    for _ in range(3 * len(identities)):
        missing = [(s, label) for s in SPLIT_ORDER for label in sorted(identities)
                   if not counts[s][label]]
        if not missing:
            break
        target, label = missing[0]
        options = []
        for gid, source in assignments.items():
            members = sizes[gid]
            if source == target or label not in members:
                continue
            if all(counts[source][name] > n for name, n in members.items()):
                options.append((source == 'train', sum(members.values()), gid, source))
        if not options:
            raise ValueError(f'Cannot maintain source-photo grouping and identity coverage: {target}/{label}')
        _, _, gid, source = min(options)
        assignments[gid] = target
        counts[source].subtract(sizes[gid])
        counts[target].update(sizes[gid])
        transfers.append({'source_group': gid, 'from': source, 'to': target,
                          'reason': 'Maintain each identity in all three splits.'})
    else:
        raise ValueError('Could not complete closed-set source-group assignments.')
    for group in groups:
        for c in group:
            c['split'] = assignments[c['source_group']]
    return transfers


def snapshot_copy(source, target, expected=None):
    """Copy independent bytes, check both ends, and reject concurrent edits."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as inp, target.open('xb') as out:
        for block in iter(lambda: inp.read(1024 * 1024), b''):
            digest.update(block)
            out.write(block)
    shutil.copystat(source, target)
    after = source.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(before) != stamp(after):
        raise ValueError('Source changed during copy: ' + str(source))
    value = digest.hexdigest()
    if (expected and value != expected) or sha(target) != value:
        raise ValueError('Copy checksum differs: ' + str(source))
    return value


def prepare(review_dir):
    review = ReviewStore(review_dir)
    earlier = CorrectionStore(review)
    originals = OriginalPhotoStore(review)
    faces = OrganizedFaceStore(review, originals)
    exported = faces.export()
    for row in exported['decisions']:
        if row['decision'] not in {'corrected', 'exclude'}:
            raise ValueError('Incomplete current face crop: ' + json.dumps(row))
    return review, earlier, originals, faces, exported


def assemble(review_dir, output, plan_only=False, workers=4):
    output = Path(output).resolve()
    review, earlier, originals, faces, exported = prepare(Path(review_dir))
    registry = IdentityRegistry(review.inventory['registry_path'])
    inputs = [review.output / name for name in (
        'inventory.json', 'review_state.json', 'corrections/state.json',
        'original_photo_edits/catalog.json', 'original_photo_edits/state.json',
        'original_photo_edits/face_stage/corrections/state.json')]
    inputs += [Path(review.inventory[k]) for k in ('manifest_path', 'registry_path', 'source_manifest_path')]
    input_hashes = {str(p): sha(p) for p in inputs}
    with ExitStack() as locks:
        for store in (earlier, originals, faces):
            locks.enter_context(store.guard())
        if read(review.path) != review.state:
            raise ValueError('Review changed while preparing assembly; retry.')
        original_state, correction_state = originals.read(), earlier.read()
        changed, original_decisions, current_expected, retirements = set(), {}, {}, []
        source_hashes = defaultdict(set)
        for group, record in originals.catalog['records'].items():
            entry = original_state['entries'].get(group, {})
            if not entry.get('history') or not entry.get('draft_applied'):
                raise ValueError('Original-photo decision not complete: ' + record['filename'])
            txn = read(originals.root / 'transactions' / (entry['history'][-1] + '.json'))
            decision = txn['effect']['decision']
            if decision == 'defer' and record['active_files']:
                raise ValueError('Unresolved active original: ' + record['filename'])
            frame = camera_key(record['filename'])
            for v in record['variants']:
                if v['kind'] == 'archive':
                    source_hashes[frame].add(v['sha256'])
            for v in record['active_files']:
                original_decisions[v['path']] = decision
                current_expected[v['path']] = v['sha256']
            for op in txn.get('operations', []):
                if op['before_sha256'] != op['after_sha256']:
                    changed.add(op['path'])
                if op['after_sha256'] is None:
                    current_expected.pop(op['path'], None)
                    retirements.append(op['path'])
                    if Path(op['path']).exists():
                        raise ValueError('Retired original still exists: ' + op['path'])
                else:
                    current_expected[op['path']] = op['after_sha256']
        effects = {}
        for key, entry in correction_state['entries'].items():
            if not entry.get('history') or not entry.get('draft_applied'):
                raise ValueError('Earlier face correction is not applied: ' + key)
            effects[key] = earlier.effects(key, correction_state)
        for face in exported['approved_faces']:
            record = originals.catalog['records'][face['parent_group']]
            face['camera_key'] = camera_key(record['filename'])
        candidates, exclusions = merge_candidates(review.inventory['records'], review.state['annotations'],
            effects, changed, original_decisions, exported['approved_faces'])
        source_manifest = read(review.inventory['source_manifest_path'])
        source_records = {r['path']: r for rows in source_manifest.values() for r in rows}
        old_manifest = read(review.inventory['manifest_path'])
        old_records = {r['path']: r for rows in old_manifest.values() for r in rows}
        historical_hints = defaultdict(set)
        for r in review.inventory['records']:
            frame = camera_key(r['source_photo'])
            historical_hints[frame].add(r['split'])
            crop = old_records.get(r['crop_path'], {})
            original = source_records.get(r.get('source_path') or crop.get('source_path'), {})
            if original.get('sha256'):
                source_hashes[frame].add(original['sha256'])
        print(f'Fingerprinting {len(candidates)} candidate faces...', flush=True)

        def fingerprint(candidate):
            result = file_fingerprint(candidate['input_path'])
            if result['sha256'] != candidate['sha256']:
                raise ValueError('Face changed after review: ' + candidate['input_path'])
            candidate.update(result)
            candidate['source_content_hashes'] = sorted(source_hashes[candidate['camera_key']])
            if candidate['id'] not in registry.identities:
                raise ValueError('Noncanonical identity: ' + candidate['id'])
            return candidate

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fingerprinted = []
            for n, candidate in enumerate(pool.map(fingerprint, candidates), 1):
                fingerprinted.append(candidate)
                if n % 1000 == 0:
                    print(f'Face fingerprints verified: {n}/{len(candidates)}', flush=True)
            candidates = fingerprinted
        groups, duplicates = deduplicate_and_group(candidates)
        transfers = assign_splits(groups, historical_hints)
        retained = [c for group in groups for c in group]
        if {c['id'] for c in retained} != set(registry.identities):
            raise ValueError('Identity coverage changed during review assembly.')
        print('Scanning current original photo folders...', flush=True)
        original_paths = photo_paths(originals.source_root)
        directories = {str(originals.source_root / info['source_directory']): name
                       for name, info in registry.identities.items()}
        for p in original_paths:
            if str(p.parent) not in directories:
                raise ValueError('Original is in an unregistered identity folder: ' + str(p))
        original_set = {str(p) for p in original_paths}
        for candidate in retained:
            if candidate['original_path'] not in original_set:
                raise ValueError('Approved face has no current original: ' + candidate['input_path'])
        report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'review_dataset_id': review.inventory['dataset_id'], 'generation': str(output),
                  'originals': len(original_paths), 'faces': len(retained),
                  'identities': len(registry.identities), 'source_photo_groups': len(groups),
                  'split_counts': dict(Counter(c['split'] for c in retained)),
                  'face_origins': dict(Counter(c['origin'] for c in retained)),
                  'excluded_or_superseded_records': len(exclusions),
                  'exact_face_duplicates_consolidated': len(duplicates),
                  'retired_originals': retirements,
                  'source_group_coverage_transfers': transfers,
                  'input_sha256': input_hashes,
                  'split_policy': 'Preserve historical membership when possible; every camera frame and exact source-content group stays in one split. Prioritize train, then val, then test for historical conflicts; repair identity coverage as whole groups.',
                  'evaluation_policy': 'Retrain from this generation before reporting evaluation results. Historical checkpoints and galleries describe earlier data.',
                  'infant_policy': 'Infants alone do not require a multi-individual original-photo review.',
                  'archive_only_deferred_frames': [record['filename'] for key, record in originals.catalog['records'].items()
                      if original_state['entries'][key]['draft']['decision'] == 'defer' and not record['active_files']]}
        print(json.dumps({k: report[k] for k in ('originals', 'faces', 'identities', 'source_photo_groups',
              'split_counts', 'face_origins', 'excluded_or_superseded_records', 'exact_face_duplicates_consolidated')}), flush=True)
        if plan_only:
            write(review.output / 'finalization_2026-11-25' / 'assembly_plan.json', report)
            return report
        with fresh_output_directory(output) as stage:
            original_manifest = []

            def copy_original(path):
                relative = path.relative_to(originals.source_root)
                target = stage / 'originals' / relative
                digest = snapshot_copy(path, target, current_expected.get(str(path)))
                return {'path': str(relative), 'id': directories[str(path.parent)], 'sha256': digest,
                        'file_bytes': target.stat().st_size, 'source_path': str(path)}

            print('Copying and verifying original photographs...', flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for n, item in enumerate(pool.map(copy_original, original_paths), 1):
                    original_manifest.append(item)
                    if n % 250 == 0:
                        print(f'Originals copied and verified: {n}/{len(original_paths)}', flush=True)
            payload = {s: [] for s in SPLIT_ORDER}
            original_lookup = {item['source_path']: item for item in original_manifest}

            def copy_face(candidate):
                split, name = candidate['split'], candidate['id']
                stem = hashlib.sha256(candidate['input_path'].encode()).hexdigest()[:20]
                relative = Path(split) / name / (stem + Path(candidate['input_path']).suffix.lower())
                target = stage / 'faces' / relative
                snapshot_copy(candidate['input_path'], target, candidate['sha256'])
                metadata = registry.metadata(name)
                current_original = original_lookup.get(candidate['original_path'])
                if current_original is None:
                    raise ValueError('Approved face has no current original: ' + candidate['input_path'])
                return split, {'path': str(relative), **metadata,
                    'sha256': candidate['sha256'], 'pixels_sha256': candidate['pixels_sha256'],
                    'file_bytes': target.stat().st_size, 'mtime_ns': target.stat().st_mtime_ns,
                    'master_source_path': current_original['path'],
                    'source_path': current_original['path'], 'source_group': candidate['source_group'],
                    'review_key': candidate['review_key'], 'review_status': candidate['review_status'],
                    'provenance': candidate['origin'], 'input_path': candidate['input_path'],
                    'camera_key': candidate['camera_key'], 'camera_aliases': candidate['camera_aliases']}

            print('Copying and verifying retained face crops...', flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for n, (split, item) in enumerate(pool.map(copy_face, retained), 1):
                    payload[split].append(item)
                    if n % 1000 == 0:
                        print(f'Faces copied and verified: {n}/{len(retained)}', flush=True)
            for rows in payload.values():
                rows.sort(key=lambda item: item['path'])
            print('Validating complete training manifest and camera-frame separation...', flush=True)
            registry.validate(payload)
            validate_manifest(payload, stage / 'faces', verify_content=True,
                              expected_labels=set(registry.identities))
            owners = {}
            for split, rows in payload.items():
                for row in rows:
                    for token in [row['source_group'], *row['camera_aliases']]:
                        if owners.setdefault(token, split) != split:
                            raise ValueError('Camera-frame derivatives cross splits: ' + token)
            final_originals = photo_paths(originals.source_root)
            if original_paths != final_originals:
                raise ValueError('Original photo inventory changed during assembly.')
            for path, digest in input_hashes.items():
                if sha(path) != digest:
                    raise ValueError('Review inputs changed during assembly: ' + path)
            write(stage / 'faces' / 'splits.json', payload)
            updated_registry = read(registry.path)
            updated_registry['source_root'] = str(output / 'originals')
            write(stage / 'identity_registry.json', updated_registry)
            write(stage / 'faces' / 'identity_registry.json', updated_registry)
            write(stage / 'originals_manifest.json', original_manifest)
            write(stage / 'audit' / 'excluded_or_superseded.json', exclusions)
            write(stage / 'audit' / 'exact_duplicate_consolidation.json', duplicates)
            write(stage / 'audit' / 'retained_faces_provenance.json', retained)
            write(stage / 'audit' / 'approved_organised_faces.json', exported)
            for p in inputs:
                relative = p.relative_to(review.output) if p.is_relative_to(review.output) else Path('historical_inputs') / p.name
                destination = stage / 'audit' / 'inputs' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, destination)
            for sub in ('corrections', 'original_photo_edits', 'original_photo_edits/face_stage/corrections'):
                shutil.copytree(review.output / sub / 'transactions', stage / 'audit' / 'transactions' / sub)
            report.update(published=True, verified_file_hashes=True, verified_decoded_face_hashes=True,
                          cross_split_source_groups=0, exact_face_duplicates=0,
                          all_identities_in_each_split=True, originals_are_independent_copies=True,
                          historical_review_inputs_preserved=True, training_ready=True,
                          requires_new_training_and_gallery=True)
            write(stage / 'summary.json', report)
        print('Published verified dataset: ' + str(output), flush=True)
        return report


def activate_generation(output):
    """Update active configs and one current-generation pointer with rollback."""
    import yaml

    output = Path(output).resolve()
    report = read(output / 'summary.json')
    if not report.get('published') or not report.get('training_ready'):
        raise ValueError('Only a published, verified generation can be activated.')
    registry = IdentityRegistry(output / 'identity_registry.json')
    manifest = read(output / 'faces/splits.json')
    registry.validate(manifest)
    validate_manifest(manifest, output / 'faces', expected_labels=set(registry.identities))
    pointer = ROOT / 'data/macaque_faces/current_dataset.json'
    plans = {}
    for path in sorted((ROOT / 'configs').glob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        if config.get('data', {}).get('dataset_name') != 'macaque_faces':
            continue
        if config['data'].get('comparison_role') == 'historical_baseline':
            continue
        original = path.read_bytes()
        suffix = '_reviewed_' + output.name.removeprefix('reviewed_').replace('-', '')
        if not config['name'].endswith(suffix):
            config['name'] += suffix
        config['data'].update(raw_root=str(output / 'faces'),
            splits_path=str(output / 'faces/splits.json'),
            identity_registry=str(output / 'identity_registry.json'), num_classes=report['identities'])
        config.setdefault('inference', {}).update(
            checkpoint='artifacts/' + config['name'] + '_best.pt',
            gallery_path='artifacts/' + config['name'] + '_gallery.pkl')
        encoded = ('# Reviewed dataset generation; train a fresh model and gallery.\n'
                   + yaml.safe_dump(config, sort_keys=False)).encode()
        plans[path] = (original, encoded)
    if not plans:
        raise ValueError('No active macaque training configurations were found.')
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    audit = output / 'audit/configs_before_activation'
    audit.mkdir(parents=True, exist_ok=False)
    for path, (original, _) in plans.items():
        (audit / path.name).write_bytes(original)
    if pointer_before is not None:
        (audit / 'current_dataset.before.json').write_bytes(pointer_before)

    def replace(path, data):
        temporary = path.with_name('.' + path.name + '.review-activation.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)

    activation = {'generation': str(output), 'configs': [str(p) for p in plans],
                  'status': 'prepared', 'created_at': datetime.now(timezone.utc).isoformat()}
    write(output / 'audit/activation.json', activation)
    try:
        for path, (original, encoded) in plans.items():
            if path.read_bytes() != original:
                raise ValueError('Config changed during activation: ' + str(path))
            replace(path, encoded)
        current = {'generation': str(output), 'originals': str(output / 'originals'),
                   'faces': str(output / 'faces'), 'splits_path': str(output / 'faces/splits.json'),
                   'identity_registry': str(output / 'identity_registry.json'),
                   'summary': str(output / 'summary.json'),
                   'original_photo_count': report['originals'], 'face_crop_count': report['faces'],
                   'identity_count': report['identities'], 'split_counts': report['split_counts'],
                   'training_ready': True, 'model_and_gallery_retraining_required': True,
                   'activated_at': datetime.now(timezone.utc).isoformat()}
        replace(pointer, (json.dumps(current, indent=2) + '\n').encode())
        activation['status'] = 'active'
        write(output / 'audit/activation.json', activation)
    except BaseException:
        for path, (original, _) in plans.items():
            replace(path, original)
        if pointer_before is None:
            pointer.unlink(missing_ok=True)
        else:
            replace(pointer, pointer_before)
        activation['status'] = 'rolled_back'
        write(output / 'audit/activation.json', activation)
        raise
    print(f'Activated generation in {len(plans)} training configs: {pointer}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=ROOT / 'reports/crop_review_2026-11-10')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--activate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    if args.activate_only:
        if args.plan_only:
            parser.error('--activate-only and --plan-only cannot be combined.')
        activate_generation(args.output)
    else:
        assemble(args.review_dir, args.output, args.plan_only, args.workers)


if __name__ == '__main__':
    main()
