"""Expand deferred photo sequences and edit multiple whole-animal originals.

The untouched archive is read-only. User-approved batches update the active
identity folders, with a separate correction state and the existing WAL/undo.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import threading
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs

from PIL import Image, ImageOps

spec = importlib.util.spec_from_file_location('originals_transactions', Path(__file__).with_name('crop_corrections.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp', '.cr2'}


def frame_name(name):
    return re.sub(r'[\s_-]+', ' ', unicodedata.normalize('NFKC', Path(name).stem).casefold()).strip()


def family_name(name):
    # Strip short frame counters following letters, not numeric timestamps.
    stem = re.sub(r'(?<=[^\W\d_])[ _-]*\d{1,3}$', '', Path(name).stem)
    return frame_name(stem + '.jpg')


def candidate_name(name):
    """Camera-frame identity for reviewed derivatives, without changing old IDs."""
    while name.lower().endswith('.jpg.jpg'):
        name = name[:-4]
    stem = re.sub(r'(?i)_(?:cropped|crop\d+)$', '', Path(name).stem)
    return frame_name(stem + '.jpg')


def oriented_size(path):
    with Image.open(path) as image:
        w, h = image.size
        return (h, w) if image.getexif().get(274) in (5, 6, 7, 8) else (w, h)


def developed_image(path, digest, root):
    if path.suffix.lower() != '.cr2':
        return path
    output = root / 'developed_raw' / (digest + '.jpg')
    if not output.exists():
        dcraw = shutil.which('dcraw')
        if not dcraw:
            raise ValueError('dcraw is required to open this Canon RAW photo: ' + str(path))
        print('Developing full-resolution RAW: ' + path.name, flush=True)
        result = subprocess.run([dcraw, '-c', '-w', '-q', '0', str(path)], capture_output=True, check=True)
        with Image.open(io.BytesIO(result.stdout)) as image:
            stream = io.BytesIO()
            image.convert('RGB').save(stream, format='JPEG', quality=95, subsampling=0)
        C.atomic_bytes(output, stream.getvalue())
    return output


def build_catalog(review, archive):
    root = review.output / 'original_photo_edits'
    root.mkdir(exist_ok=True)
    state_path = root / 'state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('entries'):
        raise ValueError('This original-photo workspace already has saved work. Its catalog is preserved; open the editor without rebuilding.')
    archive = Path(archive).resolve()
    active = Path(review.inventory['source_root']).resolve()
    if archive == active or not archive.is_dir():
        raise ValueError('Choose the separate, untouched original-photo archive.')
    registry = json.loads(Path(review.inventory['registry_path']).read_text())['identities']
    directories = {str((active / item['source_directory']).resolve()): name for name, item in registry.items() if item.get('source_directory')}
    corrections = json.loads((review.output / 'corrections/state.json').read_text())
    seeds = defaultdict(list)
    for key, entry in corrections['entries'].items():
        if entry.get('draft', {}).get('action') == 'defer':
            record = review.records[key]
            seeds[family_name(Path(record['source_photo']).name)].append({
                'key': key, 'identity': record['identity'], 'filename': Path(record['source_photo']).name,
                'notes': entry['draft'].get('notes', ''), 'target_identity': entry['draft']['target_identity'],
                'review_notes': review.state['annotations'].get(key, {}).get('notes', '')})
    if not seeds:
        raise ValueError('No deferred original-photo cases were found.')
    paths = defaultdict(list)
    for source_root, kind in [(archive, 'archive'), (active, 'active')]:
        for directory, _, filenames in os.walk(source_root):
            for filename in filenames:
                if filename.startswith('.') or Path(filename).suffix.lower() not in EXTENSIONS:
                    continue
                family = family_name(filename)
                if family in seeds:
                    paths[frame_name(filename)].append((kind, Path(directory) / filename, family))
    records, issues = {}, []
    for number, (frame, versions) in enumerate(sorted(paths.items()), 1):
        key = C.sha(frame.encode())[:24]
        candidates, active_files = [], []
        family = versions[0][2]
        for kind, path, _ in versions:
            path = C.confined(archive if kind == 'archive' else active, path)
            try:
                digest = C.file_sha(path)
                pixels = developed_image(path, digest, root)
                w, h = oriented_size(pixels)
                version = {'id': C.sha((kind + str(path)).encode())[:20], 'kind': kind, 'path': str(path),
                           'sha256': digest, 'pixel_path': str(pixels), 'pixel_sha256': C.file_sha(pixels) if pixels != path else digest,
                           'width': w, 'height': h, 'identity': directories.get(str(path.parent), '')}
                candidates.append(version)
                if kind == 'active':
                    active_files.append(version.copy())
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                issues.append({'path': str(path), 'error': str(exc)})
        if not candidates:
            raise ValueError('No readable source for ' + frame)
        # Identical archive copies in different identity folders are one view.
        candidates.sort(key=lambda v: (v['width'] * v['height'], v['kind'] == 'archive'), reverse=True)
        variants, seen = [], set()
        for variant in candidates:
            duplicate = (variant['kind'], variant['sha256'])
            if duplicate not in seen:
                variants.append(variant)
                seen.add(duplicate)
        matching = [s for s in seeds[family] if frame_name(s['filename']) == frame]
        filename = next((p.name for kind, p, _ in versions if kind == 'archive'), versions[0][1].name)
        records[key] = {'key': key, 'filename': filename, 'family': family, 'seed': bool(matching),
                        'seeds': matching, 'family_notes': seeds[family], 'variants': variants,
                        'default_variant': variants[0]['id'], 'active_files': active_files}
        print(f'Catalogued {number}/{len(paths)}: {filename}', flush=True)
    summary = {'frames': len(records), 'families': len(seeds), 'seed_frames': sum(r['seed'] for r in records.values()),
               'additional_frames': sum(not r['seed'] for r in records.values()),
               'raw_frames': sum(any(v['path'].lower().endswith('.cr2') for v in r['variants']) for r in records.values()),
               'families_counts': dict(Counter(r['family'] for r in records.values())), 'issues': issues}
    catalog = {'schema_version': 1, 'dataset_id': review.inventory['dataset_id'], 'created_at': C.now(),
               'archive_root': str(archive), 'active_root': str(active), 'summary': summary, 'records': records}
    C.atomic_json(root / 'catalog.json', catalog)
    with (root / 'expanded_photo_queue.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key', 'family', 'filename', 'originally_deferred', 'source', 'active_identity_folders'])
        writer.writeheader()
        for r in records.values():
            writer.writerow({'key': r['key'], 'family': r['family'], 'filename': r['filename'], 'originally_deferred': r['seed'],
                             'source': r['variants'][0]['path'], 'active_identity_folders': '; '.join(v['identity'] for v in r['active_files'])})
    print(json.dumps(summary, indent=2), flush=True)
    return catalog


def append_goblin_candidates(review, audit_path):
    """Append the visual-audit candidates; existing records and state are immutable.

    Work is prepared outside the operation lock. Publishing checks that no other
    catalog extension happened meanwhile, and saves an exact catalog backup.
    """
    store = OriginalPhotoStore(review)
    catalog_path = store.root / 'catalog.json'
    baseline = catalog_path.read_bytes()
    catalog = json.loads(baseline)
    audit_raw = Path(audit_path).read_bytes()
    with Path(audit_path).open(newline='') as handle:
        cases = [r for r in csv.DictReader(handle) if r['visual_category'] in (
            'additional_adult_or_juvenile', 'adult_with_infant_or_young', 'possible_occluded_infant')]
    if not cases:
        raise ValueError('No visually flagged Goblin candidates in this audit.')
    by_frame = defaultdict(list)
    for row in cases:
        C.confined(store.source_root, row['path'])
        if Path(row['path']).parent != store.source_root / store.identities['Goblin']['source_directory']:
            raise ValueError('This candidate is outside the registered Goblin folder.')
        by_frame[candidate_name(row['filename'])].append(row)
    existing = {candidate_name(r['filename']): key for key, r in catalog['records'].items()}
    missing = set(by_frame) - set(existing)
    if not missing:
        return {'added_frames': 0, 'selected_files': len(cases), 'selected_photos': len(by_frame),
                'already_queued': len(by_frame), 'total_frames': len(catalog['records'])}
    versions = defaultdict(list)
    for source_root, kind in [(store.archive_root, 'archive'), (store.source_root, 'active')]:
        for directory, _, names in os.walk(source_root):
            for name in names:
                if name.startswith('.') or Path(name).suffix.lower() not in EXTENSIONS:
                    continue
                frame = candidate_name(name)
                if frame in missing:
                    versions[frame].append((kind, Path(directory) / name))
    directories = {str((store.source_root / item['source_directory']).resolve()): identity
                   for identity, item in store.identities.items() if item.get('source_directory')}
    new_keys = []
    family_notes = defaultdict(list)
    for frame in sorted(missing):
        rows = by_frame[frame]
        if not versions[frame]:
            raise ValueError('No current source versions found for ' + rows[0]['filename'])
        candidates, active_files = [], []
        for kind, path in versions[frame]:
            path = C.confined(store.archive_root if kind == 'archive' else store.source_root, path)
            digest = C.file_sha(path)
            pixels = developed_image(path, digest, store.root)
            w, h = oriented_size(pixels)
            version = {'id': C.sha((kind + str(path)).encode())[:20], 'kind': kind, 'path': str(path),
                       'sha256': digest, 'pixel_path': str(pixels), 'pixel_sha256': C.file_sha(pixels) if pixels != path else digest,
                       'width': w, 'height': h, 'identity': directories.get(str(path.parent), '')}
            candidates.append(version)
            if kind == 'active':
                active_files.append(version.copy())
        # Keep every listed active copy, including cropped derivatives. Only
        # deduplicate identical bytes in the source-version selector.
        candidates.sort(key=lambda v: (v['width'] * v['height'], v['kind'] == 'archive'), reverse=True)
        distinct, seen = [], set()
        for v in candidates:
            signature = (v['kind'], v['sha256'])
            if signature not in seen:
                distinct.append(v)
                seen.add(signature)
        source = next((v for v in distinct if v['kind'] == 'archive'), distinct[0])
        filename = Path(source['path']).name
        while filename.lower().endswith('.jpg.jpg'):
            filename = filename[:-4]
        filename = re.sub(r'(?i)_(cropped|crop\d+)(?=\.[^.]+$)', '', filename)
        key = C.sha(frame.encode())[:24]
        if key in catalog['records']:
            raise ValueError('Candidate key collision; catalog left unchanged.')
        categories = sorted({r['visual_category'] for r in rows})
        descriptions = {
            'additional_adult_or_juvenile': 'Another adult/juvenile is visible, sometimes partly in frame. Check whether the intended individual is sufficiently isolated.',
            'adult_with_infant_or_young': 'An adult and infant/young monkey are visible. This is a review candidate; retaining the existing original may be appropriate.',
            'possible_occluded_infant': 'A possible mostly hidden infant needs a closer look. Keep the original if no correction is needed.'}
        notes = 'Goblin visual audit: ' + ' '.join(descriptions[c] for c in categories) + ' Identity assignments have not been re-verified. No cropping or relabelling was applied by adding this photo.'
        seed = {'key': 'goblin_audit_' + key, 'identity': 'Goblin', 'filename': filename,
                'target_identity': 'Goblin', 'notes': notes, 'review_notes': notes, 'origin': 'goblin_visual_audit'}
        family = family_name(filename)
        record = {'key': key, 'filename': filename, 'family': family, 'seed': False,
                  'seeds': [seed], 'family_notes': [seed], 'variants': distinct, 'default_variant': distinct[0]['id'],
                  'active_files': active_files, 'review_categories': categories,
                  'audit_files': [r['path'] for r in rows], 'added_reason': 'goblin_visual_audit'}
        catalog['records'][key] = record
        family_notes[family].append(seed)
        new_keys.append(key)
        print('Prepared Goblin review: ' + filename, flush=True)
    for key in new_keys:
        catalog['records'][key]['family_notes'] = family_notes[catalog['records'][key]['family']]
    records = list(catalog['records'].values())
    catalog['summary'].update(frames=len(records), families=len({r['family'] for r in records}),
        seed_frames=sum(r['seed'] for r in records), additional_frames=sum(not r['seed'] for r in records),
        raw_frames=sum(any(v['path'].lower().endswith('.cr2') for v in r['variants']) for r in records),
        families_counts=dict(Counter(r['family'] for r in records)))
    result = {'added_frames': len(new_keys), 'selected_files': len(cases), 'selected_photos': len(by_frame),
              'already_queued': len(by_frame) - len(new_keys), 'total_frames': len(records),
              'added_keys': new_keys, 'categories_file_counts': dict(Counter(r['visual_category'] for r in cases)),
              'created_at': C.now(), 'audit_sha256': C.sha(audit_raw)}
    catalog.setdefault('append_history', []).append(result)
    with store.guard():
        if catalog_path.read_bytes() != baseline:
            raise C.Conflict('The catalog changed while preparing additions. Run the extension again.')
        # Recheck file versions so no stale before-hash is introduced into a
        # newly reviewable record while another tool updates an original.
        for key in new_keys:
            for version in catalog['records'][key]['variants']:
                if C.file_sha(version['path']) != version['sha256']:
                    raise C.Conflict('An original changed during queue preparation. Retry the extension.')
        backup = store.root / 'catalog_backups' / ('before_goblin_' + uuid.uuid4().hex + '.json')
        C.atomic_bytes(backup, baseline)
        C.atomic_json(catalog_path, catalog)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['key', 'family', 'filename', 'originally_deferred', 'source', 'active_identity_folders'])
        writer.writeheader()
        for r in records:
            writer.writerow({'key': r['key'], 'family': r['family'], 'filename': r['filename'], 'originally_deferred': r['seed'],
                             'source': r['variants'][0]['path'], 'active_identity_folders': '; '.join(v['identity'] for v in r['active_files'])})
        C.atomic_bytes(store.root / 'expanded_photo_queue.csv', stream.getvalue().encode())
        C.atomic_json(store.root / 'goblin_queue_extension.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'added_keys'}, indent=2), flush=True)
    return result


class OriginalPhotoStore(C.CorrectionStore):
    """Separate state; reuse the tested transaction, recovery and undo engine."""
    def __init__(self, review):
        self.review = review
        self.root = review.output / 'original_photo_edits'
        if not (self.root / 'catalog.json').is_file():
            raise ValueError('Build the expanded queue first with source-build.')
        self.catalog = json.loads((self.root / 'catalog.json').read_text())
        self.catalog_stamp = None
        if self.catalog['dataset_id'] != review.inventory['dataset_id']:
            raise ValueError('Original-photo catalog belongs to another review.')
        self.source_root = Path(self.catalog['active_root']).resolve()
        self.archive_root = Path(self.catalog['archive_root']).resolve()
        self.path = self.root / 'state.json'
        self.lock = threading.RLock()
        self.identities = json.loads(Path(review.inventory['registry_path']).read_text())['identities']
        for name in ('previews', 'transactions', 'backups', 'faces', 'image_cache'):
            (self.root / name).mkdir(exist_ok=True)
        with self.guard():
            self.recover()

    def record(self, key):
        self.refresh_catalog()
        if key not in self.catalog['records']:
            raise ValueError('Unknown original photograph.')
        return self.catalog['records'][key]

    def refresh_catalog(self):
        path = self.root / 'catalog.json'
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self.catalog_stamp:
            catalog = json.loads(path.read_text())
            if catalog['dataset_id'] != self.review.inventory['dataset_id'] or Path(catalog['active_root']).resolve() != self.source_root:
                raise ValueError('The updated catalog has incompatible dataset roots.')
            self.catalog = catalog
            self.catalog_stamp = stamp

    def context(self):
        self.refresh_catalog()
        state = self.read()
        categories = defaultdict(set)
        audit_path = self.review.output / 'goblin_audit_2026-11-25' / 'all_95_goblin_images.csv'
        if audit_path.is_file():
            with audit_path.open(newline='') as handle:
                for item in csv.DictReader(handle):
                    categories[candidate_name(item['filename'])].add(item['visual_category'])
        rows = []
        for r in self.catalog['records'].values():
            entry = state['entries'].get(r['key'], {})
            applied = bool(entry.get('history'))
            rows.append({'key': r['key'], 'filename': r['filename'], 'family': r['family'], 'seed': r['seed'],
                         'goblin_candidate': r.get('added_reason') == 'goblin_visual_audit' or any(v.get('identity') == 'Goblin' for v in r['active_files']),
                         'review_categories': sorted(set(r.get('review_categories', [])) | categories[candidate_name(r['filename'])]),
                         'identities': sorted({v['identity'] for v in r['active_files'] if v['identity']}),
                         'status': entry.get('draft', {}).get('decision', 'edit') if applied else 'draft' if entry.get('draft') else 'pending',
                         'applied': applied})
        return {'dataset_id': state['dataset_id'], 'revision': state['revision'], 'records': rows,
                'identities': sorted(self.identities), 'summary': self.catalog['summary']}

    def metadata(self, key):
        r = self.record(key)
        state = self.read()
        entry = state['entries'].get(key, {})
        txn = json.loads((self.root / 'transactions' / (entry['history'][-1] + '.json')).read_text()) if entry.get('history') else None
        return {**r, 'revision': state['revision'], 'entry': entry, 'applied_effect': txn.get('effect') if txn else None}

    def variant(self, key, variant_id):
        r = self.record(key)
        for v in r['variants']:
            if v['id'] == variant_id:
                C.confined(self.archive_root if v['kind'] == 'archive' else self.source_root, v['path'])
                if Path(v['pixel_path']) != Path(v['path']):
                    C.confined(self.root / 'developed_raw', v['pixel_path'])
                return v
        raise ValueError('Unknown original image version.')

    def validate(self, key, draft, require_outputs=True):
        r = self.record(key)
        if not isinstance(draft, dict) or draft.get('decision') not in ('edit', 'keep', 'defer'):
            raise ValueError('Choose Edit originals, Keep originals, or Defer.')
        variant = self.variant(key, draft.get('variant'))
        if draft.get('source_sha256') != variant['sha256']:
            raise C.Conflict('Original source changed. Reload it.')
        notes = draft.get('notes', '')
        outputs, retire = draft.get('outputs', []), draft.get('retire', [])
        if not isinstance(notes, str) or len(notes) > 10000 or not isinstance(outputs, list) or len(outputs) > 30:
            raise ValueError('Invalid notes or number of outputs.')
        if not isinstance(retire, list) or len(set(retire)) != len(retire):
            raise ValueError('Invalid old-file removal list.')
        active = {v['id']: v for v in r['active_files']}
        if any(v not in active for v in retire):
            raise ValueError('Only listed active folder copies may be retired.')
        clean, seen, destinations = [], set(), set()
        for output in outputs:
            if not isinstance(output, dict) or not re.fullmatch('[a-f0-9]{16}', str(output.get('id', ''))) or output['id'] in seen:
                raise ValueError('Invalid or duplicate crop identifier.')
            seen.add(output['id'])
            target = output.get('identity')
            if target not in self.identities or not self.identities[target].get('source_directory'):
                raise ValueError('Choose an identity with a registered original folder.')
            box = C.clean_box(output.get('box'))
            if not box:
                raise ValueError('Draw a whole-monkey rectangle for every output.')
            replace = output.get('replace') or None
            if replace and (replace not in active or active[replace]['identity'] != target):
                raise ValueError('Replacement file must belong to the selected identity in this photo group.')
            if replace and (replace in destinations or replace in retire):
                raise ValueError('An existing file may be replaced once or retired, not both.')
            if replace:
                destinations.add(replace)
            clean.append({'id': output['id'], 'identity': target, 'box': box, 'replace': replace})
        if require_outputs and draft['decision'] == 'edit' and not clean and not retire:
            raise ValueError('Draw at least one output crop, select an old copy to retire, or choose Keep/Defer.')
        return {'decision': draft['decision'], 'variant': variant['id'], 'source_sha256': variant['sha256'],
                'notes': notes, 'outputs': clean, 'retire': retire}

    def save_draft(self, payload):
        self.check_dataset(payload)
        key = payload.get('key')
        draft = self.validate(key, payload.get('draft'), require_outputs=False)
        with self.guard():
            state = self.read()
            if payload.get('base_revision') != state['revision']:
                raise C.Conflict('Another tab changed this workspace. Reload before saving.')
            if state['entries'].get(key, {}).get('history'):
                raise ValueError('Undo this photo’s applied batch before changing its draft.')
            state['entries'].setdefault(key, {'history': []}).update(draft=draft, draft_applied=False)
            state['revision'] += 1
            C.atomic_json(self.path, state)
            return {'revision': state['revision'], 'saved_at': C.now()}

    def preview(self, payload):
        self.check_dataset(payload)
        key = payload.get('key')
        draft = self.validate(key, payload.get('draft'))
        with self.guard():
            state = self.read()
            if state['entries'].get(key, {}).get('history'):
                raise ValueError('This photo is already applied. Undo it before making another batch.')
            record, source = self.record(key), self.variant(key, draft['variant'])
            token = uuid.uuid4().hex
            folder = self.root / 'previews' / token
            folder.mkdir()
            inputs = {source['path']: source['sha256'], source['pixel_path']: source['pixel_sha256']}
            for path, expected in inputs.items():
                if C.file_sha(path) != expected:
                    raise C.Conflict('The chosen source has changed: ' + path)
            active = {v['id']: v for v in record['active_files']}
            # Check every existing copy, including the ones retained unchanged.
            # Avoid losing a concurrent source-folder correction in another UI.
            inputs.update({v['path']: v['sha256'] for v in active.values()})
            for path, expected in inputs.items():
                if C.file_sha(path) != expected:
                    raise C.Conflict('An existing original-folder copy changed. Reload/reconcile before applying: ' + path)
            plan = {'id': token, 'key': key, 'draft': draft, 'created_at': C.now(), 'base_revision': state['revision'],
                    'inputs': inputs, 'operations': [], 'effect': {'decision': draft['decision'], 'outputs': [],
                    'retired': [], 'related_review_keys': [s['key'] for s in record['family_notes']],
                    'face_dataset_needs_rebuild': draft['decision'] == 'edit'}, 'warnings': []}
            if draft['decision'] == 'edit':
                raw = Path(source['pixel_path']).read_bytes()
                for index, output in enumerate(draft['outputs']):
                    if output['replace']:
                        dest = Path(active[output['replace']]['path'])
                    else:
                        directory = C.confined(self.source_root, self.source_root / self.identities[output['identity']]['source_directory'])
                        stem = Path(record['filename']).stem.encode()[:150].decode('utf-8', errors='ignore')
                        dest = directory / (stem + '__' + output['identity'] + '__' + output['id'][:8] + '.jpg')
                        if dest.exists():
                            raise C.Conflict('New output already exists; choose its existing copy instead of adding a duplicate.')
                    dest = self.check_operation_path(dest)
                    data, size = C.image_bytes(raw, output['box'], Image.registered_extensions().get(dest.suffix.lower(), 'JPEG'))
                    if min(size) < 2:
                        raise ValueError('An output rectangle is too small.')
                    payload_path = folder / (str(index) + '.bin')
                    C.atomic_bytes(payload_path, data)
                    plan['operations'].append({'path': str(dest), 'before_sha256': C.file_sha(dest), 'after_sha256': C.sha(data),
                                                'payload': str(payload_path), 'description': 'Replace original' if output['replace'] else 'Add original'})
                    face = {'identity': output['identity'], 'path': str(dest), 'width': size[0], 'height': size[1],
                            'sha256': C.sha(data), 'box': output['box'], 'source_path': source['path'],
                            'source_sha256': source['sha256'], 'source_group': key,
                            'preview_url': f'/api/originals/preview/{token}/{index}'}
                    plan['effect']['outputs'].append(face)
                    with Image.open(io.BytesIO(data)) as im:
                        im = ImageOps.exif_transpose(im).convert('RGB')
                        im.thumbnail((1000, 1000))
                        im.save(folder / (str(index) + '.jpg'), quality=90)
                for old_id in draft['retire']:
                    old = active[old_id]
                    plan['operations'].append({'path': old['path'], 'before_sha256': old['sha256'],
                                                'after_sha256': None, 'description': 'Retire old folder copy (backed up)'})
                    plan['effect']['retired'].append(old['path'])
                plan['warnings'].append('These changes update the active original identity folders. Verify/regenerate affected face crops before retraining.')
            elif draft['outputs'] or draft['retire']:
                plan['warnings'].append('Keep/Defer records a decision only. Drawn boxes and removal selections will not change files.')
            plan['backup_directory'] = str(self.root / 'backups' / token)
            C.atomic_json(folder / 'plan.json', plan)
            return plan

    def image(self, key, image_id, full=False, existing=False):
        if existing:
            matches = [v for v in self.record(key)['active_files'] if v['id'] == image_id]
            if not matches:
                raise ValueError('Unknown existing copy.')
            path = C.confined(self.source_root, matches[0]['path'])
            if not path.exists():
                entry = self.read()['entries'].get(key, {})
                if entry.get('history'):
                    txn = json.loads((self.root / 'transactions' / (entry['history'][-1] + '.json')).read_text())
                    old = next((op for op in txn['operations'] if op['path'] == str(path) and op.get('backup')), None)
                    if old:
                        path = C.confined(self.root / 'backups', old['backup'])
            digest = C.file_sha(path)
            pixels = path
        else:
            version = self.variant(key, image_id)
            path, pixels = Path(version['path']), Path(version['pixel_path'])
            digest = C.file_sha(pixels) if version['kind'] == 'active' else version['pixel_sha256']
        cached = self.root / 'image_cache' / (digest + ('_full' if full else '_view') + '.jpg')
        if not cached.exists():
            with Image.open(pixels) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                if not full:
                    im.thumbnail((2048, 2048))
                data = io.BytesIO()
                im.save(data, format='JPEG', quality=92, subsampling=0)
            C.atomic_bytes(cached, data.getvalue())
        return cached.read_bytes()

    def handle_get(self, handler, route, query):
        if not C.own_user_connection(handler):
            return handler.send_json({'error': 'Use your own CSD3 SSH/VS Code forwarding connection.'}, 403)
        try:
            q = parse_qs(query)
            if route == '/originals':
                return handler.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            if route == '/api/originals/context':
                return handler.send_json(self.context())
            if route == '/api/originals/export':
                data = self.read()
                data['catalog'] = self.catalog
                data['transactions'] = [json.loads((self.root / 'transactions' / (t + '.json')).read_text()) for t in data['commits']]
                return handler.send(json.dumps(data, indent=2).encode(), 'application/json', filename='original_photo_edits_backup.json')
            if route.startswith('/api/originals/record/'):
                return handler.send_json(self.metadata(route.rsplit('/', 1)[1]))
            if route.startswith('/api/originals/image/'):
                return handler.send(self.image(route.rsplit('/', 1)[1], q.get('variant', [''])[0],
                                               q.get('full') == ['1'], q.get('existing') == ['1']), 'image/jpeg')
            match = re.fullmatch(r'/api/originals/preview/([a-f0-9]{32})/(\d+)', route)
            if match:
                return handler.send((self.root / 'previews' / match[1] / (match[2] + '.jpg')).read_bytes(), 'image/jpeg')
            return handler.send_json({'error': 'Unknown original-photo route.'}, 404)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return handler.send_json({'error': str(exc)}, 409 if isinstance(exc, C.Conflict) else 400)

    def handle_post(self, handler, route, payload):
        if not C.own_user_connection(handler):
            return handler.send_json({'error': 'Use your own CSD3 forwarding connection.'}, 403)
        try:
            method = {'draft': self.save_draft, 'preview': self.preview, 'apply': self.apply, 'undo': self.undo}.get(route.rsplit('/', 1)[1])
            if not method:
                raise ValueError('Unknown original-photo operation.')
            return handler.send_json(method(payload))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return handler.send_json({'error': str(exc)}, 409 if isinstance(exc, C.Conflict) else 400)
