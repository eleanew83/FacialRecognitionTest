"""Portable, file:// correction packages and checked, resumable batch import.

The returned JSON contains instructions, never trusted filesystem paths. A
private manifest on CSD3 binds every instruction to the exported source bytes.
"""
from __future__ import annotations

import base64
import copy
import importlib.util
import json
import math
import re
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def corrections_module():
    spec = importlib.util.spec_from_file_location('crop_corrections', Path(__file__).with_name('crop_corrections.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = corrections_module()
SCHEMA = 'macaque-offline-corrections/v1'


def fingerprint(value):
    return C.sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def export_package(store, scope='attention', workers=4):
    context = store.context()
    records = [r for r in context['records'] if scope == 'all' or
               r['status'] != 'ok' or scope == 'work' and r['uncertain_filename']]
    package_id = uuid.uuid4().hex
    folder = store.root / 'offline_packages' / package_id
    bundle = folder / 'macaque_offline_editor'
    assets = bundle / 'assets'
    assets.mkdir(parents=True)
    state = store.read()
    manifest = {'schema': SCHEMA, 'package_id': package_id, 'dataset_id': context['dataset_id'],
                'created_at': C.now(), 'records': {}, 'scope': scope}
    metas, inputs = {}, {}
    print(f'Packaging {len(records)} images with their full-resolution sources…', flush=True)
    for row in records:
        key = row['key']
        meta = store.metadata(key)
        # Old immutable offline snapshots are provenance, not extra UI variants.
        meta['variants'] = [v for v in meta['variants'] if not v['id'].startswith('offline_')]
        item = {'baseline_entry': copy.deepcopy(state['entries'].get(key)), 'variants': {},
                'source_path': meta['effects']['source_path']}
        for variant in meta['variants']:
            if not variant['available']:
                continue
            item['variants'][variant['id']] = {k: variant[k] for k in ('path', 'sha256', 'width', 'height')}
            variant['asset'] = variant['sha256']
            inputs[variant['sha256']] = Path(variant['path'])
        try:
            face = store.selected_image(key, 'face', state)
            raw = face.read_bytes()
            face_sha = C.sha(raw)
            item['face'] = {'path': str(face), 'sha256': face_sha}
            meta['face_asset'] = face_sha
            inputs[face_sha] = face
        except (ValueError, FileNotFoundError):
            meta['face_asset'] = None
        if not item['variants']:
            raise ValueError(f'No readable original for {key}; export cannot omit it silently.')
        manifest['records'][key] = item
        metas[key] = meta

    def asset(pair):
        digest, path = pair
        raw = path.read_bytes()
        if C.sha(raw) != digest:
            raise C.Conflict('Photo changed during export: ' + str(path))
        jpeg = C.image_bytes(raw)[0]
        script = 'window.CROP_OFFLINE_IMAGES[' + json.dumps(digest) + ']=' + json.dumps(
            'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()) + ';\n'
        (assets / (digest + '.js')).write_text(script)
        return len(jpeg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(asset, inputs.items()))
    # Do not publish a package whose baseline changed while it was generated.
    latest = store.read()
    for key, item in manifest['records'].items():
        if latest['entries'].get(key) != item['baseline_entry']:
            raise C.Conflict('Corrections changed during export; export again.')
    context['records'] = records
    data = {'manifest': {k: v for k, v in manifest.items() if k != 'records'},
            'context': context, 'metadata': metas}
    template = Path(__file__).with_name('crop_corrections.html').read_text()
    adapter = Path(__file__).with_suffix('.js').read_text()
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    injection = '<script id="offlineData" type="application/json">' + embedded + '</script><script>' + adapter + '</script>'
    template = template.replace('</main><script>', '</main>' + injection + '<script>', 1)
    (bundle / 'index.html').write_text(template)
    (bundle / 'README.txt').write_text(
        'MACAQUE OFFLINE CORRECTION EDITOR\n\n'
        'Extract this entire ZIP, then open index.html in Chrome, Edge or Firefox.\n'
        'Keep the assets folder beside index.html. No server, Python or internet is required.\n'
        'Use one tab. Work from the full-size source photo; draw and Preview, then Save offline correction.\n'
        'Download results after every session and before closing or moving the folder.\n'
        'Browser autosave is a convenience; the downloaded results JSON is your portable backup.\n'
        'Load results restores that JSON into this same package. Existing photos are immutable here.\n'
        'Original-photo crop/move/copy actions are queued for CSD3, not performed on this laptop.\n'
        'After an original edit you may draw a face box using the export-time original again.\n'
        'Only confirmed offline corrections are applied by import; unfinished drafts are retained in JSON.\n'
        'Upload the results JSON to CSD3. Run offline-import FILE to check it, then add --apply.\n'
        'Keep this package and the CSD3 corrections directory until final dataset assembly.\n'
        f'Package: {package_id}\nImages: {len(records)}\nCreated: {manifest["created_at"]}\n')
    C.atomic_json(folder / 'manifest.json', manifest)
    archive = store.review.output / ('offline_corrections_' + manifest['created_at'][:10] + '_' + package_id[:8] + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(folder))
    summary = {'package_id': package_id, 'records': len(records), 'already_applied': sum(r['applied'] for r in records),
               'image_assets': len(inputs), 'image_bytes': sum(sizes), 'zip_bytes': archive.stat().st_size,
               'zip_path': str(archive), 'index_path': str(bundle / 'index.html')}
    C.atomic_json(folder / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def read_results(store, path):
    results = json.loads(Path(path).read_text())
    if results.get('schema') != SCHEMA or results.get('dataset_id') != store.review.inventory['dataset_id']:
        raise ValueError('Offline results belong to a different dataset or format.')
    for field in ('package_id', 'session_id'):
        if not isinstance(results.get(field), str) or not re.fullmatch('[a-f0-9]{32}', results[field]):
            raise ValueError('Invalid offline ' + field)
    manifest_path = store.root / 'offline_packages' / results['package_id'] / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('The CSD3 manifest for this package is missing; use the original correction workspace.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['dataset_id'] != results['dataset_id']:
        raise ValueError('Package dataset mismatch.')
    if not isinstance(results.get('items'), dict):
        raise ValueError('Offline results need an items object.')
    seen = set()
    for key, item in results['items'].items():
        if key not in manifest['records'] or not isinstance(item, dict):
            raise ValueError('Unknown offline image: ' + key)
        if not isinstance(item.get('events', []), list) or len(item.get('events', [])) > 1000:
            raise ValueError('Invalid correction history.')
        for event in item.get('events', []):
            token = event.get('id')
            if not isinstance(token, str) or not re.fullmatch('[a-f0-9]{32}', token) or token in seen:
                raise ValueError('Invalid or duplicate offline event.')
            seen.add(token)
            draft = store.validate_draft(key, event.get('draft'))
            variant = manifest['records'][key]['variants'].get(draft['variant'])
            if not variant or variant['sha256'] != draft['source_sha256']:
                raise ValueError('Offline source version/hash differs from its package.')
            if draft['action'] in ('face_crop', 'edit_source') and not draft['box']:
                raise ValueError('A crop action is missing its rectangle.')
    return results, manifest


def committed_events(store, results, state, journal_cache=None):
    found = {}
    journal_cache = {} if journal_cache is None else journal_cache
    # All commits include undo journals; only still-active histories can count
    # as completed imports. An online undo after import is a conflict.
    active = {t for entry in state['entries'].values() for t in entry.get('history', [])}
    for token in state['commits']:
        if token not in journal_cache:
            journal_cache[token] = json.loads((store.root / 'transactions' / (token + '.json')).read_text())
        txn = journal_cache[token]
        tag = txn.get('offline')
        if tag and tag['package_id'] == results['package_id'] and tag['session_id'] == results['session_id']:
            if token not in active:
                raise C.Conflict('An imported correction was undone online; export a fresh package.')
            found[tag['event_id']] = txn
    return found


def check_import(store, results, manifest, only_key=None, journal_cache=None):
    state = store.read()
    done = committed_events(store, results, state, journal_cache)
    events = [(key, event) for key, item in results['items'].items() for event in item.get('events', [])]
    requested = {event['id']: event for _, event in events}
    for event_id, txn in done.items():
        if event_id not in requested or txn['offline']['event_sha256'] != fingerprint(requested[event_id]):
            raise C.Conflict('Previously imported history was removed or changed in this results file.')
    expected_files = {}
    for key in dict.fromkeys(key for key, _ in events):
        if only_key is not None and key != only_key:
            continue
        baseline = manifest['records'][key]
        entry = state['entries'].get(key)
        previous = [done[e['id']] for k, e in events if k == key and e['id'] in done]
        # Applied events must be an unchanged prefix, never reordered.
        ids = [e['id'] in done for k, e in events if k == key]
        if ids != sorted(ids, reverse=True):
            raise C.Conflict('Offline operations were reordered after import.')
        if previous:
            expected = {'history': (baseline['baseline_entry'] or {}).get('history', []) + [t['id'] for t in previous],
                        'draft': previous[-1]['draft'], 'draft_applied': True}
        else:
            expected = baseline['baseline_entry']
        if entry != expected:
            raise C.Conflict(f'Online corrections changed for {key}; export a fresh package or resolve this image first.')
        face_exists = bool(baseline.get('face'))
        source_directory = Path(baseline['source_path']).parent
        for k, event in events:
            if k != key:
                continue
            draft = event['draft']
            if draft['action'] in C.RELABEL_ACTIONS and not face_exists:
                raise ValueError('Create a face crop before relabelling an original-only photo.')
            if draft['action'] == 'face_crop':
                face_exists = True
            variant = baseline['variants'][draft['variant']]
            if draft['box'] and draft['action'] in C.SOURCE_ACTIONS | {'face_crop'}:
                x, y, w, h = draft['box']
                size = (math.ceil((x + w) * variant['width']) - math.floor(x * variant['width']),
                        math.ceil((y + h) * variant['height']) - math.floor(y * variant['height']))
                if min(size) < 2:
                    raise ValueError('An offline crop is too small.')
            if draft['action'] in C.SOURCE_ACTIONS and not store.identities[draft['target_identity']].get('source_directory'):
                raise ValueError('Target has no registered original-photo folder.')
            if draft['action'] in C.SOURCE_ACTIONS:
                target_directory = store.source_root / store.identities[draft['target_identity']]['source_directory']
                if draft['action'] == 'edit_source' and source_directory != target_directory:
                    raise ValueError('Use Move original when changing its identity folder.')
                if draft['action'] in ('move_source', 'relabel_move'):
                    source_directory = target_directory
        for variant in list(baseline['variants'].values()) + ([baseline['face']] if baseline.get('face') else []):
            expected_files[variant['path']] = variant['sha256']
    # Account for changes this very session has already committed, including a
    # cropped/moved original. This also supports safe retry after interruption.
    by_transaction = {txn['id']: txn for txn in done.values()}
    for token in state['commits']:
        txn = by_transaction.get(token)
        if txn:
            for operation in txn['operations']:
                if only_key is None or txn['key'] == only_key or operation['path'] in expected_files:
                    expected_files[operation['path']] = operation['after_sha256']
    for path, expected in expected_files.items():
        if C.file_sha(path) != expected:
            raise C.Conflict('Image changed since export/import: ' + path)
    return events, done


def import_results(store, path, apply=False):
    results, manifest = read_results(store, path)
    journal_cache = {}
    events, done = check_import(store, results, manifest, journal_cache=journal_cache)
    pending = [(key, event) for key, event in events if event['id'] not in done]
    report = {'package_id': results['package_id'], 'confirmed_operations': len(events),
              'already_imported': len(done), 'pending_operations': len(pending),
              'unfinished_drafts': sum(bool(item.get('draft')) and
                  (not item.get('events') or item['draft'] != item['events'][-1]['draft'])
                  for item in results['items'].values()), 'applied': 0}
    if not apply or not pending:
        print(json.dumps(report, indent=2))
        return report
    # Preserve baseline bytes before any original-folder operation. Offline
    # rectangles always refer to these immutable EXIF-oriented source sizes.
    input_root = store.root / 'offline_inputs'
    index_path = input_root / 'index.json'
    with store.guard():
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        for key, event in pending:
            variant = manifest['records'][key]['variants'][event['draft']['variant']]
            digest = variant['sha256']
            snapshot = input_root / results['package_id'] / (digest + '.bin')
            if not snapshot.exists():
                raw = Path(variant['path']).read_bytes()
                if C.sha(raw) != digest:
                    raise C.Conflict('Offline baseline source changed before snapshotting.')
                C.atomic_bytes(snapshot, raw)
            if C.file_sha(snapshot) != digest:
                raise C.Conflict('Offline input snapshot changed.')
            name = 'offline_' + results['package_id'][:8] + '_' + digest[:24]
            index.setdefault(key, {})[name] = {'path': str(snapshot), 'label': 'Offline export original ' + results['package_id'][:8]}
        C.atomic_json(index_path, index)
    for key, event in pending:
        # Stop if an online tab changed a record between batch operations.
        check_import(store, results, manifest, only_key=key, journal_cache=journal_cache)
        draft = copy.deepcopy(event['draft'])
        draft['variant'] = 'offline_' + results['package_id'][:8] + '_' + draft['source_sha256'][:24]
        plan = store.preview({'dataset_id': results['dataset_id'], 'key': key, 'draft': draft})
        plan['offline'] = {'package_id': results['package_id'], 'session_id': results['session_id'],
                           'event_id': event['id'], 'event_sha256': fingerprint(event)}
        C.atomic_json(store.root / 'previews' / plan['id'] / 'plan.json', plan)
        store.apply({'dataset_id': results['dataset_id'], 'token': plan['id']})
        report['applied'] += 1
        print(f'Applied {report["applied"]}/{len(pending)}: {store.record(key)["identity"]} — {C.ACTIONS[draft["action"]]}', flush=True)
    archive = store.root / 'offline_imports' / (results['session_id'] + '.json')
    C.atomic_json(archive, results)
    print(json.dumps(report, indent=2))
    return report


def run(args, review_store):
    store = C.CorrectionStore(review_store)
    if args.command == 'offline-export':
        return export_package(store, args.scope, args.workers)
    return import_results(store, args.results, args.apply)
