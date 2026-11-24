"""Build/serve a crop audit and an explicit, journalled correction editor.

Review decisions alone never modify images. Correction Apply buttons can edit
original-folder files with backups; face outputs are stored separately.
Use the project Python environment (Pillow required for building and editing).
See docs/CROP_REVIEW.md for the review workflow and SSH/VS Code instructions.
"""
from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import importlib.util
import io
import json
import mimetypes
import os
import signal
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'reports/crop_review_2026-11-10'
STATUSES = {
    'unreviewed': 'Unreviewed',
    'ok': 'OK — no obvious issue',
    'wrong_animal': 'Wrong animal selected',
    'wrong_label': 'Wrong identity label',
    'uncertain_identity': 'Unsure identity / ask Sylvain',
    'bad_crop': 'Bad crop / no usable face',
    'unreadable': 'Cannot open image',
    'not_croppable': 'No usable face / cannot crop',
}
CSV_FIELDS = [
    'identity', 'split', 'record_type', 'crop_path', 'source_path', 'sha256', 'source_photo', 'width', 'height',
    'status', 'suggested_identity', 'notes', 'updated_at', 'load_error',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes its dataset root: {relative}')
    return path


def publish_inventory(output, inventory):
    """Publish the grid without changing any saved annotation or existing key."""
    if (output / 'review_state.json').exists():
        state = json.loads((output / 'review_state.json').read_text())
        keys = {r['key'] for r in inventory['records']}
        if state['dataset_id'] != inventory['dataset_id'] or not set(state['annotations']) <= keys:
            raise ValueError('The new inventory would invalidate saved reviews. Use a new output directory.')
    atomic_json(output / 'inventory.json', inventory)
    data = json.dumps(inventory, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    temp = output / 'index.html.tmp'
    temp.write_text(template.replace('__INVENTORY__', data), encoding='utf-8')
    temp.replace(output / 'index.html')
    crops = [r for r in inventory['records'] if r.get('record_type', 'crop') == 'crop']
    originals = [r for r in inventory['records'] if r.get('record_type') == 'source_only']
    summary = {
        'dataset_id': inventory['dataset_id'], 'crops': len(crops),
        'source_photos_without_crop': len(originals), 'review_images': len(inventory['records']),
        'identities': len({r['identity'] for r in inventory['records']}),
        'splits': dict(Counter(r['split'] for r in crops)),
        'source_only_splits': dict(Counter(r['split'] for r in originals)),
        'missing_sources': sum(not r['source_exists'] for r in inventory['records']),
        'unreadable_crops': sum(bool(r['load_error']) for r in crops),
        'unreadable_originals': sum(bool(r['load_error']) for r in originals),
        'crop_hashes_verified': sum(not r['load_error'] for r in crops),
    }
    atomic_json(output / 'build_summary.json', summary)
    return summary


def include_uncropped_sources(inventory, source_manifest_path, output, workers=4):
    """Add original photos that have no active crop, using a separate key space.

    The dataset fingerprint still identifies the crop manifest and registry, so
    existing backups, browser progress and crop annotations remain compatible.
    Original-photo keys additionally bind their source path and displayed bytes.
    """
    from PIL import Image

    source_manifest_path = source_manifest_path.resolve()
    source_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_bytes)
    crop_manifest = json.loads(Path(inventory['manifest_path']).read_text())
    crop_items = [r for rows in crop_manifest.values() for r in rows]
    if any(not r.get('source_path') for r in crop_items):
        raise ValueError('Crop provenance needs source_path for each crop before adding originals.')
    linked = {r['source_path'] for r in crop_items}
    source_paths = {r['path'] for rows in source_manifest.values() for r in rows}
    if not linked <= source_paths:
        raise ValueError('Crop provenance does not match the selected source manifest.')
    source_root = Path(inventory['source_root'])
    jobs = [(split, r) for split, rows in source_manifest.items() for r in rows if r['path'] not in linked]
    print(f'Preparing {len(jobs)} original photographs without an active crop...', flush=True)

    def prepare(job):
        split, item = job
        if not item.get('master_source_path'):
            raise ValueError(f'No master source path for {item["path"]}')
        path = inside(source_root, item['master_source_path'])
        if path.parent.name != item['id']:
            raise ValueError(f'Source folder identity disagrees with manifest: {item["path"]}')
        error = ''
        try:
            image_bytes = path.read_bytes()
            actual_hash = digest(image_bytes)
        except OSError as exc:
            image_bytes = None
            actual_hash = item['sha256']
            error = str(exc)
        key = digest(('source_only:' + item['path'] + ':' + actual_hash).encode())[:24]
        record = {
            'key': key, 'record_type': 'source_only', 'identity': item['id'], 'split': split,
            'crop_path': '', 'crop_absolute': '', 'crop_file': '', 'source_path': item['path'],
            'sha256': actual_hash, 'source_photo': str(path), 'source_file': path.as_uri(),
            'source_exists': image_bytes is not None, 'group': item.get('current_group', ''),
            'sex': item.get('gender', ''), 'thumbnail': f'thumbs/{key}.jpg',
            'width': 0, 'height': 0, 'load_error': error,
        }
        if image_bytes is not None:
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    record['width'], record['height'] = image.size
                    image.thumbnail((360, 360), Image.Resampling.LANCZOS)
                    image.convert('RGB').save(output / record['thumbnail'], quality=83)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                record['load_error'] = str(exc)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        originals = list(pool.map(prepare, jobs))
    crops = [{**r, 'record_type': 'crop'} for r in inventory['records'] if r.get('record_type', 'crop') == 'crop']
    records = crops + originals
    if len({r['key'] for r in records}) != len(records):
        raise ValueError('Review key collision.')
    records.sort(key=lambda r: (r['identity'].casefold(), r['record_type'] != 'crop',
                               ('train', 'val', 'test').index(r['split']),
                               (r['crop_path'] or r['source_path']).casefold()))
    return {**inventory, 'records': records, 'statuses': STATUSES,
            'source_manifest_path': str(source_manifest_path),
            'source_manifest_sha256': digest(source_bytes), 'source_only_added_at': timestamp()}


def add_sources(args):
    output = args.output.resolve()
    inventory = json.loads((output / 'inventory.json').read_text())
    for field in ('manifest', 'registry'):
        if digest(Path(inventory[field + '_path']).read_bytes()) != inventory[field + '_sha256']:
            raise ValueError('Dataset changed since this review inventory was built.')
    inventory = include_uncropped_sources(inventory, args.source_manifest, output, args.workers)
    print(json.dumps(publish_inventory(output, inventory), indent=2), flush=True)


def build(args):
    from PIL import Image

    manifest_path = args.manifest.resolve()
    registry_path = args.registry.resolve()
    manifest_bytes = manifest_path.read_bytes()
    registry_bytes = registry_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    registry = json.loads(registry_bytes)
    source_root = Path(registry['source_root']).resolve()
    crop_root = args.crops.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'thumbs').mkdir(exist_ok=True)
    dataset_id = digest(manifest_bytes + b'\x00' + registry_bytes)
    index_path = output / 'inventory.json'
    previous_inventory = json.loads(index_path.read_text()) if index_path.exists() else {}
    if previous_inventory and previous_inventory['dataset_id'] != dataset_id:
        raise ValueError('This review directory belongs to another dataset. Choose a new --output.')

    jobs = [(split, item) for split in ('train', 'val', 'test') for item in manifest[split]]
    paths = [item['path'] for _, item in jobs]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate crop paths in manifest.')
    print(f'Checking and thumbnailing {len(jobs):,} crops. Original files are read only.', flush=True)

    def prepare(job):
        split, item = job
        path = inside(crop_root, item['path'])
        key = digest(item['path'].encode())[:24]
        source = inside(source_root, item['master_source_path']) if item.get('master_source_path') else None
        record = {
            'key': key, 'record_type': 'crop', 'identity': item['id'], 'split': split, 'crop_path': item['path'],
            'source_path': item.get('source_path', ''),
            'sha256': item.get('sha256', ''), 'crop_absolute': str(path), 'crop_file': path.as_uri(),
            'source_photo': str(source) if source else '',
            'source_file': source.as_uri() if source else '',
            'source_exists': bool(source and source.is_file()),
            'group': item.get('current_group', ''), 'sex': item.get('gender', ''),
            'thumbnail': f'thumbs/{key}.jpg', 'width': 0, 'height': 0, 'load_error': '',
        }
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            record['load_error'] = str(exc)
            return record
        if not item.get('sha256') or digest(image_bytes) != item['sha256']:
            raise ValueError(f'Crop does not match manifest SHA-256: {item["path"]}')
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                record['width'], record['height'] = image.size
                image.thumbnail((360, 360), Image.Resampling.LANCZOS)
                image.convert('RGB').save(output / record['thumbnail'], quality=83)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            record['load_error'] = str(exc)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(prepare, jobs), 1):
            records.append(record)
            if i % 500 == 0 or i == len(jobs):
                print(f'  {i:,}/{len(jobs):,}', flush=True)
    if len({r['key'] for r in records}) != len(records):
        raise ValueError('Review key collision.')
    records.sort(key=lambda r: (r['identity'].casefold(), ('train', 'val', 'test').index(r['split']), r['crop_path'].casefold()))
    inventory = {
        'schema_version': 1, 'dataset_id': dataset_id, 'built_at': timestamp(),
        'manifest_path': str(manifest_path), 'manifest_sha256': digest(manifest_bytes),
        'registry_path': str(registry_path), 'registry_sha256': digest(registry_bytes),
        'crop_root': str(crop_root), 'source_root': str(source_root),
        'statuses': STATUSES, 'records': records,
        'counts': dict(Counter(r['split'] for r in records)),
    }
    if previous_inventory.get('source_manifest_path'):
        inventory = include_uncropped_sources(inventory, Path(previous_inventory['source_manifest_path']), output, args.workers)
    summary = publish_inventory(output, inventory)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Ready: {output / "index.html"}\nStart: python {Path(__file__)} serve --output {output}', flush=True)


class ReviewStore:
    def __init__(self, output):
        self.output = output.resolve()
        self.inventory = json.loads((self.output / 'inventory.json').read_text())
        self.records = {r['key']: r for r in self.inventory['records']}
        self.path = self.output / 'review_state.json'
        self.lock = threading.Lock()
        self.state = {'schema_version': 1, 'dataset_id': self.inventory['dataset_id'], 'revision': 0, 'annotations': {}}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['dataset_id'] != self.inventory['dataset_id']:
                raise ValueError('Saved review belongs to another dataset.')

    def update(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object.')
        if payload.get('dataset_id') != self.inventory['dataset_id']:
            raise ValueError('Dataset fingerprint does not match.')
        changes = payload.get('changes')
        if not isinstance(changes, dict) or len(changes) > len(self.records):
            raise ValueError('Invalid annotation batch.')
        validated = {}
        for key, annotation in changes.items():
            if key not in self.records or not isinstance(annotation, dict):
                raise ValueError('Unknown crop in annotation batch.')
            status = annotation.get('status', 'unreviewed')
            if status not in STATUSES:
                raise ValueError('Unknown review status.')
            checked = {'status': status, 'updated_at': timestamp()}
            for field, limit in [('suggested_identity', 150), ('notes', 5000)]:
                value = annotation.get(field, '')
                if not isinstance(value, str) or len(value) > limit:
                    raise ValueError(f'Invalid {field}.')
                checked[field] = value
            validated[key] = checked
        with self.lock:
            if payload.get('base_revision') != self.state['revision']:
                return None
            state = {**self.state, 'revision': self.state['revision'] + 1,
                     'annotations': {**self.state['annotations'], **validated}}
            atomic_json(self.path, state)
            self.state = state
            return {'revision': state['revision'], 'saved_at': timestamp()}

    def csv_bytes(self, flagged=False):
        with self.lock:
            annotations = dict(self.state['annotations'])
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in self.inventory['records']:
            annotation = annotations.get(record['key'], {'status': 'unreviewed'})
            if flagged and annotation['status'] in ('unreviewed', 'ok'):
                continue
            row = {field: record.get(field, '') for field in CSV_FIELDS}
            row['record_type'] = record.get('record_type', 'crop')
            row.update({field: annotation.get(field, '') for field in ('status', 'suggested_identity', 'notes', 'updated_at')})
            # Prevent reviewed free text / filenames being interpreted as spreadsheet formulae.
            row = {k: "'" + str(v) if str(v).startswith(('=', '+', '-', '@', '\t', '\r')) else v for k, v in row.items()}
            writer.writerow(row)
        return ('\ufeff' + stream.getvalue()).encode('utf-8')


def handler_for(store):
    thumbnail_names = {r['key'] + '.jpg' for r in store.records.values()}

    def corrections():
        with store.lock:
            if not hasattr(store, 'corrections'):
                spec = importlib.util.spec_from_file_location('macaque_crop_corrections', Path(__file__).with_name('crop_corrections.py'))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                store.corrections = module.CorrectionStore(store)
            return store.corrections

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, mime, code=200, filename=None):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, code=200):
            self.send(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', code)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == '/corrections' or route.startswith('/api/corrections/'):
                try:
                    return corrections().handle_get(self, route, urlsplit(self.path).query)
                except (ValueError, OSError) as exc:
                    return self.send_json({'error': str(exc)}, 400)
            if route == '/api/review':
                with store.lock:
                    return self.send_json(store.state)
            if route in ('/api/export/all.csv', '/api/export/flagged.csv'):
                flagged = route.endswith('flagged.csv')
                return self.send(store.csv_bytes(flagged), 'text/csv; charset=utf-8', filename='flagged_crops.csv' if flagged else 'all_crop_reviews.csv')
            if route == '/api/export/backup.json':
                with store.lock:
                    return self.send(json.dumps(store.state, indent=2).encode(), 'application/json', filename='crop_review_backup.json')
            if route.startswith(('/crop/', '/source/')):
                kind, key = route.strip('/').split('/', 1)
                record = store.records.get(key)
                if not record:
                    return self.send_json({'error': 'Unknown crop.'}, 404)
                field = 'crop_absolute' if kind == 'crop' else 'source_photo'
                root = Path(store.inventory['crop_root' if kind == 'crop' else 'source_root'])
                candidate = Path(record[field]) if record[field] else None
                if kind == 'source' and hasattr(store, 'corrections'):
                    candidate = store.corrections.selected_image(key, 'master')
                if not candidate or not candidate.resolve().is_relative_to(root.resolve()):
                    return self.send_json({'error': 'Source unavailable.'}, 404)
            elif route in ('/', '/index.html'):
                candidate = store.output / 'index.html'
            elif route.startswith('/thumbs/') and route.count('/') == 2:
                candidate = store.output / 'thumbs' / route.rsplit('/', 1)[1]
                if candidate.name not in thumbnail_names:
                    return self.send_json({'error': 'Unknown thumbnail.'}, 404)
            else:
                return self.send_json({'error': 'Not found.'}, 404)
            try:
                self.send(candidate.read_bytes(), mimetypes.guess_type(str(candidate))[0] or 'application/octet-stream')
            except OSError:
                self.send_json({'error': 'Image unavailable on this machine.'}, 404)

        def do_POST(self):
            if self.path != '/api/review' and not self.path.startswith('/api/corrections/'):
                return self.send_json({'error': 'Not found.'}, 404)
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.send_json({'error': 'Origin does not match.'}, 403)
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected JSON.')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16_000_000:
                    raise ValueError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                if self.path.startswith('/api/corrections/'):
                    return corrections().handle_post(self, self.path, payload)
                result = store.update(payload)
                if result is None:
                    return self.send_json({'error': 'Another tab changed this review. Export a backup, then reload before continuing.'}, 409)
                self.send_json(result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except OSError:
                self.send_json({'error': 'Could not write review_state.json; download a backup.'}, 500)

    return Handler


def listener_pids(port):
    """Find this user's listening processes using Linux's read-only proc tables."""
    inodes = set()
    for table in ('tcp', 'tcp6'):
        try:
            lines = Path('/proc/net', table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(f'socket:[{fields[9]}]')
    pids = set()
    if not inodes:
        return pids
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            for fd in (process / 'fd').iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        pids.add(int(process.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def is_review_process(pid):
    """Require the same user, this exact script and the serve subcommand."""
    if pid == os.getpid():
        return False
    process = Path('/proc') / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            return False
        argv = process.joinpath('cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
        cwd = process.joinpath('cwd').resolve()
        for index, argument in enumerate(argv[1:], 1):
            if Path(argument).name == Path(__file__).name:
                candidate = Path(argument) if Path(argument).is_absolute() else cwd / argument
                if candidate.resolve() == Path(__file__).resolve() and argv[index + 1:index + 2] == ['serve']:
                    return True
    except (OSError, UnicodeError, ValueError):
        pass
    return False


def stop_previous_review_server(port, timeout=10):
    pids = listener_pids(port)
    if not pids or not all(is_review_process(pid) for pid in pids):
        return False
    for pid in sorted(pids):
        # Check again immediately before signalling in case a process exited.
        if not is_review_process(pid):
            continue
        print(f'Stopping previous crop-review server on port {port} (PID {pid})...', flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    # Wait for process exit, not just socket closure: request threads may still
    # be completing an atomic save after the listener closes.
    while any(is_review_process(pid) for pid in pids):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'The previous review server did not release port {port} within {timeout} seconds.')
        time.sleep(0.1)
    return True


class ReviewHTTPServer(ThreadingHTTPServer):
    # Finish any in-flight disk save before the old process exits on restart.
    daemon_threads = False


def open_review_server(args):
    store = ReviewStore(args.output)
    for field in ('manifest', 'registry'):
        if digest(Path(store.inventory[field + '_path']).read_bytes()) != store.inventory[field + '_sha256']:
            raise ValueError('Dataset inventory changed since build. Build a new review directory.')
    if store.inventory.get('source_manifest_path'):
        if digest(Path(store.inventory['source_manifest_path']).read_bytes()) != store.inventory['source_manifest_sha256']:
            raise ValueError('Source manifest changed since originals were added. Rebuild the review inventory.')
    if not 0 <= args.port <= 65535:
        raise RuntimeError('Port must be between 0 and 65535.')
    first_port = args.port
    for port in range(first_port, min(first_port + 20, 65536)):
        for attempt in range(2):
            try:
                server = ReviewHTTPServer(('127.0.0.1', port), handler_for(store))
                args.port = server.server_port
                return server
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
            if attempt == 0 and stop_previous_review_server(port):
                # Include any decisions saved by the old server during shutdown.
                store = ReviewStore(args.output)
                continue
            print(f'Port {port} is occupied by another service; trying the next port.', flush=True)
            break
    raise RuntimeError(f'No available port found starting at {first_port}. Try --port with a different starting port.')


def serve(args):
    server = open_review_server(args)
    stopping = threading.Event()

    def stop(signum, frame):
        if not stopping.is_set():
            stopping.set()
            # shutdown() must run outside the serve_forever() thread.
            threading.Thread(target=server.shutdown, daemon=True).start()

    old_handler = signal.signal(signal.SIGTERM, stop)
    print(f'Open http://localhost:{args.port}\nProgress saves to {args.output.resolve() / "review_state.json"}\nUse VS Code port forwarding or SSH -L for a remote HPC session. Ctrl-C stops the server.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('build', help='Verify crops and build review thumbnails.')
    make.add_argument('--manifest', type=Path, default=ROOT / 'data/macaque_faces/splits.json')
    make.add_argument('--registry', type=Path, default=ROOT / 'data/macaque_faces/identity_registry.json')
    make.add_argument('--crops', type=Path, default=ROOT.parent / 'yolo_detection/yolo_detection_code/output/macaque_crops')
    make.add_argument('--workers', type=int, default=4, choices=range(1, 17))
    originals = commands.add_parser('add-sources', help='Add original photos without active crops to an existing review, preserving saved decisions.')
    originals.add_argument('--source-manifest', type=Path, default=ROOT.parent / 'macaque_split_data/splits.json')
    originals.add_argument('--workers', type=int, default=4, choices=range(1, 17))
    show = commands.add_parser('serve', help='Open the review UI with source-photo access and disk autosave.')
    show.add_argument('--port', type=int, default=8765, help='Starting port; replace your previous crop-review server or choose the next available port.')
    for command in (make, show, originals):
        command.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        {'build': build, 'serve': serve, 'add-sources': add_sources}[args.command](args)
    except RuntimeError as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
