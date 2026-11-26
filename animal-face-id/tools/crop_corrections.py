"""Manual correction workspace for the localhost macaque review server.

Review decisions and historical training crops are never overwritten. Explicit
source actions can change original-folder files, with backups and a write-ahead
journal. Applied face corrections are immutable outputs in this workspace.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

from PIL import Image, ImageOps

ACTIONS = {
    'face_crop': 'Recrop face only',
    'relabel_crop': 'Relabel face crop only',
    'relabel_move': 'Relabel crop + move original',
    'relabel_copy': 'Relabel crop + copy original',
    'edit_source': 'Crop original-folder photo',
    'move_source': 'Move original to another identity',
    'copy_source': 'Copy original to another identity',
    'keep': 'Keep as reviewed',
    'exclude': 'Exclude from corrected dataset',
    'defer': 'Defer identity decision',
}
SOURCE_ACTIONS = {'relabel_move', 'relabel_copy', 'edit_source', 'move_source', 'copy_source'}
RELABEL_ACTIONS = {'relabel_crop', 'relabel_move', 'relabel_copy'}


class Conflict(ValueError):
    pass


def own_user_connection(handler):
    """A shared HPC localhost is reachable by other users: verify the peer UID.

    SSH/VS Code forwarding creates the client socket under the connected user's
    account. The kernel's reverse TCP entry identifies that account without a
    browser password or a public, retrievable API token.
    """
    try:
        client_ip, client_port = handler.client_address[:2]
        server_ip, server_port = handler.connection.getsockname()[:2]
        address = lambda ip, port: socket.inet_aton(ip)[::-1].hex().upper() + ':' + f'{port:04X}'
        client, server = address(client_ip, client_port), address(server_ip, server_port)
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1] == client and fields[2] == server:
                return int(fields[7]) == os.getuid()
    except (OSError, ValueError, IndexError):
        pass
    return False


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_bytes(path, data):
    """Replace the directory entry, never write through a hard-linked inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('File is outside its configured image root.')
    return path


def file_sha(path):
    return sha(Path(path).read_bytes()) if Path(path).exists() else None


def image_bytes(raw, box=None, image_format='JPEG'):
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if box is not None:
            x, y, w, h = box
            left, top = math.floor(x * image.width), math.floor(y * image.height)
            right, bottom = math.ceil((x + w) * image.width), math.ceil((y + h) * image.height)
            image = image.crop((left, top, min(right, image.width), min(bottom, image.height)))
        result = io.BytesIO()
        image.save(result, format=image_format, **({'quality': 95, 'subsampling': 0} if image_format == 'JPEG' else {}))
        return result.getvalue(), image.size


def clean_box(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('Draw a valid rectangle first.')
    x, y, w, h = value
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
        raise ValueError('The rectangle must be inside the image.')
    return [float(x), float(y), min(float(w), 1 - x), min(float(h), 1 - y)]


class CorrectionStore:
    def __init__(self, review):
        self.review = review
        self.root = review.output / 'corrections'
        self.root.mkdir(exist_ok=True)
        for name in ('previews', 'transactions', 'backups', 'faces'):
            (self.root / name).mkdir(exist_ok=True)
        self.path = self.root / 'state.json'
        self.lock = threading.RLock()
        registry = json.loads(Path(review.inventory['registry_path']).read_text())
        self.identities = registry.get('identities') or {r['identity']: {} for r in review.records.values()}
        self.source_root = Path(review.inventory['source_root']).resolve()
        self.crop_root = Path(review.inventory['crop_root']).resolve()
        self.crop_manifest = {r['path']: r for rows in json.loads(Path(review.inventory['manifest_path']).read_text()).values() for r in rows}
        self.source_manifest = {}
        source_manifest = review.inventory.get('source_manifest_path')
        self.full_root = Path(source_manifest).parent.resolve() if source_manifest else None
        if source_manifest:
            self.source_manifest = {r['path']: r for rows in json.loads(Path(source_manifest).read_text()).values() for r in rows}
        with self.guard():
            self.recover()

    @contextlib.contextmanager
    def guard(self):
        with self.lock, (self.root / 'operations.lock').open('a+b') as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Conflict('Another correction is being saved; try again in a moment.') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self):
        state = json.loads(self.path.read_text()) if self.path.exists() else {
            'schema_version': 1, 'dataset_id': self.review.inventory['dataset_id'],
            'revision': 0, 'entries': {}, 'commits': [],
        }
        if state['dataset_id'] != self.review.inventory['dataset_id']:
            raise ValueError('Correction workspace belongs to another dataset.')
        return state

    def check_dataset(self, payload):
        if not isinstance(payload, dict) or payload.get('dataset_id') != self.review.inventory['dataset_id']:
            raise ValueError('Correction dataset does not match this review.')

    def record(self, key):
        if key not in self.review.records:
            raise ValueError('Unknown review image.')
        return self.review.records[key]

    def effects(self, key, state=None):
        state = state or self.read()
        result = {'faces': {}, 'drop_original': False, 'source_path': self.record(key)['source_photo'],
                  'copies': {}, 'face_pending': False, 'decision': None, 'current_face': None}
        for txn_id in state['entries'].get(key, {}).get('history', []):
            txn = json.loads((self.root / 'transactions' / (txn_id + '.json')).read_text())
            effect = txn['effect']
            for field in ('drop_original', 'source_path', 'face_pending', 'decision'):
                if field in effect:
                    result[field] = effect[field]
            if effect.get('clear_faces'):
                result['faces'] = {}
            if effect.get('remove_face_identity'):
                result['faces'].pop(effect['remove_face_identity'], None)
            if effect.get('face'):
                result['faces'][effect['face']['identity']] = effect['face']
                result['current_face'] = effect['face']
            if effect.get('copy'):
                result['copies']['copy_' + txn_id] = effect['copy']
        return result

    def variants(self, key, state=None):
        r = self.record(key)
        effects = self.effects(key, state)
        paths = {}
        if effects['source_path']:
            paths['master'] = ('Original folder photo', confined(self.source_root, effects['source_path']))
        source_path = r.get('source_path') or self.crop_manifest.get(r['crop_path'], {}).get('source_path')
        if self.full_root and source_path:
            paths['full'] = ('Preserved full source before face cropping', confined(self.full_root, self.full_root / source_path))
        for name, item in effects['copies'].items():
            paths[name] = ('Copied original — ' + item['identity'], confined(self.source_root, item['path']))
        # Offline boxes refer to the export-time photograph, even if a previous
        # imported operation has since cropped or moved its live original.
        index = self.root / 'offline_inputs' / 'index.json'
        if index.exists():
            for name, item in json.loads(index.read_text()).get(key, {}).items():
                paths[name] = (item['label'], confined(index.parent, item['path']))
        return paths

    def selected_image(self, key, variant, state=None):
        if variant == 'face':
            effects = self.effects(key, state)
            if effects['current_face']:
                return confined(self.root, effects['current_face']['path'])
            record = self.record(key)
            if not record.get('crop_absolute'):
                raise ValueError('This original has no existing face crop. Choose Recrop face only.')
            return confined(self.crop_root, record['crop_absolute'])
        paths = self.variants(key, state)
        if variant not in paths:
            raise ValueError('Unknown source image version.')
        return paths[variant][1]

    def metadata(self, key):
        with self.guard():
            state = self.read()
            variants = []
            for name, (label, path) in self.variants(key, state).items():
                item = {'id': name, 'label': label, 'path': str(path), 'available': path.is_file()}
                if path.is_file():
                    raw = path.read_bytes()
                    with Image.open(io.BytesIO(raw)) as image:
                        oriented = ImageOps.exif_transpose(image)
                        item.update(sha256=sha(raw), width=oriented.width, height=oriented.height)
                variants.append(item)
            entry = state['entries'].get(key, {})
            return {'record': self.record(key), 'review': self.review.state['annotations'].get(key, {}),
                    'variants': variants, 'draft': entry.get('draft'), 'history': entry.get('history', []),
                    'draft_applied': entry.get('draft_applied', False),
                    'effects': self.effects(key, state), 'revision': state['revision']}

    def context(self):
        with self.guard():
            state = self.read()
            records = []
            for key, r in self.review.records.items():
                a = self.review.state['annotations'].get(key, {})
                filename = Path(r['crop_path'] or r['source_path']).name
                normalized = re.sub('[_-]+', ' ', re.sub(r'_crop\d+$', '', Path(filename).stem, flags=re.I))
                uncertain = bool(re.search(r'\b(?:unsure|likely|maybe|unclear|or|uncertain|possibly|probably)\b|\?', normalized, re.I))
                history = state['entries'].get(key, {}).get('history', [])
                records.append({'key': key, 'identity': r['identity'], 'split': r['split'], 'filename': filename,
                                'status': a.get('status', 'unreviewed'), 'notes': a.get('notes', ''),
                                'uncertain_filename': uncertain, 'applied': bool(history),
                                'has_draft': bool(state['entries'].get(key, {}).get('draft')) and not state['entries'].get(key, {}).get('draft_applied', False)})
            return {'dataset_id': state['dataset_id'], 'revision': state['revision'], 'records': records,
                    'identities': sorted(self.identities), 'actions': ACTIONS}

    def validate_draft(self, key, draft):
        self.record(key)
        if not isinstance(draft, dict) or draft.get('action') not in ACTIONS:
            raise ValueError('Select a correction action.')
        if draft.get('target_identity') not in self.identities:
            raise ValueError('Choose an identity from the registry.')
        note = draft.get('notes', '')
        if not isinstance(note, str) or len(note) > 5000:
            raise ValueError('Correction notes must be at most 5,000 characters.')
        variant = draft.get('variant', 'master')
        if not isinstance(variant, str) or len(variant) > 60:
            raise ValueError('Invalid source version.')
        return {'action': draft['action'], 'target_identity': draft['target_identity'], 'variant': variant,
                'source_sha256': draft.get('source_sha256', ''), 'box': clean_box(draft.get('box')),
                'notes': note, 'keep_existing_face': draft.get('keep_existing_face') is True}

    def save_draft(self, payload):
        self.check_dataset(payload)
        key = payload.get('key')
        draft = self.validate_draft(key, payload.get('draft'))
        with self.guard():
            state = self.read()
            if payload.get('base_revision') != state['revision']:
                raise Conflict('Corrections changed in another tab. Reload before saving.')
            state['entries'].setdefault(key, {'history': []}).update(draft=draft, draft_applied=False)
            state['revision'] += 1
            atomic_json(self.path, state)
            return {'revision': state['revision'], 'saved_at': now()}

    def destination(self, key, target, name, token):
        directory = self.identities[target].get('source_directory')
        if not directory:
            raise ValueError('This identity has no registered original-folder destination.')
        result = confined(self.source_root, self.source_root / directory / Path(name).name)
        if result.exists():
            result = result.with_name(result.stem + '__corr_' + key[:8] + '_' + token[:6] + result.suffix)
        if result.exists():
            raise Conflict('Destination collision. Make a fresh preview.')
        return result

    def preview(self, payload):
        self.check_dataset(payload)
        key = payload.get('key')
        draft = self.validate_draft(key, payload.get('draft'))
        with self.guard():
            state = self.read()
            r, action, target = self.record(key), draft['action'], draft['target_identity']
            effects = self.effects(key, state)
            token = uuid.uuid4().hex
            folder = self.root / 'previews' / token
            folder.mkdir()
            plan = {'id': token, 'key': key, 'draft': draft, 'created_at': now(),
                    'base_revision': state['revision'], 'inputs': {}, 'operations': [], 'effect': {}, 'warnings': []}

            def read(path):
                raw = Path(path).read_bytes()
                plan['inputs'][str(path)] = sha(raw)
                return raw

            def write(path, data, description):
                path = Path(path)
                before = file_sha(path)
                if before is not None:
                    plan['inputs'][str(path)] = before
                payload_name = str(len(plan['operations'])) + '.bin'
                atomic_bytes(folder / payload_name, data)
                plan['operations'].append({'path': str(path), 'before_sha256': before,
                                           'after_sha256': sha(data), 'payload': str(folder / payload_name),
                                           'description': description})

            source_raw = None
            if action not in ('keep', 'exclude', 'defer'):
                source_path = self.selected_image(key, draft['variant'], state)
                source_raw = read(source_path)
                if sha(source_raw) != draft['source_sha256']:
                    raise Conflict('Source image changed. Reload it and redraw the box.')
            if action == 'face_crop' or action in RELABEL_ACTIONS:
                if action == 'face_crop':
                    if not draft['box']:
                        raise ValueError('Draw a face box before previewing.')
                    face_raw, size = image_bytes(source_raw, draft['box'])
                else:
                    face_raw = read(self.selected_image(key, 'face', state))
                    with Image.open(io.BytesIO(face_raw)) as image:
                        size = ImageOps.exif_transpose(image).size
                if min(size) < 2:
                    raise ValueError('The face crop is too small.')
                # Do not introduce an identical face under two conflicting IDs.
                if action == 'face_crop' and draft['keep_existing_face'] and r.get('crop_absolute') and target != r['identity']:
                    if sha(face_raw) == file_sha(r['crop_absolute']):
                        raise ValueError('The same face pixels cannot be kept under two identities.')
                dest = self.root / 'faces' / token / r['split'] / target / (key + ('.jpg' if action == 'face_crop' else Path(r['crop_path']).suffix or '.jpg'))
                write(dest, face_raw, 'Write face crop under ' + target)
                plan['effect'].update(face={'identity': target, 'path': str(dest), 'sha256': sha(face_raw),
                                            'width': size[0], 'height': size[1], 'source_group': key,
                                            'source_image': str(source_path), 'source_sha256': sha(source_raw),
                                            'face_box': draft['box'] if action == 'face_crop' else None,
                                            'provenance': 'manual_face_box' if action == 'face_crop' else 'retained_face_pixels'},
                                      face_pending=False, decision='corrected')
                if action == 'face_crop':
                    plan['effect'].update(clear_faces=not draft['keep_existing_face'],
                                          drop_original=effects['drop_original'] or not draft['keep_existing_face'])
                else:
                    plan['effect']['drop_original'] = effects['drop_original'] or effects['current_face'] is None
                    if effects['current_face']:
                        plan['effect']['remove_face_identity'] = effects['current_face']['identity']
                atomic_bytes(folder / 'preview.jpg', image_bytes(face_raw)[0])
                plan['face_image_url'] = '/api/corrections/preview/' + token + '/image'
            if action in SOURCE_ACTIONS:
                master = confined(self.source_root, effects['source_path'])
                master_raw = read(master)
                if action == 'edit_source' and not draft['box']:
                    raise ValueError('Draw a region for the original-folder photo.')
                if action == 'edit_source':
                    directory = self.identities[target].get('source_directory')
                    if directory and master.parent != confined(self.source_root, self.source_root / directory):
                        raise ValueError('Use Move original when changing its identity folder.')
                    dest = master
                else:
                    dest = self.destination(key, target, master.name, token)
                expected_format = Image.registered_extensions().get(dest.suffix.lower(), 'JPEG')
                with Image.open(io.BytesIO(source_raw)) as original:
                    same_format = original.format == expected_format
                output_raw = image_bytes(source_raw, draft['box'], expected_format)[0] if draft['box'] or not same_format else source_raw
                write(dest, output_raw, ('Copy' if action in ('copy_source', 'relabel_copy') else 'Save/move') + ' original under ' + target)
                atomic_bytes(folder / 'source_preview.jpg', image_bytes(output_raw)[0])
                plan['source_image_url'] = '/api/corrections/preview/' + token + '/source-image'
                if action in ('copy_source', 'relabel_copy') and not draft['box']:
                    plan['warnings'].append('This copies the full photograph. Draw a source region if you want to isolate one individual.')
                if action in ('move_source', 'relabel_move') and dest != master:
                    plan['operations'].append({'path': str(master), 'before_sha256': sha(master_raw),
                                               'after_sha256': None, 'description': 'Remove old-folder original after backing it up'})
                if action in ('copy_source', 'relabel_copy'):
                    plan['effect']['copy'] = {'identity': target, 'path': str(dest), 'sha256': sha(output_raw)}
                else:
                    plan['effect']['source_path'] = str(dest)
                if action not in RELABEL_ACTIONS:
                    plan['effect']['face_pending'] = True
                    plan['warnings'].append('Original-folder change saved separately. Create or verify the face crop before rebuilding training data.')
                if not (folder / 'preview.jpg').exists():
                    atomic_bytes(folder / 'preview.jpg', image_bytes(output_raw)[0])
            if action in ('keep', 'exclude', 'defer'):
                plan['effect']['decision'] = action
            plan['image_url'] = '/api/corrections/preview/' + token + '/image' if (folder / 'preview.jpg').exists() else None
            plan['action_label'] = ACTIONS[action]
            plan['target_identity'] = target
            plan['backup_directory'] = str(self.root / 'backups' / token)
            atomic_json(folder / 'plan.json', plan)
            return plan

    def check_operation_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.source_root) and not path.is_relative_to(self.root / 'faces'):
            raise ValueError('Correction operation points outside its allowed roots.')
        return path

    def restore_operations(self, operations):
        for operation in reversed(operations):
            path = self.check_operation_path(operation['path'])
            current = file_sha(path)
            if current == operation['before_sha256']:
                continue
            if current != operation['after_sha256']:
                raise Conflict('A corrected file changed externally; automatic undo cannot overwrite it: ' + str(path))
            if operation['before_sha256'] is None:
                path.unlink()
            else:
                backup = confined(self.root / 'backups', operation['backup'])
                data = backup.read_bytes()
                if sha(data) != operation['before_sha256']:
                    raise ValueError('Backup hash does not match.')
                atomic_bytes(path, data)

    def recover(self):
        state = self.read()
        for path in (self.root / 'transactions').glob('*.json'):
            txn = json.loads(path.read_text())
            if txn['status'] not in ('prepared', 'recovery_required'):
                continue
            if txn['id'] in state['commits']:
                txn['status'] = 'committed'
            else:
                self.restore_operations(txn['operations'])
                txn['status'] = 'rolled_back'
            atomic_json(path, txn)

    def commit(self, state, key, txn, history_after):
        journal = self.root / 'transactions' / (txn['id'] + '.json')
        backups = self.root / 'backups' / txn['id']
        backups.mkdir(exist_ok=True)
        for i, op in enumerate(txn['operations']):
            path = self.check_operation_path(op['path'])
            if file_sha(path) != op['before_sha256']:
                raise Conflict('A file changed since preview. Generate a new preview: ' + str(path))
            if op['before_sha256'] is not None:
                op['backup'] = str(backups / (str(i) + '.bin'))
                atomic_bytes(Path(op['backup']), path.read_bytes())
        txn['status'] = 'prepared'
        atomic_json(journal, txn)
        try:
            for op in txn['operations']:
                path = self.check_operation_path(op['path'])
                if file_sha(path) != op['before_sha256']:
                    raise Conflict('A file changed during apply: ' + str(path))
                if op['after_sha256'] is None:
                    path.unlink()
                else:
                    payload = Path(op['payload']).read_bytes()
                    if sha(payload) != op['after_sha256']:
                        raise ValueError('Preview payload changed.')
                    atomic_bytes(path, payload)
            state['entries'][key] = {'history': history_after, 'draft': txn.get('draft'),
                                     'draft_applied': not bool(txn.get('undo_of'))}
            state['commits'].append(txn['id'])
            state['revision'] += 1
            atomic_json(self.path, state)
        except Exception:
            try:
                self.restore_operations(txn['operations'])
                txn['status'] = 'rolled_back'
            except Exception:
                txn['status'] = 'recovery_required'
                atomic_json(journal, txn)
                raise
            atomic_json(journal, txn)
            raise
        txn['status'] = 'committed'
        atomic_json(journal, txn)
        return {'revision': state['revision'], 'transaction': txn['id'], 'saved_at': now()}

    def apply(self, payload):
        self.check_dataset(payload)
        token = payload.get('token', '')
        if not re.fullmatch('[a-f0-9]{32}', token):
            raise ValueError('Invalid preview token.')
        with self.guard():
            self.recover()
            state = self.read()
            plan = json.loads((self.root / 'previews' / token / 'plan.json').read_text())
            if token in state['commits']:
                raise Conflict('This preview has already been applied.')
            if plan['base_revision'] != state['revision']:
                raise Conflict('Corrections changed after preview. Make a fresh preview.')
            for path, expected in plan['inputs'].items():
                if file_sha(path) != expected:
                    raise Conflict('A source changed after preview. Reload the image.')
            history = state['entries'].get(plan['key'], {}).get('history', []) + [token]
            return self.commit(state, plan['key'], {**plan, 'applied_at': now()}, history)

    def undo(self, payload):
        self.check_dataset(payload)
        key = payload.get('key')
        self.record(key)
        with self.guard():
            self.recover()
            state = self.read()
            if payload.get('base_revision') != state['revision']:
                raise Conflict('Corrections changed in another tab. Reload before undo.')
            history = state['entries'].get(key, {}).get('history', [])
            if not history:
                raise ValueError('This image has no applied correction to undo.')
            previous = json.loads((self.root / 'transactions' / (history[-1] + '.json')).read_text())
            token = uuid.uuid4().hex
            folder = self.root / 'previews' / token
            folder.mkdir()
            operations = []
            for i, old in enumerate(reversed(previous['operations'])):
                if file_sha(old['path']) != old['after_sha256']:
                    raise Conflict('A file changed after this correction. Undo would overwrite an external edit.')
                op = {'path': old['path'], 'before_sha256': old['after_sha256'], 'after_sha256': old['before_sha256']}
                if old['before_sha256'] is not None:
                    data = Path(old['backup']).read_bytes()
                    if sha(data) != old['before_sha256']:
                        raise ValueError('Undo backup changed.')
                    op['payload'] = str(folder / (str(i) + '.bin'))
                    atomic_bytes(Path(op['payload']), data)
                operations.append(op)
            txn = {'id': token, 'key': key, 'undo_of': previous['id'], 'operations': operations,
                   'effect': {}, 'draft': previous['draft'], 'created_at': now()}
            return self.commit(state, key, txn, history[:-1])

    def image(self, key, variant):
        with self.guard():
            raw = self.selected_image(key, variant).read_bytes()
        return image_bytes(raw)[0]

    def handle_get(self, handler, route, query):
        if not own_user_connection(handler):
            return handler.send_json({'error': 'Use SSH/VS Code port forwarding from the same CSD3 user account as this server.'}, 403)
        try:
            if route == '/corrections':
                return handler.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            if route == '/api/corrections/context':
                return handler.send_json(self.context())
            if route == '/api/corrections/export':
                with self.guard():
                    data = self.read()
                    data['applied_effects'] = {key: self.effects(key, data) for key, entry in data['entries'].items() if entry.get('history')}
                    data['transactions'] = [json.loads((self.root / 'transactions' / (txn_id + '.json')).read_text()) for txn_id in data['commits']]
                return handler.send(json.dumps(data, indent=2).encode(), 'application/json', filename='corrections_backup.json')
            if route.startswith('/api/corrections/record/'):
                return handler.send_json(self.metadata(route.rsplit('/', 1)[1]))
            if route.startswith('/api/corrections/image/'):
                key = route.rsplit('/', 1)[1]
                variant = parse_qs(query).get('variant', ['master'])[0]
                return handler.send(self.image(key, variant), 'image/jpeg')
            match = re.fullmatch(r'/api/corrections/preview/([a-f0-9]{32})/(image|source-image)', route)
            if match:
                path = self.root / 'previews' / match.group(1) / ('source_preview.jpg' if match.group(2) == 'source-image' else 'preview.jpg')
                return handler.send(path.read_bytes(), 'image/jpeg')
            return handler.send_json({'error': 'Unknown correction route.'}, 404)
        except Conflict as exc:
            return handler.send_json({'error': str(exc)}, 409)
        except (ValueError, KeyError, OSError) as exc:
            return handler.send_json({'error': str(exc)}, 400)

    def handle_post(self, handler, route, payload):
        if not own_user_connection(handler):
            return handler.send_json({'error': 'Correction requests must come through your own CSD3 account.'}, 403)
        try:
            action = {'/api/corrections/draft': self.save_draft, '/api/corrections/preview': self.preview,
                      '/api/corrections/apply': self.apply, '/api/corrections/undo': self.undo}.get(route)
            if action is None:
                return handler.send_json({'error': 'Unknown correction route.'}, 404)
            return handler.send_json(action(payload))
        except Conflict as exc:
            return handler.send_json({'error': str(exc)}, 409)
        except (ValueError, KeyError, OSError) as exc:
            return handler.send_json({'error': str(exc)}, 400)
