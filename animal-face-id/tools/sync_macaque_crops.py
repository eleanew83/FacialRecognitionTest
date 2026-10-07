#!/usr/bin/env python3
"""Validate crops, transfer into a fresh staging directory, then publish atomically."""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.split_integrity import SPLIT_ORDER, validate_manifest, verified_fingerprint

# Runs locally or over SSH using only the receiver's Python standard library.
# Input images have already been checked for exact RGB duplicates at the sender;
# matching every received SHA-256 guarantees those same pixels arrive intact.
RECEIVER = r'''
import hashlib
import json
import os
import sys

phase, target, stage = sys.argv[1:]
lock = os.path.join(os.path.dirname(target), '.' + os.path.basename(target) + '.transfer.lock')

def check_target():
    if os.path.lexists(target) and (os.path.islink(target) or not os.path.isdir(target) or os.listdir(target)):
        raise RuntimeError('Destination is not empty; choose a fresh dataset directory: ' + target)

try:
    check_target()
    if phase == 'reserve':
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise RuntimeError('A transfer is active or incomplete; inspect its lock before retrying: ' + lock)
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(stage)
        os.mkdir(stage)
        print('Reserved fresh staging directory: ' + stage)
    elif phase == 'publish':
        with open(lock) as handle:
            if handle.read() != stage:
                raise RuntimeError('Transfer lock does not belong to this generation.')
        payload = json.load(sys.stdin)
        expected = {}
        for split in ('train', 'val', 'test'):
            for sample in payload[split]:
                relative = sample['path']
                parts = relative.split('/')
                if os.path.isabs(relative) or '..' in parts or len(parts) != 3 or parts[:2] != [split, sample['id']]:
                    raise RuntimeError('Unsafe transferred manifest path: ' + relative)
                if relative in expected:
                    raise RuntimeError('Repeated transferred manifest path: ' + relative)
                expected[relative] = sample
        actual = set()
        for folder, dirs, files in os.walk(stage):
            dirs[:] = [name for name in dirs if not name.startswith('.')]
            for name in files:
                if not name.startswith('.') and os.path.splitext(name)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    actual.add(os.path.relpath(os.path.join(folder, name), stage))
        if actual != set(expected):
            raise RuntimeError('Received image inventory does not match the source manifest.')
        for relative, sample in expected.items():
            filename = os.path.join(stage, relative)
            if os.path.islink(filename):
                raise RuntimeError('Transferred image is a symbolic link: ' + relative)
            digest = hashlib.sha256()
            with open(filename, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != sample['sha256']:
                raise RuntimeError('Received image hash disagrees with source: ' + relative)
            stat = os.stat(filename)
            sample['file_bytes'], sample['mtime_ns'] = stat.st_size, stat.st_mtime_ns
        with open(os.path.join(stage, 'splits.json'), 'w') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
        with open(os.path.join(stage, 'transfer_verified.json'), 'w') as handle:
            json.dump({'image_count': len(expected), 'byte_hashes_verified': True,
                       'source_exact_pixel_duplicates': 0}, handle, indent=2)
            handle.write('\n')
        check_target()
        if os.path.exists(target):
            os.rmdir(target)
        os.rename(stage, target)
        os.unlink(lock)
        print('Published verified dataset: ' + target)
    else:
        raise RuntimeError('Unknown receiver phase: ' + phase)
except Exception as error:
    print(str(error), file=sys.stderr)
    sys.exit(2)
'''


def destination_parts(destination: str) -> tuple[str | None, str]:
    if '\n' in destination or '\r' in destination:
        raise ValueError('Destination must be a single path.')
    if ':' in destination:
        host, path = destination.split(':', 1)
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host) or not path.startswith('/'):
            raise ValueError('Remote destination must be [user@]hostname:/absolute/path.')
        return host, posixpath.normpath(path)
    return None, str(Path(destination).resolve())


def receiver_call(host, phase, target, stage, payload=None):
    arguments = ['python3', '-c', RECEIVER, phase, target, stage]
    command = ['ssh', host, shlex.join(arguments)] if host else [sys.executable, *arguments[1:]]
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else '',
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Receiver validation failed.')
    if result.stdout:
        print(result.stdout.strip(), flush=True)


def source_snapshot(source, manifest):
    payload = json.loads(Path(manifest).read_text())
    validate_manifest(payload, source, verify_content=True)
    snapshot = {split: [] for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        for sample in payload[split]:
            fingerprint = verified_fingerprint(Path(source) / sample['path'])
            snapshot[split].append({'id': sample['id'], 'path': sample['path'], **fingerprint})
    return snapshot


def sync_dataset(source, destination, manifest, *, retries=5, retry_delay=10, dry_run=False):
    source = Path(source).resolve()
    if retries < 1 or retry_delay < 0:
        raise ValueError('Retries must be positive and retry delay nonnegative.')
    host, target = destination_parts(destination)
    if target == '/':
        raise ValueError('The filesystem root cannot be a dataset destination.')
    if host is None:
        target_path = Path(target)
        if target_path == source or target_path in source.parents or source in target_path.parents:
            raise ValueError('Source and destination must be separate, non-nested directories.')
    print('Checking source image content and manifest...', flush=True)
    payload = source_snapshot(source, manifest)
    count = sum(len(payload[split]) for split in SPLIT_ORDER)
    print(f'Validated {count} images with no exact duplicates.', flush=True)
    if dry_run:
        print(f'Dry run: would publish to fresh destination {destination}; no connection or transfer performed.')
        return
    stage = str(Path(target).parent / ('.' + Path(target).name + '.incoming-' + uuid.uuid4().hex))
    receiver_call(host, 'reserve', target, stage)
    receiving = (host + ':' if host else '') + stage + '/'
    try:
        for attempt in range(1, retries + 1):
            print(f'Rsync attempt {attempt}/{retries} into isolated staging...', flush=True)
            result = subprocess.run(['rsync', '-az', '--protect-args', '--partial', '--', str(source) + '/', receiving])
            if result.returncode == 0:
                break
            if attempt == retries:
                raise RuntimeError(f'Rsync failed with exit code {result.returncode}.')
            time.sleep(retry_delay)
        # Check the sender again; ctime-aware caching rehashes files changed during transfer.
        validate_manifest(payload, source, verify_content=True)
        receiver_call(host, 'publish', target, stage, payload)
    except BaseException:
        print(f'Transfer was not published. Incoming files and lock remain for inspection: {receiving}', file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', required=True, help='Fresh local directory or user@host:/absolute/path')
    parser.add_argument('--manifest', type=Path, help='Default: source/splits.json, then the project manifest')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--retry-delay', type=float, default=10)
    parser.add_argument('--dry-run', action='store_true', help='Validate source only; do not connect or copy')
    args = parser.parse_args()
    manifest = args.manifest or (args.source / 'splits.json')
    if args.manifest is None and not manifest.is_file():
        manifest = ROOT / 'data/macaque_faces/splits.json'
    try:
        sync_dataset(args.source, args.destination, manifest, retries=args.retries,
                     retry_delay=args.retry_delay, dry_run=args.dry_run)
    except Exception as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
