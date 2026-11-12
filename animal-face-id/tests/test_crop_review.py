"""Review integrity: complete inventory, exact-data binding, durable decisions."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/review_macaque_crops.py'
SPEC = importlib.util.spec_from_file_location('crop_review', MODULE_PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class CropReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.crops, self.sources, self.output = root / 'crops', root / 'sources', root / 'review'
        self.sources.mkdir()
        manifest = {'train': [], 'val': [], 'test': []}
        for i, split in enumerate(manifest):
            relative = f'{split}/A/source {i}.jpg'
            path = self.crops / relative
            path.parent.mkdir(parents=True)
            Image.new('RGB', (45 + i, 50), (10, i * 40, 90)).save(path)
            (self.sources / path.name).write_bytes(path.read_bytes())
            manifest[split].append({'id': 'A', 'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'master_source_path': path.name, 'source_path': relative})
        self.manifest = root / 'manifest.json'
        self.registry = root / 'registry.json'
        self.manifest.write_text(json.dumps(manifest))
        self.registry.write_text(json.dumps({'source_root': str(self.sources)}))
        self.args = argparse.Namespace(manifest=self.manifest, registry=self.registry, crops=self.crops, output=self.output, workers=2)
        with contextlib.redirect_stdout(io.StringIO()):
            review.build(self.args)
        self.store = review.ReviewStore(self.output)
        self.key = next(iter(self.store.records))

    def payload(self, changes, revision=0):
        return {'dataset_id': self.store.inventory['dataset_id'], 'base_revision': revision, 'changes': changes}

    def add_source_fixture(self):
        source_manifest = json.loads(self.manifest.read_text())
        path = self.sources / 'A' / 'uncropped original.jpg'
        path.parent.mkdir()
        Image.new('RGB', (180, 130), (90, 130, 80)).save(path)
        source_manifest['val'].append({
            'id': 'A', 'path': 'val/A/uncropped original.jpg', 'master_source_path': 'A/uncropped original.jpg',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        source_manifest_path = Path(self.temp.name) / 'source_manifest.json'
        source_manifest_path.write_text(json.dumps(source_manifest))
        args = argparse.Namespace(output=self.output, source_manifest=source_manifest_path, workers=2)
        with contextlib.redirect_stdout(io.StringIO()):
            review.add_sources(args)
        return args

    def test_add_uncropped_original_preserves_existing_review_bytes_and_keys(self):
        self.store.update(self.payload({self.key: {'status': 'wrong_animal', 'notes': 'Keep my existing review'}}))
        before = self.store.path.read_bytes()
        keys = set(self.store.records)
        self.add_source_fixture()
        store = review.ReviewStore(self.output)
        self.assertEqual(store.path.read_bytes(), before)
        self.assertEqual(store.inventory['dataset_id'], self.store.inventory['dataset_id'])
        self.assertTrue(keys < set(store.records))
        originals = [r for r in store.records.values() if r['record_type'] == 'source_only']
        self.assertEqual(len(originals), 1)
        original = originals[0]
        self.assertEqual(original['identity'], 'A')
        self.assertEqual(original['split'], 'val')
        self.assertEqual(original['crop_path'], '')
        self.assertEqual(original['crop_absolute'], '')
        self.assertEqual((original['width'], original['height']), (180, 130))
        self.assertTrue((self.output / original['thumbnail']).is_file())
        result = store.update(self.payload({original['key']: {'status': 'not_croppable', 'notes': 'Face is turned away'}}, revision=1))
        self.assertEqual(result['revision'], 2)
        self.assertEqual(store.state['annotations'][self.key]['notes'], 'Keep my existing review')

    def test_adding_originals_is_idempotent_and_rebuild_keeps_their_reviews(self):
        args = self.add_source_fixture()
        store = review.ReviewStore(self.output)
        key = next(k for k, r in store.records.items() if r['record_type'] == 'source_only')
        store.update(self.payload({key: {'status': 'uncertain_identity', 'notes': 'Ask Sylvain'}}))
        before = store.path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            review.add_sources(args)
            review.build(self.args)
        restored = review.ReviewStore(self.output)
        self.assertEqual(len(restored.records), 4)
        self.assertEqual(restored.path.read_bytes(), before)
        self.assertIn(key, restored.records)

    def test_cannot_publish_inventory_that_drops_reviewed_original(self):
        args = self.add_source_fixture()
        store = review.ReviewStore(self.output)
        key = next(k for k, r in store.records.items() if r['record_type'] == 'source_only')
        store.update(self.payload({key: {'status': 'not_croppable'}}))
        before = (self.output / 'inventory.json').read_bytes()
        source_manifest = json.loads(args.source_manifest.read_text())
        source_manifest['val'] = source_manifest['val'][:1]
        args.source_manifest.write_text(json.dumps(source_manifest))
        with self.assertRaisesRegex(ValueError, 'invalidate saved reviews'):
            review.add_sources(args)
        self.assertEqual((self.output / 'inventory.json').read_bytes(), before)

    def test_original_http_route_and_export_do_not_invent_a_crop(self):
        self.add_source_fixture()
        store = review.ReviewStore(self.output)
        original = next(r for r in store.records.values() if r['record_type'] == 'source_only')
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler_for(store))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base + '/source/' + original['key']) as response:
            self.assertEqual(hashlib.sha256(response.read()).hexdigest(), original['sha256'])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(base + '/crop/' + original['key'])
        self.assertEqual(caught.exception.code, 404)
        rows = list(csv.DictReader(io.StringIO(store.csv_bytes().decode('utf-8-sig'))))
        row = next(r for r in rows if r['record_type'] == 'source_only')
        self.assertEqual(row['identity'], 'A')
        self.assertEqual(row['crop_path'], '')
        self.assertEqual(row['source_path'], 'val/A/uncropped original.jpg')
        self.assertEqual(row['status'], 'unreviewed')

    def test_inventory_includes_every_split_and_links_sources(self):
        records = list(self.store.records.values())
        self.assertEqual({r['split'] for r in records}, {'train', 'val', 'test'})
        self.assertTrue(all(r['source_exists'] and r['width'] > 0 for r in records))
        self.assertTrue(all((self.output / r['thumbnail']).is_file() for r in records))
        self.assertEqual(len(self.store.records), 3)

    def test_changed_crop_hash_is_rejected_without_changing_review(self):
        record = self.store.records[self.key]
        Path(record['crop_absolute']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            review.build(self.args)
        self.assertEqual(review.ReviewStore(self.output).inventory['dataset_id'], self.store.inventory['dataset_id'])

    def test_missing_crop_is_visible_instead_of_omitted(self):
        Path(self.store.records[self.key]['crop_absolute']).unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            review.build(self.args)
        store = review.ReviewStore(self.output)
        self.assertEqual(len(store.records), 3)
        self.assertTrue(store.records[self.key]['load_error'])

    def test_annotations_survive_restart_and_stale_updates_cannot_overwrite(self):
        changes = {self.key: {'status': 'uncertain_identity', 'notes': 'Ask Sylvain', 'suggested_identity': 'B'}}
        self.assertEqual(self.store.update(self.payload(changes))['revision'], 1)
        self.assertIsNone(self.store.update(self.payload({self.key: {'status': 'ok'}})))
        restored = review.ReviewStore(self.output)
        self.assertEqual(restored.state['annotations'][self.key]['notes'], 'Ask Sylvain')
        self.assertEqual(restored.state['annotations'][self.key]['status'], 'uncertain_identity')

    def test_invalid_batch_does_not_partially_save(self):
        changes = {self.key: {'status': 'ok'}, 'unknown': {'status': 'wrong_label'}}
        with self.assertRaises(ValueError):
            self.store.update(self.payload(changes))
        self.assertEqual(self.store.state['revision'], 0)
        self.assertFalse(self.store.path.exists())
        with self.assertRaises(ValueError):
            self.store.update({**self.payload({}), 'dataset_id': 'wrong dataset'})

    def test_exports_include_unreviewed_and_preserve_notes(self):
        self.store.update(self.payload({self.key: {'status': 'wrong_animal', 'notes': '=not a formula,\nsecond line'}}))
        all_rows = list(csv.DictReader(io.StringIO(self.store.csv_bytes().decode('utf-8-sig'))))
        flagged_rows = list(csv.DictReader(io.StringIO(self.store.csv_bytes(True).decode('utf-8-sig'))))
        self.assertEqual(len(all_rows), 3)
        self.assertEqual(sum(r['status'] == 'unreviewed' for r in all_rows), 2)
        self.assertEqual(len(flagged_rows), 1)
        self.assertEqual(flagged_rows[0]['notes'], "'=not a formula,\nsecond line")

    def test_http_save_images_exports_and_path_restrictions(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler_for(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        for route in ['/', f'/crop/{self.key}', f'/source/{self.key}', '/api/review']:
            with urllib.request.urlopen(base + route) as response:
                self.assertEqual(response.status, 200)
        data = json.dumps(self.payload({self.key: {'status': 'ok'}})).encode()
        request = urllib.request.Request(base + '/api/review', data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(json.load(response)['revision'], 1)
        with urllib.request.urlopen(base + '/api/export/all.csv') as response:
            self.assertIn(b'unreviewed', response.read())
        for route in ['/crop/../../registry.json', '/source/unknown', '/review_state.json', '/thumbs/%2e%2e']:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + route)
            self.assertEqual(caught.exception.code, 404)

    def start_review_process(self, port):
        process = subprocess.Popen(
            [sys.executable, str(MODULE_PATH), 'serve', '--output', str(self.output), '--port', str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        def cleanup():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdout.close()

        self.addCleanup(cleanup)
        return process

    def wait_for_review(self, process, port):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail('Review server exited: ' + process.stdout.read())
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/review', timeout=0.5) as response:
                    return json.load(response)
            except (OSError, ValueError):
                time.sleep(0.05)
        self.fail('Timed out waiting for review server.')

    def test_restart_replaces_previous_process_and_reloads_saved_decisions(self):
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        old = self.start_review_process(port)
        self.wait_for_review(old, port)
        changes = {self.key: {'status': 'uncertain_identity', 'notes': 'Keep this decision across restart'}}
        request = urllib.request.Request(
            f'http://127.0.0.1:{port}/api/review', data=json.dumps(self.payload(changes)).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(json.load(response)['revision'], 1)
        replacement = self.start_review_process(port)
        old.wait(timeout=12)
        self.assertEqual(old.returncode, 0)
        state = self.wait_for_review(replacement, port)
        self.assertEqual(state['revision'], 1)
        self.assertEqual(state['annotations'][self.key]['notes'], 'Keep this decision across restart')
        self.assertIn('Stopping previous crop-review server', old.stdout.read() + self.read_startup_log(replacement))

    @staticmethod
    def read_startup_log(process):
        # The replacement prints the stop message before its ready message.
        lines = []
        for _ in range(2):
            lines.append(process.stdout.readline())
        return ''.join(lines)

    def test_unrelated_listener_is_left_running_and_next_port_is_used(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler_for(self.store))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        args = argparse.Namespace(output=self.output, port=server.server_port)
        replacement = review.open_review_server(args)
        self.addCleanup(replacement.server_close)
        self.assertGreater(args.port, server.server_port)
        self.assertEqual(args.port, replacement.server_port)
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/api/review') as response:
            self.assertEqual(response.status, 200)

    def test_fallback_port_replaces_own_previous_server_and_keeps_progress(self):
        # Reserve two consecutive ports for the foreign service and old review.
        for _ in range(30):
            server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler_for(self.store))
            port = server.server_port
            try:
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port + 1))
            except OSError:
                server.server_close()
                continue
            break
        else:
            self.fail('Could not find two consecutive test ports.')
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        old = self.start_review_process(port + 1)
        self.wait_for_review(old, port + 1)
        replacement = self.start_review_process(port)
        old.wait(timeout=12)
        self.assertEqual(old.returncode, 0)
        state = self.wait_for_review(replacement, port + 1)
        self.assertEqual(state['dataset_id'], self.store.inventory['dataset_id'])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/review') as response:
            self.assertEqual(response.status, 200)


if __name__ == '__main__':
    unittest.main()
