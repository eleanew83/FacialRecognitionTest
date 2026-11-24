"""Correction transactions must preserve originals/review state and undo cleanly."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review = load('review_macaque_crops')
correction = load('crop_corrections')


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output = root / 'review'
        self.sources, self.crops, self.full = root / 'master', root / 'crops', root / 'full'
        self.master = self.sources / 'Group/females/A/photo.jpg'
        self.master.parent.mkdir(parents=True)
        (self.sources / 'Group/males/B').mkdir(parents=True)
        image = Image.new('RGB', (200, 120), 'white')
        for x in range(100):
            for y in range(120):
                image.putpixel((x, y), (170, 20, 40))
        image.save(self.master)
        crop = self.crops / 'train/A/photo_crop0.jpg'
        crop.parent.mkdir(parents=True)
        image.crop((0, 0, 100, 120)).save(crop)
        self.crop = crop
        self.master_before, self.crop_before = self.master.read_bytes(), crop.read_bytes()
        manifest = {'train': [{'id': 'A', 'path': 'train/A/photo_crop0.jpg',
                    'sha256': hashlib.sha256(self.crop_before).hexdigest(),
                    'source_path': 'train/A/photo.jpg', 'master_source_path': 'Group/females/A/photo.jpg'}], 'val': [], 'test': []}
        manifest_path = root / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        self.registry = root / 'registry.json'
        self.registry.write_text(json.dumps({'source_root': str(self.sources), 'identities': {
            'A': {'source_directory': 'Group/females/A'}, 'B': {'source_directory': 'Group/males/B'}}}))
        with contextlib.redirect_stdout(io.StringIO()):
            review.build(argparse.Namespace(manifest=manifest_path, registry=self.registry, crops=self.crops,
                                            output=self.output, workers=1))
        self.review = review.ReviewStore(self.output)
        self.key = next(iter(self.review.records))
        self.review.update({'dataset_id': self.review.inventory['dataset_id'], 'base_revision': 0,
                            'changes': {self.key: {'status': 'wrong_label', 'notes': 'Human review is immutable'}}})
        self.review_before = self.review.path.read_bytes()
        self.store = correction.CorrectionStore(self.review)
        self.dataset = self.review.inventory['dataset_id']

    def draft(self, action='face_crop', target='A', box=None):
        meta = self.store.metadata(self.key)
        source = next(v for v in meta['variants'] if v['id'] == 'master')
        return {'action': action, 'target_identity': target, 'variant': 'master',
                'source_sha256': source['sha256'], 'box': box, 'notes': 'Correction note', 'keep_existing_face': False}

    def preview(self, draft):
        return self.store.preview({'dataset_id': self.dataset, 'key': self.key, 'draft': draft})

    def apply(self, plan):
        return self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})

    def undo(self):
        return self.store.undo({'dataset_id': self.dataset, 'key': self.key, 'base_revision': self.store.read()['revision']})

    def test_face_crop_outputs_full_resolution_and_never_edits_source_or_review(self):
        plan = self.preview(self.draft(box=[0.5, 0, 0.5, 1]))
        self.assertFalse(self.store.path.exists())
        self.apply(plan)
        face = self.store.effects(self.key)['current_face']
        with Image.open(face['path']) as image:
            self.assertEqual(image.size, (100, 120))
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(self.crop.read_bytes(), self.crop_before)
        self.assertEqual(self.review.path.read_bytes(), self.review_before)
        self.undo()
        self.assertFalse(Path(face['path']).exists())
        self.assertIsNone(self.store.effects(self.key)['current_face'])

    def test_relabel_only_preserves_exact_crop_pixels_and_original_folder(self):
        plan = self.preview(self.draft('relabel_crop', 'B'))
        self.apply(plan)
        effect = self.store.effects(self.key)
        self.assertEqual(effect['current_face']['identity'], 'B')
        self.assertEqual(Path(effect['current_face']['path']).read_bytes(), self.crop_before)
        self.assertTrue(effect['drop_original'])
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(len(plan['operations']), 1)

    def test_relabel_and_move_original_is_reversible(self):
        plan = self.preview(self.draft('relabel_move', 'B'))
        self.apply(plan)
        effect = self.store.effects(self.key)
        destination = Path(effect['source_path'])
        self.assertFalse(self.master.exists())
        self.assertEqual(destination.parent, self.sources / 'Group/males/B')
        self.assertEqual(destination.read_bytes(), self.master_before)
        self.assertEqual(effect['current_face']['identity'], 'B')
        self.assertEqual(self.review.path.read_bytes(), self.review_before)
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertFalse(destination.exists())

    def test_relabel_and_copy_original_keeps_old_folder_and_creates_target_copy(self):
        plan = self.preview(self.draft('relabel_copy', 'B', [0.5, 0, 0.5, 1]))
        self.assertTrue(plan['face_image_url'] and plan['source_image_url'])
        self.apply(plan)
        effect = self.store.effects(self.key)
        copy_ = next(iter(effect['copies'].values()))
        with Image.open(copy_['path']) as image:
            self.assertEqual(image.size, (100, 120))
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(effect['current_face']['identity'], 'B')
        self.undo()
        self.assertFalse(Path(copy_['path']).exists())
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_original_crop_uses_atomic_replacement_preserving_hardlinked_archive(self):
        historical = self.sources / 'archive.jpg'
        os.link(self.master, historical)
        self.apply(self.preview(self.draft('edit_source', box=[0, 0, 0.5, 1])))
        self.assertEqual(historical.read_bytes(), self.master_before)
        with Image.open(self.master) as image:
            self.assertEqual(image.size, (100, 120))
        self.assertTrue(self.store.effects(self.key)['face_pending'])
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_second_face_relabel_does_not_leave_a_stale_label_or_revive_original(self):
        self.apply(self.preview(self.draft('relabel_crop', 'B')))
        self.apply(self.preview(self.draft('relabel_crop', 'A')))
        effect = self.store.effects(self.key)
        self.assertEqual(set(effect['faces']), {'A'})
        self.assertTrue(effect['drop_original'])
        self.undo()
        self.assertEqual(set(self.store.effects(self.key)['faces']), {'B'})

    def test_face_addition_keeps_original_only_if_it_was_not_already_replaced(self):
        draft = self.draft('face_crop', 'B', [0.5, 0, 0.5, 1])
        draft['keep_existing_face'] = True
        self.apply(self.preview(draft))
        self.assertFalse(self.store.effects(self.key)['drop_original'])
        self.apply(self.preview(self.draft('face_crop', 'A', [0, 0, 0.5, 1])))
        self.assertTrue(self.store.effects(self.key)['drop_original'])
        self.assertEqual(set(self.store.effects(self.key)['faces']), {'A'})

    def test_earlier_full_source_is_distinct_from_master_and_coordinates_follow_orientation(self):
        full_path = self.full / 'train/A/photo.jpg'
        full_path.parent.mkdir(parents=True)
        Image.new('RGB', (400, 240), 'green').save(full_path)
        source_manifest = self.full / 'splits.json'
        source_manifest.write_text(json.dumps({'train': [{'id': 'A', 'path': 'train/A/photo.jpg'}], 'val': [], 'test': []}))
        self.review.inventory['source_manifest_path'] = str(source_manifest)
        store = correction.CorrectionStore(self.review)
        versions = {v['id']: v for v in store.metadata(self.key)['variants']}
        self.assertEqual(versions['master']['width'], 200)
        self.assertEqual(versions['full']['width'], 400)
        exif = Image.Exif();exif[274] = 6
        Image.new('RGB', (200, 120), 'red').save(self.master, exif=exif)
        oriented = store.metadata(self.key)['variants'][0]
        self.assertEqual((oriented['width'], oriented['height']), (120, 200))
        payload = self.draft(box=[0, 0, 0.5, 1])
        payload['source_sha256'] = oriented['sha256']
        plan = store.preview({'dataset_id': self.dataset, 'key': self.key, 'draft': payload})
        self.assertEqual((plan['effect']['face']['width'], plan['effect']['face']['height']), (60, 200))

    def test_destination_collision_does_not_overwrite_other_photo(self):
        other = self.sources / 'Group/males/B/photo.jpg'
        other.write_bytes(b'Another original')
        plan = self.preview(self.draft('move_source', 'B'))
        self.apply(plan)
        dest = Path(self.store.effects(self.key)['source_path'])
        self.assertNotEqual(dest, other)
        self.assertEqual(other.read_bytes(), b'Another original')
        self.undo()
        self.assertEqual(other.read_bytes(), b'Another original')

    def test_preview_rejects_stale_images_and_apply_rechecks_inputs(self):
        draft = self.draft(box=[0, 0, 0.5, 1])
        plan = self.preview(draft)
        self.master.write_bytes(b'Changed outside the editor')
        with self.assertRaises(correction.Conflict):
            self.apply(plan)
        with self.assertRaises(correction.Conflict):
            self.preview(draft)
        self.assertFalse(self.store.path.exists())

    def test_invalid_boxes_ids_and_version_are_rejected(self):
        for box in [[0, 0, 2, 1], [0, 0, float('nan'), 1], [-0.1, 0, 0.5, 0.5], [0, 0, 0, 0.2]]:
            with self.assertRaises(ValueError):
                self.preview(self.draft(box=box))
        with self.assertRaises(ValueError):
            self.preview(self.draft(target='../../outside', box=[0, 0, 1, 1]))
        draft = self.draft(box=[0, 0, 1, 1]);draft['variant'] = '../../other'
        with self.assertRaises(ValueError):
            self.preview(draft)

    def test_failed_move_rolls_back_files_and_does_not_commit_state(self):
        plan = self.preview(self.draft('relabel_move', 'B'))
        real = correction.atomic_bytes
        failed = [False]

        def fail_once(path, data):
            if str(path) == str(self.sources / 'Group/males/B/photo.jpg') and not failed[0]:
                failed[0] = True
                raise OSError('Injected write failure')
            return real(path, data)

        with patch.object(correction, 'atomic_bytes', side_effect=fail_once):
            with self.assertRaises(OSError):
                self.apply(plan)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(self.store.read()['revision'], 0)
        self.assertFalse((self.sources / 'Group/males/B/photo.jpg').exists())
        self.assertFalse(Path(plan['effect']['face']['path']).exists())

    def test_undo_refuses_to_overwrite_an_external_edit(self):
        plan = self.preview(self.draft('edit_source', box=[0, 0, 0.5, 1]))
        self.apply(plan)
        self.master.write_bytes(b'An external edit')
        with self.assertRaises(correction.Conflict):
            self.undo()
        self.assertEqual(self.master.read_bytes(), b'An external edit')

    def test_interrupted_apply_is_recovered_from_journal_on_restart(self):
        plan = self.preview(self.draft('relabel_move', 'B'))
        real = correction.atomic_json

        def crash_before_state(path, value):
            if Path(path) == self.store.path:
                raise SystemExit('Simulated process interruption')
            return real(path, value)

        with patch.object(correction, 'atomic_json', side_effect=crash_before_state):
            with self.assertRaises(SystemExit):
                self.apply(plan)
        self.assertFalse(self.master.exists())
        recovered = correction.CorrectionStore(self.review)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(recovered.read()['revision'], 0)
        self.assertFalse(Path(plan['effect']['face']['path']).exists())

    def test_committed_apply_is_not_rolled_back_after_final_journal_write_failure(self):
        plan = self.preview(self.draft('relabel_move', 'B'))
        real = correction.atomic_json

        def fail_final_journal(path, value):
            if value.get('status') == 'committed':
                raise OSError('Simulated journal write failure after commit')
            return real(path, value)

        with patch.object(correction, 'atomic_json', side_effect=fail_final_journal):
            with self.assertRaises(OSError):
                self.apply(plan)
        recovered = correction.CorrectionStore(self.review)
        self.assertFalse(self.master.exists())
        self.assertEqual(recovered.read()['revision'], 1)
        self.assertEqual(recovered.effects(self.key)['current_face']['identity'], 'B')
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_drafts_and_stale_tabs_and_double_apply(self):
        payload = {'dataset_id': self.dataset, 'key': self.key, 'base_revision': 0,
                   'draft': self.draft(box=[0, 0, 0.5, 1])}
        self.store.save_draft(payload)
        with self.assertRaises(correction.Conflict):
            self.store.save_draft(payload)
        restored = correction.CorrectionStore(self.review)
        self.assertEqual(restored.read()['entries'][self.key]['draft'], payload['draft'])
        plan = self.preview(payload['draft']);self.apply(plan)
        with self.assertRaises(correction.Conflict):
            self.apply(plan)

    def test_keep_exclude_defer_are_separate_from_completed_review(self):
        for action in ['keep', 'exclude', 'defer']:
            self.apply(self.preview(self.draft(action)))
            self.assertEqual(self.store.effects(self.key)['decision'], action)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(self.crop.read_bytes(), self.crop_before)
        self.assertEqual(self.review.path.read_bytes(), self.review_before)

    def test_http_editor_routes_and_origin_protection(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler_for(self.review))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        for route in ['/corrections', '/api/corrections/context', '/api/corrections/record/' + self.key,
                      '/api/corrections/image/' + self.key + '?variant=master']:
            with urllib.request.urlopen(base + route) as response:
                self.assertEqual(response.status, 200)
        payload = {'dataset_id': self.dataset, 'key': self.key, 'draft': self.draft(box=[0, 0, 0.5, 1])}
        request = urllib.request.Request(base + '/api/corrections/preview', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            plan = json.load(response)
        with urllib.request.urlopen(base + plan['face_image_url']) as response:
            self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
        request = urllib.request.Request(base + '/api/corrections/apply', data=json.dumps({'dataset_id': self.dataset, 'token': plan['id']}).encode(), headers={'Content-Type': 'application/json', 'Origin': 'http://other.example'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
