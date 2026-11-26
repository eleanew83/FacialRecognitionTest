"""Whole-subject batches must preserve archive bytes and existing face work."""
import contextlib
import copy
import io
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import test_crop_corrections as fixtures

originals = fixtures.load('original_photo_editor')


class OriginalTests(unittest.TestCase):
    setUpBase = fixtures.CorrectionTests.setUp
    draftFace = fixtures.CorrectionTests.draft

    def setUp(self):
        self.setUpBase()
        self.face_store = self.store
        draft = self.draftFace('defer')
        plan = self.face_store.preview({'dataset_id': self.dataset, 'key': self.key, 'draft': draft})
        self.face_store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        self.face_state = self.face_store.path.read_bytes()
        self.archive = self.output.parent / 'archive'
        archive_file = self.archive / 'OldGroup/oldA/photo.jpg'
        archive_file.parent.mkdir(parents=True)
        archive_file.write_bytes(self.master_before)
        archive_file.with_name('photo2.jpg').write_bytes(self.master_before)
        archive_file.with_name('photo3.jpg').write_bytes(self.master_before)
        archive_file.with_name('other1.jpg').write_bytes(self.master_before)
        self.archive_file = archive_file
        with contextlib.redirect_stdout(io.StringIO()):
            self.catalog = originals.build_catalog(self.review, self.archive)
        self.store = originals.OriginalPhotoStore(self.review)
        self.photo = next(r for r in self.catalog['records'].values() if r['seed'])
        self.photo_key = self.photo['key']
        self.old_id = self.photo['active_files'][0]['id']

    def payload(self, decision='edit'):
        v = self.photo['variants'][0]
        return {'dataset_id': self.dataset, 'key': self.photo_key, 'base_revision': self.store.read()['revision'],
                'draft': {'decision': decision, 'variant': v['id'], 'source_sha256': v['sha256'],
                          'notes': 'Whole monkey batch', 'outputs': [], 'retire': []}}

    def two_outputs(self):
        p = self.payload()
        p['draft']['outputs'] = [
            {'id': 'a' * 16, 'identity': 'A', 'box': [0, 0, 0.5, 1], 'replace': self.old_id},
            {'id': 'b' * 16, 'identity': 'B', 'box': [0.5, 0, 0.5, 1], 'replace': None}]
        return p

    def apply(self, payload):
        plan = self.store.preview(payload)
        result = self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        return plan, result

    def undo(self):
        return self.store.undo({'dataset_id': self.dataset, 'key': self.photo_key,
                               'base_revision': self.store.read()['revision']})

    def assert_preserved(self):
        self.assertEqual(self.archive_file.read_bytes(), self.master_before)
        self.assertEqual(self.crop.read_bytes(), self.crop_before)
        self.assertEqual(self.review.path.read_bytes(), self.review_before)
        self.assertEqual(self.face_store.path.read_bytes(), self.face_state)

    def test_number_matching_handles_sylv_and_timestamps_and_unrelated_files(self):
        self.assertEqual(originals.family_name('Something Sylv_1.JPG'), originals.family_name('something Sylv12.jpg'))
        self.assertEqual(originals.family_name('Something Martin2.JPG'), originals.family_name('Something Martin3.CR2'))
        self.assertNotEqual(originals.family_name('20221217_174649095.jpg'), originals.family_name('20221217_174649096.jpg'))
        self.assertEqual(self.catalog['summary']['frames'], 3)
        self.assertEqual(self.catalog['summary']['seed_frames'], 1)
        self.assertEqual(self.catalog['summary']['additional_frames'], 2)
        self.assert_preserved()

    def test_empty_draft_can_be_saved_but_cannot_be_applied_as_an_edit(self):
        p = self.payload()
        self.store.save_draft(p)
        self.assertTrue(self.store.path.exists())
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.store.preview(p)
        self.assert_preserved()

    def test_retire_only_batch_backs_up_original_and_undo_restores_it(self):
        payload = self.payload()
        payload['draft']['retire'] = [self.old_id]
        plan, _ = self.apply(payload)
        self.assertEqual(plan['effect']['outputs'], [])
        self.assertFalse(self.master.exists())
        journal = json.loads((self.store.root / 'transactions' / (plan['id'] + '.json')).read_text())
        backup = Path(journal['operations'][0]['backup'])
        self.assertEqual(backup.read_bytes(), self.master_before)
        self.assert_preserved()
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_two_full_resolution_subject_outputs_replace_and_add_with_batch_undo(self):
        historical = self.sources / 'hardlinked_archive.jpg'
        os.link(self.master, historical)
        plan = self.store.preview(self.two_outputs())
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(len(plan['operations']), 2)
        self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        for output in plan['effect']['outputs']:
            with Image.open(output['path']) as image:
                self.assertEqual(image.size, (100, 120))
                if output['identity'] == 'A': self.assertLess(image.getpixel((50, 60))[1], 50)
                else: self.assertGreater(image.getpixel((60, 60))[1], 240)
        self.assertEqual(historical.read_bytes(), self.master_before)
        self.assert_preserved()
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertFalse(Path(plan['effect']['outputs'][1]['path']).exists())

    def test_move_by_adding_correct_identity_then_retiring_wrong_folder_copy(self):
        p = self.two_outputs()
        p['draft']['outputs'] = p['draft']['outputs'][1:]
        p['draft']['retire'] = [self.old_id]
        plan, result = self.apply(p)
        self.assertFalse(self.master.exists())
        self.assertEqual(len(plan['operations']), 2)
        # The old copy remains viewable from its backup after retirement.
        data = self.store.image(self.photo_key, self.old_id, existing=True)
        with Image.open(io.BytesIO(data)) as image: self.assertEqual(image.size, (200, 120))
        self.assert_preserved()
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_keep_and_defer_record_without_modifying_any_source(self):
        for decision in ['keep', 'defer']:
            plan, _ = self.apply(self.payload(decision))
            self.assertEqual(plan['operations'], [])
            self.assert_preserved()
            self.undo()

    def test_new_numbered_frame_without_active_copies_can_create_individual_originals(self):
        related = next(r for r in self.catalog['records'].values() if not r['seed'])
        self.assertEqual(related['active_files'], [])
        payload = self.two_outputs()
        payload['key'] = related['key']
        payload['draft'].update(variant=related['default_variant'], source_sha256=related['variants'][0]['sha256'])
        payload['draft']['outputs'][0]['replace'] = None
        plan, _ = self.apply(payload)
        self.assertEqual(len(plan['effect']['outputs']), 2)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        for output in plan['effect']['outputs']:
            self.assertTrue(Path(output['path']).exists())
        self.store.undo({'dataset_id': self.dataset, 'key': related['key'], 'base_revision': self.store.read()['revision']})
        for output in plan['effect']['outputs']:
            self.assertFalse(Path(output['path']).exists())
        self.assert_preserved()

    def test_modified_archive_or_destination_invalidates_preview(self):
        plan = self.store.preview(self.two_outputs())
        self.archive_file.write_bytes(self.master_before + b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        self.archive_file.write_bytes(self.master_before)
        self.master.write_bytes(self.master_before + b'external change')
        with self.assertRaises(ValueError):
            self.store.preview(self.two_outputs())

    def test_invalid_identity_boxes_replacements_and_retire_lists(self):
        for mutate in [lambda d:d['outputs'][0].update(identity='../outside'),
                       lambda d:d['outputs'][0].update(box=[0, 0, 2, 1]),
                       lambda d:d['outputs'][1].update(replace=self.old_id),
                       lambda d:d.update(retire=['unknown']),
                       lambda d:d.update(retire=[self.old_id])]:
            p = self.two_outputs()
            mutate(p['draft'])
            with self.assertRaises(ValueError): self.store.preview(p)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assert_preserved()

    def test_failed_second_write_rolls_back_first_output(self):
        plan = self.store.preview(self.two_outputs())
        second_path = Path(plan['operations'][1]['path'])
        original = originals.C.atomic_bytes
        def fail(path, data):
            if Path(path) == second_path: raise OSError('Simulated destination failure')
            return original(path, data)
        with patch.object(originals.C, 'atomic_bytes', side_effect=fail):
            with self.assertRaises(OSError):
                self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertFalse(second_path.exists())
        self.assertEqual(self.store.read()['commits'], [])
        self.assert_preserved()

    def test_reapply_and_overwriting_an_applied_batch_are_blocked(self):
        plan, _ = self.apply(self.two_outputs())
        with self.assertRaises(ValueError): self.store.apply({'dataset_id': self.dataset, 'token': plan['id']})
        with self.assertRaises(ValueError): self.store.preview(self.two_outputs())
        with self.assertRaises(ValueError): self.store.save_draft(self.two_outputs())
        with self.assertRaisesRegex(ValueError, 'already has saved work'):
            originals.build_catalog(self.review, self.archive)

    def test_undo_refuses_to_overwrite_external_edit(self):
        plan, _ = self.apply(self.two_outputs())
        self.master.write_bytes(b'external edits')
        with self.assertRaisesRegex(ValueError, 'externally|external edit'):
            self.undo()
        self.assertEqual(self.master.read_bytes(), b'external edits')

    def test_http_page_owner_origin_draft_preview_apply_and_undo(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), fixtures.review.handler_for(self.review))
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        self.addCleanup(server.shutdown)
        origin = f'http://127.0.0.1:{server.server_port}'
        def request(path, payload=None, external=False):
            headers = {'Content-Type': 'application/json', 'Origin': 'http://untrusted.example' if external else origin}
            req = urllib.request.Request(origin + path, data=json.dumps(payload).encode() if payload else None, headers=headers)
            return urllib.request.urlopen(req)
        self.assertIn(b'Whole-monkey original photo editor', request('/originals').read())
        data = json.load(request('/api/originals/context'))
        self.assertEqual(data['summary']['frames'], 3)
        payload = self.two_outputs()
        with self.assertRaises(urllib.error.HTTPError) as denied:
            request('/api/originals/preview', payload, external=True)
        self.assertEqual(denied.exception.code, 403)
        saved = json.load(request('/api/originals/draft', payload))
        plan = json.load(request('/api/originals/preview', payload))
        self.assertIn(b'\xff\xd8', request(plan['effect']['outputs'][0]['preview_url']).read()[:2])
        json.load(request('/api/originals/apply', {'dataset_id': self.dataset, 'token': plan['id']}))
        state = json.load(request('/api/originals/context'))
        json.load(request('/api/originals/undo', {'dataset_id': self.dataset, 'key': self.photo_key, 'base_revision': state['revision']}))
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assert_preserved()


if __name__ == '__main__':
    unittest.main()
