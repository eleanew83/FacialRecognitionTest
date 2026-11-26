"""Only changed organised originals enter the new face-cropping stage."""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import test_crop_corrections as fixtures
import test_original_photo_editor as original_tests

organized = fixtures.load('organized_face_editor')


class OrganizedFaceTests(unittest.TestCase):
    setUpBase = fixtures.CorrectionTests.setUp
    draftFace = fixtures.CorrectionTests.draft
    payload = original_tests.OriginalTests.payload
    two_outputs = original_tests.OriginalTests.two_outputs
    apply_original = original_tests.OriginalTests.apply
    assert_preserved = original_tests.OriginalTests.assert_preserved

    def setUp(self):
        original_tests.OriginalTests.setUp(self)
        self.faces = organized.OrganizedFaceStore(self.review, self.store)

    def source_edit(self):
        return self.apply_original(self.two_outputs())[0]

    def face_payload(self, key=None, action='face_crop'):
        if key is None:
            key = self.faces.context()['records'][0]['key']
        meta = self.faces.metadata(key)
        return {'dataset_id': self.dataset, 'key': key, 'base_revision': self.faces.read()['revision'],
                'draft': {'action': action, 'target_identity': meta['record']['identity'], 'variant': 'master',
                          'source_sha256': meta['variants'][0]['sha256'], 'box': [0.2, 0.1, 0.5, 0.5],
                          'keep_existing_face': False, 'notes': 'Face after original organisation'}}

    def apply_face(self, payload):
        preview = self.faces.preview(payload)
        self.faces.apply({'dataset_id': self.dataset, 'token': preview['id']})
        return preview

    def test_no_changes_needed_and_unfinished_photos_never_enter_face_queue(self):
        self.assertEqual(self.faces.context()['records'], [])
        self.apply_original(self.payload('keep'))
        self.assertEqual(self.faces.context()['records'], [])
        self.assertEqual(self.faces.export()['approved_faces'], [])
        self.assert_preserved()

    def test_only_new_or_replaced_outputs_are_queued_not_retained_copies(self):
        payload = self.two_outputs()
        payload['draft']['outputs'] = payload['draft']['outputs'][1:]
        self.apply_original(payload)
        rows = self.faces.context()['records']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['identity'], 'B')
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assert_preserved()

    def test_multiple_outputs_keep_assigned_identity_and_common_photo_group(self):
        self.source_edit()
        rows = self.faces.context()['records']
        self.assertEqual({r['identity'] for r in rows}, {'A', 'B'})
        self.assertEqual({r['source_group'] for r in rows}, {self.photo_key})
        for row in rows:
            meta = self.faces.metadata(row['key'])
            self.assertEqual(meta['variants'][0]['width'], 100)
            self.assertEqual(meta['variants'][0]['height'], 120)

    def test_full_resolution_face_save_and_undo_leave_organised_original_unchanged(self):
        original_plan = self.source_edit()
        snapshots = {o['path']: Path(o['path']).read_bytes() for o in original_plan['effect']['outputs']}
        payload = self.face_payload()
        self.faces.save_draft(payload)
        face_plan = self.apply_face(payload)
        face = face_plan['effect']['face']
        with Image.open(face['path']) as image:
            self.assertEqual(image.size, (50, 60))
        self.assertEqual(face['source_group'], self.photo_key)
        self.assertEqual(len(self.faces.export()['approved_faces']), 1)
        for path, raw in snapshots.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        self.assert_preserved()
        self.faces.undo({'dataset_id': self.dataset, 'key': payload['key'], 'base_revision': self.faces.read()['revision']})
        self.assertEqual(self.faces.export()['approved_faces'], [])
        self.assertFalse(Path(face['path']).exists())

    def test_original_undo_invalidates_saved_faces_and_stale_previews(self):
        self.source_edit()
        payload = self.face_payload()
        self.apply_face(payload)
        pending = self.faces.preview(payload)
        self.store.undo({'dataset_id': self.dataset, 'key': self.photo_key, 'base_revision': self.store.read()['revision']})
        self.assertEqual(self.faces.context()['records'], [])
        exported = self.faces.export()
        self.assertEqual(exported['approved_faces'], [])
        self.assertIn(payload['key'], exported['inactive_saved_keys'])
        with self.assertRaises(ValueError):
            self.faces.apply({'dataset_id': self.dataset, 'token': pending['id']})
        self.assert_preserved()

    def test_external_source_change_blocks_save_and_export_approval(self):
        self.source_edit()
        payload = self.face_payload()
        self.apply_face(payload)
        member = self.faces.source_members[payload['key']]
        path = Path(member['path'])
        path.write_bytes(path.read_bytes() + b'external change')
        with self.assertRaisesRegex(ValueError, 'organised original changed'):
            self.faces.preview(payload)
        exported = self.faces.export()
        self.assertEqual(exported['approved_faces'], [])
        self.assertIn('source_changed', [d['decision'] for d in exported['decisions']])

    def test_stage_cannot_relabel_or_edit_originals(self):
        self.source_edit()
        for field, value in [('target_identity', 'B'), ('action', 'edit_source'), ('keep_existing_face', True)]:
            payload = self.face_payload(next(r['key'] for r in self.faces.context()['records'] if r['identity'] == 'A'))
            payload['draft'][field] = value
            with self.assertRaises(ValueError):
                self.faces.preview(payload)
        self.assert_preserved()

    def test_excluded_faces_stay_out_of_manifest_and_deferred_faces_stay_pending(self):
        self.source_edit()
        rows = self.faces.context()['records']
        for row, decision in zip(rows, ['exclude', 'defer']):
            self.apply_face(self.face_payload(row['key'], decision))
        self.assertEqual(self.faces.export()['approved_faces'], [])
        statuses = {r['decision']: r['applied'] for r in self.faces.context()['records']}
        self.assertTrue(statuses['exclude'])
        self.assertFalse(statuses['defer'])
        self.assert_preserved()

    def test_http_stage_two_page_crop_export_and_origin_protection(self):
        self.source_edit()
        server = ThreadingHTTPServer(('127.0.0.1', 0), fixtures.review.handler_for(self.review))
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        origin = f'http://127.0.0.1:{server.server_port}'
        def request(path, payload=None, external=False):
            req = urllib.request.Request(origin + path, data=json.dumps(payload).encode() if payload else None,
                headers={'Content-Type': 'application/json', 'Origin': 'http://elsewhere.invalid' if external else origin})
            return urllib.request.urlopen(req, timeout=20)
        self.assertIn(b'Stage 2', request('/organized-faces').read())
        rows = json.load(request('/api/organized-faces/context'))['records']
        self.assertEqual(len(rows), 2)
        payload = self.face_payload(rows[0]['key'])
        with self.assertRaises(urllib.error.HTTPError) as blocked:
            request('/api/organized-faces/draft', payload, external=True)
        self.assertEqual(blocked.exception.code, 403)
        json.load(request('/api/organized-faces/draft', payload))
        preview = json.load(request('/api/organized-faces/preview', payload))
        self.assertTrue(request(preview['face_image_url']).read().startswith(b'\xff\xd8'))
        json.load(request('/api/organized-faces/apply', {'dataset_id': self.dataset, 'token': preview['id']}))
        self.assertEqual(len(json.load(request('/api/organized-faces/export'))['approved_faces']), 1)
        self.assert_preserved()


if __name__ == '__main__':
    unittest.main()
