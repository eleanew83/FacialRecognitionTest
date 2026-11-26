"""Round-trip portable boxes through the journalled editor without real data."""
import contextlib
import copy
import io
import json
import uuid
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import test_crop_corrections as fixtures

offline = fixtures.load('offline_corrections')


class OfflineTests(unittest.TestCase):
    setUp = fixtures.CorrectionTests.setUp
    draft = fixtures.CorrectionTests.draft
    preview = fixtures.CorrectionTests.preview
    apply = fixtures.CorrectionTests.apply
    undo = fixtures.CorrectionTests.undo

    def package(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = offline.export_package(self.store, workers=1)
        self.summary = summary
        self.manifest = json.loads((self.store.root / 'offline_packages' / summary['package_id'] / 'manifest.json').read_text())
        self.results = {'schema': offline.SCHEMA, 'dataset_id': self.dataset,
                        'package_id': summary['package_id'], 'session_id': uuid.uuid4().hex, 'items': {}}
        return summary

    def event(self, draft):
        entry = self.results['items'].setdefault(self.key, {'draft': None, 'events': []})
        event = {'id': uuid.uuid4().hex, 'draft': draft, 'saved_at': '2026-11-24T12:00:00Z'}
        entry['events'].append(event)
        entry['draft'] = copy.deepcopy(draft)
        return event

    def import_file(self, apply=False):
        path = self.output / 'results.json'
        path.write_text(json.dumps(self.results))
        with contextlib.redirect_stdout(io.StringIO()):
            return offline.import_results(self.store, path, apply=apply)

    def test_package_contains_self_contained_editor_and_full_size_assets(self):
        summary = self.package()
        self.assertEqual(summary['records'], 1)
        self.assertEqual(summary['image_assets'], 2)
        with zipfile.ZipFile(summary['zip_path']) as archive:
            index = archive.read('macaque_offline_editor/index.html').decode()
            self.assertIn('window.CROP_OFFLINE =', index)
            self.assertIn('"width": 200', index)
            self.assertEqual(len([n for n in archive.namelist() if n.endswith('.js')]), 2)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertEqual(self.review.path.read_bytes(), self.review_before)
        self.assertFalse(self.store.path.exists())

    def test_validation_never_changes_original_or_correction_state(self):
        self.package()
        self.event(self.draft(box=[0.5, 0, 0.5, 1]))
        report = self.import_file()
        self.assertEqual(report['pending_operations'], 1)
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertFalse(self.store.path.exists())

    def test_full_resolution_face_round_trip_idempotent_and_undoable(self):
        self.package()
        self.event(self.draft(box=[0.5, 0, 0.5, 1]))
        self.assertEqual(self.import_file(True)['applied'], 1)
        face = self.store.effects(self.key)['current_face']
        with Image.open(face['path']) as image:
            self.assertEqual(image.size, (100, 120))
            self.assertGreater(image.getpixel((60, 60))[1], 240)
        self.assertEqual(self.import_file(True)['applied'], 0)
        self.assertEqual(len(self.store.read()['commits']), 1)
        self.assertEqual(self.review.path.read_bytes(), self.review_before)
        self.undo()
        self.assertIsNone(self.store.effects(self.key)['current_face'])

    def test_crop_original_then_crop_face_uses_immutable_export_coordinates(self):
        self.package()
        self.event(self.draft('edit_source', box=[0, 0, 0.5, 1]))
        self.event(self.draft('face_crop', box=[0.5, 0, 0.5, 1]))
        self.import_file(True)
        with Image.open(self.master) as image:
            self.assertEqual(image.size, (100, 120))
            self.assertLess(image.getpixel((50, 60))[1], 50)
        face = self.store.effects(self.key)['current_face']
        with Image.open(face['path']) as image:
            self.assertEqual(image.size, (100, 120))
            self.assertGreater(image.getpixel((60, 60))[1], 240)
        self.assertFalse(self.store.effects(self.key)['face_pending'])
        self.undo()
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_relabel_move_and_copy_round_trip(self):
        self.package()
        self.event(self.draft('relabel_copy', 'B', [0.5, 0, 0.5, 1]))
        self.event(self.draft('relabel_move', 'B'))
        self.import_file(True)
        effects = self.store.effects(self.key)
        self.assertEqual(effects['current_face']['identity'], 'B')
        self.assertEqual(Path(effects['current_face']['path']).read_bytes(), self.crop_before)
        self.assertFalse(self.master.exists())
        self.assertEqual(Path(effects['source_path']).read_bytes(), self.master_before)
        copied = next(iter(effects['copies'].values()))
        with Image.open(copied['path']) as image:
            self.assertEqual(image.size, (100, 120))
        self.assertEqual(self.import_file(True)['applied'], 0)
        self.undo()
        self.undo()
        self.assertEqual(self.master.read_bytes(), self.master_before)
        self.assertFalse(Path(copied['path']).exists())

    def test_changed_online_draft_or_source_blocks_import(self):
        self.package()
        self.event(self.draft(box=[0, 0, 0.5, 1]))
        self.master.write_bytes(self.master_before + b'changed')
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            self.import_file(True)
        self.master.write_bytes(self.master_before)
        self.store.save_draft({'dataset_id': self.dataset, 'key': self.key, 'base_revision': 0, 'draft': self.draft()})
        with self.assertRaisesRegex(ValueError, 'Online corrections changed'):
            self.import_file(True)

    def test_tampered_hash_identity_box_and_unknown_key_rejected_before_any_apply(self):
        self.package()
        self.event(self.draft(box=[0, 0, 0.5, 1]))
        original = copy.deepcopy(self.results)
        for field, value in [('source_sha256', 'wrong'), ('target_identity', '../bad'), ('box', [0, 0, 2, 1])]:
            self.results = copy.deepcopy(original)
            self.results['items'][self.key]['events'][0]['draft'][field] = value
            with self.assertRaises(ValueError):
                self.import_file(True)
        self.results = copy.deepcopy(original)
        self.results['items']['unknown'] = self.results['items'].pop(self.key)
        with self.assertRaises(ValueError):
            self.import_file(True)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.master.read_bytes(), self.master_before)

    def test_preexisting_applied_correction_is_preserved_and_not_replayed(self):
        self.apply(self.preview(self.draft('relabel_crop', 'B')))
        before = copy.deepcopy(self.store.read())
        self.package()
        self.assertEqual(self.summary['already_applied'], 1)
        self.assertEqual(self.import_file(True)['applied'], 0)
        self.assertEqual(self.store.read(), before)
        self.event(self.draft('keep'))
        self.import_file(True)
        self.assertEqual(self.store.read()['entries'][self.key]['history'][0], before['entries'][self.key]['history'][0])

    def test_interrupted_batch_resumes_without_duplicate_commit(self):
        self.package()
        self.event(self.draft('edit_source', box=[0, 0, 0.5, 1]))
        self.event(self.draft('face_crop', box=[0.5, 0, 0.5, 1]))
        real_apply = self.store.apply
        calls = []

        def fail_second(payload):
            calls.append(payload)
            if len(calls) == 2:
                raise OSError('simulated interruption')
            return real_apply(payload)

        with patch.object(self.store, 'apply', side_effect=fail_second):
            with self.assertRaises(OSError):
                self.import_file(True)
        self.assertEqual(len(self.store.read()['commits']), 1)
        self.assertEqual(self.import_file(True)['applied'], 1)
        self.assertEqual(len(self.store.read()['commits']), 2)

    def test_later_session_download_can_extend_unchanged_imported_history(self):
        self.package()
        self.event(self.draft('keep'))
        self.import_file(True)
        self.event(self.draft('defer'))
        report = self.import_file(True)
        self.assertEqual(report['already_imported'], 1)
        self.assertEqual(report['applied'], 1)
        self.results['items'][self.key]['events'][0]['draft']['notes'] = 'changed history'
        with self.assertRaisesRegex(ValueError, 'history was removed or changed'):
            self.import_file(True)

    def test_additional_offline_face_crop_still_works_after_imported_source_move(self):
        self.package()
        face_draft = self.draft('face_crop', 'B', [0.5, 0, 0.5, 1])
        self.event(self.draft('move_source', 'B'))
        self.import_file(True)
        self.assertFalse(self.master.exists())
        self.event(face_draft)
        self.assertEqual(self.import_file(True)['applied'], 1)
        face = self.store.effects(self.key)['current_face']
        self.assertEqual(face['identity'], 'B')
        with Image.open(face['path']) as image:
            self.assertEqual(image.size, (100, 120))

    def test_changes_to_an_unrelated_online_record_do_not_block_import(self):
        second = 'b' * 24
        self.review.records[second] = {**self.review.records[self.key], 'key': second, 'identity': 'B'}
        self.package()
        self.event(self.draft('keep'))
        self.store.save_draft({'dataset_id': self.dataset, 'key': second, 'base_revision': 0,
                               'draft': self.draft('defer', 'B')})
        self.assertEqual(self.import_file(True)['applied'], 1)
        self.assertEqual(self.store.read()['entries'][second]['draft']['action'], 'defer')


if __name__ == '__main__':
    unittest.main()
