"""Incremental queue additions must preserve in-progress original edits."""
import contextlib
import copy
import csv
import io
import json
import unittest
from pathlib import Path

from PIL import Image

import test_crop_corrections as fixtures
import test_original_photo_editor as original_tests

O = original_tests.originals


class QueueExtensionTests(unittest.TestCase):
    setUpBase = fixtures.CorrectionTests.setUp
    draftFace = fixtures.CorrectionTests.draft
    payload = original_tests.OriginalTests.payload
    two_outputs = original_tests.OriginalTests.two_outputs
    apply_original = original_tests.OriginalTests.apply
    assert_preserved = original_tests.OriginalTests.assert_preserved

    def setUp(self):
        original_tests.OriginalTests.setUp(self)
        registry = json.loads(self.registry.read_text())
        registry['identities']['Goblin'] = {'source_directory': 'Group/females/Goblin'}
        self.registry.write_text(json.dumps(registry))
        active = self.sources / 'Group/females/Goblin'
        archive = self.archive / 'Group/females/Goblin'
        active.mkdir(parents=True)
        archive.mkdir(parents=True)
        self.audit = self.output / 'audit.csv'
        cases = []
        for number, category in [(1, 'additional_adult_or_juvenile'), (2, 'adult_with_infant_or_young'),
                                 (3, 'possible_occluded_infant'), (4, 'one_visible_monkey')]:
            name = f'Group Goblin Sylv{number}.jpg'
            (active / name).write_bytes(self.master_before)
            (archive / name).write_bytes(self.master_before)
            cases.append({'filename': name, 'path': str(active / name), 'visual_category': category})
        for name, category in [('Group Goblin Sylv1_cropped.jpg', 'additional_adult_or_juvenile'),
                               ('Group Goblin Sylv2.jpg.jpg', 'adult_with_infant_or_young')]:
            with Image.open(io.BytesIO(self.master_before)) as image:
                image.crop((0, 0, 80, 100)).save(active / name)
            cases.append({'filename': name, 'path': str(active / name), 'visual_category': category})
        with self.audit.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['filename', 'path', 'visual_category'])
            writer.writeheader();writer.writerows(cases)

    def append(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return O.append_goblin_candidates(self.review, self.audit)

    def test_candidates_added_without_changing_saved_edits_or_existing_records(self):
        self.apply_original(self.payload('keep'))
        state_before = self.store.path.read_bytes()
        before = copy.deepcopy(self.store.catalog['records'])
        self.store.context()  # Cache the old catalog, as a running server does.
        result = self.append()
        self.assertEqual(result['selected_files'], 5)
        self.assertEqual(result['selected_photos'], 3)
        self.assertEqual(result['added_frames'], 3)
        self.assertEqual(result['total_frames'], 6)
        self.assertEqual(self.store.path.read_bytes(), state_before)
        for key, record in before.items():
            self.assertEqual(json.loads((self.store.root / 'catalog.json').read_text())['records'][key], record)
        # A running updated store discovers new catalog records automatically.
        rows = self.store.context()['records']
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r['goblin_candidate'] for r in rows), 3)
        self.assertEqual(len(list((self.store.root / 'catalog_backups').glob('*.json'))), 1)
        self.assert_preserved()

    def test_repeating_extension_does_not_duplicate_candidates_or_touch_catalog(self):
        self.append()
        before = (self.store.root / 'catalog.json').read_bytes()
        result = self.append()
        self.assertEqual(result['added_frames'], 0)
        self.assertEqual((self.store.root / 'catalog.json').read_bytes(), before)

    def test_derivatives_are_grouped_and_single_monkey_photo_is_not_flagged(self):
        self.append()
        catalog = json.loads((self.store.root / 'catalog.json').read_text())
        added = [r for r in catalog['records'].values() if r.get('added_reason')]
        self.assertEqual(len(added), 3)
        self.assertFalse(any('Sylv4' in r['filename'] for r in added))
        first = next(r for r in added if 'Sylv1' in r['filename'])
        second = next(r for r in added if 'Sylv2' in r['filename'])
        self.assertEqual(len(first['active_files']), 2)
        self.assertEqual(len(second['active_files']), 2)
        self.assertEqual(first['variants'][0]['kind'], 'archive')
        self.assertEqual(len(first['audit_files']), 2)


if __name__ == '__main__':
    unittest.main()
