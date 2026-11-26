"""Regression cases for combining review generations without stale pixels/leakage."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import json

from PIL import Image
import yaml

spec = importlib.util.spec_from_file_location('reviewed_assembly',
    Path(__file__).resolve().parents[1] / 'tools/assemble_reviewed_dataset.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def face(name, identity='A', camera='frame1', byte_hash=None, pixel_hash=None,
         split='train', priority=2):
    return {'id': identity, 'input_path': name, 'camera_key': camera,
            'sha256': byte_hash or name, 'pixels_sha256': pixel_hash or name,
            'split_hint': split, 'priority': priority}


class ReviewedAssemblyTests(unittest.TestCase):
    def test_distinct_crops_of_one_camera_frame_survive_and_share_split(self):
        candidates = [face('old-face', split='test'),
                      face('different-face', split='train'),
                      face('another-individual', identity='B', split='val')]
        groups, omitted = A.deduplicate_and_group(candidates)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(omitted, [])

    def test_rgb_duplicate_prefers_manual_pixels_and_unites_camera_aliases(self):
        candidates = [face('old.jpg', pixel_hash='same-pixels', camera='original-name'),
                      face('new.jpg', pixel_hash='same-pixels', camera='renamed', priority=0),
                      face('other-crop.jpg', camera='original-name')]
        groups, omitted = A.deduplicate_and_group(candidates)
        self.assertEqual(len(groups), 1)
        self.assertEqual({c['input_path'] for c in groups[0]}, {'new.jpg', 'other-crop.jpg'})
        self.assertEqual(omitted[0]['input_path'], 'old.jpg')

    def test_identical_face_assigned_to_two_animals_blocks_publication(self):
        with self.assertRaisesRegex(ValueError, 'conflicting identities'):
            A.deduplicate_and_group([face('a', byte_hash='same'),
                                     face('b', identity='B', byte_hash='same')])

    def test_source_replacement_and_explicit_exclusion_override_earlier_faces(self):
        records = [{'key': key, 'source_photo': source, 'split': 'train',
                    'crop_absolute': key+'.jpg', 'sha256': key, 'identity': 'A'}
                   for key, source in [('replaced','changed.jpg'), ('retired','retired.jpg'),
                                       ('excluded','keep.jpg'), ('retained','same.jpg')]]
        annotations = {r['key']: {'status': 'ok'} for r in records}
        effects = {'replaced': {'faces': {'A': {'identity': 'A', 'path': 'stale.jpg', 'sha256': 'stale'}}},
                   'excluded': {'decision': 'exclude'}}
        current = {'identity': 'A', 'path': 'fresh.jpg', 'sha256': 'fresh',
                   'organised_original': 'changed.jpg', 'split_hint': 'train',
                   'camera_key': 'changed', 'source_group': 'group'}
        candidates, omitted = A.merge_candidates(records, annotations, effects,
            {'changed.jpg','retired.jpg'}, {}, [current])
        self.assertEqual({c['input_path'] for c in candidates}, {'retained.jpg','fresh.jpg'})
        self.assertEqual(len(omitted), 3)

    def test_deferred_case_requires_completed_source_followup(self):
        record = {'key': 'k', 'source_photo': 'source.jpg', 'split': 'train',
                  'crop_absolute': 'old.jpg', 'sha256': 'old', 'identity': 'A'}
        args = ([record], {'k': {'status': 'bad_crop'}}, {'k': {'decision': 'defer'}}, set())
        with self.assertRaisesRegex(ValueError, 'Unresolved deferred'):
            A.merge_candidates(*args, {}, [])
        result, _ = A.merge_candidates(*args, {'source.jpg': 'keep'}, [])
        self.assertEqual(result[0]['input_path'], 'old.jpg')
        result, omitted = A.merge_candidates(*args, {'source.jpg': 'edit'}, [])
        self.assertEqual(result, [])
        self.assertEqual(len(omitted), 1)

    def test_closed_set_coverage_repair_moves_whole_camera_groups(self):
        candidates = [face(f'{name}-{i}', identity=name, camera=f'camera-{i}', split='train')
                      for name in ['A','B'] for i in range(4)]
        groups, _ = A.deduplicate_and_group(candidates)
        A.assign_splits(groups, {})
        for group in groups:
            self.assertEqual(len({r['split'] for r in group}), 1)
        for split in A.SPLIT_ORDER:
            self.assertEqual({r['id'] for g in groups for r in g if r['split']==split}, {'A','B'})


class ActivationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'data/macaque_faces/reviewed_2026-11-25'
        self.output.mkdir(parents=True)
        self.configs = self.root / 'configs'
        self.configs.mkdir()
        self.before = {}
        for name in ['first','second']:
            p = self.configs / (name+'.yaml')
            p.write_text(yaml.safe_dump({'name': name, 'data': {'dataset_name': 'macaque_faces',
                'raw_root': '/old', 'splits_path': '/old/splits.json', 'num_classes': 1}}))
            self.before[p] = p.read_bytes()
        registry = {'version': 'test', 'source_root': '/originals', 'aliases': {},
                    'identities': {'A': {'current_group': 'G', 'gender': 'females',
                                        'source_directory': 'G/females/A'}}}
        A.write(self.output / 'identity_registry.json', registry)
        payload = {}
        for number, split in enumerate(A.SPLIT_ORDER):
            path = self.output / 'faces' / split / 'A/photo.jpg'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (8,8), (number*50,30,100)).save(path)
            payload[split] = [{'id': 'A', 'path': f'{split}/A/photo.jpg',
                'current_group': 'G', 'gender': 'females', **A.file_fingerprint(path)}]
        A.write(self.output / 'faces/splits.json', payload)
        A.write(self.output / 'summary.json', {'published': True, 'training_ready': True,
            'originals': 3, 'faces': 3, 'identities': 1,
            'split_counts': {s: 1 for s in A.SPLIT_ORDER}})
        self.root_patch = patch.object(A, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_activation_updates_all_configs_and_preserves_previous_bytes(self):
        A.activate_generation(self.output)
        for path, original in self.before.items():
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config['data']['raw_root'], str(self.output / 'faces'))
            self.assertTrue(config['name'].endswith('_reviewed_20261125'))
            self.assertEqual((self.output / 'audit/configs_before_activation' / path.name).read_bytes(), original)
        pointer = A.read(self.root / 'data/macaque_faces/current_dataset.json')
        self.assertEqual(pointer['generation'], str(self.output))

    def test_interrupted_config_activation_rolls_back_all_references(self):
        original_replace = A.os.replace
        failed = False
        def fail_once(source, target):
            nonlocal failed
            if Path(target) == self.configs / 'second.yaml' and not failed:
                failed = True
                raise OSError('simulated config write failure')
            return original_replace(source, target)
        with patch.object(A.os, 'replace', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                A.activate_generation(self.output)
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / 'data/macaque_faces/current_dataset.json').exists())
        self.assertEqual(A.read(self.output / 'audit/activation.json')['status'], 'rolled_back')

    def test_historical_comparison_config_is_preserved_during_activation(self):
        path=self.configs/'first.yaml'
        cfg=yaml.safe_load(path.read_text())
        cfg['data']['comparison_role']='historical_baseline'
        path.write_text(yaml.safe_dump(cfg))
        original=path.read_bytes()
        A.activate_generation(self.output)
        self.assertEqual(path.read_bytes(),original)
        updated=yaml.safe_load((self.configs/'second.yaml').read_text())
        self.assertEqual(updated['data']['raw_root'],str(self.output/'faces'))


if __name__ == '__main__':
    unittest.main()
