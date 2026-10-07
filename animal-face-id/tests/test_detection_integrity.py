"""Regression coverage for detector leakage, unsafe rebuilds and incremental edits."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.detection_integrity import apply_dataset_changes, parse_labels, validate_detection_dataset

SCRIPTS = ROOT.parent / 'yolo_detection/yolo_detection_code/scripts'
BOX = b'0 0.5 0.5 0.4 0.4\n'


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_image(root, split, name, color):
    image = root / 'images' / split / name
    label = root / 'labels' / split / Path(name).with_suffix('.txt')
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (16, 16), color).save(image)
    label.write_bytes(BOX)
    return image, label


def dataset(root, name='face_abcdef01.png'):
    add_image(root, 'train', name, (30, 50, 70))
    add_image(root, 'val', name, (200, 50, 70))
    (root / 'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: macaque_face\n')
    return root


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class DetectionIntegrityTests(unittest.TestCase):
    def test_same_name_distinct_crops_remain_in_separate_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp))
            self.assertEqual(validate_detection_dataset(root)['image_counts'], {'train': 1, 'val': 1})

    def test_exact_duplicates_are_rejected_within_and_across_splits(self):
        for split in ('train', 'val'):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as temp:
                root = dataset(Path(temp))
                image, _ = add_image(root, split, 'duplicate.png', (1, 2, 3))
                shutil.copy2(root / 'images/train/face_abcdef01.png', image)
                with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                    validate_detection_dataset(root)

    def test_identical_rgb_in_different_encodings_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp))
            path = root / 'images/val/face_abcdef01.png'
            Image.new('RGB', (16, 16), (30, 50, 70)).save(path, compress_level=0)
            self.assertNotEqual(path.read_bytes(), (root / 'images/train/face_abcdef01.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                validate_detection_dataset(root)

    def test_bad_boxes_and_repeated_annotations_are_rejected(self):
        for content in (b'1 0.5 0.5 0.2 0.2', b'0 nan 0.5 0.2 0.2', b'0 0.1 0.5 0.5 0.5',
                        b'0 0.5 0.5 0 0.2', BOX + BOX):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'Invalid YOLO'):
                parse_labels(content)

    def test_stale_orphans_and_unconfigured_splits_are_rejected(self):
        for kind in ('orphan', 'missing', 'test'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = dataset(Path(temp))
                if kind == 'orphan':
                    (root / 'labels/train/stale.txt').write_bytes(BOX)
                elif kind == 'missing':
                    (root / 'labels/train/face_abcdef01.txt').unlink()
                else:
                    add_image(root, 'test', 'stale.png', (12, 14, 16))
                with self.assertRaises(ValueError):
                    validate_detection_dataset(root)

    def test_duplicate_update_fails_without_touching_data_or_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data')
            (root / 'labels/train.cache').write_bytes(b'old cache')
            before = snapshot(root)
            with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                apply_dataset_changes(root, {'images/val/face_abcdef01.png': root / 'images/train/face_abcdef01.png'}, {})
            self.assertEqual(snapshot(root), before)

    def test_rename_preserves_split_bytes_and_label_and_backs_up_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data')
            old_image = root / 'images/train/face_abcdef01.png'
            original = old_image.read_bytes()
            (root / 'labels/train.cache').write_bytes(b'old cache')
            result = apply_dataset_changes(root,
                {'images/train/face_abcdef01.png': None, 'images/train/renamed.png': old_image},
                {'labels/train/face_abcdef01.txt': None, 'labels/train/renamed.txt': BOX})
            self.assertEqual((root / 'images/train/renamed.png').read_bytes(), original)
            self.assertEqual((root / 'labels/train/renamed.txt').read_bytes(), BOX)
            self.assertTrue((root / 'images/val/face_abcdef01.png').exists())
            self.assertFalse((root / 'labels/train.cache').exists())
            self.assertEqual((Path(result['backup_directory']) / 'labels/train.cache').read_bytes(), b'old cache')

    def test_failed_commit_rolls_back_images_labels_and_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data')
            before = snapshot(root)
            real_replace, calls = os.replace, [0]
            def interrupt_second_replace(source, target):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError('interrupted commit')
                return real_replace(source, target)
            with patch('src.datasets.detection_integrity.os.replace', side_effect=interrupt_second_replace):
                with self.assertRaisesRegex(OSError, 'interrupted commit'):
                    apply_dataset_changes(root,
                        {'images/train/face_abcdef01.png': None,
                         'images/train/renamed.png': root / 'images/train/face_abcdef01.png'},
                        {'labels/train/face_abcdef01.txt': None, 'labels/train/renamed.txt': BOX})
            self.assertEqual(snapshot(root), before)

    def test_dry_run_validates_without_writing_or_backups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data'); before = snapshot(root)
            apply_dataset_changes(root, {}, {'labels/train/face_abcdef01.txt': b'0 0.5 0.5 0.3 0.3\n'}, dry_run=True)
            self.assertEqual(snapshot(root), before)
            self.assertFalse(list(root.parent.glob('*.update-backup-*')))

    def test_interrupted_update_journal_blocks_training_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data')
            backup = root.parent / '.data.update-backup-interrupted'; backup.mkdir()
            (backup / 'operation.json').write_text(json.dumps({'status': 'applying'}))
            with self.assertRaisesRegex(RuntimeError, 'Interrupted dataset update'):
                validate_detection_dataset(root)

    def test_training_invalidates_label_cache_before_model_load(self):
        module = load_script('train_yolo_detection.py')
        fake_yolo, fake_torch = ModuleType('ultralytics'), ModuleType('torch')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = dataset(base / 'data')
            cache = root / 'labels/train.cache'; cache.write_bytes(b'old annotation cache')
            calls = []
            def load_model(path):
                self.assertFalse(cache.exists())
                return SimpleNamespace(train=lambda **kwargs: calls.append(kwargs))
            fake_yolo.YOLO = load_model
            with patch.dict(sys.modules, {'ultralytics': fake_yolo, 'torch': fake_torch}), patch.multiple(
                    module, MODEL_DIR=str(base / 'models'), RUNS_DIR=str(base / 'models/runs'),
                    LEGACY_DIR=str(base / 'models/legacy'), LAST_RUN_FILE=str(base / 'models/latest_run.txt')):
                module.train_model(data_dir=root, device='cpu', run_name='fixture')
            self.assertEqual(calls[0]['data'], str(root / 'dataset.yaml'))

    def test_bad_dataset_blocks_training_before_model_load(self):
        module = load_script('train_yolo_detection.py')
        fake = ModuleType('ultralytics')
        fake.YOLO = lambda _: self.fail('Model loaded before dataset validation')
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'ultralytics': fake}):
            root = dataset(Path(temp))
            shutil.copy2(root / 'images/train/face_abcdef01.png', root / 'images/val/face_abcdef01.png')
            with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                module.train_model(data_dir=root)


class DetectionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.prepare = load_script('prepare_yolo_dataset.py')
        self.update = load_script('update_yolo_detection_data.py')

    def sources(self, base):
        source, labels = base / 'source', base / 'annotations'
        for relative, color in [('A/same.png', (10, 30, 50)), ('B/same.png', (70, 90, 110)),
                                ('C/copy.png', (10, 30, 50))]:
            image, label = source / relative, labels / Path(relative).with_suffix('.txt')
            image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (16, 16), color).save(image); label.write_bytes(BOX)
        return source, labels

    def test_content_grouping_is_deterministic_and_preserves_distinct_same_named_crops(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source, labels = self.sources(base)
            first = self.prepare.prepare_dataset(source, base / 'one', labels_dir=labels, seed=7)
            second = self.prepare.prepare_dataset(source, base / 'two', labels_dir=labels, seed=7)
            self.assertEqual(len(first['records']), 2)
            self.assertEqual([(r['path'], r['sha256']) for r in first['records']],
                             [(r['path'], r['sha256']) for r in second['records']])
            self.assertEqual(json.loads((base / 'one/preparation_manifest.json').read_text())['exact_copies_omitted'], 1)

    def test_existing_reviewed_dataset_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source, labels = self.sources(base); output = dataset(base / 'reviewed')
            before = snapshot(output)
            with self.assertRaises(FileExistsError):
                self.prepare.prepare_dataset(source, output, labels_dir=labels)
            self.assertEqual(snapshot(output), before)

    def test_conflicting_duplicate_labels_are_never_published(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source, labels = self.sources(base)
            (labels / 'C/copy.txt').write_bytes(b'0 0.5 0.5 0.2 0.2\n')
            with self.assertRaisesRegex(ValueError, 'conflicting annotations'):
                self.prepare.prepare_dataset(source, base / 'output', labels_dir=labels)
            self.assertFalse((base / 'output').exists())

    def test_person_checkpoint_and_empty_detections_cannot_generate_placeholders(self):
        with self.assertRaisesRegex(ValueError, 'COCO class 0 is person'):
            self.prepare.detection_labels(SimpleNamespace(names={0: 'person'}), Path('unused'), 0.3)
        class EmptyModel:
            names = {0: 'macaque_face'}
            def __call__(self, *args, **kwargs):
                return [SimpleNamespace(boxes=[])]
        with self.assertRaisesRegex(ValueError, 'No face detected'):
            self.prepare.detection_labels(EmptyModel(), Path('unused'), 0.3)

    def test_all_detected_faces_are_kept_in_proposals(self):
        def scalar(value):
            return SimpleNamespace(item=lambda: value)
        class Model:
            names = {0: 'macaque_face'}
            def __call__(self, *args, **kwargs):
                boxes = [SimpleNamespace(cls=scalar(0), conf=scalar(0.9),
                         xywhn=[SimpleNamespace(tolist=lambda x=x: [x, 0.5, 0.2, 0.2])]) for x in (0.3, 0.7)]
                return [SimpleNamespace(boxes=boxes)]
        self.assertEqual(len(parse_labels(self.prepare.detection_labels(Model(), Path('unused'), 0.3))), 2)

    def test_ambiguous_same_named_targets_require_split_and_leave_both_crops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp) / 'data'); source = Path(temp) / 'source'; source.mkdir()
            before = snapshot(root)
            with self.assertRaisesRegex(ValueError, 'Ambiguous target filename'):
                self.update.build_update_plan([{'image': 'face_abcdef01.png d'}], source, root)
            self.assertEqual(snapshot(root), before)

    def test_replacing_pixels_requires_reviewed_replacement_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = dataset(base / 'data'); source = base / 'source'; source.mkdir()
            Image.new('RGB', (16, 16), (80, 90, 100)).save(source / 'face.png')
            row = {'image': 'face_abcdef01.png o', 'split': 'train'}
            with self.assertRaisesRegex(ValueError, 'replacement-labels-dir'):
                self.update.build_update_plan([row], source, root)
            labels = base / 'reviewed/train'; labels.mkdir(parents=True)
            new_box = b'0 0.5 0.5 0.2 0.2\n'; (labels / 'face_abcdef01.txt').write_bytes(new_box)
            images, annotations, _ = self.update.build_update_plan([row], source, root, replacement_labels_dir=labels.parent)
            self.assertEqual(annotations['labels/train/face_abcdef01.txt'], new_box)
            result = apply_dataset_changes(root, images, annotations)
            self.assertEqual(result['image_counts'], {'train': 1, 'val': 1})
            self.assertEqual((root / 'labels/train/face_abcdef01.txt').read_bytes(), new_box)


if __name__ == '__main__':
    unittest.main()
