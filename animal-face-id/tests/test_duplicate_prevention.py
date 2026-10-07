"""Exercise stale folders, image edits, failed builds, and safe crop transfers."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from src.datasets.split_integrity import SPLIT_ORDER, file_fingerprint, fresh_output_directory, validate_manifest
from sync_macaque_crops import receiver_call, sync_dataset


def create_dataset(root, labels=('A',)):
    payload = {split: [] for split in SPLIT_ORDER}
    for index, (split, label) in enumerate((split, label) for split in SPLIT_ORDER for label in labels):
        relative = Path(split) / label / 'same_name.png'
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 12), (index * 30, 40, 60)).save(path)
        payload[split].append({'id': label, 'path': str(relative), **file_fingerprint(path)})
    manifest = root / 'splits.json'
    manifest.write_text(json.dumps(payload))
    return payload, manifest


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DuplicatePreventionTests(unittest.TestCase):
    def test_unlisted_stale_images_block_validation_even_in_nested_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, _ = create_dataset(root)
            stale = root / 'val/A/stale_backup/copy.png'
            stale.parent.mkdir(parents=True)
            shutil.copy2(root / 'train/A/same_name.png', stale)
            with self.assertRaisesRegex(ValueError, 'unlisted=.*stale_backup'):
                validate_manifest(payload, root, verify_content=True)

    def test_cached_verification_detects_edits_with_original_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, _ = create_dataset(root)
            path = root / payload['train'][0]['path']
            # Reserve padding after PNG's IEND so both valid encodings have identical sizes.
            encoded = path.read_bytes()
            path.write_bytes(encoded + b'\0' * (1024 - len(encoded)))
            payload['train'][0].update(file_fingerprint(path))
            validate_manifest(payload, root, verify_content=True)
            original = path.stat()
            Image.new('RGB', (12, 12), (210, 40, 60)).save(path)
            encoded = path.read_bytes()
            path.write_bytes(encoded + b'\0' * (original.st_size - len(encoded)))
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'hash disagrees'):
                validate_manifest(payload, root, verify_content=True)

    def test_legacy_hashless_manifest_cannot_hide_cross_label_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, _ = create_dataset(root, ('A', 'B'))
            shutil.copy2(root / 'train/A/same_name.png', root / 'test/B/same_name.png')
            legacy = {split: [{'id': r['id'], 'path': r['path']} for r in payload[split]] for split in SPLIT_ORDER}
            with self.assertRaisesRegex(ValueError, 'conflicting identities'):
                validate_manifest(legacy, root)

    def test_verified_same_named_distinct_crops_are_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, _ = create_dataset(root, ('A', 'B'))
            validate_manifest(payload, root, verify_content=True)

    def test_failed_generation_is_never_published(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'dataset'
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with fresh_output_directory(target) as stage:
                    (stage / 'partial.txt').write_text('partial')
                    raise RuntimeError('interrupted')
            self.assertFalse(target.exists())
            self.assertTrue((stage / 'partial.txt').is_file())
            self.assertFalse(json.loads((stage / '.generation_failed.json').read_text())['published'])

    def test_concurrent_generations_cannot_use_the_same_target(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'dataset'
            with fresh_output_directory(target) as stage:
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with fresh_output_directory(target):
                        self.fail('Concurrent builder acquired the same target.')
                (stage / 'complete.txt').write_text('done')
                self.assertFalse(target.exists())
            self.assertEqual((target / 'complete.txt').read_text(), 'done')

    def test_interrupted_split_copy_does_not_publish_partial_splits(self):
        module = import_script('atomic_split_test', ROOT.parent / 'gorillavision/reid-system/scripts/prepare_macaque_dataset.py')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source = base / 'source'; (source / 'A').mkdir(parents=True)
            for index in range(3):
                Image.new('RGB', (8, 8), (index * 50, 20, 30)).save(source / 'A' / f'p{index}.png')
            target = base / 'split'
            real_copy = shutil.copy2
            counter = [0]
            def fail_second_copy(*args, **kwargs):
                counter[0] += 1
                if counter[0] == 2:
                    raise OSError('simulated copy failure')
                return real_copy(*args, **kwargs)
            with patch.object(module.shutil, 'copy2', side_effect=fail_second_copy):
                with self.assertRaisesRegex(OSError, 'copy failure'):
                    module.split_dataset(source, target)
            self.assertFalse(target.exists())
            self.assertEqual(len(list(source.glob('A/*.png'))), 3)

    def test_empty_and_identity_dropped_manifests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no images'):
            validate_manifest({split: [] for split in SPLIT_ORDER})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); payload, _ = create_dataset(root)
            with self.assertRaisesRegex(ValueError, 'Identity coverage changed'):
                validate_manifest(payload, root, expected_labels={'A', 'B'})

    def test_crop_failures_and_missing_identities_never_publish(self):
        import numpy as np
        fake_yolo = ModuleType('ultralytics')
        box = SimpleNamespace(conf=SimpleNamespace(item=lambda: 0.99),
                              xyxy=[SimpleNamespace(tolist=lambda: [0, 0, 8, 8])])
        fake_cv = ModuleType('cv2')
        def read_image(path):
            with Image.open(path) as image:
                return np.asarray(image.convert('RGB'))
        fake_cv.imread = read_image
        def write_image(path, array):
            Image.fromarray(array).save(path)
            return True
        module_path = ROOT.parent / 'yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py'
        with patch.dict(sys.modules, {'ultralytics': fake_yolo, 'torch': ModuleType('torch'), 'cv2': fake_cv}):
            for scenario in ('missing_identity', 'write_failure'):
                with tempfile.TemporaryDirectory() as temp:
                    base = Path(temp); source = base / 'source'; create_dataset(source, ('A', 'B'))
                    target = base / 'crops'
                    fake_cv.imwrite = (lambda *args: False) if scenario == 'write_failure' else write_image
                    fake_yolo.YOLO = lambda _: lambda path, **kwargs: [SimpleNamespace(boxes=[] if scenario == 'missing_identity' and Path(path).parent.name == 'B' else [box])]
                    module = import_script('crop_atomic_' + scenario, module_path)
                    expected = 'Crop export failed' if scenario == 'write_failure' else 'Identity coverage changed'
                    with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                        module.crop_faces('fake.pt', str(target), source_folder=str(source))
                    self.assertFalse(target.exists())

    def test_populated_transfer_destination_is_untouched_and_rsync_never_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source = base / 'source'; _, manifest = create_dataset(source)
            destination = base / 'active'; destination.mkdir(); sentinel = destination / 'keep.txt'; sentinel.write_text('preserve')
            real_run = subprocess.run
            def no_rsync(command, *args, **kwargs):
                if command[0] == 'rsync':
                    self.fail('Rsync ran against a populated target.')
                return real_run(command, *args, **kwargs)
            with patch('sync_macaque_crops.subprocess.run', side_effect=no_rsync):
                with self.assertRaisesRegex(RuntimeError, 'Destination is not empty'):
                    sync_dataset(source, str(destination), manifest, retry_delay=0)
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertFalse(list(base.glob('.active.incoming-*')))

    @unittest.skipUnless(shutil.which('rsync'), 'rsync is not installed')
    def test_local_transfer_publishes_verified_manifest_and_no_extra_images(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source = base / 'source'; _, manifest = create_dataset(source, ('A', 'B'))
            destination = base / "new dataset's crops"
            sync_dataset(source, str(destination), manifest, retry_delay=0)
            received = json.loads((destination / 'splits.json').read_text())
            validate_manifest(received, destination, verify_content=True)
            self.assertEqual(len(list(destination.glob('*/*/*.png'))), 6)
            self.assertTrue((destination / 'transfer_verified.json').is_file())
            self.assertFalse((base / ('.' + destination.name + '.transfer.lock')).exists())

    def test_corrupted_transfer_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source = base / 'source'; _, manifest = create_dataset(source)
            destination = base / 'received'
            real_run = subprocess.run
            def corrupt_copy(command, *args, **kwargs):
                if command[0] == 'rsync':
                    stage = Path(command[-1])
                    shutil.copytree(source, stage, dirs_exist_ok=True)
                    Image.new('RGB', (12, 12), (255, 0, 0)).save(stage / 'train/A/same_name.png')
                    return SimpleNamespace(returncode=0)
                return real_run(command, *args, **kwargs)
            with patch('sync_macaque_crops.subprocess.run', side_effect=corrupt_copy):
                with self.assertRaisesRegex(RuntimeError, 'hash disagrees'):
                    sync_dataset(source, str(destination), manifest, retry_delay=0)
            self.assertFalse(destination.exists())
            self.assertTrue(list(base.glob('.received.incoming-*')))

    def test_dry_run_never_connects_or_transfers(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'; _, manifest = create_dataset(source)
            with patch('sync_macaque_crops.subprocess.run') as run:
                sync_dataset(source, 'user@example.com:/fresh/crops', manifest, dry_run=True)
            run.assert_not_called()

    def test_invalid_duplicate_source_blocks_transfer_before_any_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'; payload, manifest = create_dataset(source)
            copy = source / 'test/A/copy.png'; shutil.copy2(source / 'train/A/same_name.png', copy)
            payload['test'].append({'id': 'A', 'path': str(copy.relative_to(source)), **file_fingerprint(copy)})
            manifest.write_text(json.dumps(payload))
            with patch('sync_macaque_crops.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                    sync_dataset(source, 'user@example.com:/fresh/crops', manifest)
            run.assert_not_called()

    def test_remote_paths_are_passed_as_literal_shell_arguments(self):
        target = "/datasets/new run's crops"
        stage = "/datasets/.new run's crops.incoming-123"
        with patch('sync_macaque_crops.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as run:
            receiver_call('user@example.com', 'reserve', target, stage)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ['ssh', 'user@example.com'])
        self.assertEqual(shlex.split(command[2])[-3:], ['reserve', target, stage])


if __name__ == '__main__':
    unittest.main()
