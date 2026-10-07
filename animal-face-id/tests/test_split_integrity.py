"""Regression checks for duplicate-safe macaque dataset preparation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.split_integrity import (
    SPLIT_ORDER, file_fingerprint, group_photos, partition_photos, source_photo_key, validate_manifest,
)


def record(split, label, filename, content):
    return {"id": label, "path": f"{split}/{label}/{filename}", "split": split,
            "sha256": content, "pixels_sha256": content}


class SplitIntegrityTests(unittest.TestCase):
    def test_source_versions_match_but_frames_and_dates_remain_distinct(self):
        self.assertEqual(source_photo_key("PHOTO_cropped_crop0.JPG"),
                         source_photo_key("photo.jpg"))
        self.assertEqual(source_photo_key("photo_abcdef12.jpg"), "photo")
        self.assertEqual(source_photo_key("photo.JPG_crop1.JPG"), "photo")
        self.assertNotEqual(source_photo_key("photo_1.jpg"), source_photo_key("photo_2.jpg"))
        self.assertNotEqual(source_photo_key("photo_20221230.jpg"),
                            source_photo_key("photo_20221231.jpg"))

    def test_groups_match_renamed_content_and_shared_photos_across_labels(self):
        rows = [record("train", "A", "photo.jpg", "x"),
                record("val", "A", "renamed.jpg", "x"),
                record("test", "B", "photo_cropped.jpg", "y")]
        self.assertEqual(len(group_photos(rows)), 2)
        self.assertEqual(len(group_photos(rows, include_source_names=True)), 1)

    def test_loader_accepts_distinct_crops_with_matching_source_names(self):
        payload = {"train": [record("train", "A", "photo.jpg", "one")],
                   "val": [record("val", "A", "photo_cropped.jpg", "two")],
                   "test": [record("test", "A", "photo_crop1.jpg", "three")]}
        validate_manifest(payload)
        self.assertEqual(len(group_photos([r for rows in payload.values() for r in rows])), 3)

    def test_loader_rejects_renamed_duplicate_content_and_conflicting_labels(self):
        payload = {"train": [record("train", "A", "one.jpg", "x")],
                   "val": [record("val", "A", "two.jpg", "x")], "test": []}
        with self.assertRaises(ValueError):
            validate_manifest(payload)
        payload["val"] = []
        payload["train"].append(record("train", "B", "other.jpg", "x"))
        with self.assertRaisesRegex(ValueError, "conflicting identities"):
            validate_manifest(payload)

    def test_partition_is_independent_of_input_listing_order(self):
        rows = [record("train", label, f"{label}_{index}.jpg", f"{label}{index}")
                for label in ["A", "B"] for index in range(8)]
        first = partition_photos(group_photos(rows), (0.7, 0.15, 0.15), 42)
        second = partition_photos(group_photos(list(reversed(rows))), (0.7, 0.15, 0.15), 42)
        self.assertEqual(first, second)
        self.assertTrue(all({r["id"] for r in entries} == {"A", "B"}
                            for entries in first.values()))

    def test_partition_small_identity_fails_before_copying(self):
        with self.assertRaisesRegex(ValueError, "three independent photos"):
            partition_photos(group_photos([record("train", "A", f"p{i}.jpg", str(i))
                                          for i in range(2)]), (0.7, 0.15, 0.15), 42)

    def test_fingerprint_matches_rgb_pixels_despite_different_encodings(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "one.png", Path(temp) / "two.png"
            image = Image.new("RGB", (12, 9), (20, 30, 40))
            image.save(a, compress_level=0); image.save(b, compress_level=9)
            first, second = file_fingerprint(a), file_fingerprint(b)
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["pixels_sha256"], second["pixels_sha256"])

    def test_upstream_refuses_existing_output_and_handles_three_photos(self):
        from PIL import Image
        path = ROOT.parent / "gorillavision/reid-system/scripts/prepare_macaque_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_macaque_dataset", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"; (source / "A").mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (5, 5), (index * 50, 0, 0)).save(source / "A" / f"photo{index}.png")
            target = Path(temp) / "split"
            payload = module.split_dataset(source, target)
            self.assertTrue(all(len(rows) == 1 for rows in payload.values()))
            sentinel = target / "keep.txt"; sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                module.split_dataset(source, target)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_loader_rejects_images_edited_after_content_audit(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {split: [] for split in ["train", "val", "test"]}
            for index, split in enumerate(payload):
                path = root / split / "A" / f"photo{index}.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (5, 5), (index * 40, 0, 0)).save(path)
                payload[split].append({"id": "A", "path": str(path.relative_to(root)),
                                       **file_fingerprint(path)})
            validate_manifest(payload, root)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed after split audit"):
                validate_manifest(payload, root)

    def test_crop_exporter_rejects_stale_outputs_and_leaking_source_splits(self):
        from PIL import Image
        from types import ModuleType
        from unittest.mock import Mock
        fake_yolo = ModuleType("ultralytics"); fake_yolo.YOLO = Mock()
        fake_torch = ModuleType("torch")
        path = ROOT.parent / "yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py"
        spec = importlib.util.spec_from_file_location("train_yolo_detection_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"ultralytics": fake_yolo, "torch": fake_torch}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"; output = Path(temp) / "output"
            for index, split in enumerate(["train", "val", "test"]):
                folder = source / split / "A"; folder.mkdir(parents=True)
                Image.new("RGB", (5, 5), (index * 40, 0, 0)).save(folder / f"photo{index}.png")
            output.mkdir(); sentinel = output / "keep.txt"; sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                module.crop_faces("model.pt", str(output), source_folder=str(source))
            self.assertEqual(sentinel.read_text(), "preserve")
            fresh = Path(temp) / "fresh"
            import shutil
            (source / "val/A/photo1.png").unlink()
            shutil.copy2(source / "train/A/photo0.png", source / "val/A/photo0_cropped.png")
            with self.assertRaisesRegex(ValueError, "Exact duplicate content"):
                module.crop_faces("model.pt", str(fresh), source_folder=str(source))
            self.assertFalse(fresh.exists())
            fake_yolo.YOLO.assert_not_called()

    def test_same_filename_different_identities_and_crops_are_retained(self):
        rows = [record(split, label, "shared.jpg" if split == "train" else f"{split}.jpg", label + split)
                for label in ["A", "B"] for split in SPLIT_ORDER]
        payload = partition_photos(group_photos(rows), (0.7, 0.15, 0.15), 42)
        retained = [row for entries in payload.values() for row in entries]
        self.assertEqual({row["path"] for row in retained}, {row["path"] for row in rows})
        self.assertEqual(len(retained), len(rows))
        self.assertTrue(all({row["id"] for row in entries} == {"A", "B"}
                            for entries in payload.values()))

    def test_loader_rejects_exact_duplicates_within_one_split(self):
        payload = {split: [record(split, "A", f"{split}.jpg", split)] for split in SPLIT_ORDER}
        payload["train"].append(record("train", "A", "copy.jpg", "train"))
        with self.assertRaisesRegex(ValueError, "Exact duplicate content"):
            validate_manifest(payload)

    def test_partition_rejects_exact_pixels_with_two_identity_labels(self):
        rows = [record(split, label, f"{label}_{split}.jpg", label + split)
                for label in ["A", "B"] for split in SPLIT_ORDER]
        rows += [record("train", "A", "shared.jpg", "conflict"),
                 record("val", "B", "other.jpg", "conflict")]
        with self.assertRaisesRegex(ValueError, "conflicting identities"):
            partition_photos(group_photos(rows), (0.7, 0.15, 0.15), 42)

    def test_crop_exporter_checks_output_pixels_and_keeps_distinct_same_named_crops(self):
        import numpy as np
        from PIL import Image
        from types import ModuleType, SimpleNamespace
        fake_yolo = ModuleType("ultralytics")
        box = SimpleNamespace(conf=SimpleNamespace(item=lambda: 0.99),
                              xyxy=[SimpleNamespace(tolist=lambda: [0, 0, 8, 8])])
        fake_yolo.YOLO = lambda _: lambda *args, **kwargs: [SimpleNamespace(boxes=[box])]
        fake_cv = ModuleType("cv2")
        def read_image(path):
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        def write_image(path, array):
            Image.fromarray(array).save(path)
            return True
        fake_cv.imread = read_image
        fake_cv.imwrite = write_image
        path = ROOT.parent / "yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py"
        spec = importlib.util.spec_from_file_location("crop_output_integrity_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"ultralytics": fake_yolo, "torch": ModuleType("torch"), "cv2": fake_cv}):
            spec.loader.exec_module(module)
            for conflicting in (True, False):
                with tempfile.TemporaryDirectory() as temp:
                    base = Path(temp); source = base / "source"; output = base / "crops"
                    for index, (split, label) in enumerate((split, label) for split in SPLIT_ORDER for label in ("A", "B")):
                        folder = source / split / label; folder.mkdir(parents=True)
                        image = Image.new("RGB", (12, 12), (150 + index, 0, 0))
                        # Distinct full images have identical face pixels only in the failure case.
                        color = 10 if conflicting and (split, label) == ("test", "B") else 10 + index * 20
                        image.paste((color, 30, 40), (0, 0, 8, 8))
                        image.save(folder / "shared_filename.png")
                    if conflicting:
                        with self.assertRaisesRegex(ValueError, "conflicting identities"):
                            module.crop_faces("fake.pt", str(output), source_folder=str(source))
                    else:
                        module.crop_faces("fake.pt", str(output), source_folder=str(source))
                        self.assertEqual(len(list(output.glob("*/*/*.png"))), 6)


if __name__ == "__main__":
    unittest.main()
