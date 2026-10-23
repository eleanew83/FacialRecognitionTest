"""Check probability pooling, ArcFace inference, and deterministic rotation views."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.base import load_config
from src.datasets.transforms import GaussianNoise, build_transforms
from src.training.evaluate import _collect_outputs, run_final_evaluation
from src.training.tta import ROTATION_ANGLES, pool_view_probabilities


class RotationTTATests(unittest.TestCase):
    def test_mean_and_max_have_distinct_expected_predictions(self):
        probs = torch.tensor([[[0.90, 0.10], [0.20, 0.80], [0.20, 0.80]]])
        mean = pool_view_probabilities(probs, "mean")
        maximum = pool_view_probabilities(probs, "max")
        self.assertEqual(mean.argmax(1).item(), 1)
        self.assertEqual(maximum.argmax(1).item(), 0)
        torch.testing.assert_close(mean, probs.sum(1) / 3)
        torch.testing.assert_close(maximum.sum(1), torch.ones(1))

    def test_zero_view_matches_original_and_rotations_are_deterministic(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (45, 60, 3), dtype=np.uint8))
        cfg = {"image_size": 32}
        original = build_transforms("test", cfg)(image)
        transform = build_transforms("test", {**cfg, "tta_angles": ROTATION_ANGLES})
        views = transform(image)
        self.assertEqual(views.shape, (7, 3, 32, 32))
        torch.testing.assert_close(views[3], original, rtol=0, atol=0)
        torch.testing.assert_close(views, transform(image), rtol=0, atol=0)
        self.assertFalse(torch.equal(views[0], views[3]))

    def test_arcface_uses_no_margin_and_keeps_seven_outputs_per_image(self):
        class NoMarginHead(torch.nn.Module):
            def forward(self, *args):
                raise AssertionError("Training-only ArcFace margin used during evaluation")

            def logits_eval(self, embeddings):
                return embeddings

        # Each view encodes logits directly, so expected pooled scores are known.
        images = torch.zeros(2, 7, 3, 1, 1)
        images[:, :, 0] = 2
        images[:, 0, 1] = 5
        loader = [{"image": images, "label": torch.tensor([0, 1]), "path": ["a", "b"]}]
        for mode in ("mean", "max"):
            output = _collect_outputs(torch.nn.Flatten(), NoMarginHead(), loader,
                                      torch.device("cpu"), "arcface", tta_mode=mode)
            expected_views = images.flatten(2).softmax(-1)
            torch.testing.assert_close(torch.from_numpy(output["logits"]),
                                       pool_view_probabilities(expected_views, mode))
            self.assertEqual(output["view_probabilities"].shape, (2, 7, 3))
            self.assertEqual(output["labels"].tolist(), [0, 1])
            self.assertEqual(output["paths"], ["a", "b"])

    def test_single_view_and_one_angle_tta_agree(self):
        images = torch.randn(2, 3, 2, 2)
        labels = torch.tensor([0, 1])
        model = torch.nn.Flatten()
        head = torch.nn.Linear(12, 3)
        single = _collect_outputs(model, head, [{"image": images, "label": labels}],
                                  torch.device("cpu"), "linear")
        tta = _collect_outputs(model, head, [{"image": images[:, None], "label": labels}],
                               torch.device("cpu"), "linear", tta_mode="mean")
        np.testing.assert_allclose(single["logits"], tta["logits"], atol=1e-7)

    def test_aggressive_profiles_produce_finite_training_tensors(self):
        root = Path(__file__).resolve().parents[1]
        image = Image.fromarray(np.full((60, 75, 3), 125, dtype=np.uint8))
        for profile in ("rot30", "rot100", "rot150"):
            cfg = load_config(str(root / f"configs/train_macaque_arcface_aggressive_{profile}.yaml"))
            training = build_transforms("train", cfg.data)
            for _ in range(3):
                tensor = training(image)
                self.assertEqual(tensor.shape, (3, 224, 224))
                self.assertTrue(torch.isfinite(tensor).all())
            # Ordinary training validation stays a single, unaugmented image.
            self.assertEqual(build_transforms("val", cfg.data)(image).ndim, 3)
            self.assertEqual(cfg.trainer["max_epochs"], 30)

    def test_noise_fallback_is_stochastic_clipped_and_zero_noise_is_identity(self):
        image = torch.full((3, 32, 32), 0.5)
        torch.testing.assert_close(GaussianNoise(0)(image), image)
        noisy = GaussianNoise(0.3)(image)
        self.assertTrue(((0 <= noisy) & (noisy <= 1)).all())
        self.assertGreater((noisy - image).std().item(), 0.1)

    def test_evaluation_writes_all_seven_views_with_class_and_angle_order(self):
        class Loader(list):
            class dataset:
                classes = ("A", "B", "C", "D", "E")

        images = torch.randn(5, 7, 3, 2, 2)
        loader = Loader([{"image": images, "label": torch.arange(5),
                          "path": [f"image_{i}.jpg" for i in range(5)]}])
        head = torch.nn.Linear(12, 5)
        head.output_class_names = list(loader.dataset.classes)
        config = {"data": {"dataset_name": "animal_faces", "num_classes": 5},
                  "model": {"head": "linear"}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("src.training.evaluate.build_dataloader", return_value=loader) as build, \
                patch("src.training.evaluate._load_checkpoint", return_value=(torch.nn.Flatten(), head)):
            summary = run_final_evaluation(config, "fixture.pt", device="cpu", split="val",
                                           tta_mode="mean", out_dir=tmp,
                                           make_plots=False, write_report=False)
            self.assertEqual(summary["num_samples"], 5)
            self.assertEqual(summary["split"], "val")
            self.assertEqual(build.call_args.kwargs["split"], "val")
            self.assertEqual(build.call_args.kwargs["config"]["tta_angles"], list(ROTATION_ANGLES))
            with np.load(next(Path(tmp).glob("*_view_probabilities.npz"))) as saved:
                self.assertEqual(saved["probabilities"].shape, (5, 7, 5))
                np.testing.assert_array_equal(saved["angles"], ROTATION_ANGLES)
                np.testing.assert_array_equal(saved["class_names"], loader.dataset.classes)
                np.testing.assert_allclose(saved["pooled_probabilities"],
                                           saved["probabilities"].mean(1), atol=1e-7)
            self.assertEqual(len(list(Path(tmp).glob("*_metrics_summary.json"))), 1)
            self.assertFalse(list(Path(tmp).glob("*.png")))


if __name__ == "__main__":
    unittest.main()
