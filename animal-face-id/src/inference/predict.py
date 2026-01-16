"""Prediction-facing helpers (CLI/FastAPI can import these)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config.base import TrainingConfig, load_config
from src.datasets.transforms import build_transforms
from src.inference.knn import EmbeddingIndex
from src.models.backbones import build_backbone


def _load_gallery(index_path: str) -> EmbeddingIndex:
    if not Path(index_path).exists():
        msg = (
            f"Gallery index '{index_path}' not found. "
            "Run the embedding export script after training to build it."
        )
        raise FileNotFoundError(msg)
    return EmbeddingIndex.load(index_path)


def _load_backbone_from_checkpoint(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_backbone(
        model_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        pretrained=False,
    )
    if "model_state" not in checkpoint:
        msg = f"Checkpoint {checkpoint_path} missing 'model_state'."
        raise KeyError(msg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def _prepare_embedding(image_path: str, model: torch.nn.Module, transform, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy()[0]
    return embedding


def predict_image(
    image_path: str,
    config: dict[str, Any] | TrainingConfig,
    checkpoint_path: str | None = None,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Return ID prediction for one image using gallery k-NN."""
    if isinstance(config, TrainingConfig):
        config = config.as_dict()
    model_cfg = config["model"]
    inference_cfg = config.get("inference", {})
    transforms = build_transforms(stage="val", config=config["data"])
    device = torch.device(device_str)

    ckpt_path = checkpoint_path or inference_cfg.get("checkpoint", "artifacts/chimp-min10-resnet18_best.pt")
    model = _load_backbone_from_checkpoint(config, ckpt_path, device)

    gallery_index = _load_gallery(inference_cfg.get("gallery_path", "artifacts/gallery_index.pkl"))
    embedding = _prepare_embedding(image_path, model, transforms, device)
    neighbors = gallery_index.query(embedding)
    best_id, distance = neighbors[0]
    return {
        "image": image_path,
        "predicted_id": best_id,
        "distance": distance,
        "neighbors": neighbors,
        "backbone": model_cfg.get("backbone", "resnet18"),
        "checkpoint": ckpt_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict animal ID for a cropped face.")
    parser.add_argument("--image", required=True, help="Path to the image to evaluate.")
    parser.add_argument("--config", default="configs/train_chimp_min10.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint (defaults to best).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = predict_image(args.image, config=cfg, checkpoint_path=args.checkpoint, device_str=args.device)
    print(result)


if __name__ == "__main__":
    main()
