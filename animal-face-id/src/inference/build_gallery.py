"""Export embeddings for a split and build a gallery index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

from src.config.base import TrainingConfig, load_config
from src.datasets.dataset_registry import build_dataloader
from src.inference.knn import EmbeddingIndex
from src.models.backbones import build_backbone


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


def _resolve_label_names(batch: dict, dataset) -> List[str]:
    # Prefer explicit id strings if provided by the dataset.
    if "id" in batch:
        ids = batch["id"]
        if isinstance(ids, list):
            return ids
        # When collated into tensors, strings remain as list of strings.
        return list(ids)
    # Fallback: map label indices to class names if available.
    labels = batch["label"].tolist()
    if hasattr(dataset, "classes"):
        classes = list(dataset.classes)
        return [classes[idx] for idx in labels]
    return [str(idx) for idx in labels]


def export_embeddings(
    config: dict[str, Any] | TrainingConfig,
    checkpoint_path: str,
    split: str = "train",
    output_npz: str | None = None,
    device_str: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    if isinstance(config, TrainingConfig):
        config = config.as_dict()
    data_cfg = config["data"]
    device = torch.device(device_str)

    loader = build_dataloader(
        name=data_cfg.get("dataset_name", "chimpanzee_faces"),
        split=split,
        config=data_cfg,
        shuffle=False,
        drop_last=False,
    )
    model = _load_backbone_from_checkpoint(config, checkpoint_path, device)

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            feats = model(images).cpu().numpy()
            embeddings.append(feats)
            labels.extend(_resolve_label_names(batch, loader.dataset))

    all_embeddings = np.concatenate(embeddings, axis=0)

    if output_npz:
        np.savez(output_npz, embeddings=all_embeddings, labels=np.array(labels))

    return all_embeddings, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gallery index from a trained checkpoint.")
    parser.add_argument("--config", default="configs/train_chimp_min10.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load (default: best checkpoint).")
    parser.add_argument("--split", default="train", help="Dataset split to export embeddings from.")
    parser.add_argument("--output", default=None, help="Optional .npz path to save embeddings + labels.")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu or cuda).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inference_cfg = cfg.inference or {} if isinstance(cfg, TrainingConfig) else cfg.get("inference", {})
    default_ckpt = "artifacts/chimp-min10-resnet18_best.pt"
    ckpt_path = args.checkpoint or inference_cfg.get("checkpoint", default_ckpt)
    output_npz = args.output or f"artifacts/{args.split}_embeddings.npz"

    embeddings, labels = export_embeddings(
        config=cfg,
        checkpoint_path=ckpt_path,
        split=args.split,
        output_npz=output_npz,
        device_str=args.device,
    )

    knn_cfg = inference_cfg.get("knn", {}) if inference_cfg else {}
    index = EmbeddingIndex(metric=knn_cfg.get("metric", "cosine"), top_k=knn_cfg.get("top_k", 5))
    index.fit(embeddings, labels)

    gallery_path = inference_cfg.get("gallery_path", "artifacts/gallery_index.pkl")
    Path(gallery_path).parent.mkdir(parents=True, exist_ok=True)
    index.save(gallery_path)

    print(f"Saved embeddings to: {output_npz}")
    print(f"Saved gallery index to: {gallery_path}")
    print(f"Total embeddings: {len(labels)}")


if __name__ == "__main__":
    main()
