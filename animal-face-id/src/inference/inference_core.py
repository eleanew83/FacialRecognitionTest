"""Core inference helpers for loading models and running single-image predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config.base import TrainingConfig, load_config
from src.datasets.chimpanzee_faces import ChimpanzeeFacesDataset
from src.datasets.transforms import build_transforms
from src.models.backbones import build_backbone
from src.models.losses import ArcFaceHead, build_classifier_head


def _prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model_and_head(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, str]:
    model_cfg = config["model"]
    data_cfg = config["data"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_backbone(
        model_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        pretrained=False,
    )
    head_type = model_cfg.get("head", "linear")
    head = build_classifier_head(
        head_type=head_type,
        embedding_dim=model_cfg.get("embedding_dim", 256),
        num_classes=data_cfg.get("num_classes", 1),
        margin=model_cfg.get("margin", 0.5),
        scale=model_cfg.get("scale", 30.0),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    model.to(device).eval()
    head.to(device).eval()
    return model, head, head_type


def _load_class_names(data_cfg: dict[str, Any]) -> List[str]:
    dataset = ChimpanzeeFacesDataset(
        raw_root=data_cfg["raw_root"],
        splits_path=data_cfg["splits_path"],
        split="train",
        transform=None,
    )
    return list(dataset.classes)


def load_model_from_config(config_path: str, ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load model/head/transform + metadata for inference."""
    cfg_obj = load_config(config_path)
    config = cfg_obj.as_dict() if isinstance(cfg_obj, TrainingConfig) else cfg_obj
    device_t = _prepare_device(device)
    model, head, head_type = _load_model_and_head(config, ckpt_path, device_t)
    transform = build_transforms(stage="val", config=config["data"])
    class_names = _load_class_names(config["data"])
    return {
        "model": model,
        "head": head,
        "head_type": head_type,
        "transform": transform,
        "device": device_t,
        "class_names": class_names,
        "config": config,
    }


def _compute_logits_from_head(embeddings: torch.Tensor, head: torch.nn.Module, head_type: str) -> torch.Tensor:
    if head_type == "arcface" and isinstance(head, ArcFaceHead):
        norm_emb = F.normalize(embeddings)
        norm_w = F.normalize(head.weight)
        logits = F.linear(norm_emb, norm_w) * head.scale
        return logits
    return head(embeddings)


def infer_single_image(model_bundle: Dict[str, Any], image_path: str, topk: int = 5) -> Dict[str, Any]:
    """Run inference on one image path and return top-k predictions and embedding."""
    model = model_bundle["model"]
    head = model_bundle["head"]
    head_type = model_bundle["head_type"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    class_names = model_bundle.get("class_names", [])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embeddings = model(tensor)
        logits = _compute_logits_from_head(embeddings, head, head_type)
        probs = torch.softmax(logits, dim=1)

    topk = min(topk, probs.shape[1])
    values, indices = torch.topk(probs, k=topk, dim=1)
    indices_list = indices[0].cpu().tolist()
    scores_list = values[0].cpu().tolist()
    labels_list = [class_names[idx] if idx < len(class_names) else str(idx) for idx in indices_list]

    return {
        "image": image_path,
        "embedding": embeddings[0].detach().cpu().numpy(),
        "topk_indices": indices_list,
        "topk_labels": labels_list,
        "topk_scores": scores_list,
        "logits": logits[0].detach().cpu().numpy(),
    }


def infer_image_pil(model_bundle: Dict[str, Any], image: Image.Image, topk: int = 5) -> Dict[str, Any]:
    """Run inference on an in-memory PIL image."""
    temp_path = Path("__in_memory_placeholder.png")
    image.save(temp_path)
    try:
        return infer_single_image(model_bundle, str(temp_path), topk=topk)
    finally:
        if temp_path.exists():
            temp_path.unlink()
