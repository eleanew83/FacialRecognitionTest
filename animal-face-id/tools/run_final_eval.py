#!/usr/bin/env python3
"""Final evaluation entry point for chimpanzee faces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.base import TrainingConfig, load_config
from src.datasets.dataset_registry import build_dataloader
from src.models.backbones import build_backbone
from src.models.losses import build_classifier_head
from src.utils.metrics import compute_basic_metrics, compute_confusion, per_class_report, topk_accuracies
from src.utils.plotting import plot_confusion_matrix, plot_per_class_accuracy, plot_topk


def _prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model_cfg = config["model"]
    data_cfg = config["data"]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_backbone(
        model_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        pretrained=False,
    )
    head = build_classifier_head(
        head_type=model_cfg.get("head", "linear"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        num_classes=data_cfg.get("num_classes", 1),
        margin=model_cfg.get("margin", 0.5),
        scale=model_cfg.get("scale", 30.0),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    model.to(device)
    head.to(device)
    model.eval()
    head.eval()
    return model, head


def _collect_outputs(
    model,
    head,
    dataloader,
    device: torch.device,
    head_type: str,
) -> Dict[str, Any]:
    all_labels: List[int] = []
    all_logits: List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []
    all_paths: List[str] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            embeddings = model(images)
            if head_type == "arcface":
                logits = head(embeddings, labels)
            else:
                logits = head(embeddings)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.append(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            if "path" in batch:
                all_paths.extend(batch["path"])
            else:
                all_paths.extend([""] * labels.shape[0])
    return {
        "labels": np.array(all_labels, dtype=np.int64),
        "logits": np.concatenate(all_logits, axis=0),
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "paths": all_paths,
    }


def _tsne_or_pca(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D embeddings and labels for plotting (TSNE with PCA fallback)."""
    if embeddings.shape[0] < 2:
        return embeddings[:, :2], labels
    try:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, embeddings.shape[0] - 1))
        reduced = tsne.fit_transform(embeddings)
    except Exception:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
    return reduced, labels


def _plot_embeddings_2d(emb2d: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, s=8, cmap="tab20")
    plt.title("Test Embeddings (t-SNE/PCA)")
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_per_class_csv(per_class: Dict[str, Dict[str, float]], out_path: Path) -> None:
    import csv

    rows = []
    for class_id, stats in per_class.items():
        rows.append(
            {
                "class_id": class_id,
                "precision": stats.get("precision", 0.0),
                "recall": stats.get("recall", 0.0),
                "f1": stats.get("f1-score", 0.0),
                "num_samples": stats.get("support", 0),
                "accuracy": stats.get("accuracy", stats.get("recall", 0.0)),  # recall acts as per-class acc
            },
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "num_samples", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def _write_hard_examples(
    labels: np.ndarray,
    logits: np.ndarray,
    paths: List[str],
    class_names: List[str],
    out_path: Path,
) -> None:
    import csv

    preds = logits.argmax(axis=1)
    top1_probs = logits.max(axis=1)
    rows = []
    for lbl, pred, prob, path in zip(labels, preds, top1_probs, paths):
        rows.append(
            {
                "image_path": path,
                "true_id": class_names[lbl] if lbl < len(class_names) else str(lbl),
                "pred_id": class_names[pred] if pred < len(class_names) else str(pred),
                "top1_prob": prob,
                "is_correct": int(lbl == pred),
            },
        )
    rows.sort(key=lambda r: (r["is_correct"], r["top1_prob"]))  # errors first, then low confidence
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_id", "pred_id", "top1_prob", "is_correct"])
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_report(
    summary: Dict[str, Any],
    files: Dict[str, Path],
    class_acc_sorted: list[tuple[str, float]],
) -> None:
    report_path = Path("FINAL_EVAL_REPORT.md")
    soft_topk = summary.get("topk", {})
    content = [
        "# Final Evaluation Report — Chimpanzee Faces (ResNet50 + ArcFace)",
        "",
        f"- Config: `{summary['config']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Date: {summary['timestamp']}",
        f"- Device: {summary['device']}",
        f"- Test samples: {summary['num_samples']}",
        f"- Num classes: {summary['num_classes']}",
        "",
        "## 1. Overall Metrics",
        f"- Top-1 accuracy: {summary['top1']:.4f}",
        f"- Top-3 accuracy: {soft_topk.get(3, 0.0):.4f}",
        f"- Top-5 accuracy: {soft_topk.get(5, 0.0):.4f}",
        f"- Macro F1: {summary['f1_macro']:.4f}",
        f"- Weighted F1: {summary['f1_weighted']:.4f}",
        "",
        "## 2. Per-class Summary",
        f"- Per-class metrics CSV: `{files['per_class_csv']}`",
        f"- Confusion matrix: `{files['conf_png']}`",
        "",
    ]
    if class_acc_sorted:
        hardest = class_acc_sorted[:5]
        content.append("- Hardest IDs by accuracy:")
        for cid, acc in hardest:
            content.append(f"  - {cid}: acc = {acc:.3f}")
        content.append("")

    content.extend(
        [
            "## 3. Embedding Space Visualization",
            f"- t-SNE / PCA plot: `{files.get('tsne_png', '')}`",
            "",
            "## 4. Known Limitations / Notes",
            "- Test set size is modest; per-class variation may influence stability.",
            "- Some individuals may have low support; interpret per-class metrics accordingly.",
            "- Consider augmentations/backbone/loss tuning for further gains.",
            "",
        ],
    )
    report_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on test split.")
    parser.add_argument("--config", required=True, help="Path to training config yaml.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to evaluate.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation.")
    args = parser.parse_args()

    cfg_obj = load_config(args.config)
    config = cfg_obj.as_dict() if isinstance(cfg_obj, TrainingConfig) else cfg_obj
    data_cfg = dict(config["data"])
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size

    device = _prepare_device(args.device)

    # Build dataloader
    test_loader = build_dataloader(
        name=data_cfg.get("dataset_name", "chimpanzee_faces"),
        split="test",
        config=data_cfg,
        shuffle=False,
        drop_last=False,
    )
    class_names = list(getattr(test_loader.dataset, "classes", [str(i) for i in range(data_cfg.get("num_classes", 0))]))

    # Load model + head
    model, head = _load_checkpoint(config, args.ckpt, device)

    outputs = _collect_outputs(
        model=model,
        head=head,
        dataloader=test_loader,
        device=device,
        head_type=config["model"].get("head", "linear"),
    )

    labels = outputs["labels"]
    logits = outputs["logits"]
    embeddings = outputs["embeddings"]
    paths = outputs["paths"]

    # Metrics
    basic = compute_basic_metrics(logits, labels)
    topk = topk_accuracies(logits, labels, ks=(1, 3, 5))
    cm = compute_confusion(logits, labels)
    per_class = per_class_report(logits, labels, class_names)

    # Prepare output paths
    out_dir = Path("artifacts/final_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        Path(args.config).stem.replace(".yaml", "")
        + "_"
        + Path(args.ckpt).stem.replace(".pt", "")
    )
    summary_path = out_dir / f"{stem}_metrics_summary.json"
    per_class_csv = out_dir / f"{stem}_per_class_metrics.csv"
    conf_npy = out_dir / f"{stem}_confusion_matrix.npy"
    conf_png = out_dir / f"{stem}_confusion_matrix.png"
    topk_png = out_dir / f"{stem}_topk_accuracy.png"
    tsne_png = out_dir / f"{stem}_embeddings_tsne.png"
    hard_csv = out_dir / f"{stem}_hard_examples.csv"
    per_class_bar_png = out_dir / f"{stem}_per_class_accuracy_bar.png"

    # Save numerical artifacts
    summary_payload = {
        "config": args.config,
        "checkpoint": args.ckpt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "num_samples": int(len(labels)),
        "num_classes": int(logits.shape[1]),
        "top1": basic["top1"],
        "topk": topk,
        "f1_macro": basic["f1_macro"],
        "f1_weighted": basic["f1_weighted"],
        "precision_macro": basic["precision_macro"],
        "recall_macro": basic["recall_macro"],
        "precision_weighted": basic["precision_weighted"],
        "recall_weighted": basic["recall_weighted"],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_per_class_csv(per_class, per_class_csv)
    np.save(conf_npy, cm)
    _write_hard_examples(labels, logits, paths, class_names, hard_csv)

    # Plots
    plot_confusion_matrix(cm, class_labels=class_names, out_path=str(conf_png))
    plot_topk(topk, out_path=str(topk_png))

    class_acc_sorted = sorted(
        [(cid, stats.get("recall", 0.0)) for cid, stats in per_class.items()],
        key=lambda x: x[1],
    )
    plot_per_class_accuracy(class_acc_sorted, out_path=str(per_class_bar_png))

    try:
        emb2d, emb_labels = _tsne_or_pca(embeddings, labels)
        _plot_embeddings_2d(emb2d, emb_labels, out_path=str(tsne_png))
    except Exception:
        tsne_png = None

    # Generate Markdown report
    files = {
        "per_class_csv": per_class_csv,
        "conf_png": conf_png,
        "tsne_png": tsne_png or "",
    }
    _write_summary_report(summary_payload, files, class_acc_sorted)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-class metrics: {per_class_csv}")
    print(f"Saved confusion matrix: {conf_npy} and {conf_png}")
    print(f"Saved top-k plot: {topk_png}")
    if tsne_png:
        print(f"Saved embeddings plot: {tsne_png}")
    print(f"Saved hard examples: {hard_csv}")
    print("Report updated: FINAL_EVAL_REPORT.md")


if __name__ == "__main__":
    main()
