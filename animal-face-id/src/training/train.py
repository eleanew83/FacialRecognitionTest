"""Closed-set training entry points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import optim

from src.config.base import TrainingConfig, load_config
from src.datasets.dataset_registry import build_dataloader
from src.models.backbones import build_backbone
from src.models.losses import build_classifier_head, build_loss


def _prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_topk(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:
    """Return top-k correct counts keyed by k."""
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = {}
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results[k] = correct_k.item()
    return results


def _run_phase(
    model,
    head,
    dataloader,
    criterion,
    device: torch.device,
    head_type: str,
    optimizer=None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    head.train(train_mode)

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model(images)
            if head_type == "arcface":
                logits = head(embeddings, labels)
            else:
                logits = head(embeddings)
            loss = criterion(logits, labels)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
                optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        topk = _compute_topk(logits, labels, topk=(1, 5))
        total_top1 += topk[1]
        total_top5 += topk.get(5, 0.0)

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "top1": total_top1 / max(total_samples, 1),
    }
    if dataloader.dataset and getattr(dataloader.dataset, "class_to_idx", None):  # type: ignore[attr-defined]
        num_classes = len(getattr(dataloader.dataset, "class_to_idx"))
    else:
        num_classes = None
    if num_classes is None or num_classes >= 5:
        metrics["top5"] = total_top5 / max(total_samples, 1)
    return metrics


def _save_checkpoint(
    path: Path,
    model,
    head,
    optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "best_top1": best_metric,
        "model_state": model.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _append_metrics_csv(
    csv_path: Path,
    epoch: int,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["epoch", "lr", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5"]
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_metrics.get("loss"),
        "train_top1": train_metrics.get("top1"),
        "train_top5": train_metrics.get("top5"),
        "val_loss": val_metrics.get("loss"),
        "val_top1": val_metrics.get("top1"),
        "val_top5": val_metrics.get("top5"),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_closed_set(config: dict[str, Any] | TrainingConfig) -> None:
    """Run the closed-set baseline training loop."""
    if isinstance(config, TrainingConfig):
        config = config.as_dict()
    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]
    run_name = config.get("name", "chimp-face-id")

    _set_seed(trainer_cfg.get("seed"))

    train_loader = build_dataloader(
        name=data_cfg.get("dataset_name", "chimpanzee_faces"),
        split="train",
        config=data_cfg,
    )
    val_loader = build_dataloader(
        name=data_cfg.get("dataset_name", "chimpanzee_faces"),
        split="val",
        config=data_cfg,
    )

    model = build_backbone(
        model_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        pretrained=model_cfg.get("pretrained", True),
    )
    head_type = model_cfg.get("head", "linear")
    head = build_classifier_head(
        head_type=head_type,
        embedding_dim=model_cfg.get("embedding_dim", 256),
        num_classes=data_cfg.get("num_classes", 1),
        margin=model_cfg.get("margin", 0.5),
        scale=model_cfg.get("scale", 30.0),
    )
    criterion = build_loss(trainer_cfg.get("loss", "cross_entropy"))

    device = _prepare_device(trainer_cfg.get("device", "cuda"))
    model.to(device)
    head.to(device)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(
        params,
        lr=trainer_cfg.get("lr", 3e-4),
        weight_decay=trainer_cfg.get("weight_decay", 0.01),
    )
    max_epochs = trainer_cfg.get("max_epochs", 10)

    scheduler = None
    scheduler_name = trainer_cfg.get("scheduler", None)
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=trainer_cfg.get("step_size", 10),
            gamma=trainer_cfg.get("gamma", 0.1),
        )

    use_amp = bool(trainer_cfg.get("amp", device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = trainer_cfg.get("grad_clip", 1.0)

    checkpoint_dir = Path(trainer_cfg.get("checkpoint_dir", "artifacts"))
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    last_path = checkpoint_dir / f"{run_name}_last.pt"
    metrics_csv = checkpoint_dir / f"{run_name}_metrics.csv"
    best_top1 = 0.0

    for epoch in range(1, max_epochs + 1):
        train_metrics = _run_phase(
            model=model,
            head=head,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            head_type=head_type,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            use_amp=use_amp,
        )
        val_metrics = _run_phase(
            model=model,
            head=head,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            head_type=head_type,
            optimizer=None,
            scaler=None,
            grad_clip=None,
            use_amp=False,
        )

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{max_epochs}] "
            f"lr={lr_current:.5f} | "
            f"train_loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} top1={val_metrics['top1']:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        _append_metrics_csv(metrics_csv, epoch, lr_current, train_metrics, val_metrics)

        # Save checkpoints
        if val_metrics["top1"] >= best_top1:
            best_top1 = val_metrics["top1"]
            _save_checkpoint(best_path, model, head, optimizer, epoch, best_top1, config)
        _save_checkpoint(last_path, model, head, optimizer, epoch, best_top1, config)

    print(f"Best val top-1: {best_top1:.4f}")
    print(f"Checkpoints saved to: {best_path} (best), {last_path} (last)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train chimpanzee face ID model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_chimp_min10.yaml",
        help="Path to YAML config.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_closed_set(cfg)


if __name__ == "__main__":
    main()
