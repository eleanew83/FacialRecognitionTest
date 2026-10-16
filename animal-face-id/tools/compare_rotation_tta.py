#!/usr/bin/env python3
"""Compare three trained models with single-view, mean, and max rotation TTA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.base import load_config
from src.training.evaluate import run_final_evaluation
from src.training.tta import ROTATION_ANGLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in args.configs:
        config = load_config(path)
        checkpoint = Path(config.trainer.get("checkpoint_dir", "artifacts")) / f"{config.name}_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training must finish first: {checkpoint}")
        runs.append((path, config.name, checkpoint))

    rows = []
    fields = ["run", "split", "tta_mode", "top1", "top3", "top5", "f1_macro", "num_samples", "checkpoint"]
    # Complete all validation comparisons and lock the choice before scoring test.
    for split in ("val", "test"):
        for config_path, name, checkpoint in runs:
            for mode in ("none", "mean", "max"):
                print(f"Evaluating {name}: split={split}, TTA={mode}", flush=True)
                summary = run_final_evaluation(
                    config=config_path, ckpt=str(checkpoint), device=args.device,
                    batch_size=args.batch_size, split=split, tta_mode=mode,
                    out_dir=out_dir, tag=f"{name}_{mode}",
                    make_plots=False, write_report=False,
                )
                rows.append({
                    "run": name, "split": split, "tta_mode": mode,
                    "top1": summary["top1"], "top3": summary["topk"][3],
                    "top5": summary["topk"][5], "f1_macro": summary["f1_macro"],
                    "num_samples": summary["num_samples"], "checkpoint": str(checkpoint),
                })
                # Save progress after each result, including during long eval jobs.
                with (out_dir / "comparison.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        if split == "val":
            selected = max(rows, key=lambda row: (row["top1"], row["f1_macro"]))
            (out_dir / "selection.json").write_text(json.dumps({
                "selection_split": "val", "selection_metric": "top1; macro F1 breaks ties",
                "angles": list(ROTATION_ANGLES), "selected": selected,
            }, indent=2) + "\n")
            print(f"Selected on validation: {selected['run']} / {selected['tta_mode']}", flush=True)

    selected_test = next(row for row in rows if row["split"] == "test"
                         and row["run"] == selected["run"] and row["tta_mode"] == selected["tta_mode"])
    print(f"Validation-selected test result: top1={selected_test['top1']:.4f}, "
          f"macro F1={selected_test['f1_macro']:.4f}")
    print(f"Comparison: {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
