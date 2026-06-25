#!/usr/bin/env python3
"""CLI wrapper for final evaluation on the test split.

All logic lives in ``src/training/evaluate.py``; this just parses args and calls
:func:`run_final_evaluation`.

    python tools/run_final_eval.py --config configs/train_macaque_arcface_ltr.yaml \\
        --ckpt artifacts/macaque-resnet50-arcface_ltr_best.pt --device cuda \\
        [--logit-adjust-tau 1.0 --tag logitadj]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.evaluate import run_final_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on test split.")
    parser.add_argument("--config", required=True, help="Path to training config yaml.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to evaluate.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation.")
    parser.add_argument("--logit-adjust-tau", type=float, default=0.0,
                        help="Post-hoc logit adjustment strength (0 = off; 1.0 typical). "
                             "Subtracts tau*log(train prior) from logits to boost rare IDs.")
    parser.add_argument("--tag", default=None, help="Optional suffix for output filenames (e.g. 'logitadj').")
    args = parser.parse_args()

    summary = run_final_evaluation(
        config=args.config,
        ckpt=args.ckpt,
        device=args.device,
        batch_size=args.batch_size,
        logit_adjust_tau=args.logit_adjust_tau,
        tag=args.tag,
    )

    print(f"Top-1={summary['top1']:.4f}  Macro-F1={summary['f1_macro']:.4f}  "
          f"Macro-P={summary['precision_macro']:.4f}  Macro-R={summary['recall_macro']:.4f}")
    print(f"Per-class metrics: {summary['_per_class_csv']}")


if __name__ == "__main__":
    main()
