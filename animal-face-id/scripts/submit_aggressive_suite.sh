#!/bin/bash
# Submit three training array tasks plus one dependent rotation-TTA comparison.
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$BASE"
PYTHON_BIN="${MACAQUE_PYTHON:-/rds/user/ylj20/hpc-work/venvs/macaque/bin/python}"
mkdir -p "$BASE/artifacts/augmentation_sweeps"
RUN_DIR=$(mktemp -d "$BASE/artifacts/augmentation_sweeps/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
mkdir -p "$RUN_DIR/configs" "$RUN_DIR/checkpoints" "$RUN_DIR/logs"
"$PYTHON_BIN" - "$BASE" "$RUN_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import yaml

base, run = map(Path, sys.argv[1:])
for profile in ("rot30", "rot100", "rot150"):
    source = base / f"configs/train_macaque_arcface_aggressive_{profile}.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg["trainer"]["checkpoint_dir"] = str(run / "checkpoints")
    cfg["inference"]["checkpoint"] = str(run / "checkpoints" / f"{cfg['name']}_best.pt")
    cfg["inference"]["gallery_path"] = str(run / f"{cfg['name']}_gallery.pkl")
    (run / "configs" / source.name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    if profile == "rot30":
        inputs = {key: {"path": cfg["data"][key],
                        "sha256": hashlib.sha256(Path(cfg["data"][key]).read_bytes()).hexdigest()}
                  for key in ("splits_path", "identity_registry")}
        (run / "data_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
PY
TRAIN_ID=$(sbatch --parsable --output="$RUN_DIR/logs/%A_%a.out" \
    --error="$RUN_DIR/logs/%A_%a.err" \
    scripts/run_aggressive_train_gpu.sh "$RUN_DIR" "$BASE")
TRAIN_ID="${TRAIN_ID%%;*}"
printf 'Training array: %s (tasks 0=rot30, 1=rot100, 2=rot150)\n' "$TRAIN_ID"
printf 'Training array: %s\n' "$TRAIN_ID" > "$RUN_DIR/jobs.txt"
EVAL_ID=$(sbatch --parsable --dependency="afterok:$TRAIN_ID" \
    --output="$RUN_DIR/logs/%j_tta.out" --error="$RUN_DIR/logs/%j_tta.err" \
    scripts/run_rotation_tta_gpu.sh "$RUN_DIR" "$BASE")
EVAL_ID="${EVAL_ID%%;*}"
printf 'Rotation-TTA job: %s (waits for all three training tasks)\n' "$EVAL_ID"
printf 'Rotation-TTA job: %s\n' "$EVAL_ID" >> "$RUN_DIR/jobs.txt"
printf 'Outputs: %s\nMonitor: squeue -j %s,%s\n' "$RUN_DIR" "$TRAIN_ID" "$EVAL_ID"
