#!/bin/bash
# Evaluate the detector, retrain YOLOv8n, or retrain and export fresh crops.
# From the repository root:
#   bash yolo_detection/yolo_detection_code/scripts/setup_detection_env.sh
#   sbatch yolo_detection/yolo_detection_code/scripts/run_detection_gpu.sh eval
#   sbatch yolo_detection/yolo_detection_code/scripts/run_detection_gpu.sh train
#   sbatch --time=06:00:00 yolo_detection/yolo_detection_code/scripts/run_detection_gpu.sh both
# Optional: DETECTION_DATA_DIR=/absolute/path/to/reviewed_detector_dataset
# Optional for both: DETECTION_SOURCE_SPLITS and DETECTION_CROP_OUTPUT.
# Crop exports retain source splits; existing recognition inputs are preserved.
#SBATCH -J macaque_detect
#SBATCH -A LEMOINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/rds/user/ylj20/hpc-work/FacialRecognitionTest
#SBATCH -o /rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_code/models/slurm_logs/%j.out
#SBATCH -e /rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_code/models/slurm_logs/%j.err

set -euo pipefail

MODE="${1:-eval}"
case "$MODE" in
    eval|train|both) ;;
    -h|--help)
        printf 'Usage: sbatch %s {eval|train|both} [checkpoint-for-eval]\n' "$0"
        printf 'Set DETECTION_DATA_DIR to select a different reviewed detector dataset.\n'
        printf 'Mode both trains, then crops DETECTION_SOURCE_SPLITS into a fresh DETECTION_CROP_OUTPUT.\n'
        printf 'Allow additional time for both stages, e.g. sbatch --time=06:00:00.\n'
        exit 0
        ;;
    *) printf 'Unknown mode: %s (choose eval, train or both)\n' "$MODE" >&2; exit 2 ;;
esac

REPO_ROOT=/rds/user/ylj20/hpc-work/FacialRecognitionTest
YOLO_BASE="$REPO_ROOT/yolo_detection/yolo_detection_code"
PYTHON=/rds/user/ylj20/hpc-work/venvs/macaque/bin/python
DATA_DIR="${DETECTION_DATA_DIR:-$REPO_ROOT/yolo_detection/yolo_detection_data}"
MODEL="${2:-$YOLO_BASE/models/runs/macaque_face_detector_20260120_v1/weights/best.pt}"
JOB_ID="${SLURM_JOB_ID:?Submit this script with sbatch}"
RUN_NAME="macaque_face_detector_$(date +%Y%m%d)_job${JOB_ID}"
SOURCE_SPLITS="${DETECTION_SOURCE_SPLITS:-$REPO_ROOT/macaque_split_data}"
CROP_OUTPUT="${DETECTION_CROP_OUTPUT:-$YOLO_BASE/output/macaque_crops_${RUN_NAME}}"
DETECTION_DEPS="$YOLO_BASE/models/python_deps/ultralytics-8.4.14"

[[ -d "$DETECTION_DEPS/ultralytics" ]] || {
    printf 'Run bash %s/scripts/setup_detection_env.sh before submitting.\n' "$YOLO_BASE" >&2
    exit 1
}
export PYTHONPATH="$DETECTION_DEPS"
export YOLO_CONFIG_DIR="$YOLO_BASE/models/ultralytics_config"
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO_ROOT"
printf 'Job %s | mode=%s | detector data=%s\n' "$JOB_ID" "$MODE" "$DATA_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$MODE" == eval ]]; then
    [[ -f "$MODEL" ]] || { printf 'Checkpoint missing: %s\n' "$MODEL" >&2; exit 1; }
    # Check images, exact duplicates, and box syntax before loading the detector.
    "$PYTHON" -u -B "$YOLO_BASE/scripts/train_yolo_detection.py" \
        --data-dir "$DATA_DIR" --check-data-only
    "$PYTHON" -u -B - "$MODEL" "$DATA_DIR/dataset.yaml" \
        "$YOLO_BASE/models/evaluations" "$RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path
from ultralytics import YOLO

checkpoint, dataset, project, name = sys.argv[1:]
# Ultralytics' cache key does not fully identify annotation contents. Rebuild it
# so this evaluation uses the exact coordinates that passed the integrity check.
for cache in (Path(dataset).parent / "labels").rglob("*.cache"):
    cache.unlink(missing_ok=True)
metrics = YOLO(checkpoint).val(
    data=dataset, split="val", device=0, imgsz=416, batch=16,
    workers=8, project=project, name=name, exist_ok=False, plots=True,
)
summary = {
    "checkpoint": str(Path(checkpoint).resolve()),
    "dataset": str(Path(dataset).resolve()),
    "split": "val",
    "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
}
output = Path(metrics.save_dir)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"Evaluation outputs: {output}")
PY
else
    # The training entrypoint runs the same dataset checks before model loading.
    # Match the existing checkpoint's 100 epochs, batch 4, and 416-pixel images.
    TRAIN_ARGS=(--data-dir "$DATA_DIR" --mode "$MODE" --device 0
        --epochs 100 --batch 4 --img-size 416 --run-name "$RUN_NAME")
    if [[ "$MODE" == both ]]; then
        # Reject unsafe destinations before spending time on detector training.
        "$PYTHON" -u -B - "$SOURCE_SPLITS" "$CROP_OUTPUT" <<'PY'
import sys
from pathlib import Path

source, output = (Path(value).resolve() for value in sys.argv[1:])
if not all((source / split).is_dir() for split in ("train", "val", "test")):
    raise ValueError(f"Source train/val/test folders are missing: {source}")
if source == output or source in output.parents or output in source.parents:
    raise ValueError("Source splits and crop output must be separate, non-nested directories")
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise FileExistsError(f"Crop output is not empty; choose a fresh destination: {output}")
print(f"New crop destination: {output}")
PY
        TRAIN_ARGS+=(--source-split-dir "$SOURCE_SPLITS" --output-dir "$CROP_OUTPUT")
    fi
    "$PYTHON" -u -B "$YOLO_BASE/scripts/train_yolo_detection.py" \
        "${TRAIN_ARGS[@]}"
    printf 'Detector checkpoint: %s/models/runs/%s/weights/best.pt\n' "$YOLO_BASE" "$RUN_NAME"
    if [[ "$MODE" == both ]]; then
        printf 'Verified crop generation: %s\n' "$CROP_OUTPUT"
        printf 'Use its bundled splits.json for a subsequent recognition training run.\n'
    fi
fi
