#!/bin/bash
# Add detector-only packages without changing the shared recognition environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YOLO_BASE="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON=/rds/user/ylj20/hpc-work/venvs/macaque/bin/python
DEPS="$YOLO_BASE/models/python_deps/ultralytics-8.4.14"
mkdir -p "$YOLO_BASE/models/slurm_logs" "$YOLO_BASE/models/ultralytics_config"

# Core libraries (including CUDA PyTorch) come from the existing environment.
# --no-deps prevents pip from replacing or downloading those libraries.
"$PYTHON" -m pip install --disable-pip-version-check --no-deps --upgrade \
    --target "$DEPS" --requirement "$SCRIPT_DIR/requirements_detection.txt"

PYTHONPATH="$DEPS" YOLO_CONFIG_DIR="$YOLO_BASE/models/ultralytics_config" \
    YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 "$PYTHON" -u -B - <<'PY'
import ultralytics
import polars
import thop
print(f"Detector dependencies ready: ultralytics={ultralytics.__version__}, polars={polars.__version__}")
PY
