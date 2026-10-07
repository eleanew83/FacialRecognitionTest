#!/usr/bin/env bash
set -euo pipefail

# Crop datasets must be validated and published into a fresh destination.
# Existing datasets are never merged into or deleted by this helper.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MACAQUE_PYTHON:-python3}"
CROP_SOURCE="${MACAQUE_CROP_SOURCE:-/home/ylj20/FacialRecognitionTest/yolo_detection/yolo_detection_code/output/macaque_crops/}"
CROP_DESTINATION="${MACAQUE_CROP_DESTINATION:-ylj20@login.hpc.cam.ac.uk:/home/ylj20/rds/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_code/output/macaque_crops/}"

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo 'Use the project Python environment (3.10+), or set MACAQUE_PYTHON to its interpreter.' >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/animal-face-id/tools/sync_macaque_crops.py" \
  --source "${CROP_SOURCE}" --destination "${CROP_DESTINATION}" "$@"
