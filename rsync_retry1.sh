#!/usr/bin/env bash
set -euo pipefail

# Edit these as needed before running.
SOURCE="/home/ylj20/FacialRecognitionTest/animal-face-id/artifacts/"
DEST="ylj20@login.hpc.cam.ac.uk:/home/ylj20/rds/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/"
RETRIES=5
SLEEP_SECONDS=10

attempt=1
while true; do
  echo "Rsync attempt ${attempt}/${RETRIES}..."
  if rsync -avz --progress --partial "${SOURCE}" "${DEST}"; then
    echo "Rsync completed successfully."
    break
  fi

  if [[ "${attempt}" -ge "${RETRIES}" ]]; then
    echo "Rsync failed after ${RETRIES} attempts."
    exit 1
  fi

  echo "Rsync failed. Retrying in ${SLEEP_SECONDS}s..."
  sleep "${SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done
