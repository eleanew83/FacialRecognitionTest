#!/bin/bash
set -e

# Directory paths
WORKSPACE_DIR="$(pwd)"
SPLIT_DATA_DIR="${WORKSPACE_DIR}/macaque_split_data"
MODELS_DIR="${WORKSPACE_DIR}/macaque_models"
LOG_FILE="${WORKSPACE_DIR}/training_log.txt"

# Create models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

echo "Running training in Docker container with patched TripletLoss class and original wandb..."
echo "Logs will be saved to ${LOG_FILE}"

# Run with tee to capture output
docker run \
    -v "${SPLIT_DATA_DIR}:/data" \
    -v "${MODELS_DIR}:/models" \
    -v "${WORKSPACE_DIR}/gorillavision/reid-system:/gorilla-reidentification/reid-system" \
    -v "${WORKSPACE_DIR}/macaque_training_config.json:/gorilla-reidentification/reid-system/gorillavision/configs/custom/macaque_config.json" \
    -v "${WORKSPACE_DIR}/fix_wandb_in_container.sh:/fix_wandb_in_container.sh" \
    -v "${HOME}/.netrc:/root/.netrc" \
    -v "${HOME}/.config/wandb:/root/.config/wandb" \
    --ipc="host" \
    -it gorilla_triplet \
    bash -c "/fix_wandb_in_container.sh && python3 /gorilla-reidentification/reid-system/scripts/simple_train.py -c custom/macaque_config.json" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code: $EXIT_CODE"
    echo "See log file for details: ${LOG_FILE}"
    exit $EXIT_CODE
else
    echo "Training complete. Stored models in ${MODELS_DIR}"
fi 
