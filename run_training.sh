#!/bin/bash
set -e

# Directory paths
WORKSPACE_DIR="$(pwd)"
SPLIT_DATA_DIR="${WORKSPACE_DIR}/macaque_split_data"
MODELS_DIR="${WORKSPACE_DIR}/macaque_models"
LOG_FILE="${WORKSPACE_DIR}/training_log.txt"

# Create models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

echo "Running macaque facial recognition training..."
echo "Logs will be saved to ${LOG_FILE}"

# Run with tee to capture output
docker run \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    -e PYTHONUNBUFFERED=1 \
    -e PL_MAX_EPOCHS=10 \
    --stop-timeout 300 \
    -v "${SPLIT_DATA_DIR}:/data" \
    -v "${MODELS_DIR}:/models" \
    -v "${WORKSPACE_DIR}/gorillavision/reid-system:/gorilla-reidentification/reid-system" \
    -v "${WORKSPACE_DIR}/macaque_training_config.json:/gorilla-reidentification/reid-system/gorillavision/configs/custom/macaque_config.json" \
    -v "${HOME}/.netrc:/root/.netrc" \
    -v "${HOME}/.config/wandb:/root/.config/wandb" \
    --ipc="host" \
    -it gorilla_triplet \
    bash -c "python3 /gorilla-reidentification/reid-system/scripts/simple_train.py -c custom/macaque_config.json" 2>&1 | tee "${LOG_FILE}"

echo "Training complete. Stored models in ${MODELS_DIR}"
