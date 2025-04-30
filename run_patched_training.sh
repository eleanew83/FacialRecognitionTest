#!/bin/bash

# Directory paths
WORKSPACE_DIR="$(pwd)"
SPLIT_DATA_DIR="${WORKSPACE_DIR}/macaque_split_data"
MODELS_DIR="${WORKSPACE_DIR}/macaque_models"

# Create models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

echo "Running training in Docker container with patched TripletLoss class..."
docker run \
    -v "${SPLIT_DATA_DIR}:/data" \
    -v "${MODELS_DIR}:/models" \
    -v "${WORKSPACE_DIR}/gorillavision/reid-system:/gorilla-reidentification/reid-system" \
    -v "${WORKSPACE_DIR}/macaque_training_config.json:/gorilla-reidentification/reid-system/gorillavision/configs/custom/macaque_config.json" \
    --ipc="host" \
    -it gorilla_triplet \
    bash -c "python3 /gorilla-reidentification/reid-system/scripts/simple_train.py -c custom/macaque_config.json"

echo "Training complete. Stored models in ${MODELS_DIR}" 