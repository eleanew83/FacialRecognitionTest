# Macaque Facial Recognition

This repository contains code for training and evaluating a facial recognition system for macaques using PyTorch Lightning and a Triplet Loss approach with Vision Transformer (ViT) backbone.

## Table of Contents

- [Overview](#overview)  
- [Prerequisites](#prerequisites)  
- [Repository Structure](#repository-structure)  
- [Data Organization](#data-organization)  
- [Data Preparation](#data-preparation)  
- [Training](#training)  
- [Configuration](#configuration)  
- [Evaluation & Inference](#evaluation--inference)  
- [Scripts & Utilities](#scripts--utilities)  
- [Contributing](#contributing)  
- [License](#license)  

## Overview

This project demonstrates how to train a face‐recognition/identification model for macaques using a patched version of the GorillaVision open-set re-identification system. It includes data preparation, Docker-based training, database creation, and evaluation pipelines.

## Prerequisites

- Docker with GPU support (NVIDIA Container Toolkit / `--gpus` flag)  
- Python 3.8+ (for local scripts)  
- A properly structured dataset of macaque images (see [Data Organization](#data-organization))  

## Repository Structure
Can you generate the content in the readme for me?

Below is the full contents of the README.md for the GorillaVision project:
Apply to README.md
!
Can you generate the readme for FacialRecognitionTest instead?

Here’s where we stand:
You’re working on a project called FacialRecognitionTest, whose goal is to train and evaluate a macaque (and potentially other primate) face‐recognition/identification system using components from the GorillaVision codebase.
You asked me to generate a more complete README.md for the project, replacing the existing three‐line placeholder.
To gather context, I inspected the repository root and found:
Shell scripts (run_patched_training.sh, patch_triplet.py)
A JSON config (macaque_training_config.json)
Data/model directories (data/, models/, macaque_models/, macaque_split_data/)
A nested GorillaVision subfolder with its own scripts, configs, and docs
I also read the INSTRUCTIONS.md, which details:
How to prepare the Gibraltar_Macaques dataset
How to build and run the Docker‐based training pipeline
Configuration options and troubleshooting tips
Next up: draft a comprehensive README that covers project overview, setup, data organization, training & evaluation steps, directory structure, and pointers to configuration and troubleshooting.
yes do that

Tool call timed out after 25s (grep)
FacialRecognitionTest/
├── data/ # (Optional) raw datasets
├── gorillavision/ # Forked GorillaVision module
│ ├── reid-system/ # Core re-identification code
│ ├── scripts/ # Training & dataset utilities
│ ├── gorillavision.png # Logo
│ ├── gorillavisionarchitecture.png
│ ├── LICENSE
│ └── README.md # GorillaVision docs
├── macaque_split_data/ # Auto-generated train/db/eval splits
├── macaque_models/ # Output trained models & checkpoints
├── run_patched_training.sh # Entrypoint for Dockerized training
├── patch_triplet.py # Fixes Lightning v2.0 API in TripletLoss
├── macaque_training_config.json
├── patch_triplet.py
├── patch_triplet.py
└── README.md # ← You are here!

## Data Organization

Your raw images should live under a directory structured by individual name:
Gibraltar_Macaques/
├── Alice/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
├── Bob/
│ └── ...
└── ...


Place this folder anywhere (e.g. in `data/`), then use our split script to generate:

- `train/`
- `database_set/`
- `eval/`

under `macaque_split_data/`.

## Data Preparation

Use the provided data splitter to stratify your dataset:

```bash
python3 gorillavision/reid-system/scripts/prepare_macaque_dataset.py \
  --source /path/to/Gibraltar_Macaques \
  --target ./macaque_split_data \
  --train-ratio 0.7 \
  --db-ratio 0.15 \
  --eval-ratio 0.15
```

- **train**: for model fitting  
- **database_set**: for building the identity database  
- **eval**: for final performance metrics  

## Training

We ship a convenience script that patches the TripletLoss API for compatibility and kicks off training inside Docker:

```bash
chmod +x run_patched_training.sh
./run_patched_training.sh
```

What it does:

1. Mounts `macaque_split_data/` and `macaque_models/` into the container  
2. Mounts your custom `macaque_training_config.json`  
3. Installs specific dependencies (`traitlets==5.9.0`, `wandb==0.15.0`)  
4. Runs `simple_train.py` (no wandb) for **nb_epochs** epochs  
5. Saves the best checkpoint under `macaque_models/`  

## Configuration

All hyperparameters and paths are controlled via `macaque_training_config.json`. Key sections:

```jsonc
{
  "main": {
    "experiment": "macaque_identification",
    "datasets": ["/data"]             // Mounted dataset root
  },
  "model": {
    "backbone": "vit",                // VisionTransformer
    "input_width": 224,               // Crop size
    "embedding_size": 256,
    "cutoff_classes": true
  },
  "train": {
    "batch_size": 64,
    "learning_rate": 1e-5,
    "nb_epochs": 5,
    "sampler": "ensure_positive",     // Triplet sampling strategy
    "use_augmentation": true,
    // ...
  },
  "create_db": {
    "image_folder": "/data/database_set",
    "db_path": "/data/db/"
  },
  "eval": {
    "img_folder": "/data/eval",
    "db_path": "/data/db/"
  }
}
```

Modify any of these before launching training.

## Usage

### Training

To train the model, run the following command:

```bash
./run_training.sh
```

This script will:
- Load data from the `macaque_split_data` directory
- Train a ViT-based facial recognition model using triplet loss
- Save models to the `macaque_models` directory
- Log training progress to WandB

### Configuration

The model training parameters are defined in `macaque_training_config.json`. Key parameters include:

- Learning rate: 1e-6 (reduced for stability)
- Batch size: 32
- Embedding size: 256
- Backbone: Vision Transformer (ViT)

### Known Issues and Solutions

1. **Training Stability**: The training uses debug mode with a limited dataset size to ensure stability. This prevents hanging issues with the batch sampler.

2. **NaN Validation Loss**: You may occasionally see NaN values in validation loss. This is handled gracefully and won't crash training.

3. **GPU Support**: For faster training, enable GPU support by adding `--gpus all` to the Docker run command if you have NVIDIA drivers installed.

## Evaluation & Inference

- **Automatic evaluation**: After training, the best model path is injected into your config. You can then run the GorillaVision pipeline:

  ```bash
  docker run \
    -v "$(pwd)/macaque_split_data:/data" \
    -v "$(pwd)/macaque_models:/models" \
    -v "$(pwd)/gorillavision/reid-system:/gorilla-reidentification/reid-system" \
    --gpus device=0 --ipc="host" -it gorilla_triplet \
    python3 identification_pipeline.py -c custom/macaque_config.json
  ```

- **Prediction on images/videos**: See [gorillavision/README.md](gorillavision/README.md) for instructions on using `predict.py`.

## Scripts & Utilities

- **patch_triplet.py**: Patches Lightning v1→v2 epoch hooks  
- **run_patched_training.sh**: Docker wrapper for training  
- `gorillavision/reid-system/scripts/`:
  - `prepare_macaque_dataset.py`  
  - `simple_train.py` (wandb-free trainer)  
  - `simple_train_offline.py`  

## Contributing

Contributions welcome! Please fork, file an issue, or submit a pull request. For updates to core re-identification code, refer to the [GorillaVision submodule](gorillavision/README.md).

## License

This project incorporates code from the GorillaVision project. See `gorillavision/LICENSE` for full license terms.