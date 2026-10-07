# Macaque Facial Recognition

This repository contains code for training and evaluating a facial recognition system for macaques using PyTorch Lightning and a Triplet Loss approach with Vision Transformer (ViT) backbone following [GorillaVision](https://github.com/Lasklu/gorillavision).

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
- [YOLO Detection Pipeline](#yolo-detection-pipeline)  
- [Identification Training (GorillaVision)](#identification-training-gorillavision)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)  

## Overview

This project demonstrates how to train a face‐recognition/identification model for macaques using a patched version of the GorillaVision open-set re-identification system. It includes data preparation, Docker-based training, database creation, and evaluation pipelines.

## Prerequisites
 
- Python 3.8+ (for local scripts)  
- A properly structured dataset of macaque images (see [Data Organization](#data-organization))
- GPU not required, but recommended

## Repository Structure

```
FacialRecognitionTest/
├── data/                           # (Optional) raw datasets
├── gorillavision/                  # Forked GorillaVision module
│   ├── reid-system/               # Core re-identification code
│   ├── scripts/                   # Training & dataset utilities
│   ├── gorillavision.png          # Logo
│   ├── gorillavisionarchitecture.png
│   ├── LICENSE
│   └── README.md                  # GorillaVision docs
├── macaque_split_data/            # Auto-generated train/val/test splits
├── macaque_models/                # Output trained models & checkpoints
├── run_patched_training.sh        # Entrypoint for Dockerized training
├── patch_triplet.py               # Fixes Lightning v2.0 API in TripletLoss
├── macaque_training_config.json   # Training configuration
└── README.md                      # This file
```

## Data Organization

Your raw images should live under a directory structured by individual name:

```
Gibraltar_Macaques/
├── Alice/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Bob/
│   └── ...
└── ...
```

Place this folder anywhere (e.g. in `data/`), then use our split script to generate:

- `train/`
- `val/`
- `test/`

under `macaque_split_data/`.

## Data Preparation

Use the provided data splitter to stratify your dataset:

```bash
python3 gorillavision/reid-system/scripts/prepare_macaque_dataset.py \
  --source /path/to/Gibraltar_Macaques \
  --target ./macaque_split_data \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

- **train**: for model fitting  
- **val**: for building the identity database  
- **test**: for final performance metrics  

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
    "use_augmentation": true
  },
  "create_db": {
    "image_folder": "/data/val",
    "db_path": "/data/db/"
  },
  "eval": {
    "img_folder": "/data/test",
    "db_path": "/data/db/"
  }
}
```

Modify any of these before launching training.

## Evaluation & Inference

**Automatic evaluation**: After training, the best model path is injected into your config. You can then run the GorillaVision pipeline:

```bash
docker run \
  -v "$(pwd)/macaque_split_data:/data" \
  -v "$(pwd)/macaque_models:/models" \
  -v "$(pwd)/gorillavision/reid-system:/gorilla-reidentification/reid-system" \
  --gpus device=0 --ipc="host" -it gorilla_triplet \
  python3 identification_pipeline.py -c custom/macaque_config.json
```

**Prediction on images/videos**: See [gorillavision/README.md](gorillavision/README.md) for instructions on using `predict.py`.

## Scripts & Utilities

- **patch_triplet.py**: Patches Lightning v1→v2 epoch hooks  
- **run_patched_training.sh**: Docker wrapper for training  
- `scripts/`:
  - `flatten_macaque_dirs.py`  
  - `extract_metadata_to_excel.py`
  - `clean_double_spaces.py`
- `gorillavision/reid-system/scripts/`:
  - `prepare_macaque_dataset.py`  
  - `simple_train.py` (wandb-free trainer)  
  - `simple_train_offline.py`  

## YOLO Detection Pipeline

This section describes the steps for preparing data and training a YOLO-based macaque face detection model.

### 1. Flatten Folder Structure

In the project root, run:

```bash
python3 scripts/flatten_macaque_dirs.py
```

### 2. Create `macaque_split_data`

Use a fresh output directory for each generation. The splitter and crop exporter
publish only validated, complete outputs. The crop transfer helper also refuses
populated destinations; see [Data Preparation safeguards](animal-face-id/docs/DATA_PREPARATION.md).

Prepare the split data using:

```bash
python3 gorillavision/reid-system/scripts/prepare_macaque_dataset.py --source /home/ylj20/macaque_flattened --target /home/ylj20/FacialRecognitionTest/macaque_split_data --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### 3. YOLO Detection Preparation

The existing reviewed detector dataset can be used directly for training. To build
another dataset, provide source photos and reviewed YOLO labels with matching
relative paths, and choose a **new or empty** output directory. From the repository root:

```bash
python3 yolo_detection/yolo_detection_code/scripts/prepare_yolo_dataset.py \
  --source-dir /path/to/source_images \
  --labels-dir /path/to/reviewed_labels \
  --output-dir yolo_detection/yolo_detection_data_new --seed 42
```

Preparation groups exact file bytes or decoded RGB pixels before deterministic
splitting. It keeps distinct crops with matching names, rejects conflicting labels
on identical content, preserves original image bytes, and checks the complete
output before publishing. It refuses populated destinations, including the current
reviewed dataset.

Alternatively, `--model /path/to/macaque_face_checkpoint.pt` generates proposals
for visual review. Its class must be `macaque_face`; a COCO `person` checkpoint
cannot supply face labels. Every detected face is retained. Missing detections
stop generation for manual annotation rather than writing a fixed centre box.

### 4. Reviewing Face Boxes (Labeling)

Use LabelImg in a graphical environment to edit existing YOLO face boxes. Its arguments are `image_dir class_file save_dir`; `classes.txt` supplies class names, while the third argument selects the directory used to load and save the annotations. From the repository root:

```bash
cd yolo_detection/yolo_detection_data
labelImg images/train labels/train/classes.txt labels/train
labelImg images/val labels/val/classes.txt labels/val
```

Select **YOLO** format and keep the single class `macaque_face` (class ID 0). Click a box and drag its corners to adjust it. **Ctrl+S** saves, **D/A** move to the next/previous image, and **W** creates another box. Annotate every eligible visible face under a consistent blur/profile/occlusion policy. If you use **Open Dir** again, confirm **Change Save Dir** still points to the matching labels directory.

After editing, invalidate any YOLO label caches so training rereads the boxes. From `yolo_detection/yolo_detection_data/`:

```bash
rm -f labels/train.cache labels/val.cache
```

Reference: [LabelImg usage](https://github.com/HumanSignal/labelImg) and [command-line / label-directory implementation](https://github.com/HumanSignal/labelImg/blob/master/labelImg.py).

### 5. Training YOLO Detector

Training validates the configured dataset, annotation format, image/label pairing,
and exact content within and across splits before loading a model. It also
invalidates label caches so training rereads the current boxes. The default
path is relative to this repository; `--data-dir` selects another checked dataset.
To run the same checks without training, from the repository root:

```bash
python3 yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py --check-data-only
```

Change to the scripts directory and start training:

```bash
cd yolo_detection/yolo_detection_code/scripts
python3 train_yolo_detection.py --mode train --epochs 100 --batch 4 --img-size 416 --device cpu
```

## Identification Training (GorillaVision)

This section describes how to run the identification training pipeline with GorillaVision.

### Prerequisites

1. Install Docker with GPU support.
2. Build the `gorilla_triplet` image:

```bash
docker build -t gorilla_triplet .
```

3. Ensure the Gibraltar_Macaques dataset is available (typically in the parent directory of this repository).

### Run the training pipeline

The training pipeline is automated via `gorillavision/reid-system/scripts/train_macaque_identification.sh`.

From the repo root:

```bash
cd /home/ylj20/FacialRecognitionTest
./gorillavision/reid-system/scripts/train_macaque_identification.sh
```

This script will:
- split the dataset into `train/val/test`
- train the identification model
- create the identification database
- evaluate model performance

All trained models and the identification database are stored in `macaque_models`.

### Configuration

To change training parameters, edit:

`gorillavision/reid-system/gorillavision/configs/custom/macaque_config.json`

Common knobs include input size, batch size, learning rate, epochs, and augmentation settings.

## Troubleshooting

- If training fails, check console output for errors.
- Verify GPU access in Docker (`nvidia-smi`).
- For memory issues, reduce batch size in the config.
- Ensure each individual has at least 3 images.

## Contributing

Contributions welcome! Please fork, file an issue, or submit a pull request. For updates to core re-identification code, refer to the [GorillaVision submodule](gorillavision/README.md).

## License

This project incorporates code from the GorillaVision project. See `gorillavision/LICENSE` for full license terms.