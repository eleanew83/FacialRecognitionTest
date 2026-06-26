# Model Training Guide

This guide explains how to train the macaque face identification model using the prepared dataset. The entire training process is controlled by a configuration file.

## 1. Understanding the Configuration Files

All training parameters are defined in YAML files located in the `configs/` directory. You can select a configuration or create a new one to run an experiment.

Here are the key configurations provided:

-   **`configs/train_macaque_arcface_aug2.yaml`**:
    -   **Purpose**: The main configuration for the current best model.
    -   **Backbone**: `resnet50` with an `arcface` head.
    -   **Augmentation**: basic; loss: cross-entropy. Produces `macaque-resnet50-arcface_aug2_best.pt`.

-   **`configs/train_macaque_arcface_aug1.yaml`**:
    -   **Purpose**: An earlier augmentation variant (kept for reference).

-   **`configs/train_macaque_arcface_ltr.yaml`**:
    -   **Purpose**: A long-tail-recognition experiment — strong augmentation (restricted RandAugment, no horizontal flip) + class-balanced loss. Use if tail performance needs improving on harder data.

### Key Parameters to Know

-   `data.num_classes`: Must match the number of individuals in your dataset (`156` for the current macaque set).
-   `model.backbone`: The neural network architecture (e.g., `resnet18`, `resnet50`).
-   `model.head`: The final layer type. `arcface` is recommended for face ID tasks as it encourages better feature separation than a standard `linear` classifier.
-   `trainer.max_epochs`: The total number of times the training loop will iterate over the entire dataset.
-   `trainer.lr`: The learning rate.
-   `trainer.device`: `cuda` for GPU training, `cpu` for CPU.
-   `trainer.amp`: (Automatic Mixed Precision) Set to `true` to speed up training on modern NVIDIA GPUs.

## 2. How to Start Training

To start a training run, you execute the main training script `src/training/train.py` and point it to your desired configuration file.

**Command:**

```bash
python -m src.training.train --config <path_to_your_config.yaml>
```

**Example: Training the current best model:**

```bash
python -m src.training.train --config configs/train_macaque_arcface_aug2.yaml
```

**Example: Running the long-tail experiment (strong aug + class-balanced loss):**

```bash
python -m src.training.train --config configs/train_macaque_arcface_ltr.yaml
```

While the script is running, you will see progress updates for each epoch printed to your console, including loss and accuracy metrics.

```
[Epoch 1/200] lr=0.00040 | train_loss=4.1234 top1=0.0567 | val_loss=3.9876 top1=0.0890
[Epoch 2/200] lr=0.00040 | train_loss=3.5678 top1=0.1234 | val_loss=3.4567 top1=0.1567
...
```

## 3. Training Outputs

All artifacts generated during training are saved in the `artifacts/` directory.

-   **Checkpoints (Model Weights)**:
    -   `artifacts/<config_name>_best.pt`: The model weights from the epoch with the **highest validation accuracy**. This is usually the file you'll want to use for inference.
    -   `artifacts/<config_name>_last.pt`: The model weights from the very last epoch of training.

-   **Metrics Log**:
    -   `artifacts/<config_name>_metrics.csv`: A CSV file containing the detailed metrics (loss, top-1 accuracy, top-5 accuracy) for every epoch. You can use this file to plot the training curves and analyze the model's learning progress.

    *Example content of `_metrics.csv`:*
    ```csv
    epoch,lr,train_loss,train_top1,train_top5,val_loss,val_top1,val_top5
    1,0.0004,4.1234,0.0567,0.1234,3.9876,0.0890,0.2012
    2,0.0004,3.5678,0.1234,0.2345,3.4567,0.1567,0.3123
    ...
    ```

## 4. Resuming from a Checkpoint

The current `train.py` script does not automatically resume from a checkpoint. It will always start a new training run from scratch. If you need to resume, you would need to modify the script to load the `optimizer_state` and `epoch` from a saved `.pt` file.

---

**After training is complete, you will have a trained model file (`.pt`) ready for evaluation and inference.**
