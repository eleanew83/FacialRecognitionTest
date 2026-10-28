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

## Aggressive augmentation comparison (October 2026)

Submit three independent training tasks and a dependent fourth evaluation task:

```bash
cd /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id
bash scripts/submit_aggressive_suite.sh
```

The submitter creates a Slurm array with three GPU tasks, then submits the
rotation-TTA job with `afterok` on the entire array. Each training task uses one
GPU, eight CPUs, 32 GB RAM, and a two-hour wall-time limit. Evaluation requests one
GPU for up to two hours. This command submits work; it does not wait for completion.

For the controlled rotation-only follow-up, submit just the ±45° and ±60° policies
and their dependent single-view/mean/max evaluation:

```bash
bash scripts/submit_aggressive_suite.sh rot45 rot60
```

These two configs copy the ±30° policy and change only the affine rotation range
and output names. The remaining augmentation settings, model, loss, seed 42,
learning-rate schedule and 30-epoch budget stay fixed. The submitter saves the
selected profile order in `profiles.json` and maps it to array tasks from zero.

Both GPU launchers send Slurm email notifications to `ylj20@cam.ac.uk` on start,
completion, failure and time-limit termination. Training enables `ARRAY_TASKS`,
so each array task generates its own notifications. The dependent evaluation job
also requests notification if its training dependency becomes invalid. Email
settings are part of the submitted Slurm job record; this does not send a separate
test email.

Both GPU launchers set `TMPDIR=/rds/user/ylj20/hpc-work/tmp` directly, matching the
successful baseline launcher. Python manages its temporary worker files within
this existing shared directory. The short path avoids the `AF_UNIX path too long`
error caused by putting `TMPDIR` inside a long suite directory.
The original array `38544155` timed out on 23 October before completing an epoch
because of this launcher error; its dependent evaluation `38544156` did not run.
The launcher fix is in place for subsequent submissions; no augmentation results
were produced by that original array.

| Config suffix | Rotation | Affine scale | Translation | Noise sigma / probability | Blur sigma / probability | Erasing probability / area |
| --- | --- | --- | --- | --- | --- | --- |
| `aggressive_rot30` | ±30° | 0.80–1.25 | ±10% | 0.03 / 35% | 0.2–2.0 / 40% | 20% / 2–10% |
| `aggressive_rot100` | ±100° | 0.70–1.35 | ±12% | 0.05 / 45% | 0.2–2.5 / 50% | 25% / 2–12% |
| `aggressive_rot150` | ±150° | 0.60–1.50 | ±15% | 0.08 / 50% | 0.3–3.0 / 60% | 30% / 2–15% |

All three use the current repaired split and 149 canonical identities from
`train_macaque_arcface_aug3_clean.yaml`. Images resize to 224×224 before a single
bilinear affine transform, with mean-colour fill. All add mild colour jitter,
7×7 Gaussian blur, Gaussian noise before normalization, and random erasing.
There is no mirroring or extra RandAugment rotation. The ±100° and ±150° settings
are deliberate stress tests; stronger augmentation may reduce accuracy.

The model/loss stays ResNet50 + ArcFace with cross-entropy. Each fit starts from
ImageNet weights with a fresh classifier and optimizer. All use seed 42 to help
compare the augmentation policies; these are separate fits, not three seed
repeats of one policy. Class-balanced loss already exists in the LTR configs and
can be compared separately after choosing augmentation.

Training runs for 30 epochs with no early stopping, matching the previous baseline
budget so the comparison focuses on augmentation. AdamW starts at 3e-4 with the
original cosine schedule (minimum LR 0); both validation-best and final checkpoints
are retained. Review the new validation curves before deciding whether to extend
any run; stronger augmentation alone does not establish a need for more epochs.
The scheduler wall-time limit can still terminate a job. There is no automatic
resume; monitor the initial epoch timings and request more wall time before
submitting if needed. Longer training does not guarantee validation improvement.

The installed torchvision 0.17.2 lacks `v2.GaussianNoise`. The pipeline uses a
compatible `randn_like` implementation with clipping on float [0,1] images; it
uses native `v2.GaussianNoise` when available. See the
[torchvision API](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.GaussianNoise.html).
No package upgrade is required.

The fourth job evaluates **each model separately** with single-image prediction,
seven-angle probability averaging, and probability max pooling. Angles are
`[-30, -20, -10, 0, 10, 20, 30]` degrees. Each view uses margin-free ArcFace logits,
then softmax. Mean pools class probabilities; max takes the maximum probability
per class and renormalizes. Summing and averaging rank classes identically for
seven views. Views run sequentially within each batch to limit GPU memory use.

Validation picks the model and pooling method by top-1 (macro F1 breaks ties)
before any test evaluation. The best checkpoint for each model was already chosen
using ordinary, unaugmented validation during training. All test comparisons are
reported, while `selection.json` records the choice made on validation.

Each submission prints a unique directory under `artifacts/augmentation_sweeps/`:

- `configs/`: snapshots with separate checkpoint paths, plus `data_inputs.json`
  at the suite root recording split/identity-registry hashes.
- `checkpoints/`: validation-best and final model weights, plus epoch metrics.
- `logs/` and `jobs.txt`: Slurm logs and IDs.
- `eval/comparison.csv`: top-1/3/5 and macro F1 for all models, splits and methods.
- `eval/selection.json`: validation-selected model/method.
- `eval/*_view_probabilities.npz`: all seven probabilities per image, pooled
  probabilities, labels, paths, angle order and class names. Per-class metrics,
  confusion matrices and prediction CSVs are also saved. Plotting is skipped in
  the comparison job.

To evaluate another compatible checkpoint with rotation averaging directly:

```bash
python tools/run_final_eval.py --config <config.yaml> --ckpt <checkpoint.pt> \
  --device cuda --tta mean --tag rotation_mean
# Use --tta max for the max comparison; --split val for validation.
```
