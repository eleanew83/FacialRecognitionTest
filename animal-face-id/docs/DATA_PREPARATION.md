# Data Preparation Guide

This guide describes how the macaque face data is produced and how this repo consumes it. Unlike a single bundled dataset, the data flows through an **upstream detection + split stage** and this repo reads the result.

The pipeline is:
1. **Detect & crop** faces from source images/video using YOLO (in the wider project's `yolo_detection/`).
2. **Split** the per-individual crops into train/val/test.
3. **Consume** the crops + split manifest here for training and evaluation.

---

## Step 1: Crops (upstream)

Faces are detected and cropped by the YOLO stage, producing one folder per individual:

```
…/yolo_detection/yolo_detection_code/output/macaque_crops/
└── <split>/<IndividualID>/<image>.jpg          # split ∈ {train, val, test}
```

This repo does **not** run the detector; it points at the existing crops via `data.raw_root` in the config.

---

## Step 2: Split (upstream)

The crops are split per-individual into train/val/test (70/15/15, fixed seed) by:

```
gorillavision/reid-system/scripts/prepare_macaque_dataset.py
```

It is a **stratified** split — every individual appears in all three sets — which is required for closed-set evaluation. The split is done a level above this repo because it operates on the YOLO output (you must crop before you can split).

Current dataset: **156 individuals**, **6607** train / **1361** val / **1584** test images.

---

## Step 3: Split manifest consumed here

Training/evaluation in this repo read a single manifest:

```
data/macaque_faces/splits.json
```

Each entry is `{"id": "<IndividualID>", "path": "<split>/<ID>/<file>"}`, joined onto `data.raw_root` from the config. The config keys that wire this up:

```yaml
data:
  dataset_name: macaque_faces
  raw_root:    /…/yolo_detection/.../output/macaque_crops
  splits_path: /…/animal-face-id/data/macaque_faces/splits.json
  num_classes: 156
```

To sanity-check that every listed image exists on disk, a quick scan over `splits.json` vs `raw_root` is enough (all paths should resolve; the dataset loader also validates them at startup).

---

**With crops + `splits.json` in place, proceed to [Model Training](./TRAINING.md).**
