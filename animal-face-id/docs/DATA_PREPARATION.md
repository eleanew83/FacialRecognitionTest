# Data Preparation Guide

This guide details the steps required to download, structure, and prepare the Chimpanzee Faces dataset for model training. Following these steps ensures data integrity and reproducibility.

The entire data preparation workflow consists of four main steps:
1.  **Download**: Obtain the raw dataset from its original source.
2.  **Structure**: Organize the files into the directory structure expected by the scripts.
3.  **Validate**: Run a script to verify file integrity and correct structure.
4.  **Split**: Generate a JSON file that defines the training, validation, and test sets.

---

## Step 1: Download the Dataset

This project uses the **Chimpanzee Faces Dataset**. Due to its size and licensing, it is not bundled with this repository.

-   **Source**: [https://github.com/cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)

Please visit the link above and download the dataset. The key data folders you will need are `data_CTai` and `data_CZoo`, which contain the cropped face images.

---

## Step 2: Structure the Data

After downloading, you must place the image folders into a specific directory structure within this project. The scripts rely on this exact layout to find the data.

Create the following folder hierarchy and place the corresponding data inside:

```
animal-face-id/
└── data/
    └── chimpanzee_faces/
        ├── raw/
        │   └── datasets_cropped_chimpanzee_faces/
        │       ├── data_CTai/
        │       │   └── face_images/          # <-- Place the 5,078 CTai images here
        │       └── data_CZoo/
        │           └── face_images/          # <-- Place the 2,109 CZoo images here
        │
        ├── annotations/
        │   ├── annotations_merged_all.txt    # (Already in the repo)
        │   ├── annotations_merged_min10.txt  # (Already in the repo)
        │   └── kept_ids_min10.txt            # (Already in the repo)
        │
        └── processed/
            # This folder should exist but remain empty for now.
```

**Important**:
- The `raw/` directory is where you place the downloaded images.
- The `annotations/` directory (already provided in this repository) contains the text files that link image paths to individual chimpanzee IDs.
- The `processed/` directory is currently unused but reserved for future data processing steps.
- The `.gitignore` file is configured to ignore the `raw/` directory, so you don't accidentally commit thousands of images.

---

## Step 3: Validate the Dataset

Before proceeding, it is crucial to verify that all files are correctly placed and that the annotation files match the image files.

Run the provided validation script from the project's root directory:

```bash
python validate_dataset.py
```

This script performs several critical checks:
-   Verifies that the folder structure from Step 2 is correct.
-   Reads `annotations_merged_min10.txt` and confirms that every image path listed exists in the `raw/` directory.
-   Checks for ID consistency to ensure the data is clean.

**Expected Output**:
If everything is set up correctly, you will see a confirmation message at the end:

```
================================================================================
✓✓✓ Dataset structure verified — ready for model training.
================================================================================
```

If the script reports missing files or other errors, please double-check the folder structure and file locations from Step 2. A detailed report is also saved to `validation_results.json`.

---

## Step 4: Generate Train/Val/Test Splits

Our training process does not use the raw annotation files directly. Instead, it relies on a pre-defined split to ensure reproducible results.

Run the split generation script:

```bash
python scripts/prepare_chimpanzee_splits.py
```

This script does the following:
-   It reads the `annotations_merged_min10.txt` file, which contains 87 individuals with at least 10 images each.
-   It performs a **stratified split**, dividing the images into training (70%), validation (15%), and test (15%) sets.
-   The "stratified" approach guarantees that **every individual chimpanzee appears in all three sets**, which is vital for robust evaluation.
-   The script uses a fixed random seed (`42`) so that everyone who runs it gets the exact same splits.

**Expected Output**:
The script will create a new file and print a summary:

```
Wrote splits to: data/chimpanzee_faces/annotations/splits_min10.json
Train: 5005 images, 87 ids
Val:   1073 images, 87 ids
Test:  1072 images, 87 ids
```

The generated file, `data/chimpanzee_faces/annotations/splits_min10.json`, is the "source of truth" that the model trainer will use to load data.

---

**You have now successfully prepared the dataset. You are ready to proceed to model training.**
