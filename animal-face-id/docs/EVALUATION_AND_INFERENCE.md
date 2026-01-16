# Evaluation and Inference Guide

This guide covers how to use a trained model checkpoint (`.pt` file) for both scientific validation and practical identification tasks.

---

## Conceptual Preamble: Evaluation vs. Inference

Before running the commands, it's important to understand the two primary use cases for a trained model:

1.  **Evaluation (Scientific Validation):** This is the process of measuring your model's performance on a held-out test dataset. It answers the question: **"How good is this model?"** by generating objective metrics like accuracy, precision, and confusion matrices. This is crucial for benchmarking and understanding the model's reliability.

2.  **Inference (Practical Application):** This is the process of using the model to identify an unknown individual. It answers the question: **"Who is this chimpanzee?"** This involves comparing a new image against a "gallery" of known individuals.

For a deeper dive into these concepts, including how the model "remembers" faces and the hardware requirements, please read the **[Conceptual Overview & FAQ](./CONCEPTS.md)**.

---

## The Workflow: Gallery, then Predict

The core of the identification process is a **gallery**. Think of this as the model's "memory"—an indexed collection of "faceprints" (embeddings) from all the known individuals the model was trained on.

The workflow is always:
1.  **Build the Gallery**: Use a trained model to create a gallery file. This only needs to be done once per model.
2.  **Run Inference/Evaluation**: Use the model and the gallery to perform predictions.

---

## Step 1: Build the Gallery Index

The gallery index is a file that acts as a searchable database of facial features.

**Command:**
Run `build_gallery.py`, pointing it to the config file of your trained model.

```bash
python -m src.inference.build_gallery --config <path_to_your_config.yaml> --device cuda
```
*Note: While gallery building can run on a CPU (`--device cpu`), using a GPU is much faster if you have one.*

**Example:**
```bash
python -m src.inference.build_gallery --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cuda
```

This script will:
1.  Load the best checkpoint specified in the config (`inference.checkpoint`).
2.  Process all images in the **training set** to create the "memory" of known individuals.
3.  Save the computed embeddings and a k-NN index to the `artifacts/` directory.

**Outputs:**
-   `artifacts/gallery_index.pkl`: The crucial k-NN index for fast similarity search.
-   `artifacts/train_embeddings.npz`: The raw embeddings and labels.

---

## Step 2 (Option A): Predict a Single Image (Inference)

With the gallery built, you can identify a single image. This task is very lightweight and runs quickly on a CPU.

**Command:**
Use `predict.py` with the path to an image and the corresponding config file.

```bash
python -m src.inference.predict --image <path_to_image.png> --config <path_to_your_config.yaml> --device cpu
```

**Example:**
```bash
python -m src.inference.predict --image /path/to/some_chimp_face.png --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cpu
```

**Output:**
The script prints the predicted ID and the distance (lower is better). A very high distance may indicate an unknown individual.
```json
{
  "image": "/path/to/some_chimp_face.png",
  "predicted_id": "Frodo",
  "distance": 0.2345,
  ...
}
```

---

## Step 2 (Option B): Run Full Evaluation on the Test Set

To get an objective measure of your model's performance, run the final evaluation script on the **test set**.

**Command:**
Use `run_final_eval.py`, providing the config file and the path to the specific checkpoint you want to evaluate.

```bash
python tools/run_final_eval.py --config <path_to_config.yaml> --ckpt <path_to_checkpoint.pt> --device cuda
```

**Example:**
```bash
python tools/run_final_eval.py \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  --device cuda
```

**Outputs:**
This script generates a comprehensive set of reports in `artifacts/final_eval/` and updates `FINAL_EVAL_REPORT.md`, including confusion matrices, per-class metrics, and embedding visualizations. Analyzing these outputs provides a deep understanding of your model's strengths and weaknesses.

---

## Enrolling New Individuals (The Future)

What if you have a new chimpanzee that wasn't in the original training set? You don't need to retrain the model. You just need to **enroll** them into your gallery.

While the scripts do not yet automate this, the process would be:
1.  **Get Pictures:** Collect one or more clear photos of the new individual's face.
2.  **Generate Embeddings:** Use the `predict.py` script (or a similar utility) to turn these photos into feature embeddings using your trained model.
3.  **Update Gallery:** Add the new embeddings and a new ID to your gallery files (`gallery_index.pkl` and `train_embeddings.npz`).

This makes the system extensible. The core model provides the intelligence, and the gallery provides a flexible memory that can be updated over time.
