# Conceptual Overview & FAQ

This document answers key conceptual questions about the Animal Face ID project, its architecture, and how to use it effectively.

## 1. High-Level Architecture

The project is more than just a single model; it's a complete system with three core pipelines:

1.  **Training Pipeline (`src/training/`):** This system takes a configured dataset and trains a deep learning model to learn the facial features of individuals. It supports YAML configurations, mixed-precision training, and robust metric logging. Its final output is a model file (e.g., `.pt`).
2.  **Evaluation Pipeline (`tools/run_final_eval.py`):** This is the canonical final-eval tool for test split metrics and reports (confusion matrix, per-class metrics, embeddings plots). The older `src/training/evaluate.py` is a stub/placeholder.
3.  **Inference Pipeline (`src/inference/`):** This system uses the trained model to perform the actual identification task. It takes a new image, converts it into a feature vector (embedding), and compares it against a "gallery" of known individuals to find the best match.

---

## 2. How Does the Model "Learn" and "Remember" New Faces?

This is a critical concept. The process is split into two distinct phases: **Training** and **Enrollment**.

### Training: Building the "Brain"

-   The model (`.pt` file) is the "brain." It's trained on thousands of images to understand the subtle features that make each chimpanzee unique.
-   This process is computationally expensive and requires a powerful GPU (like the RTX 5080 used for this project).
-   The goal of training is **not** to memorize specific faces, but to learn a "feature space" where faces of the same individual are grouped closely together, and faces of different individuals are pushed far apart.
-   You only need to retrain the model when you want to make the "brain" itself smarter, for example, by adding a massive amount of new data, supporting a new species, or changing the model architecture.

### Enrollment: Updating the "Memory"

-   The "gallery" (e.g., `gallery_embeddings.npz`) is the model's "memory." It stores the feature embeddings of all known individuals.
-   When you want to add a **new chimpanzee**, you do **not** need to retrain the entire model.
-   Instead, you simply "enroll" the new individual:
    1.  Use the already-trained model to generate embeddings for one or more photos of the new chimpanzee.
    2.  Add these new embeddings (along with a new ID) to the gallery file.
-   This process is extremely fast and can be done on a standard computer with a CPU.

**In short: The GPU-intensive training happens once to create a powerful "brain." Adding new faces is a fast, CPU-friendly process of updating the "memory" (gallery).**

---

## 3. Hardware Requirements: GPU vs. CPU

-   **Training:** Requires a powerful GPU. This is the "heavy lifting" phase where the model learns from data.
-   **Inference & Enrollment:** Can be run efficiently on a standard CPU. A GPU will make it faster, but is not required. This means the trained model can be deployed on laptops or other devices without specialized hardware for day-to-day identification tasks.

---

## 4. Why is This Model a "Chimpanzee Expert"?

-   This model was trained exclusively on a dataset of chimpanzee faces. All of its learned weights and biases are optimized to detect subtle variations between individual chimpanzees.
-   **Can it identify other primates?** Maybe, but performance is not guaranteed and will likely be much lower. The model may have learned features that are transferable to other apes (like gorillas or orangutans), but it has no specific knowledge of them.
-   **What happens if I add other animals (dogs, lizards) to the gallery?** This will contaminate the gallery and severely degrade performance. The model will try to find the "closest" match, and a dog's embedding might accidentally be closer to one chimp than another, leading to incorrect identifications.

**Recommendation:** Use this model exclusively for chimpanzees. For other species, the best practice is to train a new, dedicated model using a high-quality dataset for that species.

---

## 5. What is the Difference Between "Final Evaluation" and "Inference"?

-   **Final Evaluation (Step A):** This is a **scientific validation** of the model's quality. It answers the question: "How good is this model?" It runs the model on a held-out test set and produces a formal report with metrics like accuracy, precision, recall, and confusion matrices. This step is essential for benchmarking, finding model weaknesses, and having confidence in your system.
-   **Inference (Step B):** This is the **practical application** of the model. It answers the question: "Who is this chimpanzee?" It involves building a gallery of known individuals and using the model to identify new, incoming images. This is the core feature you would demonstrate to an end-user.

A robust project does both: **Step A** to prove the model is reliable, and **Step B** to use it for its intended purpose.
