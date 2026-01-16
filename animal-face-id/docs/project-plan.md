# Animal Face Identification POC Plan

## Technical Requirements Extracted from Context
- Target hardware is a local Windows workstation with an NVIDIA RTX 5080; leverage PyTorch + CUDA for rapid iteration.
- Pipeline must detect animal faces (species filter + face localization), classify known individuals (closed set), flag unknowns (open set), and enroll new identities safely.
- Baseline should start from cropped animal face datasets before adding detection.
- Downstream system must support incremental gallery updates without retraining the entire network; embeddings stored for k-NN retrieval.
- Open-set logic requires distance-thresholding on embeddings plus ROC-style calibration to balance FPR/FNR.
- Enrollment must accumulate repeated unknown embeddings, cluster them (e.g., DBSCAN) and only create a new ID when clusters exceed a minimum sample count (≈10) and optionally pass human verification.
- Provide simple CLI / script (`python predict.py --image ...`) that outputs species confidence, ID confidence, and open-set score.
- Keep raw datasets out of version control (use `/data` as untracked storage).

## Project Goals & Success Criteria
- Deliver a reproducible POC that, on local GPU, can ingest animal face images and return predicted individual IDs or "Unknown."
- Achieve ≥85% Top-1 accuracy on a held-out closed-set validation split built from public datasets.
- Demonstrate open-set rejection with ≤1% false accept rate at ≥70% true accept rate for known individuals (tunable).
- Implement automated unknown clustering that proposes new IDs only after sufficient evidence; log enrollment events.
- Produce clear documentation enabling Cursor / Claude Code to extend training, detection, or UI layers.

## Problem Breakdown
1. **Input Normalization** – Collect datasets, clean labels, create train/val/test splits, and standardize image crops/resolution.
2. **Detection & Species Filtering** – (Phase 0/extension) Detect animal faces using YOLOv7/YOLOv8 or specialized detectors.
3. **Embedding Backbone** – Fine-tune a lightweight CNN/ViT (e.g., ResNet-18, ViT-S/16) with metric-learning loss (Triplet, ArcFace) to produce 128–256D embeddings.
4. **Closed-Set ID** – Maintain gallery vectors per individual; perform cosine similarity / k-NN; report Top-k predictions.
5. **Open-Set Thresholding** – Model distance distributions for positive vs negative pairs; learn threshold τ; optionally calibrate with logistic regression score.
6. **Unknown Pool & Clustering** – Buffer embeddings labeled unknown, run DBSCAN/HDBSCAN to form clusters; ensure min samples & temporal diversity.
7. **Enrollment & Metadata Store** – Persist newly approved identities with representative embeddings + metadata (name, source frames).

## Phase 0–3 Roadmap
| Phase | Focus | Key Outputs |
| --- | --- | --- |
| **Phase 0 – Environment & Data Prep** | Repo bootstrap, environment configs, dataset download scripts, EDA notebooks. | Functional repo, Makefile, data manifests, exploratory stats. |
| **Phase 1 – Closed-Set Baseline** | Train embedding backbone on cropped faces, build k-NN inference, CLI demo. | `train.py`, `predict.py`, saved weights, accuracy report. |
| **Phase 2 – Open-Set Thresholding** | Model distance distributions, add unknown classification, logging. | Threshold calibration notebook, updated inference returning Unknown. |
| **Phase 3 – Unknown Clustering & Enrollment** | Implement unknown buffer, clustering, auto-ID proposal, manual review hooks. | `enroll.py`, cluster visualizations, metadata store updates. |

## Recommended Architecture
1. **Data Layer**
   - `data/raw/` for original datasets; `data/processed/` for resized crops and metadata parquet/JSON.
   - Dataset registry JSON describing splits, augmentation configs.
2. **Model Layer**
   - Backbone: Torchvision ResNet-18 or ViT-S/16 fine-tuned with Triplet + ArcFace hybrid loss.
   - Optimizer: AdamW with cosine LR schedule; mixed precision training (torch.cuda.amp) for speed on 5080.
3. **Embedding Store**
   - Use FAISS (CPU) or torch.kNNGraph for retrieval; gallery stored in `artifacts/gallery.pkl`.
4. **Inference Service (CLI)**
   - Pipeline: optional YOLO detector → crop align → embedding → similarity scoring → threshold decision.
   - Outputs JSON blob with species prob, predicted ID, similarity score, unknown flag, metadata path.
5. **Enrollment Engine**
   - Unknown buffer persisted as SQLite/Parquet table.
   - Clustering job (cron or manual) using DBSCAN (metric=cosine, eps tuned) + heuristics (min_samples, min captures per day).
6. **Observability**
   - Store tensorboard logs, confusion matrices, ROC plots (`reports/`).
7. **Probabilistic Embedding Upgrade (Future)**
   - Evaluate GroVE (Probabilistic VLM Embeddings) to predict per-sample mean/covariance, enabling KL-divergence scoring and uncertainty-aware enrollment.
   - Keep GroVE head modular so existing ResNet/ViT weights remain usable; enable toggling deterministic vs probabilistic pipelines at inference.

## File Structure (Repo Root)
```
.
├── README.md
├── docs/
│   ├── context-raw.md
│   └── project-plan.md   ← this document
├── src/
│   ├── data/
│   │   ├── download_datasets.py
│   │   └── dataset_registry.yaml
│   ├── models/
│   │   ├── backbones.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── enroll.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── data/                # gitignored storage for datasets & embeddings
├── configs/
│   ├── train_closed_set.yaml
│   ├── open_set.yaml
│   └── clustering.yaml
├── experiments/
│   └── notebooks/
│       ├── eda.ipynb
│       └── threshold_calibration.ipynb
└── reports/
    ├── metrics.json
    └── figures/
```

## Detailed Tasks – Phase 1 (Closed-Set Baseline)
1. **Data Preparation**
   - Implement `download_datasets.py` with Kaggle + ResearchGate links (manual authentication as needed).
   - Build preprocessing script to crop/resize faces, normalize color, generate train/val/test splits ensuring individual-disjoint partitions.
2. **Model Setup**
   - Implement backbone definitions (ResNet-18, ViT-S fallback) with configurable embedding dimension.
   - Implement Triplet loss sampler (batch-hard or semi-hard) plus optional ArcFace/CosFace classification head.
3. **Training Loop**
   - Mixed precision training, gradient accumulation flags, checkpointing best val accuracy.
   - Log metrics (accuracy, loss curves) to TensorBoard + JSON.
4. **Evaluation**
   - Compute Top-1/Top-5 accuracy, confusion matrix, per-individual precision/recall.
   - Export gallery embeddings (mean of support images per ID) and serialized backbone weights.
5. **Inference Demo**
   - Build `predict.py` CLI (accepts image path) that loads embeddings, runs forward pass, prints species & ID with similarity score.
   - Include simple FastAPI stub or notebook for manual visual verification (optional).

## Detailed Tasks – Phase 2 (Open-Set Thresholding)
1. **Distance Distribution Analysis**
   - Generate positive/negative pair distance histograms from validation embeddings.
   - Fit ROC/DET curves; select τ for target FAR (≤1%).
2. **Threshold Calibration Module**
   - Add `open_set.yaml` config for τ, fallback logistic reg calibrator, and temperature scaling.
   - Update inference pipeline to output `Unknown` when `d_min > τ`; log decision metadata.
3. **Unknown Buffering**
   - Append embeddings + metadata (timestamp, frame id) to `data/unknown_pool.parquet`.
   - Build monitoring notebook to visualize unknown influx and threshold stability.
4. **Regression Tests**
   - Unit tests for threshold logic, ensuring known samples below τ and synthetic unknowns above τ.
   - Evaluate Open-Set metrics: AUROC, Open-set Identification Rate (OSIR), False Alarm Rate.

## Detailed Tasks – Phase 3 (Unknown Clustering & Enrollment)
1. **Clustering Pipeline**
   - Implement `cluster_unknowns.py` using DBSCAN/HDBSCAN over normalized embeddings with cosine metric.
   - Tune `eps`, `min_samples`, and minimum distinct capture timestamps to avoid duplicate IDs.
2. **ID Proposal Workflow**
   - For each stable cluster (≥10 samples or configurable), compute centroid embedding, aggregate thumbnails, and auto-assign provisional ID (`gorilla_###`).
   - Generate HTML/Markdown report for manual review (link frames, allow override names).
3. **Enrollment Storage**
   - Persist approved IDs into gallery store, update catalog JSON, and regenerate FAISS index.
   - Archive raw cluster members and remove them from `unknown_pool`.
4. **Feedback Loop**
   - Track enrollment quality metrics (accept/reject counts, time-to-approval).
   - Add hooks to retrain/fine-tune backbone periodically with new labeled data plus hard negatives.

## Libraries & Model Choices
- **Core**: Python 3.11, PyTorch 2.x, Lightning (optional), torchvision, timm, CUDA 12.
- **Detection**: YOLOv7/YOLOv8 (Ultralytics) or adapt GorillaVision detector weights.
- **Metric Learning**: `pytorch-metric-learning` for Triplet/ArcFace, or custom heads.
- **Retrieval**: FAISS for k-NN search; alternatively `sklearn.NearestNeighbors`.
- **Clustering**: `hdbscan` / `scikit-learn` DBSCAN.
- **Augmentation**: `albumentations` for blur, lighting, partial occlusion simulation.
- **Logging**: TensorBoard, Weights & Biases (optional, if network ok), Rich console.
- **Utilities**: Hydra or OmegaConf for configs; Typer for CLI ergonomics.

## Training Pipeline
1. **Data Ingestion** – Sync datasets via script, verify checksums, generate metadata manifest.
2. **Preprocessing** – Apply augmentations (color jitter, blur, cutout), align faces (optional landmarks).
3. **Batching & Sampling** – Ensure each batch contains multiple images per gorilla to enable effective Triplet mining.
4. **Model Training** – Run mixed-precision training with patience-based early stopping; save checkpoints & best weights.
5. **Embedding Export** – After training, compute embeddings for gallery/reference splits; store as `.npz` or parquet.
6. **Evaluation** – Run closed-set metrics + open-set ROC to determine τ; log to `reports/`.
7. **Packaging** – Bundle weights, configs, and gallery store into `artifacts/` for inference scripts.

## Evaluation Metrics
- **Closed-Set**: Top-1/Top-5 accuracy, mean Average Precision (mAP), per-individual recall.
- **Verification**: ROC/AUC, Equal Error Rate (EER), Detection Error Tradeoff (DET) curves.
- **Open-Set**: False Accept Rate (FAR), False Reject Rate (FRR), Open-set Identification Rate at target thresholds.
- **Clustering/Enrollment**: Purity, Adjusted Rand Index on simulated unknowns, average samples per accepted cluster.
- **Operational**: Inference latency per frame, GPU memory footprint, dataset coverage (images per individual).

## Future Extensions
- **YOLO Detection Upgrade** – Train YOLOv8/YOLO-NAS on mixed gorilla datasets + manual annotations to handle uncropped inputs; integrate tracking for ROI extraction.
- **Multi-Frame Tracking** – Add DeepSORT/ByteTrack to stitch detections into tracklets so multiple frames vote on identity before decision.
- **Enrollment UI** – Build lightweight web dashboard (FastAPI + React or Gradio) for reviewing unknown clusters, approving IDs, and updating metadata.
- **Active Learning Loop** – Surface low-confidence predictions for human labeling, automatically queue re-training jobs.
- **Edge Deployment** – Optimize inference with TensorRT or ONNX Runtime for on-site cameras once POC stabilizes.
- **Probabilistic Embeddings (GroVE)** – Integrate cvjena/GroVE probabilistic VLM head to attach mean/covariance predictions onto existing embeddings, unlocking uncertainty-aware open-set thresholds and enrollment gating.
