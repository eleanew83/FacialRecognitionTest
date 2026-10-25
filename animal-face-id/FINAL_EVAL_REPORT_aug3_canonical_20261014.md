# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug3_clean.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best.pt`
- Date: 2026-10-23 13:15:10
- Device: cuda
- Logit-adjust tau: 0.0
- Split: test
- TTA: none; angles: []
- Samples: 1498
- Num classes: 149

## 1. Overall Metrics
- Top-1 accuracy: 0.8251
- Top-3 accuracy: 0.8865
- Top-5 accuracy: 0.8985
- Macro F1: 0.8027
- Macro precision: 0.8246
- Macro recall: 0.8042
- Weighted F1: 0.8239

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Racoon: acc = 0.000
  - Brookes: acc = 0.200
  - Caro: acc = 0.333
  - Leigh: acc = 0.333
  - SAM to be named: acc = 0.333

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
