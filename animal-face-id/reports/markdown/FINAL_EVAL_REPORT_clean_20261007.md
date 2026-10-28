# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_clean.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_clean_photo_split_20261007_best.pt`
- Date: 2026-10-08 00:39:53
- Device: cuda
- Logit-adjust tau: 0.0
- Test samples: 1498
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8278
- Top-3 accuracy: 0.8798
- Top-5 accuracy: 0.8992
- Macro F1: 0.8102
- Macro precision: 0.8320
- Macro recall: 0.8101
- Weighted F1: 0.8268

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_clean_macaque-resnet50-arcface_clean_photo_split_20261007_best_clean_20261007_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_clean_macaque-resnet50-arcface_clean_photo_split_20261007_best_clean_20261007_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Brookes: acc = 0.200
  - Caro: acc = 0.333
  - Thief: acc = 0.333
  - Scarlet: acc = 0.400
  - Harry: acc = 0.429

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_clean_macaque-resnet50-arcface_clean_photo_split_20261007_best_clean_20261007_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
