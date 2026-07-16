# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug2.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_best.pt`
- Date: 2026-06-26 04:44:39
- Device: cpu
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8068
- Top-3 accuracy: 0.8718
- Top-5 accuracy: 0.8889
- Macro F1: 0.7891
- Macro precision: 0.8106
- Macro recall: 0.7867
- Weighted F1: 0.8052

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_best_corrected_best_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_best_corrected_best_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Racoon: acc = 0.000
  - Brookes: acc = 0.200
  - Leigh: acc = 0.250
  - Caro: acc = 0.333
  - Castaf: acc = 0.333

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_best_corrected_best_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
