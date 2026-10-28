# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug3_ltr.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug3_ltr_best.pt`
- Date: 2026-07-17 05:30:54
- Device: cuda
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8188
- Top-3 accuracy: 0.8801
- Top-5 accuracy: 0.8958
- Macro F1: 0.8044
- Macro precision: 0.8371
- Macro recall: 0.7983
- Weighted F1: 0.8177

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug3_ltr_macaque-resnet50-arcface_aug3_ltr_best_aug3_ltr_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug3_ltr_macaque-resnet50-arcface_aug3_ltr_best_aug3_ltr_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Brookes: acc = 0.200
  - Racoon: acc = 0.250
  - Caro: acc = 0.333
  - SAM to be named: acc = 0.333
  - Goblin: acc = 0.400

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug3_ltr_macaque-resnet50-arcface_aug3_ltr_best_aug3_ltr_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
