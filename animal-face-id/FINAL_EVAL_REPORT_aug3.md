# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug3.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug3_best.pt`
- Date: 2026-07-17 05:12:50
- Device: cuda
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8201
- Top-3 accuracy: 0.8744
- Top-5 accuracy: 0.8908
- Macro F1: 0.8010
- Macro precision: 0.8183
- Macro recall: 0.8001
- Weighted F1: 0.8174

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug3_macaque-resnet50-arcface_aug3_best_aug3_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug3_macaque-resnet50-arcface_aug3_best_aug3_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Caro: acc = 0.000
  - Racoon: acc = 0.000
  - Brookes: acc = 0.200
  - Adele: acc = 0.375
  - Tack before emigrating: acc = 0.400

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug3_macaque-resnet50-arcface_aug3_best_aug3_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
