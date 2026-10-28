# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug2.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug1_best.pt`
- Date: 2026-06-26 04:49:11
- Device: cpu
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8194
- Top-3 accuracy: 0.8826
- Top-5 accuracy: 0.8958
- Macro F1: 0.8091
- Macro precision: 0.8351
- Macro recall: 0.8038
- Weighted F1: 0.8194

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug1_best_corrected_aug1_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug1_best_corrected_aug1_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Caro: acc = 0.333
  - Thief: acc = 0.333
  - Harry: acc = 0.429
  - Nathalie: acc = 0.444
  - Goblin: acc = 0.467

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug1_best_corrected_aug1_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
