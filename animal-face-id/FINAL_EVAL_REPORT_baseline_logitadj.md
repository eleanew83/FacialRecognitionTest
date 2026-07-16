# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug2.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug2_best.pt`
- Date: 2026-06-27 01:43:10
- Device: cuda
- Logit-adjust tau: 1.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8308
- Top-3 accuracy: 0.8801
- Top-5 accuracy: 0.8990
- Macro F1: 0.8158
- Macro precision: 0.8309
- Macro recall: 0.8187
- Weighted F1: 0.8308

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug2_best_baseline_logitadj_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug2_best_baseline_logitadj_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Racoon: acc = 0.250
  - Goblin: acc = 0.400
  - Tonka: acc = 0.400
  - Harry: acc = 0.429
  - Leigh: acc = 0.500

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug2_macaque-resnet50-arcface_aug2_best_baseline_logitadj_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
