# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_ltr.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_ltr_best.pt`
- Date: 2026-06-27 01:43:27
- Device: cuda
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8182
- Top-3 accuracy: 0.8737
- Top-5 accuracy: 0.8914
- Macro F1: 0.8028
- Macro precision: 0.8309
- Macro recall: 0.7970
- Weighted F1: 0.8161

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_ltr_macaque-resnet50-arcface_ltr_best_ltr_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_ltr_macaque-resnet50-arcface_ltr_best_ltr_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Brookes: acc = 0.200
  - Racoon: acc = 0.250
  - Caro: acc = 0.333
  - SAM to be named: acc = 0.333
  - Fleur: acc = 0.400

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_ltr_macaque-resnet50-arcface_ltr_best_ltr_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
