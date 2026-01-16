# Final Evaluation Report — Chimpanzee Faces (ResNet50 + ArcFace)

- Config: `configs/train_chimp_min10_resnet50_arc_full.yaml`
- Checkpoint: `artifacts/chimp-min10-resnet50-arcface-full_best.pt`
- Date: 2025-11-16 15:15:49
- Device: cuda
- Test samples: 1072
- Num classes: 87

## 1. Overall Metrics
- Top-1 accuracy: 0.7920
- Top-3 accuracy: 0.8293
- Top-5 accuracy: 0.8424
- Macro F1: 0.7072
- Weighted F1: 0.7856

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_chimp_min10_resnet50_arc_full_chimp-min10-resnet50-arcface-full_best_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_chimp_min10_resnet50_arc_full_chimp-min10-resnet50-arcface-full_best_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Alina: acc = 0.000
  - Celine: acc = 0.000
  - Max: acc = 0.000
  - Rubra: acc = 0.000
  - Totem: acc = 0.000

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_chimp_min10_resnet50_arc_full_chimp-min10-resnet50-arcface-full_best_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals may have low support; interpret per-class metrics accordingly.
- Consider augmentations/backbone/loss tuning for further gains.
