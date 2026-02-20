# Final Evaluation Report — Chimpanzee Faces (ResNet50 + ArcFace)

- Config: `/home/ylj20/FacialRecognitionTest/animal-face-id/configs/train_macaque_arcface.yaml`
- Checkpoint: `/home/ylj20/FacialRecognitionTest/animal-face-id/artifacts/macaque-resnet50-arcface_aug2_best.pt`
- Date: 2026-02-06 10:14:00
- Device: cpu
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.6793
- Top-3 accuracy: 0.7298
- Top-5 accuracy: 0.7456
- Macro F1: 0.6324
- Weighted F1: 0.6780

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_macaque-resnet50-arcface_aug2_best_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_macaque-resnet50-arcface_aug2_best_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Carole: acc = 0.000
  - Castaf: acc = 0.000
  - Leigh: acc = 0.000
  - Nadia: acc = 0.000
  - Racoon: acc = 0.000

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_macaque-resnet50-arcface_aug2_best_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals may have low support; interpret per-class metrics accordingly.
- Consider augmentations/backbone/loss tuning for further gains.
