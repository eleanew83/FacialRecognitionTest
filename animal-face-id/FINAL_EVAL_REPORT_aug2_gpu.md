# Final Evaluation Report — Macaque Faces (ResNet50 + ArcFace)

- Config: `configs/train_macaque_arcface_aug2_gpu.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug2_gpu_best.pt`
- Date: 2026-07-19 12:10:37
- Device: cuda
- Logit-adjust tau: 0.0
- Test samples: 1584
- Num classes: 156

## 1. Overall Metrics
- Top-1 accuracy: 0.8194
- Top-3 accuracy: 0.8807
- Top-5 accuracy: 0.9003
- Macro F1: 0.8046
- Macro precision: 0.8340
- Macro recall: 0.7992
- Weighted F1: 0.8170

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug2_gpu_macaque-resnet50-arcface_aug2_gpu_best_aug2_gpu_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug2_gpu_macaque-resnet50-arcface_aug2_gpu_best_aug2_gpu_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Brookes: acc = 0.200
  - Racoon: acc = 0.250
  - Caro: acc = 0.333
  - Scarlet: acc = 0.400
  - Gregory: acc = 0.429

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug2_gpu_macaque-resnet50-arcface_aug2_gpu_best_aug2_gpu_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
