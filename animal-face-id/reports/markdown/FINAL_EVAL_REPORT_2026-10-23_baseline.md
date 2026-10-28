# Final Evaluation Report — Baseline Macaque Model — 23 October 2026

This report preserves the baseline evaluation performed on 23 October 2026.
See the [augmentation comparison dated 28 October 2026](FINAL_EVAL_REPORT_2026-10-28_augmentation_comparison.md)
for the subsequent training and seven-angle test-time evaluation results.

- Config: `configs/train_macaque_arcface_aug3_clean.yaml`
- Checkpoint: `artifacts/macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best.pt`
- Evaluation date: 2026-10-23 13:15:10 (timestamp as logged)
- Report updated: 2026-10-28; original evaluation metrics retained
- Word version: [formatted baseline report](../word/Macaque_ReID_Report_2026-10-23_Baseline.docx)
- Device: cuda
- Logit-adjust tau: 0.0
- Split: test
- TTA: none; angles: []
- Samples: 1498
- Num classes: 149

The baseline trained for 30 epochs with the `strong` augmentation profile. Its
explicit affine rotation range was **−15° to +15°**. Restricted RandAugment then
selected two operations at magnitude 5; in the installed torchvision 0.17.2,
each selected `Rotate` operation added +5° or −5°. Thus, the baseline already
included rotation augmentation and its combined rotation could exceed ±15°.
Its best validation top-1 accuracy was **80.49% at epoch 28**.

Implementation references: [local transforms](../../src/datasets/transforms.py),
[baseline training metrics](../../artifacts/macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_metrics.csv),
and [torchvision 0.17.2 RandAugment](https://github.com/pytorch/vision/blob/v0.17.2/torchvision/transforms/autoaugment.py).

## 1. Overall Metrics
- Top-1 accuracy: 0.8251
- Top-3 accuracy: 0.8865
- Top-5 accuracy: 0.8985
- Macro F1: 0.8027
- Macro precision: 0.8246
- Macro recall: 0.8042
- Weighted F1: 0.8239

## 2. Per-class Summary
- Per-class metrics CSV: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_per_class_metrics.csv`
- Confusion matrix: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_confusion_matrix.png`

- Hardest IDs by accuracy:
  - Racoon: acc = 0.000
  - Brookes: acc = 0.200
  - Caro: acc = 0.333
  - Leigh: acc = 0.333
  - SAM to be named: acc = 0.333

## 3. Embedding Space Visualization
- t-SNE / PCA plot: `artifacts/final_eval/train_macaque_arcface_aug3_clean_macaque-resnet50-arcface_aug3_clean_photo_split_20261007_canonical_20261014_best_aug3_canonical_20261014_embeddings_tsne.png`

## 4. Known Limitations / Notes
- Test set size is modest; per-class variation may influence stability.
- Some individuals have low support; interpret tail per-class metrics accordingly.
