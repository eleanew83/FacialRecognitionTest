# Macaque ReID — Long-Tail Check & Evaluation Correction

**Model:** ResNet50 + ArcFace, `artifacts/macaque-resnet50-arcface_aug2_best.pt`
**Data:** 156 individuals · 6607 train / 1361 val / 1584 test crops (all verified present)
**Date:** 2026-06-25

---

## TL;DR
1. **An evaluation bug was deflating our reported metrics by ~15 points.** Eval was applying
   the ArcFace angular *margin* (a training-only device, and it needs the ground-truth label)
   at test time, which lowers the true class score. Scoring correctly with plain cosine
   similarity, the **same checkpoint** goes from Macro-F1 0.63 → **0.82**.
2. **The "long-tail problem" was mostly that bug, not a real failure.** After the fix, the
   head→tail F1 gap shrinks from 0.28 to **0.03**, and the "7 individuals at 0% accuracy"
   become **0**. There is only a mild residual effect: rare IDs have slightly lower *recall*.
3. Dataset imbalance itself is **mild** (Gini 0.29, max/min ratio 24.5×, median 40 imgs/ID).

---

## 1. The evaluation fix (same checkpoint, nothing retrained)

| Metric | Original report (buggy eval) | **Corrected eval (no margin)** |
|--------|-----------------------------:|-------------------------------:|
| Top-1 accuracy | 0.679 | **0.831** |
| Macro-F1 | 0.632 | **0.820** |
| Macro-Precision | 0.662 | **0.841** |
| Macro-Recall | 0.631 | **0.816** |
| Weighted-F1 | 0.678 | **0.835** |

*Why:* ArcFace adds an angular margin during **training** to push classes apart. At **test**
time you score with plain (scaled) cosine similarity. The old code re-applied the margin using
the true labels, penalising the correct class and leaking label info. Fix:
`ArcFaceHead.logits_eval()` in `src/models/losses.py`; eval logic now in
`src/training/evaluate.py`.

*Macro-F1 broken into precision and recall (supervisor request):*
Macro-Precision **0.841**, Macro-Recall **0.816** → recall is the (slightly) weaker half,
i.e. we miss some true matches more than we make false ones.

---

## 2. ID distribution (the long-tail curve)

![ID distribution](artifacts/longtail_baseline_corrected/id_distribution_rank.png)

- 156 IDs, 6607 training images. Images/ID: min **4**, median **40**, max **98**.
- Imbalance ratio (max/min) = **24.5×**; **Gini = 0.29** (0 = balanced).
- Top 10% of IDs hold only **19%** of images — a gentle tail, not a steep one.
- Groups used below: **head** ≥30 imgs (103 IDs), **mid** 15–29 (42 IDs), **tail** <15 (11 IDs).

---

## 3. Does the head dominate performance? (corrected eval)

![Head vs tail](artifacts/longtail_baseline_corrected/head_vs_tail_bars.png)

| Group | #IDs | mean train imgs | accuracy | precision | recall | f1 |
|-------|-----:|----------------:|---------:|----------:|-------:|---:|
| head (≥30) | 103 | 53.9 | 0.84 | 0.84 | 0.84 | **0.84** |
| mid (15–29) | 42 | 22.7 | 0.77 | 0.82 | 0.77 | 0.79 |
| tail (<15) | 11 | 9.5 | 0.74 | 0.90 | 0.74 | **0.80** |

- Correlation between log(training images) and per-class accuracy: **r = 0.20** (weak).
- Head–tail F1 gap = **0.03**. The only residual tail effect is **recall** (0.74 vs 0.84):
  rare IDs are occasionally missed, but when predicted they're *more* precise (0.90).
- **0 individuals at 0% accuracy.**

![Precision/Recall/F1 vs support](artifacts/longtail_baseline_corrected/support_vs_prf.png)
![Per-class accuracy head→tail](artifacts/longtail_baseline_corrected/per_class_accuracy_by_support.png)

---

## 4. Conclusion & recommendation

- **There is no significant long-tail recognition problem at the corrected numbers.** The
  dramatic head-domination in the earlier analysis was an artifact of the eval bug.
- **Recommended action:** present the corrected metrics (0.82 macro-F1) and the accurate
  long-tail breakdown. Lead with macro/per-class metrics, not Top-1.
- **Long-tail mitigations are not needed now.** Code for them is implemented and ready if a
  real tail problem appears later (e.g. on harder data): strong augmentation profile
  (`augment: strong`, RandAugment minus identity-destroying ops, no horizontal flip — current
  eval code also revealed the old training pipeline *was* horizontally flipping, which it
  shouldn't), class-balanced loss (`loss: class_balanced`), and inference-time logit
  adjustment (`--logit-adjust-tau`). See `configs/train_macaque_arcface_ltr.yaml` and
  `scripts/run_ltr_train_eval_gpu.sh`.

### Caveat / how to independently verify the 0.82
The corrected eval is standard ArcFace inference (argmax of cosine similarity to class
prototypes). If the supervisor wants an independent confirmation, we can run a gallery/kNN
cosine eval (embed train as gallery, nearest-neighbour the test set) — it should land near 0.82.

---

## Files
- Corrected metrics: `artifacts/final_eval/..._baseline_metrics_summary.json` (+ per-class CSV)
- Corrected long-tail figures + report: `artifacts/longtail_baseline_corrected/`
- Augmentation preview (if we ever augment): `artifacts/longtail_analysis/augmentation_preview.png`
- ⚠️ The original `FINAL_EVAL_REPORT.md` and `artifacts/longtail_analysis/*` reflect the
  **buggy** eval — superseded by the `_baseline`/`_corrected` outputs above.
