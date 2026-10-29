# Final Evaluation Report — Augmentation and Rotation Averaging — 28 October 2026

- Report date: 2026-10-28
- Revision: 2026-10-29; completed ±45°/±60° training and averaging results added; original report date retained
- Word version: [formatted report matching the earlier progress report](../word/Macaque_ReID_Report_2026-10-28_Augmentation_Comparison.docx)
- Training dates: 2026-10-26 (±30°/±100°/±150°) and 2026-10-28 (±45°/±60°); all five policies completed 30 epochs
- Evaluation dates: 2026-10-28 (original suite) and 2026-10-29 (±45°/±60° follow-up); latest selected test result logged at 13:52:23 GMT
- Model: ImageNet-pretrained ResNet50; 256-dimensional embeddings; ArcFace head
- Training loss: cross-entropy on ArcFace margin logits; margin 0.5; scale 30
- Dataset: 149 canonical identities; 6,665 train / 1,302 validation / 1,498 test crops
- Identity-registry version: 2026-10-14; this is the label-registry date, not the report date
- Device: CUDA / NVIDIA A100; one GPU per job
- Evaluation jobs: `39152253` (00:17:55) and `39484294` (00:30:12); both completed with exit code `0:0`
- Baseline report: [23 October 2026 baseline evaluation](FINAL_EVAL_REPORT_2026-10-23_baseline.md)
- Original comparison: [18 validation/test comparisons](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/eval/comparison.csv)
- Follow-up comparison: [12 validation/test comparisons](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/comparison.csv)
- Follow-up directory: [±45°/±60° artifacts](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV)
- Original experiment directory: [artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT)

## 1. Overall Metrics and Comparison with Baseline

**The ±45° training policy with seven-angle probability averaging is the
validation-selected follow-up configuration and has the highest measured test
top-1 accuracy: 85.11%, with 83.74% macro F1.** The previous ±30° + mean result
was 84.51% and 82.84%. Both mean configurations tie on validation top-1 (83.56%);
±45° has higher validation macro F1 (80.11% versus 79.82%), consistent with the
saved selection rule. These are measured results from one fit per policy;
repeated-seed reliability and statistical significance have not been established.

**Validation and test accuracy are different measurements.** Without averaging,
±30° remains highest on validation (82.49%, versus 82.10% for ±45° and 82.18%
for ±60°). On the test set, ±30° and ±45° tie at 83.51%; ±60° scores 83.11%.
The ±45° advantage appears after averaging, not in single-image test top-1.

**Original baseline** means the previous model, trained with affine rotation from −15° to +15° plus possible extra RandAugment rotation. **±30° policy, single image** means the newly trained model, tested once per image. **±30° + seven-angle mean** reuses that same new checkpoint and averages class probabilities across test views at −30°, −20°, −10°, 0°, +10°, +20° and +30°. **±45° + seven-angle mean** uses the follow-up ±45° checkpoint with the same seven inference angles. TTA stands for test-time augmentation. Relative to the original baseline, the new policies also change scale, noise, blur and other settings. Within the ±30°/±45°/±60° follow-up, only the rotation range changes.

| Configuration | Test top-1 | Test top-3 | Test top-5 | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Original baseline, single image | 82.51% | 88.65% | 89.85% | 80.27% | 82.39% |
| ±30° policy, single image | 83.51% | 88.45% | 89.99% | 81.89% | 83.49% |
| ±30° + seven-angle mean (original selection) | 84.51% | 89.65% | 90.72% | 82.84% | 84.44% |
| ±45° policy, single image | 83.51% | 88.92% | 90.52% | 81.67% | 83.33% |
| ±45° + seven-angle mean (follow-up selection) | 85.11% | 89.92% | 91.32% | 83.74% | 84.97% |

For ±45°, single-image inference improves over the original baseline by **1.00
percentage point**, and averaging adds **1.60 points** to that same model.
The combined improvement is **2.60 points** in top-1 and **3.47 points** in macro
F1 over the baseline. The selected follow-up correctly identifies **1,275/1,498
images**, versus 1,236 for the baseline and 1,266 for ±30° + mean: gains of
**39 and 9 images**, respectively. The ±45° mean result exceeds ±30° mean by
**0.60 points** in top-1 and **0.90 points** in macro F1. Comparing predictions
image by image, it fixes 72 baseline errors and introduces 33 errors on images
the baseline got right.

### Averaging run results — job 39152253

The averaging evaluation **completed successfully on 28 October 2026 at 00:14:34**
(job start: 27 October at 23:56:39; elapsed **17m 55s**, exit code `0:0`). It scored
the saved ±30°, ±100° and ±150° models using a single image, probability averaging
and max pooling, on validation and test: **18 completed comparisons** in total.
Here, **Mean TTA means seven-angle probability averaging** and **Max TTA means
seven-angle maximum pooling**.

The table below isolates inference for the **same ±30° checkpoint**, selected at
training epoch 29. All seven views use −30°, −20°, −10°, 0°, +10°, +20° and +30°.

| Scoring method | Validation top-1 | Test top-1 | Test top-3 | Test top-5 | Test macro F1 | Correct test images |
| --- | --- | --- | --- | --- | --- | --- |
| Single image | 82.49% | 83.51% | 88.45% | 89.99% | 81.89% | 1251/1,498 |
| Seven-angle mean (selected) | 83.56% | 84.51% | 89.65% | 90.72% | 82.84% | 1266/1,498 |
| Seven-angle max | 83.26% | 84.31% | 89.45% | 90.65% | 82.51% | 1263/1,498 |

For this model, averaging adds **1.00 percentage point** over single-image
prediction: **84.51% versus 83.51%**, or **15 additional correct test images**.
Its macro F1 rises from **81.89% to 82.84%**. Max pooling reaches **84.31%**;
mean is higher by three test images. Mean was selected on validation (83.56%),
not by choosing the highest test score.

![Completed averaging run: baseline and the same ±30° model with single, mean and max scoring](../figures/augmentation_comparison_2026-10-28/averaging_run_39152253.png)

[PDF of the averaging-results figure](../figures/augmentation_comparison_2026-10-28/averaging_run_39152253.pdf).
The baseline bar uses ordinary inference; baseline TTA was not evaluated.

The averaging job evaluates existing weights. The loss-versus-epoch graphs later
in this report belong to the training runs; averaging does not produce a separate
training-loss curve. Its [job log](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/logs/39152253_tta.out)
and [full comparison CSV](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/eval/comparison.csv)
provide the original source results. The completed ±45°/±60° follow-up is reported below. The ±30° selection and figure above describe the original suite.

### Follow-up averaging results — job 39484294 (29 October revision)

The follow-up evaluation **completed successfully on 29 October 2026 at 13:53:14
GMT**, after starting at 13:23:02: **30m 12s**, exit code `0:0`. Both training jobs,
`39484293_0` (±45°) and `39484293_1` (±60°), completed all 30 epochs on 28 October
at 23:12:39, with exit code `0:0`. Evaluation produced **12 comparisons**:
two models × three scoring methods × validation/test.

The follow-up changes only the training rotation range relative to ±30°; scale,
translation, noise, blur, erasing, split, seed and training budget remain the same.
Inference still uses **−30°, −20°, −10°, 0°, +10°, +20°, +30°**, including for the
±45° and ±60° training policies. These are training ranges, not new TTA angles.

| Training policy | Scoring method | Validation top-1 | Test top-1 | Test macro F1 | Correct test images |
| --- | --- | --- | --- | --- | --- |
| ±45° | Single image | 82.10% | 83.51% | 81.67% | 1,251/1,498 |
| ±45° | Seven-angle mean | 83.56% | 85.11% | 83.74% | 1,275/1,498 |
| ±45° | Seven-angle max | 83.49% | 84.78% | 83.49% | 1,270/1,498 |
| ±60° | Single image | 82.18% | 83.11% | 81.20% | 1,245/1,498 |
| ±60° | Seven-angle mean | 83.41% | 84.98% | 83.37% | 1,273/1,498 |
| ±60° | Seven-angle max | 83.03% | 84.98% | 83.78% | 1,273/1,498 |

The saved validation-only selection chose **±45° + mean** before test scoring.
It improves its own single-image test top-1 from **83.51% to 85.11%**: **24 more
correct images**. The ±60° mean and max methods both score 84.98% (1,273/1,498),
two images fewer than ±45° mean. ±60° max has a slightly higher test macro F1
(83.78% versus 83.74%); this does not change the validation-selected choice.
The small differences do not establish a robust optimal training angle.

![Single-image, mean and max test accuracy for ±30°, ±45° and ±60° training](../figures/augmentation_comparison_2026-10-28/rotation_followup_39484294.png)

[PDF of the follow-up comparison](../figures/augmentation_comparison_2026-10-28/rotation_followup_39484294.pdf).
Source: [follow-up comparison CSV](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/comparison.csv),
[validation selection](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/selection.json),
[job log](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/logs/39484294_tta.out),
and [figure data and hashes](../figures/augmentation_comparison_2026-10-28/rotation_followup_39484294_data.json).

### Test results for every policy and pooling method

| Training policy | Single-image top-1 | Mean TTA top-1 | Max TTA top-1 | Mean TTA macro F1 |
| --- | --- | --- | --- | --- |
| Previous baseline | 82.51% | Not evaluated | Not evaluated | Not evaluated |
| ±30° policy | 83.51% | 84.51% | 84.31% | 82.84% |
| ±45° policy | 83.51% | 85.11% | 84.78% | 83.74% |
| ±60° policy | 83.11% | 84.98% | 84.98% | 83.37% |
| ±100° policy | 81.44% | 82.44% | 81.84% | 79.99% |
| ±150° policy | 56.94% | 58.41% | 57.48% | 46.36% |

Baseline TTA was not evaluated. Therefore the combined 2.60-point gain compares a
new training policy plus TTA against the original baseline without TTA; it is not
an estimate of training augmentation alone. Mean exceeds max for the ±30° model by
only 0.20 percentage points (three test images), so that small difference should
not be treated as a robust general finding.

### Validation results used to select the configuration

| Training policy | Single-image top-1 | Mean TTA top-1 | Max TTA top-1 |
| --- | --- | --- | --- |
| Previous baseline | 80.49% | Not evaluated | Not evaluated |
| ±30° policy | 82.49% | 83.56% | 83.26% |
| ±45° policy | 82.10% | 83.56% | 83.49% |
| ±60° policy | 82.18% | 83.41% | 83.03% |
| ±100° policy | 79.49% | 81.26% | 80.72% |
| ±150° policy | 56.30% | 58.06% | 55.99% |

Each checkpoint was first chosen by its highest ordinary validation top-1
during training. The original job compared nine model/pooling combinations and
selected ±30° + mean; the follow-up compared six and selected ±45° + mean.
Each job locked its validation choice before test scoring. The selection records
remain separate: [original](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/eval/selection.json)
and [follow-up](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/selection.json).
Across the two suites, ±30° mean and ±45° mean tie at exactly 1,088/1,302 correct
validation images; ±45° has higher validation macro F1 (80.11% versus 79.82%),
so the recorded tie-break rule favours ±45° without using test results.

For the selected follow-up, test macro precision is **85.15%** and macro recall
is **84.09%**. Logit adjustment was disabled (`tau=0`); evaluation uses
margin-free ArcFace logits throughout.

## 2. What Rotation Was Used Before?

The previous `strong` profile explicitly used **`RandomAffine(degrees=15)`**, which
samples an affine rotation from **−15° to +15°**. It then applied restricted
RandAugment with two operations and magnitude 5. In the installed torchvision
0.17.2 implementation, a selected `Rotate` operation contributes **+5° or −5°**;
operations are sampled sequentially and can repeat. The total baseline rotation
could therefore exceed ±15°. It should not be described as an unaugmented or
strictly ±15° baseline.

The new policies use one affine rotation step and do not add RandAugment rotations.
The names ±30°, ±45°, ±60°, ±100° and ±150° describe training angle ranges, not fixed rotations. Only rotation differs among the ±30°/±45°/±60° policies; the ±100°/±150° policies also change other augmentation settings.

| Policy | Training rotation | Affine scale | Translation | Noise | Gaussian blur | Random erasing |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | Affine ±15°; RandAugment may add rotations | 0.90–1.10 | ±8% | No explicit Gaussian noise | 35%; σ 0.1–2.0; 5×5 | 30%; area 2–12% |
| ±30° | Single affine: −30° to +30° | 0.80–1.25 | ±10% | 35%; σ 0.03 | 40%; σ 0.2–2.0; 7×7 | 20%; area 2–10% |
| ±45° | Single affine: −45° to +45° | 0.80–1.25 | ±10% | 35%; σ 0.03 | 40%; σ 0.2–2.0; 7×7 | 20%; area 2–10% |
| ±60° | Single affine: −60° to +60° | 0.80–1.25 | ±10% | 35%; σ 0.03 | 40%; σ 0.2–2.0; 7×7 | 20%; area 2–10% |
| ±100° | Single affine: −100° to +100° | 0.70–1.35 | ±12% | 45%; σ 0.05 | 50%; σ 0.2–2.5; 7×7 | 25%; area 2–12% |
| ±150° | Single affine: −150° to +150° | 0.60–1.50 | ±15% | 50%; σ 0.08 | 60%; σ 0.3–3.0; 7×7 | 30%; area 2–15% |

The baseline also uses a random resized crop (area scale 0.8–1.0) and restricted
RandAugment. The new policies resize to 224×224 before the affine operation and
use explicit colour jitter instead of RandAugment. None of these six policies
uses horizontal mirroring. Noise sigma is specified for float images in [0,1]
before normalization. Scale and translation in the baseline row describe its
explicit affine step; its other geometry operations can add further variation.

At test time, TTA uses exactly **−30°, −20°, −10°, 0°, +10°, +20°, +30°**, regardless
of the training policy. Each view produces margin-free class logits followed by
softmax. Mean pools the seven probability vectors; max takes each class's maximum
view probability and renormalizes. TTA does not retrain or average model weights.

Implementation references: [local transforms](../../src/datasets/transforms.py),
[RandomAffine 0.17 documentation](https://docs.pytorch.org/vision/0.17/generated/torchvision.transforms.RandomAffine.html),
and [RandAugment 0.17.2 source](https://github.com/pytorch/vision/blob/v0.17.2/torchvision/transforms/autoaugment.py).

## 3. Loss-Function Graphs and Learning Curves

The existing curves below cover the baseline and the original ±30°/±100°/±150° policies, using the actual 30 logged epochs without smoothing. They do not include the follow-up ±45°/±60° runs; those results are added to the training table and the dedicated follow-up figure. The model,
optimizer, learning-rate schedule, batch size and seed are held fixed across the
baseline and new runs: AdamW, initial LR 3e-4, weight decay 0.01, cosine schedule,
batch size 32 and seed 42.

**Loss interpretation:** both training and validation loss are the logged
cross-entropy on **margin-adjusted ArcFace logits**. The validation loop applies
the margin for its loss calculation, while validation accuracy and all final
evaluation scores use **margin-free** logits. Consequently these loss values are
not the cross-entropy of the final evaluation probabilities. Checkpoint selection
uses validation top-1 accuracy, not the minimum loss.

### Loss overview: baseline and the original three policies

![Training and validation loss for all four policies](../figures/augmentation_comparison_2026-10-28/loss_comparison.png)

[Exportable PDF of the loss comparison](../figures/augmentation_comparison_2026-10-28/loss_comparison.pdf).
Both panels use the same loss scale.

### Training and validation loss for each original policy

![Per-policy training and validation loss, with baseline validation as reference](../figures/augmentation_comparison_2026-10-28/loss_by_policy.png)

[Exportable PDF of the per-policy loss curves](../figures/augmentation_comparison_2026-10-28/loss_by_policy.pdf).
Vertical dashed lines mark the checkpoint epochs selected by validation accuracy.
The dotted grey line is the previous baseline's validation loss.

### Accuracy corresponding to the original loss curves

![Training and validation top-1 accuracy for baseline and all policies](../figures/augmentation_comparison_2026-10-28/accuracy_comparison.png)

[Exportable PDF of the accuracy curves](../figures/augmentation_comparison_2026-10-28/accuracy_comparison.pdf).
Training accuracy is measured on each policy's augmented inputs; validation uses
the common unaugmented inputs. Training accuracies therefore reflect different
input difficulty, rather than a shared clean-training benchmark.

| Policy | Training job | Best epoch | Best val top-1 | Final train top-1 | Final train loss | Final val loss | Wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (affine ±15° + RandAugment) | 38494871 | 28 | 80.49% | 99.79% | 0.096 | 5.742 | 00:24:36 |
| ±30° policy | 39152252_0 | 29 | 82.49% | 99.82% | 0.153 | 5.407 | 00:26:12 |
| ±45° policy | 39484293_0 | 27 | 82.10% | 99.61% | 0.235 | 5.325 | 00:26:24 |
| ±60° policy | 39484293_1 | 30 | 82.18% | 99.70% | 0.299 | 5.492 | 00:26:24 |
| ±100° policy | 39152252_1 | 30 | 79.49% | 96.83% | 2.191 | 6.160 | 00:12:15 |
| ±150° policy | 39152252_2 | 30 | 56.30% | 59.29% | 9.456 | 9.814 | 00:27:54 |

Loss/accuracy data and source CSV hashes are included in
[plot_data.json](../figures/augmentation_comparison_2026-10-28/plot_data.json). Wall times include startup,
data validation and I/O, so they are not a controlled comparison of compute speed.

## 4. Per-class Summary for the Selected Model

- Per-class metrics: [selected ±45° + mean per-class CSV](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/train_macaque_arcface_aggressive_rot45_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_best_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_mean_test_tta_mean_per_class_metrics.csv)
- Per-image predictions: [selected model predictions and hard examples](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/train_macaque_arcface_aggressive_rot45_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_best_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_mean_test_tta_mean_hard_examples.csv)
- Full evaluation summary: [selected model metrics JSON](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/train_macaque_arcface_aggressive_rot45_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_best_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_mean_test_tta_mean_metrics_summary.json)
- Confusion matrix: [149×149 matrix in NumPy format](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/eval/train_macaque_arcface_aggressive_rot45_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_best_macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_mean_test_tta_mean_confusion_matrix.npy)

The five lowest-recall identities for the selected configuration are:

| Identity | Test images | Recall / per-class accuracy | F1 |
| --- | --- | --- | --- |
| Caro | 3 | 0.00% | 0.00% |
| Thief | 3 | 33.33% | 50.00% |
| Harry | 7 | 42.86% | 54.55% |
| Racoon | 4 | 50.00% | 66.67% |
| Nathalie | 9 | 55.56% | 62.50% |

One identity has zero test recall. Low-support identities
remain sensitive to individual errors; overall accuracy alone does not describe
all individuals equally. New embedding projections were not computed by this
comparison job, so the baseline's t-SNE plot is not reused for the new model.

## 5. Interpretation

The ±30° policy improves both single-image validation and test accuracy over the
baseline. Averaging seven rotated views adds a further measured gain. This supports
moderate additional augmentation and rotation averaging for this experiment.

The rotation-only follow-up now compares ±45° and ±60° against ±30° while
holding the remaining settings fixed. ±30° retains the highest single-image
validation accuracy; ±30° and ±45° tie on single-image test top-1. Averaging
benefits the follow-ups more: ±45° mean reaches 85.11% and ±60° mean 84.98%,
compared with 84.51% for ±30° mean. The ±45°/±30° mean validation top-1 tie is
broken by macro F1. This supports ±45° + mean as the current validation-based
choice, with a small measured test gain that needs repeated-seed confirmation.

The ±100° and ±150° policies perform worse under the same 30-epoch budget. The
±150° model finishes with only 59.29% accuracy on its augmented training batches
and 56.30% validation accuracy. This is consistent with difficulty fitting the
strongly transformed inputs at this budget. It does not establish that longer
training would outperform the moderate policy.

A plausible mechanism is that large rotations, zoom and translation push useful
regions outside the fixed crop, while stronger noise, blur and erasing reduce
the remaining identity cues. This is an interpretation, not a separately measured
cause. The original policy comparison changes several operations together, and the baseline uses a different crop/RandAugment pipeline. The ±30°/±45°/±60° follow-up does isolate the configured rotation range while holding other settings fixed, but one fit per angle cannot establish a robust optimal angle.

## 6. Known Limitations and Reproducibility

- One seed and one training fit per policy; no repeated-seed confidence interval
  or significance claim is made.
- Baseline TTA was not tested, so the effect of adding the same inference ensemble
  to the old model remains unknown.
- This is a fixed-budget comparison. More epochs or a different optimization
  schedule could change the ranking, but that has not been evaluated.
- All 15 non-baseline test evaluations were checked against the baseline's exact 1,498
  `(image path, true identity)` pairs. No new split or label order is introduced.
- The split manifest and identity-registry hashes match the suite's saved
  [original data-input provenance](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/data_inputs.json)
  and [follow-up provenance](../../artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/data_inputs.json). Both saved SHA-256 pairs match.
- Training logs: [artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/logs](../../artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT/logs). Training
  used jobs `39152252_0`, `_1`, `_2`; evaluation used `39152253`; all completed
  successfully with exit code `0:0`. Follow-up training used `39484293_0` and `_1`; evaluation used `39484294`, also all completed with exit code `0:0`.
- The complete evidence comprises 30 validation/test comparisons across both suites. The existing loss figures cover only the original suite; the follow-up has its own result figure.
- Best and final checkpoints, 30-epoch CSVs, per-image/per-class evaluation files,
  and the seven-view probability arrays remain in the experiment directory.

The selected follow-up checkpoint is:

`artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV/checkpoints/macaque-resnet50-arcface_aggressive_rot45_canonical_20261014_best.pt`

The ±45°/±60° follow-up proposed in the original report has now completed.
The current result favours **±45° training with mean probability TTA** under
the validation top-1/macro-F1 rule. Repeat the strongest candidates across seeds
before interpreting the small differences as robust. Applying the same TTA to
the old baseline checkpoint remains an unevaluated comparison. This revision
documents saved results; it launches no additional training or evaluation jobs.
