# Macaque identification error analysis — 2026-10-31

**Selected model: ±45° training + seven-angle mean**, chosen using validation. Saved outputs only; no model or dataset changes.

[PDF report](../pdf/Macaque_ReID_Error_Analysis_2026-10-31.pdf) · [Searchable report and all error images](../figures/error_analysis_2026-10-31/error_analysis.html) · [Per-individual CSV](../figures/error_analysis_2026-10-31/per_individual_metrics.csv) · [Review queue](../figures/error_analysis_2026-10-31/error_review_queue.csv)

**Photo coverage:** 32 selected error queries (16 validation, 16 test), with two illustrative training references per query; one flagged training crop is shown separately. These are not all 437 errors. Labels have not been independently verified.

[Every displayed photo: filenames, PDF pages and original sources](../figures/error_analysis_2026-10-31/pdf_image_filenames.csv) · [All 437 error filenames](../figures/error_analysis_2026-10-31/all_error_filenames.csv)

## 1. Measured findings

| Measure | Validation | Test |
| --- | --- | --- |
| Correct / images | 1088/1302 | 1275/1498 |
| Top-1 accuracy | 83.56% | 85.11% |
| Macro recall | 80.00% | 84.09% |
| Macro F1 | 80.11% | 83.74% |
| Errors | 214 | 223 |
| Individuals with errors | 109 | 108 |
| Individuals with recall below 50% | 6 | 3 |
| Same-sex / cross-sex errors | 169 / 45 | 174 / 49 |
| Wrong with model score ≥90% | 42 | 44 |
| Original crops <128px: correct / images | 17/36 | 11/24 |
| Errors with true ID still in top 5 | 95/214 | 93/223 |
| Distinct directed error pairs | 209 | 216 |
| Maximum directed pair count | 2 | 2 |

Most errors are between individuals of the same sex. Counts are not chance-adjusted: there are 97 female and 52 male identities. This does not predict the benefit of sex supervision. High-confidence mistakes also occur. No directed pair occurs more than twice in either split; errors are distributed over many pairs. **Small crops are a measurable concern:** accuracy for original crops under 128px on the shorter side is 47.22% (17/36) on validation and 45.83% (11/24) on test, compared with about 87% for crops at least 224px on both sides. These small samples differ in other ways too, so this is association rather than a controlled resolution effect.

![Error overview](../figures/error_analysis_2026-10-31/error_overview.png)

## 2. Per-individual results

Per-class accuracy here means **recall**: correctly identified images / evaluation images of that individual. The CSV also includes precision, F1 and false positives. All 149 individuals appear below in validation-recall order, with test results and training support alongside. The dotted line is overall image accuracy, not macro recall. One error among three images changes recall by 33.3 percentage points.

![All individuals, panel 1/3](../figures/error_analysis_2026-10-31/per_individual_recall_1.png)

![All individuals, panel 2/3](../figures/error_analysis_2026-10-31/per_individual_recall_2.png)

![All individuals, panel 3/3](../figures/error_analysis_2026-10-31/per_individual_recall_3.png)

### Lowest recall: validation

| Individual | Correct / images | Recall | Training crops | False positives |
| --- | --- | --- | --- | --- |
| Caro | 0/1 | 0.00% | 9 | 1 |
| Castaf | 0/1 | 0.00% | 9 | 0 |
| Thief | 0/1 | 0.00% | 8 | 0 |
| Rachid | 1/3 | 33.33% | 23 | 1 |
| Kasha | 2/5 | 40.00% | 31 | 1 |
| Beate | 4/9 | 44.44% | 32 | 4 |
| Adele | 3/6 | 50.00% | 30 | 2 |
| Bernie | 3/6 | 50.00% | 48 | 4 |
| Gustave | 2/4 | 50.00% | 39 | 2 |
| Koral | 2/4 | 50.00% | 24 | 2 |
| Pernille | 2/4 | 50.00% | 18 | 0 |
| Tatiana | 2/4 | 50.00% | 22 | 1 |
| Ursa | 2/4 | 50.00% | 27 | 2 |
| Vanessa | 2/4 | 50.00% | 25 | 2 |
| Nadia | 1/2 | 50.00% | 12 | 0 |

### Lowest recall: test

| Individual | Correct / images | Recall | Training crops | False positives |
| --- | --- | --- | --- | --- |
| Caro | 0/3 | 0.00% | 9 | 0 |
| Thief | 1/3 | 33.33% | 8 | 0 |
| Harry | 3/7 | 42.86% | 26 | 1 |
| Racoon | 2/4 | 50.00% | 14 | 0 |
| Nathalie | 5/9 | 55.56% | 44 | 2 |
| Goblin | 7/12 | 58.33% | 70 | 2 |
| Drew | 10/17 | 58.82% | 83 | 4 |
| Brookes | 3/5 | 60.00% | 19 | 1 |
| Rachid | 3/5 | 60.00% | 23 | 0 |
| Scarlet | 3/5 | 60.00% | 18 | 3 |
| Tatiana | 3/5 | 60.00% | 22 | 0 |
| Trinity | 3/5 | 60.00% | 21 | 1 |
| Adele | 5/8 | 62.50% | 30 | 2 |
| Charline | 7/11 | 63.64% | 44 | 1 |
| Pauline | 7/11 | 63.64% | 46 | 2 |

### Individuals below 75% recall in both splits

The 75% cutoff is a descriptive review filter, not a statistical test or model-selection rule. Caro and Thief each have only one validation image. Individuals such as Bijou and Goblin also struggle despite more training images.

| Individual | Validation correct / images | Test correct / images | Training images |
| --- | --- | --- | --- |
| Caro | 0/1 | 0/3 | 9 |
| Thief | 0/1 | 1/3 | 8 |
| Rachid | 1/3 | 3/5 | 23 |
| Tatiana | 2/4 | 3/5 | 22 |
| Adele | 3/6 | 5/8 | 30 |
| Trinity | 2/3 | 3/5 | 21 |
| Nathalie | 5/7 | 5/9 | 44 |
| Goblin | 8/11 | 7/12 | 70 |
| Lola | 3/5 | 5/7 | 26 |
| Gina | 5/8 | 7/10 | 39 |
| Bijou | 7/11 | 9/13 | 55 |
| Etienne | 2/3 | 4/6 | 23 |
| Gregory | 4/6 | 4/6 | 29 |

[Review-priority CSV](../figures/error_analysis_2026-10-31/individual_review_priorities.csv)

![Training support](../figures/error_analysis_2026-10-31/training_support_vs_recall.png)

Training support is descriptive, not a controlled experiment. Image diversity, pose, crop quality and identity difficulty may differ.

## 3. Confused identity pairs

![Confused pairs](../figures/error_analysis_2026-10-31/confused_pairs.png)

### Validation

| A / B | A→B | B→A | Same sex? | Same current group? |
| --- | --- | --- | --- | --- |
| Athena / Josie | 0 | 2 | Yes | Yes |
| Bank / Leny | 1 | 1 | No | No |
| Faustine / Mickey | 2 | 0 | No | Yes |
| Georgina / Moutarde | 1 | 1 | Yes | Yes |
| Mickey / Rocky | 0 | 2 | Yes | No |
| Moutarde / Nathalie | 0 | 2 | Yes | Yes |
| Richard / Triton | 0 | 2 | Yes | No |
| Abby / Cyril | 0 | 1 | No | No |
| Abby / Sonia | 1 | 0 | Yes | Yes |
| Adele / Batman | 1 | 0 | No | Yes |

### Test

| A / B | A→B | B→A | Same sex? | Same current group? |
| --- | --- | --- | --- | --- |
| Adele / Caro | 0 | 2 | Yes | Yes |
| Alexa / Athena | 0 | 2 | Yes | No |
| Athena / Josie | 0 | 2 | Yes | Yes |
| Bijou / Kate | 1 | 1 | Yes | Yes |
| Cactus / Ksenia | 0 | 2 | Yes | No |
| Cerise / Theresa | 1 | 1 | Yes | Yes |
| Cleo / Drew | 0 | 2 | Yes | Yes |
| Julia / Taylor | 1 | 1 | Yes | Yes |
| Lola / Pauline | 0 | 2 | Yes | Yes |
| Noname / Philip | 2 | 0 | Yes | No |

Ties are alphabetical, not ranked by importance. Group is current registry membership, not necessarily group or location at capture.

## 4. Metadata, confidence and original crop size

![Sex and resolution](../figures/error_analysis_2026-10-31/sex_and_resolution.png)

| Split | Attribute | Value | Correct / images | Accuracy | Macro recall |
| --- | --- | --- | --- | --- | --- |
| Validation | true_sex | female | 648/802 | 80.80% | 76.32% |
| Validation | true_sex | male | 440/500 | 88.00% | 86.87% |
| Validation | true_group | Apes Den | 268/307 | 87.30% | 85.45% |
| Validation | true_group | Cable Car | 268/324 | 82.72% | 77.87% |
| Validation | true_group | Middle Hill | 70/96 | 72.92% | 68.80% |
| Validation | true_group | O'Hara | 72/83 | 86.75% | 85.35% |
| Validation | true_group | Prince Philip Arch | 136/166 | 81.93% | 81.08% |
| Validation | true_group | Rock Gun | 57/75 | 76.00% | 75.58% |
| Validation | true_group | Royal Anglian Way | 217/251 | 86.45% | 86.61% |
| Test | true_sex | female | 751/910 | 82.53% | 81.95% |
| Test | true_sex | male | 524/588 | 89.12% | 88.09% |
| Test | true_group | Apes Den | 284/344 | 82.56% | 82.27% |
| Test | true_group | Cable Car | 328/382 | 85.86% | 82.63% |
| Test | true_group | Middle Hill | 104/119 | 87.39% | 87.94% |
| Test | true_group | O'Hara | 85/95 | 89.47% | 87.36% |
| Test | true_group | Prince Philip Arch | 162/189 | 85.71% | 84.25% |
| Test | true_group | Rock Gun | 76/91 | 83.52% | 83.77% |
| Test | true_group | Royal Anglian Way | 236/278 | 84.89% | 84.73% |

| Split | Model score | Images | Errors | Observed accuracy |
| --- | --- | --- | --- | --- |
| Validation | 0%-50% | 109 | 84 | 22.94% |
| Validation | 50%-80% | 134 | 76 | 43.28% |
| Validation | 80%-90% | 35 | 12 | 65.71% |
| Validation | 90%-100% | 1024 | 42 | 95.90% |
| Test | 0%-50% | 107 | 84 | 21.50% |
| Test | 50%-80% | 120 | 79 | 34.17% |
| Test | 80%-90% | 60 | 16 | 73.33% |
| Test | 90%-100% | 1211 | 44 | 96.37% |

Scores are maximum mean class probabilities without calibration; these bins do not establish a safe operational threshold.

| Split | Original shorter side | Images | Errors | Accuracy |
| --- | --- | --- | --- | --- |
| Validation | <64 px | 7 | 5 | 28.57% |
| Validation | 64-127 px | 29 | 14 | 51.72% |
| Validation | 128-223 px | 94 | 39 | 58.51% |
| Validation | >=224 px | 1172 | 156 | 86.69% |
| Test | <64 px | 6 | 4 | 33.33% |
| Test | 64-127 px | 18 | 9 | 50.00% |
| Test | 128-223 px | 125 | 34 | 72.80% |
| Test | >=224 px | 1349 | 176 | 86.95% |

Crop dimensions are measured before resizing. Size is not a blur or crop-correctness label; image populations differ between bins.

## 5. Visual observations: selected examples, not prevalence estimates

Sixteen pages show 32 selected errors: eight highest-confidence and eight smallest-top-two-margin errors per split. Each query is paired with training references for the labelled and predicted identities. Each displayed crop has a complete filename caption and example ID; the filename CSV maps it to its PDF page and original source photograph. References are the middle file in alphabetical training-path order, not nearest neighbours or proof of similarity. Crops are contained without stretching; no training augmentation is applied.

- **val — Beate → Anara**: Strong orange/red colour cast and very dark eye regions. This is an illumination/colour-shift example; crop correctness and identity are not adjudicated. Source: val/Beate/121223 RG AF Beate wpt894 Sylv_6_crop0.JPG
- **val — Cynthia → Jack**: Branches cross the face and deep shadows obscure the eye region. External occlusion and uneven lighting are visible. Source: val/Cynthia/150124 AW AF Cynthia 14h39 AM Jack Martin1_crop0.JPG
- **val — Dorothee → Mel**: Oblique/profile view with strong side/back lighting and bright haze. Facial appearance differs substantially from a uniformly lit frontal view. Source: val/Dorothee/151223 PPA AF Dorothee wpt978 Sylv_2_crop0.JPG
- **val — Pauline → Cerise**: Fairly clear, near-frontal face with both eyes visible. No obvious gross occlusion or low-resolution problem in this displayed crop; a confident error also occurs in apparently usable images. Source: val/Pauline/080124 AD AF Pauline faecal wpt1246 Sylv_M9_crop0.JPG
- **val — Renee → Rachida**: Near-frontal face with both eyes visible, but soft detail and relatively flat contrast. No gross crop failure established. Source: val/Renee/131223 AD AF Renee wpt896 Sylv_2_crop0.JPG
- **val — Rocky → Mickey**: Largely frontal face with dark shadow over the eye region. The face is otherwise visibly present; no identity-label conclusion is drawn. Source: val/Rocky/170124 AW AF Cynthia AM Rocky wpt2902 Martin1_crop0.JPG
- **val — Tack → Jamie**: Fairly clear, near-frontal face with both eyes visible. No obvious gross crop failure in the displayed image; fine-grained identity confusion remains a possibility. Source: val/Tack/260823 OH AM Tack wpt450 Jess16_crop0.JPG
- **val — Bijou → Drew**: Strongly blurred/soft oblique face, with little fine texture visible. Limited image detail is a plausible difficulty, not a demonstrated cause. Source: val/Bijou/190823 AW AF Bijou wpt1820 Sylv1_crop0.JPG
- **test — Cerise → Richard**: Very dark face with poor eye visibility; a hand is visible at the lower edge near the mouth/chin. Illumination and partial obstruction merit review. Source: test/Cerise/131224 AD AM Richard AF Cerise wpt4756 Sylv1_crop0.JPG
- **test — Drew → Cleo**: The two selected Drew-to-Cleo errors show different views: a near-frontal/downward face and a more oblique face. No gross crop failure is established from the crops alone. Source: test/Drew/290823 AW AF Drew during follow 14h05 Sylv_2_crop0.JPG
- **test — Gina → Julia**: Strong profile/three-quarter view: the far eye is not visible and only one side of the facial structure is exposed. Source: test/Gina/171223 AW AF Gina unsure wpt1033 Sylv_3_crop0.JPG
- **test — Mickey → Drew**: Downward/oblique head view with the eyes narrowed or shadowed. Viewpoint and eye visibility are possible difficulties. Source: test/Mickey/P1010611_crop0.JPG
- **test — Nathalie → Sophie**: Close oblique face with the far eye hidden and the face/fur meeting the crop borders. Whether the crop excludes useful information requires the source photo. Source: test/Nathalie/120124 CC AF Nathalie unsure wpt1332 SylvM1_crop0.JPG
- **test — Rocky → Cleo**: Small but apparently usable, mostly frontal crop with both eyes visible. No obvious gross occlusion; an error is not automatically a crop error. Source: test/Rocky/071223 AW AM Rocky wpt746 Sylv_3_crop0.JPG
- **test — Drew → Cleo**: The two selected Drew-to-Cleo errors show different views: a near-frontal/downward face and a more oblique face. No gross crop failure is established from the crops alone. Source: test/Drew/290823 AW AF Drew during follow 14h05 Sylv_4_crop0.JPG
- **test — Richard → Eliane**: Head tilted/downward with a dark eye region; much of the visible crop is forehead and surrounding fur. Viewpoint and eye visibility merit review. Source: test/Richard/080123 AD AM Richard 11h23 Martin3_crop0.JPG

These examples are deliberately selected, without matched correct-image controls. They cannot quantify how common or causal each problem is. Identity labels were not adjudicated. The complete manual YOLO crop audit remains separate.

![val high_confidence, page 1/4](../figures/error_analysis_2026-10-31/examples_val_high_confidence.png)

![val high_confidence, page 2/4](../figures/error_analysis_2026-10-31/examples_val_high_confidence_page_2.png)

![val high_confidence, page 3/4](../figures/error_analysis_2026-10-31/examples_val_high_confidence_page_3.png)

![val high_confidence, page 4/4](../figures/error_analysis_2026-10-31/examples_val_high_confidence_page_4.png)

![val ambiguous, page 1/4](../figures/error_analysis_2026-10-31/examples_val_ambiguous.png)

![val ambiguous, page 2/4](../figures/error_analysis_2026-10-31/examples_val_ambiguous_page_2.png)

![val ambiguous, page 3/4](../figures/error_analysis_2026-10-31/examples_val_ambiguous_page_3.png)

![val ambiguous, page 4/4](../figures/error_analysis_2026-10-31/examples_val_ambiguous_page_4.png)

![test high_confidence, page 1/4](../figures/error_analysis_2026-10-31/examples_test_high_confidence.png)

![test high_confidence, page 2/4](../figures/error_analysis_2026-10-31/examples_test_high_confidence_page_2.png)

![test high_confidence, page 3/4](../figures/error_analysis_2026-10-31/examples_test_high_confidence_page_3.png)

![test high_confidence, page 4/4](../figures/error_analysis_2026-10-31/examples_test_high_confidence_page_4.png)

![test ambiguous, page 1/4](../figures/error_analysis_2026-10-31/examples_test_ambiguous.png)

![test ambiguous, page 2/4](../figures/error_analysis_2026-10-31/examples_test_ambiguous_page_2.png)

![test ambiguous, page 3/4](../figures/error_analysis_2026-10-31/examples_test_ambiguous_page_3.png)

![test ambiguous, page 4/4](../figures/error_analysis_2026-10-31/examples_test_ambiguous_page_4.png)

### A concrete training-crop issue to inspect

- **Eliane**: The selected training reference shows fur/body and a railing, with no recognizable face or visible eyes/nose. Review this YOLO crop against its original photograph. The image was not removed or relabelled; its causal contribution to errors is unknown. [Open the crop](../../../yolo_detection/yolo_detection_code/output/macaque_crops/train/Eliane/131224%20AD%20AF%20Eliane%20wpt4735%20Sylv3_crop0.JPG).

[Recorded training-reference observations](../figures/error_analysis_2026-10-31/training_reference_observations.json)

## 6. What changed relative to earlier predictions

| Comparison | Errors fixed | Errors introduced | Both wrong | Both correct |
| --- | --- | --- | --- | --- |
| Validation: single → mean | 31 | 12 | 202 | 1057 |
| Test: single → mean | 32 | 8 | 215 | 1243 |
| Test: original baseline → selected policy + mean | 72 | 33 | 190 | 1203 |

Single→mean uses the same checkpoint and isolates inference averaging. The baseline comparison changes both training policy and inference. All comparisons are matched by image and true identity; one fit per policy.

## 7. Review priorities and limits

1. Review validation errors for truncation, occlusion, viewpoint, limited detail and possible label issues. Inspect the source photograph before deciding a crop or identity is incorrect.
2. Compare the same conditions among correct predictions before assigning causes. The review CSV contains all errors and blank annotation fields; no guessed cause is pre-filled.
3. Prioritise individuals with poor results across both splits and repeated confident errors, keeping support counts visible. Select experiments using validation.
4. If test-error observations guide redesign, keep this cohort as a historical benchmark and reserve a fresh, source-photo-separated final holdout.

No existing pose, age, visibility, blur or crop-quality labels support automatic causal attribution. Sex/group slices cannot establish mechanisms. This analysis does not establish statistical significance or performance on unseen identities; related photographs may be correlated.

## 8. Reproduction and source integrity

Run from the repository root using the existing project environment: /rds/user/ylj20/hpc-work/venvs/macaque/bin/python animal-face-id/tools/analyze_identification_errors.py

Requires NumPy, Matplotlib and Pillow. No GPU, model loading or internet. Checks: current manifest/registry hashes match the saved run; probabilities reproduce predictions, top-k results, per-class metrics and saved confusion matrices; all 2,800 evaluation crop hashes plus 149 illustrative training-reference hashes match the manifest.

[Source hashes and analysis data](../figures/error_analysis_2026-10-31/analysis_data.json) · [All predictions](../figures/error_analysis_2026-10-31/all_predictions.csv) · [All confused pairs](../figures/error_analysis_2026-10-31/confused_pairs.csv) · [Example manifest](../figures/error_analysis_2026-10-31/example_manifest.csv)
