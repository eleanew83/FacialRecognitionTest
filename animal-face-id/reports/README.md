# Macaque ReID reports

## Current reports

| Evaluation date | Report | Markdown | Word |
| --- | --- | --- | --- |
| 31 October 2026 | Selected-model error analysis: every individual, confusion pairs and visual review | [Read Markdown](markdown/ERROR_ANALYSIS_2026-10-31.md) | — |
| 28 October 2026 | Augmentation and seven-angle averaging comparison | [Read Markdown](markdown/FINAL_EVAL_REPORT_2026-10-28_augmentation_comparison.md) | [Open Word](word/Macaque_ReID_Report_2026-10-28_Augmentation_Comparison.docx) |
| 23 October 2026 | Current 149-identity baseline | [Read Markdown](markdown/FINAL_EVAL_REPORT_2026-10-23_baseline.md) | [Open Word](word/Macaque_ReID_Report_2026-10-23_Baseline.docx) |

The 31 October error analysis uses the selected ±45° + seven-angle mean model.
[Open the searchable offline report](figures/error_analysis_2026-10-31/error_analysis.html)
for all 149 individuals and every validation/test error, with true/predicted-ID
training references. [PDF and example sheets](pdf/Macaque_ReID_Error_Analysis_2026-10-31.pdf),
[per-individual metrics](figures/error_analysis_2026-10-31/per_individual_metrics.csv),
and a [blank error-review queue](figures/error_analysis_2026-10-31/error_review_queue.csv)
are included. Each PDF photo now has a complete crop filename;
[a filename index](figures/error_analysis_2026-10-31/pdf_image_filenames.csv)
maps selected examples and training references to PDF pages and original source photos.
[All 437 error filenames](figures/error_analysis_2026-10-31/all_error_filenames.csv)
are listed separately. This analysis does not replace the complete manual crop audit.

The 28 October report was consolidated on 30 October for supervisor review.
The main figures compare the baseline and all five augmentation policies together:
shared training/validation loss curves and final held-out test accuracy plus cross-entropy.
Final test NLL is derived from saved probability arrays for the five augmented
models; baseline test NLL was not recorded. Test NLL is margin-free and is shown
separately from the margin-adjusted training/validation loss.

The validation-based choice remains ±45° + seven-angle mean: 85.11% test top-1
versus 84.51% for ±30° mean and 82.51% for the original baseline without TTA.
All-policy diagnostic/accuracy plots, source hashes and meeting-image examples
are included. One fit per policy does not establish repeated-seed reliability.
[Consolidated comparison figures and data](figures/augmentation_comparison_2026-10-28/supervisor_comparison/)
and [full-size meeting image sheets](figures/augmentation_comparison_2026-10-28/meeting_examples/)
are available as PNG/PDF exports.

[Full report PDF for sharing](pdf/Macaque_ReID_Report_2026-10-28_Augmentation_Comparison.pdf)
· [One-page comparison sheet](figures/augmentation_comparison_2026-10-28/supervisor_comparison/one_page_overview.pdf).

## Folder layout

- [markdown/](markdown/): all Markdown evaluation reports and the historical
  long-tail meeting summary. Ordinary setup, training and inference guides remain
  in `docs/`.
- [word/](word/): Word reports, including the original
  [Macaque_ReID_Report.docx](word/Macaque_ReID_Report.docx) used as the formatting
  reference and [Model_Results_Updated.docx](word/Model_Results_Updated.docx).
- [word/backups/](word/backups/): preserved historical report snapshots.
- [figures/](figures/): shared PNG/PDF charts and their source data. The latest
  figures are in [augmentation_comparison_2026-10-28/](figures/augmentation_comparison_2026-10-28/).

Older reports describe earlier splits, identity counts and software states.
Retain those dates and caveats when comparing results; the October 2026 comparison
uses 149 canonical identities and the same 1,498 test images for all models.
Paths shown in backticks in older reports are relative to `animal-face-id/` unless
stated otherwise. Markdown hyperlinks and embedded figures use paths relative to
their report file.

New evaluations save Markdown summaries under `reports/markdown/`; raw evaluation
CSVs, JSON summaries and model artifacts remain in `artifacts/`. The dated Word
reports are presentation documents following the original report's styles.
