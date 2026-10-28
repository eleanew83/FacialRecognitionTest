# Macaque ReID reports

## Current reports

| Evaluation date | Report | Markdown | Word |
| --- | --- | --- | --- |
| 28 October 2026 | Augmentation and seven-angle averaging comparison | [Read Markdown](markdown/FINAL_EVAL_REPORT_2026-10-28_augmentation_comparison.md) | [Open Word](word/Macaque_ReID_Report_2026-10-28_Augmentation_Comparison.docx) |
| 23 October 2026 | Current 149-identity baseline | [Read Markdown](markdown/FINAL_EVAL_REPORT_2026-10-23_baseline.md) | [Open Word](word/Macaque_ReID_Report_2026-10-23_Baseline.docx) |

The current comparison reports measured test top-1 accuracy of 82.51% for the
baseline, 83.51% for the ±30° policy with single-image prediction, and 84.51% for
that policy with seven-angle probability averaging. One training fit was run per
policy; these measurements do not establish repeated-seed reliability.
The latest report includes a dedicated section and figure for completed averaging
job `39152253`, distinguishing it from the newer ±45°/±60° evaluation submission.

## Folder layout

- [markdown/](markdown/): all Markdown evaluation reports and the historical
  long-tail meeting summary. Ordinary setup, training and inference guides remain
  in `docs/`.
- [word/](word/): Word reports, including the original
  [Macaque_ReID_Report.docx](word/Macaque_ReID_Report.docx) used as the formatting
  reference and [Model_Results_Updated.docx](word/Model_Results_Updated.docx).
- [word/backups/](word/backups/): the two preserved historical Word snapshots.
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
