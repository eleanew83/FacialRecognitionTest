# Data Preparation Guide

## Current reviewed generation, 25 November 2026

The active dataset is [`../data/macaque_faces/reviewed_2026-11-25/`](../data/macaque_faces/reviewed_2026-11-25/).
The current-generation pointer is
[`current_dataset.json`](../data/macaque_faces/current_dataset.json).
It contains **9,583 original photographs** and **9,574 face crops**
across **149 canonical identities**, split into
**6,876 train / 1,271 validation / 1,427 test**.

`originals/` contains independent, checksum-verified copies of the completed active
identity folders. `faces/` combines unchanged reviewed historical crops, earlier
manual face corrections and the 74 approved crops from organised originals.
Excluded, retired and superseded crops are omitted. Infant-only original-photo
cases are retained under the user's policy that infants do not count as additional
individuals. The two explicitly selected Goblin retirements were applied with
transaction backups. The deferred Wally/Pom RAW frame remains only in the preserved
archive and is absent from this generation.

`faces/splits.json` includes byte/RGB hashes, canonical identity metadata, original
photo paths and camera-source groups. Distinct face crops are preserved; exact
bytes or decoded RGB duplicates are consolidated. Every camera photograph and its
derivatives remain in one split, including different individuals from one photo.
Historical membership is retained where possible, with shared-frame conflicts
assigned to train first and identity coverage repaired as whole groups. Each
identity occurs in all three splits. These new splits require fresh training;
historical checkpoints and galleries are not results on this generation.

`summary.json` contains validation results; `audit/` contains input snapshots,
retained-face provenance, exclusions, transaction journals and pre-activation
configs. Active macaque configs now use this generation's face root, manifest and
registry. The historical review manifest and registry remain unchanged.

For a future completed review, assemble into a **new** output directory:

```bash
python tools/assemble_reviewed_dataset.py --output data/macaque_faces/reviewed_NEW_DATE
python tools/assemble_reviewed_dataset.py --output data/macaque_faces/reviewed_NEW_DATE --activate-only
```

The build publishes only after file/content, folder inventory, identity coverage
and source-group separation checks pass. Activation backs up existing configs and
rolls them back if updating the references fails. Failed build stages are kept
separately and do not become the active generation.

The sections below preserve the earlier repairs and their historical counts.

## Comparison with previous results

The November generation was **not randomly reshuffled**. Of 9,389 unchanged face
crops, **9,239 retained their split** and **150 moved** so derivatives of a camera
photo stay together. New/replacement faces follow those camera-source groups.
Physical split counts changed from **6,665 / 1,302 / 1,498** to
**6,876 / 1,271 / 1,427** (train/validation/test).
The final extra test-to-training move fixes a trailing-space filename that hid
a shared camera frame; it is recorded with backups in the generation's audit.

For a matched evaluation, [`benchmark.json`](../data/macaque_faces/comparison_2026-11-25/benchmark.json)
freezes **1,236 validation** and **1,391 test** images
that retain identical bytes, RGB fingerprints and canonical identities in the
corresponding old and reviewed splits. Both subsets cover all **149 identities**.
Photos exposed in training, and test photos exposed in validation, are excluded
by camera frame and content. The primary provenance scope is the October
canonical dataset with a preserved manifest, not the unavailable pre-repair
July training folds.

The main macaque configs use `data.evaluation_subset` and its SHA-256 to select
these samples **after validating the full dataset**. Training still uses all
6,876 reviewed training images. Class indices remain bound to the complete
149-class taxonomy. A changed benchmark or missing/changed sample fails loading;
there is no automatic replacement of evaluation photos. Subset selection does not rewrite the physical face folders
or the published split manifest.

The primary reference is the validation-selected canonical ArcFace ±45° checkpoint
evaluated with seven-angle mean probability averaging on 29 October 2026. Its complete saved predictions reproduce the original full-test
metrics exactly. Restricted to the common test set, its baseline is
**Top-1 85.1905%**, **macro-F1 83.5737%**.
The original full-test Top-1 was **85.1135%** on 1,498 photos.
These are different evaluation cohorts; compare future models with the common
baseline. Older 156-identity headline scores are historical and use a different
taxonomy and photo set.

Two controlled training configurations keep model, augmentation, optimiser,
epoch budget, seed and common validation/test photos matched. Only training data
differs (6,665 pre-review versus 6,876 reviewed crops):

```bash
sbatch scripts/run_train_eval_gpu.sh configs/compare_before_review.yaml before_review_common
sbatch scripts/run_train_eval_gpu.sh configs/compare_after_review.yaml after_review_common
```

These configurations are prepared; no new training results are claimed. The saved
historical checkpoint was selected using its original validation set, so retrain
both configurations for a controlled measurement of the review's training-data
effect. The historical-baseline config is protected from normal dataset-reference
activation.

Evaluation result filenames include the benchmark name, preserving historical
reports. After evaluating a candidate model, compare its complete per-image CSV:

```bash
python tools/compare_reviewed_results.py compare \
  --benchmark data/macaque_faces/comparison_2026-11-25/benchmark.json \
  --predictions /path/to/new_hard_examples.csv \
  --metrics /path/to/new_metrics_summary.json \
  --output /path/to/paired_comparison.json
```

The tool matches exact photos regardless of CSV order, checks their current
bytes and canonical labels, reports Top-1 and macro-F1 deltas, and gives a paired
accuracy interval by resampling camera-source groups. That interval describes
evaluation-sample variability, not variation between training seeds. The primary
comparison uses seven-angle mean TTA and no logit adjustment, matching the reference.
Both matched configs use the original winning ±45° training policy. The evaluation
CLI reads `data.evaluation_tta: mean` automatically; `--tta none` explicitly requests
a secondary single-view score. The earlier aug3 single-view reference is preserved
in `comparison_2026-11-25/reference_aug3_none/`. Alternative
evaluation methods should be reported separately.

For a secondary score on all 1,427 reviewed test crops, use
`data/macaque_faces/comparison_2026-11-25/eval_full_reviewed.yaml` with the same
candidate checkpoint. Keep that score separate from the paired common-cohort
comparison. [`summary.json`](../data/macaque_faces/comparison_2026-11-25/summary.json)
and `split_changes.csv` record the exact counts and changes.

## Reviewed labels and exact duplicate policy, 7 October 2026

The initial repair used normalized source-photo names as well as content hashes.
It quarantined 443 files: 403 source-photo copies/variants and 40 files in 19
conflicting-label groups. That filename-based rule was too broad for different
faces or crops sharing a name.

The subsequent manual review in `label_conflicts_relative_paths.csv` resolved all
19 groups. The current policy consolidates **only exact bytes or decoded RGB
pixels**, including image dimensions. Distinct crops remain, even when names or
source-photo stems match across identity folders. This policy also applies to the
splitter, crop exporter, transfer helper, and dataset loader.

The correction restored **337 distinct crops** and **19 reviewed representatives**.
All 9109 images active after the first repair remain. **85 exact duplicate copies**
remain excluded, alongside the bad Stich crop and the extra Victor version the
review explicitly requested excluding. Excluded originals are preserved in
quarantine. Each exact-content group retains one representative, prioritizing
existing training membership, then validation, then test. Label corrections move
retained images into the correct identity folder and update the manifest.

The initial history remains under `data/macaque_faces/split_repair_2026_10_07/`.
The current correction audit is under
`data/macaque_faces/label_correction_2026_10_07/`:

- `user_review.csv`: a byte-for-byte copy of the user's annotated review.
- `reviewed_labels.json`: interpreted identity corrections and exclusions.
- `label_corrections_applied.csv`: outcomes and current file/representative paths.
- `restored_crops.csv`: all 356 restored files and their current paths.
- `shared_filenames_distinct_crops.csv`: retained crops sharing a filename across
  identities; this is an inventory, not a list of presumed errors.
- `splits.before_correction.json`: manifest backup before reviewed corrections.
- `moves.json`: reversible source/destination moves.
- `exclusions.json`: all 87 remaining excluded files and their reasons.
- `summary.json` and `verification.json`: counts, policy, hashes, and checks.
- `fingerprints.json`: a refreshed inventory of all 9552 original files.

There are no remaining active exact duplicates or conflicting labels on identical
content. No exact image pixels present in the original training set occur in
validation/test. Hash checks cannot establish whether a unique crop has the wrong
identity; that requires visual/reference annotation review. This is an
**exact-content split**, not a guarantee of independent source photos, near-duplicate
crops, or capture sessions. The audit does not use perceptual similarity to remove
images.

## Full source photo split repair, 7 October 2026

The separate `macaque_split_data` source-photo copy was audited by actual file
bytes and decoded RGB pixels. Its 9641 original images contained 72 duplicate
groups with 86 redundant copies. The repair keeps 9555 distinct source images:
**6715 train / 1324 validation / 1516 test**, with all 156 identities in each split.
It quarantines the 86 redundant copies outside the active source tree and
relocates 97 retained photos to match previously reviewed identities and split
assignments. Existing training content stays in training. Distinct photos sharing
a filename remain.

Nineteen source label conflicts follow the existing reviewed crop assignments.
The additional source-only frame
`100824 MH AF Pia SAF Ilary during follow Sylv2.JPG` was explicitly reviewed as
**Pia** by the user. It is retained only in
`macaque_split_data/train/Pia/`; its incorrect `test/Ilary` copy is quarantined.
The decision applies to this exact frame and does not relabel frames 1, 3 or 4.

The source tree now has its own `splits.json` and `split_summary.json`. Records
include current identity/path assignments and content hashes. The audit under
`data/macaque_faces/source_split_repair_2026_10_07/` contains the user's decision,
`reference_updates.csv` mapping every original source path to its current file
location and active representative, reversible move records, quarantined copies
and `verification.json`. Original audit snapshots preserve the pre-repair folder
labels; they are historical evidence rather than active assignments.

The repair rereads every retained file and requires its bytes to match the full
RGB audit before publishing the source manifest. No exact duplicates or
conflicting labels remain on identical active source content. The existing
recognition crop manifest remains unchanged; this particular frame has no crop
entry. The detector annotation for the frame uses the identity-free
`macaque_face` class. Source identity review does not validate its face boxes.

The final alignment check links all **9465 existing recognition crops** to exactly
one retained source-content group with the same identity and train/validation/test
assignment. The source tree has **90 photos without an active crop**, explaining
the different source and crop counts. Current alignment and content verification
records are saved in the source audit directory, along with results for 48 pipeline
integrity regression tests and 5 source repair tests.

## Why duplicates appeared in the earlier pipeline

The original splitter selected disjoint filename slices and cleared its source
split output. One clean call cannot put the same identity/filename in all three
splits. It did not check image content, however, so identical images with different
filenames or labels could be split independently. It also shuffled unsorted
`os.listdir()` results: the same seed did not fix the initial ordering.

Two downstream operations could preserve old assignments:

- The old crop exporter reused populated `macaque_crops` folders. Rebuilding
  `macaque_split_data` did not clear that separate output tree. A photo moved from
  train to test would leave its old training crop behind.
- The previous `rsync_retry.sh` used merge-only rsync transfers, preserving paths
  that disappeared from the source. A later transfer after re-splitting could
  therefore leave both old and new copies.

The original audit found 14 same-identity/same-filename crop groups in all three
splits. Both accumulation mechanisms reproduce that pattern in temporary tests;
original logs are unavailable to distinguish their contribution. The evidence is
saved in `reports/review_2026_10_07/duplicate_cause_reproduction.json`.

The splitter and exporter now require fresh output directories and publish only
complete, validated generations. The exporter checks exact output crop hashes,
because different full images can yield identical highest-confidence face crops.
The crop transfer helper now also requires a fresh destination, verifies received
file hashes, and publishes a complete generation. Existing datasets are preserved.

## Build future splits

Use the project Python environment. The upstream splitter now sorts inputs, uses
a local seed, groups exact duplicate content, and balances identities across the
three splits. Different crops with the same filename remain independent. Identical
content with conflicting labels blocks splitting before output is written. It
requires at least three distinct exact-content groups per identity.

```bash
python ../gorillavision/reid-system/scripts/prepare_macaque_dataset.py \
  --source /path/to/individual_photo_folders \
  --target /path/to/new_source_splits --seed 42
```

The target must be new or empty. The splitter writes into a separate staging
directory under a build lock, checks the copied image contents, then renames the
complete directory into place. It also writes `splits.json` and
`split_summary.json` into that generation. Interrupted builds are kept separately
for review and cannot appear as a completed output. Ratios default to 70/15/15;
grouping and small identity counts mean achieved ratios can differ slightly.

Export YOLO crops into another new or empty directory:

```bash
python ../yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py \
  --mode crop --model /path/to/detector.pt \
  --source-split-dir /path/to/new_source_splits \
  --output-dir /path/to/new_crop_output
```

The exporter fingerprints source images and checks exact-content duplicates before
inference. It writes into a locked staging directory, checks exact output hashes,
image inventory, and identity coverage, and includes a `splits.json` for those
crops. It publishes the complete directory only after validation. Failed writes,
read errors, duplicate crops, and missing identity coverage prevent publication;
failed staging directories remain available for review. Target-face identity on
multi-animal photographs still
requires annotation/manual checking: selecting the highest-confidence face does
not establish its identity.

## Transfer crop datasets safely

`rsync_retry.sh` now calls `tools/sync_macaque_crops.py`. It validates source
contents and the manifest, refuses populated destinations, reserves a fresh
incoming directory under a transfer lock, and retries only inside that directory.
The receiver checks the full image inventory and every SHA-256 before publishing
with a directory rename. It writes a matching `splits.json` with receiver file
timestamps and `transfer_verified.json`. It does not merge into or delete an
existing dataset. The separate `rsync_retry1.sh` artifact helper is unchanged.

Use the project Python environment (3.10+) and a new destination for each dataset
generation. From `animal-face-id/`, first validate without connecting or copying:

```bash
python tools/sync_macaque_crops.py \
  --source /path/to/crops --manifest data/macaque_faces/splits.json \
  --destination user@host:/path/to/macaque_crops_v2 --dry-run
```

Run the same command without `--dry-run` to transfer. If `--manifest` is omitted,
the tool prefers the source directory's bundled `splits.json`, then the project
manifest. After a successful transfer, point `data.raw_root` at the new directory
and `data.splits_path` at its bundled `splits.json` together. Receiver validation
needs only Python's standard library; Pillow is required in the sender's project
environment.

The root wrapper accepts the same arguments, plus `MACAQUE_PYTHON`,
`MACAQUE_CROP_SOURCE`, and `MACAQUE_CROP_DESTINATION` environment overrides.
Its old default destination will be refused if already populated; select a fresh
versioned path. A failed transfer leaves its incoming directory and lock for
inspection. Resolve the failure and inspect/remove that transfer's lock before
starting another attempt at the same destination; existing datasets stay intact.

## Checks before training

`MacaqueFacesDataset` verifies actual byte and decoded RGB hashes and compares the
complete visible image tree with the manifest before loading. Extra stale files,
missing files, changed contents, exact duplicates, label conflicts, and missing
split identities stop training with an error. Hashless legacy manifests also have
their image contents checked.

The first load hashes the dataset. Subsequent train/validation/test loaders in the
same process reuse verified hashes while device, inode, size, modification time,
and change time still agree. Editing an image invalidates that cache even when its
original size and modification time are restored. No persistent hash cache is
trusted at training startup. Different crops sharing a filename remain accepted.

## Training and historical results

The saved July results refer to the old **6607 / 1361 / 1584** manifest. They are
historical and must not be presented as measurements on the repaired splits.
Existing checkpoints can be scored on the remaining held-out photos as a
diagnostic, but use a fresh training run for results on the repaired dataset.

A separate baseline config avoids overwriting earlier checkpoints:

```bash
sbatch scripts/run_train_eval_gpu.sh \
  configs/train_macaque_arcface_clean.yaml clean_photo_split_20261007
```

That command submits a new GPU job; no retraining was submitted during the repair.
Rebuild gallery embeddings from the new checkpoint before using them for
identification. Old galleries include photos excluded by this repair.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```


## Detector dataset safeguards

`src/datasets/detection_integrity.py` is the reusable detector counterpart to
`split_integrity.py`. It is called by `prepare_yolo_dataset.py`,
`update_yolo_detection_data.py`, and `train_yolo_detection.py`. It checks the
actual dataset selected by `dataset.yaml`, every visible image and matching
label, class-0 normalized boxes, missing/orphan labels, repeated boxes and exact
byte/RGB duplicates within or across configured train/val/test splits. Different
crops sharing filenames are allowed. It does not establish semantic annotation
accuracy or independence of near-duplicate frames.

Detector preparation requires a new output directory, groups exact content before
seeded splitting, and publishes only a checked generation under the existing
build lock. Manually reviewed annotations can be supplied with `--labels-dir`,
mirroring source image paths. Contradictory boxes on identical content stop the
build. Optional model proposals require a macaque-face checkpoint, keep all
face boxes and never use a fabricated centre-box fallback. Proposed boxes still
need visual review.

The permanent TSV updater plans the complete edit before touching the dataset.
It validates proposed image and label content, refuses filename collisions and
ambiguous sources/targets, and keeps each target in its existing split. A TSV
`split` column disambiguates same-named distinct crops in train and validation;
`source_relative_path` disambiguates source files. When replacement image pixels
change, supply reviewed labels using `--replacement-labels-dir`; paths in that
directory are `train/original-stem.txt` or `val/original-stem.txt`. Replacing an
image while retaining unrelated old box coordinates is rejected.

Run the updater with `--dry-run` to check an edit without changing the data. Actual
edits stage replacements first, keep backups and an operation journal outside the
active dataset, invalidate label caches, and roll back if the update fails. Script
updates share the preparation lock. A forced process termination between file
replacements can require recovery from the journal; training checks prevent a
partial image/label inventory or duplicate content from being used silently.

Training runs the detector checks before model loading and invalidates derived
label caches so reviewed coordinates are reread even when label file lengths
are unchanged. From the repository root:

```bash
python yolo_detection/yolo_detection_code/scripts/train_yolo_detection.py \
  --data-dir yolo_detection/yolo_detection_data --check-data-only
```

No one-off repair or label-application scripts are needed. The 7 October detector
review records and label backups remain as evidence; training reads only the
configured active image and label folders. Regression coverage is in
`tests/test_detection_integrity.py`.

For a full detector rerun followed by fresh face crops, submit from the repository
root:

```bash
sbatch --time=06:00:00 --job-name=macaque_detect_retrain_crop \
  yolo_detection/yolo_detection_code/scripts/run_detection_gpu.sh both
```

The job checks the separate annotated dataset in `yolo_detection_data`, retrains
YOLOv8n for up to 100 epochs using batch 4 and 416-pixel inputs, and runs the
training validation. It then uses that run's `best.pt` to crop the repaired
`macaque_split_data` while retaining its train/validation/test assignments. Crops
are published only after validation into a new job-specific directory under
`yolo_detection/yolo_detection_code/output/`, with a bundled `splits.json`.
Existing crop generations and recognition inputs remain available.

Old-detector reevaluation is an optional comparison baseline; it does not update
the model for changed detector training data. The source split repair and the
annotated detector dataset are separate. The combined job stops if either
dataset fails its applicable checks. After the new crops pass identity review,
use their directory and bundled manifest together for a new recognition run.
