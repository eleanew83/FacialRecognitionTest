# Data Preparation Guide

The active dataset uses a checked manifest at `data/macaque_faces/splits.json`.
Each record contains an identity, its path relative to `data.raw_root`, file and
decoded RGB hashes, and the file size/timestamp checked during the repair.

The current dataset has **156 identity labels** and **6665 train / 1302 validation /
1498 test crops** (9465 total). All 156 labels appear in every split.

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
