# Macaque dataset correction and retraining plan — 24 November 2026

Recommendation: keep CSD3 as the authoritative storage location and use the
existing browser review tool as the interface for a new correction stage. Add
a face-recropping editor, an optional original-folder photo editor, a target-ID
selector, and correction previews. Choose the operation separately for each
image. Most crop fixes should produce a tight face crop similar to YOLO's output,
leaving the original photo unchanged. Source-photo cropping, copying and moving
are additional options for cases that need original-folder correction. Save
drafts and back up any source file before applying an approved source edit.
Publish the combined corrections as a new downstream training dataset. A local
package can be a fallback for offline work. No image, label, split, or training
configuration has been changed by this analysis.

## Completed review and planning files

All **9,555 records** have saved decisions, revision **1637**, last saved
24 November 2026 at 03:04:56 UTC. The grid contains 9,465 active recognition
crops and 90 source photographs without active crops. There are 9,335 records
marked OK and 220 requiring attention.

The frozen review and proposed action lists are in
[`correction_plan_2026-11-24/`](../crop_review_2026-11-10/correction_plan_2026-11-24/):

- [`summary.json`](../crop_review_2026-11-10/correction_plan_2026-11-24/summary.json): counts, input hashes, caveats and coverage.
- [`completed_review_state.json`](../crop_review_2026-11-10/correction_plan_2026-11-24/completed_review_state.json): byte-preserved review snapshot.
- [`attention_queue.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/attention_queue.csv): all 220 flagged records and their original notes.
- [`explicit_exclusions.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/explicit_exclusions.csv): 14 explicit exclusion instructions.
- [`new_crop_candidates.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/new_crop_candidates.csv): 78 original-photo candidates after explicit exclusions.
- [`label_and_target_review.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/label_and_target_review.csv): 16 proposed identity/target checks.
- [`uncertain_filename_candidates.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/uncertain_filename_candidates.csv): 322 filename-screen candidates, with review decisions retained.
- [`uncertainty_impact_by_individual.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/uncertainty_impact_by_individual.csv): per-split consequences of excluding every filename candidate.
- [`all_review_actions.csv`](../crop_review_2026-11-10/correction_plan_2026-11-24/all_review_actions.csv): all 9,555 records, proposed actions and blank final-decision/box fields.

These planning files sit under the generated review directory, which is ignored
by Git. Keep separate backups of human decisions. Only the small analysis tool
and written plan need to be version-controlled; no staging was performed here.

The action proposals are a text-based triage of the notes, not confirmed
corrections. They divide the 220 flagged records as follows:

| Proposed work | Records | Interpretation |
| --- | ---: | --- |
| Explicit exclusion | 14 | Two active crops and 12 originals without active crops. |
| Create/inspect a missing face crop | 78 | 76 notes explicitly request manual cropping; Foxy and Victor need source inspection. |
| Existing crop box check or redraw | 21 | Nine explicit/implied recrop requests and 12 crop/target inspection requests. |
| Identity or target check | 16 | Two explicit identity questions, 12 target/relabel checks and two unresolved suggested names. |
| Quality review; retain unless unusable | 91 | Angle, light, blur and partial-face notes without an explicit removal instruction. |

Use the written notes to interpret intent. In particular, the 90 original photos
have UI statuses of 79 `not_croppable` and 11 `unreadable`, but 76 notes request a
manual crop. All 11 `unreadable` originals were successfully decoded during this
read-only check. Neither UI category is a basis for automatically discarding the
whole group.

## Uncertain identities: count and impact

The scan reads filenames only, removes `_cropN` suffixes and normalises
underscores/hyphens before matching uncertainty wording. Folder names and review
notes do not inflate this filename count.

| Filename wording | Records |
| --- | ---: |
| `unsure` or `not sure` | 258 |
| Alternatives containing the word `or` | 19 |
| `likely` | 31 |
| `maybe`, `unclear`, or related wording | 14 |
| Unique candidates | **322** |

These cover 61 assigned individuals: **318 active crops** and **four originals
without crops**. Of the candidates, 312 were marked OK in the completed visual
review. That is evidence of no obvious problem on inspection, not independent
identity confirmation.

The 318 active-crop candidates comprise 229 training, 46 validation and 43 test
images. Excluding all of them would remove 3.36% of active crops, leaving
6,436 train / 1,256 validation / 1,455 test images before any other changes.
This is a sensitivity calculation, not a recommendation.

| Individual | Training before → after blanket exclusion | Validation before → after | Test before → after |
| --- | --- | --- | --- |
| Adele | 30 → 15 | 6 → 3 | 8 → 3 |
| Caro | 9 → 5 | 1 → 0 | 3 → 1 |
| SAM to be named | 8 → 6 | 1 → 0 | 3 → 1 |

A filename can be uncertain about another animal. Four Bobby crops contain
`Yvonne or Fanta ... Bobby`; that wording alone does not say the Bobby target is
uncertain. A Richard filename contains `Richard AF Cerise or NEW2`, which likewise
does not automatically invalidate Richard. Conversely, the Warty examples with
`Jason unsure` and `unclear ID` merit direct expert checking. Do not merge Jason
and Warty without confirmation; the current registry treats them separately.

Recommended policy:

1. Retain the files while classifying the uncertainty as **confirmed target**,
   **uncertainty concerns another animal**, or **target identity unresolved**.
2. Ask Sylvain about the target-unresolved subset, with originals and several
   trusted references. Group related capture events to save review time, but
   confirm each frame before propagating a decision.
3. Include confirmed targets, recording the evidence even if the historical
   filename still says `unsure`.
4. Put genuinely unresolved target labels into a pending/excluded set for the
   first supervised training generation. Keep them out of primary validation
   and test scoring. Resolve the low-support identities first so this does not
   silently eliminate their evaluation coverage.
5. If useful later, run a separate training-only sensitivity experiment adding
   unresolved labels back. Select using verified validation labels. This is an
   optional experiment after a clean baseline, not a prerequisite for correction.

Incorrect labels can misdirect training and make evaluation scores misleading;
removing correct but difficult images can also reduce coverage. The direction or
magnitude of an accuracy change cannot be predicted from the filename counts.
Label-quality research addresses this distinction between predictions and label
quality; model scores can help prioritise a review, but this plan does not use
them as authority for relabelling. [Confident Learning](https://arxiv.org/abs/1911.00068).

## Correction workflow

Implementation update: the correction editor is now available at `/corrections`
on the review server. It supports both face-only cropping and optional original
photo crops/moves/copies, including combined face relabelling with a source move
or copy. Drafts, previews, transactional application, backups and per-record undo
are implemented. The following workflow describes how to use these capabilities
and then prepare the subsequent corrected training generation.

### 1. Freeze the audit; prepare a correction ledger

Use the frozen review snapshot as the input to a new correction ledger. Each
record needs a stable review/source key, original label, proposed/confirmed target
ID, original split, chosen source path and byte hash, action, reviewer, evidence,
and any face box. Preserve review notes verbatim. Keep identity certainty separate
from crop suitability and image difficulty.

The CSVs retain the frozen initial triage and blank final-action/box fields.
Live correction decisions and boxes now save through the editor to
`corrections/state.json`, with applied file operations in transaction journals.
They do not overwrite `review_state.json`.

### 2. Choose the correction operation per image

The editor must support two distinct crop operations, plus identity/folder
actions. Original-folder editing is optional, not a prerequisite for every fix.

| Editor option | Use and result |
| --- | --- |
| **Recrop face** — default for ordinary crop fixes | Open the full-resolution source, draw a tight face box similar to YOLO output, preview it, and save a replacement/new recognition crop. Leave the original-folder photo unchanged. Also works for originals that currently have no crop. |
| **Edit original-folder photo** | Crop the source photo itself to isolate the intended individual, back up the pre-edit photo, then save the edited photo in its identity folder. Regenerate or revalidate downstream face crops for this changed source. |
| **Move original to another folder** | Correct the original photo's confirmed identity/folder assignment, optionally after isolating that individual with a source crop. |
| **Copy original to another folder** | Retain the current assignment and add another individual's source image, optionally with a different source crop. Preserve their common capture/source-group identity. |
| **Relabel face crop** | Keep useful face pixels and change the confirmed recognition identity without necessarily editing or moving the original photo. |
| **Exclude or defer** | Record a removal reason or pending identity question. |

Notes such as “manual crop” or a poorly framed face normally suggest **Recrop
face**. Notes such as “wrong individual”, “recrop original” or “move folder”
should bring original-folder and identity options to attention, but must not
automatically select or apply a source edit. Show the original notes, provide a
suggested operation, and let the user explicitly choose the action per record.

In face mode, show the current crop alongside the selected original and the
replacement face-crop preview. In original-folder mode, show the larger source
region that will replace the folder photo. Label the save buttons distinctly:
**Save face crop** versus **Save original-folder edit**. The face crop and source
region are separate annotations with separate coordinate frames/source hashes.

For example, a Kasha-folder photo containing three individuals may need an
original-folder edit: select a region containing Kasha alone and save that
region as the folder photo. Another Kasha photo whose face is simply cut off
by YOLO needs only a new face box; its original can remain exactly as it is.
If both Kasha and Gustave should be retained from a scene, separately isolate
their source regions or draw their face boxes according to the selected mode.
All resulting samples retain the same capture group and split.

Offer **Open original folder photo** and **Open earlier full-frame version**
where available in either mode. Show the chosen path, dimensions and version:
a master-folder photo may already have an older manual crop. Tie every saved
box to the exact displayed image bytes and orientation. Render edited pixels
from full-resolution images, not thumbnails or browser screenshots.

Only original-folder operations replace or move source files. Before applying
one, preserve the pre-edit image, record the old/new paths and hashes, and use
a new temporary file and atomic replacement rather than truncating a shared
hard-linked inode. Keep undo and collision checks. Face-only operations write
derived crop outputs and never modify the source folder.

Manual face crops are authoritative outputs for those records; a later YOLO
export must not silently overwrite them. For records with source-folder edits,
mark the old face crop as stale and regenerate it or explicitly revalidate a
manual face box against the edited source. Do not reuse coordinates from a
different source version without transforming or rechecking them.

### 3. Resolve identity decisions

Prepare Sylvain's packet with the current crop, original, candidate identities,
trusted references, exact filename and a specific question. Prioritise:

- Caro, SAM to be named and Adele because their support is most sensitive to
  blanket exclusions;
- Warty/Jason and the `unclear ID` examples;
- the Bank and Bijou identity questions;
- the proposed Moutarde/Nathalie crop swap, Kasha/Gustave cases, Coco/Drew,
  Kate/Jack and Wally/Pom target choices;
- the suggested name `Natalie`, which is not canonical (`Nathalie` is), and `INF`,
  which describes an infant rather than an available canonical identity.

Do not automatically convert tentative wording such as “I think” into an expert
identity decision. Record whether Sylvain confirms the old ID, a replacement ID,
or cannot identify the animal.

### 4. Assemble the combined corrections into a new dataset generation

Publish a new versioned dataset and manifests combining approved manual face
crops, face-label corrections, optional original-folder edits, valid unchanged
crops, and approved exclusions. Keep historical training generations and
pre-edit source backups intact. Original-folder photos change only for records
with approved source operations.

Use saved manual face crops/boxes directly for face-only fixes. Regenerate or
revalidate face crops for sources that were edited, moved or copied; preserve
manual overrides during future exports. Retain valid unchanged crop bytes
where source version, identity and split still agree. The original-folder
collection, correction ledger and crop manifest must remain consistent. Record
source-region transforms separately from final face boxes, and retain common
capture-group provenance for samples derived from the same photograph.

Twelve explicit exclusions already have no active crop, so those twelve do not
reduce the current recognition sample count. Excluding the two active crops
leaves 9,463. Recovering all 78 remaining original-photo candidates could raise
that to at most 9,541 before unresolved-ID exclusions or other final decisions.
This is a bookkeeping bound assuming no extra individuals are added from a
photograph, not a claim that all 78 will yield useful crops.

The repository also documents a tighter master-photo version versus an older
full-frame version for one Fanta source. Tie every edit to the exact image bytes
and dimensions displayed; never apply coordinates to a different source version.

### 5. Validate and freeze evaluation eligibility

Before training, require:

- existing files match the new manifest; no stale extras or missing crops;
- valid, nonempty boxes within the selected source dimensions;
- canonical and evidence-backed labels;
- byte/RGB duplicate and contradictory-label checks after recropping;
- per-individual train/validation/test counts, especially after exclusions;
- all crops and variants of a source photograph belong to the same split;
- capture bursts/sessions are kept together for any newly designed independent
  evaluation split, while preserving identity coverage across splits.

Keep current split assignments during annotation. Do not randomly reshuffle as
part of moving labels or restoring crops. Any leakage-driven split repair should
be a separate, documented protocol change. Grouped splitting supports keeping
related observations together while stratification helps retain class balance.
The grouping unit here is a source/capture event, **not the macaque identity**:
this remains closed-set identification with identities represented across splits.
[Grouped and stratified validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

The old test errors have already informed development. Retain that cohort as a
historical diagnostic and, where feasible, reserve new capture sessions for a
final independent assessment. Human label/crop corrections should apply by the
same stated rules across splits; they should not be driven by a model being
wrong on a particular test image.

### 6. Retrain and compare

Use the same selected baseline configuration first: ±45° training augmentation
and seven-angle mean inference. Start a fresh fit on the corrected manifest,
keep the architecture and evaluation policy fixed for this first comparison,
and select checkpoints using verified validation data. Rebuild galleries from
the new checkpoint rather than retaining embeddings from the old model.

Evaluate the frozen old checkpoint and the newly trained checkpoint on the same
corrected, verified test set where the identity/protocol remains compatible.
Report the historical 85.11% result separately because the test examples and
labels may change. Report top-1, macro recall/F1, per-ID counts and difficult-image
slices. This separates changes to evaluation data from the new model comparison;
one seed alone does not establish a reliable causal gain from cleaning. Repeat
seeds if resources permit before making strong comparative claims.

Detector retraining is a separate decision. Manually overriding recognition
crops does not require it. If new boxes are later used for YOLO training, only
use detection-training-eligible source groups; never add recognition test photos
to detector training and then claim those crops as an independent pipeline test.

## CSD3 versus local correction

| Choice | Advantages | Requirements |
| --- | --- | --- |
| CSD3 storage + browser editor (recommended) | One authoritative source set, no full-dataset transfer, direct manifest/hash validation, familiar review interface. | SSH/port forwarding while editing; correction mode is available at `/corrections`. |
| Local offline package | Works without a CSD3 connection; convenient zoom and local annotation tools. | Export relevant original-folder photos plus keys and hashes; return an edit manifest and approved source images if edited externally; validate before importing on CSD3. |

For a local package, use opaque review keys as filenames and ship a mapping CSV
with the real labels and paths. Keep annotations in the same ledger format and
original-image coordinate system. Prefer returning source-region, destination-ID
and explicit face-only versus source-edit instructions so CSD3 can reproduce
the selected operation. If using an
external local image editor, return the edited source files with the same key
mapping and an explicit operation manifest; import only those approved changes,
checking hashes, dimensions and destinations rather than merging a whole folder
tree. Assemble face-only corrections directly and regenerate affected YOLO
crops only where the chosen source operation requires it.

The immediate next work is using the correction editor to resolve its proposed
queues. A new training dataset and retraining should follow the completed
correction ledger and its validated file operations.

## Reproduce the planning inventory

Run from any directory:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/summarize_crop_review_actions.py
```

The default output refuses to overwrite a different frozen review snapshot.
Use `--output` for a later revision. The tool reads saved reviews and manifests,
copies the review state, and writes planning files only; it never modifies
original photos, crop pixels, labels or split assignments.
