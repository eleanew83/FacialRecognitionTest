# Review every macaque crop

The completed review was assembled on **25 November 2026** into
[`data/macaque_faces/reviewed_2026-11-25/`](../data/macaque_faces/reviewed_2026-11-25/): **9,583 originals** and
**9,574 face crops across 149 identities**.
Use [`current_dataset.json`](../data/macaque_faces/current_dataset.json) for the
current folder paths, and the generation's `summary.json` and `audit/` for
verification and provenance. Active training configs point to this generation.
The review grids and correction workspaces below remain the historical decision
records. Infant-only Goblin cases were retained by policy; the two selected
Goblin removals were applied with backups. All 74 current organised-original
face crops were complete when this generation was built.


The review grid covers **9,555 images across 149 individuals**: all **9,465
active recognition crops**, plus **90 original photographs without active
crops**. The crop manifest contains 6,665 training, 1,302 validation and 1,498
test crops. The grid records manual decisions without editing images or labels.
The prepared grid is in `reports/crop_review_2026-11-10/`.

## What this grid covers

The grid contains all 9,465 active recognition crops, including bad or
wrong-animal crops that have not yet been corrected. Flagging a crop records a
decision; it does not remove it, change its label or regenerate it.

The cleaned source collection contains **9,555 photographs**. A check on
12 November 2026 found **90 source photographs with no active crop** (50 training,
22 validation, 18 test). These photographs are now included in the same grid,
using their existing source-folder identities. Their absence from the crop
dataset does not establish why no crop exists. The standalone list is in
[`source_photos_without_active_crop.csv`](../reports/crop_review_2026-11-10/source_photos_without_active_crop.csv).

Click **Review 90 originals without crops** to see them together across all
individuals. Or use **Images → Originals without crops** and select an
individual. Each has an orange **Assigned folder identity** header and a
**No active crop · original photo** badge. Click it to enlarge the original
and use the full-image link for browser zoom. You can still mark a suspected
wrong identity, ask Sylvain, add notes, or select **No usable face / cannot
crop**. For an original, **Reviewed — no obvious identity issue** records your
identity review; it does not imply that the photo has a usable face crop.

The overall total is now 9,555. Crop and uncropped-photo progress are also shown
separately. Existing crop decisions and backup compatibility are preserved.
Select **Images → Existing crops** to continue reviewing only the original
9,465 crop entries.

For flagged existing crops, inspect the source to decide whether to recover a
better crop of the intended individual, confirm an identity correction, or
exclude an unusable example with a recorded reason. A sideways face, difficult
lighting or blur can still be a valid example; such flags are review candidates,
not automatic exclusion decisions. For a source with no crop, decide whether a
usable target face can be recovered at all. Preserve current review progress
and apply confirmed changes later to a new dataset generation.

## Open the grid

Run this single line on the HPC machine, from any directory. Copy only the
command; do not add `bash` or a backslash:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py serve
```

The terminal prints the actual address, for example `Open http://localhost:8765`.
In VS Code Remote SSH, open the **Ports** panel and forward the port printed
there, then use the forwarded address in your normal browser.
If VS Code assigns a different local port, use that address.
Alternatively run `ssh -L 8765:localhost:8765 YOUR_HPC_SSH_ALIAS` on your laptop
and keep that SSH connection open. The server must run on the machine reached
by the tunnel. No GPU or scheduler job is needed for the review server; follow
your HPC site's rules for long-running login-node processes.

Keep the server terminal running during review. Run the same command again to
restart it or resume another day. If an earlier crop-review server is using
port 8765, the script automatically stops it and starts the new server on the
same port. It reloads your saved decisions after the previous server stops.

Only a server running this exact script under your own user account is stopped.
If a different application or another user's process occupies the port, the
script leaves it running and automatically tries the next port. For example,
it can skip occupied port 8765 and replace your old review server on port 8766.
Use the port in the printed `Open http://localhost:...` address for forwarding.
You can also choose a different starting port explicitly:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py serve --port 8766
```

Forward **8766** in VS Code's Ports panel and open <http://localhost:8766>.
Use only one review server and one review tab at a time. Changing the port
does not change the review folder or saved decisions.

Opening `reports/crop_review_2026-11-10/index.html` directly is also possible if
the review folder is on your computer. In this mode only browser storage is
available; original-photo links require access to the same filesystem paths.
The server is recommended for an HPC dataset.

## Review process

1. Start with the first individual. Browse **all splits**, 24 crops per page.
   Comparing the same animal across multiple photographs makes outliers easier
   to spot. Group labels and photographs are existing assignments, not trusted
   reference annotations.
2. Enlarge any suspicious crop. Compare it with the source photograph: is there
   a usable face, is it the intended animal, and is the identity plausible?
   Open the source in a separate browser tab for full-resolution zoom.
3. Select one primary problem category and describe additional issues in notes.
   Add a suggested identity only when you have evidence. Use **Unsure identity /
   ask Sylvain** instead of guessing.
4. Inspect every visible crop, then select **Mark unreviewed crops on this page
   OK** for the remaining ones. This preserves flags already entered. Move to
   the next page; pages do not reshuffle while you mark them.
5. Use **Next individual with unreviewed crops** when finished with an individual.
   The progress counters show coverage. Reviewing all crops includes correct
   model predictions and all training images. No predictions are shown in the
   grid, to help you judge the images independently.
6. At the end of each session, **Download backup JSON**. Export **flagged CSV**
   for detailed source checks and questions for Sylvain. Export **all reviews
   CSV** for the complete audit, including crops still unreviewed.

| Decision | Meaning |
| --- | --- |
| Unreviewed | You have not made a decision. Viewing a page does not mark it. |
| OK — no obvious issue | A usable face with no obvious crop or identity problem on visual review; this is not expert identity certification. |
| Wrong animal selected | Another animal was cropped from the source; recover the intended animal's face. |
| Wrong identity label | The visible face appears assigned to the wrong individual; record the evidence and proposed identity. |
| Unsure identity / ask Sylvain | Identity, historical name or target animal needs expert confirmation. |
| Bad crop / no usable face | No recognizable face, severe truncation or another crop failure. |
| Cannot open image | The image cannot be read or accessed. |
| No usable face / cannot crop | For an original without a crop: you reviewed it and cannot recover a usable target face. |

Blur, profile views, shadows and occlusion alone are not reasons to discard an
otherwise valid image. Keep realistic hard examples. This tool flags decisions;
apply confirmed corrections later to a new, traceable dataset version. Do not
automatically change identities to agree with model predictions.

## Correction editor

Use the same server command, then click **Open correction editor** above the
grid or the correction link inside an enlarged image. The direct address is
<http://localhost:8766/corrections> when your forwarded port is 8766. The grid's
**review decisions remain a separate completed audit**.

1. Choose a queue: flagged review records, uncertain filenames, saved correction
   drafts, applied corrections, or all images. Filter by individual or filename.
   Your original review note and suggested ID are visible beside each image.
2. Choose the source version. **Original folder photo** opens the current photo
   in the master identity folder. **Preserved full source before face cropping**
   opens the source-split version, which can contain more context than an already
   manually cropped master photo. Paths, dimensions and hashes are shown.
3. Choose an operation from the table below and confirm its target identity.
4. For crop operations, drag a rectangle on the full source. Move it by dragging
   inside; resize it by dragging a corner. Pixel-coordinate inputs allow precise
   adjustment. Use Zoom, Clear rectangle, or Undo rectangle as needed.
5. Drafts autosave to the correction workspace. **Preview changes** shows the
   output image(s), destination paths and backup directory. Only **Apply this
   correction** performs the displayed file operations.
6. **Undo last applied correction for this image** restores the last operation
   for that record, including original-folder moves/crops. Undo refuses to
   overwrite a file changed outside the editor after the operation.

| Operation | What Apply does |
| --- | --- |
| Recrop face only | Writes a new face crop from the selected rectangle. Original-folder photo remains unchanged. |
| Relabel face crop only | Keeps existing face bytes and writes them under the corrected ID in the correction outputs. Original stays in its folder. |
| Relabel crop + move original | Writes the corrected-ID face output and moves the original into that individual's registered source folder. |
| Relabel crop + copy original | Writes the corrected-ID face output and creates an original-photo copy under that individual, retaining the old-folder original. |
| Crop original-folder photo | Replaces the actual folder photo with the selected region, after preserving its pre-edit bytes. |
| Move original / Copy original | Moves or copies the source photo to the target folder. A rectangle optionally isolates the intended individual as part of that operation. |
| Keep / Exclude / Defer | Records the disposition for the corrected dataset; preserved source and historical crop files remain available. |

Relabel operations use the existing/current face crop. For a photograph with no
face crop, first use **Recrop face only** to create one. To retain more than one
individual from a photograph, choose **Keep existing face too** when adding a
face crop. All derived samples retain their common source-group identifier.

**Original-folder operations really change the master-folder files when Apply
is clicked.** Each operation preserves old bytes, handles destination-name
collisions, and uses atomic replacements rather than writing through existing
hard links. For a combined relabel and source operation, the preview shows both
the face output and the source-photo output.

Use your own CSD3 SSH/VS Code connection. Because the login node is shared, the
editor checks that the forwarded connection belongs to the same user as the
server before accepting correction requests.

Files are saved below `reports/crop_review_2026-11-10/corrections/`:

- `state.json`: correction drafts, applied histories and revision checks.
- `faces/`: corrected face outputs, arranged by transaction, split and target ID.
- `backups/`: pre-edit bytes needed to undo changes to original-folder files.
- `transactions/`: write-ahead journals and applied operation provenance.
- `previews/`: generated previews and source/target checks for a proposed apply.

**Download correction backup** exports draft decisions, applied effects and
transaction metadata. It does not contain the image/backup bytes: retain the
whole correction directory on CSD3 as well. The folder is already ignored by Git.

Historical active face crops and `data/macaque_faces/splits.json` are not
overwritten by editor actions. After the correction pass, assemble a new training
generation from the approved corrections, exclusions and source changes before
retraining. An original-folder edit marks its face output as needing verification;
recrop the face or verify its replacement before that final assembly. See the
[correction and retraining plan](../reports/markdown/CORRECTION_PLAN_2026-11-24.md).

## Whole-monkey original-photo editor

This is a separate page for fixing the actual identity-folder photographs. The
initial queue contains **48 frames across 12 sequences**: the 17 photographs
behind the 18 deferred review entries, plus **31 additional numbered frames**.
It includes all eight Wally/Pom Canon RAW frames, developed at full resolution
with the installed `dcraw` tool. Identical filenames across identity folders and
RAW/JPEG versions of a frame appear together. Related frames are candidates for
inspection, not automatically incorrect images.

The Goblin visual audit added **58 further photographs**, bringing Stage 1 to
**106 photographs**. Its complete catalog contains 65 Goblin camera-photo groups
representing 67 audited files. Following the user's policy, the default Show
filter is now **Goblin · multiple individuals (excluding infants)**: it shows
the **33 groups** flagged with another adult or juvenile. Infant-only and possible
occluded-infant cases do not appear in that view and were retained unchanged by
policy. The complete catalog remains available through **All photographs**.
Changing Show resets Photo sequence to All sequences; sequence choices include
only the selected view's photographs.

All non-infant Goblin cases were reviewed before publishing the generation above.
The two explicit old-copy retirements were applied with transaction backups.
An Edit batch can retire selected old copies without creating a new output crop;
Keep and Defer still record decisions without changing files.

The additions are recorded in `original_photo_edits/goblin_queue_extension.json`;
an exact pre-extension catalog backup is in `catalog_backups/`. The repeatable
`source-add-goblin` command appends missing candidates from the visual-audit CSV
without rebuilding existing records. Updated servers reload an extended catalog
automatically. Servers started before that capability was added need restarting,
or use the updated server on port 8769 at `/originals` (and `/organized-faces`
for Stage 2 on the same port).

The queue groups names ending in short frame counters, for example `Sylv_1`,
`Sylv2`, `Martin3` and `EL_J4`. It preserves numeric timestamp filenames so it
does not accidentally group unrelated photographs. The archive was also checked
for frames within 60 seconds of the timestamp-named Tack photograph; no additional
frames were found in that interval.

Start the separate server with this **single-line command**, from any directory:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py source-serve
```

Forward port **8767** in VS Code and open **http://localhost:8767/originals**.
If occupied, the command tries subsequent ports and prints the actual URL. It
does not stop or replace the existing review/correction servers.

1. Choose a photo sequence or inspect the unfinished queue. The source defaults
   to the widest available photograph, favouring the untouched archive when
   dimensions are equal. The archive is `Gibraltar_Macaques_Photos`; the active
   folder collection being corrected is `Gibraltar_Macaques_Photos_Cleaned_Updated`.
2. Draw around each **whole visible monkey**, using **Draw another monkey** for
   additional outputs. Select a rectangle to assign its identity, move it, resize
   it or adjust original-pixel coordinates. Use **Full resolution** for closer
   inspection; crop exports always use full-resolution source pixels.
3. For each output choose **Replace existing** or **Add a new image** in that
   identity's registered folder. A single existing copy for the identity is
   selected as its replacement by default; check that choice. New images get
   distinct names that include the target identity.
4. Inspect the existing folder copies below the canvas. Explicitly select
   **Retire this old folder copy** for a wrong assignment that should disappear.
   Replacing a file and retiring it are mutually exclusive. A new output alone
   does not remove a wrong old-folder copy.
5. Click **Preview entire photo batch**. Check every crop, identity, destination
   path and retirement, then **Apply changes to original folders**. Every output
   comes from the same untouched source, so creating the first crop does not
   destroy the source for the next individual.
6. For related photos that need no changes click **No changes needed — next photo**.
   This records the photo as checked, keeps its files and advances through the
   unfinished queue. Find those decisions later using the **No changes needed**
   filter. Alternatively select **No changes needed — keep current originals**
   in the operation menu and preview/record it. For uncertain cases choose
   **Defer** and add notes, then preview and record the decision. Boxes/removal selections are retained as draft context but do not
   change files for Keep/Defer.

Drafts autosave in the separate workspace. **Download workspace backup** exports
the source catalog, drafts and applied transaction metadata. Image/backup bytes
stay on CSD3. Use one editing tab. To change an already applied photo, first use
**Undo this photo's applied batch**, then adjust and preview the draft again.
Undo refuses to overwrite later external file edits.

The workspace is `reports/crop_review_2026-11-10/original_photo_edits/`:

- `catalog.json` and `expanded_photo_queue.csv`: grouped source versions, seed
  notes, numbered frames and existing identity-folder copies.
- `state.json`: separate draft decisions and applied photo histories.
- `developed_raw/` and `image_cache/`: full-resolution RAW development and fast
  browser views. The camera originals remain untouched in the archive.
- `previews/`, `transactions/`, `backups/`: staged output pixels, write-ahead
  journals, provenance and pre-edit bytes for rollback/undo.

The initial queue has already been built. For a fresh workspace it can be built
with `source-build` (optional `--archive /absolute/archive/path`). Rebuilding a
workspace with saved drafts or applied work is refused to preserve its catalog.

This page updates the active original folders. It records which source/frame
each output came from and that the face dataset needs verification/rebuilding;
it does not rewrite the completed review or the existing face-correction state.
Before retraining, consume this workspace alongside the face corrections,
regenerate/verify faces for affected originals, apply exclusions and keep
derivatives of the same photo together when checking train/validation/test splits.

## Stage 2: crop faces after organising originals

The new face workflow uses only **new or replaced originals from applied Edit
batches** in the whole-monkey editor. **No changes needed** photographs and
unchanged copies retained within an edited batch do not enter this queue: their
earlier approved face crops remain in use. Original-photo drafts and deferred
source decisions are also excluded until their organisation is complete.

Start the face-stage server with this single-line command:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py faces-serve
```

Forward port **8768** and open **http://localhost:8768/organized-faces**. If that
port is occupied, use the actual port printed by the command. Existing review
and original-photo servers are left running. Updated servers also expose Stage 1
at `/originals`, so its link and Stage 2 can be used on the same server.

1. Finish and apply the original-folder changes in Stage 1. Each new/replaced
   individual image becomes a separate face-stage item under its assigned ID.
2. Open Stage 2, or click **Reload newly organised originals** if it is already
   open. The default queue shows unfinished face crops for changed originals.
3. Check the displayed identity, draw a tight rectangle around that individual's
   face, and use **Preview changes → Save face crop / decision**. Zoom and the
   original-pixel coordinate controls work as in the earlier face editor. The
   identity is fixed to the organised folder; change its assignment in Stage 1
   if necessary. The whole-monkey original is preserved.
4. Choose **No usable face** to exclude that face output, or **Defer** when
   identification/face suitability needs another look. Deferred faces remain
   unfinished and are available in the Deferred faces filter. Applied face crops
   and exclusions can be undone in this stage.
5. Use **Export approved faces + decisions** for the JSON inventory of approved
   outputs, exclusions and pending decisions. Face images remain on CSD3 under
   `original_photo_edits/face_stage/corrections/faces/`; drafts and transaction
   histories are in that workspace's `state.json` and `transactions/`.

Every output records its organised original's hash, parent photo group and
original-organisation transaction. Undoing/reapplying Stage 1 creates a new
version: faces from an inactive version are omitted from export. External source
changes or modified face-output files also prevent their export as approved.
Reload the page after changing the original organisation.

This stage preserves the earlier review and face-correction workspace. For
retraining, assemble the new faces together with approved unchanged earlier faces
and exclusions, consuming the original-photo transactions to replace superseded
examples. Keep derivatives of the same camera photograph in one split. Exported
split hints (including `unassigned` for genuinely new photos) are provisional;
this stage does not itself rebuild the final training manifest or retrain a model.

## Offline correction editor (no CSD3 connection needed while editing)

The portable editor runs by opening a local HTML file on your laptop. It includes
full-resolution source photographs, existing face crops, identity labels and your
review notes. It supports the same face-crop, relabel, original-photo crop/move/copy,
exclude, keep and defer operations. No Python, web server or internet is required
on the laptop. The online review and correction pages remain separate and available.

1. Download the generated `offline_corrections_*.zip` from
   `reports/crop_review_2026-11-10/` using VS Code Explorer → right-click → **Download**.
2. Extract the **whole ZIP** on your laptop. Keep the `assets` folder beside
   `index.html`, then open `index.html` in Chrome, Edge or Firefox. You can now
   disconnect from CSD3. Do not try to edit inside the compressed ZIP.
3. Draw a rectangle, choose the operation/identity, click **Preview changes**, then
   **Save offline correction**. A saved draft alone is unfinished work; it does
   not confirm an operation for application to the dataset.
4. Click **Download results** after every session, and before closing the page.
   This creates a small JSON file containing your boxes, notes, identities and
   confirmed operations. Keep the latest download. Browser autosave is convenient
   but browser storage can be cleared or unavailable, especially for local files.
5. To resume, reopen the same extracted package. If progress is missing, use
   **Load results JSON** to restore your latest download. Use one active tab.

The default package includes the **220 flagged images**. The default unfinished
queue skips corrections already applied when the package was exported, and images
you have saved offline. Use **Flagged in completed review** or **All images** to
revisit them. Undo removes the last offline operation for that image; it does not
undo existing CSD3 corrections.

Photos in the offline package are immutable. Original-photo edits/moves/copies
are saved as instructions, not performed on your laptop. After queueing an original
crop or move, you can also queue a face crop: draw its box using the displayed
**export-time original**, which remains the coordinate reference. The importer
preserves those original bytes so later boxes remain correct after a source edit.
To inspect the actual resulting source folder, import the batch, then open the
online editor or export a fresh package.

To generate another package on CSD3, run this **single-line command**:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py offline-export
```

Use `--scope work` to include flagged images plus uncertain-filename candidates,
or `--scope all` for the entire database. Each export creates a new package ID.
Results must be restored into the same package they came from. The CSD3 package
manifest is retained under `corrections/offline_packages/`; do not delete it.

When ready to apply the corrections, upload the latest results JSON to CSD3.
First check it (replace the results path with the uploaded file):

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py offline-import /absolute/path/to/macaque_results.json
```

This checks the dataset, exported source hashes, identity choices, rectangles and
whether online corrections have changed. It reports confirmed operations and
unfinished drafts without changing photographs or correction state. To actually
perform the confirmed corrections, use the same command with `--apply`:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py offline-import /absolute/path/to/macaque_results.json --apply
```

Import uses the existing backups, transaction journals and undo mechanism.
Already imported operations are skipped when retrying or importing a later
download with additional operations. An interrupted batch can be rerun; earlier
committed operations stay applied. An unexpected change stops the batch, so check
the error before retrying. Do not edit the same images in the online editor while
using/importing an offline package. Downloaded results and a copy are retained in
`corrections/offline_imports/` after successful application. Unconfirmed drafts
remain in your results file for resuming offline; they are not applied.

Offline correction does not rebuild the training manifest. After all changes,
assemble/validate the corrected dataset and its splits before retraining.

## Progress and backups

While the server is running, decisions save atomically to
`reports/crop_review_2026-11-10/review_state.json`. Browser storage protects
pending changes if a save fails. The status banner distinguishes disk saves
from browser-only saves. Download a backup if it reports a problem.

Use one active review tab. Conflicting edits from another tab stop further
disk saves until you export a backup and reload. **Undo last review action**
undoes the last crop decision or batch of OK marks in the current session.
Backup JSON can be restored/merged for the same dataset. Imported decisions
replace existing decisions for crops included in the backup; download the
current backup first if both contain useful work.

Review keys are tied to the manifest and identity-registry fingerprints. The
builder verifies existing crop SHA-256 values before producing thumbnails.
Missing or unreadable crops remain visible with an error instead of being
silently omitted. Original source photographs are linked but not identity
verified. Starting the server checks that the manifest and registry still
match the inventory. Rebuild into a new directory after dataset changes.

## Time to allow

These are planning assumptions, not measured review times:

| Work | Assumption | Approximate time |
| --- | --- | --- |
| Visual pass over all 9,465 crops | 4–6 seconds per crop | 10.5–15.8 hours of viewing |
| Navigation, short notes and breaks | Add roughly 15–25% | About 12–20 hours total for the pass |
| Detailed follow-up | If 2–5% need 1 minute each | About 3–8 additional hours |

Plan **15–28 hours overall**, spread over short sessions. More ambiguous
identities or frequent source-photo inspection will increase this. Expert
confirmation and actually correcting/re-exporting the data are additional.

Time your first **200 crops**. Estimate remaining first-pass time as:

```text
minutes for first 200 × (9,465 − 200) / 200
```

For example, 15 minutes per 200 crops means about 11.8 hours for the entire
visual pass at that pace, before additional source checks and breaks. Use a
30–45 minute session length if attention drops; avoiding missed mistakes matters
more than finishing the grid quickly.

## Rebuild or use another dataset

Pillow is required to build thumbnails and use the correction editor. Serving
the original review grid alone uses Python's standard library. Run from the
repository root:

```bash
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python \
  animal-face-id/tools/review_macaque_crops.py build
```

Optional build flags are `--manifest`, `--registry`, `--crops`, `--workers`, and
`--output`. Choose a new `--output` for a changed manifest or registry; the tool
refuses to overwrite a review belonging to another dataset. Rebuilding the same
inventory preserves `review_state.json`. Pass that directory to `serve --output`.

The generated `build_summary.json` records coverage and missing/unreadable
images. The UI template and generator live together under `tools/`.

To add the no-crop originals to a newly built review, use this single-line
command. It does not rebuild existing crop thumbnails or change review decisions:

```
/rds/user/ylj20/hpc-work/venvs/macaque/bin/python /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/tools/review_macaque_crops.py add-sources
```

Use `--source-manifest` and `--output` for another dataset. Once added, the source
manifest is remembered and subsequent `build` commands keep these originals.
Restart the review server after updating its inventory.
