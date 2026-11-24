# Review every macaque crop

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
