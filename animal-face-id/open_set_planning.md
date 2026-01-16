  # Open-Set Planning (Identify → Enroll)

  ## 1) Current Pipeline (high-level)

  - Model/Config: ResNet50 + ArcFace, min10 config/checkpoint.
  - Inference core: src/inference/inference_core.py loads model + transforms; returns embedding + classifier logits/
    probs.
  - Index/galleries: src/inference/index_store.py provides cosine index; tools/build_chimp_index_from_annotations.py
    builds an index from annotations/splits.
  - GUI: tools/chimp_gui_app.py loads model, auto-loads an index if present, Identify tab runs classifier top-k and
    gallery top-k, Enroll tab adds new individuals to index and saves.
  - Final eval: tools/run_final_eval.py (not directly relevant for UX).

  ## 2) Goal for Open-Set UX (high-level)

  - After Identify, show both:
      - Model top-1 probability (classifier confidence).
      - Gallery/index top-1 similarity (cosine).
  - If either is below a threshold, flag: “⚠️ 可能為新個體”.
  - Provide a one-click action to send the just-processed face to Enroll (reuse cached image; no re-upload).

  ## 3) Proposed Open-Set Integration (clean, minimal)

  - Add a small “decision layer” in the Identify callback:
      - Inputs: model top-1 prob, index top-1 similarity.
      - Configurable thresholds: e.g., model_prob_thresh, gallery_sim_thresh.
      - Outcome: status flag (known/unknown/ambiguous), message string.
  - Expose thresholds in GUI settings (defaults loaded from config-like constants).
  - When flagged as “possible new individual”, show:
      - The warning badge/message.
      - A button to “Send to Enroll” that passes along the cached image path/bytes to the Enroll tab.

  ## 4) GUI data flow changes

  - Identify callback currently returns status + model top-k rows + gallery top-k rows.
  - Extend return payload to include:
      - top1_prob (classifier), top1_sim (gallery), and an open-set flag message.
      - A hidden/cached reference to the uploaded image (temp file path or in-memory bytes) for reuse.
  - UI changes:
      - Add threshold controls (sliders/textboxes) with defaults.
      - Display an “Open-set warning” text/Markdown below the results.
      - Add a button “Send to Enroll” that writes the cached image reference into a shared gr.State, then switches tab
        or pre-fills Enroll with the image list.

  ## 5) Modules/files impacted

  - tools/chimp_gui_app.py: main changes (UI + Identify callback output formatting + state passing to Enroll).
  - src/inference/index_store.py: no logic change; reuse existing similarity scores.
  - Optional: a small helper in src/utils for open-set decision logic (pure function).
  - No changes to training, inference core, index building.

  ## 6) Data flow specifics

  - Identify:
      1. Get embedding, model probs, gallery sims.
      2. Compute open-set flag: flag = (top1_prob < p_thresh) or (top1_sim < s_thresh).
      3. Return model top-k table, gallery top-k table, flag_message, top1_prob, top1_sim, and a cached image reference.
  - Enroll:
      - Accept an optional image reference from Identify. If present, pre-populate the upload list (or show a thumbnail)
        and allow enrolling with a name.

  ## 7) Thresholds & UX

  - Defaults (tunable):
      - model_prob_thresh: e.g., 0.35–0.45 (prob based on softmax).
      - gallery_sim_thresh: e.g., 0.45–0.55 (cosine similarity).
  - Expose in GUI so researchers can adjust.
  - Show the actual scores next to the warning to aid manual decisions.

  ## 8) Logging / future extensibility

  - Add lightweight logging of open-set decisions (image path, scores, thresholds, final decision) to a CSV in
    artifacts/open_set_logs.csv.
  - Future: per-ID thresholds, dynamic calibration, or distance-to-centroid instead of single-vector cosine.
  - Future: “buffer” tab to hold flagged samples before enrollment.

  ## 9) Safety / non-intrusive changes

  - Do not alter training, index-building, or final eval.
  - Keep Identify/Enroll functional even without an index (warn that open-set uses gallery sim; if no index, fall back
    to classifier-only and flag based on model prob alone).

  ## 10) Summary of additions (no code yet)

  - Add open-set decision helper (pure function) with thresholds.
  - Update GUI Identify callback to compute and display open-set flag, scores, and pass image reference to Enroll.
  - Add GUI controls for thresholds and a “send to enroll” action.
  - Optional log CSV for open-set events.

  End of plan.