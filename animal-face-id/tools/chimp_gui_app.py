#!/usr/bin/env python3
"""Gradio MVP for chimpanzee face identification and enrollment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import numpy as np

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.inference_core import infer_single_image, load_model_from_config
from src.inference.index_store import SimpleIndex, add_individual


DEFAULT_CONFIG = "configs/train_chimp_min10_resnet50_arc_full.yaml"
DEFAULT_CKPT = "artifacts/chimp-min10-resnet50-arcface-full_best.pt"
DEFAULT_INDEX_CANDIDATES = [
    "artifacts/index/chimp_min10_auto",
    "artifacts/index/chimp_index",
]
# Preferred save location (falls back to first candidate)
DEFAULT_INDEX_SAVE_PREFIX = DEFAULT_INDEX_CANDIDATES[0]
MODEL_PROB_THRESH = 0.5
GALLERY_SIM_THRESH = 0.75


class AppState:
    def __init__(self) -> None:
        self.model_bundle: Dict[str, Any] | None = None
        self.index: SimpleIndex | None = None
        self.last_image_path: str | None = None
        self.index_prefix: str | None = None


STATE = AppState()


def _find_existing_index() -> str | None:
    """Return a prefix for an existing index if found, checking common candidates."""
    for cand in DEFAULT_INDEX_CANDIDATES:
        emb_path = Path(cand + "_embeddings.npy")
        meta_path = Path(cand + "_meta.json")
        if emb_path.exists() and meta_path.exists():
            return cand
    # Fallback: first matching chimp_min10_auto*.npy
    auto = list(Path("artifacts/index").glob("chimp_*_auto*_embeddings.npy"))
    if auto:
        prefix = str(auto[0]).replace("_embeddings.npy", "")
        meta = Path(prefix + "_meta.json")
        if meta.exists():
            return prefix
    return None


def _load_index(prefix: str | None) -> SimpleIndex | None:
    if prefix is None:
        return None
    emb_path = Path(prefix + "_embeddings.npy")
    meta_path = Path(prefix + "_meta.json")
    if emb_path.exists() and meta_path.exists():
        try:
            return SimpleIndex.load(prefix)
        except Exception:
            return None
    return None


def init_model(config: str = DEFAULT_CONFIG, ckpt: str = DEFAULT_CKPT, device: str = "cuda") -> str:
    STATE.model_bundle = load_model_from_config(config, ckpt, device=device)
    prefix = _find_existing_index()
    STATE.index = _load_index(prefix)
    STATE.index_prefix = prefix if prefix else DEFAULT_INDEX_SAVE_PREFIX
    msg = f"Model loaded from {ckpt} on {STATE.model_bundle['device']}. "
    if STATE.index and STATE.index.size > 0:
        unique_ids = len(set(STATE.index.labels))
        msg += f"Index loaded ({STATE.index.size} embeddings, {unique_ids} IDs) from prefix: {prefix}."
    else:
        msg += "No index found; build or enroll to add a gallery."
    return msg


def identify(image, topk: int, model_thresh: float, gallery_thresh: float):
    """Identify a single uploaded face.

    Returns:
    - status string
    - model_topk: list rows [rank, id, prob] (model classifier top-k on logits/probs)
    - gallery_topk: list rows [rank, id, similarity] (gallery index cosine results)
    - open_set_msg: string warning/info about confidence
    - image_path: cached file path for reuse in Enroll
    """
    if STATE.model_bundle is None:
        return "Model not loaded yet.", [], [], "Model not loaded", None
    if image is None:
        return "Please upload an image.", [], [], "No image", None

    tmp_path = Path(".__tmp_upload.png")
    image.save(tmp_path)
    try:
        result = infer_single_image(STATE.model_bundle, str(tmp_path), topk=topk)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Model classifier top-k (classes/IDs + probabilities from logits)
    model_topk = [
        [rank, label, float(score)]
        for rank, (label, score) in enumerate(zip(result["topk_labels"], result["topk_scores"]), start=1)
    ]

    # Gallery index top-k (cosine similarity)
    if STATE.index is None or STATE.index.size == 0:
        gallery_topk = [["-", "No index loaded", "-"]]
        status = "No index loaded; showing classifier top-k only."
    else:
        matches = STATE.index.query(result["embedding"], topk=topk)
        gallery_topk = [[i + 1, m["label"], float(m["score"])] for i, m in enumerate(matches)]
        status = "Done"

    top1_prob = result["topk_scores"][0] if result["topk_scores"] else 0.0
    top1_sim = gallery_topk[0][2] if (STATE.index and STATE.index.size > 0 and gallery_topk) else None

    # Open-set decision
    open_set_flag = (top1_prob < model_thresh) or (top1_sim is not None and top1_sim < gallery_thresh)
    sim_str = f"{top1_sim:.3f}" if top1_sim is not None else "N/A"
    if open_set_flag:
        open_set_msg = (
            "⚠️ Possibly a new individual (open-set triggered)\n"
            f"Model confidence: {top1_prob:.3f} (threshold: {model_thresh:.3f})\n"
            f"Gallery similarity: {sim_str} (threshold: {gallery_thresh:.3f})"
        )
    else:
        open_set_msg = (
            "✓ Known individual (confidence above thresholds)\n"
            f"Model confidence: {top1_prob:.3f}\n"
            f"Gallery similarity: {sim_str}"
        )

    # Cache last image path for Enroll reuse
    STATE.last_image_path = str(tmp_path)

    return status, model_topk, gallery_topk, open_set_msg, str(tmp_path)


def enroll(name: str, files: List[Any], aggregate: bool):
    if STATE.model_bundle is None:
        return "Model not loaded yet.", 0
    if not name:
        return "Please provide a name/ID.", 0
    if not files:
        return "Please upload at least one image.", 0

    if STATE.index is None:
        STATE.index = SimpleIndex()

    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        else:
            # gradio returns tempfile objects with .name
            paths.append(getattr(f, "name", None))
    paths = [p for p in paths if p]

    add_individual(STATE.model_bundle, STATE.index, name, paths, aggregate=aggregate)
    save_prefix = STATE.index_prefix or DEFAULT_INDEX_SAVE_PREFIX
    STATE.index.save(save_prefix)
    return f"Added {len(paths)} images under '{name}'. Index size: {STATE.index.size}. Saved to {save_prefix}_*.npy/json", len(paths)


def build_interface():
    with gr.Blocks(title="Chimpanzee Face ID") as demo:
        status = gr.State("")

        gr.Markdown("# Chimpanzee Face Identification (MVP)\nUpload cropped face images to identify or enroll.")
        with gr.Row():
            cfg = gr.Textbox(label="Config path", value=DEFAULT_CONFIG)
            ckpt = gr.Textbox(label="Checkpoint path", value=DEFAULT_CKPT)
            device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device")
            init_btn = gr.Button("Load model")
        init_out = gr.Textbox(label="Init status")
        init_btn.click(init_model, inputs=[cfg, ckpt, device], outputs=init_out)

        with gr.Tab("Identify"):
            img = gr.Image(type="pil", label="Cropped chimp face")
            topk_in = gr.Slider(1, 5, value=5, step=1, label="Top-k")
            model_thresh_in = gr.Slider(0.0, 1.0, value=MODEL_PROB_THRESH, step=0.01, label="Model prob threshold")
            gallery_thresh_in = gr.Slider(0.0, 1.0, value=GALLERY_SIM_THRESH, step=0.01, label="Gallery sim threshold")
            identify_btn = gr.Button("Identify")
            msg = gr.Textbox(label="Status")
            open_set_box = gr.Textbox(label="Open-set status", interactive=False)
            last_img_hidden = gr.Textbox(label="Last image path", interactive=False, visible=False)
            gallery_table = gr.Dataframe(headers=["rank", "id", "similarity"], label="Gallery (index) top-k", interactive=False, datatype=["number", "str", "number"])
            clf_table = gr.Dataframe(headers=["rank", "id", "prob"], label="Model top-k", interactive=False, datatype=["number", "str", "number"])
            identify_btn.click(
                identify,
                inputs=[img, topk_in, model_thresh_in, gallery_thresh_in],
                outputs=[msg, clf_table, gallery_table, open_set_box, last_img_hidden],
            )

        with gr.Tab("Enroll"):
            name_in = gr.Textbox(label="New individual name/ID")
            files_in = gr.File(file_count="multiple", type="filepath", label="Images (cropped faces)")
            aggregate = gr.Checkbox(value=False, label="Average embeddings into one vector")
            enroll_btn = gr.Button("Add to index")
            enroll_msg = gr.Textbox(label="Enroll status")
            enroll_count = gr.Number(label="Images added", precision=0)
            # Button to reuse last identified image
            reuse_btn = gr.Button("Use last identified image")
            reuse_btn.click(lambda p: [p] if p else [], inputs=[last_img_hidden], outputs=[files_in])
            enroll_btn.click(enroll, inputs=[name_in, files_in, aggregate], outputs=[enroll_msg, enroll_count])

    return demo


def main():
    # Preload model/index on startup
    init_model()
    demo = build_interface()
    demo.launch(server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_PORT", 7860)))


if __name__ == "__main__":
    main()
