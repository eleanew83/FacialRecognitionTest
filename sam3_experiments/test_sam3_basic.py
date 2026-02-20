#!/usr/bin/env python
"""Basic SAM 3 test script to verify installation and GPU access."""

import torch
from PIL import Image
import sys

print("=" * 60)
print("SAM 3 Basic Test")
print("=" * 60)

# --- Device setup ---
print("\n1. Checking hardware...")
print(f"   PyTorch version : {torch.__version__}")
print(f"   CUDA compiled   : {torch.version.cuda}")
print(f"   CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU             : {gpu_name}")
    print(f"   GPU memory      : {gpu_mem:.1f} GB")
else:
    device = "cpu"
    print("   WARNING: No GPU found — running on CPU (very slow, for testing only)")

print(f"   Using device    : {device}")

# --- Load model ---
print("\n2. Loading SAM 3 model (downloads checkpoint on first run)...")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model)
    print("   Model loaded successfully!")

    if device == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU memory used : {mem_used:.2f} GB")

except Exception as e:
    import traceback
    print(f"   Failed to load model: {e}")
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  1. Request access at: https://huggingface.co/facebook/sam3")
    print("  2. Check auth:  python sam3_experiments/hf_login.py")
    print("  3. Check quota: mybalance")
    sys.exit(1)

# --- Test inference ---
print("\n3. Running inference on a test image...")
try:
    dummy_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    inference_state = processor.set_image(dummy_image)
    print("   Image set successfully!")

    output = processor.set_text_prompt(state=inference_state, prompt="face")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"   Text prompt works! Detected {len(masks)} objects")

    if device == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU memory after inference: {mem_used:.2f} GB")

except Exception as e:
    import traceback
    print(f"   Inference failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS! SAM 3 is ready on", device.upper())
print("=" * 60)
print("\nNext steps:")
print("  1. Test on real macaque images")
print("  2. Try prompts: 'macaque face', 'monkey', 'primate'")
print("  3. Submit full job: sbatch sam3_experiments/run_sam3_gpu.sh")
