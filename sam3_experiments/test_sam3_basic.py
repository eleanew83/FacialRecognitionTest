#!/usr/bin/env python
"""Basic SAM 3 test script to verify installation and access."""

import os
# Force CPU mode before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from PIL import Image
from pathlib import Path
import sys

print("=" * 60)
print("SAM 3 Basic Test")
print("=" * 60)

# Add sam3 to path
sam3_root = Path(__file__).parent
sys.path.insert(0, str(sam3_root))

print("\n1. Checking PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

print("\n2. Loading SAM 3 model...")
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # Force CPU mode
    device = 'cpu'
    print(f"   Using device: {device}")
    
    # Load model (will download checkpoints if needed)
    print("   Building model (this may take a while on first run)...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model)
    print("   ✓ Model loaded successfully!")
    
except Exception as e:
    import traceback
    print(f"   ✗ Failed to load model: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Make sure you requested access at: https://huggingface.co/facebook/sam3")
    print("2. Verify authentication with: huggingface-cli whoami")
    sys.exit(1)

print("\n3. Testing with a dummy image...")
try:
    # Create a small test image
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    # Set image
    inference_state = processor.set_image(dummy_image)
    print("   ✓ Image processing works!")
    
    # Test text prompt
    output = processor.set_text_prompt(state=inference_state, prompt="face")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    
    print(f"   ✓ Text prompt works!")
    print(f"   Detected {len(masks)} objects")
    
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS! SAM 3 is ready to use.")
print("=" * 60)
print("\nNext steps:")
print("1. Test on real macaque images")
print("2. Try text prompts: 'macaque', 'monkey face', 'Barbary macaque'")
print("3. Compare detection quality with YOLO baseline")
