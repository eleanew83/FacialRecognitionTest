# SAM 3 Exploration for Macaque Face Detection

This folder contains scripts and experiments for exploring SAM 3 (Segment Anything Model 3) for macaque face detection and identification.

## Setup

1. **Clone SAM 3 repository** (in parent directory):
   ```bash
   cd /home/ylj20/FacialRecognitionTest
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   ```

2. **Install dependencies**:
   ```bash
   conda create -n sam3 python=3.12 -y
   conda activate sam3
   pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -e ".[notebooks]"
   ```

3. **Authenticate with HuggingFace**:
   ```bash
   python ../sam3_experiments/hf_login.py
   ```
   - Get token from: https://huggingface.co/settings/tokens
   - Request access: https://huggingface.co/facebook/sam3

## Scripts

- `hf_login.py`: Helper script for HuggingFace authentication
- `test_sam3_basic.py`: Basic SAM 3 functionality test (requires GPU)

## Known Limitations

- **GPU Required**: SAM 3 does not support CPU-only inference (hardcoded CUDA dependencies throughout)
- To run SAM 3, you need:
  - NVIDIA GPU
  - CUDA 12.6+
  - Access to troughton GPU nodes or cloud GPU instance

## Goals

1. Test SAM 3 zero-shot segmentation on macaque faces
2. Compare with current YOLO detection pipeline
3. Evaluate multi-face detection and human face filtering
4. Assess individual macaque identification capabilities

## Status

- [x] SAM 3 repository cloned
- [x] Environment setup documented
- [x] HuggingFace authentication completed
- [ ] GPU access obtained
- [ ] Basic inference test completed
- [ ] Multi-face detection experiments
- [ ] Comparison with YOLO baseline
