# SAM 3 Exploration for Macaque Face Detection

This folder contains scripts and experiments for exploring SAM 3 (Segment Anything Model 3) for macaque face detection and identification.

## Setup

1. **SAM 3 repository** is cloned at:
   ```
   /home/ylj20/FacialRecognitionTest/sam3
   ```

2. **Python virtual environment** (`macaque`) is at `~/venvs/macaque`.
   Activate it with:
   ```bash
   module load python/3.11.0-icl
   source ~/venvs/macaque/bin/activate
   ```
   SAM 3 is installed as an editable package (`pip install -e .`) with pip
   cache and tmp on hpc-work to avoid home quota issues.

3. **Authenticate with HuggingFace** (run once):
   ```bash
   module load python/3.11.0-icl
   source ~/venvs/macaque/bin/activate
   python /rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/hf_login.py
   ```
   - Get token from: https://huggingface.co/settings/tokens
   - Request access: https://huggingface.co/facebook/sam3

## Scripts

- `hf_login.py`: Helper script for HuggingFace authentication
- `test_sam3_basic.py`: Basic SAM 3 functionality test (requires GPU)
- `test_sam3_macaque.py`: Basic macaque facial recognition test with 3 images each (requires GPU)

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

- [x] SAM 3 repository cloned to `/home/ylj20/FacialRecognitionTest/sam3`
- [x] `macaque` venv created at `~/venvs/macaque` (Python 3.11)
- [x] SAM 3 installed in venv (`pip install -e .`)
- [x] HuggingFace authentication completed
- [ ] GPU access obtained
- [ ] Basic inference test completed
- [ ] Multi-face detection experiments
- [ ] Comparison with YOLO baseline
