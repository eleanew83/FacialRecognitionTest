#!/bin/bash
# Long-tail recognition retrain + before/after evaluation for macaque ReID.
#
#   sbatch scripts/run_ltr_train_eval_gpu.sh
#
# Produces a clean comparison (all with the corrected no-margin eval):
#   1. baseline (existing aug2 checkpoint)
#   2. baseline + logit adjustment        (free, no retrain)
#   3. LTR model (strong aug + class-balanced loss)
#   4. LTR model + logit adjustment
#
#SBATCH -J macaque_ltr
#SBATCH -A LEMOINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ylj20@cam.ac.uk
#SBATCH -o /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j.out
#SBATCH -e /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j.err

set -e
BASE=/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id
mkdir -p "$BASE/artifacts/ltr_logs"

unset PYTHONPATH
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate
export TMPDIR=/rds/user/ylj20/hpc-work/tmp
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface
export PYTHONUNBUFFERED=1

cd "$BASE"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME | python=$(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

BASELINE_CKPT=artifacts/macaque-resnet50-arcface_aug2_best.pt
LTR_CFG=configs/train_macaque_arcface_ltr.yaml
BASE_CFG=configs/train_macaque_arcface_aug2.yaml
LTR_CKPT=artifacts/macaque-resnet50-arcface_ltr_best.pt
TAU=1.0

echo "==================== 1. TRAIN LTR MODEL ===================="
python -m src.training.train --config "$LTR_CFG"

echo "==================== 2. EVAL (corrected, no margin) ===================="
echo "--- baseline ---"
python tools/run_final_eval.py --config "$BASE_CFG" --ckpt "$BASELINE_CKPT" --device cuda --tag baseline
echo "--- baseline + logit-adjust ---"
python tools/run_final_eval.py --config "$BASE_CFG" --ckpt "$BASELINE_CKPT" --device cuda \
    --logit-adjust-tau "$TAU" --tag baseline_logitadj
echo "--- LTR model ---"
python tools/run_final_eval.py --config "$LTR_CFG" --ckpt "$LTR_CKPT" --device cuda --tag ltr
echo "--- LTR model + logit-adjust ---"
python tools/run_final_eval.py --config "$LTR_CFG" --ckpt "$LTR_CKPT" --device cuda \
    --logit-adjust-tau "$TAU" --tag ltr_logitadj

echo "ALL DONE. Per-class CSVs + summaries in artifacts/final_eval/ (stems: *_baseline, *_ltr, *_logitadj)"
echo "Compare with: tools/longtail_analysis.py (re-point PER_CLASS to the *_ltr CSV)"
