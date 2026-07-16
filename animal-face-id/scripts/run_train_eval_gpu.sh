#!/bin/bash
# Train a config on GPU, then run the corrected (no-margin) eval on its best checkpoint.
#
#   sbatch scripts/run_train_eval_gpu.sh <config.yaml> <tag>
#
# Example:
#   sbatch scripts/run_train_eval_gpu.sh configs/train_macaque_arcface_aug3.yaml aug3
#
#SBATCH -J macaque_train
#SBATCH -A LEMOINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ylj20@cam.ac.uk
#SBATCH -o /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j.out
#SBATCH -e /rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j.err

set -e
BASE=/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id
mkdir -p "$BASE/artifacts/ltr_logs"

CONFIG="${1:?usage: sbatch run_train_eval_gpu.sh <config.yaml> <tag>}"
TAG="${2:?usage: sbatch run_train_eval_gpu.sh <config.yaml> <tag>}"

unset PYTHONPATH
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate
export TMPDIR=/rds/user/ylj20/hpc-work/tmp
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface
export PYTHONUNBUFFERED=1

cd "$BASE"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME | config=$CONFIG | tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Resolve the checkpoint name from the config's `name:` field
NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['name'])")
CKPT="artifacts/${NAME}_best.pt"

echo "==================== 1. TRAIN ($NAME) ===================="
python -m src.training.train --config "$CONFIG"

echo "==================== 2. EVAL (corrected, no margin) ===================="
python tools/run_final_eval.py --config "$CONFIG" --ckpt "$CKPT" --device cuda --tag "$TAG"

echo "DONE. checkpoint=$CKPT | per-class CSV + summary in artifacts/final_eval/ (tag: $TAG)"
