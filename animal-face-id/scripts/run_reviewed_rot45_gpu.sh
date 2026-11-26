#!/bin/bash
# Reviewed data, original winning ±45-degree training and seven-view mean TTA.
#SBATCH --job-name=macaque_reviewed_rot45
#SBATCH --account=LEMOINE-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --mail-user=ylj20@cam.ac.uk
#SBATCH --output=/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j_reviewed_rot45.out
#SBATCH --error=/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id/artifacts/ltr_logs/%j_reviewed_rot45.err

set -euo pipefail
BASE=/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface
export TMPDIR=/rds/user/ylj20/hpc-work/tmp
cd "$BASE"
python tools/train_reviewed_best.py "$@"
