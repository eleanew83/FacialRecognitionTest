#!/bin/bash
# Fourth job: compare all trained checkpoints with seven-angle probability pooling.
#SBATCH --job-name=macaque_rotation_tta
#SBATCH --account=LEMOINE-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
BASE="${2:?Pass the suite directory and project base}"
RUN_DIR="${1:?Pass the suite directory and project base}"
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/rds/user/ylj20/hpc-work/tmp
cd "$BASE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for this job"'
python tools/compare_rotation_tta.py \
    --configs "$RUN_DIR"/configs/*.yaml \
    --out-dir "$RUN_DIR/eval" --device cuda --batch-size 32
