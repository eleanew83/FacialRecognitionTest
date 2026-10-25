#!/bin/bash
# Array of three independent model fits. Submit through submit_aggressive_suite.sh.
#SBATCH --job-name=macaque_aggressive
#SBATCH --account=LEMOINE-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2

set -euo pipefail
# Slurm copies batch scripts into its spool; the submitter passes the real base.
BASE="${2:?Pass the suite directory and project base}"
RUN_DIR="${1:?Pass the suite directory and project base}"
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface
source "$BASE/scripts/setup_job_tmp.sh"
cd "$BASE"
PROFILES=(rot30 rot100 rot150)
PROFILE="${PROFILES[$SLURM_ARRAY_TASK_ID]}"
CONFIG="$RUN_DIR/configs/train_macaque_arcface_aggressive_${PROFILE}.yaml"
echo "Job ${SLURM_JOB_ID}: $PROFILE on ${SLURMD_NODENAME} | config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for this job"'
python -m src.training.train --config "$CONFIG"
