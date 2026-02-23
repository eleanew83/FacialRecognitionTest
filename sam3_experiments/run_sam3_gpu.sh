#!/bin/bash
# Usage:
#   sbatch run_sam3_gpu.sh              # main scenario test (~9 images)
#   sbatch run_sam3_gpu.sh --benchmark  # prompt benchmark (~9 images × 9 prompts)
#
#SBATCH -J sam3_macaque
#SBATCH -A LEMOINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/logs/%j.out
#SBATCH -e /rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/logs/%j.err

mkdir -p /rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/logs

unset PYTHONPATH
source /rds/user/ylj20/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export TMPDIR=/rds/user/ylj20/hpc-work/tmp
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

BASE=/rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments

if [[ "$1" == "--benchmark" ]]; then
    echo "Running: prompt benchmark"
    python -u "$BASE/benchmark_prompts.py"
else
    echo "Running: main scenario test"
    python -u "$BASE/test_sam3_macaque.py"
fi
