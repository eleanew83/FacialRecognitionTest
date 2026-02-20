#!/bin/bash
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

module load python/3.11.0-icl
source /rds/user/ylj20/hpc-work/venvs/macaque/bin/activate

export TMPDIR=/rds/user/ylj20/hpc-work/tmp
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo ""

python /rds/user/ylj20/hpc-work/FacialRecognitionTest/sam3_experiments/test_sam3_macaque.py
