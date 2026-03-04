#!/bin/bash
# Run PrimateReID zero-shot baseline on macaque test crops (all 4 backbones).
#
# Usage:
#   sbatch run_baseline_gpu.sh                  # all 4 backbones
#   sbatch run_baseline_gpu.sh --backbone dinov2  # single backbone
#
#SBATCH -J primateid_baseline
#SBATCH -A LEMOINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /rds/user/ylj20/hpc-work/FacialRecognitionTest/PrimateReID_experiments/logs/%j.out
#SBATCH -e /rds/user/ylj20/hpc-work/FacialRecognitionTest/PrimateReID_experiments/logs/%j.err

mkdir -p /rds/user/ylj20/hpc-work/FacialRecognitionTest/PrimateReID_experiments/logs

unset PYTHONPATH
source /rds/user/ylj20/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export TMPDIR=/rds/user/ylj20/hpc-work/tmp
export HF_HOME=/rds/user/ylj20/hpc-work/.cache/huggingface

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo ""

BASE=/rds/user/ylj20/hpc-work/FacialRecognitionTest/PrimateReID
RESULTS_DIR=/rds/user/ylj20/hpc-work/FacialRecognitionTest/PrimateReID_experiments/results
CROPS=/rds/user/ylj20/hpc-work/FacialRecognitionTest/yolo_detection/yolo_detection_code/output/macaque_crops/test

cd "$BASE"

# If a specific backbone is requested, run just that one
if [[ "$1" == "--backbone" && -n "$2" ]]; then
    echo "Running single backbone: $2"
    PYTHONPATH=src python -u -m primateid.run \
        --crops "$CROPS" \
        --backbone "$2" \
        --device cuda \
        --output "$RESULTS_DIR/$2_$(date +%Y%m%d_%H%M%S)"
else
    # Run all 4 backbones sequentially
    for BACKBONE in resnet50 arcface dinov2 facenet; do
        echo ""
        echo "========================================"
        echo "Running backbone: $BACKBONE"
        echo "========================================"
        PYTHONPATH=src python -u -m primateid.run \
            --crops "$CROPS" \
            --backbone "$BACKBONE" \
            --device cuda \
            --output "$RESULTS_DIR/${BACKBONE}_$(date +%Y%m%d_%H%M%S)"
    done

    echo ""
    echo "========================================"
    echo "ALL BACKBONES DONE — results summary:"
    echo "========================================"
    for f in "$RESULTS_DIR"/*/summary.json; do
        backbone=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('backbone','?'))" 2>/dev/null || echo "?")
        auc=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('auc',0):.4f}\")" 2>/dev/null || echo "?")
        eer=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('eer_pct',0):.1f}\")" 2>/dev/null || echo "?")
        d=$(cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('decidability',0):.3f}\")" 2>/dev/null || echo "?")
        echo "  $backbone: AUC=$auc  EER=$eer%  d'=$d"
    done
fi
