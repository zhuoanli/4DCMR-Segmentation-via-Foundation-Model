#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --job-name=dinov2-seg
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/dinov2-%j.log

set -e
echo "==== dinov2_seg  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft

mkdir -p /scratch/gautschi/li4533/MIUA_2026/results/dinov2
mkdir -p /scratch/gautschi/li4533/MIUA_2026/logs

python /scratch/gautschi/li4533/MIUA_2026/train_eval_dinov2.py \
    --epochs 50 \
    --batch  8  \
    --lr     1e-4

echo "==== dinov2_seg DONE $(date) ===="
