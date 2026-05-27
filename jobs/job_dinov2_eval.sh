#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --job-name=dinov2-eval
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/dinov2_eval-%j.log

set -e
echo "==== dinov2_eval  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft

python /scratch/gautschi/li4533/MIUA_2026/train_eval_dinov2.py --eval_only

echo "==== dinov2_eval DONE $(date) ===="
