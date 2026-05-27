#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --job-name=figures
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/figures-%j.log

set -e
echo "==== figures  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft

python /scratch/gautschi/li4533/MIUA_2026/evaluate_and_figures.py

echo "==== figures DONE $(date) ===="
