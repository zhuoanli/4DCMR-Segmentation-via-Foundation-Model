#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:30:00
#SBATCH --job-name=unet-acdc
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/unet-%j.log

set -e
echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load anaconda
conda activate cinema_ft

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

export WANDB_MODE=disabled

echo "=== Training U-Net on ACDC (Exp E) ==="
python /scratch/gautschi/li4533/MIUA_2026/train_eval_unet.py \
    --db     /scratch/gautschi/li4533/MIUA_2026/database/training \
    --out    /scratch/gautschi/li4533/MIUA_2026/results/unet \
    --epochs 30 \
    --batch  16 \
    --lr     1e-4

echo "=== DONE: $(date) ==="
