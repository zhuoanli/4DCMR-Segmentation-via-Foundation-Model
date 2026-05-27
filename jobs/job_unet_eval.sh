#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --job-name=unet-eval
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/unet_eval-%j.log

set -e
echo "==== unet_eval  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

python /scratch/gautschi/li4533/MIUA_2026/train_eval_unet.py \
    --db     /scratch/gautschi/li4533/MIUA_2026/database/training \
    --out    /scratch/gautschi/li4533/MIUA_2026/results/unet \
    --eval_only

echo "==== unet_eval DONE $(date) ===="
