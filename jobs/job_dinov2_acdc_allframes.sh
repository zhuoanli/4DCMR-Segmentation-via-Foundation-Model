#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --job-name=dinov2-allframes
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/dinov2-allframes-%j.log

set -e
echo "==== dinov2_acdc_allframes  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate train_gpu_env

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

PROJ=/scratch/gautschi/li4533/MIUA_2026

python $PROJ/infer_dinov2_acdc_allframes.py \
    --prep_dir $PROJ/preprocessed \
    --out      $PROJ/results/dinov2_acdc_allframes \
    --overwrite

echo "=== DONE $(date) ==="
