#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --job-name=unet-allframes
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/unet-allframes-%j.log

set -e
echo "==== unet_acdc_allframes  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

PROJ=/scratch/gautschi/li4533/MIUA_2026

python $PROJ/infer_unet_acdc_allframes.py \
    --prep_dir $PROJ/preprocessed \
    --out      $PROJ/results/unet_acdc_allframes \
    --overwrite

echo "=== DONE $(date) ==="
