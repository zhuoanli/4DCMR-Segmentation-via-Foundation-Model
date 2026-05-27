#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --job-name=medsam2-acdc-test
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/medsam2_acdc_test-%j.log
#SBATCH --chdir=/scratch/gautschi/li4533/MIUA_2026/MedSAM2

set -e
echo "==== medsam2_acdc_test  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft
pip install --user --quiet iopath

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="/scratch/gautschi/li4533/MIUA_2026/MedSAM2:$USER_SITE:${PYTHONPATH:-}"

python /scratch/gautschi/li4533/MIUA_2026/infer_medsam2.py \
    --ckpt checkpoints/MedSAM2_latest.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data /scratch/gautschi/li4533/MIUA_2026/preprocessed_acdc_test \
    --out  /scratch/gautschi/li4533/MIUA_2026/results/medsam2_acdc_test

echo "==== medsam2_acdc_test DONE $(date) ===="
