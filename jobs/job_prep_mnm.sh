#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --job-name=prep-mnm
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/prep_mnm-%j.log

set -e
echo "==== prep_mnm  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

pip install --user --quiet pandas nibabel pillow tqdm

python /scratch/gautschi/li4533/MIUA_2026/prep_mnm.py \
    --test_dir /scratch/gautschi/li4533/MIUA_2026/MnM/Testing \
    --out      /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm

echo "==== prep_mnm DONE $(date) ===="
