#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=prep-mnm2
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/prep_mnm2-%j.log

set -e
echo "==== prep_mnm2  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

cd /scratch/gautschi/li4533/MIUA_2026
python prep_mnm2_sa.py \
    --data_dir /scratch/gautschi/li4533/MIUA_2026/MnM2/dataset \
    --out      /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm2 \
    --overwrite \
    --workers 32

echo "==== prep_mnm2 DONE $(date) ===="
