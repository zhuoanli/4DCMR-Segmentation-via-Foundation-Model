#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=prep-mnm-all
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/prep_mnm_all-%j.log

set -e
echo "==== prep_mnm_all  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

cd /scratch/gautschi/li4533/MIUA_2026
python prep_mnm.py --overwrite --workers 32

echo "==== prep_mnm_all done $(date) ===="
