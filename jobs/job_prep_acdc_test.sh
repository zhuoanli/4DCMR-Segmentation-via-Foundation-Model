#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=prep-acdc-test
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/prep_acdc_test-%j.log

set -e
echo "==== prep_acdc_test  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

python /scratch/gautschi/li4533/MIUA_2026/prep_acdc_test.py \
    --db  /scratch/gautschi/li4533/MIUA_2026/database/testing \
    --out /scratch/gautschi/li4533/MIUA_2026/preprocessed_acdc_test

echo "==== prep_acdc_test DONE $(date) ===="
