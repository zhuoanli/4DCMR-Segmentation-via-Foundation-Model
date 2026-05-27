#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=metrics-final
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/metrics_final-%j.log

set -e
echo "==== metrics_final  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

cd /scratch/gautschi/li4533/MIUA_2026
python compute_all_metrics.py --dataset mnm
python evaluate_and_figures.py

echo "==== metrics_final done $(date) ===="
echo ""
echo "=== paper_table1_mnm_complete.csv ==="
cat results/paper_table1_mnm_complete.csv
echo ""
echo "=== paper_table_clinical_mnm_complete.csv ==="
cat results/paper_table_clinical_mnm_complete.csv
