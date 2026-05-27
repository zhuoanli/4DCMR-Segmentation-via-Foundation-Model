#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --job-name=compute-metrics
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/metrics-%j.log

set -e
echo "==== compute_metrics  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="

module load conda
conda activate cinema_ft

pip install --user --quiet medpy scipy pandas

# ACDC validation set (existing predictions, no GPU needed)
python /scratch/gautschi/li4533/MIUA_2026/compute_all_metrics.py --dataset acdc_val

# ACDC test set (run after medsam2_acdc_test job)
if [ -d /scratch/gautschi/li4533/MIUA_2026/results/medsam2_acdc_test ]; then
    python /scratch/gautschi/li4533/MIUA_2026/compute_all_metrics.py --dataset acdc_test
fi

# MnM (run after medsam2_mnm job)
if [ -d /scratch/gautschi/li4533/MIUA_2026/results/medsam2_mnm ]; then
    python /scratch/gautschi/li4533/MIUA_2026/compute_all_metrics.py --dataset mnm
fi

# MnM2 (run after medsam2_mnm2 job)
if [ -d /scratch/gautschi/li4533/MIUA_2026/results/medsam2_mnm2 ]; then
    python /scratch/gautschi/li4533/MIUA_2026/compute_all_metrics.py --dataset mnm2
fi

echo "==== compute_metrics DONE $(date) ===="
echo "Results in /scratch/gautschi/li4533/MIUA_2026/results/table_*.csv"
