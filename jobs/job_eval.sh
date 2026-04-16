#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=eval-figures
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/eval-%j.log

set -e
echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================"

module load anaconda
conda activate cinema_ft

pip install --user --quiet matplotlib seaborn nibabel scipy

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

echo "=== Computing Dice tables and generating figures ==="
python /scratch/gautschi/li4533/MIUA_2026/evaluate_and_figures.py \
    --results_dir /scratch/gautschi/li4533/MIUA_2026/results \
    --db          /scratch/gautschi/li4533/MIUA_2026/database/training \
    --fig_dir     /scratch/gautschi/li4533/MIUA_2026/figures

echo "=== DONE: $(date) ==="
