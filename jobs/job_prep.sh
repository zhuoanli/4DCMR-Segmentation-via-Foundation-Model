#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=acdc-prep
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/acdc-prep-%j.log

set -e
echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================"

module load anaconda
conda activate cinema_ft

pip install --user --quiet nibabel SimpleITK scikit-image pillow

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

echo "=== Preprocessing ACDC 4D data ==="
python /scratch/gautschi/li4533/MIUA_2026/prep_acdc_4d.py \
    --db  /scratch/gautschi/li4533/MIUA_2026/database/training \
    --out /scratch/gautschi/li4533/MIUA_2026/preprocessed

echo "=== DONE: $(date) ==="
