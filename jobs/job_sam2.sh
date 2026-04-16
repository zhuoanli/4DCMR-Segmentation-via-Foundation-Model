#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:30:00
#SBATCH --job-name=sam2-ablation
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/sam2-%j.log
#SBATCH --chdir=/scratch/gautschi/li4533/MIUA_2026/MedSAM2

set -e
echo "========================================"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load anaconda
conda activate cinema_ft

pip install --user --quiet iopath

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="/scratch/gautschi/li4533/MIUA_2026/MedSAM2:$USER_SITE:${PYTHONPATH:-}"

echo "=== Running SAM2-tiny ablation (Exp D) ==="
python /scratch/gautschi/li4533/MIUA_2026/infer_sam2.py \
    --ckpt checkpoints/sam2.1_hiera_tiny.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data /scratch/gautschi/li4533/MIUA_2026/preprocessed \
    --out  /scratch/gautschi/li4533/MIUA_2026/results/sam2

echo "=== DONE: $(date) ==="
