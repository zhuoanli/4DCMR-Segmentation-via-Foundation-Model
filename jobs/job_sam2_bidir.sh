#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --job-name=sam2-bidir
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/sam2_bidir-%j.log
#SBATCH --chdir=/scratch/gautschi/li4533/MIUA_2026/MedSAM2

set -e
echo "==== sam2_bidir  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate cinema_ft
pip install --user --quiet iopath

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="/scratch/gautschi/li4533/MIUA_2026/MedSAM2:$USER_SITE:${PYTHONPATH:-}"

# SAM2-tiny (vanilla, no medical fine-tuning) with dual-anchored propagation
# Reuses infer_medsam2.py pipeline but with sam2.1_hiera_tiny.pt checkpoint
python /scratch/gautschi/li4533/MIUA_2026/infer_medsam2.py \
    --ckpt checkpoints/sam2.1_hiera_tiny.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data /scratch/gautschi/li4533/MIUA_2026/preprocessed \
    --out  /scratch/gautschi/li4533/MIUA_2026/results/sam2_bidir

echo "==== sam2_bidir DONE $(date) ===="
