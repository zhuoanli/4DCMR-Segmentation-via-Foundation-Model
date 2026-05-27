#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --job-name=sam2-bidir-mnm
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/sam2_bidir_mnm-%j.log
#SBATCH --chdir=/scratch/gautschi/li4533/MIUA_2026/MedSAM2

set -e
echo "==== sam2_bidir_mnm  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda/2026.03
conda activate train_gpu_env
pip install --user --quiet iopath

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="/scratch/gautschi/li4533/MIUA_2026/MedSAM2:$USER_SITE:${PYTHONPATH:-}"

python /scratch/gautschi/li4533/MIUA_2026/infer_medsam2.py \
    --ckpt checkpoints/sam2.1_hiera_tiny.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm \
    --out  /scratch/gautschi/li4533/MIUA_2026/results/sam2_bidir_mnm \
    --overwrite

echo "==== sam2_bidir_mnm DONE $(date) ===="
