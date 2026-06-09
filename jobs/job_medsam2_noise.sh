#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=medsam2-noise
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/medsam2-noise-%j.log

set -e
echo "==== medsam2_noise  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate train_gpu_env

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"

PROJ=/scratch/gautschi/li4533/MIUA_2026
cd $PROJ/MedSAM2

echo "--- Noise=0.10 ---"
python $PROJ/infer_medsam2.py \
    --ckpt checkpoints/MedSAM2_latest.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data $PROJ/preprocessed \
    --out  $PROJ/results/medsam2_noise10 \
    --bbox_noise 0.10 \
    --seed 42 \
    --overwrite

echo "--- Noise=0.20 ---"
python $PROJ/infer_medsam2.py \
    --ckpt checkpoints/MedSAM2_latest.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --data $PROJ/preprocessed \
    --out  $PROJ/results/medsam2_noise20 \
    --bbox_noise 0.20 \
    --seed 42 \
    --overwrite

echo "=== DONE $(date) ==="
