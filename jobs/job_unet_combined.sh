#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --job-name=unet-combined
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/unet_combined-%j.log

set -e
echo "==== unet_combined  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate train_gpu_env

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd /scratch/gautschi/li4533/MIUA_2026

echo "--- Training UNet on ACDC(100) + MnM2(360) ---"
python train_unet_combined.py \
    --acdc_dir    /scratch/gautschi/li4533/MIUA_2026/preprocessed \
    --mnm2_dir    /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm2 \
    --mnm_val_dir /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm \
    --out         /scratch/gautschi/li4533/MIUA_2026/results/unet_combined \
    --epochs 30 --batch 16 --lr 1e-4 --amp

echo "--- UNet combined inference on MnM (all 136 patients) ---"
python infer_unet_mnm.py \
    --ckpt  /scratch/gautschi/li4533/MIUA_2026/results/unet_combined/best_model.pth \
    --out   /scratch/gautschi/li4533/MIUA_2026/results/metrics_mnm.json \
    --prep_dir /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm

echo "==== unet_combined DONE $(date) ===="
