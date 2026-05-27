#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --job-name=dinov2-combined
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/dinov2_combined-%j.log

set -e
echo "==== dinov2_combined  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda/2026.03
conda activate train_gpu_env

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd /scratch/gautschi/li4533/MIUA_2026

echo "--- Training DINOv2 decoder on ACDC(100) + MnM2(360) ---"
python train_dinov2_combined.py \
    --acdc_dir    /scratch/gautschi/li4533/MIUA_2026/preprocessed \
    --mnm2_dir    /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm2 \
    --mnm_val_dir /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm \
    --out         /scratch/gautschi/li4533/MIUA_2026/results/dinov2_combined \
    --epochs 100 --patience 15 --batch 8 --lr 5e-5

echo "--- DINOv2 combined inference on MnM (all 136 patients) ---"
python infer_dinov2_mnm.py \
    --ckpt  /scratch/gautschi/li4533/MIUA_2026/results/dinov2_combined/best_model.pth \
    --out   /scratch/gautschi/li4533/MIUA_2026/results/metrics_mnm.json \
    --prep_dir /scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm

echo "==== dinov2_combined DONE $(date) ===="
