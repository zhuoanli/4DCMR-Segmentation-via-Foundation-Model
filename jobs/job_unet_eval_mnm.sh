#!/bin/bash
#SBATCH --account=life
#SBATCH --partition=smallgpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=unet-eval-mnm
#SBATCH --output=/scratch/gautschi/li4533/MIUA_2026/logs/unet_eval_mnm-%j.log

set -e
echo "==== unet_eval_mnm  Job: $SLURM_JOB_ID  Node: $(hostname)  $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load conda
conda activate train_gpu_env

cd /scratch/gautschi/li4533/MIUA_2026
python infer_unet_mnm.py

echo "==== unet_eval_mnm done $(date) ===="
