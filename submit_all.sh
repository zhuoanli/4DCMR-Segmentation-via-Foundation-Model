#!/bin/bash
# submit_all.sh
# Downloads MedSAM2 checkpoints then submits the full experiment pipeline.
# Run this interactively: bash submit_all.sh
#
# Timeline (with parallel GPU jobs):
#   T+0:00  Download checkpoints (interactive, ~5 min)
#   T+0:05  Submit job_prep     (~15 min, no GPU)
#   T+0:20  Submit jobs 1-3 in parallel after prep (~35 min each on H100)
#   T+0:55  Submit job_eval after all 3 GPU jobs finish (~10 min)
#   T+1:05  ALL RESULTS READY

set -e
MIUA=/scratch/gautschi/li4533/MIUA_2026
MEDSAM2=$MIUA/MedSAM2

# ── Step 0: Download checkpoints ─────────────────────────────────────────────
echo "=== Downloading MedSAM2 and SAM2 checkpoints ==="
module load anaconda
conda activate cinema_ft

CKPT_DIR=$MEDSAM2/checkpoints
mkdir -p "$CKPT_DIR"

# MedSAM2_latest.pt (~300 MB)
if [ ! -f "$CKPT_DIR/MedSAM2_latest.pt" ]; then
    echo "Downloading MedSAM2_latest.pt ..."
    wget -q --show-progress -O "$CKPT_DIR/MedSAM2_latest.pt" \
        "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
    echo "  Done."
else
    echo "  MedSAM2_latest.pt already exists."
fi

# sam2.1_hiera_tiny.pt (~38 MB) — vanilla SAM2 for ablation
if [ ! -f "$CKPT_DIR/sam2.1_hiera_tiny.pt" ]; then
    echo "Downloading sam2.1_hiera_tiny.pt ..."
    wget -q --show-progress -O "$CKPT_DIR/sam2.1_hiera_tiny.pt" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    echo "  Done."
else
    echo "  sam2.1_hiera_tiny.pt already exists."
fi

echo ""
ls -lh "$CKPT_DIR/"
echo ""

# ── Step 1: Submit preprocessing job ─────────────────────────────────────────
echo "=== Submitting job_prep ==="
J0=$(sbatch --parsable "$MIUA/jobs/job_prep.sh")
echo "  job_prep: $J0"

# ── Step 2: Submit 3 GPU jobs in PARALLEL after prep ─────────────────────────
echo "=== Submitting GPU inference/training jobs (parallel, depend on prep) ==="
J1=$(sbatch --parsable --dependency=afterok:$J0 "$MIUA/jobs/job_medsam2.sh")
J2=$(sbatch --parsable --dependency=afterok:$J0 "$MIUA/jobs/job_sam2.sh")
J3=$(sbatch --parsable --dependency=afterok:$J0 "$MIUA/jobs/job_unet.sh")
echo "  job_medsam2: $J1"
echo "  job_sam2:    $J2"
echo "  job_unet:    $J3"

# ── Step 3: Submit evaluation after ALL 3 GPU jobs ───────────────────────────
echo "=== Submitting job_eval (depends on all 3 GPU jobs) ==="
J4=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3 "$MIUA/jobs/job_eval.sh")
echo "  job_eval: $J4"

echo ""
echo "========================================"
echo "Pipeline submitted! Job IDs:"
echo "  Prep:     $J0"
echo "  MedSAM2:  $J1"
echo "  SAM2:     $J2"
echo "  U-Net:    $J3"
echo "  Eval:     $J4"
echo ""
echo "Monitor:  squeue -u $USER"
echo "Logs:     $MIUA/logs/"
echo "Results:  $MIUA/results/"
echo "Figures:  $MIUA/figures/"
echo "========================================"
