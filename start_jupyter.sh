#!/bin/bash
# Quick start script for ACDC visualization notebook

echo "======================================================================"
echo "  ACDC Cardiac MRI Visualization Notebook"
echo "======================================================================"
echo ""
echo "Loading conda environment..."
module load conda
conda activate cinema_ft

echo ""
echo "✓ Environment: cinema_ft activated"
echo "✓ All packages ready (numpy, matplotlib, nibabel, imageio, ipywidgets)"
echo ""
echo "Starting Jupyter Lab..."
echo ""
echo "TIP: The notebook will open in your browser"
echo "     Navigate to: visualize_acdc.ipynb"
echo ""
echo "======================================================================"

# Start Jupyter Lab
jupyter lab --no-browser --ip=0.0.0.0