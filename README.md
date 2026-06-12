# Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation via Dual-Anchored Medical Video Foundation Model

**MIUA 2026 — Main Track (Accepted)**

---

## Abstract

Cardiac cine MRI segmentation across the full cardiac cycle is essential for computing ejection fraction, ventricular volumes, and time-resolved functional indices, yet dense frame-level annotation remains costly. Conventional supervised methods are trained on two clinically labelled key frames—end-diastole (ED) and end-systole (ES)—leaving intermediate frames unsegmented. We present a zero-shot framework using **MedSAM2**, a medical video foundation model, with a **dual-anchored propagation strategy** that prompts the model at both ED and ES and merges bidirectional predictions at the temporal midpoint. On ACDC (n=20, stratified), MedSAM2 (Dual-anchored) achieves Dice 0.850/0.809/0.843 for RV/Myo/LV with RV HD95 2.94 mm—matching supervised baselines on RV without any cardiac-specific training. Full-cycle propagation further enables extraction of time-resolved biomarkers (PER, PFR, SV) that reveal pathology-specific functional signatures inaccessible to ED/ES-only methods.

---

## Method Overview

![Method Overview](results/figures/fig1_methods.png)

**(a) Data and prompt preparation** — 4D cine MRI sequences preprocessed slice-by-slice to 512×512; bounding-box prompts at ED and ES anchor frames.  
**(b) Dual-anchored MedSAM2 propagation** — forward pass from ED + backward pass from ES, merged at temporal midpoint mid = ⌊(t_ED + t_ES)/2⌋.  
**(c) Full-cycle functional analysis** — V(t) curve yields PER, PFR, SV, TMS biomarkers per patient.  
**(d) Study design** — ACDC 100 patients, 5 pathology groups, evaluated against SAM2, U-Net, and DINOv2 baselines.

---

## Key Results

| Method | RV Dice | Myo Dice | LV Dice | RV HD95 (mm) | RV ASSD (mm) |
|--------|---------|----------|---------|-------------|-------------|
| SAM2 (Dual-anchored) | 0.745 | 0.647 | 0.806 | 5.86 | 1.20 |
| MedSAM2 (ED-anchored) | 0.716 | 0.667 | 0.699 | 8.64 | 2.99 |
| MedSAM2 (ES-anchored) | 0.784 | 0.789 | 0.856 | 7.71 | 2.70 |
| **MedSAM2 (Dual-anchored) †** | **0.850** | **0.809** | 0.843 | **2.94** | **0.55** |
| U-Net (supervised) ‡ | 0.730 | **0.861** | **0.868** | 2.23 | 0.37 |
| DINOv2 (supervised) ‡ | 0.553 | 0.719 | 0.793 | 12.91 | 2.81 |

† Proposed zero-shot method. ‡ Supervised methods trained and evaluated at ES only.

---

## Repository Structure

```
MIUA_2026/
├── preprocessing/
│   ├── prep_acdc_4d.py       # ACDC training set preprocessing (NIfTI → NPZ)
│   └── prep_acdc_test.py     # ACDC test set preprocessing
├── inference/
│   ├── infer_medsam2.py      # Zero-shot MedSAM2: ED/ES/dual-anchor propagation
│   ├── infer_sam2.py         # SAM2 baseline inference
│   ├── infer_unet_acdc_allframes.py  # U-Net all-frame inference for HD95
│   └── infer_dinov2_acdc_allframes.py
├── training/
│   ├── train_eval_unet.py    # U-Net training and evaluation
│   └── train_eval_dinov2.py  # DINOv2 segmentation head training and evaluation
├── compute_all_metrics.py    # Dice, HD95, ASSD, EF/EDV/ESV computation
├── evaluate_and_figures.py   # All paper figures and tables
├── requirements.txt
└── results/
    ├── figures/              # Paper figures (Fig. 1–5)
    ├── tables/               # Paper tables (CSV)
    ├── metrics_acdc_val.json # All method metrics (ACDC validation)
    └── metrics_acdc_test.json
```

---

## Installation

### Requirements

- Python 3.10
- CUDA 12.4
- PyTorch 2.5.0+cu124

### Setup

```bash
conda create -n medsam2_cardiac python=3.10
conda activate medsam2_cardiac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### MedSAM2 Checkpoint

Download the MedSAM2 checkpoint from the [MedSAM2 repository](https://github.com/bowang-lab/MedSAM2) and place it at:

```
MedSAM2/checkpoints/MedSAM2.pt
MedSAM2/configs/sam2.1_hiera_t.yaml
```

---

## Dataset Preparation (ACDC)

1. Download the ACDC dataset from [acdc.creatis.insa-lyon.fr](https://acdc.creatis.insa-lyon.fr)
2. Place under `ACDC_training/` and `ACDC_testing/`
3. Run preprocessing:

```bash
python preprocessing/prep_acdc_4d.py \
    --acdc_dir ACDC_training/ \
    --output_dir preprocessed/ \
    --split train
```

This produces per-patient, per-slice NPZ files with keys: `imgs` (T×512×512), `gts` (T×512×512), `ed_idx`, `es_idx`, `group`.

---

## Running Inference (Zero-Shot)

### Dual-anchored (proposed)

```bash
python inference/infer_medsam2.py \
    --preprocessed_dir preprocessed/ \
    --output_dir results/medsam2/ \
    --model_cfg MedSAM2/configs/sam2.1_hiera_t.yaml \
    --checkpoint MedSAM2/checkpoints/MedSAM2.pt \
    --mode bidir
```

### Single-anchor variants 

```bash
# ED-anchored
python inference/infer_medsam2.py --mode forward --output_dir results/medsam2_ed/ ...

# ES-anchored
python inference/infer_medsam2.py --mode backward --output_dir results/medsam2_es/ ...
```

### Bbox noise robustness 

```bash
python inference/infer_medsam2.py --mode bidir --bbox_noise 0.10 --output_dir results/medsam2_noise10/ ...
python inference/infer_medsam2.py --mode bidir --bbox_noise 0.20 --output_dir results/medsam2_noise20/ ...
```

---

## Evaluation

```bash
python compute_all_metrics.py --dataset acdc
```

Outputs metrics to `results/metrics_acdc_val.json` including Dice, HD95, ASSD per patient per method.

---

## Training Baselines

### U-Net (supervised)

```bash
python training/train_eval_unet.py --mode train \
    --preprocessed_dir preprocessed/ \
    --output_dir results/unet/

python training/train_eval_unet.py --mode eval \
    --preprocessed_dir preprocessed/ \
    --checkpoint results/unet/best_model.pth
```

### DINOv2 (supervised)

```bash
python training/train_eval_dinov2.py --mode train \
    --preprocessed_dir preprocessed/ \
    --output_dir results/dinov2/

python training/train_eval_dinov2.py --mode eval \
    --preprocessed_dir preprocessed/ \
    --checkpoint results/dinov2/best_model.pth
```

---

## Evaluation

```bash
python evaluate_and_figures.py
```

Outputs all paper figures to `results/figures/` and tables to `results/tables/`.

---

## Results

All final figures and tables used in the paper are in `results/`:

| File | Paper label |
|------|------------|
| `results/figures/fig1_methods.png` | Fig. 1 — Methods overview |
| `results/figures/fig2_qualitative.png` | Fig. 2 — Qualitative results |
| `results/figures/fig3_timevolume.png` | Fig. 3 — LV time-volume curves |
| `results/figures/fig4_dice_boxplot.png` | Fig. 4 — Dice ablation box plots |
| `results/figures/fig5_heatmap.png` | Fig. 5 — Pathology-stratified Dice heatmap |
| `results/tables/table1_segmentation.csv` | Table 1 — Dice/HD95/ASSD |
| `results/tables/table4_biomarkers.csv` | Table 4 — Per-group biomarkers |
| `results/tables/tableA_noise.csv` | Supp. Table A — Bbox noise robustness |
| `results/tables/tableB_clinical.csv` | Supp. Table B — EF/EDV/ESV |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2026zeroshotcardiac,
  title     = {Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation via Dual-Anchored Medical Video Foundation Model},
  author    = {Li, Zhuoan et al.},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026}
}
```

---

## Acknowledgements

This work uses the [MedSAM2](https://github.com/bowang-lab/MedSAM2) framework and the [ACDC dataset](https://acdc.creatis.insa-lyon.fr). We thank the ACDC organisers for providing a publicly available benchmark.
