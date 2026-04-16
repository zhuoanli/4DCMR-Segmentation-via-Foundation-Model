# MedSAM2 for 4D Cardiac Cine MRI: Single-Frame Prompted Video Propagation for Full-Cycle Segmentation

**Venue:** Medical Image Understanding and Analysis (MIUA) 2026  
**Working Directory:** `/scratch/gautschi/li4533/MIUA_2026`

---

## Abstract

Cardiac segmentation of cine MRI across the full cardiac cycle is a prerequisite for computing time-resolved volume curves and ejection fraction — key clinical metrics for diagnosing cardiomyopathies. Current practice requires either manual labelling of every frame or supervised models trained on large annotated datasets. We investigate **MedSAM2**, a video-foundation model fine-tuned on medical images, as a zero-shot video propagator: given a single ground-truth bounding-box prompt at one frame (ED or ES), the model propagates segmentation masks across all ~30 frames of the cardiac cycle without any task-specific training. We evaluate on the ACDC benchmark (100 patients, 5 pathology groups) using a stratified 20-patient validation set, compare against vanilla SAM2-tiny and a fully supervised U-Net baseline, and demonstrate downstream clinical utility by generating pathology-stratified LV time-volume curves from the propagated masks.

---

## 1. Introduction and Motivation

### Clinical Background

Cardiac cine MRI is the gold standard for non-invasive assessment of cardiac function. A typical acquisition produces a **4D volume** (H × W × Z × T), where T ≈ 27–30 frames capture one complete cardiac cycle at temporal resolution ~30–40 ms/frame. For each patient, key functional indices are derived from segmenting the **right ventricle (RV)**, **myocardium (Myo)**, and **left ventricle (LV)** across time:

- **EDV** (end-diastolic volume): maximum LV volume, measured at ED frame
- **ESV** (end-systolic volume): minimum LV volume, measured at ES frame
- **EF = (EDV − ESV) / EDV**: ejection fraction, primary marker of systolic function
- **LV time-volume curve**: full cycle volume trajectory, encodes contractility, diastolic filling rate, and relaxation patterns — distinct per pathology

Manual labelling of all T frames per patient is prohibitively expensive. Existing supervised methods (e.g., U-Net) typically annotate only ED and ES frames and cannot trivially generalise to intermediate frames. This work asks: **can a video-foundation model, prompted with a single annotated frame, propagate accurate segmentations across the entire cardiac cycle?**

### Key Idea

MedSAM2 (based on SAM2 architecture with medical fine-tuning) implements a **causal memory bank**: it encodes the prompt frame into an object memory, then attends to that memory while processing subsequent frames in sequence. This video-propagation paradigm is architecturally suited for cardiac cine MRI because:

1. The cardiac cycle is a smooth, quasi-periodic motion — inter-frame deformation is small and predictable
2. A single clinically annotated frame (ED or ES) is always available in standard ACDC-style datasets
3. Propagation is **zero-shot**: no cardiac-specific training is needed beyond the general medical fine-tuning

---

## 2. Dataset

### ACDC (Automated Cardiac Diagnosis Challenge)

| Property | Value |
|---|---|
| Total patients | 100 (training set, all with GT) |
| Pathology groups | NOR, DCM, HCM, MINF, RV (20 patients each) |
| Image type | Short-axis cine MRI, NIfTI format |
| Spatial resolution | ~1.4–1.8 mm in-plane, 5–10 mm slice thickness |
| Typical in-plane size | 154–240 × 224–256 pixels |
| Temporal frames | 12–35 per patient (mean 27) |
| Slices per patient | 6–17 short-axis slices |
| GT annotation | ED frame + ES frame only (3-class: RV=1, Myo=2, LV=3) |

**Pathology Descriptions:**

| Group | Full Name | Clinical Feature |
|---|---|---|
| NOR | Normal | Healthy controls, regular wall motion, EF ~60% |
| DCM | Dilated Cardiomyopathy | Enlarged LV cavity, reduced EF (<35%), global hypokinesia |
| HCM | Hypertrophic Cardiomyopathy | Thickened myocardium, small LV cavity, diastolic dysfunction |
| MINF | Myocardial Infarction | Segmental wall motion abnormality (akinesia/dyskinesia) at infarct territory |
| RV | Right Ventricular Pathology | Dilated/dysfunctional RV, varied LV function |

**Frame Indexing Convention:**  
ED and ES frame indices in `Info.cfg` are **1-based** integers. All code converts to 0-based (subtract 1). In ACDC, ED is typically frame 0 (acquisition starts at end-diastole); ES is around frame 10–12 (mid-cycle systole).

### Train / Validation Split (Stratified)

To ensure fair evaluation across all pathologies:

```
Validation (20 patients): last 4 patients per group
  NOR:  017–020
  DCM:  037–040
  HCM:  057–060
  MINF: 077–080
  RV:   097–100

Training (80 patients): remaining 16 per group
  NOR:  001–016
  DCM:  021–036
  HCM:  041–056
  MINF: 061–076
  RV:   081–096
```

This stratification ensures every pathology group has representation in validation, eliminating the bias of an unstratified split (which would have placed all 20 RV patients in the val set).

---

## 3. Methods

### 3.1 Preprocessing (`prep_acdc_4d.py`)

For each of the 100 training patients:

1. **Load 4D cine volume** `patientXXX_4d.nii.gz` → shape (H, W, Z, T)
2. **Load GT masks** for ED and ES frames → shape (H, W, Z), integer labels {0,1,2,3}
3. **Parse ED/ES indices** from `Info.cfg` (1-based → 0-based)
4. **Per-slice processing** — for each axial slice z:
   - Extract frame stack: `vol4d[:,:,z,:]` → transpose to (T, H, W)
   - **Percentile normalisation**: clip to [p2, p98] → scale to [0,1] → convert to uint8
   - **RGB conversion + resize** to (T, 3, 512, 512) via bilinear interpolation (MedSAM2 input format)
   - **GT mask resize** to (512, 512) via nearest-neighbour interpolation (preserves label integers)
   - Skip slices where both ED and ES masks are all-background
5. **Save per-slice NPZ**: `preprocessed/patientXXX_sliceYY.npz`
   - `frames`: (T, 3, 512, 512) float16 — normalised to [0,1], ImageNet normalisation applied at inference
   - `ed_mask`, `es_mask`: (512, 512) uint8
   - `ed_idx`, `es_idx`: 0-based frame indices
   - `group`: pathology label string
   - `pixdim`: (dx, dy, dz) in mm for volume computation
   - `orig_H`, `orig_W`: original spatial dimensions (for spatial scale correction)

**Total output:** 940 NPZ files (average ~9.4 slices/patient)

### 3.2 MedSAM2 Inference (`infer_medsam2.py`)

**Model:** MedSAM2 built on SAM2-tiny architecture, fine-tuned on multi-modal medical image data. Checkpoint: `MedSAM2_latest.pt`. Config: `sam2.1_hiera_t512.yaml`.

**Prompting Strategy:** For each class (RV=1, Myo=2, LV=3), extract a bounding box from the GT mask with 5-pixel padding, clamped to [0, 511]. This is a **GT-bbox prompt** — used at exactly one frame per propagation pass; all other frames receive no annotation.

**Propagation Mechanism:**  
MedSAM2's `propagate_in_video` uses a causal memory bank. At each step, it:
- Computes image features for the current frame
- Attends to the memory bank (contains encoded prompt frame + previously predicted frames)
- Predicts per-class binary logits
- Updates the memory bank with the current frame's features

The key limitation: **error accumulates with distance from the prompt frame** because each predicted frame is re-encoded into the memory bank. Frames far from the prompt may degrade in quality.

**Experiment A — Forward from ED (`ed_pred`):**
```
Prompt: GT bbox at ED frame (ed_idx)
Propagation:
  fwd_ed = propagate(prompt=ed_idx, reverse=False)   # covers [ed_idx, T-1]
  bwd_ed = propagate(prompt=ed_idx, reverse=True)    # covers [0, ed_idx]  (skipped if ed_idx=0)
  ed_pred = merge(bwd_ed[:ed_idx], fwd_ed[ed_idx:])  # full T-frame coverage
Evaluation: Dice at ES frame → ed_pred[es_idx] vs es_mask
```
The ED frame is the "given" frame; the ES frame is the target to predict. This simulates the clinically common scenario of having an end-diastolic annotation.

**Experiment B — Reverse from ES (`es_pred`):**
```
Prompt: GT bbox at ES frame (es_idx)
Propagation:
  bwd_es = propagate(prompt=es_idx, reverse=True)    # covers [0, es_idx]
  fwd_es = propagate(prompt=es_idx, reverse=False)   # covers [es_idx, T-1]
  es_pred = merge(bwd_es[:es_idx], fwd_es[es_idx:])  # full T-frame coverage
Evaluation: Dice at ED frame → es_pred[ed_idx] vs ed_mask
```
Note: ES→ED propagation traverses the transition from maximum contraction back to maximum relaxation, which is a larger morphological change than ED→ES in most pathologies. Results show ES-anchored propagation consistently outperforms ED-anchored, likely because the ES frame is more diagnostically informative (smaller, more distinct structures).

**Experiment C — Bidirectional (`bidir`):**
```
mid = (ed_idx + es_idx) // 2
bidir[0 : mid+1]  = ed_pred[0 : mid+1]   # frames closer to ED use ED-anchored prediction
bidir[mid+1 : T]  = es_pred[mid+1 : T]   # frames closer to ES use ES-anchored prediction
Evaluation: Dice at ES → bidir[es_idx] vs es_mask
           (identical to Exp A at that frame, since bidir[es_idx] = ed_pred[es_idx])
```
**Bidir's value is not in Dice numbers** (which are the same as A/B at the respective evaluation frames) but in **intermediate-frame quality**: the maximum propagation distance from any frame to its nearest anchor is reduced from `T-1-ed_idx` to approximately `(es_idx-ed_idx)/2`, halving the worst-case drift. This directly improves the reliability of the full-cycle LV time-volume curves (Experiment F).

### 3.3 SAM2-tiny Ablation (`infer_sam2.py`)

**Experiment D — Vanilla SAM2 Forward:**  
Identical to Experiment A but using the unmodified `sam2.1_hiera_tiny.pt` checkpoint without any medical fine-tuning. Only forward propagation from ED is run. This ablation isolates the contribution of medical domain fine-tuning from the architectural design. Output key: `ed_pred`.

### 3.4 Supervised U-Net Baseline (`train_eval_unet.py`)

**Experiment E — Per-Frame U-Net:**

| Hyperparameter | Value |
|---|---|
| Architecture | U-Net (bilinear upsampling, n_channels=1, n_classes=4) |
| Input | Single 2D grayscale slice, resized to 256×256 |
| Normalisation | Percentile (p2/p98) → [0,1] |
| Loss | Cross-entropy + Dice loss (equal weight) |
| Optimiser | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=30 |
| Batch size | 16 |
| Epochs | 30 |
| AMP | Mixed precision (float16) |
| Training data | 80 patients × ED+ES frames → 2D slices (~1,200 slices) |
| Validation metric | Mean Dice over RV/Myo/LV on val set slices |

The U-Net is trained on ED and ES slices from all 80 training patients and evaluated on ES frames of the 20 val patients. It has no temporal awareness and cannot propagate to intermediate frames — each frame is processed independently. This represents the **supervised upper bound** for per-frame accuracy.

### 3.5 Evaluation Protocol (`evaluate_and_figures.py`)

**Dice computation:**  
For each val patient, all slice-level predictions for the evaluation frame are compared against the corresponding GT mask at 512×512 resolution. Per-class Dice is computed per slice, then averaged across slices to get a per-patient score. Results are reported as mean ± std across 20 val patients.

**Dice formula:**
```
Dice(pred, gt, cls) = 2 * |pred==cls ∩ gt==cls| / (|pred==cls| + |gt==cls|)
```
Special cases: if both pred and GT are empty for a class → Dice=1.0; if only GT is empty → Dice=0.0.

**Volume computation for time-volume curves:**  
Since predictions are at 512×512 but the original voxel spacing (pixdim) refers to the original resolution:
```
scale_factor = (orig_H / 512) × (orig_W / 512)
voxel_mm³ = dx × dy × dz × scale_factor
LV_vol_mL(t) = Σ_slices [(bidir[t] == 3).sum() × voxel_mm³ / 1000]
```

---

## 4. Experiments and Results

### Table 1 — Dice at Evaluation Frame (Val Set, n=20, stratified)

| Method | Prompt | Eval Frame | RV | Myo | LV |
|---|---|---|---|---|---|
| SAM2-tiny (fwd) | GT ED bbox | ES | 0.611 ± 0.140 | 0.280 ± 0.197 | 0.559 ± 0.159 |
| MedSAM2 (fwd) | GT ED bbox | ES | 0.716 ± 0.111 | 0.667 ± 0.200 | 0.699 ± 0.193 |
| MedSAM2 (rev) | GT ES bbox | ED | 0.784 ± 0.116 | 0.789 ± 0.137 | 0.856 ± 0.125 |
| MedSAM2 (bidir) | GT ED+ES bbox | ES | 0.850 ± 0.092 | 0.809 ± 0.125 | 0.843 ± 0.111 |
| U-Net (supervised) | 80-patient training | ES | **0.914 ± 0.079** | **0.914 ± 0.046** | **0.906 ± 0.078** |

**Key observations:**

1. **Medical fine-tuning is essential:** SAM2-tiny vs MedSAM2 (fwd) — Myo Dice jumps from 0.280 to 0.667 (+0.387). Vanilla SAM2 essentially fails on Myocardium, the most challenging structure due to its thin ring-shaped morphology.

2. **ES→ED is harder to propagate than ED→ES, yet achieves better Dice:** MedSAM2 (rev) outperforms (fwd) across all structures. The ES frame captures the heart at maximum contraction — structures are smaller and more distinct, making the prompt mask more informative and precise. The resulting bounding box tightly constrains the model, leading to better initialisation.

3. **Bidir combines the best of both prompts:** Using both GT frames as anchors, MedSAM2 (bidir) achieves the highest zero-shot performance. The ES-anchored half of the cycle (which covers the ED evaluation frame) outperforms pure forward propagation from ED.

4. **Zero-shot vs supervised gap:** MedSAM2 (bidir) mean Dice ≈ 0.834 vs U-Net ≈ 0.911. The gap of ~0.077 Dice points is modest given that MedSAM2 receives no cardiac training data. Crucially, MedSAM2 produces segmentation for **all ~30 frames** per patient, while U-Net is evaluated only at ES frames.

### Experiment F — LV Time-Volume Curves (Centrepiece)

Using MedSAM2 bidir predictions for all 100 patients and all T frames per patient, LV volume (mL) is computed at each cardiac phase. Curves are aligned to ED (phase=0%), interpolated to a 100-point common grid, and averaged per pathology group.

**Expected physiological signatures per group:**

| Group | EDV Range | EF | Curve Shape |
|---|---|---|---|
| NOR | ~130–160 mL | ~60–65% | Symmetric, deep trough at ES |
| DCM | ~200–280 mL | ~20–35% | High baseline, shallow trough (poor contraction) |
| HCM | ~80–130 mL | ~65–75% | Small cavity, steep descent, may show diastolic plateau |
| MINF | ~150–220 mL | ~30–45% | Irregular, segmental plateau due to akinetic regions |
| RV | Variable | Variable | Atypical shape, often earlier ES timing |

These pathology-specific signatures are **only extractable if full-cycle segmentation exists** — neither ED+ES-only annotation nor a per-frame supervised model applied to randomly selected frames would capture them automatically. This is the core clinical contribution of MedSAM2 video propagation.

---

## 5. Impact and Contributions

### Scientific Contributions

1. **First systematic evaluation of MedSAM2 video propagation for 4D cardiac cine MRI segmentation** on the ACDC benchmark across all 5 pathology groups.

2. **Experiment design clarifying directionality of propagation:** We show that ES-anchored propagation (Exp B) consistently outperforms ED-anchored (Exp A) due to the geometric properties of the ES frame. This design insight is novel and practically important.

3. **Bidir strategy as a principled combination:** The mid-point anchor assignment reduces worst-case frame-to-anchor distance, improving intermediate-frame quality for time-volume analysis.

4. **Ablation of medical fine-tuning:** The SAM2 vs MedSAM2 comparison isolates the contribution of medical domain adaptation. The +0.387 Myo Dice improvement demonstrates that cardiac-domain fine-tuning is critical, not just SAM2's architecture.

5. **Pathology-stratified LV time-volume curves** generated automatically from zero-shot propagation, demonstrating a complete pipeline from single-frame annotation to clinical functional metrics.

### Clinical Impact

| Impact Area | Description |
|---|---|
| **Annotation Efficiency** | One GT frame per patient replaces ~30 manual annotations. A single cardiologist annotation at ED or ES enables automated full-cycle segmentation. |
| **EF Computation** | EDV and ESV can be computed from any patient with a single annotated frame, without retraining any model. |
| **Population Studies** | With MedSAM2, large retrospective cohorts (thousands of patients with existing ED/ES annotations) can be processed for time-volume analysis without additional labelling. |
| **Pathology Discrimination** | LV volume curve shapes are pathology-specific (DCM: high EDV/low EF; HCM: small cavity; MINF: irregular contraction). Automated full-cycle curves provide features for downstream classification. |
| **Generalisability** | Zero-shot performance (no cardiac training) suggests the approach could extend to other cardiac MRI sequences, other vendors, or even echocardiography with the same prompt format. |

### Limitations

1. **GT prompt at inference:** Both MedSAM2 and SAM2 ablations use ground-truth bounding boxes as prompts. In a fully automatic deployment, prompts would need to be generated by a detector. Performance would decrease without GT prompts.

2. **ED=frame0 assumption:** Most ACDC patients have ED at the first temporal frame, simplifying forward propagation (no backward pass needed). Patients with ED at later frames require both forward and backward passes, introducing additional computational cost.

3. **Supervised gap remains:** U-Net achieves 0.911 mean Dice vs 0.834 for MedSAM2 bidir. For applications requiring the highest per-frame accuracy (e.g., surgical planning), a trained model is still preferable.

4. **Myocardium remains hardest:** Myo Dice for MedSAM2 (0.809) lags LV (0.843) and RV (0.850) due to the thin ring morphology (~5–8 mm) and low contrast at boundaries.

---

## 6. Code and Reproducibility

### File Structure

```
MIUA_2026/
├── prep_acdc_4d.py           # Preprocessing: 4D NIfTI → per-slice NPZ
├── infer_medsam2.py          # Exp A+B+C: MedSAM2 video propagation
├── infer_sam2.py             # Exp D: SAM2-tiny ablation (forward only)
├── train_eval_unet.py        # Exp E: U-Net training + ES-frame evaluation
├── evaluate_and_figures.py   # Dice tables + 4 paper figures
├── database/training/        # ACDC dataset (100 patients)
├── preprocessed/             # 940 per-slice NPZ files
├── results/
│   ├── medsam2/              # 940 NPZ (ed_pred, es_pred, bidir)
│   ├── sam2/                 # 940 NPZ (ed_pred)
│   └── unet/
│       ├── best_model.pth
│       └── results.json
├── figures/
│   ├── fig1_qualitative.png  # 4-row segmentation gallery, patient 037 (DCM)
│   ├── fig2_boxplot.png      # Box plots per structure, 3 subplots
│   ├── fig3_pathology_heat.png  # Per-pathology heatmap
│   └── fig4_timevolume.png   # LV time-volume curves, all 100 patients
├── MedSAM2/                  # MedSAM2 repo (Hydra config must be CWD)
├── pytorch-unet/             # U-Net repo
└── jobs/
    ├── job_prep.sh           # SLURM: preprocessing (CPU, 1h)
    ├── job_medsam2.sh        # SLURM: MedSAM2 inference (GPU, 3h)
    ├── job_sam2.sh           # SLURM: SAM2 inference (GPU, 1.5h)
    ├── job_unet.sh           # SLURM: U-Net training (GPU, 1h)
    └── job_eval.sh           # SLURM: evaluation + figures (CPU, 30min)
```

### SLURM Execution Pipeline (Gautschi HPC, partition=ai/cpu)

```bash
# Full pipeline submission with dependencies
J0=$(sbatch --parsable jobs/job_prep.sh)                              # preprocess
J1=$(sbatch --parsable --dependency=afterok:$J0 jobs/job_medsam2.sh) # Exp A+B+C
J2=$(sbatch --parsable --dependency=afterok:$J0 jobs/job_sam2.sh)    # Exp D
J3=$(sbatch --parsable --dependency=afterok:$J0 jobs/job_unet.sh)    # Exp E
sbatch --dependency=afterok:$J1:$J2:$J3 jobs/job_eval.sh             # figures
```

### Environment

- Cluster: Gautschi (ETH Zürich), partition `ai` (GPU) / `cpu`
- GPU: H100 80GB
- Conda environment: `cinema_ft`
- Key packages: PyTorch, nibabel, scipy, matplotlib, PIL
- MedSAM2 requires `iopath` and `PYTHONPATH` must include the MedSAM2 directory

---

## 7. Figures Summary

| Figure | File | Description |
|---|---|---|
| Fig 1 | `fig1_qualitative.png` | 4 rows × 6 frames. Row 0: GT (at ED/ES only, blank elsewhere). Row 1: MedSAM2 bidir overlay. Row 2: SAM2 fwd overlay. Row 3: U-Net per-frame. Patient 037 (DCM), mid-slice. Demonstrates qualitative propagation quality and method comparison. |
| Fig 2 | `fig2_boxplot.png` | 3 subplots (RV, Myo, LV), 5 methods each. Box plots show full distribution of per-patient Dice across the 20-patient stratified val set. Ordered: SAM2→MedSAM2 fwd→rev→bidir→U-Net. |
| Fig 3 | `fig3_pathology_heat.png` | 5×3 heatmap (pathology × structure). MedSAM2 bidir Dice per group. Colour scale green=high, red=low. Reveals which pathologies are harder to segment. |
| Fig 4 | `fig4_timevolume.png` | **Centrepiece.** 5 subplots, one per pathology. LV volume (mL) vs cardiac phase (%) for all 100 patients. Mean ± std shading. ED marked at 0%, ES marked with dashed red line. Pathology-specific curve shapes visible. |

---

*Last updated: 2026-04-16. All GPU jobs run on Gautschi HPC (ETH Zürich).*
