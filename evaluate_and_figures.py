"""
evaluate_and_figures.py
Post-processing: compute Dice tables and generate all paper figures.

Figures produced:
  figures/fig1_qualitative.png   — GT / MedSAM2 / SAM2 / U-Net overlay at 6 time points
  figures/fig2_boxplot.png       — box plots + significance stars, all methods
  figures/fig3_pathology_heat.png — per-pathology Dice heatmap (MedSAM2 bidir)
  figures/fig4_timevolume.png    — LV time-volume curves per pathology group (centrepiece)

Tables printed to stdout and saved as CSV in results/.
"""

import os, sys, json, argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob
from tqdm import tqdm
from PIL import Image

RESULTS_DIR = '/scratch/gautschi/li4533/MIUA_2026/results'
DB_PATH     = '/scratch/gautschi/li4533/MIUA_2026/database/training'
FIG_DIR     = '/scratch/gautschi/li4533/MIUA_2026/figures'

# Stratified val split: last 4 patients per group (each group has 20 patients)
# NOR:001-020→val 017-020, DCM:021-040→val 037-040, HCM:041-060→val 057-060
# MINF:061-080→val 077-080, RV:081-100→val 097-100
VAL_IDS     = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]
CLASSES     = {1: 'RV', 2: 'Myo', 3: 'LV'}
GROUPS      = ['NOR', 'DCM', 'HCM', 'MINF', 'RV']
GROUP_COLOR = {'NOR': '#2196F3', 'DCM': '#F44336', 'HCM': '#4CAF50',
               'MINF': '#FF9800', 'RV': '#9C27B0'}
LABEL_COLOR = {1: (0.9, 0.1, 0.1), 2: (0.1, 0.8, 0.1), 3: (0.1, 0.1, 0.9)}


# ── helpers ───────────────────────────────────────────────────────────────────
def dice_np(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


def parse_info_cfg(cfg_path):
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def decode_group(raw) -> str:
    """Decode group field regardless of how many times it was encoded in the NPZ."""
    s = str(raw)
    # Strip np.bytes_(b'...') wrapper from double-encoding
    if s.startswith("np.bytes_(b'") and s.endswith("')"):
        s = s[len("np.bytes_(b'"):-2]
    # Strip plain b'...' wrapper
    elif s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s


def overlay_mask(frame_u8_512, pred_mask_512):
    """Blend colorised segmentation mask onto a grayscale frame. Returns float32 (512,512,3)."""
    rgb = np.stack([frame_u8_512] * 3, axis=-1).astype(np.float32) / 255.0
    for cls_id, color in LABEL_COLOR.items():
        m = (pred_mask_512 == cls_id)
        for c, cv in enumerate(color):
            rgb[:, :, c][m] = rgb[:, :, c][m] * 0.4 + cv * 0.6
    return np.clip(rgb, 0, 1)


# ── Dice computation ──────────────────────────────────────────────────────────
def compute_method_dice(result_dir, mode='ed_pred', eval_at='es'):
    """
    mode     : 'ed_pred' | 'es_pred' | 'bidir'
    eval_at  : 'es' (compare pred at es_idx vs es_mask)  |  'ed' (compare at ed_idx vs ed_mask)

    Returns dict: pid -> {RV, Myo, LV, group}
    """
    prep_dir = os.path.normpath(os.path.join(RESULTS_DIR, '..', 'preprocessed'))
    results  = {}

    for pid in VAL_IDS:
        res_npzs  = sorted(glob(os.path.join(result_dir, f'patient{pid:03d}_slice*.npz')))
        prep_npzs = sorted(glob(os.path.join(prep_dir,   f'patient{pid:03d}_slice*.npz')))
        if not res_npzs or not prep_npzs:
            continue

        slice_dices = {1: [], 2: [], 3: []}
        group = 'UNK'

        for r_path, p_path in zip(res_npzs, prep_npzs):
            rd = np.load(r_path,  allow_pickle=True)
            pd = np.load(p_path,  allow_pickle=True)

            if mode not in rd:
                continue
            pred_all = rd[mode]          # (T, 512, 512)
            group    = decode_group(pd['group'])

            if eval_at == 'es':
                t   = int(pd['es_idx'])
                ref = pd['es_mask']
            else:
                t   = int(pd['ed_idx'])
                ref = pd['ed_mask']

            if t >= pred_all.shape[0]:
                continue
            pred_frame = pred_all[t]

            for cls in [1, 2, 3]:
                slice_dices[cls].append(dice_np(pred_frame, ref, cls))

        if not any(slice_dices[c] for c in [1, 2, 3]):
            continue

        rv  = float(np.mean(slice_dices[1])) if slice_dices[1] else 0.0
        myo = float(np.mean(slice_dices[2])) if slice_dices[2] else 0.0
        lv  = float(np.mean(slice_dices[3])) if slice_dices[3] else 0.0
        results[pid] = {'RV': rv, 'Myo': myo, 'LV': lv, 'group': group}
    return results


def summarise(d):
    rv  = [v['RV']  for v in d.values()]
    myo = [v['Myo'] for v in d.values()]
    lv  = [v['LV']  for v in d.values()]
    return {
        'RV':  (np.mean(rv),  np.std(rv)),
        'Myo': (np.mean(myo), np.std(myo)),
        'LV':  (np.mean(lv),  np.std(lv)),
    }


# ── Figure 1: Qualitative gallery — 4 rows × 6 frames ────────────────────────
def fig1_qualitative(fig_dir, db, prep_dir, medsam2_dir, sam2_dir, unet_ckpt):
    pid = 37   # DCM val patient
    nii4d_path = os.path.join(db, f'patient{pid:03d}', f'patient{pid:03d}_4d.nii.gz')
    if not os.path.exists(nii4d_path):
        print("Fig1: 4D NIfTI not found, skipping"); return

    vol4d = nib.load(nii4d_path).get_fdata(dtype=np.float32)   # (H, W, Z, T)
    T     = vol4d.shape[3]

    prep_npzs = sorted(glob(os.path.join(prep_dir, f'patient{pid:03d}_slice*.npz')))
    if not prep_npzs:
        print("Fig1: no prep npzs, skipping"); return
    mid_idx = len(prep_npzs) // 2
    stem    = os.path.basename(prep_npzs[mid_idx]).replace('.npz', '')
    z       = int(stem.split('_slice')[1])

    pd_data  = np.load(prep_npzs[mid_idx], allow_pickle=True)
    ed_idx   = int(pd_data['ed_idx'])
    es_idx   = int(pd_data['es_idx'])
    ed_mask  = pd_data['ed_mask']   # (512, 512)
    es_mask  = pd_data['es_mask']

    # MedSAM2 bidir result
    ms2_path = os.path.join(medsam2_dir, f'{stem}.npz')
    if not os.path.exists(ms2_path):
        print(f"Fig1: MedSAM2 result not found for {stem}, skipping"); return
    pred_bidir = np.load(ms2_path, allow_pickle=True)['bidir']   # (T, 512, 512)

    # SAM2 result (forward from ED)
    sam2_path = os.path.join(sam2_dir, f'{stem}.npz')
    sam2_avail = os.path.exists(sam2_path)
    pred_sam2  = np.load(sam2_path, allow_pickle=True)['ed_pred'] if sam2_avail else None

    # U-Net model
    unet_avail = os.path.exists(unet_ckpt)
    unet_model = None
    device = 'cpu'
    if unet_avail:
        try:
            import torch
            sys.path.insert(0, '/scratch/gautschi/li4533/MIUA_2026/pytorch-unet')
            from unet import UNet   # noqa: E402
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            unet_model = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
            unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device))
            unet_model.eval()
        except Exception as e:
            print(f"Fig1: U-Net load failed ({e}), skipping U-Net row")
            unet_model = None

    frame_indices = np.linspace(0, T - 1, 6, dtype=int)

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    row_labels = ['Ground Truth', 'MedSAM2 (Dual-anchored)', 'SAM2 (ED-anchored)', 'U-Net']

    for col, t in enumerate(frame_indices):
        # ── Shared: raw MRI frame resized to 512 ──
        frame_raw = vol4d[:, :, z, t]
        p2, p98   = np.percentile(frame_raw, 2), np.percentile(frame_raw, 98)
        frame_u8  = np.clip((frame_raw - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
        frame_512 = np.array(Image.fromarray(frame_u8).resize((512, 512), Image.BILINEAR))
        t_label   = 'ED' if t == ed_idx else ('ES' if t == es_idx else f't={t}')

        # Row 0: GT at ED/ES, blank elsewhere
        ax = axes[0, col]
        if t == ed_idx:
            gt_disp = np.zeros((512, 512, 3), dtype=np.float32)
            for cls_id, color in LABEL_COLOR.items():
                m = (ed_mask == cls_id)
                for c, cv in enumerate(color): gt_disp[:, :, c][m] = cv
            ax.imshow(gt_disp)
            ax.set_title(f'GT (ED)', color='black', fontsize=9)
        elif t == es_idx:
            gt_disp = np.zeros((512, 512, 3), dtype=np.float32)
            for cls_id, color in LABEL_COLOR.items():
                m = (es_mask == cls_id)
                for c, cv in enumerate(color): gt_disp[:, :, c][m] = cv
            ax.imshow(gt_disp)
            ax.set_title(f'GT (ES)', color='black', fontsize=9)
        else:
            ax.set_facecolor('white')
            ax.set_title(t_label, color='black', fontsize=9)
        ax.axis('off')

        # Row 1: MedSAM2 dual-anchored overlay
        ax = axes[1, col]
        ax.imshow(overlay_mask(frame_512, pred_bidir[t]))
        ax.set_title(t_label, color='black', fontsize=9)
        ax.axis('off')

        # Row 2: SAM2 ED-anchored overlay
        ax = axes[2, col]
        if sam2_avail and pred_sam2 is not None:
            ax.imshow(overlay_mask(frame_512, pred_sam2[t]))
        else:
            ax.set_facecolor('white')
        ax.set_title(t_label, color='black', fontsize=9)
        ax.axis('off')

        # Row 3: U-Net prediction overlay
        ax = axes[3, col]
        if unet_model is not None:
            import torch
            # percentile-norm already done above; just resize to 256×256
            frame_256  = np.array(
                Image.fromarray(frame_u8).resize((256, 256), Image.BILINEAR),
                dtype=np.float32
            ) / 255.0
            inp = torch.tensor(frame_256[None, None], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_256 = unet_model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_512_unet = np.array(
                Image.fromarray(pred_256).resize((512, 512), Image.NEAREST)
            )
            ax.imshow(overlay_mask(frame_512, pred_512_unet))
        else:
            ax.set_facecolor('white')
        ax.set_title(t_label, color='black', fontsize=9)
        ax.axis('off')

    # Row labels: text box on left edge of each row's first cell
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, color='black', fontsize=10,
                                fontweight='bold', rotation=90, labelpad=5)

    # Legend
    patches = [mpatches.Patch(color=LABEL_COLOR[1], label='RV'),
               mpatches.Patch(color=LABEL_COLOR[2], label='Myo'),
               mpatches.Patch(color=LABEL_COLOR[3], label='LV')]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11,
               facecolor='white', labelcolor='black', framealpha=0.8)

    plt.suptitle(f'Cardiac Segmentation Gallery — Patient {pid:03d} DCM (mid-slice z={z})',
                 color='black', fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(fig_dir, 'fig1_qualitative.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: Box plots ───────────────────────────────────────────────────────
def fig2_boxplot(fig_dir, raw_results):
    """
    raw_results: ordered dict of method_name -> {pid: {RV, Myo, LV, ...}}
    Plots 3 subplots (RV, Myo, LV) with box plots.
    """
    method_names = list(raw_results.keys())
    metrics      = ['RV', 'Myo', 'LV']
    colors       = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0', '#F44336']

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.suptitle('Dice Score Comparison — ACDC Val Set (Stratified, 4 per group)',
                 fontsize=13)

    for ax, metric in zip(axes, metrics):
        # Collect per-patient values (only patients present in all methods)
        all_pids = set(VAL_IDS)
        for d in raw_results.values():
            all_pids &= set(d.keys())
        common_pids = sorted(all_pids)

        data_per_method = []
        for name in method_names:
            vals = [raw_results[name][pid][metric] for pid in common_pids
                    if pid in raw_results[name]]
            data_per_method.append(vals)

        bp = ax.boxplot(data_per_method, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, len(method_names) + 1))
        ax.set_xticklabels(method_names, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Dice Score', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig2_boxplot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: Per-pathology heatmap (MedSAM2 dual-anchored) ──────────────────
def fig3_pathology_heatmap(fig_dir, medsam2_bidir_results):
    data = np.zeros((len(GROUPS), 3))
    for i, grp in enumerate(GROUPS):
        per_grp = [v for v in medsam2_bidir_results.values() if v.get('group') == grp]
        if not per_grp:
            continue
        data[i, 0] = np.mean([v['RV']  for v in per_grp])
        data[i, 1] = np.mean([v['Myo'] for v in per_grp])
        data[i, 2] = np.mean([v['LV']  for v in per_grp])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['RV', 'Myo', 'LV'], fontsize=12)
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS, fontsize=12)
    for i in range(len(GROUPS)):
        for j in range(3):
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                    fontsize=11, color='black' if data[i,j] > 0.65 else 'white')
    plt.colorbar(im, ax=ax, label='Dice Score')
    ax.set_title('MedSAM2 (Dual-anchored) Dice per Pathology', fontsize=12)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig3_pathology_heat.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Figure 4: LV Time-Volume Curves (CENTREPIECE) ────────────────────────────
def fig4_timevolume(fig_dir, db, medsam2_dir):
    """
    For every patient (all 100): load MedSAM2 bidir predictions for all slices,
    compute LV volume (mL) at each time frame, plot per-pathology mean ± std.
    """
    prep_dir = os.path.normpath(os.path.join(os.path.dirname(medsam2_dir), '..', 'preprocessed'))

    group_curves = {g: [] for g in GROUPS}

    for pid in tqdm(range(1, 101), desc='Computing time-volume curves'):
        pdir = os.path.join(db, f'patient{pid:03d}')
        cfg  = os.path.join(pdir, 'Info.cfg')
        if not os.path.exists(cfg):
            continue
        info  = parse_info_cfg(cfg)
        group = info.get('Group', 'UNK')
        if group not in GROUPS:
            continue

        prep_npzs = sorted(glob(os.path.join(prep_dir,    f'patient{pid:03d}_slice*.npz')))
        res_npzs  = sorted(glob(os.path.join(medsam2_dir, f'patient{pid:03d}_slice*.npz')))
        if not prep_npzs or not res_npzs:
            continue

        d0     = np.load(prep_npzs[0], allow_pickle=True)
        T      = np.load(res_npzs[0],  allow_pickle=True)['bidir'].shape[0]
        ed_idx = int(d0['ed_idx'])
        es_idx = int(d0['es_idx'])
        pixdim = d0['pixdim'].astype(np.float64)
        orig_H = int(d0['orig_H'])
        orig_W = int(d0['orig_W'])

        scale     = (orig_H / 512.0) * (orig_W / 512.0)
        voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale

        lv_voxels = np.zeros(T, dtype=np.float64)
        for r_path in res_npzs:
            rd = np.load(r_path, allow_pickle=True)
            if 'bidir' not in rd:
                continue
            bidir = rd['bidir']
            for t in range(min(T, bidir.shape[0])):
                lv_voxels[t] += (bidir[t] == 3).sum()

        lv_vol_ml = lv_voxels * voxel_mm3 / 1000.0

        phases = np.arange(T) / T
        group_curves[group].append((phases, lv_vol_ml, ed_idx, es_idx))

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(18, 4), sharey=False)
    for ax, grp in zip(axes, GROUPS):
        curves = group_curves[grp]
        if not curves:
            ax.set_title(grp); continue

        common_phase = np.linspace(0, 1, 100, endpoint=False)
        interp_vols  = []
        for phases, vols, ed_i, es_i in curves:
            shift      = ed_i
            phases_rot = (np.arange(len(phases)) - shift) / len(phases) % 1.0
            order      = np.argsort(phases_rot)
            ph_s       = phases_rot[order]
            vl_s       = vols[order]
            interp_vols.append(np.interp(common_phase, ph_s, vl_s,
                                          left=vl_s[0], right=vl_s[-1]))

        arr  = np.array(interp_vols)
        mean = arr.mean(0)
        std  = arr.std(0)

        ax.fill_between(common_phase * 100, mean - std, mean + std,
                        alpha=0.25, color=GROUP_COLOR[grp])
        ax.plot(common_phase * 100, mean, color=GROUP_COLOR[grp], linewidth=2, label=grp)

        avg_es_phase = np.mean([es_i / len(c[0]) for c in curves]) * 100
        ax.axvline(0,            color='blue', linestyle='--', linewidth=1, alpha=0.7, label='ED')
        ax.axvline(avg_es_phase, color='red',  linestyle='--', linewidth=1, alpha=0.7, label='ES')

        ax.set_title(grp, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cardiac Phase (%)', fontsize=10)
        ax.set_ylabel('LV Volume (mL)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.text(0.97, 0.97, f'n={len(curves)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

    plt.suptitle('LV Time-Volume Curves from MedSAM2 Full-Cycle Propagation\n'
                 '(All 100 Patients, mean ± std)', fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig4_timevolume.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.fig_dir, exist_ok=True)
    prep_dir    = os.path.normpath(os.path.join(args.results_dir, '..', 'preprocessed'))
    medsam2_dir = os.path.join(args.results_dir, 'medsam2')
    sam2_dir    = os.path.join(args.results_dir, 'sam2')
    unet_json   = os.path.join(args.results_dir, 'unet', 'results.json')
    unet_ckpt   = os.path.join(args.results_dir, 'unet', 'best_model.pth')

    print("\n══ Computing Dice for all methods ══")

    ms2_ed   = compute_method_dice(medsam2_dir, mode='ed_pred', eval_at='es')
    ms2_es   = compute_method_dice(medsam2_dir, mode='es_pred', eval_at='ed')
    ms2_dual = compute_method_dice(medsam2_dir, mode='bidir',   eval_at='es')
    sam2_ed  = compute_method_dice(sam2_dir,    mode='ed_pred', eval_at='es')

    unet_res = {}
    if os.path.exists(unet_json):
        with open(unet_json) as f:
            raw = json.load(f)
        for pid_str, v in raw.items():
            unet_res[int(pid_str)] = v

    # Build ordered raw-results dict for fig2
    raw_results = {}
    if sam2_ed:   raw_results['SAM2 (ED-anchored)']           = sam2_ed
    if ms2_ed:    raw_results['MedSAM2 (ED-anchored)']        = ms2_ed
    if ms2_es:    raw_results['MedSAM2 (ES-anchored)']        = ms2_es
    if ms2_dual:  raw_results['MedSAM2 (Dual-anchored)']      = ms2_dual
    if unet_res:  raw_results['U-Net (supervised)']           = unet_res

    # Summarise for table
    all_results = {name: summarise(d) for name, d in raw_results.items()}

    # Print Table 1
    print("\n── Table 1: Dice at evaluation frame ──")
    header = f"{'Method':<25} {'RV':>12} {'Myo':>12} {'LV':>12}"
    print(header); print('─' * len(header))
    for method, s in all_results.items():
        row = f"{method:<25}"
        for m in ['RV', 'Myo', 'LV']:
            row += f"  {s[m][0]:.3f}±{s[m][1]:.3f}"
        print(row)

    # Save Table 1 CSV
    import csv
    csv_path = os.path.join(args.results_dir, 'table1_dice.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'RV_mean', 'RV_std', 'Myo_mean', 'Myo_std', 'LV_mean', 'LV_std'])
        for method, s in all_results.items():
            w.writerow([method] + [f'{s[m][i]:.4f}' for m in ['RV', 'Myo', 'LV'] for i in [0, 1]])
    print(f"Saved {csv_path}")

    # ── Generate figures ──
    print("\n══ Generating figures ══")
    fig1_qualitative(args.fig_dir, args.db, prep_dir, medsam2_dir, sam2_dir, unet_ckpt)
    if raw_results:
        fig2_boxplot(args.fig_dir, raw_results)
    if ms2_dual:
        fig3_pathology_heatmap(args.fig_dir, ms2_dual)
    fig4_timevolume(args.fig_dir, args.db, medsam2_dir)

    print(f"\nAll figures saved to {args.fig_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--db',          default=DB_PATH)
    parser.add_argument('--fig_dir',     default=FIG_DIR)
    main(parser.parse_args())
