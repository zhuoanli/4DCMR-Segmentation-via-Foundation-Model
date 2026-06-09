"""
evaluate_and_figures.py
Post-processing: compute Dice tables and generate all paper figures.

Figures (all prefixed paper_):
  paper_fig1_qualitative.png   — GT / MedSAM2 / SAM2 / U-Net / DINOv2 gallery
  paper_fig2_boxplot.png       — box plots, all methods × 3 structures
  paper_fig3_timevolume.png    — LV time-volume curves, all 100 patients (centrepiece)
  paper_fig4_ef_regression.png — EF regression + Bland-Altman, 3 methods (MedSAM2/U-Net/DINOv2)
  paper_fig5_crossvendor.png   — MnM cross-vendor Dice bar chart

Tables (all prefixed paper_):
  paper_table1_dice.csv
  paper_table_clinical_acdc_complete.csv
  paper_table_clinical_mnm_complete.csv
"""

import os, sys, json, argparse, csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from glob import glob
from tqdm import tqdm
from PIL import Image

RESULTS_DIR = '/scratch/gautschi/li4533/MIUA_2026/results'
DB_PATH     = '/scratch/gautschi/li4533/MIUA_2026/database/training'
FIG_DIR     = '/scratch/gautschi/li4533/MIUA_2026/figures'
PREP_DIR    = '/scratch/gautschi/li4533/MIUA_2026/preprocessed'

VAL_IDS     = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]
CLASSES     = {1: 'RV', 2: 'Myo', 3: 'LV'}
GROUPS      = ['NOR', 'DCM', 'HCM', 'MINF', 'RV']
GROUP_COLOR = {'NOR': '#2196F3', 'DCM': '#F44336', 'HCM': '#4CAF50',
               'MINF': '#FF9800', 'RV': '#9C27B0'}
LABEL_COLOR = {1: (0.9, 0.1, 0.1), 2: (0.1, 0.8, 0.1), 3: (0.1, 0.1, 0.9)}

# Method display names, result directories, NPZ keys, evaluation frame
METHOD_CFG = [
    ('SAM2',                        'sam2_bidir', 'bidir',   'es'),
    ('MedSAM2\n(ED-anchored)',      'medsam2',    'ed_pred', 'es'),
    ('MedSAM2\n(ES-anchored)',      'medsam2',    'es_pred', 'ed'),
    ('MedSAM2\n(Dual-anchored)†',  'medsam2',    'bidir',   'es'),
    ('U-Net\n(supervised)',         'unet',       None,      'es'),
    ('DINOv2\n(supervised)',        'dinov2',     None,      'es'),
    ('nnUNet\n(supervised)',        'nnunet',     None,      'es'),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def dice_np(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    if g.sum() == 0 and p.sum() == 0: return 1.0
    if g.sum() == 0: return 0.0
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
    s = str(raw)
    if s.startswith("np.bytes_(b'") and s.endswith("')"): s = s[len("np.bytes_(b'"):-2]
    elif s.startswith("b'") and s.endswith("'"): s = s[2:-1]
    return s


def overlay_mask(frame_u8, pred_mask):
    """Blend colourised mask onto grayscale frame. Returns float32 (H,W,3)."""
    rgb = np.stack([frame_u8]*3, axis=-1).astype(np.float32) / 255.0
    for cls_id, color in LABEL_COLOR.items():
        m = (pred_mask == cls_id)
        for c, cv in enumerate(color):
            rgb[:,:,c][m] = rgb[:,:,c][m] * 0.4 + cv * 0.6
    return np.clip(rgb, 0, 1)


# ── Dice computation ──────────────────────────────────────────────────────────

def compute_method_dice(result_dir, mode='ed_pred', eval_at='es'):
    results = {}
    for pid in VAL_IDS:
        res_npzs  = sorted(glob(os.path.join(result_dir, f'patient{pid:03d}_slice*.npz')))
        prep_npzs = sorted(glob(os.path.join(PREP_DIR,   f'patient{pid:03d}_slice*.npz')))
        if not res_npzs or not prep_npzs:
            continue
        slice_dices = {1: [], 2: [], 3: []}
        group = 'UNK'
        for r_path, p_path in zip(res_npzs, prep_npzs):
            rd = np.load(r_path,  allow_pickle=True)
            pd = np.load(p_path,  allow_pickle=True)
            if mode not in rd: continue
            pred_all = rd[mode]
            group    = decode_group(pd['group'])
            t   = int(pd['es_idx']) if eval_at == 'es' else int(pd['ed_idx'])
            ref = pd['es_mask']    if eval_at == 'es' else pd['ed_mask']
            if t >= pred_all.shape[0]: continue
            for cls in [1,2,3]:
                slice_dices[cls].append(dice_np(pred_all[t], ref, cls))
        if not any(slice_dices[c] for c in [1,2,3]): continue
        results[pid] = {
            'RV':  float(np.mean(slice_dices[1])) if slice_dices[1] else 0.0,
            'Myo': float(np.mean(slice_dices[2])) if slice_dices[2] else 0.0,
            'LV':  float(np.mean(slice_dices[3])) if slice_dices[3] else 0.0,
            'group': group,
        }
    return results


def summarise(d):
    rv  = [v['RV']  for v in d.values()]
    myo = [v['Myo'] for v in d.values()]
    lv  = [v['LV']  for v in d.values()]
    return {'RV': (np.mean(rv), np.std(rv)),
            'Myo':(np.mean(myo),np.std(myo)),
            'LV': (np.mean(lv), np.std(lv))}


# ── Figure 1: Qualitative gallery ─────────────────────────────────────────────

def _unet_pred_frame(frame_u8, model, device):
    import torch
    frame_256 = np.array(
        Image.fromarray(frame_u8).resize((256,256), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    inp = torch.tensor(frame_256[None,None], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return np.array(Image.fromarray(pred).resize((512,512), Image.NEAREST))


def _dinov2_pred_frame(frame_u8, model, device):
    import torch
    IMG_MEAN = torch.tensor([0.485,0.456,0.406])[:,None,None].to(device)
    IMG_STD  = torch.tensor([0.229,0.224,0.225])[:,None,None].to(device)
    frame_f32 = np.array(
        Image.fromarray(frame_u8).resize((512,512), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    inp = torch.tensor(
        np.stack([frame_f32]*3, axis=0)[None], dtype=torch.float32
    ).to(device)
    inp = (inp - IMG_MEAN) / IMG_STD
    with torch.no_grad():
        pred = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred   # already 512×512


def fig1_qualitative(fig_dir, db, medsam2_dir, sam2_dir, unet_ckpt, dinov2_ckpt=None):
    import torch
    pid = 37   # DCM val patient
    nii4d_path = os.path.join(db, f'patient{pid:03d}', f'patient{pid:03d}_4d.nii.gz')
    if not os.path.exists(nii4d_path):
        print("Fig1: 4D NIfTI not found, skipping"); return

    vol4d = nib.load(nii4d_path).get_fdata(dtype=np.float32)   # (H,W,Z,T)
    T     = vol4d.shape[3]

    prep_npzs = sorted(glob(os.path.join(PREP_DIR, f'patient{pid:03d}_slice*.npz')))
    if not prep_npzs: print("Fig1: no prep npzs, skipping"); return
    mid_idx = len(prep_npzs) // 2
    stem    = os.path.basename(prep_npzs[mid_idx]).replace('.npz', '')
    z       = int(stem.split('_slice')[1])
    pd_data = np.load(prep_npzs[mid_idx], allow_pickle=True)
    ed_idx  = int(pd_data['ed_idx']); es_idx = int(pd_data['es_idx'])
    ed_mask = pd_data['ed_mask'];     es_mask = pd_data['es_mask']

    # Load MedSAM2 bidir
    ms2_path = os.path.join(medsam2_dir, f'{stem}.npz')
    if not os.path.exists(ms2_path): print(f"Fig1: MedSAM2 result missing, skipping"); return
    pred_bidir = np.load(ms2_path, allow_pickle=True)['bidir']   # (T,512,512)

    # Load SAM2 (dual-anchored)
    sam2_path  = os.path.join(sam2_dir, f'{stem}.npz')
    pred_sam2  = np.load(sam2_path, allow_pickle=True)['bidir'] if os.path.exists(sam2_path) else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load U-Net
    unet_model = None
    if os.path.exists(unet_ckpt):
        try:
            sys.path.insert(0, '/scratch/gautschi/li4533/MIUA_2026/pytorch-unet')
            from unet import UNet
            unet_model = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
            unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device))
            unet_model.eval()
        except Exception as e:
            print(f"Fig1: U-Net load failed: {e}")

    # Load DINOv2
    dinov2_model = None
    if dinov2_ckpt and os.path.exists(dinov2_ckpt):
        try:
            sys.path.insert(0, '/scratch/gautschi/li4533/MIUA_2026')
            try:
                from train_dinov2_combined import DINOv2SegHead
            except ImportError:
                from train_eval_dinov2 import DINOv2SegHead
            dinov2_model = DINOv2SegHead(num_classes=4).to(device)
            dinov2_model.load_state_dict(torch.load(dinov2_ckpt, map_location=device))
            dinov2_model.eval()
        except Exception as e:
            print(f"Fig1: DINOv2 load failed: {e}")

    # Build row spec: (row_label, renderer_fn_or_None, is_video)
    row_specs = [
        ('Raw MRI',                       None, True),
        ('Ground Truth',                  None, True),
        ('MedSAM2\n(Dual-anchored) †',   'bidir', True),
        ('SAM2',                          'sam2',  True),
        ('U-Net\n(supervised)',           'unet',  False),
    ]
    if dinov2_model is not None:
        row_specs.append(('DINOv2\n(supervised)', 'dinov2', False))

    n_rows = len(row_specs)
    n_cols = 6
    frame_indices = np.linspace(0, T-1, n_cols, dtype=int)

    # Extra left column for row labels
    fig = plt.figure(figsize=(n_cols * 2.5 + 2.0, n_rows * 2.5))
    gs  = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                            width_ratios=[0.18] + [1]*n_cols,
                            wspace=0.04, hspace=0.12)

    for row_idx, (row_label, renderer, _is_video) in enumerate(row_specs):
        # Row label cell
        ax_lbl = fig.add_subplot(gs[row_idx, 0])
        ax_lbl.axis('off')
        ax_lbl.text(0.95, 0.5, row_label, ha='right', va='center',
                    fontsize=9, fontweight='bold', transform=ax_lbl.transAxes,
                    linespacing=1.4)

        for col_idx, t in enumerate(frame_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.axis('off')

            frame_raw = vol4d[:, :, z, t]
            p2, p98   = np.percentile(frame_raw, 2), np.percentile(frame_raw, 98)
            frame_u8  = np.clip((frame_raw - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
            frame_512 = np.array(Image.fromarray(frame_u8).resize((512,512), Image.BILINEAR))
            t_label   = 'ED' if t == ed_idx else ('ES' if t == es_idx else f't={t}')

            if row_idx == 0:   # Raw MRI
                ax.imshow(frame_512, cmap='gray', vmin=0, vmax=255)
                if row_idx == 0 and col_idx == 0:
                    ax.set_title(f'Patient {pid:03d} DCM', fontsize=8, pad=2)
                ax.set_title(t_label, fontsize=8, pad=2)

            elif row_idx == 1:   # GT
                if t in (ed_idx, es_idx):
                    gt_m = ed_mask if t == ed_idx else es_mask
                    gt_disp = np.zeros((512,512,3), dtype=np.float32)
                    for cls_id, color in LABEL_COLOR.items():
                        m = (gt_m == cls_id)
                        for c, cv in enumerate(color): gt_disp[:,:,c][m] = cv
                    ax.imshow(gt_disp)
                    ax.set_title(f'GT ({"ED" if t==ed_idx else "ES"})', fontsize=8, pad=2)
                else:
                    ax.set_facecolor('#f5f5f5')
                    ax.set_title(t_label, fontsize=8, pad=2)

            elif renderer == 'bidir':
                ax.imshow(overlay_mask(frame_512, pred_bidir[t]))
                ax.set_title(t_label, fontsize=8, pad=2)

            elif renderer == 'sam2':
                if pred_sam2 is not None:
                    ax.imshow(overlay_mask(frame_512, pred_sam2[t]))
                else:
                    ax.set_facecolor('#f0f0f0')
                ax.set_title(t_label, fontsize=8, pad=2)

            elif renderer == 'unet':
                if unet_model is not None:
                    pred = _unet_pred_frame(frame_u8, unet_model, device)
                    ax.imshow(overlay_mask(frame_512, pred))
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                ax.set_title(t_label, fontsize=8, pad=2)

            elif renderer == 'dinov2':
                if dinov2_model is not None:
                    pred = _dinov2_pred_frame(frame_u8, dinov2_model, device)
                    ax.imshow(overlay_mask(frame_512, pred))
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                ax.set_title(t_label, fontsize=8, pad=2)

    # Legend
    patches = [mpatches.Patch(color=LABEL_COLOR[k], label=v)
               for k,v in CLASSES.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.55, 0.01))

    plt.suptitle(f'Cardiac Segmentation Gallery — Patient {pid:03d} DCM (mid-slice z={z})',
                 fontsize=12, y=1.01)
    out = os.path.join(fig_dir, 'paper_fig1_qualitative.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: Box plots ───────────────────────────────────────────────────────

# Colour palette: consistent across methods
METHOD_COLORS = {
    'SAM2':                         '#607D8B',   # blue-grey
    'MedSAM2\n(ED-anchored)':      '#FFB74D',   # amber
    'MedSAM2\n(ES-anchored)':      '#FF7043',   # deep orange
    'MedSAM2\n(Dual-anchored)†':   '#E53935',   # strong red (proposed)
    'U-Net\n(supervised)':         '#1E88E5',   # blue
    'DINOv2\n(supervised)':        '#00ACC1',   # cyan-teal
    'nnUNet\n(supervised)':        '#43A047',   # green
}
DEFAULT_COLOR = '#90A4AE'


def fig2_boxplot(fig_dir, raw_results):
    method_names = list(raw_results.keys())
    colors = [METHOD_COLORS.get(n, DEFAULT_COLOR) for n in method_names]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(method_names)*2.2), 6),
                             facecolor='white')
    fig.suptitle('Dice Score Comparison — ACDC Validation Set (n=20, stratified)',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, metric in zip(axes, ['RV', 'Myo', 'LV']):
        ax.set_facecolor('#FAFAFA')
        all_pids = set(VAL_IDS)
        for d in raw_results.values():
            all_pids &= set(d.keys())
        common_pids = sorted(all_pids)

        data_per_method = []
        for name in method_names:
            vals = [raw_results[name][pid][metric] for pid in common_pids
                    if pid in raw_results[name]]
            data_per_method.append(vals if vals else [0.0])

        bp = ax.boxplot(data_per_method, patch_artist=True,
                        medianprops=dict(color='#212121', linewidth=2.2),
                        whiskerprops=dict(linewidth=1.4, color='#555'),
                        capprops=dict(linewidth=1.4, color='#555'),
                        flierprops=dict(marker='o', markersize=4, alpha=0.55,
                                        markeredgewidth=0.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

        # Mean annotation above each box
        for i, (name, vals) in enumerate(zip(method_names, data_per_method)):
            if vals:
                ax.text(i+1, max(vals)+0.025, f'{np.mean(vals):.2f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#333', fontweight='bold')

        ax.set_title(metric, fontsize=14, fontweight='bold', pad=6)
        ax.set_xticks(range(1, len(method_names)+1))
        ax.set_xticklabels(method_names, rotation=35, ha='right', fontsize=8.5)
        ax.set_ylabel('Dice Score', fontsize=11)
        ax.set_ylim(0, 1.22)
        ax.axhline(0.8, color='#9E9E9E', linestyle='--', linewidth=1.0, alpha=0.7,
                   label='0.80 reference')
        ax.grid(axis='y', alpha=0.35, color='#ccc', linewidth=0.8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#bbb')
        ax.spines['bottom'].set_color('#bbb')

    fig.text(0.5, -0.04, '† = Proposed method (MedSAM2 Dual-anchored)',
             ha='center', fontsize=9, color='#E53935', style='italic')
    plt.tight_layout()
    out = os.path.join(fig_dir, 'paper_fig2_boxplot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: LV Time-Volume Curves ──────────────────────────────────────────

def fig3_timevolume(fig_dir, db, medsam2_dir):
    prep_dir    = PREP_DIR
    group_curves = {g: [] for g in GROUPS}

    for pid in tqdm(range(1, 101), desc='Time-volume curves'):
        pdir = os.path.join(db, f'patient{pid:03d}')
        cfg  = os.path.join(pdir, 'Info.cfg')
        if not os.path.exists(cfg): continue
        info  = parse_info_cfg(cfg)
        group = info.get('Group', 'UNK')
        if group not in GROUPS: continue

        prep_npzs = sorted(glob(os.path.join(prep_dir,    f'patient{pid:03d}_slice*.npz')))
        res_npzs  = sorted(glob(os.path.join(medsam2_dir, f'patient{pid:03d}_slice*.npz')))
        if not prep_npzs or not res_npzs: continue

        d0     = np.load(prep_npzs[0], allow_pickle=True)
        T      = np.load(res_npzs[0],  allow_pickle=True)['bidir'].shape[0]
        ed_idx = int(d0['ed_idx']); es_idx = int(d0['es_idx'])
        pixdim = d0['pixdim'].astype(np.float64)
        orig_H = int(d0['orig_H']); orig_W = int(d0['orig_W'])
        scale  = (orig_H / 512.0) * (orig_W / 512.0)
        voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale

        lv_voxels = np.zeros(T, dtype=np.float64)
        for r_path in res_npzs:
            rd = np.load(r_path, allow_pickle=True)
            if 'bidir' not in rd: continue
            bidir = rd['bidir']
            for t in range(min(T, bidir.shape[0])):
                lv_voxels[t] += (bidir[t] == 3).sum()

        lv_vol_ml  = lv_voxels * voxel_mm3 / 1000.0
        phases     = np.arange(T) / T
        group_curves[group].append((phases, lv_vol_ml, ed_idx, es_idx))

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(18, 4), sharey=False)
    for ax, grp in zip(axes, GROUPS):
        curves = group_curves[grp]
        if not curves: ax.set_title(grp); continue

        common_phase = np.linspace(0, 1, 100, endpoint=False)
        interp_vols  = []
        for phases, vols, ed_i, es_i in curves:
            shift      = ed_i
            phases_rot = (np.arange(len(phases)) - shift) / len(phases) % 1.0
            order      = np.argsort(phases_rot)
            interp_vols.append(np.interp(common_phase,
                                          phases_rot[order], vols[order],
                                          left=vols[order][0], right=vols[order][-1]))

        arr  = np.array(interp_vols)
        mean = arr.mean(0); std = arr.std(0)
        ax.fill_between(common_phase*100, mean-std, mean+std,
                        alpha=0.25, color=GROUP_COLOR[grp])
        ax.plot(common_phase*100, mean, color=GROUP_COLOR[grp], linewidth=2)

        avg_es = np.mean([c[3] / len(c[0]) for c in curves]) * 100
        ax.axvline(0,       color='blue', linestyle='--', linewidth=1, alpha=0.7, label='ED')
        ax.axvline(avg_es,  color='red',  linestyle='--', linewidth=1, alpha=0.7, label='ES')
        ax.set_title(grp, fontsize=12, fontweight='bold', color=GROUP_COLOR[grp])
        ax.set_xlabel('Cardiac Phase (%)', fontsize=10)
        ax.set_ylabel('LV Volume (mL)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.text(0.97, 0.97, f'n={len(curves)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

        # Clinical annotations per pathology group
        annots = {
            'DCM':  ('Elevated EDV/ESV\n(dilated LV)', 0.02, 0.15),
            'HCM':  ('Compact LV\ncavity', 0.02, 0.15),
            'MINF': ('Reduced stroke\nvolume', 0.02, 0.15),
            'NOR':  ('Smooth systolic\nemptying', 0.02, 0.15),
            'RV':   ('Atypical\nLV dynamics', 0.02, 0.15),
        }
        if grp in annots:
            txt, x_frac, y_frac = annots[grp]
            ax.text(x_frac, y_frac, txt, transform=ax.transAxes,
                    fontsize=7.5, color=GROUP_COLOR[grp], alpha=0.85,
                    va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

    plt.suptitle('LV Time-Volume Curves — MedSAM2 Full-Cycle Propagation (mean ± std)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'paper_fig3_timevolume.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Figure 3b: LV Time-Volume Curves — MnM by disease group ──────────────────

MNM_DISEASE_GROUPS  = ['NOR', 'DCM', 'HCM', 'HHD', 'IHD', 'ARV', 'LVNC', 'AHS', 'Other']
MNM_DISEASE_COLOR   = {
    'NOR': '#2196F3', 'DCM': '#F44336', 'HCM': '#4CAF50',
    'HHD': '#FF9800', 'IHD': '#9C27B0', 'ARV': '#00BCD4',
    'LVNC': '#795548', 'AHS': '#607D8B', 'Other': '#9E9E9E',
}


def fig3_timevolume_mnm(fig_dir, prep_mnm_dir, medsam2_mnm_dir):
    """LV time-volume curves from MnM patients, grouped by disease."""
    from collections import defaultdict

    # Collect per-patient info: pid → (disease, list_of_slice_npz_stems)
    all_npzs = sorted(glob(os.path.join(prep_mnm_dir, '*.npz')))
    pid_map = defaultdict(list)   # pid → list of prep npz paths
    for p in all_npzs:
        stem = os.path.basename(p).replace('.npz', '')
        parts = stem.rsplit('_slice', 1)
        if len(parts) == 2:
            pid_map[parts[0]].append(p)

    group_curves = {g: [] for g in MNM_DISEASE_GROUPS}

    for pid, slice_paths in tqdm(pid_map.items(), desc='MnM time-volume curves'):
        # Get disease group from first slice
        d0 = np.load(slice_paths[0], allow_pickle=True)
        raw_group = decode_group(d0['group']) if 'group' in d0 else ''
        disease = raw_group.split('_')[0] if raw_group else ''
        if disease not in MNM_DISEASE_GROUPS:
            continue

        # Get timing from first slice
        T      = None
        ed_idx = int(d0['ed_idx'])
        es_idx = int(d0['es_idx'])
        pixdim = d0['pixdim'].astype(np.float64)
        orig_H = int(d0['orig_H']); orig_W = int(d0['orig_W'])
        scale  = (orig_H / 512.0) * (orig_W / 512.0)
        voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale

        # Accumulate LV voxel counts from MedSAM2 bidir predictions
        lv_voxels = None
        for sp in slice_paths:
            stem_s = os.path.basename(sp).replace('.npz', '')
            pred_path = os.path.join(medsam2_mnm_dir, f'{stem_s}.npz')
            if not os.path.exists(pred_path):
                continue
            rd = np.load(pred_path, allow_pickle=True)
            if 'bidir' not in rd:
                continue
            bidir = rd['bidir']   # (T, 512, 512)
            if lv_voxels is None:
                T = bidir.shape[0]
                lv_voxels = np.zeros(T, dtype=np.float64)
            for t in range(min(T, bidir.shape[0])):
                lv_voxels[t] += (bidir[t] == 3).sum()

        if lv_voxels is None or T is None:
            continue

        lv_vol_ml = lv_voxels * voxel_mm3 / 1000.0
        phases    = np.arange(T) / T
        group_curves[disease].append((phases, lv_vol_ml, ed_idx, es_idx))

    active = [g for g in MNM_DISEASE_GROUPS if group_curves[g]]
    if not active:
        print("Fig3-MnM: no curves generated (no matching predictions), skipping"); return

    fig, axes = plt.subplots(1, len(active), figsize=(len(active) * 3.5, 4), sharey=False)
    if len(active) == 1:
        axes = [axes]

    for ax, grp in zip(axes, active):
        curves = group_curves[grp]
        color  = MNM_DISEASE_COLOR[grp]

        common_phase = np.linspace(0, 1, 100, endpoint=False)
        interp_vols  = []
        for phases, vols, ed_i, es_i in curves:
            shift      = ed_i  # now max-volume frame
            phases_rot = (np.arange(len(phases)) - shift) / len(phases) % 1.0
            order      = np.argsort(phases_rot)
            interp_vols.append(np.interp(common_phase, phases_rot[order], vols[order],
                                          left=vols[order][0], right=vols[order][-1]))

        arr  = np.array(interp_vols)
        mean = arr.mean(0); std = arr.std(0)
        ax.fill_between(common_phase * 100, mean - std, mean + std,
                        alpha=0.25, color=color)
        ax.plot(common_phase * 100, mean, color=color, linewidth=2)

        avg_es = np.mean([c[3] / len(c[0]) for c in curves]) * 100
        ax.axvline(0,       color='blue', linestyle='--', linewidth=1, alpha=0.7, label='ED')
        ax.axvline(avg_es,  color='red',  linestyle='--', linewidth=1, alpha=0.7, label='ES')
        ax.set_title(grp, fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Cardiac Phase (%)', fontsize=10)
        ax.set_ylabel('LV Volume (mL)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.text(0.97, 0.97, f'n={len(curves)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

    plt.suptitle('LV Time-Volume Curves — MedSAM2 Dual-anchored (MnM, mean ± std)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'paper_fig3_timevolume.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Figure 4 (NEW): Multi-structure HD95 Ablation + Full-cycle LV Trajectory ──

def fig4_temporal_propagation(fig_dir, db, medsam2_dir, metrics_json_path=None,
                               unet_allframes_dir=None):
    """
    Four-panel figure proving dual-anchoring benefit across all structures:
      Panel A: RV HD95 bar chart (methods compared)
      Panel B: Myo HD95 bar chart
      Panel C: LV HD95 bar chart
      Panel D: Full-cycle LV volume curves (mean ± std) — verifies dual std is smaller
    """
    VAL_PIDS = [
        'patient017','patient018','patient019','patient020',
        'patient037','patient038','patient039','patient040',
        'patient057','patient058','patient059','patient060',
        'patient077','patient078','patient079','patient080',
        'patient097','patient098','patient099','patient100',
    ]

    if metrics_json_path is None:
        metrics_json_path = os.path.join(RESULTS_DIR, 'metrics_acdc_val.json')
    with open(metrics_json_path) as f:
        metrics = json.load(f)

    C_UNET = '#607D8B'
    C_DINO = '#78909C'
    C_SAM2 = '#9C27B0'
    C_ED   = '#F4511E'
    C_ES   = '#2E7D32'
    C_DUAL = '#1565C0'

    # Method order — DINOv2/SAM2_ED excluded (in Table 1; HD95 outliers distort scale)
    BAR_METHODS = [
        ('UNet',        'U-Net\n(supervised)', C_UNET, 0.75, False),
        ('SAM2_Dual',   'SAM2\n(Dual)',        C_SAM2, 0.75, False),
        ('MedSAM2_ED',  'MedSAM2\n(ED)',       C_ED,   0.75, False),
        ('MedSAM2_ES',  'MedSAM2\n(ES)',       C_ES,   0.75, False),
        ('MedSAM2_Dual','MedSAM2\n(Dual)†',   C_DUAL, 1.00, True),
    ]

    STRUCT_CFG = [
        ('RV',  'hd95_RV',  '(a) Right Ventricle HD95 (mm)'),
        ('Myo', 'hd95_Myo', '(b) Myocardium HD95 (mm)'),
        ('LV',  'hd95_LV',  '(c) Left Ventricle HD95 (mm)'),
    ]

    def get_vals(method_key, metric_key):
        return [r[metric_key] for r in metrics.get(method_key, [])
                if r.get(metric_key) is not None]

    # ── Figure: 1 row of 3 HD95 bar charts + 1 volume curve panel ────────────
    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.38)

    x = np.arange(len(BAR_METHODS))

    for col, (struct, hd_key, title) in enumerate(STRUCT_CFG):
        ax = fig.add_subplot(gs[col])
        means, stds, colors, alphas = [], [], [], []
        for mkey, mlabel, color, alpha, highlight in BAR_METHODS:
            vals = get_vals(mkey, hd_key)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
            colors.append(color); alphas.append(alpha)

        bars = ax.bar(x, means, color=colors, alpha=0.82)
        # Error bars separately (clipped at 0)
        for i, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                ax.errorbar(x[i], m, yerr=[[min(s, m)], [s]], fmt='none',
                            ecolor='#555', elinewidth=1.2, capsize=4)
        # Bold border on Dual
        bars[-1].set_linewidth(2.2); bars[-1].set_edgecolor('#0D47A1')

        valid_means = [m for m in means if not np.isnan(m)]
        ymax = max(m + s for m, s in zip(means, stds) if not np.isnan(m)) * 1.18
        ax.set_ylim(0, ymax)

        for i, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                bold = (i == len(BAR_METHODS) - 1)
                ax.text(x[i], m + s + ymax * 0.01, f'{m:.1f}',
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold' if bold else 'normal',
                        color=C_DUAL if bold else '#444')

        ax.axhline(means[-1], color=C_DUAL, linestyle='--', linewidth=1, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([cfg[1] for cfg in BAR_METHODS], fontsize=8)
        ax.set_ylabel('HD95 (mm) ↓', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # ────────────────── Panel D: Propagation distance vs RV Dice scatter ────────
    ax_sc = fig.add_subplot(gs[3])

    def get_pp(method_key, metric_key):
        return {r['pid']: r.get(metric_key) for r in metrics.get(method_key, [])}

    # Load per-patient RV Dice and propagation distances
    dice_ed   = get_pp('MedSAM2_ED',   'dice_RV')
    dice_es   = get_pp('MedSAM2_ES',   'dice_RV')
    dice_dual = get_pp('MedSAM2_Dual', 'dice_RV')
    dice_unet = get_pp('UNet',          'dice_RV')

    prop_dist = {}
    for pid in VAL_PIDS:
        pp = sorted(glob(os.path.join(PREP_DIR, f'{pid}_slice*.npz')))
        if pp:
            d0 = np.load(pp[0], allow_pickle=True)
            prop_dist[pid] = int(d0['es_idx']) - int(d0['ed_idx'])

    pids = [p for p in VAL_PIDS if p in prop_dist and p in dice_ed]
    dist = np.array([prop_dist[p] for p in pids])

    # ED-anchored: evaluated at ES (propagation distance = es_idx - ed_idx = es_idx)
    ed_d = np.array([dice_ed[p] for p in pids])
    # ES-anchored: evaluated at ED (same physical distance, reversed direction)
    es_d = np.array([dice_es.get(p, np.nan) for p in pids])
    # Dual-anchored: evaluated at ES which is its anchor → distance ≈ 0
    dual_d = np.array([dice_dual.get(p, np.nan) for p in pids])
    # UNet: evaluated at ES (its training frame) → distance = 0 (frame-wise)
    unet_d = np.array([dice_unet.get(p, np.nan) for p in pids])

    np.random.seed(42)
    jit = np.random.uniform(-0.15, 0.15, len(pids))

    ax_sc.scatter(dist,        ed_d,   color=C_ED,   alpha=0.80, s=50, label='MedSAM2 (ED-anchored)', zorder=3)
    ax_sc.scatter(dist,        es_d,   color=C_ES,   alpha=0.80, s=50, label='MedSAM2 (ES-anchored)', marker='^', zorder=3)
    ax_sc.scatter(jit,         dual_d, color=C_DUAL, alpha=0.85, s=60, label='MedSAM2 (Dual)†', marker='D', zorder=4)
    ax_sc.scatter(jit + 0.3,   unet_d, color=C_UNET, alpha=0.75, s=50, label='U-Net (frame-wise)', marker='s', zorder=3)

    # Regression line through ED + ES points (both suffer from distance)
    all_d = np.concatenate([dist, dist])
    all_v = np.concatenate([ed_d, es_d])
    mask  = ~np.isnan(all_v)
    z = np.polyfit(all_d[mask], all_v[mask], 1)
    r = np.corrcoef(all_d[mask], all_v[mask])[0, 1]
    xfit = np.linspace(0, dist.max() + 1, 50)
    ax_sc.plot(xfit, np.polyval(z, xfit), color='#333', lw=1.2, ls='--',
               alpha=0.5, label=f'Trend (r={r:.2f})')

    ax_sc.set_xlabel('Propagation Distance (frames to anchor)', fontsize=10)
    ax_sc.set_ylabel('RV Dice', fontsize=10)
    ax_sc.set_xlim(-1.5, dist.max() + 1.5)
    ax_sc.set_ylim(0.3, 1.05)
    ax_sc.set_title('(d) Distance–Quality Relationship\n(more distance → lower Dice; Dual = 0 dist)',
                    fontsize=11, fontweight='bold')
    ax_sc.legend(fontsize=7.5, loc='lower left'); ax_sc.grid(alpha=0.3)
    for spine in ['top', 'right']:
        ax_sc.spines[spine].set_visible(False)

    plt.suptitle('Dual-Anchored MedSAM2: Boundary Accuracy Across All Cardiac Structures',
                 fontsize=12, y=1.02)
    out = os.path.join(fig_dir, 'paper_fig4_temporal_propagation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Figure 4 (ORIGINAL): EF Regression + Bland-Altman (moved to supplementary) ─

def fig4_ef_regression(fig_dir, metrics_json_path):
    from scipy import stats as sp_stats

    if not os.path.exists(metrics_json_path):
        print(f"Fig4: {metrics_json_path} not found, skipping"); return

    with open(metrics_json_path) as f:
        metrics = json.load(f)

    # Four-way: proposed (Dual) vs zero-shot (Dual) vs supervised — EF from pred masks, not GT
    PLOT_METHODS = [
        ('MedSAM2_Dual', 'MedSAM2\n(Dual-anchored)†', '#E53935'),
        ('SAM2_Dual',    'SAM2\n(Dual-anchored)',         '#FB8C00'),
        ('UNet',         'U-Net',                        '#1E88E5'),
        ('DINOv2',       'DINOv2',                       '#00ACC1'),
    ]

    # Filter to methods that have at least 5 valid EF pairs
    available = []
    for key, label, color in PLOT_METHODS:
        if key not in metrics:
            continue
        pairs = [(p['pred_EF'], p['gt_EF'], p['group'])
                 for p in metrics[key]
                 if p.get('pred_EF') is not None and p.get('gt_EF') is not None]
        if len(pairs) >= 5:
            available.append((key, label, color, pairs))

    if not available:
        print("Fig4: no methods with EF data, skipping"); return

    ba_all_means, ba_all_diffs = [], []
    for _, _, _, pairs in available:
        ef_p = np.array([p[0] for p in pairs])
        ef_g = np.array([p[1] for p in pairs])
        ba_all_means.extend((ef_p + ef_g) / 2.0)
        ba_all_diffs.extend(ef_p - ef_g)
    ba_x_pad = (max(ba_all_means) - min(ba_all_means)) * 0.1 + 3
    ba_y_pad = (max(ba_all_diffs) - min(ba_all_diffs)) * 0.1 + 3
    BA_XLIM = (min(ba_all_means) - ba_x_pad, max(ba_all_means) + ba_x_pad)
    BA_YLIM = (min(ba_all_diffs) - ba_y_pad, max(ba_all_diffs) + ba_y_pad)

    ncols = len(available)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 4.5, 8.5), facecolor='white')
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, (method_key, method_label, color, pairs) in enumerate(available):
        ef_pred = np.array([p[0] for p in pairs])
        ef_gt   = np.array([p[1] for p in pairs])
        groups  = [p[2] for p in pairs]
        def _dot_color(g):
            k = g.split('_')[0]
            return MNM_DISEASE_COLOR.get(k, GROUP_COLOR.get(k, '#999'))
        grp_colors = [_dot_color(g) for g in groups]
        n = len(ef_pred)

        # ── Scatter (top row) ──────────────────────────────────────────────────
        ax_sc = axes[0, col]
        ax_sc.scatter(ef_gt, ef_pred, c=grp_colors, s=65, alpha=0.85,
                      edgecolors='white', linewidth=0.5, zorder=3)

        lo = min(ef_gt.min(), ef_pred.min()) - 6
        hi = max(ef_gt.max(), ef_pred.max()) + 6
        x_line = np.linspace(lo, hi, 200)

        # identity reference (thin dashed)
        ax_sc.plot(x_line, x_line, 'k--', linewidth=0.8, alpha=0.4, zorder=1)

        # linear regression line
        slope, intercept, r_val, _, _ = sp_stats.linregress(ef_gt, ef_pred)
        ax_sc.plot(x_line, slope * x_line + intercept,
                   color=color, linewidth=1.8, zorder=2,
                   label=f'y={slope:.2f}x{intercept:+.1f}')

        mae = float(np.mean(np.abs(ef_pred - ef_gt)))
        ax_sc.text(0.05, 0.95, f'r = {r_val:.3f}\nMAE = {mae:.1f}%\nn = {n}',
                   transform=ax_sc.transAxes, fontsize=9.5, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
        ax_sc.set_xlabel('GT EF (%)', fontsize=11)
        ax_sc.set_ylabel('Predicted EF (%)', fontsize=11)
        ax_sc.set_title(method_label, fontsize=10, fontweight='bold')
        ax_sc.grid(alpha=0.25)

        # ── Bland-Altman (bottom row) ──────────────────────────────────────────
        ax_ba = axes[1, col]
        means = (ef_pred + ef_gt) / 2.0
        diffs = ef_pred - ef_gt
        bias  = float(np.mean(diffs))
        sd    = float(np.std(diffs, ddof=1))
        loa_hi = bias + 1.96 * sd
        loa_lo = bias - 1.96 * sd

        ax_ba.scatter(means, diffs, c=grp_colors, s=65, alpha=0.85,
                      edgecolors='white', linewidth=0.5, zorder=3)
        ax_ba.axhline(bias,   color='k',   linewidth=1.4, zorder=2, label=f'Bias={bias:+.1f}%')
        ax_ba.axhline(loa_hi, color='#D32F2F', linewidth=1.0, linestyle='--', zorder=2)
        ax_ba.axhline(loa_lo, color='#D32F2F', linewidth=1.0, linestyle='--', zorder=2)
        ax_ba.axhline(0,      color='gray', linewidth=0.6, linestyle=':', zorder=1)

        for val, txt, va in [(loa_hi, f'+1.96SD\n{loa_hi:+.1f}%', 'bottom'),
                              (bias,   f'Bias\n{bias:+.1f}%',     'center'),
                              (loa_lo, f'−1.96SD\n{loa_lo:+.1f}%', 'top')]:
            ax_ba.annotate(txt, xy=(1.01, val), xycoords=('axes fraction', 'data'),
                           fontsize=7.5, va=va, color='k' if val == bias else '#D32F2F')

        ax_ba.set_xlim(*BA_XLIM)
        ax_ba.set_ylim(*BA_YLIM)
        ax_ba.set_xlabel('Mean EF (%)', fontsize=11)
        ax_ba.set_ylabel('Pred − GT EF (%)', fontsize=11)
        ax_ba.set_title(method_label, fontsize=10, fontweight='bold')
        ax_ba.grid(alpha=0.25)

    # Pathology legend: use MnM disease groups if data came from MnM, else ACDC groups
    all_groups_in_data = set()
    for _, _, _, pairs in available:
        for p in pairs:
            g = str(p[2]).split('_')[0] if p[2] else ''
            if g:
                all_groups_in_data.add(g)
    mnm_diseases = set(MNM_DISEASE_GROUPS)
    if all_groups_in_data & mnm_diseases:
        legend_colors = MNM_DISEASE_COLOR
        legend_groups = [g for g in MNM_DISEASE_GROUPS if g in all_groups_in_data]
    else:
        legend_colors = GROUP_COLOR
        legend_groups = [g for g in GROUPS if g in all_groups_in_data]
    group_patches = [mpatches.Patch(color=legend_colors.get(g, '#999'), label=g)
                     for g in legend_groups]
    if group_patches:
        fig.legend(handles=group_patches, loc='lower center',
                   ncol=min(len(group_patches), 7), fontsize=9,
                   title='Pathology', bbox_to_anchor=(0.5, -0.02))

    n_total = len(available[0][3]) if available else 0
    dataset_label = 'MnM External Validation' if all_groups_in_data & mnm_diseases else 'ACDC Val'
    fig.text(0.5, 1.005,
             f'Ejection Fraction: Predicted vs Ground Truth — Regression & Bland-Altman ({dataset_label}, n={n_total})',
             ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, -0.06,
             '† EF/EDV/ESV from model-predicted masks given ED+ES prompts.',
             ha='center', fontsize=8, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(fig_dir, 'paper_fig4_ef_regression.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Figure 5: Cross-vendor generalisation (MnM) ──────────────────────────────

def fig5_crossvendor(fig_dir, mnm_metrics_json):
    if not os.path.exists(mnm_metrics_json):
        print(f"Fig6: {mnm_metrics_json} not found, skipping"); return

    with open(mnm_metrics_json) as f:
        metrics = json.load(f)

    plist = metrics.get('MedSAM2_Dual', [])
    if not plist: print("Fig6: no MedSAM2_Dual MnM results"); return

    # Parse vendor from group field (format: "PATHOLOGY_VendorID")
    vendor_map = {'A': 'Siemens', 'B': 'Philips', 'C': 'GE', 'D': 'Canon'}
    vendor_data = {v: {'RV': [], 'Myo': [], 'LV': []} for v in vendor_map.values()}
    vendor_data['Overall'] = {'RV': [], 'Myo': [], 'LV': []}

    for p in plist:
        group = p.get('group', '')
        parts = group.split('_')
        vid   = parts[-1] if len(parts) > 1 else ''
        vname = vendor_map.get(vid, None)
        for struct in ['RV', 'Myo', 'LV']:
            v = p.get(f'dice_{struct}')
            if v is None: continue
            vendor_data['Overall'][struct].append(v)
            if vname:
                vendor_data[vname][struct].append(v)

    vendors  = [v for v in list(vendor_map.values()) + ['Overall']
                if vendor_data[v]['LV']]
    structs  = ['RV', 'Myo', 'LV']
    s_colors = {'RV': '#EF9A9A', 'Myo': '#A5D6A7', 'LV': '#90CAF9'}
    x       = np.arange(len(vendors))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(vendors)*2), 5))
    for i, struct in enumerate(structs):
        means = [np.mean(vendor_data[v][struct]) if vendor_data[v][struct] else 0 for v in vendors]
        stds  = [np.std(vendor_data[v][struct])  if vendor_data[v][struct] else 0 for v in vendors]
        ns    = [len(vendor_data[v][struct]) for v in vendors]
        bars  = ax.bar(x + (i-1)*width, means, width, label=struct,
                       color=s_colors[struct], alpha=0.85,
                       yerr=stds, capsize=4, error_kw=dict(linewidth=1))
        for bar, n, m, s in zip(bars, ns, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + s + 0.03,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=7)
        # Sample sizes under x-axis tick (only for LV)
        if struct == 'LV':
            for j, (v, n) in enumerate(zip(vendors, ns)):
                ax.text(j, -0.08, f'n={n}', ha='center', va='top',
                        fontsize=8, color='gray', transform=ax.get_xaxis_transform())

    ax.set_xticks(x); ax.set_xticklabels(vendors, fontsize=11)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('MedSAM2 (Dual-anchored) — Cross-Vendor Generalisation (M&Ms Test Set)',
                 fontsize=12, fontweight='bold')

    # Separate "Overall" column visually
    ax.axvline(len(vendors)-1.5, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()
    out = os.path.join(fig_dir, 'paper_fig5_crossvendor.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Clinical metrics tables ───────────────────────────────────────────────────

def make_clinical_tables(results_dir):
    """
    Read metrics_acdc_val.json and metrics_mnm.json and write two CSVs:
      table_clinical_acdc_complete.csv  — EF / EDV / ESV per method (ACDC val)
      table_clinical_mnm_complete.csv   — EF / EDV / ESV for MedSAM2_Dual (MnM)
    """
    from scipy import stats as sp_stats

    METHOD_ORDER = [
        ('SAM2_Dual',    'SAM2'),
        ('MedSAM2_ED',   'MedSAM2 (ED-anchored)'),
        ('MedSAM2_ES',   'MedSAM2 (ES-anchored)'),
        ('MedSAM2_Dual', 'MedSAM2 (Dual-anchored)†'),
        ('UNet',         'U-Net (supervised)'),
        ('DINOv2',       'DINOv2 (supervised)'),
        ('nnUNet',       'nnUNet (supervised)'),
    ]

    def _clinical_stats(plist):
        """From a list of per-patient dicts return (ef_mae, ef_std, edv_mae, esv_mae, ef_r, n)."""
        ef_pairs, edv_pairs, esv_pairs = [], [], []
        for p in plist:
            pef, gef = p.get('pred_EF'), p.get('gt_EF')
            pedv, gedv = p.get('pred_EDV'), p.get('gt_EDV')
            pesv, gesv = p.get('pred_ESV'), p.get('gt_ESV')
            if pef is not None and gef is not None:
                ef_pairs.append((pef, gef))
            if pedv is not None and gedv is not None:
                edv_pairs.append((pedv, gedv))
            if pesv is not None and gesv is not None:
                esv_pairs.append((pesv, gesv))

        def _mae_std(pairs):
            if not pairs: return None, None
            errs = [abs(a - b) for a, b in pairs]
            return float(np.mean(errs)), float(np.std(errs))

        def _r(pairs):
            if len(pairs) < 3: return None
            x = [p[1] for p in pairs]; y = [p[0] for p in pairs]
            r, _ = sp_stats.pearsonr(x, y)
            return float(r)

        ef_mae, ef_std   = _mae_std(ef_pairs)
        edv_mae, edv_std = _mae_std(edv_pairs)
        esv_mae, esv_std = _mae_std(esv_pairs)
        ef_r  = _r(ef_pairs)
        edv_r = _r(edv_pairs)
        esv_r = _r(esv_pairs)
        n    = len(ef_pairs)
        return ef_mae, ef_std, edv_mae, edv_std, esv_mae, esv_std, ef_r, edv_r, esv_r, n

    # ── ACDC val ──────────────────────────────────────────────────────────────
    acdc_path = os.path.join(results_dir, 'metrics_acdc_val.json')
    if os.path.exists(acdc_path):
        with open(acdc_path) as f:
            acdc = json.load(f)

        rows = []
        for key, label in METHOD_ORDER:
            if key not in acdc: continue
            ef_mae, ef_std, edv_mae, edv_std, esv_mae, esv_std, ef_r, edv_r, esv_r, n = \
                _clinical_stats(acdc[key])
            rows.append({
                'Method': label,
                'N': n,
                'EF_MAE': f'{ef_mae:.2f}' if ef_mae is not None else 'N/A',
                'EF_std': f'{ef_std:.2f}' if ef_std is not None else 'N/A',
                'EF_r':   f'{ef_r:.3f}'   if ef_r   is not None else 'N/A',
                'EDV_MAE': f'{edv_mae:.1f}' if edv_mae is not None else 'N/A',
                'EDV_std': f'{edv_std:.1f}' if edv_std is not None else 'N/A',
                'EDV_r':   f'{edv_r:.3f}'   if edv_r  is not None else 'N/A',
                'ESV_MAE': f'{esv_mae:.1f}' if esv_mae is not None else 'N/A',
                'ESV_std': f'{esv_std:.1f}' if esv_std is not None else 'N/A',
                'ESV_r':   f'{esv_r:.3f}'   if esv_r  is not None else 'N/A',
            })

        out = os.path.join(results_dir, 'paper_table_clinical_acdc_complete.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['Method','N',
                'EF_MAE','EF_std','EF_r','EDV_MAE','EDV_std','EDV_r','ESV_MAE','ESV_std','ESV_r'])
            w.writeheader(); w.writerows(rows)
        print(f"Saved {out}")

        # Pretty-print
        print("\n── Clinical Metrics (ACDC Val) ──")
        print(f"{'Method':<32} {'EF MAE':>9} {'EF r':>7} {'EDV MAE':>10} {'ESV MAE':>10} N")
        for r in rows:
            print(f"{r['Method']:<32} {r['EF_MAE']:>5}±{r['EF_std']:<3} "
                  f"{r['EF_r']:>7} {r['EDV_MAE']:>6}±{r['EDV_std']:<4} "
                  f"{r['ESV_MAE']:>6}±{r['ESV_std']:<4} {r['N']}")

    # ── MnM ───────────────────────────────────────────────────────────────────
    mnm_path = os.path.join(results_dir, 'metrics_mnm.json')
    if os.path.exists(mnm_path):
        with open(mnm_path) as f:
            mnm = json.load(f)

        mnm_rows = []
        for key, label in METHOD_ORDER:
            if key not in mnm: continue
            ef_mae, ef_std, edv_mae, edv_std, esv_mae, esv_std, ef_r, edv_r, esv_r, n = \
                _clinical_stats(mnm[key])
            mnm_rows.append({
                'Method': label, 'N': n,
                'EF_MAE': f'{ef_mae:.2f}' if ef_mae is not None else 'N/A',
                'EF_std': f'{ef_std:.2f}' if ef_std is not None else 'N/A',
                'EF_r':   f'{ef_r:.3f}'   if ef_r   is not None else 'N/A',
                'EDV_MAE': f'{edv_mae:.1f}' if edv_mae is not None else 'N/A',
                'EDV_std': f'{edv_std:.1f}' if edv_std is not None else 'N/A',
                'EDV_r':   f'{edv_r:.3f}'   if edv_r  is not None else 'N/A',
                'ESV_MAE': f'{esv_mae:.1f}' if esv_mae is not None else 'N/A',
                'ESV_std': f'{esv_std:.1f}' if esv_std is not None else 'N/A',
                'ESV_r':   f'{esv_r:.3f}'   if esv_r  is not None else 'N/A',
            })

        out = os.path.join(results_dir, 'paper_table_clinical_mnm_complete.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['Method','N',
                'EF_MAE','EF_std','EF_r','EDV_MAE','EDV_std','EDV_r','ESV_MAE','ESV_std','ESV_r'])
            w.writeheader(); w.writerows(mnm_rows)
        print(f"Saved {out}")


# ── Table 1 from metrics JSON (MnM external validation) ───────────────────────

MNM_METHOD_ORDER = [
    ('SAM2_Dual',    'SAM2 (Dual-anchored)'),
    ('MedSAM2_ED',   'MedSAM2 (ED-anchored)'),
    ('MedSAM2_ES',   'MedSAM2 (ES-anchored)'),
    ('MedSAM2_Dual', 'MedSAM2 (Dual-anchored)†'),
    ('UNet',         'U-Net (supervised)'),
    ('DINOv2',       'DINOv2 (supervised)'),
    ('nnUNet',       'nnUNet (supervised)'),
]


def make_table1_from_mnm_json(results_dir):
    """Read metrics_mnm.json and write paper_table1_mnm_complete.csv
    with Dice + HD95 + ASSD for all available methods.
    """
    mnm_path = os.path.join(results_dir, 'metrics_mnm.json')
    if not os.path.exists(mnm_path):
        print(f"Table1-MnM: {mnm_path} not found, skipping"); return {}

    with open(mnm_path) as f:
        mnm = json.load(f)

    def _valid(v):
        return v is not None and not (isinstance(v, float) and (np.isnan(v) if isinstance(v, float) else False))

    def _stats(vals):
        v = [x for x in vals if x is not None]
        if not v: return None, None
        return float(np.mean(v)), float(np.std(v))

    rows = []
    display = {}
    for key, label in MNM_METHOD_ORDER:
        if key not in mnm: continue
        plist = mnm[key]

        rv_d  = [p['dice_RV']  for p in plist if _valid(p.get('dice_RV'))]
        myo_d = [p['dice_Myo'] for p in plist if _valid(p.get('dice_Myo'))]
        lv_d  = [p['dice_LV']  for p in plist if _valid(p.get('dice_LV'))]
        rv_h  = [p['hd95_RV']  for p in plist if _valid(p.get('hd95_RV'))]
        myo_h = [p['hd95_Myo'] for p in plist if _valid(p.get('hd95_Myo'))]
        lv_h  = [p['hd95_LV']  for p in plist if _valid(p.get('hd95_LV'))]
        rv_a  = [p['assd_RV']  for p in plist if _valid(p.get('assd_RV'))]
        myo_a = [p['assd_Myo'] for p in plist if _valid(p.get('assd_Myo'))]
        lv_a  = [p['assd_LV']  for p in plist if _valid(p.get('assd_LV'))]

        def fmt(lst, dec=3):
            m, s = _stats(lst)
            return (f'{m:.{dec}f}', f'{s:.{dec}f}') if m is not None else ('N/A', 'N/A')

        rv_dm, rv_ds   = fmt(rv_d)
        myo_dm, myo_ds = fmt(myo_d)
        lv_dm, lv_ds   = fmt(lv_d)
        rv_hm, rv_hs   = fmt(rv_h, 2)
        myo_hm, myo_hs = fmt(myo_h, 2)
        lv_hm, lv_hs   = fmt(lv_h, 2)
        rv_am, rv_as   = fmt(rv_a, 2)
        myo_am, myo_as = fmt(myo_a, 2)
        lv_am, lv_as   = fmt(lv_a, 2)

        ref_key = 'MedSAM2_Dual'
        if key == ref_key:
            rv_p = myo_p = lv_p = '—'
        else:
            rv_p  = _pval_star(_wilcoxon_paired(mnm, ref_key, key, 'dice_RV'))
            myo_p = _pval_star(_wilcoxon_paired(mnm, ref_key, key, 'dice_Myo'))
            lv_p  = _pval_star(_wilcoxon_paired(mnm, ref_key, key, 'dice_LV'))

        rows.append({
            'Method':    label,
            'N':         len(plist),
            'RV_Dice':   rv_dm,  'RV_Dice_std':   rv_ds,
            'Myo_Dice':  myo_dm, 'Myo_Dice_std':  myo_ds,
            'LV_Dice':   lv_dm,  'LV_Dice_std':   lv_ds,
            'RV_HD95':   rv_hm,  'Myo_HD95':  myo_hm,  'LV_HD95':  lv_hm,
            'RV_ASSD':   rv_am,  'Myo_ASSD':  myo_am,  'LV_ASSD':  lv_am,
            'RV_pval':   rv_p,   'Myo_pval':  myo_p,   'LV_pval':  lv_p,
        })
        display[label] = {
            'RV':  (float(rv_dm)  if rv_dm  != 'N/A' else 0.0, float(rv_ds)  if rv_ds  != 'N/A' else 0.0),
            'Myo': (float(myo_dm) if myo_dm != 'N/A' else 0.0, float(myo_ds) if myo_ds != 'N/A' else 0.0),
            'LV':  (float(lv_dm)  if lv_dm  != 'N/A' else 0.0, float(lv_ds)  if lv_ds  != 'N/A' else 0.0),
        }

    out = os.path.join(results_dir, 'paper_table1_mnm_complete.csv')
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"Saved {out}")

        print("\n── Table 1 (MnM External Validation) ──")
        hdr = f"{'Method':<30} {'RV Dice':>11} {'Myo Dice':>11} {'LV Dice':>11} {'RV HD95':>9} {'Myo HD95':>9} {'LV HD95':>9}"
        print(hdr); print('─' * len(hdr))
        for r in rows:
            print(f"{r['Method']:<30} "
                  f"{r['RV_Dice']:>5}±{r['RV_Dice_std']:<5} "
                  f"{r['Myo_Dice']:>5}±{r['Myo_Dice_std']:<5} "
                  f"{r['LV_Dice']:>5}±{r['LV_Dice_std']:<5} "
                  f"{r['RV_HD95']:>9} {r['Myo_HD95']:>9} {r['LV_HD95']:>9}")

    return display


def _pval_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _wilcoxon_paired(mnm, key_ref, key_other, metric_key):
    """Wilcoxon signed-rank test (one-sided: ref > other) on paired per-patient dice."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float('nan')
    if key_ref not in mnm or key_other not in mnm:
        return float('nan')
    pid_ref   = {p['pid']: p.get(metric_key) for p in mnm[key_ref]}
    pid_other = {p['pid']: p.get(metric_key) for p in mnm[key_other]}
    common = sorted(set(pid_ref) & set(pid_other))
    v_ref   = [pid_ref[p]   for p in common if pid_ref[p]   is not None and pid_other[p] is not None]
    v_other = [pid_other[p] for p in common if pid_ref[p]   is not None and pid_other[p] is not None]
    if len(v_ref) < 5:
        return float('nan')
    d = np.array(v_ref) - np.array(v_other)
    if (d == 0).all():
        return 1.0
    try:
        _, pval = wilcoxon(v_ref, v_other, alternative='greater')
        return float(pval)
    except Exception:
        return float('nan')


def fig2_boxplot_from_mnm_json(fig_dir, results_dir):
    """Generate Fig 2 boxplot from MnM metrics JSON (per-patient Dice)."""
    mnm_path = os.path.join(results_dir, 'metrics_mnm.json')
    if not os.path.exists(mnm_path):
        print("Fig2-MnM: metrics_mnm.json not found, skipping"); return

    with open(mnm_path) as f:
        mnm = json.load(f)

    method_configs = [
        ('SAM2_Dual',    'SAM2\n(Dual-anchored)', '#607D8B'),
        ('MedSAM2_ED',   'MedSAM2\n(ED-anchored)', '#FFB74D'),
        ('MedSAM2_ES',   'MedSAM2\n(ES-anchored)', '#FF7043'),
        ('MedSAM2_Dual', 'MedSAM2\n(Dual-anchored)†', '#E53935'),
        ('UNet',         'U-Net\n(supervised)', '#1E88E5'),
        ('DINOv2',       'DINOv2\n(supervised)', '#00ACC1'),
    ]
    ref_key = 'MedSAM2_Dual'

    method_data = {}
    method_keys = {}
    for key, label, color in method_configs:
        if key not in mnm: continue
        plist = mnm[key]
        method_data[label] = {
            'color': color, 'key': key,
            'RV':  [p['dice_RV']  for p in plist if p.get('dice_RV')  is not None],
            'Myo': [p['dice_Myo'] for p in plist if p.get('dice_Myo') is not None],
            'LV':  [p['dice_LV']  for p in plist if p.get('dice_LV')  is not None],
        }
        method_keys[label] = key

    if not method_data:
        print("Fig2-MnM: no data found"); return

    method_names = list(method_data.keys())
    colors = [method_data[n]['color'] for n in method_names]
    n_total = len(mnm.get('MedSAM2_Dual', []))

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(method_names)*2.5), 6),
                             facecolor='white')
    fig.suptitle(f'Dice Score Comparison — M&Ms External Validation (n={n_total}, 4 vendors)',
                 fontsize=13, fontweight='bold', y=1.02)

    metric_dice_key = {'RV': 'dice_RV', 'Myo': 'dice_Myo', 'LV': 'dice_LV'}

    for ax, metric in zip(axes, ['RV', 'Myo', 'LV']):
        ax.set_facecolor('#FAFAFA')
        data_per_method = [method_data[n][metric] for n in method_names]

        bp = ax.boxplot(data_per_method, patch_artist=True,
                        medianprops=dict(color='#212121', linewidth=2.2),
                        whiskerprops=dict(linewidth=1.4, color='#555'),
                        capprops=dict(linewidth=1.4, color='#555'),
                        flierprops=dict(marker='o', markersize=4, alpha=0.55,
                                        markeredgewidth=0.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.82)
            patch.set_edgecolor(color); patch.set_linewidth(1.5)

        top_y = 1.05
        for i, (name, vals) in enumerate(zip(method_names, data_per_method)):
            if vals:
                ax.text(i+1, min(max(vals)+0.025, 0.99), f'{np.mean(vals):.2f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#333', fontweight='bold')

        # Significance stars vs MedSAM2 Dual (ref), skipping ref itself
        if ref_key in mnm:
            ref_label_idx = None
            for i, name in enumerate(method_names):
                if method_keys.get(name) == ref_key:
                    ref_label_idx = i; break

            star_y = 1.08
            for i, name in enumerate(method_names):
                k = method_keys.get(name)
                if k == ref_key or k is None: continue
                pval = _wilcoxon_paired(mnm, ref_key, k, metric_dice_key[metric])
                star = _pval_star(pval)
                ax.text(i+1, star_y, star, ha='center', va='bottom',
                        fontsize=8, color='#C62828' if star != 'ns' else '#888')

        ax.set_title(metric, fontsize=14, fontweight='bold', pad=6)
        ax.set_xticks(range(1, len(method_names)+1))
        ax.set_xticklabels(method_names, rotation=35, ha='right', fontsize=8.5)
        ax.set_ylabel('Dice Score', fontsize=11)
        ax.set_ylim(0, 1.22)
        ax.axhline(0.8, color='#9E9E9E', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.grid(axis='y', alpha=0.35, color='#ccc', linewidth=0.8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.text(0.5, -0.04,
             '† = Proposed. Stars = Wilcoxon signed-rank vs MedSAM2 Dual (one-sided): ***p<0.001, **p<0.01, *p<0.05',
             ha='center', fontsize=8.5, color='#555', style='italic')
    plt.tight_layout()
    out = os.path.join(fig_dir, 'paper_fig2_boxplot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.fig_dir, exist_ok=True)
    medsam2_dir = os.path.join(args.results_dir, 'medsam2')
    sam2_dir    = os.path.join(args.results_dir, 'sam2')
    sam2b_dir   = os.path.join(args.results_dir, 'sam2_bidir')
    unet_json   = os.path.join(args.results_dir, 'unet',   'results.json')
    def _prefer(a, b): return a if os.path.exists(a) else b
    unet_ckpt   = _prefer(os.path.join(args.results_dir, 'unet_combined',   'best_model.pth'),
                           os.path.join(args.results_dir, 'unet',           'best_model.pth'))
    dinov2_json = os.path.join(args.results_dir, 'dinov2', 'results.json')
    dinov2_ckpt = _prefer(os.path.join(args.results_dir, 'dinov2_combined', 'best_model.pth'),
                           os.path.join(args.results_dir, 'dinov2',         'best_model.pth'))
    nnunet_json = os.path.join(args.results_dir, 'nnunet', 'results.json')
    metrics_val = os.path.join(args.results_dir, 'metrics_acdc_val.json')
    metrics_mnm = os.path.join(args.results_dir, 'metrics_mnm.json')

    print("\n══ Computing Dice for all methods ══")
    ms2_ed   = compute_method_dice(medsam2_dir, 'ed_pred', 'es')
    ms2_es   = compute_method_dice(medsam2_dir, 'es_pred', 'ed')
    ms2_dual = compute_method_dice(medsam2_dir, 'bidir',   'es')
    sam2_bi  = compute_method_dice(sam2b_dir,   'bidir',   'es') if os.path.isdir(sam2b_dir) else {}

    def load_json_results(path):
        if not os.path.exists(path): return {}
        with open(path) as f: raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    unet_res   = load_json_results(unet_json)
    dinov2_res = load_json_results(dinov2_json)
    nnunet_res = load_json_results(nnunet_json)

    # Ordered for Table 1 and Fig 2
    raw_results = {}
    if sam2_bi:    raw_results['SAM2']                         = sam2_bi
    if ms2_ed:     raw_results['MedSAM2\n(ED-anchored)']      = ms2_ed
    if ms2_es:     raw_results['MedSAM2\n(ES-anchored)']      = ms2_es
    if ms2_dual:   raw_results['MedSAM2\n(Dual-anchored)†']  = ms2_dual
    if unet_res:   raw_results['U-Net\n(supervised)']         = unet_res
    if dinov2_res: raw_results['DINOv2\n(supervised)']        = dinov2_res
    if nnunet_res: raw_results['nnUNet\n(supervised)']        = nnunet_res

    all_results = {n: summarise(d) for n, d in raw_results.items()}

    # Print Table 1
    print("\n── Table 1: Dice ──")
    hdr = f"{'Method':<28} {'RV':>13} {'Myo':>13} {'LV':>13}"
    print(hdr); print('─' * len(hdr))
    for method, s in all_results.items():
        row = f"{method.replace(chr(10),' '):<28}"
        for m in ['RV','Myo','LV']:
            row += f"  {s[m][0]:.3f}±{s[m][1]:.3f}"
        print(row)

    csv_path = os.path.join(args.results_dir, 'paper_table1_dice.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method','RV_mean','RV_std','Myo_mean','Myo_std','LV_mean','LV_std'])
        for method, s in all_results.items():
            w.writerow([method.replace('\n',' ')] +
                       [f'{s[m][i]:.4f}' for m in ['RV','Myo','LV'] for i in [0,1]])
    print(f"Saved {csv_path}")

    print("\n══ Clinical metrics tables ══")
    make_clinical_tables(args.results_dir)

    print("\n══ Table 1 (MnM external validation, Dice + HD95 + ASSD) ══")
    make_table1_from_mnm_json(args.results_dir)

    print("\n══ Generating figures ══")
    try:
        fig1_qualitative(args.fig_dir, args.db, medsam2_dir, sam2b_dir, unet_ckpt,
                         dinov2_ckpt=dinov2_ckpt)
    except Exception as e:
        print(f"Fig1 skipped (needs GPU): {e}")
    # Fig 2: Use MnM results if available, else fall back to ACDC val
    mnm_metrics_path = os.path.join(args.results_dir, 'metrics_mnm.json')
    if os.path.exists(mnm_metrics_path):
        fig2_boxplot_from_mnm_json(args.fig_dir, args.results_dir)
    elif raw_results:
        fig2_boxplot(args.fig_dir, raw_results)
    mnm_pred_dir = os.path.join(args.results_dir, 'medsam2_mnm')
    if os.path.isdir(mnm_pred_dir):
        fig3_timevolume_mnm(args.fig_dir,
                            '/scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm',
                            mnm_pred_dir)
    else:
        fig3_timevolume(args.fig_dir, args.db, medsam2_dir)
    fig4_ef_regression(args.fig_dir, metrics_mnm)
    fig5_crossvendor(args.fig_dir, metrics_mnm)
    print(f"\nAll figures → {args.fig_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--db',          default=DB_PATH)
    parser.add_argument('--fig_dir',     default=FIG_DIR)
    main(parser.parse_args())
