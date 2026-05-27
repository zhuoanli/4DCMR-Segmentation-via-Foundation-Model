"""
compute_all_metrics.py
Compute all evaluation metrics for all datasets and methods:
  - Dice (RV / Myo / LV)
  - HD95 and ASSD (3D, physical units mm)
  - EF / EDV / ESV (LV clinical metrics)

Usage:
  python compute_all_metrics.py             # ACDC val (default)
  python compute_all_metrics.py --dataset acdc_test
  python compute_all_metrics.py --dataset mnm
  python compute_all_metrics.py --dataset mnm2

Outputs:
  results/metrics_{dataset}.json   — full per-patient results
  results/table_surface_{dataset}.csv
  results/table_clinical_{dataset}.csv
"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    from medpy.metric.binary import assd as medpy_assd
    HAS_MEDPY = True
except ImportError:
    print("WARNING: medpy not found. Install with: pip install medpy")
    print("         Falling back to scipy-based HD95 approximation.")
    HAS_MEDPY = False
    from scipy.ndimage import distance_transform_edt

RESULTS_BASE = '/scratch/gautschi/li4533/MIUA_2026/results'
PREP_DIRS    = {
    'acdc_val':  '/scratch/gautschi/li4533/MIUA_2026/preprocessed',
    'acdc_test': '/scratch/gautschi/li4533/MIUA_2026/preprocessed_acdc_test',
    'mnm':       '/scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm',
    'mnm2':      '/scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm2',
}
RESULT_DIRS  = {
    'medsam2_bidir': os.path.join(RESULTS_BASE, 'medsam2'),
    'medsam2_ed':    os.path.join(RESULTS_BASE, 'medsam2'),
    'medsam2_es':    os.path.join(RESULTS_BASE, 'medsam2'),
    'sam2_ed':       os.path.join(RESULTS_BASE, 'sam2'),
    'unet':          os.path.join(RESULTS_BASE, 'unet'),
}

ACDC_VAL_IDS = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]
CLASSES      = {1: 'RV', 2: 'Myo', 3: 'LV'}


# ── helpers ────────────────────────────────────────────────────────────────────

def dice_np(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


def _hd95_scipy(pred_bin, gt_bin, voxelspacing):
    """Fallback HD95 using scipy distance transforms."""
    from scipy.ndimage import distance_transform_edt
    if not pred_bin.any() or not gt_bin.any():
        return np.nan
    dt_pred = distance_transform_edt(~pred_bin, sampling=voxelspacing)
    dt_gt   = distance_transform_edt(~gt_bin,   sampling=voxelspacing)
    surfdist_p2g = dt_gt[pred_bin]
    surfdist_g2p = dt_pred[gt_bin]
    all_dists = np.concatenate([surfdist_p2g, surfdist_g2p])
    return float(np.percentile(all_dists, 95))


def _assd_scipy(pred_bin, gt_bin, voxelspacing):
    from scipy.ndimage import distance_transform_edt
    if not pred_bin.any() or not gt_bin.any():
        return np.nan
    dt_pred = distance_transform_edt(~pred_bin, sampling=voxelspacing)
    dt_gt   = distance_transform_edt(~gt_bin,   sampling=voxelspacing)
    d1 = dt_gt[pred_bin].mean()
    d2 = dt_pred[gt_bin].mean()
    return float((d1 + d2) / 2)


def compute_hd95_assd_3d(pred_3d, gt_3d, cls, spacing_xyz):
    """
    pred_3d, gt_3d : (H, W, Z) numpy arrays (in original resolution, NOT 512)
    spacing_xyz    : (dx, dy, dz) in mm
    Returns (hd95_mm, assd_mm) or (nan, nan) if class absent
    """
    p = (pred_3d == cls).astype(bool)
    g = (gt_3d  == cls).astype(bool)
    if not g.any() or not p.any():
        return np.nan, np.nan

    # medpy/scipy expect (z, y, x) ordering and voxelspacing in same order
    voxelspacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    p_zyx = p.transpose(2, 0, 1)
    g_zyx = g.transpose(2, 0, 1)

    if HAS_MEDPY:
        try:
            h = float(medpy_hd95(p_zyx, g_zyx, voxelspacing=voxelspacing))
            a = float(medpy_assd(p_zyx, g_zyx, voxelspacing=voxelspacing))
        except Exception:
            h = _hd95_scipy(p_zyx, g_zyx, voxelspacing)
            a = _assd_scipy(p_zyx, g_zyx, voxelspacing)
    else:
        h = _hd95_scipy(p_zyx, g_zyx, voxelspacing)
        a = _assd_scipy(p_zyx, g_zyx, voxelspacing)

    return h, a


def resize_mask_to_orig(mask_512, orig_H, orig_W):
    """Resize a 512x512 uint8 mask back to (orig_H, orig_W) using nearest-neighbour."""
    from PIL import Image
    if orig_H == 512 and orig_W == 512:
        return mask_512
    img = Image.fromarray(mask_512)
    return np.array(img.resize((orig_W, orig_H), Image.NEAREST), dtype=np.uint8)


def decode_group(raw):
    s = str(raw)
    if s.startswith("np.bytes_(b'") and s.endswith("')"):
        s = s[len("np.bytes_(b'"):-2]
    elif s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s


# ── patient-level loader ───────────────────────────────────────────────────────

def load_patient_slices(pid_pattern, prep_dir, result_dir, pred_key, ed_only=False):
    """
    Returns:
      pred_3d : (orig_H, orig_W, Z)  — predicted mask at the evaluation frame
      gt_3d   : (orig_H, orig_W, Z)  — GT mask (es_mask for most methods)
      pred_all: list of (T, orig_H, orig_W) — all frames (for EF computation)
      meta    : dict with ed_idx, es_idx, pixdim, group
    ed_only: if True, always evaluate at ED frame (for datasets with no ES GT)
    """
    prep_npzs = sorted(glob(os.path.join(prep_dir, f'{pid_pattern}_slice*.npz')))
    if not prep_npzs:
        return None

    # Load first slice for metadata
    d0      = np.load(prep_npzs[0], allow_pickle=True)
    ed_idx  = int(d0['ed_idx'])
    es_idx  = int(d0['es_idx'])
    pixdim  = d0['pixdim']
    group   = decode_group(d0['group'])
    orig_H  = int(d0['orig_H'])
    orig_W  = int(d0['orig_W'])
    scale_H = orig_H / 512.0
    scale_W = orig_W / 512.0

    Z = len(prep_npzs)
    T = d0['frames'].shape[0]

    pred_3d     = np.zeros((orig_H, orig_W, Z), dtype=np.uint8)
    gt_3d_ed    = np.zeros((orig_H, orig_W, Z), dtype=np.uint8)
    gt_3d_es    = np.zeros((orig_H, orig_W, Z), dtype=np.uint8)
    pred_all    = np.zeros((T, orig_H, orig_W, Z), dtype=np.uint8)  # all frames

    for z, prep_path in enumerate(prep_npzs):
        prep_d = np.load(prep_path, allow_pickle=True)

        # GT masks
        gt_3d_ed[:, :, z] = resize_mask_to_orig(prep_d['ed_mask'], orig_H, orig_W)
        gt_3d_es[:, :, z] = resize_mask_to_orig(prep_d['es_mask'], orig_H, orig_W)

        # Try to load predictions from result dir
        stem     = os.path.basename(prep_path).replace('.npz', '')
        res_path = os.path.join(result_dir, stem + '.npz')

        if not os.path.exists(res_path):
            continue

        res_d = np.load(res_path, allow_pickle=True)
        if pred_key not in res_d:
            continue

        pred_seq = res_d[pred_key]   # (T, 512, 512)

        # All frames
        for t in range(min(T, pred_seq.shape[0])):
            pred_all[t, :, :, z] = resize_mask_to_orig(pred_seq[t], orig_H, orig_W)

        # Evaluation frame
        if ed_only:
            eval_t = ed_idx
        else:
            eval_t = es_idx if pred_key in ('ed_pred', 'bidir') else ed_idx
        if eval_t < pred_seq.shape[0]:
            pred_3d[:, :, z] = resize_mask_to_orig(pred_seq[eval_t], orig_H, orig_W)

    if ed_only:
        gt_3d_eval = gt_3d_ed
    else:
        gt_3d_eval = gt_3d_es if pred_key in ('ed_pred', 'bidir') else gt_3d_ed

    return {
        'pred_3d':   pred_3d,
        'gt_3d':     gt_3d_eval,
        'gt_3d_ed':  gt_3d_ed,
        'gt_3d_es':  gt_3d_es,
        'pred_all':  pred_all,
        'ed_idx':    ed_idx,
        'es_idx':    es_idx,
        'pixdim':    np.array(pixdim, dtype=np.float32),
        'group':     group,
        'orig_H':    orig_H,
        'orig_W':    orig_W,
    }


# ── clinical metrics ───────────────────────────────────────────────────────────

def compute_lv_volume(mask_3d, pixdim):
    """mask_3d: (H, W, Z). Returns volume in mL."""
    voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2])
    return float((mask_3d == 3).sum()) * voxel_mm3 / 1000.0


def compute_ef_edv_esv(pred_all, ed_idx, es_idx, pixdim):
    """
    pred_all: (T, H, W, Z)
    Returns (EF%, EDV_mL, ESV_mL).
    EDV is always the LARGER of the two frame volumes to handle scanners where
    the ED/ES frame order varies by vendor (e.g., M&Ms dataset).
    """
    T = pred_all.shape[0]
    voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2])

    ed_t = min(ed_idx, T - 1)
    es_t = min(es_idx, T - 1)

    vol_ed = float((pred_all[ed_t] == 3).sum()) * voxel_mm3 / 1000.0
    vol_es = float((pred_all[es_t] == 3).sum()) * voxel_mm3 / 1000.0

    # Use max-volume frame as EDV, min-volume as ESV (robust to ED/ES convention)
    EDV = max(vol_ed, vol_es)
    ESV = min(vol_ed, vol_es)

    if EDV < 1e-3:
        return np.nan, EDV, ESV

    EF = (EDV - ESV) / EDV * 100.0
    return EF, EDV, ESV


ACDC_VAL_GROUP = {
    **{pid: 'DCM'  for pid in [17,18,19,20]},
    **{pid: 'HCM'  for pid in [37,38,39,40]},
    **{pid: 'MINF' for pid in [57,58,59,60]},
    **{pid: 'NOR'  for pid in [77,78,79,80]},
    **{pid: 'RV'   for pid in [97,98,99,100]},
}


def _load_supervised_results(dataset, result_base, metrics_json_path):
    """Load UNet/DINOv2 Dice-only results into the standard per-patient format.
    Reads existing EF data from metrics JSON if already computed by the eval job.
    Returns dict suitable for passing to summarise_results() (CSV only, not JSON).
    """
    if dataset != 'acdc_val':
        return {}

    # Pull any EF data already written by train_eval_unet.py
    existing_ef = {}
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path) as f:
                existing = json.load(f)
            for method in ['UNet', 'DINOv2']:
                if method in existing:
                    existing_ef[method] = {p['pid']: p for p in existing[method]}
        except (json.JSONDecodeError, KeyError):
            pass

    supervised = {}
    for method, subdir in [('UNet', 'unet'), ('DINOv2', 'dinov2')]:
        res_path = os.path.join(result_base, subdir, 'results.json')
        if not os.path.exists(res_path):
            continue
        with open(res_path) as f:
            res = json.load(f)

        entries = []
        for pid_str, dice in res.items():
            pid_int = int(pid_str)
            pid_padded = f'patient{pid_int:03d}'
            ef_data = existing_ef.get(method, {}).get(pid_padded, {})
            entries.append({
                'pid':      pid_padded,
                'group':    ACDC_VAL_GROUP.get(pid_int, 'UNK'),
                'dice_RV':  dice.get('RV', 0.0),
                'dice_Myo': dice.get('Myo', 0.0),
                'dice_LV':  dice.get('LV', 0.0),
                'hd95_RV':  None, 'hd95_Myo': None, 'hd95_LV': None,
                'assd_RV':  None, 'assd_Myo': None, 'assd_LV': None,
                'pred_EF':  ef_data.get('pred_EF'),
                'pred_EDV': ef_data.get('pred_EDV'),
                'pred_ESV': ef_data.get('pred_ESV'),
                'gt_EF':    ef_data.get('gt_EF'),
                'gt_EDV':   ef_data.get('gt_EDV'),
                'gt_ESV':   ef_data.get('gt_ESV'),
            })
        if entries:
            supervised[method] = entries

    return supervised


# ── main evaluation ────────────────────────────────────────────────────────────

METHODS = {
    'MedSAM2_Dual':    {'dir': 'medsam2',    'key': 'bidir',   'eval': 'es'},
    'MedSAM2_ED':      {'dir': 'medsam2',    'key': 'ed_pred', 'eval': 'es'},
    'MedSAM2_ES':      {'dir': 'medsam2',    'key': 'es_pred', 'eval': 'ed'},
    'SAM2_ED':         {'dir': 'sam2',       'key': 'ed_pred', 'eval': 'es'},
    'SAM2_Dual':       {'dir': 'sam2_bidir', 'key': 'bidir',   'eval': 'es'},
    'UNet':            {'dir': 'unet_mnm',   'key': 'bidir',   'eval': 'es'},
    'DINOv2':          {'dir': 'dinov2_mnm', 'key': 'bidir',   'eval': 'es'},
}


def get_patient_ids(dataset, prep_dir):
    """Return list of patient ID strings (matching NPZ file prefixes)."""
    all_npzs = glob(os.path.join(prep_dir, '*.npz'))
    # Extract unique patient IDs (everything before _slice)
    ids = sorted(set(os.path.basename(f).split('_slice')[0] for f in all_npzs))
    return ids


DIR_REMAP = {
    'acdc_test': {'medsam2': 'medsam2_acdc_test'},
    'mnm':       {'medsam2': 'medsam2_mnm', 'sam2_bidir': 'sam2_bidir_mnm', 'sam2': 'sam2_mnm',
                  'unet_mnm': 'unet_mnm', 'dinov2_mnm': 'dinov2_mnm'},
    'mnm2':      {'medsam2': 'medsam2_mnm2'},
}


def evaluate_dataset(dataset, prep_dir, result_base):
    """
    Evaluate all methods on a dataset.
    Returns dict: method -> list of per-patient metric dicts.
    """
    all_ids = get_patient_ids(dataset, prep_dir)
    if dataset == 'acdc_val':
        # Filter to val IDs only
        val_strs = {f'patient{pid:03d}' for pid in ACDC_VAL_IDS}
        all_ids  = [pid for pid in all_ids if pid in val_strs]
    print(f"Dataset {dataset}: {len(all_ids)} patients, methods: {list(METHODS.keys())}")

    remap = DIR_REMAP.get(dataset, {})
    # acdc_test only has ED-frame GT; evaluate at ED frame for all methods
    ed_only = (dataset == 'acdc_test')
    results = defaultdict(list)

    for pid in tqdm(all_ids, desc=dataset):
        for method, cfg in METHODS.items():
            remapped_dir = remap.get(cfg['dir'], cfg['dir'])
            res_dir = os.path.join(result_base, remapped_dir)
            if not os.path.isdir(res_dir):
                continue

            data = load_patient_slices(pid, prep_dir, res_dir, cfg['key'], ed_only=ed_only)
            if data is None:
                continue

            pred_3d  = data['pred_3d']
            gt_3d    = data['gt_3d']
            pred_all = data['pred_all']
            pixdim   = data['pixdim']

            # ── Dice ──────────────────────────────────────────────────────────
            dice = {CLASSES[c]: dice_np(pred_3d, gt_3d, c) for c in [1, 2, 3]}

            # ── HD95 + ASSD ───────────────────────────────────────────────────
            hd = {}
            ad = {}
            for c in [1, 2, 3]:
                h, a = compute_hd95_assd_3d(pred_3d, gt_3d, c, pixdim)
                hd[CLASSES[c]] = h
                ad[CLASSES[c]] = a

            # ── EF / EDV / ESV ────────────────────────────────────────────────
            ef, edv, esv = compute_ef_edv_esv(
                pred_all, data['ed_idx'], data['es_idx'], pixdim
            )
            gt_ef, gt_edv, gt_esv = compute_ef_edv_esv(
                np.concatenate([
                    data['gt_3d_ed'][np.newaxis],
                    data['gt_3d_es'][np.newaxis],
                ], axis=0).reshape(2, *pred_all.shape[1:]),
                0, 1, pixdim
            ) if False else (np.nan, np.nan, np.nan)  # GT EF computed separately below

            # GT EF: use max-volume frame as EDV (robust to ED/ES convention differences)
            gt_vol_ed = compute_lv_volume(data['gt_3d_ed'], pixdim)
            gt_vol_es = compute_lv_volume(data['gt_3d_es'], pixdim)
            gt_EDV = max(gt_vol_ed, gt_vol_es)
            gt_ESV = min(gt_vol_ed, gt_vol_es)
            gt_EF  = (gt_EDV - gt_ESV) / gt_EDV * 100.0 if gt_EDV > 1e-3 else np.nan

            results[method].append({
                'pid':      pid,
                'group':    data['group'],
                'dice_RV':  dice['RV'],
                'dice_Myo': dice['Myo'],
                'dice_LV':  dice['LV'],
                'hd95_RV':  hd['RV'],
                'hd95_Myo': hd['Myo'],
                'hd95_LV':  hd['LV'],
                'assd_RV':  ad['RV'],
                'assd_Myo': ad['Myo'],
                'assd_LV':  ad['LV'],
                'pred_EF':  ef,
                'pred_EDV': edv,
                'pred_ESV': esv,
                'gt_EF':    gt_EF,
                'gt_EDV':   gt_EDV,
                'gt_ESV':   gt_ESV,
            })

    return results


def summarise_results(results):
    """Print summary tables and save CSVs."""
    import csv, io

    print("\n" + "="*80)
    print(f"{'Method':<22} {'RV Dice':>9} {'Myo Dice':>9} {'LV Dice':>9} "
          f"{'RV HD95':>9} {'LV HD95':>9} {'LV ASSD':>8}")
    print("-"*80)

    surface_rows = []
    clinical_rows = []

    def _valid(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    for method, plist in results.items():
        if not plist:
            continue
        rv_d  = [p['dice_RV']  for p in plist if _valid(p.get('dice_RV'))]
        myo_d = [p['dice_Myo'] for p in plist if _valid(p.get('dice_Myo'))]
        lv_d  = [p['dice_LV']  for p in plist if _valid(p.get('dice_LV'))]
        rv_h  = [p['hd95_RV']  for p in plist if _valid(p.get('hd95_RV'))]
        lv_h  = [p['hd95_LV']  for p in plist if _valid(p.get('hd95_LV'))]
        lv_a  = [p['assd_LV']  for p in plist if _valid(p.get('assd_LV'))]
        ef_e  = [abs(p['pred_EF'] - p['gt_EF'])
                 for p in plist
                 if _valid(p.get('pred_EF')) and _valid(p.get('gt_EF'))]

        def m(lst): return f"{np.mean(lst):.3f}±{np.std(lst):.3f}" if lst else "N/A"

        print(f"{method:<22} {m(rv_d):>9} {m(myo_d):>9} {m(lv_d):>9} "
              f"{m(rv_h):>9} {m(lv_h):>9} {m(lv_a):>8}")

        surface_rows.append({
            'Method':     method,
            'RV_Dice':    f"{np.mean(rv_d):.4f}" if rv_d else '',
            'Myo_Dice':   f"{np.mean(myo_d):.4f}" if myo_d else '',
            'LV_Dice':    f"{np.mean(lv_d):.4f}" if lv_d else '',
            'RV_Dice_std':   f"{np.std(rv_d):.4f}" if rv_d else '',
            'Myo_Dice_std':  f"{np.std(myo_d):.4f}" if myo_d else '',
            'LV_Dice_std':   f"{np.std(lv_d):.4f}" if lv_d else '',
            'RV_HD95':    f"{np.mean(rv_h):.2f}" if rv_h else '',
            'LV_HD95':    f"{np.mean(lv_h):.2f}" if lv_h else '',
            'LV_ASSD':    f"{np.mean(lv_a):.2f}" if lv_a else '',
            'RV_HD95_std': f"{np.std(rv_h):.2f}" if rv_h else '',
            'LV_HD95_std': f"{np.std(lv_h):.2f}" if lv_h else '',
        })
        clinical_rows.append({
            'Method':     method,
            'EF_MAE':     f"{np.mean(ef_e):.2f}" if ef_e else '',
            'EF_MAE_std': f"{np.std(ef_e):.2f}" if ef_e else '',
            'N_patients': len(plist),
        })

    print("="*80)
    return surface_rows, clinical_rows


def main(args):
    dataset  = args.dataset
    prep_dir = PREP_DIRS.get(dataset)
    if prep_dir is None or not os.path.isdir(prep_dir):
        print(f"ERROR: prep dir not found for dataset '{dataset}': {prep_dir}")
        print(f"  Run the appropriate prep_*.py first.")
        sys.exit(1)

    results = evaluate_dataset(dataset, prep_dir, RESULTS_BASE)

    # Save per-patient JSON (merge: preserve keys from other scripts, e.g. UNet)
    os.makedirs(RESULTS_BASE, exist_ok=True)
    json_path = os.path.join(RESULTS_BASE, f'metrics_{dataset}.json')
    existing = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    def nan_to_none(d):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in d.items()}

    existing.update({m: [nan_to_none(p) for p in pl] for m, pl in results.items()})
    with open(json_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nPer-patient JSON saved to {json_path}")

    # Add supervised baselines (UNet, DINOv2) for CSV tables only
    # (their Dice comes from results.json; EF is pulled from the metrics JSON if present)
    results_for_csv = dict(results)
    results_for_csv.update(_load_supervised_results(dataset, RESULTS_BASE, json_path))

    surface_rows, clinical_rows = summarise_results(results_for_csv)

    import csv
    surf_csv = os.path.join(RESULTS_BASE, f'table_surface_{dataset}.csv')
    clin_csv = os.path.join(RESULTS_BASE, f'table_clinical_{dataset}.csv')

    if surface_rows:
        with open(surf_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=surface_rows[0].keys())
            w.writeheader(); w.writerows(surface_rows)
        print(f"Surface metrics CSV → {surf_csv}")

    if clinical_rows:
        with open(clin_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=clinical_rows[0].keys())
            w.writeheader(); w.writerows(clinical_rows)
        print(f"Clinical metrics CSV → {clin_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='acdc_val',
                        choices=['acdc_val', 'acdc_test', 'mnm', 'mnm2'])
    main(parser.parse_args())
