"""
compute_hd95_from_allframes.py
Compute HD95/ASSD for UNet and DINOv2 from their all-frames NPZ outputs,
then merge into metrics_acdc_val.json and regenerate paper_table1_complete.csv.

Run after job_unet_acdc_allframes and job_dinov2_acdc_allframes complete.
"""

import os, json, glob
import numpy as np
from tqdm import tqdm

try:
    from medpy.metric.binary import hd95 as compute_hd95, assd as compute_assd
    HAS_MEDPY = True
except ImportError:
    print("WARNING: medpy not available, HD95/ASSD will be skipped")
    HAS_MEDPY = False

PROJ_DIR     = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR     = os.path.join(PROJ_DIR, 'preprocessed')
METRICS_JSON = os.path.join(PROJ_DIR, 'results', 'metrics_acdc_val.json')

VAL_PIDS = [
    'patient017','patient018','patient019','patient020',
    'patient037','patient038','patient039','patient040',
    'patient057','patient058','patient059','patient060',
    'patient077','patient078','patient079','patient080',
    'patient097','patient098','patient099','patient100',
]

ALLFRAMES_DIRS = {
    'UNet':   os.path.join(PROJ_DIR, 'results', 'unet_acdc_allframes'),
    'DINOv2': os.path.join(PROJ_DIR, 'results', 'dinov2_acdc_allframes'),
}


def _decode_group(raw) -> str:
    s = str(raw)
    for pfx, sfx in [("np.bytes_(b'", "')"), ("b'", "'")]:
        if s.startswith(pfx) and s.endswith(sfx):
            return s[len(pfx):-len(sfx)]
    return s


def dice_coef(pred, gt, cls):
    p = (pred == cls); g = (gt == cls)
    if g.sum() == 0 and p.sum() == 0: return 1.0
    if g.sum() == 0: return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


def compute_patient_metrics(pid, npz_dir):
    """
    Load all-frames NPZ predictions for a patient, evaluate at ES frame.
    Returns dict with dice, hd95, assd for RV/Myo/LV.
    """
    prep_npzs = sorted(glob.glob(os.path.join(PREP_DIR, f'{pid}_slice*.npz')))
    pred_npzs = sorted(glob.glob(os.path.join(npz_dir,  f'{pid}_slice*.npz')))
    if not prep_npzs or not pred_npzs:
        return None

    d0     = np.load(prep_npzs[0], allow_pickle=True)
    es_idx = int(d0['es_idx'])
    ed_idx = int(d0['ed_idx'])
    pixdim = d0['pixdim'].astype(np.float64)
    orig_H = int(d0['orig_H'])
    orig_W = int(d0['orig_W'])
    group  = _decode_group(d0['group'])
    scale  = (orig_H / 512.0) * (orig_W / 512.0)
    voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale
    spacing   = (float(pixdim[2]), float(pixdim[1]) * orig_H / 512.0, float(pixdim[0]) * orig_W / 512.0)

    # Stack slices: ES-frame prediction vs GT ES mask
    pred_es_list = []; gt_es_list = []
    pred_ed_list = []; gt_ed_list = []
    pred_vol_vox = np.zeros(1, dtype=np.float64)  # just for EF
    gt_edv_vox = gt_esv_vox = pred_edv_vox = pred_esv_vox = 0.0

    for p_path, r_path in zip(prep_npzs, pred_npzs):
        pd_ = np.load(p_path, allow_pickle=True)
        rd  = np.load(r_path, allow_pickle=True)
        es_gt   = pd_['es_mask']   # (512,512)
        ed_gt   = pd_['ed_mask']
        pred_es = rd['bidir'][es_idx]   # (512,512)
        pred_ed = rd['bidir'][ed_idx]

        pred_es_list.append(pred_es)
        gt_es_list.append(es_gt)
        pred_ed_list.append(pred_ed)
        gt_ed_list.append(ed_gt)

        pred_esv_vox += (pred_es == 3).sum()
        pred_edv_vox += (pred_ed == 3).sum()
        gt_esv_vox   += (es_gt   == 3).sum()
        gt_edv_vox   += (ed_gt   == 3).sum()

    pred_es_3d = np.stack(pred_es_list, axis=0)   # (Z, 512, 512)
    gt_es_3d   = np.stack(gt_es_list,   axis=0)

    result = {
        'pid': pid, 'group': group,
        'dice_RV': 0.0, 'dice_Myo': 0.0, 'dice_LV': 0.0,
        'hd95_RV': None, 'hd95_Myo': None, 'hd95_LV': None,
        'assd_RV': None, 'assd_Myo': None, 'assd_LV': None,
        'pred_EF': None, 'pred_EDV': None, 'pred_ESV': None,
        'gt_EF': None,   'gt_EDV': None,   'gt_ESV': None,
    }

    for cls, name in [(1,'RV'), (2,'Myo'), (3,'LV')]:
        dices = [dice_coef(pred_es_list[z], gt_es_list[z], cls)
                 for z in range(len(pred_es_list))
                 if gt_es_list[z].max() > 0]
        result[f'dice_{name}'] = float(np.mean(dices)) if dices else 0.0

        if HAS_MEDPY:
            pred_bin = (pred_es_3d == cls).astype(bool)
            gt_bin   = (gt_es_3d   == cls).astype(bool)
            if pred_bin.any() and gt_bin.any():
                try:
                    result[f'hd95_{name}'] = float(compute_hd95(pred_bin, gt_bin, voxelspacing=spacing))
                    result[f'assd_{name}'] = float(compute_assd(pred_bin, gt_bin, voxelspacing=spacing))
                except Exception:
                    pass

    # EF/EDV/ESV
    pred_edv = pred_edv_vox * voxel_mm3 / 1000.0
    pred_esv = pred_esv_vox * voxel_mm3 / 1000.0
    gt_edv   = gt_edv_vox   * voxel_mm3 / 1000.0
    gt_esv   = gt_esv_vox   * voxel_mm3 / 1000.0
    pred_edv, pred_esv = max(pred_edv, pred_esv), min(pred_edv, pred_esv)
    gt_edv,   gt_esv   = max(gt_edv, gt_esv),     min(gt_edv, gt_esv)
    result['pred_EDV'] = float(pred_edv)
    result['pred_ESV'] = float(pred_esv)
    result['gt_EDV']   = float(gt_edv)
    result['gt_ESV']   = float(gt_esv)
    result['pred_EF']  = float(100*(pred_edv-pred_esv)/pred_edv) if pred_edv > 1e-3 else 0.0
    result['gt_EF']    = float(100*(gt_edv-gt_esv)/gt_edv)       if gt_edv > 1e-3 else 0.0

    return result


def main():
    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    for method, npz_dir in ALLFRAMES_DIRS.items():
        if not os.path.isdir(npz_dir):
            print(f"Skipping {method}: {npz_dir} not found")
            continue
        if not glob.glob(os.path.join(npz_dir, '*.npz')):
            print(f"Skipping {method}: no NPZ files in {npz_dir}")
            continue

        print(f"\nComputing HD95/ASSD for {method} from {npz_dir} ...")
        results = []
        for pid in tqdm(VAL_PIDS):
            r = compute_patient_metrics(pid, npz_dir)
            if r: results.append(r)

        if results:
            metrics[method] = results
            rv  = np.mean([r['dice_RV']  for r in results])
            myo = np.mean([r['dice_Myo'] for r in results])
            lv  = np.mean([r['dice_LV']  for r in results])
            rv_hd = [r['hd95_RV'] for r in results if r['hd95_RV'] is not None]
            lv_hd = [r['hd95_LV'] for r in results if r['hd95_LV'] is not None]
            print(f"  {method}: Dice RV={rv:.3f} Myo={myo:.3f} LV={lv:.3f}")
            if rv_hd:
                print(f"  HD95: RV={np.mean(rv_hd):.2f}  LV={np.mean(lv_hd):.2f}")

    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nUpdated {METRICS_JSON}")


if __name__ == '__main__':
    main()
