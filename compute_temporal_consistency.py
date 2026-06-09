"""
compute_temporal_consistency.py
Compute temporal jitter and physiological violation counts from existing
MedSAM2 predictions (ed_pred, es_pred, bidir keys) for ACDC val patients.

Outputs: results/temporal_consistency_acdc_val.csv
"""

import os, glob, csv
import numpy as np
from tqdm import tqdm

PROJ_DIR    = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR    = os.path.join(PROJ_DIR, 'preprocessed')
MEDSAM2_DIR = os.path.join(PROJ_DIR, 'results', 'medsam2')
OUT_CSV     = os.path.join(PROJ_DIR, 'results', 'temporal_consistency_acdc_val.csv')

VAL_PIDS = {
    'patient017', 'patient018', 'patient019', 'patient020',
    'patient037', 'patient038', 'patient039', 'patient040',
    'patient057', 'patient058', 'patient059', 'patient060',
    'patient077', 'patient078', 'patient079', 'patient080',
    'patient097', 'patient098', 'patient099', 'patient100',
}

STRATEGIES = {
    'ED-anchored': 'ed_pred',
    'ES-anchored': 'es_pred',
    'Dual-anchored': 'bidir',
}


def lv_volume_sequence(res_npzs, pred_key, voxel_mm3):
    """Return (T,) LV volume in mL for one patient across all slices."""
    if not res_npzs:
        return None
    T = np.load(res_npzs[0], allow_pickle=True)[pred_key].shape[0]
    lv_voxels = np.zeros(T, dtype=np.float64)
    for path in res_npzs:
        rd = np.load(path, allow_pickle=True)
        if pred_key not in rd:
            return None
        preds = rd[pred_key]   # (T, 512, 512)
        for t in range(min(T, preds.shape[0])):
            lv_voxels[t] += (preds[t] == 3).sum()
    return lv_voxels * voxel_mm3 / 1000.0   # mL


def temporal_jitter(vol_ml):
    """Mean |V(t+1)-2V(t)+V(t-1)| (second-order finite difference, mL/frame^2)."""
    if len(vol_ml) < 3:
        return float('nan')
    d2 = vol_ml[2:] - 2 * vol_ml[1:-1] + vol_ml[:-2]
    return float(np.mean(np.abs(d2)))


def violation_counts(vol_ml, ed_idx, es_idx):
    """
    Systolic violations:  V(t+1) > V(t) during ED→ES (volume should decrease).
    Diastolic violations: V(t+1) < V(t) during ES→end (volume should increase).
    Returns (systolic_count, diastolic_count).
    """
    T = len(vol_ml)
    # Systolic phase: ed_idx → es_idx
    sys_phase = vol_ml[ed_idx:es_idx + 1]
    sys_viols = int(np.sum(np.diff(sys_phase) > 0)) if len(sys_phase) > 1 else 0

    # Diastolic phase: es_idx → end (+ wrap-around to ed_idx if needed)
    # Use es_idx → T-1 as diastolic
    dia_phase = vol_ml[es_idx:]
    dia_viols = int(np.sum(np.diff(dia_phase) < 0)) if len(dia_phase) > 1 else 0

    return sys_viols, dia_viols


def _decode_group(raw) -> str:
    s = str(raw)
    for pfx, sfx in [("np.bytes_(b'", "')"), ("b'", "'")]:
        if s.startswith(pfx) and s.endswith(sfx):
            return s[len(pfx):-len(sfx)]
    return s


def main():
    rows = []
    pids_sorted = sorted(VAL_PIDS)

    for pid in tqdm(pids_sorted, desc='Patients'):
        prep_npzs = sorted(glob.glob(os.path.join(PREP_DIR,    f'{pid}_slice*.npz')))
        res_npzs  = sorted(glob.glob(os.path.join(MEDSAM2_DIR, f'{pid}_slice*.npz')))
        if not prep_npzs or not res_npzs:
            print(f"  Skipping {pid}: missing prep or result NPZ")
            continue

        d0 = np.load(prep_npzs[0], allow_pickle=True)
        ed_idx = int(d0['ed_idx'])
        es_idx = int(d0['es_idx'])
        pixdim = d0['pixdim'].astype(np.float64)
        orig_H = int(d0['orig_H'])
        orig_W = int(d0['orig_W'])
        group  = _decode_group(d0['group'])
        scale  = (orig_H / 512.0) * (orig_W / 512.0)
        voxel_mm3 = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale

        for strategy, pred_key in STRATEGIES.items():
            vol = lv_volume_sequence(res_npzs, pred_key, voxel_mm3)
            if vol is None:
                continue
            jitter = temporal_jitter(vol)
            sys_v, dia_v = violation_counts(vol, ed_idx, es_idx)
            rows.append({
                'pid': pid,
                'group': group,
                'strategy': strategy,
                'jitter_ml_frame2': round(jitter, 4),
                'systolic_violations': sys_v,
                'diastolic_violations': dia_v,
                'total_violations': sys_v + dia_v,
                'n_frames': len(vol),
                'ed_idx': ed_idx,
                'es_idx': es_idx,
            })

    # Write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ['pid', 'group', 'strategy', 'jitter_ml_frame2',
                  'systolic_violations', 'diastolic_violations',
                  'total_violations', 'n_frames', 'ed_idx', 'es_idx']
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT_CSV}  ({len(rows)} rows)")

    # Print summary
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    print("\n=== Temporal Consistency Summary (mean ± std across 20 patients) ===")
    for strat in ['ED-anchored', 'ES-anchored', 'Dual-anchored']:
        sub = df[df['strategy'] == strat]
        j = sub['jitter_ml_frame2']
        sv = sub['systolic_violations']
        dv = sub['diastolic_violations']
        print(f"{strat:15s}  Jitter={j.mean():.3f}±{j.std():.3f}  "
              f"SysViol={sv.mean():.2f}±{sv.std():.2f}  "
              f"DiaViol={dv.mean():.2f}±{dv.std():.2f}")


if __name__ == '__main__':
    main()
