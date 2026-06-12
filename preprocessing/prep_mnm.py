"""
prep_mnm.py
Preprocess M&Ms dataset (all labeled splits: Testing/Training/Labeled/Validation)
into per-slice NPZ files for MedSAM2.

M&Ms structure (per patient):
  MnM/{Split}/{ID}/{ID}_sa.nii.gz       — 4D SA cine (H, W, Z, T)
  MnM/{Split}/{ID}/{ID}_sa_gt.nii.gz    — 4D GT, non-zero only at ED and ES frames

ED/ES indices: read from the open-dataset CSV (ED and ES columns, 0-indexed).
Group / Vendor: from the same CSV (Pathology + Vendor columns).

Output: preprocessed_mnm/{ID}_slice{ZZ}.npz  — same schema as preprocessed/ (ACDC).

Speed: uses multiprocessing (--workers, default = all CPUs) + cv2 batch resize.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
import functools

MNM_DIR    = '/scratch/gautschi/li4533/MIUA_2026/MnM'
META_CSV   = os.path.join(MNM_DIR, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
DEFAULT_OUT = '/scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm'

SPLIT_SUBDIRS = [
    'Testing',
    'Training/Labeled',
    'Validation',
]
IMAGE_SIZE = 512


def resize_frames_batch(frames_u8, image_size):
    """
    frames_u8 : (T, H, W) uint8
    Returns   : (T, 3, image_size, image_size) float16  in [0, 1]
    Uses cv2 batch resize — one C call per frame but no Python PIL overhead.
    Grayscale→RGB via numpy stacking (avoids PIL convert).
    """
    T, H, W = frames_u8.shape
    if H == image_size and W == image_size:
        norm = frames_u8.astype(np.float32) / 255.0
    else:
        resized = np.empty((T, image_size, image_size), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames_u8[i], (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
    # Stack gray channel × 3 for RGB
    return np.stack([norm, norm, norm], axis=1).astype(np.float16)  # (T,3,H,W)


def resize_mask(mask_hw, image_size):
    """(H, W) uint8 → (image_size, image_size) uint8 nearest-neighbour."""
    if mask_hw.shape[0] == image_size and mask_hw.shape[1] == image_size:
        return mask_hw
    return cv2.resize(mask_hw, (image_size, image_size),
                      interpolation=cv2.INTER_NEAREST)


def process_patient(task):
    """Worker function — processes one patient directory."""
    pdir, meta, out_dir, overwrite = task
    pid = os.path.basename(pdir)

    if pid not in meta:
        return pid, 0, 'not_in_csv'

    ed_idx = meta[pid]['ed']
    es_idx = meta[pid]['es']
    group  = f"{meta[pid]['pathology']}_{meta[pid]['vendor_id']}"

    sa_path = os.path.join(pdir, f'{pid}_sa.nii.gz')
    gt_path = os.path.join(pdir, f'{pid}_sa_gt.nii.gz')
    if not os.path.exists(sa_path) or not os.path.exists(gt_path):
        return pid, 0, 'missing_files'

    # Fast-skip: if all expected NPZ already exist and no overwrite
    if not overwrite:
        # Peek at Z without loading full data
        nii_hdr = nib.load(sa_path)
        Z_peek = nii_hdr.shape[2]
        all_exist = all(
            os.path.exists(os.path.join(out_dir, f'{pid}_slice{z:02d}.npz'))
            for z in range(Z_peek)
        )
        if all_exist:
            return pid, 0, 'skipped_exists'

    try:
        nii_sa = nib.load(sa_path)
        vol4d  = nii_sa.get_fdata(dtype=np.float32)       # (H, W, Z, T)
        pixdim = np.array(nii_sa.header.get_zooms()[:3], dtype=np.float32)

        gt4d = nib.load(gt_path).get_fdata(dtype=np.float32).astype(np.uint8)  # (H, W, Z, T)
    except Exception as e:
        return pid, 0, f'load_error:{e}'

    H, W, Z, T = vol4d.shape

    if ed_idx >= T or es_idx >= T:
        return pid, 0, f'idx_out_of_range(T={T},ed={ed_idx},es={es_idx})'

    ed_gt3d = gt4d[:, :, :, ed_idx]  # (H, W, Z)
    es_gt3d = gt4d[:, :, :, es_idx]

    # MnM uses 1=LV, 2=Myo, 3=RV — remap to ACDC convention (1=RV, 2=Myo, 3=LV)
    for arr in (ed_gt3d, es_gt3d):
        lv = arr == 1
        rv = arr == 3
        arr[lv] = 3
        arr[rv] = 1

    # Normalise ALL frames at once (vectorised — avoids per-frame percentile)
    p2  = np.percentile(vol4d, 2)
    p98 = np.percentile(vol4d, 98)
    vol_u8 = np.clip((vol4d - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    # vol_u8: (H, W, Z, T)

    saved = 0
    for z in range(Z):
        out_path = os.path.join(out_dir, f'{pid}_slice{z:02d}.npz')
        if os.path.exists(out_path) and not overwrite:
            continue

        ed_mask_512 = resize_mask(ed_gt3d[:, :, z], IMAGE_SIZE)
        es_mask_512 = resize_mask(es_gt3d[:, :, z], IMAGE_SIZE)

        if ed_mask_512.max() == 0 and es_mask_512.max() == 0:
            continue

        # frames_thw: (T, H, W) — slice z, all time frames
        frames_thw = vol_u8[:, :, z, :].transpose(2, 0, 1)   # (T, H, W)
        frames_norm = resize_frames_batch(frames_thw, IMAGE_SIZE)  # (T,3,512,512) f16

        np.savez_compressed(
            out_path,
            frames  = frames_norm,
            ed_mask = ed_mask_512,
            es_mask = es_mask_512,
            ed_idx  = np.int32(ed_idx),
            es_idx  = np.int32(es_idx),
            group   = np.bytes_(group),
            pixdim  = pixdim,
            orig_H  = np.int32(H),
            orig_W  = np.int32(W),
        )
        saved += 1

    return pid, saved, 'ok'


def main(args):
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(META_CSV)
    meta = {}
    for _, row in df.iterrows():
        pid = str(row['External code']).strip()
        meta[pid] = {
            'pathology': str(row['Pathology']),
            'vendor':    str(row['VendorName']),
            'vendor_id': str(row['Vendor']),
            'ed':        int(row['ED']),
            'es':        int(row['ES']),
        }
    print(f"Loaded metadata for {len(meta)} patients from CSV")

    patient_dirs = []
    for split in SPLIT_SUBDIRS:
        split_path = os.path.join(MNM_DIR, split)
        if os.path.isdir(split_path):
            dirs = sorted(glob(os.path.join(split_path, '*')))
            patient_dirs.extend(dirs)
            print(f"  {split}: {len(dirs)} patient dirs")
    print(f"Total patient dirs found: {len(patient_dirs)}")

    tasks = [(pdir, meta, args.out, args.overwrite) for pdir in patient_dirs]

    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count())
    print(f"Processing with {n_workers} parallel workers ...")

    skipped = saved_total = processed = 0
    with mp.Pool(processes=n_workers) as pool:
        for pid, saved, status in tqdm(
            pool.imap_unordered(process_patient, tasks),
            total=len(tasks)
        ):
            if status == 'ok':
                if saved > 0:
                    processed += 1
                    saved_total += saved
            elif status == 'skipped_exists':
                skipped += 1
            else:
                if status not in ('not_in_csv', 'missing_files'):
                    print(f"  WARN {pid}: {status}")

    total = len(glob(os.path.join(args.out, '*.npz')))
    print(f"\nDone. Processed {processed} patients ({saved_total} NPZ written), "
          f"skipped {skipped} (already exist).")
    print(f"Total NPZ files in output dir: {total}. Output: {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',       default=DEFAULT_OUT)
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing NPZ files (default: skip)')
    parser.add_argument('--workers',   type=int, default=0,
                        help='Number of parallel workers (0 = all CPUs)')
    main(parser.parse_args())
