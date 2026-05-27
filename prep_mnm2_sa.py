"""
prep_mnm2_sa.py
Preprocess M&Ms-2 short-axis data into per-slice NPZ files for MedSAM2.

M&Ms-2 structure (per patient in dataset/):
  {ID}_SA_CINE.nii.gz      — 4D SA cine (H, W, Z, T)
  {ID}_SA_ED.nii.gz        — 3D SA ED frame image (H, W, Z)
  {ID}_SA_ED_gt.nii.gz     — 3D SA ED GT mask (H, W, Z)
  {ID}_SA_ES.nii.gz        — 3D SA ES frame image (H, W, Z)
  {ID}_SA_ES_gt.nii.gz     — 3D SA ES GT mask (H, W, Z)

ED/ES frame indices are found by NCC matching of SA_ED/SA_ES images to CINE frames.
Group: from dataset_information.csv (DISEASE column).

Label remapping: MnM2 uses 1=LV, 2=Myo, 3=RV — remapped to ACDC convention
(1=RV, 2=Myo, 3=LV) before saving.

Output: preprocessed_mnm2/{ID}_slice{ZZ}.npz  — same schema as preprocessed/ (ACDC).
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

MNM2_DIR    = '/scratch/gautschi/li4533/MIUA_2026/MnM2'
DATA_DIR    = os.path.join(MNM2_DIR, 'dataset')
META_CSV    = os.path.join(MNM2_DIR, 'dataset_information.csv')
DEFAULT_OUT = '/scratch/gautschi/li4533/MIUA_2026/preprocessed_mnm2'
IMAGE_SIZE  = 512


def resize_frames_batch(frames_u8, image_size):
    """frames_u8: (T, H, W) uint8 → (T, 3, image_size, image_size) float16 in [0,1]"""
    T, H, W = frames_u8.shape
    if H == image_size and W == image_size:
        norm = frames_u8.astype(np.float32) / 255.0
    else:
        resized = np.empty((T, image_size, image_size), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames_u8[i], (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
    return np.stack([norm, norm, norm], axis=1).astype(np.float16)  # (T,3,H,W)


def resize_mask(mask_hw, image_size):
    if mask_hw.shape[0] == image_size and mask_hw.shape[1] == image_size:
        return mask_hw
    return cv2.resize(mask_hw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def find_frame_by_ncc(cine: np.ndarray, ref: np.ndarray) -> int:
    """
    cine: (H, W, Z, T)  ref: (H, W, Z)
    Returns frame index with highest mean NCC over slices.
    Vectorised: avoids Python loop over T.
    """
    T, Z = cine.shape[3], cine.shape[2]
    # Flatten spatial dims: (H*W, Z, T) and (H*W, Z)
    HW = cine.shape[0] * cine.shape[1]
    c  = cine.reshape(HW, Z, T).astype(np.float32)
    r  = ref.reshape(HW, Z).astype(np.float32)

    # Subtract means
    c  = c - c.mean(axis=0, keepdims=True)
    r  = r - r.mean(axis=0, keepdims=True)

    c_std = c.std(axis=0, keepdims=True) + 1e-8     # (1, Z, T)
    r_std = r.std(axis=0, keepdims=True) + 1e-8     # (1, Z)

    # NCC per (z, t): mean over spatial
    ncc = (c * r[..., None]).mean(axis=0) / (c_std[0] * r_std[0, :, None])  # (Z, T)
    return int(ncc.mean(axis=0).argmax())


def remap_labels(mask):
    """MnM2: 1=LV, 2=Myo, 3=RV → ACDC: 1=RV, 2=Myo, 3=LV"""
    out = mask.copy()
    out[mask == 1] = 3  # LV→LV slot
    out[mask == 3] = 1  # RV→RV slot
    return out


def process_patient(task):
    """Worker: process one MnM2 patient directory."""
    pdir, meta, out_dir, overwrite = task
    pid = os.path.basename(pdir)

    cine_path  = os.path.join(pdir, f'{pid}_SA_CINE.nii.gz')
    ed_gt_path = os.path.join(pdir, f'{pid}_SA_ED_gt.nii.gz')
    es_gt_path = os.path.join(pdir, f'{pid}_SA_ES_gt.nii.gz')
    ed_img_path = os.path.join(pdir, f'{pid}_SA_ED.nii.gz')
    es_img_path = os.path.join(pdir, f'{pid}_SA_ES.nii.gz')

    for p in [cine_path, ed_img_path, es_img_path, ed_gt_path, es_gt_path]:
        if not os.path.exists(p):
            return pid, 0, f'missing:{os.path.basename(p)}'

    # Fast-skip check
    if not overwrite:
        nii_hdr = nib.load(cine_path)
        Z_peek = nii_hdr.shape[2]
        all_exist = all(
            os.path.exists(os.path.join(out_dir, f'{pid}_slice{z:02d}.npz'))
            for z in range(Z_peek)
        )
        if all_exist:
            return pid, 0, 'skipped_exists'

    try:
        nii_cine = nib.load(cine_path)
        vol4d    = nii_cine.get_fdata(dtype=np.float32)        # (H, W, Z, T)
        pixdim   = np.array(nii_cine.header.get_zooms()[:3], dtype=np.float32)

        ed_img = nib.load(ed_img_path).get_fdata(dtype=np.float32)  # (H, W, Z)
        es_img = nib.load(es_img_path).get_fdata(dtype=np.float32)

        ed_gt3d = nib.load(ed_gt_path).get_fdata(dtype=np.float32).astype(np.uint8)
        es_gt3d = nib.load(es_gt_path).get_fdata(dtype=np.float32).astype(np.uint8)
    except Exception as e:
        return pid, 0, f'load_error:{e}'

    H, W, Z, T = vol4d.shape

    ed_idx = find_frame_by_ncc(vol4d, ed_img)
    es_idx = find_frame_by_ncc(vol4d, es_img)

    if ed_idx == es_idx:
        ed_idx = 0

    # Remap MnM2 labels to ACDC convention
    ed_gt3d = remap_labels(ed_gt3d)
    es_gt3d = remap_labels(es_gt3d)

    info  = meta.get(pid, {'disease': 'UNK'})
    group = info['disease']

    # Global percentile normalise
    p2  = np.percentile(vol4d, 2)
    p98 = np.percentile(vol4d, 98)
    vol_u8 = np.clip((vol4d - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    # vol_u8: (H, W, Z, T)

    saved = 0
    for z in range(Z):
        out_path = os.path.join(out_dir, f'{pid}_slice{z:02d}.npz')
        if os.path.exists(out_path) and not overwrite:
            continue

        ed_slice = ed_gt3d[:, :, z] if ed_gt3d.ndim == 3 else ed_gt3d[:, :, z, 0]
        es_slice = es_gt3d[:, :, z] if es_gt3d.ndim == 3 else es_gt3d[:, :, z, 0]

        ed_mask_512 = resize_mask(ed_slice, IMAGE_SIZE)
        es_mask_512 = resize_mask(es_slice, IMAGE_SIZE)

        if ed_mask_512.max() == 0 and es_mask_512.max() == 0:
            continue

        frames_thw  = vol_u8[:, :, z, :].transpose(2, 0, 1)        # (T, H, W)
        frames_norm = resize_frames_batch(frames_thw, IMAGE_SIZE)   # (T,3,512,512) f16

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

    meta = {}
    if os.path.exists(META_CSV):
        df = pd.read_csv(META_CSV, low_memory=False)
        df = df.dropna(subset=['SUBJECT_CODE'])
        df['SUBJECT_CODE'] = df['SUBJECT_CODE'].astype(int)
        df = df.drop_duplicates(subset=['SUBJECT_CODE'])
        for _, row in df.iterrows():
            meta[str(int(row['SUBJECT_CODE'])).zfill(3)] = {
                'disease': str(row['DISEASE']),
                'vendor':  str(row.get('VENDOR', 'UNK')),
            }
    print(f"Loaded metadata for {len(meta)} patients from CSV")

    patient_dirs = sorted(glob(os.path.join(args.data_dir, '*')))
    patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patient dirs in {args.data_dir}")

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
            elif not status.startswith('missing'):
                print(f"  WARN {pid}: {status}")

    total = len(glob(os.path.join(args.out, '*.npz')))
    print(f"\nDone. Processed {processed} patients ({saved_total} NPZ written), "
          f"skipped {skipped} (already exist).")
    print(f"Total NPZ files in output dir: {total}. Output: {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--out',      default=DEFAULT_OUT)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers',  type=int, default=0,
                        help='Number of parallel workers (0 = all CPUs)')
    main(parser.parse_args())
