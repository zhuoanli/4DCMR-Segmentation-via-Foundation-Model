"""
prep_acdc_test.py
Preprocess ACDC test set (32 patients) into per-slice NPZ files for MedSAM2.
Identical pipeline to prep_acdc_4d.py; test patients have non-sequential IDs
(102, 104, 105, ...) and their Info.cfg has no Group field.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from glob import glob
from tqdm import tqdm

DEFAULT_DB  = '/scratch/gautschi/li4533/MIUA_2026/database/testing'
DEFAULT_OUT = '/scratch/gautschi/li4533/MIUA_2026/preprocessed_acdc_test'


def resize_grayscale_to_rgb(array, image_size):
    T = array.shape[0]
    out = np.zeros((T, 3, image_size, image_size), dtype=np.float32)
    for i in range(T):
        img = Image.fromarray(array[i]).convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)
        out[i] = np.array(img).transpose(2, 0, 1)
    return out


def parse_info_cfg(cfg_path):
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def find_frame_files(patient_dir, pid_str):
    nii_files = sorted(glob(os.path.join(patient_dir, f'{pid_str}_frame*.nii.gz')))
    gt_files  = [f for f in nii_files if '_gt' in f]
    img_files = [f for f in nii_files if '_gt' not in f and '4d' not in f]
    return img_files, gt_files


def main(args):
    os.makedirs(args.out, exist_ok=True)

    patient_dirs = sorted(glob(os.path.join(args.db, 'patient*')))
    print(f"Found {len(patient_dirs)} test patients in {args.db}")

    skipped = 0
    for pdir in tqdm(patient_dirs):
        pid_str = os.path.basename(pdir)          # e.g. "patient102"
        pid_int = int(pid_str.replace('patient', ''))

        cfg_path = os.path.join(pdir, 'Info.cfg')
        nii4d_path = os.path.join(pdir, f'{pid_str}_4d.nii.gz')

        if not os.path.exists(cfg_path) or not os.path.exists(nii4d_path):
            print(f"  Skipping {pid_str}: missing Info.cfg or 4D file")
            skipped += 1
            continue

        info   = parse_info_cfg(cfg_path)
        ed_idx = int(info['ED']) - 1   # 1-based → 0-based
        es_idx = int(info['ES']) - 1
        group  = info.get('Group', 'TEST')

        nii4d  = nib.load(nii4d_path)
        vol4d  = nii4d.get_fdata(dtype=np.float32)    # (H, W, Z, T)
        pixdim = np.array(nii4d.header.get_zooms()[:3], dtype=np.float32)
        H, W, Z, T = vol4d.shape

        # Load GT masks (there are exactly 2: ED and ES)
        img_files, gt_files = find_frame_files(pdir, pid_str)
        if len(gt_files) < 2:
            print(f"  Skipping {pid_str}: only {len(gt_files)} GT files found")
            skipped += 1
            continue

        gt_files.sort()
        ed_gt = nib.load(gt_files[0]).get_fdata(dtype=np.float32).astype(np.uint8)  # (H, W, Z)
        es_gt = nib.load(gt_files[1]).get_fdata(dtype=np.float32).astype(np.uint8)

        for z in range(Z):
            frames_thw = vol4d[:, :, z, :].transpose(2, 0, 1)  # (T, H, W)

            p2  = np.percentile(frames_thw, 2)
            p98 = np.percentile(frames_thw, 98)
            frames_u8 = np.clip(
                (frames_thw - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255
            ).astype(np.uint8)

            frames_rgb  = resize_grayscale_to_rgb(frames_u8, 512)
            frames_norm = (frames_rgb / 255.0).astype(np.float16)

            ed_mask_512 = np.array(
                Image.fromarray(ed_gt[:, :, z]).resize((512, 512), Image.NEAREST), dtype=np.uint8
            )
            es_mask_512 = np.array(
                Image.fromarray(es_gt[:, :, z]).resize((512, 512), Image.NEAREST), dtype=np.uint8
            )

            if ed_mask_512.max() == 0 and es_mask_512.max() == 0:
                continue

            out_path = os.path.join(args.out, f'{pid_str}_slice{z:02d}.npz')
            if os.path.exists(out_path):
                continue

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

    total = len(glob(os.path.join(args.out, '*.npz')))
    print(f"\nDone. {total} slice NPZ files saved (skipped {skipped} patients). → {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',  default=DEFAULT_DB)
    parser.add_argument('--out', default=DEFAULT_OUT)
    main(parser.parse_args())
