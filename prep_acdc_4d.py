"""
prep_acdc_4d.py
Preprocess ACDC 4D cardiac cine MRI into per-slice NPZ files ready for MedSAM2.

For each training patient:
  - Parse Info.cfg (ED/ES frame indices, pathology group, pixel spacing)
  - Load 4D NIfTI (H, W, Z, T) + ED/ES GT masks (H, W, Z)
  - For each spatial slice z: extract temporal sequence, normalize, resize to 512x512
  - Save results/preprocessed/patientXXX_sliceZZ.npz
"""

import argparse
import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
from glob import glob
from tqdm import tqdm

# ── copy of resize_grayscale_to_rgb_and_resize from MedSAM2/medsam2_infer_3D_CT.py ──
def resize_grayscale_to_rgb_and_resize(array, image_size):
    """array: (T, H, W) uint8 → returns (T, 3, image_size, image_size) float64 in [0,255]"""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, H, W)
        resized_array[i] = img_array
    return resized_array  # values in [0, 255]


def parse_info_cfg(cfg_path):
    """Parse ACDC Info.cfg file. Returns dict with ED, ES, Group, NbFrame, Height, Weight."""
    info = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                info[key.strip()] = val.strip()
    return info


def find_frame_files(patient_dir, pid):
    """Find ED and ES NIfTI frame files. Returns (ed_img, ed_gt, es_img, es_gt) paths."""
    # Files: patientXXX_frame01.nii.gz, patientXXX_frame01_gt.nii.gz etc.
    nii_files = sorted(glob(os.path.join(patient_dir, f'patient{pid:03d}_frame*.nii.gz')))
    # GT files have _gt suffix
    gt_files  = [f for f in nii_files if '_gt' in f]
    img_files = [f for f in nii_files if '_gt' not in f and '4d' not in f]
    # There should be exactly 2 of each (ED and ES)
    assert len(img_files) == 2, f"Expected 2 frame files for patient {pid}, got {img_files}"
    assert len(gt_files)  == 2, f"Expected 2 gt files for patient {pid}, got {gt_files}"
    # Sort by frame number; first is ED (lower frame number), second is ES
    img_files.sort()
    gt_files.sort()
    return img_files[0], gt_files[0], img_files[1], gt_files[1]


def main(args):
    os.makedirs(args.out, exist_ok=True)

    patient_dirs = sorted(glob(os.path.join(args.db, 'patient*')))
    print(f"Found {len(patient_dirs)} patients in {args.db}")

    for pdir in tqdm(patient_dirs):
        pid = int(os.path.basename(pdir).replace('patient', ''))

        # ── Parse Info.cfg ──
        cfg_path = os.path.join(pdir, 'Info.cfg')
        if not os.path.exists(cfg_path):
            print(f"  Skipping patient {pid}: no Info.cfg")
            continue
        info = parse_info_cfg(cfg_path)
        ed_idx = int(info['ED']) - 1   # convert 1-based → 0-based
        es_idx = int(info['ES']) - 1
        group  = info.get('Group', 'UNK')

        # ── Load 4D volume ──
        nii4d_path = os.path.join(pdir, f'patient{pid:03d}_4d.nii.gz')
        if not os.path.exists(nii4d_path):
            print(f"  Skipping patient {pid}: no 4D file")
            continue
        nii4d  = nib.load(nii4d_path)
        vol4d  = nii4d.get_fdata(dtype=np.float32)   # (H, W, Z, T)
        pixdim = np.array(nii4d.header.get_zooms()[:3], dtype=np.float32)  # (dx, dy, dz) mm

        H, W, Z, T = vol4d.shape

        # ── Load GT masks ──
        ed_img_f, ed_gt_f, es_img_f, es_gt_f = find_frame_files(pdir, pid)
        ed_gt = nib.load(ed_gt_f).get_fdata(dtype=np.float32).astype(np.uint8)  # (H, W, Z)
        es_gt = nib.load(es_gt_f).get_fdata(dtype=np.float32).astype(np.uint8)  # (H, W, Z)

        # ── Per-slice processing ──
        for z in range(Z):
            frames_hw = vol4d[:, :, z, :]          # (H, W, T)
            frames_thw = frames_hw.transpose(2, 0, 1)  # (T, H, W)

            # Percentile normalization → uint8
            p2  = np.percentile(frames_thw, 2)
            p98 = np.percentile(frames_thw, 98)
            frames_u8 = np.clip(
                (frames_thw - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255
            ).astype(np.uint8)

            # Resize to RGB 512×512: (T, 3, 512, 512), values in [0,255]
            frames_rgb = resize_grayscale_to_rgb_and_resize(frames_u8, 512)

            # Divide by 255 (ImageNet mean/std applied at inference time on GPU)
            frames_norm = (frames_rgb / 255.0).astype(np.float16)  # save as f16

            # GT masks for this slice, resized to 512×512 with NEAREST
            ed_mask_z = ed_gt[:, :, z]
            es_mask_z = es_gt[:, :, z]
            ed_mask_512 = np.array(
                Image.fromarray(ed_mask_z).resize((512, 512), Image.NEAREST), dtype=np.uint8
            )
            es_mask_512 = np.array(
                Image.fromarray(es_mask_z).resize((512, 512), Image.NEAREST), dtype=np.uint8
            )

            # Skip slices with zero cardiac content in BOTH ED and ES
            if ed_mask_512.max() == 0 and es_mask_512.max() == 0:
                continue

            out_path = os.path.join(args.out, f'patient{pid:03d}_slice{z:02d}.npz')
            if os.path.exists(out_path):
                continue  # resume: skip already-processed slices
            np.savez_compressed(
                out_path,
                frames   = frames_norm,      # (T, 3, 512, 512) float16
                ed_mask  = ed_mask_512,      # (512, 512) uint8, labels 0-3
                es_mask  = es_mask_512,      # (512, 512) uint8, labels 0-3
                ed_idx   = np.int32(ed_idx),
                es_idx   = np.int32(es_idx),
                group    = np.bytes_(group),
                pixdim   = pixdim,           # (dx, dy, dz) mm
                orig_H   = np.int32(H),
                orig_W   = np.int32(W),
            )

    npz_count = len(glob(os.path.join(args.out, '*.npz')))
    print(f"\nDone. Saved {npz_count} slice NPZ files to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',  required=True, help='Path to ACDC database/training directory')
    parser.add_argument('--out', required=True, help='Output directory for preprocessed NPZ files')
    main(parser.parse_args())
