"""
infer_sam2.py
Ablation Exp D: same pipeline as infer_medsam2.py but using vanilla SAM2 weights
(no medical fine-tuning) to quantify the benefit of MedSAM2's domain adaptation.

Runs forward propagation from ED (prompt=ED, evaluate at ES), matching Exp A.
Outputs: results/sam2/patientXXX_sliceZZ.npz  keys: ed_pred, ed_idx, es_idx, pixdim, group
"""

import os, sys, argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

MEDSAM2_DIR = '/scratch/gautschi/li4533/MIUA_2026/MedSAM2'
sys.path.insert(0, MEDSAM2_DIR)
if os.getcwd() != MEDSAM2_DIR:
    os.chdir(MEDSAM2_DIR)

from sam2.build_sam import build_sam2_video_predictor_npz   # noqa: E402

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def run_propagation(predictor, frames_f16, prompt_mask, prompt_idx):
    T      = frames_f16.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img    = torch.from_numpy(frames_f16.astype(np.float32)).to(device)
    mean   = IMG_MEAN.to(device)
    std    = IMG_STD.to(device)
    img    = (img - mean) / std

    pred = np.zeros((T, 512, 512), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(img, video_height=512, video_width=512)
        for cls_id in [1, 2, 3]:
            cls_mask = (prompt_mask == cls_id)
            if cls_mask.sum() < 10:
                continue
            y_idx, x_idx = np.where(cls_mask)
            bbox = np.array([
                max(0,   x_idx.min() - 5),
                max(0,   y_idx.min() - 5),
                min(511, x_idx.max() + 5),
                min(511, y_idx.max() + 5),
            ], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt_idx,
                obj_id=cls_id,
                box=bbox,
            )
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                pred[fidx][logits[i, 0].cpu().numpy() > 0.0] = oid
        predictor.reset_state(state)

    return pred


def main(args):
    os.makedirs(args.out, exist_ok=True)

    # Use MedSAM2 repo's NPZ API but with vanilla SAM2 checkpoint
    predictor = build_sam2_video_predictor_npz(args.cfg, args.ckpt, device='cuda')
    print(f"Loaded SAM2 predictor from {args.ckpt}")

    npz_files = sorted(glob(os.path.join(args.data, '*.npz')))
    print(f"Processing {len(npz_files)} slice files")

    for npz_path in tqdm(npz_files):
        stem     = os.path.basename(npz_path).replace('.npz', '')
        out_path = os.path.join(args.out, stem + '.npz')
        if os.path.exists(out_path):
            continue

        d      = np.load(npz_path, allow_pickle=True)
        frames = d['frames']
        ed_mask = d['ed_mask']
        ed_idx  = int(d['ed_idx'])
        es_idx  = int(d['es_idx'])

        pred_fwd = run_propagation(predictor, frames, ed_mask, ed_idx)

        np.savez_compressed(
            out_path,
            ed_pred = pred_fwd,   # key matches MedSAM2 output for consistent evaluation
            ed_idx  = np.int32(ed_idx),
            es_idx  = np.int32(es_idx),
            pixdim  = d['pixdim'],
            group   = d['group'],
        )

    print(f"Done. Results saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/sam2.1_hiera_tiny.pt')
    parser.add_argument('--cfg',  default='configs/sam2.1_hiera_t512.yaml')
    parser.add_argument('--data', default='/scratch/gautschi/li4533/MIUA_2026/preprocessed')
    parser.add_argument('--out',  default='/scratch/gautschi/li4533/MIUA_2026/results/sam2')
    main(parser.parse_args())
