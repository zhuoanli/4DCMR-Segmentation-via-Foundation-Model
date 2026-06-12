"""
infer_medsam2.py
Run MedSAM2 video propagation on ACDC 4D cardiac cine MRI slices.

MedSAM2 input:  T-frame video (T,3,512,512) + GT bbox prompt at ONE frame
MedSAM2 output: per-frame binary logit map (512,512) — run once per class (RV/Myo/LV)
Propagation: causal memory bank; error accumulates with distance from prompt.

Experiments differ in which GT frame is used as prompt (never used for evaluation):

  A) Prompt=ED → propagate to all T frames; evaluate Dice at ES
       fwd from ED: [ed_idx → T-1]
       bwd from ED: [ed_idx → 0]  (skipped if ed_idx==0, most ACDC patients)
       ed_pred = merge(bwd, fwd)
       Dice reported: ed_pred[es_idx] vs es_gt

  B) Prompt=ES → propagate to all T frames; evaluate Dice at ED
       bwd from ES: [es_idx → 0]
       fwd from ES: [es_idx → T-1]
       es_pred = merge(bwd, fwd)
       Dice reported: es_pred[ed_idx] vs ed_gt

  C) Bidir: combine A and B — each frame uses its nearest anchor:
       [0,      ed_idx] → ed_pred    (ED-anchored, pre-ED late diastole)
       [ed_idx, mid   ] → ed_pred    (ED-anchored, closer to ED)
       [mid+1,  es_idx] → es_pred    (ES-anchored, closer to ES)
       [es_idx, T-1   ] → es_pred    (ES-anchored, post-ES diastole)
       where mid = (ed_idx + es_idx) // 2
       Dice: bidir[es_idx] vs es_gt  AND  bidir[ed_idx] vs ed_gt
       (= same numbers as A@ES and B@ED respectively, since bidir reuses those passes)
       Clinical value: intermediate-frame quality is better than A alone (max drift
       reduced from T-1-ed_idx to ~(T-1-es_idx), enabling reliable time-volume curves)

Outputs one NPZ per slice: results/medsam2/patientXXX_sliceZZ.npz
  keys: ed_pred (T,512,512), es_pred (T,512,512), bidir (T,512,512),
        ed_idx, es_idx, pixdim, group, orig_H, orig_W
"""

import os, sys, argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

# MedSAM2 repo must be on sys.path AND the CWD must be the MedSAM2 dir
# (Hydra resolves configs relative to CWD).
# job_medsam2.sh sets: cd /scratch/gautschi/li4533/MIUA_2026/MedSAM2
MEDSAM2_DIR = '/scratch/gautschi/li4533/MIUA_2026/MedSAM2'
sys.path.insert(0, MEDSAM2_DIR)
if os.getcwd() != MEDSAM2_DIR:
    os.chdir(MEDSAM2_DIR)

from sam2.build_sam import build_sam2_video_predictor_npz   # noqa: E402

# ImageNet normalisation constants (applied on GPU after /255 in prep)
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def build_predictor(cfg: str, ckpt: str, device: str = 'cuda'):
    return build_sam2_video_predictor_npz(cfg, ckpt, device=device)


def run_propagation(predictor, frames_f16: np.ndarray,
                    prompt_mask: np.ndarray, prompt_idx: int,
                    reverse: bool = False,
                    bbox_noise: float = 0.0) -> np.ndarray:
    """
    frames_f16  : (T, 3, 512, 512) float16, values already in [0,1]
    prompt_mask : (512, 512) uint8, labels {0,1,2,3}
    bbox_noise  : std of per-corner jitter as fraction of box dimension (0=disabled)
    prompt_idx  : frame index of the prompt (ED or ES)
    reverse     : propagate backwards in time

    Returns pred : (T, 512, 512) uint8
    """
    T = frames_f16.shape[0]
    device = next(predictor.parameters()).device if hasattr(predictor, 'parameters') else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to GPU and apply ImageNet normalisation
    img = torch.from_numpy(frames_f16.astype(np.float32)).to(device)  # (T,3,512,512)
    mean = IMG_MEAN.to(device)
    std  = IMG_STD.to(device)
    img  = (img - mean) / std

    pred = np.zeros((T, 512, 512), dtype=np.uint8)

    # If the prompt mask is empty (e.g. apical slice with no cardiac structure
    # at the ES frame), return all-zeros without calling propagate_in_video.
    if prompt_mask.max() == 0:
        return pred

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(img, video_height=512, video_width=512)

        # Add one per-class bbox prompt (pattern from medsam2_infer_3D_CT.py)
        n_prompted = 0
        for cls_id in [1, 2, 3]:   # RV=1, Myo=2, LV=3
            cls_mask = (prompt_mask == cls_id)
            if cls_mask.sum() < 10:   # skip near-empty structures
                continue
            y_idx, x_idx = np.where(cls_mask)
            bbox = np.array([
                max(0,   x_idx.min() - 5),
                max(0,   y_idx.min() - 5),
                min(511, x_idx.max() + 5),
                min(511, y_idx.max() + 5),
            ], dtype=np.float32)
            if bbox_noise > 0.0:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                bbox[0] = np.clip(bbox[0] + np.random.normal(0, bbox_noise * w), 0, 511)
                bbox[1] = np.clip(bbox[1] + np.random.normal(0, bbox_noise * h), 0, 511)
                bbox[2] = np.clip(bbox[2] + np.random.normal(0, bbox_noise * w), 0, 511)
                bbox[3] = np.clip(bbox[3] + np.random.normal(0, bbox_noise * h), 0, 511)
                if bbox[0] > bbox[2]: bbox[0], bbox[2] = bbox[2], bbox[0]
                if bbox[1] > bbox[3]: bbox[1], bbox[3] = bbox[3], bbox[1]
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt_idx,
                obj_id=cls_id,
                box=bbox,
            )
            n_prompted += 1

        if n_prompted == 0:
            predictor.reset_state(state)
            return pred

        for fidx, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
            for i, oid in enumerate(obj_ids):
                binary = logits[i, 0].cpu().numpy() > 0.0  # (512,512) bool
                pred[fidx][binary] = oid

        predictor.reset_state(state)

    return pred


def main(args):
    os.makedirs(args.out, exist_ok=True)

    predictor = build_predictor(args.cfg, args.ckpt)
    print(f"Loaded predictor from {args.ckpt}")

    npz_files = sorted(glob(os.path.join(args.data, '*.npz')))
    print(f"Processing {len(npz_files)} slice files from {args.data}")

    for npz_path in tqdm(npz_files):
        stem = os.path.basename(npz_path).replace('.npz', '')
        out_path = os.path.join(args.out, stem + '.npz')
        if os.path.exists(out_path) and not args.overwrite:
            continue   # resume if interrupted

        d = np.load(npz_path, allow_pickle=True)
        frames   = d['frames']              # (T, 3, 512, 512) float16
        ed_mask  = d['ed_mask']             # (512, 512) uint8
        es_mask  = d['es_mask']             # (512, 512) uint8
        ed_idx   = int(d['ed_idx'])
        es_idx   = int(d['es_idx'])
        pixdim   = d['pixdim']              # (dx, dy, dz) mm
        group    = str(d['group'])
        orig_H   = int(d['orig_H'])
        orig_W   = int(d['orig_W'])

        T = frames.shape[0]

        bn = args.bbox_noise

        # ── Exp A passes: prompt=ED, cover all T frames ──────────────────────
        # fwd from ED: [ed_idx, T-1]
        fwd_ed = run_propagation(predictor, frames, ed_mask, ed_idx, reverse=False, bbox_noise=bn)
        # bwd from ED: [0, ed_idx] (skipped when ed_idx==0, most ACDC patients)
        ed_pred = fwd_ed.copy()
        if ed_idx > 0:
            bwd_ed = run_propagation(predictor, frames, ed_mask, ed_idx, reverse=True, bbox_noise=bn)
            ed_pred[:ed_idx] = bwd_ed[:ed_idx]

        # ── Exp B passes: prompt=ES, cover all T frames ──────────────────────
        # bwd from ES: [0, es_idx]
        bwd_es = run_propagation(predictor, frames, es_mask, es_idx, reverse=True, bbox_noise=bn)
        # fwd from ES: [es_idx, T-1]
        fwd_es = run_propagation(predictor, frames, es_mask, es_idx, reverse=False, bbox_noise=bn)
        es_pred = bwd_es.copy()
        if es_idx + 1 < T:
            es_pred[es_idx + 1:] = fwd_es[es_idx + 1:]

        # ── Exp C: bidir — per-frame nearest-anchor combination ───────────────
        mid = (ed_idx + es_idx) // 2
        bidir = np.zeros((T, 512, 512), dtype=np.uint8)
        bidir[:mid + 1]  = ed_pred[:mid + 1]   # frames [0, mid]: ED-anchored
        bidir[mid + 1:]  = es_pred[mid + 1:]   # frames [mid+1, T-1]: ES-anchored

        np.savez_compressed(
            out_path,
            ed_pred = ed_pred,
            es_pred = es_pred,
            bidir   = bidir,
            ed_idx  = np.int32(ed_idx),
            es_idx  = np.int32(es_idx),
            pixdim  = pixdim,
            group   = np.bytes_(group),
            orig_H  = np.int32(orig_H),
            orig_W  = np.int32(orig_W),
        )

    print(f"\nDone. Results saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/MedSAM2_latest.pt')
    parser.add_argument('--cfg',  default='configs/sam2.1_hiera_t512.yaml')
    parser.add_argument('--data', default='/scratch/gautschi/li4533/MIUA_2026/preprocessed')
    parser.add_argument('--out',  default='/scratch/gautschi/li4533/MIUA_2026/results/medsam2')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output NPZ files (default: skip)')
    parser.add_argument('--bbox_noise', type=float, default=0.0,
                        help='Bbox robustness: std of per-corner jitter as fraction of box size (0=off)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for bbox noise reproducibility')
    args = parser.parse_args()
    if args.bbox_noise > 0.0:
        np.random.seed(args.seed)
    main(args)
