"""
infer_dinov2_mnm.py
Evaluate DINOv2 (combined ACDC+MnM2 checkpoint) on M&Ms preprocessed NPZ files.
Saves per-slice NPZ predictions to results/dinov2_mnm/ (so compute_all_metrics.py
can compute HD95/ASSD), and writes 'DINOv2' key into results/metrics_mnm.json.
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJ_DIR     = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR     = os.path.join(PROJ_DIR, 'preprocessed_mnm')
RESULTS_DIR  = os.path.join(PROJ_DIR, 'results')
METRICS_JSON = os.path.join(RESULTS_DIR, 'metrics_mnm.json')
NPZ_OUT_DIR  = os.path.join(RESULTS_DIR, 'dinov2_mnm')

# Prefer combined (ACDC+MnM2) checkpoint; fall back to ACDC-only
_combined = os.path.join(RESULTS_DIR, 'dinov2_combined', 'best_model.pth')
_acdc     = os.path.join(RESULTS_DIR, 'dinov2', 'best_model.pth')
CKPT_PATH = _combined if os.path.exists(_combined) else _acdc

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class DINOv2SegHead(nn.Module):
    def __init__(self, num_classes=4, dinov2_name='dinov2_vits14'):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', dinov2_name,
            pretrained=True, verbose=False
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        feat_dim = self.backbone.embed_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        DIN_SIZE = 504
        x_din = F.interpolate(x, size=(DIN_SIZE, DIN_SIZE), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = self.backbone.forward_features(x_din)
        ph = pw = DIN_SIZE // 14
        patch_tokens = feats['x_norm_patchtokens']
        feat_map = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)
        feat_map = F.interpolate(feat_map, size=(H, W), mode='bilinear', align_corners=False)
        return self.decoder(feat_map)


def dice_np(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


def _decode_group(raw) -> str:
    s = str(raw)
    for pfx, sfx in [("np.bytes_(b'", "')"), ("b'", "'")]:
        if s.startswith(pfx) and s.endswith(sfx):
            return s[len(pfx):-len(sfx)]
    return s


def predict_frame(model, frame_f16, device):
    """frame_f16: (3,512,512) float16 [0,1] → pred (512,512) uint8."""
    img = torch.from_numpy(frame_f16.astype(np.float32)).unsqueeze(0).to(device)
    img = (img - IMG_MEAN.to(device)) / IMG_STD.to(device)
    with torch.no_grad():
        return model(img).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main(ckpt_path, prep_dir, metrics_json, npz_out_dir):
    os.makedirs(npz_out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    model = DINOv2SegHead(num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_files = sorted(glob.glob(os.path.join(prep_dir, '*.npz')))
    patients = {}
    for f in all_files:
        pid = os.path.basename(f).rsplit('_slice', 1)[0]
        patients.setdefault(pid, []).append(f)

    print(f"Evaluating {len(patients)} M&Ms patients ...")
    results = []

    for pid in tqdm(sorted(patients.keys())):
        slice_files = sorted(patients[pid])
        slice_dices = {1: [], 2: [], 3: []}
        pred_edv_vox = 0.0
        pred_esv_vox = 0.0
        gt_edv_vox   = 0.0
        gt_esv_vox   = 0.0
        voxel_vol    = None
        group        = 'UNK'

        for npz_path in slice_files:
            d = np.load(npz_path, allow_pickle=True)
            es_mask = d['es_mask']
            ed_mask = d['ed_mask']
            es_t    = int(d['es_idx'])
            ed_t    = int(d['ed_idx'])
            frames  = d['frames']   # (T, 3, 512, 512) float16
            T       = frames.shape[0]

            if voxel_vol is None:
                pixdim = d['pixdim'].astype(np.float64)
                orig_H = int(d['orig_H'])
                orig_W = int(d['orig_W'])
                scale  = (orig_H / 512.0) * (orig_W / 512.0)
                voxel_vol = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale
                group = _decode_group(d['group'])

            # Predict at ED and ES frames
            pred_ed = predict_frame(model, frames[ed_t], device)
            pred_es = predict_frame(model, frames[es_t], device)

            # Dice at ES frame
            if es_mask.max() > 0:
                for cls in [1, 2, 3]:
                    slice_dices[cls].append(dice_np(pred_es, es_mask, cls))

            # Volume accumulation
            pred_edv_vox += (pred_ed == 3).sum()
            pred_esv_vox += (pred_es == 3).sum()
            gt_edv_vox   += (ed_mask  == 3).sum()
            gt_esv_vox   += (es_mask  == 3).sum()

            # Save per-slice NPZ for HD95/ASSD computation
            bidir = np.zeros((T, 512, 512), dtype=np.uint8)
            bidir[ed_t] = pred_ed
            bidir[es_t] = pred_es
            stem = os.path.basename(npz_path).replace('.npz', '')
            np.savez_compressed(os.path.join(npz_out_dir, f'{stem}.npz'), bidir=bidir)

        if not slice_dices[3]:
            continue

        pred_vol_ed = pred_edv_vox * voxel_vol / 1000.0
        pred_vol_es = pred_esv_vox * voxel_vol / 1000.0
        pred_edv = max(pred_vol_ed, pred_vol_es)
        pred_esv = min(pred_vol_ed, pred_vol_es)
        # Fix n=135: use 0.0 instead of None when volume is zero
        pred_ef = float(100 * (pred_edv - pred_esv) / pred_edv) if pred_edv > 1e-3 else 0.0

        gt_vol_ed = gt_edv_vox * voxel_vol / 1000.0
        gt_vol_es = gt_esv_vox * voxel_vol / 1000.0
        gt_edv = max(gt_vol_ed, gt_vol_es)
        gt_esv = min(gt_vol_ed, gt_vol_es)
        gt_ef  = float(100 * (gt_edv - gt_esv) / gt_edv) if gt_edv > 1e-3 else 0.0

        results.append({
            'pid':      pid,
            'group':    group,
            'dice_RV':  float(np.mean(slice_dices[1])),
            'dice_Myo': float(np.mean(slice_dices[2])),
            'dice_LV':  float(np.mean(slice_dices[3])),
            'hd95_RV':  None, 'hd95_Myo': None, 'hd95_LV': None,
            'assd_RV':  None, 'assd_Myo': None, 'assd_LV': None,
            'pred_EF':  pred_ef,
            'pred_EDV': float(pred_edv),
            'pred_ESV': float(pred_esv),
            'gt_EF':    gt_ef,
            'gt_EDV':   float(gt_edv),
            'gt_ESV':   float(gt_esv),
        })

    rv  = np.mean([r['dice_RV']  for r in results]) if results else float('nan')
    myo = np.mean([r['dice_Myo'] for r in results]) if results else float('nan')
    lv  = np.mean([r['dice_LV']  for r in results]) if results else float('nan')
    ef_pairs = [(r['pred_EF'], r['gt_EF']) for r in results
                if r['pred_EF'] is not None and r['gt_EF'] is not None]
    ef_mae = np.mean([abs(a - b) for a, b in ef_pairs]) if ef_pairs else float('nan')
    print(f"\nDINOv2 on M&Ms (n={len(results)}):")
    print(f"  RV={rv:.3f}  Myo={myo:.3f}  LV={lv:.3f}  EF_MAE={ef_mae:.2f}% (n={len(ef_pairs)})")
    print(f"  NPZ saved to: {npz_out_dir}")

    existing = {}
    if os.path.exists(metrics_json):
        with open(metrics_json) as f:
            existing = json.load(f)
    existing['DINOv2'] = results
    with open(metrics_json, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Merged DINOv2 results into {metrics_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     default=CKPT_PATH,    help='Checkpoint path')
    parser.add_argument('--prep_dir', default=PREP_DIR,     help='Preprocessed MnM dir')
    parser.add_argument('--out',      default=METRICS_JSON, help='Output metrics JSON')
    parser.add_argument('--npz_dir',  default=NPZ_OUT_DIR,  help='Output NPZ dir')
    args = parser.parse_args()
    main(args.ckpt, args.prep_dir, args.out, args.npz_dir)
