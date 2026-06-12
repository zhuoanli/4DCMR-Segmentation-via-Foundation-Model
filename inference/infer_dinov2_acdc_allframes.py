"""
infer_dinov2_acdc_allframes.py
Run DINOv2 segmentation on ALL cardiac frames for ACDC validation patients.

Outputs per-slice NPZ to results/dinov2_acdc_allframes/ with:
  bidir: (T, 512, 512) uint8 — prediction at every frame
  ed_idx, es_idx, pixdim, group, orig_H, orig_W

Purpose: HD95/ASSD computation for DINOv2 in Table 1
"""

import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJ_DIR = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR = os.path.join(PROJ_DIR, 'preprocessed')
OUT_DIR  = os.path.join(PROJ_DIR, 'results', 'dinov2_acdc_allframes')

VAL_PIDS = {
    'patient017', 'patient018', 'patient019', 'patient020',
    'patient037', 'patient038', 'patient039', 'patient040',
    'patient057', 'patient058', 'patient059', 'patient060',
    'patient077', 'patient078', 'patient079', 'patient080',
    'patient097', 'patient098', 'patient099', 'patient100',
}

_combined = os.path.join(PROJ_DIR, 'results', 'dinov2_combined', 'best_model.pth')
_acdc     = os.path.join(PROJ_DIR, 'results', 'dinov2', 'best_model.pth')
DEFAULT_CKPT = _combined if os.path.exists(_combined) else _acdc

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


def predict_frame(model, frame_f16, device):
    """frame_f16: (3,512,512) float16 [0,1] → pred (512,512) uint8."""
    img = torch.from_numpy(frame_f16.astype(np.float32)).unsqueeze(0).to(device)
    img = (img - IMG_MEAN.to(device)) / IMG_STD.to(device)
    with torch.no_grad():
        return model(img).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _decode_group(raw) -> str:
    s = str(raw)
    for pfx, sfx in [("np.bytes_(b'", "')"), ("b'", "'")]:
        if s.startswith(pfx) and s.endswith(sfx):
            return s[len(pfx):-len(sfx)]
    return s


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Checkpoint: {args.ckpt}")

    model = DINOv2SegHead(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    all_files = sorted(glob.glob(os.path.join(args.prep_dir, '*.npz')))
    val_files = [f for f in all_files
                 if os.path.basename(f).rsplit('_slice', 1)[0] in VAL_PIDS]
    print(f"Processing {len(val_files)} slices from {len(VAL_PIDS)} val patients")

    for npz_path in tqdm(val_files):
        stem = os.path.basename(npz_path).replace('.npz', '')
        out_path = os.path.join(args.out, stem + '.npz')
        if os.path.exists(out_path) and not args.overwrite:
            continue

        d = np.load(npz_path, allow_pickle=True)
        frames = d['frames']   # (T, 3, 512, 512) float16
        T      = frames.shape[0]
        ed_idx = int(d['ed_idx'])
        es_idx = int(d['es_idx'])
        pixdim = d['pixdim']
        group  = _decode_group(d['group'])
        orig_H = int(d['orig_H'])
        orig_W = int(d['orig_W'])

        all_preds = np.zeros((T, 512, 512), dtype=np.uint8)
        for t in range(T):
            all_preds[t] = predict_frame(model, frames[t], device)

        np.savez_compressed(
            out_path,
            bidir  = all_preds,
            ed_idx = np.int32(ed_idx),
            es_idx = np.int32(es_idx),
            pixdim = pixdim,
            group  = np.bytes_(group),
            orig_H = np.int32(orig_H),
            orig_W = np.int32(orig_W),
        )

    print(f"Done. Results saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     default=DEFAULT_CKPT)
    parser.add_argument('--prep_dir', default=PREP_DIR)
    parser.add_argument('--out',      default=OUT_DIR)
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
