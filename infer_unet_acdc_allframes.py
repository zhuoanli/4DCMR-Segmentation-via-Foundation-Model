"""
infer_unet_acdc_allframes.py
Run U-Net on ALL cardiac frames for ACDC validation patients.

Outputs per-slice NPZ to results/unet_acdc_allframes/ with:
  bidir: (T, 512, 512) uint8 — prediction at every frame
  ed_idx, es_idx, pixdim, group, orig_H, orig_W

Purpose:
  (a) Full-cycle LV time-volume curves for comparison with MedSAM2 temporal drift analysis
  (b) HD95/ASSD for U-Net via compute_all_metrics.py (uses bidir[es_idx])
"""

import os, sys, glob, argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJ_DIR = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR = os.path.join(PROJ_DIR, 'preprocessed')
OUT_DIR  = os.path.join(PROJ_DIR, 'results', 'unet_acdc_allframes')

# Validation patient IDs (4-per-group stratified split, 20 patients)
VAL_PIDS = {
    'patient017', 'patient018', 'patient019', 'patient020',
    'patient037', 'patient038', 'patient039', 'patient040',
    'patient057', 'patient058', 'patient059', 'patient060',
    'patient077', 'patient078', 'patient079', 'patient080',
    'patient097', 'patient098', 'patient099', 'patient100',
}

_combined = os.path.join(PROJ_DIR, 'results', 'unet_combined', 'best_model.pth')
_acdc     = os.path.join(PROJ_DIR, 'results', 'unet', 'best_model.pth')
DEFAULT_CKPT = _combined if os.path.exists(_combined) else _acdc

UNET_DIR = os.path.join(PROJ_DIR, 'pytorch-unet')
sys.path.insert(0, UNET_DIR)
from unet import UNet  # noqa: E402


def predict_frame(model, frame_512, device):
    """frame_512: (512,512) float32 [0,1] → pred (512,512) uint8."""
    sl_256 = np.array(
        Image.fromarray((frame_512 * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    inp = torch.tensor(sl_256[None, None], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return np.array(Image.fromarray(out).resize((512, 512), Image.NEAREST), dtype=np.uint8)


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

    model = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
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
        frames = d['frames'].astype(np.float32)   # (T, 3, 512, 512)
        T      = frames.shape[0]
        ed_idx = int(d['ed_idx'])
        es_idx = int(d['es_idx'])
        pixdim = d['pixdim']
        group  = _decode_group(d['group'])
        orig_H = int(d['orig_H'])
        orig_W = int(d['orig_W'])

        # Run U-Net on every frame independently (no temporal context)
        all_preds = np.zeros((T, 512, 512), dtype=np.uint8)
        for t in range(T):
            gray = frames[t, 0]   # channel 0 as grayscale proxy
            all_preds[t] = predict_frame(model, gray, device)

        np.savez_compressed(
            out_path,
            bidir  = all_preds,    # compute_all_metrics.py reads bidir[es_idx] for Dice/HD95
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
