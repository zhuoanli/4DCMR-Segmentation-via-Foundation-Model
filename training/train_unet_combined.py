"""
train_unet_combined.py
Train U-Net on ACDC (100 patients) + MnM2 (360 patients) combined.
Validate on MnM (136 patients) for external monitoring.
Saves: results/unet_combined/best_model.pth
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm

PROJ_DIR = '/scratch/gautschi/li4533/MIUA_2026'
UNET_DIR = os.path.join(PROJ_DIR, 'pytorch-unet')
sys.path.insert(0, UNET_DIR)
from unet import UNet              # noqa: E402
from utils.dice_score import dice_loss  # noqa: E402


class NPZSliceDataset(Dataset):
    """2D slices from ED and ES frames of preprocessed NPZ files.
    Works with ACDC (preprocessed/), MnM (preprocessed_mnm/), MnM2 (preprocessed_mnm2/).
    Lazy-loads from disk to avoid pre-loading all data into RAM.
    """
    def __init__(self, npz_dir: str, target_size: int = 256, val_pids: set = None):
        self.target_size = target_size
        self.samples = []  # list of (path, 'ed' or 'es')

        all_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        skipped = 0
        for path in all_files:
            pid = os.path.basename(path).rsplit('_slice', 1)[0]
            if val_pids is not None and pid in val_pids:
                continue
            try:
                d = np.load(path, allow_pickle=True)
                ed_mask = d['ed_mask']
                es_mask = d['es_mask']
            except Exception:
                print(f"  Skipping bad file: {os.path.basename(path)}")
                continue
            if ed_mask.max() > 0:
                self.samples.append((path, 'ed'))
            else:
                skipped += 1
            if es_mask.max() > 0:
                self.samples.append((path, 'es'))
            else:
                skipped += 1
        print(f"  {npz_dir}: {len(self.samples)} training slices (skipped {skipped} background-only)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame = self.samples[idx]
        d = np.load(path, allow_pickle=True)
        t    = int(d['ed_idx']) if frame == 'ed' else int(d['es_idx'])
        mask = d['ed_mask']    if frame == 'ed' else d['es_mask']
        gray = d['frames'][t, 0].astype(np.float32)  # (512,512) [0,1] float16→float32
        gray_resized = np.array(
            Image.fromarray((gray * 255).astype(np.uint8)).resize(
                (self.target_size, self.target_size), Image.BILINEAR
            ), dtype=np.float32
        ) / 255.0
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (self.target_size, self.target_size), Image.NEAREST
            ), dtype=np.int64
        )
        return {
            'image': torch.tensor(gray_resized, dtype=torch.float32).unsqueeze(0),
            'mask':  torch.tensor(mask_resized, dtype=torch.long),
        }


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


def predict_slice(model, gray_512, device, target_size=256):
    """gray_512: (512,512) float32 [0,1] → pred (512,512) uint8."""
    sl = np.array(
        Image.fromarray((gray_512 * 255).astype(np.uint8)).resize(
            (target_size, target_size), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    inp = torch.tensor(sl[None, None], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return np.array(Image.fromarray(pred).resize((512, 512), Image.NEAREST), dtype=np.uint8)


def validate_on_mnm(model, val_dir, device, max_patients=None):
    """Quick Dice validation on MnM NPZ files."""
    all_files = sorted(glob.glob(os.path.join(val_dir, '*.npz')))
    patients = {}
    for f in all_files:
        pid = os.path.basename(f).rsplit('_slice', 1)[0]
        patients.setdefault(pid, []).append(f)
    if max_patients:
        pids = sorted(patients.keys())[:max_patients]
    else:
        pids = sorted(patients.keys())

    rv_list, myo_list, lv_list = [], [], []
    model.eval()
    for pid in pids:
        slice_dices = {1: [], 2: [], 3: []}
        for npz_path in sorted(patients[pid]):
            d = np.load(npz_path, allow_pickle=True)
            frames  = d['frames'].astype(np.float32)
            es_mask = d['es_mask']
            es_t    = int(d['es_idx'])
            if es_mask.max() == 0:
                continue
            gray = frames[es_t, 0]
            pred = predict_slice(model, gray, device)
            for cls in [1, 2, 3]:
                slice_dices[cls].append(dice_np(pred, es_mask, cls))
        if slice_dices[3]:
            rv_list.append(np.mean(slice_dices[1]))
            myo_list.append(np.mean(slice_dices[2]))
            lv_list.append(np.mean(slice_dices[3]))

    mean_dice = float(np.mean(rv_list + myo_list + lv_list)) if rv_list else 0.0
    return mean_dice, np.mean(rv_list), np.mean(myo_list), np.mean(lv_list)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    print("Building training dataset...")
    acdc_ds  = NPZSliceDataset(args.acdc_dir,  target_size=args.img_size)
    mnm2_ds  = NPZSliceDataset(args.mnm2_dir,  target_size=args.img_size)
    train_ds = ConcatDataset([acdc_ds, mnm2_ds])
    print(f"Total training slices: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    model = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_dice = 0.0
    best_ckpt = os.path.join(args.out, 'best_model.pth')
    history   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            imgs  = batch['image'].to(device, memory_format=torch.channels_last)
            masks = batch['mask'].to(device)
            with torch.autocast(device_type='cuda', enabled=args.amp):
                logits = model(imgs)
                ce     = criterion(logits, masks)
                soft   = F.softmax(logits, dim=1).float()
                oh     = F.one_hot(masks, 4).permute(0, 3, 1, 2).float()
                loss   = ce + dice_loss(soft, oh, multiclass=True)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        # Validate on subset of MnM every epoch
        mean_d, rv, myo, lv = validate_on_mnm(model, args.mnm_val_dir, device,
                                               max_patients=args.val_patients)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
              f"MnM val dice={mean_d:.4f}  RV={rv:.3f} Myo={myo:.3f} LV={lv:.3f}")
        history.append({'epoch': epoch, 'loss': avg_loss, 'val_dice': mean_d,
                        'rv': rv, 'myo': myo, 'lv': lv})

        if mean_d > best_dice:
            best_dice = mean_d
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint saved (dice={best_dice:.4f})")

    print(f"\nTraining complete. Best MnM val Dice: {best_dice:.4f}")
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Checkpoint: {best_ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_dir',     default=f'{PROJ_DIR}/preprocessed')
    parser.add_argument('--mnm2_dir',     default=f'{PROJ_DIR}/preprocessed_mnm2')
    parser.add_argument('--mnm_val_dir',  default=f'{PROJ_DIR}/preprocessed_mnm')
    parser.add_argument('--out',          default=f'{PROJ_DIR}/results/unet_combined')
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--batch',        type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--img_size',     type=int,   default=256)
    parser.add_argument('--workers',      type=int,   default=8)
    parser.add_argument('--val_patients', type=int,   default=30,
                        help='Number of MnM patients to use for quick val per epoch')
    parser.add_argument('--amp',          action='store_true', default=True)
    args = parser.parse_args()
    main(args)
