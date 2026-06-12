"""
train_dinov2_combined.py
DINOv2-S/14 + segmentation decoder trained on ACDC (100) + MnM2 (360).
Backbone frozen (Meta pretrained), only decoder is trained.
Validates on MnM (136 patients).
Saves: results/dinov2_combined/best_model.pth
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

PROJ_DIR = '/scratch/gautschi/li4533/MIUA_2026'

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class NPZSliceDataset(Dataset):
    """ED + ES frames from preprocessed NPZ directory (ACDC or MnM2 format)."""
    def __init__(self, npz_dir: str):
        self.items = []  # list of (npz_path, 'ed' or 'es')
        all_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        for path in all_files:
            try:
                d = np.load(path, allow_pickle=True)
                if d['ed_mask'].max() > 0:
                    self.items.append((path, 'ed'))
                if d['es_mask'].max() > 0:
                    self.items.append((path, 'es'))
            except Exception:
                print(f"  Skipping bad file: {os.path.basename(path)}")
        print(f"  {os.path.basename(npz_dir)}: {len(self.items)} slices")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, fr = self.items[idx]
        d = np.load(path, allow_pickle=True)
        t    = int(d['ed_idx']) if fr == 'ed' else int(d['es_idx'])
        mask = d['ed_mask']    if fr == 'ed' else d['es_mask']  # (512,512) uint8

        img = torch.from_numpy(d['frames'][t].astype(np.float32))  # (3,512,512) [0,1]
        img = (img - IMG_MEAN) / IMG_STD   # ImageNet normalisation

        lbl = torch.from_numpy(mask.astype(np.int64))  # (512,512)

        # Random horizontal flip augmentation
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            lbl = torch.flip(lbl, dims=[1])

        return img, lbl


class DINOv2SegHead(nn.Module):
    """Frozen DINOv2-S/14 + lightweight decoder. Input (B,3,512,512), output (B,4,512,512)."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14',
            pretrained=True, verbose=False
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        feat_dim = self.backbone.embed_dim  # 384

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
        DIN_SIZE = 504  # 36×14, closest multiple of 14 below 512
        xd = F.interpolate(x, size=(DIN_SIZE, DIN_SIZE), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = self.backbone.forward_features(xd)
        patch_tokens = feats['x_norm_patchtokens']  # (B, N, C)
        ph = pw = DIN_SIZE // 14  # 36
        fm = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)  # (B,C,36,36)
        fm = F.interpolate(fm, size=(H, W), mode='bilinear', align_corners=False)
        return self.decoder(fm)


def dice_np(pred, gt, cls):
    p, g = (pred == cls), (gt == cls)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


def validate_on_mnm(model, val_dir, device, max_patients=None):
    """Quick Dice on MnM NPZ ES frames."""
    all_files = sorted(glob.glob(os.path.join(val_dir, '*.npz')))
    patients = {}
    for f in all_files:
        pid = os.path.basename(f).rsplit('_slice', 1)[0]
        patients.setdefault(pid, []).append(f)
    pids = sorted(patients.keys())[:max_patients] if max_patients else sorted(patients.keys())

    rv_list, myo_list, lv_list = [], [], []
    model.eval()
    with torch.no_grad():
        for pid in pids:
            slice_dices = {1: [], 2: [], 3: []}
            for npz_path in sorted(patients[pid]):
                d = np.load(npz_path, allow_pickle=True)
                es_mask = d['es_mask']
                if es_mask.max() == 0:
                    continue
                t = int(d['es_idx'])
                img = torch.from_numpy(d['frames'][t].astype(np.float32)).unsqueeze(0)  # (1,3,512,512)
                img = (img - IMG_MEAN.unsqueeze(0)) / IMG_STD.unsqueeze(0)
                img = img.to(device)
                logits = model(img)
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                for cls in [1, 2, 3]:
                    slice_dices[cls].append(dice_np(pred, es_mask, cls))
            if slice_dices[3]:
                rv_list.append(np.mean(slice_dices[1]))
                myo_list.append(np.mean(slice_dices[2]))
                lv_list.append(np.mean(slice_dices[3]))

    if not rv_list:
        return 0.0, 0.0, 0.0, 0.0
    return (np.mean(rv_list + myo_list + lv_list),
            np.mean(rv_list), np.mean(myo_list), np.mean(lv_list))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    print("Building training datasets...")
    acdc_ds  = NPZSliceDataset(args.acdc_dir)
    mnm2_ds  = NPZSliceDataset(args.mnm2_dir)
    train_ds = ConcatDataset([acdc_ds, mnm2_ds])
    print(f"Total training slices: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    model = DINOv2SegHead(num_classes=4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M (decoder only, backbone frozen)")

    class_weights = torch.tensor([0.1, 1.0, 2.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice  = 0.0
    no_improve = 0
    best_ckpt  = os.path.join(args.out, 'best_model.pth')
    history    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validate every epoch
        mean_d, rv, myo, lv = validate_on_mnm(model, args.mnm_val_dir, device,
                                               max_patients=args.val_patients)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
              f"MnM val dice={mean_d:.4f}  RV={rv:.3f} Myo={myo:.3f} LV={lv:.3f}")
        history.append({'epoch': epoch, 'loss': avg_loss, 'val_dice': mean_d,
                        'rv': rv, 'myo': myo, 'lv': lv})

        if mean_d > best_dice:
            best_dice  = mean_d
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint saved (dice={best_dice:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs.")
                break

    print(f"\nTraining done. Best MnM val Dice: {best_dice:.4f}")
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Checkpoint: {best_ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_dir',     default=f'{PROJ_DIR}/preprocessed')
    parser.add_argument('--mnm2_dir',     default=f'{PROJ_DIR}/preprocessed_mnm2')
    parser.add_argument('--mnm_val_dir',  default=f'{PROJ_DIR}/preprocessed_mnm')
    parser.add_argument('--out',          default=f'{PROJ_DIR}/results/dinov2_combined')
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--batch',        type=int,   default=8)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--workers',      type=int,   default=8)
    parser.add_argument('--val_patients', type=int,   default=30)
    parser.add_argument('--patience',     type=int,   default=15,
                        help='Early stopping: epochs without val Dice improvement')
    args = parser.parse_args()
    main(args)
