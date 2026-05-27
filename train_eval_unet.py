"""
train_eval_unet.py
Exp E: Train U-Net on ACDC ED+ES 2D slices (stratified split: 16 per group),
       evaluate per-class Dice at ES frames of stratified val patients (4 per group).

Saves:
  results/unet/best_model.pth
  results/unet/results.json   — per-patient Dice for RV/Myo/LV + mean
"""

import os, sys, json, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image

# ── U-Net from existing repo ──────────────────────────────────────────────────
UNET_DIR = '/scratch/gautschi/li4533/MIUA_2026/pytorch-unet'
sys.path.insert(0, UNET_DIR)
from unet import UNet                           # noqa: E402
from utils.dice_score import dice_loss          # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_info_cfg(cfg_path):
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def find_frame_files(pdir, pid):
    nii = sorted(glob(os.path.join(pdir, f'patient{pid:03d}_frame*.nii.gz')))
    gt  = [f for f in nii if '_gt' in f]
    img = [f for f in nii if '_gt' not in f and '4d' not in f]
    img.sort(); gt.sort()
    assert len(img) == 2 and len(gt) == 2, f"patient {pid}: expected 2 frame/gt pairs"
    return img[0], gt[0], img[1], gt[1]   # (ed_img, ed_gt, es_img, es_gt)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ACDCSliceDataset(Dataset):
    """2D slices from ED and ES frames of given patients. Returns grayscale image + label mask."""
    def __init__(self, db_path: str, patient_ids: list, target_size: int = 256):
        self.samples = []   # list of (img_slice, gt_slice) numpy arrays
        for pid in patient_ids:
            pdir = os.path.join(db_path, f'patient{pid:03d}')
            if not os.path.isdir(pdir):
                continue
            try:
                ed_img_f, ed_gt_f, es_img_f, es_gt_f = find_frame_files(pdir, pid)
            except AssertionError:
                continue
            for img_f, gt_f in [(ed_img_f, ed_gt_f), (es_img_f, es_gt_f)]:
                img_vol = nib.load(img_f).get_fdata(dtype=np.float32)  # (H, W, Z)
                gt_vol  = nib.load(gt_f).get_fdata(dtype=np.float32).astype(np.uint8)
                for z in range(img_vol.shape[2]):
                    img_sl = img_vol[:, :, z]
                    gt_sl  = gt_vol[:, :, z]
                    if gt_sl.max() == 0:
                        continue   # skip all-background slices
                    # Percentile normalise → float32 [0,1]
                    p2, p98 = np.percentile(img_sl, 2), np.percentile(img_sl, 98)
                    img_sl  = np.clip((img_sl - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
                    # Resize to target_size × target_size
                    img_sl  = np.array(
                        Image.fromarray((img_sl * 255).astype(np.uint8)).resize(
                            (target_size, target_size), Image.BILINEAR
                        ), dtype=np.float32
                    ) / 255.0
                    gt_sl   = np.array(
                        Image.fromarray(gt_sl).resize(
                            (target_size, target_size), Image.NEAREST
                        ), dtype=np.int64
                    )
                    self.samples.append((img_sl, gt_sl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, gt = self.samples[idx]
        return {
            'image': torch.tensor(img, dtype=torch.float32).unsqueeze(0),  # (1, H, W)
            'mask':  torch.tensor(gt,  dtype=torch.long),                  # (H, W)
        }


# ── Dice evaluation (numpy) ───────────────────────────────────────────────────
def dice_np(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    p = (pred == cls)
    g = (gt   == cls)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (p & g).sum()) / float(p.sum() + g.sum())


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Stratified split: last 4 per group → val; remaining 16 per group → train
    val_ids   = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]
    train_ids = [i for i in range(1, 101) if i not in val_ids]

    train_ds = ACDCSliceDataset(args.db, train_ids, target_size=256)
    val_ds   = ACDCSliceDataset(args.db, val_ids,   target_size=256)
    print(f"Train slices: {len(train_ds)}  Val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion  = nn.CrossEntropyLoss()
    scaler     = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.out, exist_ok=True)
    best_val_dice = 0.0
    best_ckpt     = os.path.join(args.out, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            imgs  = batch['image'].to(device, memory_format=torch.channels_last)
            masks = batch['mask'].to(device)

            with torch.autocast(device_type='cuda', enabled=args.amp):
                logits = model(imgs)                          # (B, 4, H, W)
                ce_loss   = criterion(logits, masks)
                soft_pred = F.softmax(logits, dim=1).float()
                one_hot   = F.one_hot(masks, 4).permute(0, 3, 1, 2).float()
                dl        = dice_loss(soft_pred, one_hot, multiclass=True)
                loss      = ce_loss + dl

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        # ── validate ──
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                with torch.autocast(device_type='cuda', enabled=args.amp):
                    logits = model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                for p, g in zip(preds, masks):
                    val_dices.append(np.mean([dice_np(p, g, c) for c in [1, 2, 3]]))

        val_dice_mean = float(np.mean(val_dices))
        print(f"Epoch {epoch:3d}  loss={epoch_loss/len(train_loader):.4f}  val_dice={val_dice_mean:.4f}")

        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            torch.save(model.state_dict(), best_ckpt)

    print(f"\nBest val Dice: {best_val_dice:.4f}  checkpoint: {best_ckpt}")


# ── Evaluation on val set ES frames ──────────────────────────────────────────
def _infer_volume(model, vol, device):
    """Run UNet on every slice of vol (H, W, Z), return 3D prediction (H, W, Z) at original res."""
    H, W, Z = vol.shape
    pred_3d = np.zeros((H, W, Z), dtype=np.uint8)
    for z in range(Z):
        sl = vol[:, :, z]
        p2, p98 = np.percentile(sl, 2), np.percentile(sl, 98)
        sl_norm = np.clip((sl - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
        sl_256 = np.array(
            Image.fromarray((sl_norm * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
            dtype=np.float32
        ) / 255.0
        inp = torch.tensor(sl_256[None, None], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_256 = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # resize prediction back to original resolution
        pred_orig = np.array(
            Image.fromarray(pred_256).resize((W, H), Image.NEAREST), dtype=np.uint8
        )
        pred_3d[:, :, z] = pred_orig
    return pred_3d


def evaluate(args):
    """Load best checkpoint, predict ES frames of val patients, save per-patient Dice + EF."""
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_ckpt = os.path.join(args.out, 'best_model.pth')
    model     = UNet(n_channels=1, n_classes=4, bilinear=True).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    val_ids = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]
    results     = {}   # pid -> {RV, Myo, LV, mean}  (legacy Dice-only JSON)
    metrics_list = []  # entries for metrics_acdc_val.json

    for pid in tqdm(val_ids, desc='Evaluating val patients'):
        pdir = os.path.join(args.db, f'patient{pid:03d}')
        if not os.path.isdir(pdir):
            continue
        try:
            ed_img_f, ed_gt_f, es_img_f, es_gt_f = find_frame_files(pdir, pid)
        except AssertionError:
            continue

        # load volumes
        ed_nii = nib.load(ed_img_f)
        es_vol = nib.load(es_img_f).get_fdata(dtype=np.float32)  # (H, W, Z)
        ed_vol = ed_nii.get_fdata(dtype=np.float32)
        es_gt  = nib.load(es_gt_f).get_fdata(dtype=np.float32).astype(np.uint8)
        ed_gt  = nib.load(ed_gt_f).get_fdata(dtype=np.float32).astype(np.uint8)

        # voxel volume in mL
        zooms = ed_nii.header.get_zooms()[:3]  # (dy, dx, dz) mm
        voxel_mm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

        # group label
        cfg_path = os.path.join(pdir, 'Info.cfg')
        group = parse_info_cfg(cfg_path).get('Group', 'UNK') if os.path.exists(cfg_path) else 'UNK'

        # run inference on both frames for volume computation
        es_pred_3d = _infer_volume(model, es_vol, device)
        ed_pred_3d = _infer_volume(model, ed_vol, device)

        # compute Dice on annotated ES slices (resize pred to 256 for comparison)
        slice_dices = {1: [], 2: [], 3: []}
        for z in range(es_vol.shape[2]):
            gt_sl = es_gt[:, :, z]
            if gt_sl.max() == 0:
                continue
            pred_sl = es_pred_3d[:, :, z]
            pred_256 = np.array(
                Image.fromarray(pred_sl).resize((256, 256), Image.NEAREST), dtype=np.int64
            )
            gt_256 = np.array(
                Image.fromarray(gt_sl).resize((256, 256), Image.NEAREST), dtype=np.int64
            )
            for cls in [1, 2, 3]:
                slice_dices[cls].append(dice_np(pred_256, gt_256, cls))

        # clinical metrics
        pred_EDV = float((ed_pred_3d == 3).sum()) * voxel_mm3 / 1000.0
        pred_ESV = float((es_pred_3d == 3).sum()) * voxel_mm3 / 1000.0
        pred_EF  = float((pred_EDV - pred_ESV) / pred_EDV * 100.0) if pred_EDV > 1e-3 else None
        gt_EDV   = float((ed_gt == 3).sum()) * voxel_mm3 / 1000.0
        gt_ESV   = float((es_gt == 3).sum()) * voxel_mm3 / 1000.0
        gt_EF    = float((gt_EDV - gt_ESV) / gt_EDV * 100.0) if gt_EDV > 1e-3 else None

        rv  = float(np.mean(slice_dices[1])) if slice_dices[1] else 0.0
        myo = float(np.mean(slice_dices[2])) if slice_dices[2] else 0.0
        lv  = float(np.mean(slice_dices[3])) if slice_dices[3] else 0.0
        results[pid] = {'RV': rv, 'Myo': myo, 'LV': lv, 'mean': (rv + myo + lv) / 3}

        metrics_list.append({
            'pid':      f'patient{pid:03d}',
            'group':    group,
            'dice_RV':  rv,
            'dice_Myo': myo,
            'dice_LV':  lv,
            'hd95_RV':  None, 'hd95_Myo': None, 'hd95_LV': None,
            'assd_RV':  None, 'assd_Myo': None, 'assd_LV': None,
            'pred_EF':  pred_EF,
            'pred_EDV': pred_EDV,
            'pred_ESV': pred_ESV,
            'gt_EF':    gt_EF,
            'gt_EDV':   gt_EDV,
            'gt_ESV':   gt_ESV,
        })

    # Print summary
    rv_all  = [v['RV']  for v in results.values()]
    myo_all = [v['Myo'] for v in results.values()]
    lv_all  = [v['LV']  for v in results.values()]
    mn_all  = [v['mean'] for v in results.values()]
    print("\n── U-Net ES-frame Dice (stratified val set) ──")
    print(f"  RV : {np.mean(rv_all):.3f} ± {np.std(rv_all):.3f}")
    print(f"  Myo: {np.mean(myo_all):.3f} ± {np.std(myo_all):.3f}")
    print(f"  LV : {np.mean(lv_all):.3f} ± {np.std(lv_all):.3f}")
    print(f"  Mean: {np.mean(mn_all):.3f} ± {np.std(mn_all):.3f}")
    ef_vals = [e['pred_EF'] for e in metrics_list if e['pred_EF'] is not None]
    print(f"  EF  : {np.mean(ef_vals):.1f} ± {np.std(ef_vals):.1f}% (pred, n={len(ef_vals)})")

    # save legacy Dice-only JSON
    out_json = os.path.join(args.out, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-patient Dice to {out_json}")

    # merge UNet EF data into metrics_acdc_val.json
    metrics_path = os.path.join(os.path.dirname(args.out), 'metrics_acdc_val.json')
    existing = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
    existing['UNet'] = metrics_list
    with open(metrics_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Merged UNet EF data into {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',     default='/scratch/gautschi/li4533/MIUA_2026/database/training')
    parser.add_argument('--out',    default='/scratch/gautschi/li4533/MIUA_2026/results/unet')
    parser.add_argument('--epochs', type=int,   default=30)
    parser.add_argument('--batch',  type=int,   default=16)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--amp',       action='store_true', default=True)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    if not args.eval_only:
        train(args)
    evaluate(args)
