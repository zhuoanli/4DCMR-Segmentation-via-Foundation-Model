"""
train_eval_dinov2.py
DINOv2-S/14 + lightweight segmentation head baseline for ACDC 2D cardiac MRI.

Trains on same ACDC training split as U-Net (80 patients, ED+ES 2D slices).
Evaluates on val set (20 patients, ES frame).

DINOv2 is frozen; only the lightweight decoder is trained.
This is a supervised 2D baseline (no temporal context) for comparison with
MedSAM2 which uses temporal video propagation.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from PIL import Image

PROJ_DIR     = '/scratch/gautschi/li4533/MIUA_2026'
PREP_DIR     = os.path.join(PROJ_DIR, 'preprocessed')
CKPT_OUT     = os.path.join(PROJ_DIR, 'results', 'dinov2', 'best_model.pth')
RESULT_OUT   = os.path.join(PROJ_DIR, 'results', 'dinov2', 'results.json')
METRICS_JSON = os.path.join(PROJ_DIR, 'results', 'metrics_acdc_val.json')

# Stratified val split: 4 patients per pathology group
VAL_IDS = [17,18,19,20, 37,38,39,40, 57,58,59,60, 77,78,79,80, 97,98,99,100]

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Dataset ────────────────────────────────────────────────────────────────────

class ACDCSliceDataset(Dataset):
    """Each item: one 2D slice at ED or ES frame."""
    def __init__(self, prep_dir, patient_ids, frame='both', augment=False):
        self.files   = []
        self.frames  = []  # 'ed' or 'es'
        self.augment = augment

        for pid in patient_ids:
            for npz_path in sorted(glob(os.path.join(prep_dir, f'patient{pid:03d}_slice*.npz'))):
                if frame in ('both', 'ed'):
                    self.files.append(npz_path); self.frames.append('ed')
                if frame in ('both', 'es'):
                    self.files.append(npz_path); self.frames.append('es')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d      = np.load(self.files[idx], allow_pickle=True)
        fr_key = self.frames[idx]
        t      = int(d['ed_idx']) if fr_key == 'ed' else int(d['es_idx'])
        mask   = d['ed_mask'] if fr_key == 'ed' else d['es_mask']   # (512, 512) uint8

        frame_f16 = d['frames'][t]                  # (3, 512, 512) float16 [0,1]
        img = torch.from_numpy(frame_f16.astype(np.float32))   # (3, 512, 512)
        img = (img - IMG_MEAN) / IMG_STD             # ImageNet normalisation

        lbl = torch.from_numpy(mask.astype(np.int64))  # (512, 512)

        if self.augment:
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])   # horizontal flip
                lbl = torch.flip(lbl, dims=[1])

        return img, lbl


# ── Model ──────────────────────────────────────────────────────────────────────

class DINOv2SegHead(nn.Module):
    """
    Frozen DINOv2-S/14 backbone + learnable multi-scale segmentation decoder.
    Input:  (B, 3, 512, 512) — ImageNet-normalised
    Output: (B, num_classes, 512, 512) logits
    """
    def __init__(self, num_classes=4, dinov2_name='dinov2_vits14'):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', dinov2_name,
            pretrained=True, verbose=False
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        feat_dim = self.backbone.embed_dim   # 384 for vits14

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
        # DINOv2 patch size is 14 — input must be a multiple of 14
        DIN_SIZE = 504  # 36 × 14, closest multiple of 14 to 512
        x_din = F.interpolate(x, size=(DIN_SIZE, DIN_SIZE), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = self.backbone.forward_features(x_din)
        # patch tokens: (B, N_patches, feat_dim), N = (DIN_SIZE/14)^2 = 1296
        patch_tokens = feats['x_norm_patchtokens']
        ph = pw = DIN_SIZE // 14  # 36
        feat_map = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)  # (B, C, 36, 36)
        feat_map = F.interpolate(feat_map, size=(H, W), mode='bilinear', align_corners=False)
        return self.decoder(feat_map)


# ── Training ───────────────────────────────────────────────────────────────────

def dice_coeff(pred_logits, gt, cls):
    pred = (pred_logits.argmax(1) == cls)
    g    = (gt == cls)
    if g.sum() == 0 and pred.sum() == 0:
        return 1.0
    if g.sum() == 0:
        return 0.0
    return float(2 * (pred & g).sum()) / float(pred.sum() + g.sum())


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)

    all_ids = list(range(1, 101))
    train_ids = [i for i in all_ids if i not in VAL_IDS]

    train_ds = ACDCSliceDataset(PREP_DIR, train_ids, frame='both', augment=True)
    val_ds   = ACDCSliceDataset(PREP_DIR, VAL_IDS,   frame='es',   augment=False)
    print(f"Train slices: {len(train_ds)}  Val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = DINOv2SegHead(num_classes=4).to(device)
    print(f"DINOv2 backbone frozen. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # Class weights: background is overwhelming → upweight foreground
    class_weights = torch.tensor([0.1, 1.0, 2.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_lv_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            dices = {1: [], 2: [], 3: []}
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    for cls in [1, 2, 3]:
                        dices[cls].append(dice_coeff(logits, lbls, cls))
            lv = np.mean(dices[3])
            print(f"Epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  "
                  f"RV={np.mean(dices[1]):.3f}  Myo={np.mean(dices[2]):.3f}  LV={lv:.3f}")
            if lv > best_lv_dice:
                best_lv_dice = lv
                torch.save(model.state_dict(), CKPT_OUT)
                print(f"  → Saved best model (LV={lv:.3f})")

    print(f"\nTraining done. Best LV Dice: {best_lv_dice:.3f}")
    return best_lv_dice


def _predict_frame(model, d, t_idx, device):
    frame = d['frames'][t_idx].astype(np.float32)   # (3, 512, 512)
    img   = torch.from_numpy(frame).unsqueeze(0).to(device)
    img   = (img - IMG_MEAN.to(device)) / IMG_STD.to(device)
    with torch.no_grad():
        return model(img).argmax(1).squeeze(0).cpu().numpy()   # (512, 512)


def _decode_group(raw) -> str:
    s = str(raw)
    if s.startswith("np.bytes_(b'") and s.endswith("')"): s = s[len("np.bytes_(b'"):-2]
    elif s.startswith("b'") and s.endswith("'"): s = s[2:-1]
    return s


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DINOv2SegHead(num_classes=4).to(device)
    model.load_state_dict(torch.load(CKPT_OUT, map_location=device))
    model.eval()
    print(f"Loaded model from {CKPT_OUT}")

    results = {}
    for pid in tqdm(VAL_IDS, desc='Evaluating'):
        npzs = sorted(glob(os.path.join(PREP_DIR, f'patient{pid:03d}_slice*.npz')))
        if not npzs:
            continue

        slice_dices  = {1: [], 2: [], 3: []}
        pred_edv_vox = 0.0;  pred_esv_vox = 0.0
        gt_edv_vox   = 0.0;  gt_esv_vox   = 0.0
        voxel_vol    = None
        group        = 'UNK'

        for npz_path in npzs:
            d = np.load(npz_path, allow_pickle=True)
            es_t = int(d['es_idx']); ed_t = int(d['ed_idx'])
            es_mask = d['es_mask']; ed_mask = d['ed_mask']

            if voxel_vol is None:
                pixdim = d['pixdim'].astype(np.float64)
                orig_H = int(d['orig_H']); orig_W = int(d['orig_W'])
                scale  = (orig_H / 512.0) * (orig_W / 512.0)
                voxel_vol = float(pixdim[0]) * float(pixdim[1]) * float(pixdim[2]) * scale
                group = _decode_group(d['group'])

            # ES prediction → Dice + ESV
            pred_es = _predict_frame(model, d, es_t, device)
            for cls in [1, 2, 3]:
                p = (pred_es == cls); g = (es_mask == cls)
                if g.sum() == 0 and p.sum() == 0:
                    slice_dices[cls].append(1.0)
                elif g.sum() == 0:
                    slice_dices[cls].append(0.0)
                else:
                    slice_dices[cls].append(float(2*(p&g).sum()) / float(p.sum()+g.sum()))
            pred_esv_vox += (pred_es == 3).sum()
            gt_esv_vox   += (es_mask  == 3).sum()

            # ED prediction → EDV
            pred_ed = _predict_frame(model, d, ed_t, device)
            pred_edv_vox += (pred_ed == 3).sum()
            gt_edv_vox   += (ed_mask  == 3).sum()

        pred_edv = pred_edv_vox * voxel_vol / 1000.0   # mL
        pred_esv = pred_esv_vox * voxel_vol / 1000.0
        pred_ef  = (pred_edv - pred_esv) / pred_edv * 100.0 if pred_edv > 0 else None

        gt_edv = gt_edv_vox * voxel_vol / 1000.0
        gt_esv = gt_esv_vox * voxel_vol / 1000.0
        gt_ef  = (gt_edv - gt_esv) / gt_edv * 100.0 if gt_edv > 0 else None

        results[pid] = {
            'RV':      float(np.mean(slice_dices[1])),
            'Myo':     float(np.mean(slice_dices[2])),
            'LV':      float(np.mean(slice_dices[3])),
            'mean':    float(np.mean([np.mean(slice_dices[c]) for c in [1,2,3]])),
            'group':   group,
            'pred_EDV': float(pred_edv),
            'pred_ESV': float(pred_esv),
            'pred_EF':  float(pred_ef) if pred_ef is not None else None,
            'gt_EDV':   float(gt_edv),
            'gt_ESV':   float(gt_esv),
            'gt_EF':    float(gt_ef)  if gt_ef  is not None else None,
        }

    rv  = np.mean([v['RV']  for v in results.values()])
    myo = np.mean([v['Myo'] for v in results.values()])
    lv  = np.mean([v['LV']  for v in results.values()])
    ef_mae = np.mean([abs(v['pred_EF'] - v['gt_EF'])
                      for v in results.values()
                      if v['pred_EF'] is not None and v['gt_EF'] is not None])
    print(f"\nDINOv2 val — RV: {rv:.3f}  Myo: {myo:.3f}  LV: {lv:.3f}  "
          f"Mean: {np.mean([rv,myo,lv]):.3f}  EF MAE: {ef_mae:.1f}%")

    with open(RESULT_OUT, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved to {RESULT_OUT}")

    _merge_dinov2_into_metrics(results)


def _merge_dinov2_into_metrics(results):
    """Upsert DINOv2 key into metrics_acdc_val.json."""
    existing = {}
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            existing = json.load(f)

    dinov2_list = []
    for pid, v in results.items():
        dinov2_list.append({
            'pid':   f'patient{int(pid):03d}',
            'group': v.get('group', ''),
            'dice_RV':    v['RV'],
            'dice_Myo':   v['Myo'],
            'dice_LV':    v['LV'],
            'pred_EDV':   v.get('pred_EDV'),
            'pred_ESV':   v.get('pred_ESV'),
            'pred_EF':    v.get('pred_EF'),
            'gt_EDV':     v.get('gt_EDV'),
            'gt_ESV':     v.get('gt_ESV'),
            'gt_EF':      v.get('gt_EF'),
        })

    existing['DINOv2'] = dinov2_list
    with open(METRICS_JSON, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Merged DINOv2 into {METRICS_JSON}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,   default=50)
    parser.add_argument('--batch',  type=int,   default=8)
    parser.add_argument('--lr',     type=float, default=1e-4)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    if not args.eval_only:
        train(args)
    evaluate(args)


if __name__ == '__main__':
    main()
