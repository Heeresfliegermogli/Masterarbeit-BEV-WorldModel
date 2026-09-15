#!/usr/bin/env python3
# =============================================================================
# adapt_seg_head.py — Decoder-Adaptation auf Weltmodell-Vorhersagen (Seg-Strang)
# =============================================================================
# Laeuft IM DOCKER (seg_gate). Trainiert den Post-Fuser-Stack von BEVFusion-Seg
# (decoder.backbone SECOND + decoder.neck SECONDFPN + heads.map SegHead, Focal-
# Loss unveraendert) auf (Vorhersage-Latent -> echte nuScenes-GT) OHNE die
# Sensor-Pipeline: Latents aus dem Weltmodell-Train-Dump, GT aus dem gepackten
# 200x200-Pipeline-Grid-Cache (mk_gt_masks_seg --grid pipe, Geometrie-Gate
# bestanden). Alles vor dem Fuser bleibt unangetastet (wird nicht ausgefuehrt).
#
# Ausgabe: /output/adapted/seg_adapted_<mode>_best.pth (+ _last.pth) im
# load_checkpoint-Format -> direkt in der Injection-Kette (tools/test.py)
# als Checkpoint-Ersatz nutzbar.
#
# USAGE (im Container, /bevfusion als workdir):
#   python /output/adapt_seg_head.py --mode full  --epochs 8
#   python /output/adapt_seg_head.py --mode head  --epochs 8   (Ablation)
#   ... --smoke  (5 Steps + 1 Val-Batch, Pipeline-Test)
# =============================================================================
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

CFG_PATH = "configs/nuscenes/seg/fusion-bev256d2-lss.yaml"
CKPT = "pretrained/bevfusion-seg.pth"
TRAIN_LATENTS = Path("/output/dump_train_minimal")
VAL_LATENTS = Path("/output/dump_seg_minimal")
GT_TRAIN = "/output/gt_masks_seg/gt_masks_pipe_train.npz"
GT_VAL = "/output/gt_masks_seg/gt_masks_pipe_val.npz"
OUT = Path("/output/adapted")
CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
           "carpark_area", "divider"]


class LatentGTDataset(Dataset):
    """(Vorhersage-Latent [256,128,128] fp32, GT-Maske [6,200,200] float)."""

    def __init__(self, latent_dir, gt_npz_path, tokens):
        self.latent_dir = Path(latent_dir)
        self.gt_path = gt_npz_path
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        if not hasattr(self, "_gt"):                    # lazy je Worker
            self._gt = np.load(self.gt_path)
        x = np.load(self.latent_dir / f"bev_latent_{tok}.npy").astype(np.float32)
        if x.ndim == 4:
            x = x[0]
        gt = np.unpackbits(self._gt[tok]).reshape(6, 200, 200).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gt)


@torch.no_grad()
def val_iou(model, loader, device, max_batches=None):
    """Schnelle Proxy-IoU (Schwelle 0.5) auf Val-VORHERSAGEN vs GT."""
    model.eval()
    inter = torch.zeros(6, device=device)
    union = torch.zeros(6, device=device)
    for bi, (x, gt) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, gt = x.to(device), gt.to(device).bool()
        feat = model.decoder["backbone"](x)
        feat = model.decoder["neck"](feat)
        probs = model.heads["map"](feat)   # eval-Branch liefert BEREITS sigmoid!
        pred = probs > 0.5
        inter += (pred & gt).flatten(2).sum(dim=(0, 2)).float()
        union += (pred | gt).flatten(2).sum(dim=(0, 2)).float()
    model.train()
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return float(iou.mean()), {c: float(v) for c, v in zip(CLASSES, iou)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "head"], default="full",
                    help="full = decoder.backbone+neck+heads.map | head = nur heads.map")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed des Kopf-Trainings (Init+Shuffle)")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    configs.load(CFG_PATH, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CFG_PATH)
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu")
    model = model.to(device)

    # --- Einfrieren: nur der gewaehlte Post-Fuser-Teil lernt -----------------
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.heads["map"].parameters())
    if args.mode == "full":
        trainable += list(model.decoder["backbone"].parameters())
        trainable += list(model.decoder["neck"].parameters())
    for p in trainable:
        p.requires_grad = True
    model.train()
    if args.mode == "head":                      # BN im frozen Decoder fixieren
        model.decoder["backbone"].eval()
        model.decoder["neck"].eval()
    n_tr = sum(p.numel() for p in trainable)
    print(f"[adapt] mode={args.mode}, trainierbar {n_tr/1e6:.2f}M Parameter")

    gt_train = np.load(GT_TRAIN)
    tokens = sorted(f.stem.replace("bev_latent_", "")
                    for f in TRAIN_LATENTS.glob("bev_latent_*.npy"))
    tokens = [t for t in tokens if t in gt_train.files]
    gt_val = np.load(GT_VAL)
    vtokens = sorted(f.stem.replace("bev_latent_", "")
                     for f in VAL_LATENTS.glob("bev_latent_*.npy"))
    vtokens = [t for t in vtokens if t in gt_val.files][:1000]
    print(f"[adapt] {len(tokens)} Train- / {len(vtokens)} Val-Paare")

    dl = DataLoader(LatentGTDataset(TRAIN_LATENTS, GT_TRAIN, tokens),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vdl = DataLoader(LatentGTDataset(VAL_LATENTS, GT_VAL, vtokens),
                     batch_size=args.batch, shuffle=False, num_workers=4)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    OUT.mkdir(exist_ok=True)

    m0, per0 = val_iou(model, vdl, device, max_batches=(1 if args.smoke else None))
    print(f"[adapt] VOR Adaptation: Proxy-IoU {m0:.4f} | " +
          " ".join(f"{c.split('_')[0]}={v:.3f}" for c, v in per0.items()), flush=True)

    best = m0
    for ep in range(1, args.epochs + 1):
        t0, run = time.time(), 0.0
        for bi, (x, gt) in enumerate(dl):
            x, gt = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feat = model.decoder["backbone"](x)
                feat = model.decoder["neck"](feat)
                losses = model.heads["map"](feat, gt)   # Focal-Loss-Dict
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            if (bi + 1) % 100 == 0 or args.smoke:
                print(f"  ep{ep} [{bi+1:>5}/{len(dl)}] loss={run/(bi+1):.4f} "
                      f"{(bi+1)/(time.time()-t0):4.1f} it/s", flush=True)
            if args.smoke and bi >= 4:
                break
        miou, per = val_iou(model, vdl, device,
                            max_batches=(1 if args.smoke else None))
        print(f"[adapt] ep{ep}: Proxy-IoU {miou:.4f} (vor: {m0:.4f}) | " +
              " ".join(f"{c.split('_')[0]}={v:.3f}" for c, v in per.items()), flush=True)
        torch.save({"state_dict": model.state_dict()},
                   OUT / (f"seg_adapted_{args.mode}_last.pth" if args.seed == 42
                   else f"seg_adapted_{args.mode}_s{args.seed}_last.pth"))
        if miou > best:
            best = miou
            torch.save({"state_dict": model.state_dict()},
                       OUT / (f"seg_adapted_{args.mode}_best.pth" if args.seed == 42
                          else f"seg_adapted_{args.mode}_s{args.seed}_best.pth"))
            print(f"[adapt] neues Best ({best:.4f}) gespeichert", flush=True)
        if args.smoke:
            break
    print(f"[adapt] FERTIG mode={args.mode} best={best:.4f} (Start {m0:.4f})")


if __name__ == "__main__":
    main()
