#!/usr/bin/env python3
# =============================================================================
# adapt_det_head.py — Task 24.1: TransFusion-Head-Adaptation auf WM-Vorhersagen
# =============================================================================
# Laeuft IM DOCKER (det_extract). Analog adapt_seg_head.py (Task 23), aber Det:
# Post-Fuser-Stack (decoder.backbone SECOND + decoder.neck FPN + heads.object
# TransFusionHead) auf (Vorhersage-Latent [256,180,180] -> GT-Boxen) trainieren.
# GT-Boxen kommen DIREKT aus der infos-pkl (valid_flag-Filter wie
# NuScenesDataset mit use_valid_flag=True) — keine Sensor-Pipeline.
# Loss-Assembly = Original: pred = head(x, metas); head.loss(gt_b, gt_l, pred).
#
# USAGE (im Container, workdir /bevfusion):
#   python /output/adapt_det_head.py --epochs 6 [--smoke]
# Ausgabe: /output/adapted_det/det_adapted_{best,last}.pth (Voll-Format).
# =============================================================================
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.core.bbox import LiDARInstance3DBoxes

CFG_PATH = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CKPT = "pretrained/bevfusion-det.pth"
LATENTS = Path("/output/dump_train_det_baseline")
PKL = "data/nuscenes/nuscenes_infos_train.pkl"
OUT = Path("/output/adapted_det")
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


def build_gt_index():
    """token -> (boxes[N,9], labels[N]) aus der infos-pkl (valid_flag-Filter)."""
    with open(PKL, "rb") as f:
        infos = pickle.load(f)["infos"]
    idx = {}
    for info in infos:
        mask = info["valid_flag"]
        boxes = info["gt_boxes"][mask]
        vel = info["gt_velocity"][mask]
        names = info["gt_names"][mask]
        labels = np.array([CLASSES.index(n) if n in CLASSES else -1
                           for n in names])
        keep = labels >= 0
        boxes = np.concatenate([boxes[keep], vel[keep]], axis=1)  # [N,9]
        nan = np.isnan(boxes)
        boxes[nan] = 0.0
        idx[info["token"]] = (boxes.astype(np.float32), labels[keep].astype(np.int64))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)

    configs.load(CFG_PATH, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CFG_PATH)
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu")
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    trainable = (list(model.heads["object"].parameters())
                 + list(model.decoder["backbone"].parameters())
                 + list(model.decoder["neck"].parameters()))
    for p in trainable:
        p.requires_grad = True
    model.train()
    print(f"[adapt-det] trainierbar {sum(p.numel() for p in trainable)/1e6:.2f}M Param.")

    gt = build_gt_index()
    tokens = sorted(f.stem.replace("bev_latent_", "")
                    for f in LATENTS.glob("bev_latent_*.npy"))
    tokens = [t for t in tokens if t in gt and len(gt[t][1]) > 0]
    print(f"[adapt-det] {len(tokens)} Train-Paare")
    OUT.mkdir(exist_ok=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    rng = np.random.RandomState(42)

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(tokens))
        t0, run, nb = time.time(), 0.0, 0
        for bstart in range(0, len(order), args.batch):
            bidx = order[bstart:bstart + args.batch]
            xs, gbs, gls = [], [], []
            for j in bidx:
                tok = tokens[j]
                a = np.load(LATENTS / f"bev_latent_{tok}.npy").astype(np.float32)
                xs.append(torch.from_numpy(a if a.ndim == 3 else a[0]))
                b, l = gt[tok]
                gbs.append(LiDARInstance3DBoxes(torch.from_numpy(b), box_dim=9))
                gls.append(torch.from_numpy(l).to(device))
            x = torch.stack(xs).to(device)
            metas = [{} for _ in bidx]
            opt.zero_grad(set_to_none=True)
            feat = model.decoder["backbone"](x)
            feat = model.decoder["neck"](feat)
            pred = model.heads["object"](feat, metas)
            losses = model.heads["object"].loss(gbs, gls, pred)
            loss = sum(v for v in losses.values() if v.requires_grad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 35)
            opt.step()
            run += float(loss); nb += 1
            if nb % 100 == 0 or args.smoke:
                print(f"  ep{ep} [{nb:>5}/{(len(order)+args.batch-1)//args.batch}] "
                      f"loss={run/nb:.3f} {nb/(time.time()-t0):4.2f} it/s", flush=True)
            if args.smoke and nb >= 5:
                break
        torch.save({"state_dict": model.state_dict()}, OUT / "det_adapted_last.pth")
        print(f"[adapt-det] ep{ep} fertig, mean loss {run/max(nb,1):.3f} — gespeichert",
              flush=True)
        if args.smoke:
            break
    torch.save({"state_dict": model.state_dict()}, OUT / "det_adapted_best.pth")
    print("[adapt-det] FERTIG")


if __name__ == "__main__":
    main()
