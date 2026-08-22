#!/usr/bin/env python3
# =============================================================================
# visualize_thesis.py  —  Task 17 (COMPUTE-Teil, Cluster): Masken + Gate-Alpha
# =============================================================================
# Decodiert fuer N Val-Beispiele die pred- und real-Masken (Seg-Decoder, GPU)
# und greift den Gate-Alpha per Forward-Hook. Speichert ALLES als .npz --
# das RENDERING macht render_thesis.py LOKAL (matplotlib fehlt im bevwm-Env).
# KEIN Retraining, nur Inferenz + Decode.
#
# USAGE (PLAIN, kein --export):
#   python -u visualize_thesis.py --config config_rollout.yaml \
#       --checkpoint checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt \
#       --phase cell --n 6 --out predictions/task16c/../task17/thesis_masks.npz
# =============================================================================
import argparse
from pathlib import Path

import numpy as np
import torch

import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(description="Task 17 compute: pred/real Masken + Gate-Alpha -> .npz")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--n",          type=int, default=6)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="predictions/task17/thesis_masks.npz")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit("[viz] seg_decoder_torch nicht importierbar.")

    model = T.BEVWorldModel(T.ModelConfig(**cfg["model"])).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    _cap = {}
    h = model.gate.register_forward_hook(lambda m, i, o: _cap.__setitem__("alpha", o.detach()))

    seg_ckpt = cfg.get("decode_validation", {}).get("seg_checkpoint")
    seg_decoder = T.build_seg_decoder(seg_ckpt, device, verbose=True)

    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False).tolist())
    print(f"[viz] {len(idx)} Val-Beispiele: {idx}")

    preds, reals, alphas, mious = [], [], [], []
    for i in idx:
        s = dataset[i]
        inp = s["input"].unsqueeze(0).to(device).float()
        tgt = s["target"].float()
        pred = model(inp)[0].float()
        alphas.append(_cap["alpha"][0].mean(dim=0).cpu().numpy().astype(np.float32))   # [128,128]
        pm, _ = seg_decoder.latent_to_mask(pred, device=device)
        rm, _ = seg_decoder.latent_to_mask(tgt,  device=device)
        pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
        rm = (rm.cpu().numpy() if torch.is_tensor(rm) else np.asarray(rm)).astype(bool)
        preds.append(pm); reals.append(rm)
        mious.append(float(T.compute_iou(pm, rm, T.MAP_CLASSES)["mIoU"]))
        print(f"  Bsp {i}: mIoU={mious[-1]:.4f}  mask {pm.shape}  alpha_mean={alphas[-1].mean():.3f}")
    h.remove()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        idx=np.array(idx), mious=np.array(mious, dtype=np.float32),
                        preds=np.stack(preds), reals=np.stack(reals),
                        alphas=np.stack(alphas))
    print(f"[viz] gespeichert -> {out}  ({len(idx)} Beispiele, mask-shape {preds[0].shape})")


if __name__ == "__main__":
    main()
