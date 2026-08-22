#!/usr/bin/env python3
# =============================================================================
# eval_vae.py — Task 18/B1: Sample-Evaluation des CVAE (Schaerfe/Diversitaet)
# =============================================================================
#
# ZWECK
#   Beantwortet die Kapitel-Kernfrage: sind Prior-SAMPLES schaerfer und
#   vielfaeltiger als die deterministische Mittelwert-Vorhersage (z=mean)?
#   Pro Subset-Sample werden K Prior-Samples gezogen und gegen mean/real
#   verglichen — im Latent-Raum UND nach Decode (Masken).
#
# METRIKEN (Aggregat ueber das Subset, JSON-Output)
#   sharpness_ratio_{mean,sample}: Gradient-Magnitude (mittlere |dx|+|dy| ueber
#     Kanaele/Pixel) der Vorhersage relativ zur reellen Zielframe-Magnitude.
#     Naeher an 1.0 = so scharf wie die Realitaet.
#   std_ratio_{mean,sample}: Kanal-std-Verhaeltnis pred/real (16b-Metrik).
#   diversity_latent: mittlere Pixel-std ueber die K Samples, normiert auf die
#     reale Kanal-std (0 = deterministisch).
#   diversity_mask: mittlere paarweise Uneinigkeit der binaeren Masken der K
#     Samples (Anteil Zellen, 0 = alle Samples identisch).
#   miou_{mean,sample}: mIoU der Mittelwert-Vorhersage bzw. Durchschnitt der
#     Sample-mIoUs (Konsistenz zur Guard-Zahl / Kosten des Samplings).
#
# PANEL (--panel_npz): fuer die ersten --panel_n Subset-Samples werden die
#   decodierten Masken (real / mean / erste 3 Samples) als npz gesichert —
#   Render lokal via render_18_panel.py (bevwm-Env hat kein matplotlib).
#
# USAGE (Cluster, 1 GPU)
#   python -u eval_vae.py --config config_vae_0.01.yaml \
#       --checkpoint checkpoints/task18/vae_0.01/phase2/best_miou.pt \
#       --output predictions/task18/eval_vae_0.01.json \
#       --panel_npz predictions/task18/panel_0.01.npz
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(description="CVAE-Sample-Eval (Task 18/B1).")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--output",     required=True, help="Ziel-JSON")
    p.add_argument("--n_subset",   type=int, default=300,
                   help="Subset-Groesse (seed-42-Auswahl wie im Training)")
    p.add_argument("--k",          type=int, default=8, help="Prior-Samples pro Input")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--panel_npz",  default=None,
                   help="Optional: Masken-Panel-Daten (real/mean/3 Samples) als npz")
    p.add_argument("--panel_n",    type=int, default=3,
                   help="Anzahl Subset-Samples fuers Panel")
    p.add_argument("--flow_steps", type=int, default=10,
                   help="18/B2: Euler-Schritte je Flow-Sample (nur flow_head=flow)")
    p.add_argument("--flow_scale", type=float, default=1.0,
                   help="18/B2.1: Residual-Skalierung s (x = x_det + s*r_hat) "
                        "zur post-hoc Varianz-Kalibrierung")
    return p.parse_args()


def grad_mag(x):
    """Mittlere Gradient-Magnitude |dx|+|dy| eines Latents [C,H,W] (fp32)."""
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
    dy = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
    return (dx + dy).item()


def chan_std(x):
    """Mittlere Kanal-std eines Latents [C,H,W] (16b-Konvention)."""
    return x.reshape(x.shape[0], -1).std(dim=1).mean().item()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit("[eval_vae] seg_decoder_torch nicht importierbar.")

    model_config = T.ModelConfig(**cfg["model"])
    model = T.BEVWorldModel(model_config).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()
    is_vae  = getattr(model_config, "vae_mode", "off") == "cvae"
    is_flow = getattr(model_config, "flow_head", "off") == "flow"
    if is_flow:
        print(f"[eval_vae] Flow-Modus: Samples via flow_sample "
              f"(steps={args.flow_steps}); mean = deterministisches Backbone.")
    elif not is_vae:
        print("[eval_vae] WARNUNG: vae_mode=off — Samples werden identisch sein "
              "(Skript laeuft als Referenz-Baseline trotzdem durch).")

    decode_cfg = cfg.get("decode_validation", {})
    seg_decoder = T.build_seg_decoder(decode_cfg.get("seg_checkpoint"), device, verbose=True)

    print("[eval_vae] Baue DataLoader...")
    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    rng = np.random.default_rng(args.seed)
    n_take = min(args.n_subset, len(dataset))
    indices = sorted(rng.choice(len(dataset), size=n_take, replace=False).tolist())

    agg = {k: [] for k in ["sh_mean", "sh_sample", "std_mean", "std_sample",
                            "div_lat", "div_mask", "miou_mean", "miou_sample",
                            "miou_best"]}
    panel = {"real": [], "mean": [], "samples": []}
    use_amp = (device.type == "cuda")
    t0 = time.time()

    for n, i in enumerate(indices):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device).float()
        tgt = sample["target"].float()
        ego = sample["ego_delta"].unsqueeze(0).to(device).float() if "ego_delta" in sample else None

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_mean = model(inp, ego, z_mode="mean")[0].float()
            if is_flow:
                preds = [model.flow_sample(inp, ego, steps=args.flow_steps,
                                           scale=args.flow_scale)[0].float()
                         for _ in range(args.k)]
            else:
                preds = [model(inp, ego, z_mode="sample")[0].float() for _ in range(args.k)]

        tgt_dev = tgt.to(device)
        g_real = grad_mag(tgt_dev)
        s_real = chan_std(tgt_dev)
        agg["sh_mean"].append(grad_mag(pred_mean) / g_real)
        agg["sh_sample"].append(float(np.mean([grad_mag(p) / g_real for p in preds])))
        agg["std_mean"].append(chan_std(pred_mean) / s_real)
        agg["std_sample"].append(float(np.mean([chan_std(p) / s_real for p in preds])))

        stack = torch.stack(preds)                      # [K,C,H,W]
        agg["div_lat"].append((stack.std(dim=0).mean() / s_real).item())

        # --- Decode: real, mean, K Samples -> Masken + mIoU + Mask-Diversitaet
        real_mask, _ = seg_decoder.latent_to_mask(tgt, device=device)
        mean_mask, _ = seg_decoder.latent_to_mask(pred_mean, device=device)
        smasks = [seg_decoder.latent_to_mask(p, device=device)[0] for p in preds]
        agg["miou_mean"].append(T.compute_iou(mean_mask, real_mask, T.MAP_CLASSES)["mIoU"])
        _sample_ious = [T.compute_iou(m, real_mask, T.MAP_CLASSES)["mIoU"] for m in smasks]
        agg["miou_sample"].append(float(np.mean(_sample_ious)))
        # Best-of-K (SVG-Konvention): deckt die VERTEILUNG die echte Zukunft ab?
        # Nur sinnvoll interpretierbar, wo Diversitaet existiert.
        agg["miou_best"].append(float(np.max(_sample_ious)))
        sm = np.stack(smasks)                           # [K,n_cls,200,200] bool
        # paarweise Uneinigkeit = Anteil Zellen, in denen nicht alle K gleich sind
        agg["div_mask"].append(float((sm.any(axis=0) & ~sm.all(axis=0)).mean()))

        if args.panel_npz and n < args.panel_n:
            panel["real"].append(real_mask)
            panel["mean"].append(mean_mask)
            panel["samples"].append(np.stack(smasks[:3]))

        if (n + 1) % 25 == 0 or (n + 1) == n_take:
            el = time.time() - t0
            print(f"  [{n+1:>4}/{n_take}]  {el:6.1f}s  "
                  f"sh_s={np.mean(agg['sh_sample']):.4f}  div_m={np.mean(agg['div_mask']):.4f}",
                  flush=True)

    result = {
        "checkpoint": args.checkpoint, "config": args.config,
        "n_subset": n_take, "k": args.k, "is_vae": is_vae,
        "is_flow": is_flow, "flow_steps": args.flow_steps if is_flow else None,
        "flow_scale": args.flow_scale if is_flow else None,
        "sharpness_ratio_mean":   round(float(np.mean(agg["sh_mean"])), 4),
        "sharpness_ratio_sample": round(float(np.mean(agg["sh_sample"])), 4),
        "std_ratio_mean":         round(float(np.mean(agg["std_mean"])), 4),
        "std_ratio_sample":       round(float(np.mean(agg["std_sample"])), 4),
        "diversity_latent":       round(float(np.mean(agg["div_lat"])), 4),
        "diversity_mask":         round(float(np.mean(agg["div_mask"])), 4),
        "miou_mean":              round(float(np.mean(agg["miou_mean"])), 4),
        "miou_sample":            round(float(np.mean(agg["miou_sample"])), 4),
        "miou_best_of_k":         round(float(np.mean(agg["miou_best"])), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[eval_vae] FERTIG -> {out}")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v}")

    if args.panel_npz:
        pnpz = Path(args.panel_npz)
        pnpz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pnpz,
            real=np.stack(panel["real"]),        # [P,n_cls,200,200]
            mean=np.stack(panel["mean"]),
            samples=np.stack(panel["samples"]),  # [P,3,n_cls,200,200]
            classes=np.array(T.MAP_CLASSES),
        )
        print(f"[eval_vae] Panel-Daten -> {pnpz}")


if __name__ == "__main__":
    main()
