#!/usr/bin/env python3
# =============================================================================
# filmstrip_compute.py  —  Task 17: 7-Frame-Rollout-Filmstrip (COMPUTE, Cluster)
# =============================================================================
# Waehlt automatisch je EINE statische (niedriger Gate-Alpha) und EINE dynamische
# (hoher Gate-Alpha) 7-Frame-Szenensequenz, rollt sie autoregressiv aus
# (F1F2F3 -> F4p; F2F3F4p -> F5p; ...) und decodiert:
#   real  F1..F7  (Ground-Truth-Sequenz)      -> [7, n_cls,200,200]
#   pred  F4..F7  (Rollout)                    -> [4, n_cls,200,200]
#   mIoU(k) je Schritt + Gate-Alpha je Schritt ([4,128,128])
# Speichert alles als .npz -- RENDER lokal (filmstrip_render.py), da bevwm-Env
# kein matplotlib hat. KEIN Retraining.
#
# USAGE (PLAIN, kein --export):
#   python -u filmstrip_compute.py --config config_rollout.yaml \
#       --checkpoint checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt \
#       --phase cell --k 4 --scan 200 --out predictions/task17/filmstrip.npz
# =============================================================================
import argparse
from pathlib import Path
import numpy as np
import torch
import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(description="Task 17: 7-Frame-Rollout-Filmstrip (compute).")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--k",          type=int, default=4)
    p.add_argument("--scan",       type=int, default=200, help="Kandidaten-Fenster fuer die Auswahl.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="predictions/task17/filmstrip.npz")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit("[film] seg_decoder_torch nicht importierbar.")

    n_input = cfg["model"].get("n_frames", 3)
    win_len = n_input + args.k

    model = T.BEVWorldModel(T.ModelConfig(**cfg["model"])).to(device); model.eval()
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    _cap = {}
    h = model.gate.register_forward_hook(lambda m, i, o: _cap.__setitem__("alpha", o.detach()))
    seg = T.build_seg_decoder(cfg.get("decode_validation", {}).get("seg_checkpoint"), device, verbose=True)

    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    windows = []
    for scene in dataset._all_scenes:
        for i in range(len(scene) - win_len + 1):
            windows.append(scene[i:i + win_len])

    # --- Kandidaten scannen: mittlerer Gate-Alpha im ERSTEN Schritt (Proxy statisch/dynamisch)
    rng = np.random.default_rng(args.seed)
    cand = sorted(rng.choice(len(windows), size=min(args.scan, len(windows)), replace=False).tolist())
    a_of = {}
    for wi in cand:
        lat = [torch.from_numpy(dataset._load_latent(f)).float().to(device) for f in windows[wi][:n_input]]
        cur = torch.stack(lat, dim=0).unsqueeze(0)
        _ = model(cur)
        a_of[wi] = float(_cap["alpha"][0].mean().item())
    static_wi  = min(a_of, key=a_of.get)     # niedrigster Alpha
    dynamic_wi = max(a_of, key=a_of.get)      # hoechster Alpha
    print(f"[film] statisch  Fenster {static_wi}  (mean-alpha {a_of[static_wi]:.3f})")
    print(f"[film] dynamisch Fenster {dynamic_wi}  (mean-alpha {a_of[dynamic_wi]:.3f})")

    def rollout_full(win):
        lat = [torch.from_numpy(dataset._load_latent(f)).float().to(device) for f in win]
        real_masks = []
        for f in range(win_len):                      # decode ALLE echten Frames F1..F_{win_len}
            m, _ = seg.latent_to_mask(lat[f], device=device)
            real_masks.append((m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)).astype(bool))
        cur = torch.stack(lat[:n_input], dim=0).unsqueeze(0)
        pred_masks, mious, alphas = [], [], []
        for k in range(1, args.k + 1):
            pred = model(cur)
            alphas.append(_cap["alpha"][0].mean(dim=0).cpu().numpy().astype(np.float32))
            pl = pred[0].float()
            pm, _ = seg.latent_to_mask(pl, device=device)
            pm = (pm.cpu().numpy() if torch.is_tensor(pm) else np.asarray(pm)).astype(bool)
            pred_masks.append(pm)
            mious.append(float(T.compute_iou(pm, real_masks[n_input - 1 + k], T.MAP_CLASSES)["mIoU"]))
            cur = torch.cat([cur[:, 1:], pl.unsqueeze(0).unsqueeze(0)], dim=1)
        return (np.stack(real_masks), np.stack(pred_masks),
                np.array(mious, dtype=np.float32), np.stack(alphas))

    seqs = {}
    for name, wi in [("static", static_wi), ("dynamic", dynamic_wi)]:
        rm, pm, mi, al = rollout_full(windows[wi])
        seqs[name] = dict(window=wi, real=rm, pred=pm, mious=mi, alphas=al,
                          mean_alpha=a_of[wi])
        print(f"[film] {name}: mIoU je k = {np.round(mi,3).tolist()}")
    h.remove()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    save = {"n_input": n_input, "k": args.k}
    for name in seqs:
        for key in ["real", "pred", "mious", "alphas"]:
            save[f"{name}_{key}"] = seqs[name][key]
        save[f"{name}_window"]     = np.array(seqs[name]["window"])
        save[f"{name}_mean_alpha"] = np.array(seqs[name]["mean_alpha"], dtype=np.float32)
    np.savez_compressed(out, **save)
    print(f"[film] gespeichert -> {out}")


if __name__ == "__main__":
    main()
