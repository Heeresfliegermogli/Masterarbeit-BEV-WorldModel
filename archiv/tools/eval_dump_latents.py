#!/usr/bin/env python3
# =============================================================================
# eval_dump_latents.py — Task 20.1/Schritt 4: Vorhersage-Latents dumpen (Det)
# =============================================================================
#
# ZWECK
#   Det-Gegenstueck zu eval_full_val.py: statt Decode+mIoU werden die
#   VORHERSAGE-LATENTS als npy-Dateien gespeichert (je Target-Token), damit die
#   lokale Docker-Injection-Kette (Task 20.1) sie zu mAP/NDS bewerten kann.
#   - Vorhersagen entstehen fuer alle Fenster (Frames mit >=3 Vorgaengern).
#   - Frames OHNE Kontext (erste 3 je Szene) werden mit dem REALEN Latent
#     gefuellt (--fill_real, Default an) — identische Konvention fuer alle
#     Varianten inkl. Persistenz-Baseline, dokumentiert im 20.1-Protokoll.
#   - Rueckskalierung: Vorhersagen entstehen im NORMALISIERTEN Raum
#     (data.latent_scale, z.B. 0.3641) und werden vor dem Speichern durch den
#     Faktor geteilt -> Dump in ROH-Skala, fp16, Shape (1,256,H,W) wie die
#     Original-Extraktion (Injection-Hook erwartet dieses Format).
#
# USAGE (Cluster, 1 GPU; danach rsync des Output-Dirs zum lokalen Server)
#   python -u eval_dump_latents.py --config config_det_base.yaml \
#       --checkpoint checkpoints/task20/base/phase2/best_val_loss.pt \
#       --output_dir predictions/task20/dump_base
# =============================================================================

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(description="Task 20.1: Vorhersage-Latents dumpen.")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--output_dir", required=True, help="Zielordner fuer bev_latent_<tok>.npy")
    p.add_argument("--fill_real",  action="store_true", default=True,
                   help="Frames ohne Kontext mit realem Latent fuellen (Default an)")
    p.add_argument("--no_fill",    action="store_true", default=False,
                   help="Task 23: Real-Fill AUS (nur echte Vorhersagen dumpen)")
    p.add_argument("--split",      choices=["val", "train"], default="val",
                   help="Task 23: welcher Split gedumpt wird (Default val = wie bisher)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Nur N Fenster (Smoke). Default None = alle.")
    p.add_argument("--stride", type=int, default=1,
                   help="Task 24: nur jedes k-te Fenster (Platz; Default 1 = alle)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = float(cfg.get("data", {}).get("latent_scale", 1.0))
    inv = 1.0 / scale
    print(f"[dump] latent_scale={scale} -> Rueckskalierung x{inv:.4f} (Roh-Skala)")

    model_config = T.ModelConfig(**cfg["model"])
    model = T.BEVWorldModel(model_config).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    print("[dump] Baue DataLoader...")
    train_loader, val_loader, _ = T.build_dataloaders(cfg)
    dataset = (train_loader if args.split == "train" else val_loader).dataset
    if args.no_fill:
        args.fill_real = False
    print(f"[dump] split={args.split}, fill_real={args.fill_real}, "
          f"{len(dataset)} Fenster")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_amp = (device.type == "cuda")

    # --- 1) Vorhersagen fuer alle Fenster ------------------------------------
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    # Platz-Vorpruefung: ein Dump belegt (Fenster + Real-Fill) x Latentgroesse.
    # LEKTION 29.07.: ohne Pruefung stirbt der Job nach ~45 min an ENOSPC und
    # laesst einen halben Dump zurueck (Cluster-/home lief dadurch auf 100%).
    full_dump = args.max_samples is None and args.fill_real
    n_files = sum(len(sc) for sc in dataset._all_scenes) if full_dump else n
    c, h, w = dataset[0]["target"].shape[-3:]
    need = n_files * c * h * w * 2                      # fp16
    free = shutil.disk_usage(out).free
    print(f"[dump] Platzbedarf ~{need/2**30:.0f} GiB fuer {n_files} Dateien, "
          f"frei {free/2**30:.0f} GiB ({out})")
    if free < need * 1.05:
        sys.exit(f"[dump] ABBRUCH: zu wenig Platz unter {out} "
                 f"({free/2**30:.0f} GiB frei, ~{need/2**30:.0f} GiB noetig). "
                 f"Grosse Dumps NICHT ins Cluster-/home, sondern auf BeeGFS "
                 f"(/mnt/beegfs/ssd/lrt81-students/lrt81-vima/...).")
    t0 = time.time()
    seen = set()
    for i in range(0, n, args.stride):
        s = dataset[i]
        inp = s["input"].unsqueeze(0).to(device).float()
        ego = s["ego_delta"].unsqueeze(0).to(device).float() if "ego_delta" in s else None
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(inp, ego)                     # normalisierter Raum
        tok = s["target_token"]
        arr = (pred[0].float() * inv).cpu().numpy().astype(np.float16)[None]  # (1,C,H,W) roh
        np.save(out / f"bev_latent_{tok}.npy", arr)
        seen.add(tok)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            el = time.time() - t0
            print(f"  [{i+1:>5}/{n}] {(i+1)/el:5.1f} it/s", flush=True)

    # --- 2) Real-Fill fuer kontextlose Frames --------------------------------
    n_fill = 0
    if args.fill_real and args.max_samples is None:
        for scene in dataset._all_scenes:   # alle Val-Szenen (explicit-Modus)
            for fr in scene:
                tok = fr["token"]
                if tok in seen:
                    continue
                real = dataset._load_latent(fr)                    # normalisiert (dataset-scale)
                arr = (np.asarray(real, dtype=np.float32) * inv).astype(np.float16)
                if arr.ndim == 3:
                    arr = arr[None]
                np.save(out / f"bev_latent_{tok}.npy", arr)
                n_fill += 1

    print(f"[dump] FERTIG: {len(seen)} Vorhersagen + {n_fill} Real-Fill -> {out}")


if __name__ == "__main__":
    main()
