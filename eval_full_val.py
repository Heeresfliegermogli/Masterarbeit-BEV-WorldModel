#!/usr/bin/env python3
# =============================================================================
# eval_full_val.py  —  Full-Val-mIoU (6019 / alle Val-Samples)
# =============================================================================
#
# ZWECK
#   Berechnet die mIoU eines Checkpoints auf dem VOLLSTAENDIGEN Val-Set --
#   die Hauptvergleichszahl der Arbeit ("auf 300 selektieren, auf Full-Val
#   berichten"). Fuer mIoU(lambda_std) ueber die Sweep-Punkte;
#   ausserdem der Rauschboden-Rework (seed 42/43/44 auf Full-Val).
#
# WAS ES *NICHT* NEU ERFINDET
#   Es benutzt exakt die Decode-Kette des Trainings:
#     - Sample-Auswahl:  identische Logik wie build_seg_val_cache()
#     - Modell-Forward:  model(inp)                     (World-Model)
#     - Decode:          seg_decoder.latent_to_mask()   (BEVFusion, float32)
#     - Metrik:          compute_iou(pred, real, MAP_CLASSES)
#   Alle Symbole werden aus train_linux importiert (dessen Modul-Import setzt
#   sys.path auf Code/ und laedt config/bev_world_model/seg_decoder_torch).
#   main() von train_linux laeuft dabei NICHT (steht unter __main__-Guard).
#
# WARUM STREAMING STATT build_seg_val_cache(n_subset=len(dataset))
#   build_seg_val_cache() stapelt ALLE Inputs in EINEN CPU-Tensor. Bei ~5743
#   Val-Sequenzen waeren das [5743,3,256,128,128]*4B ~= 289 GB RAM -> OOM.
#   Das Vorab-Cachen lohnt nur fuer die WIEDERHOLTE Per-Epochen-Validierung
#   (real-Masken einmal decodieren, ueber Epochen wiederverwenden). Bei einem
#   EINMALIGEN Full-Val-Lauf wird jedes Sample ohnehin genau einmal beruehrt
#   -> streamen (laden -> decodieren -> IoU -> verwerfen) ist speichersicher
#   (~70 MB Peak) und liefert bitgleiche Zahlen.
#
# USAGE
#   python -u eval_full_val.py \
#       --config config_nuscenes_task15.yaml \
#       --phase  cell \
#       --checkpoint checkpoints/task15/std_sweep/lstd_3.0/phase2/best_miou.pt \
#       --output predictions/task15_3/fullval/lstd_3.0_fullval.json
#
#   # Sanity-Check gegen den 300er-Trainings-Proxy (seed 42, identisches Subset):
#   python -u eval_full_val.py --config ... --checkpoint ... \
#       --output /tmp/check.json --n_subset 300
#   -> mIoU muss den "Beste mIoU" des Trainingslaufs reproduzieren.
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- train_linux importieren: setzt sys.path (Code/) + stellt alle noetigen
#     Symbole bereit; main() bleibt inaktiv (unter __main__-Guard). -----------
import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-Val-mIoU eines Checkpoints (streamend, speichersicher)."
    )
    p.add_argument("--config",     required=True, help="YAML wie im Training")
    p.add_argument("--checkpoint", required=True, help="Pfad zu best_miou.pt")
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--output",     default=None,
                   help="Ziel-JSON (Default: neben dem Checkpoint als "
                        "miou_fullval.json)")
    p.add_argument("--n_subset",   type=int, default=None,
                   help="Nur fuer Sanity-Checks: statt Full-Val ein Subset "
                        "der Groesse N ziehen (seed-42-Auswahl wie im Training). "
                        "Default None = alle Val-Samples.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Seed der Subset-Auswahl (nur relevant wenn --n_subset "
                        "gesetzt; bei Full-Val irrelevant).")
    p.add_argument("--z_mode",     default="mean", choices=["mean", "sample"],
                   help="18/B1: z aus dem Prior — mean (deterministisch, "
                        "Default) oder sample (stochastisch). Ohne CVAE ignoriert.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, seg_decoder, dataset, device, indices, z_mode="mean"):
    """
    Streamt ueber `indices` des Val-Datasets und mittelt die IoU.
    Fusion aus validate_seg() (Decode + Aggregation) und dem Direkt-Index-
    Zugriff aus build_seg_val_cache() -- ohne Vorab-Stacking.
    """
    model.eval()
    all_ious = []
    n_total  = len(indices)
    t0       = time.time()
    use_amp  = (device.type == "cuda")

    for k, i in enumerate(indices):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device).float()   # [1,3,256,128,128]  (1d: GPU-Upcast)
        tgt = sample["target"].float()                           # [256,128,128] (CPU); fp32 fuer Decoder
        ego = sample["ego_delta"].unsqueeze(0).to(device).float() if "ego_delta" in sample else None  # Ego-Delta

        # 1) World-Model-Forward (AMP wie im Training)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_latent = model(inp, ego, z_mode=z_mode)  # [1,256,128,128]

        # 2) Decode -- Decoder laeuft bewusst in float32 (wie validate_seg)
        pred_mask, _ = seg_decoder.latent_to_mask(pred_latent[0], device=device)
        real_mask, _ = seg_decoder.latent_to_mask(tgt,            device=device)

        # 3) IoU
        iou = T.compute_iou(pred_mask, real_mask, T.MAP_CLASSES)
        all_ious.append(iou)

        # tee-fester Fortschritt alle 50 Samples
        if (k + 1) % 50 == 0 or (k + 1) == n_total:
            el   = time.time() - t0
            it_s = (k + 1) / el if el > 0 else 0.0
            eta  = (n_total - k - 1) / it_s if it_s > 0 else 0.0
            run  = float(np.mean([d["mIoU"] for d in all_ious]))
            sys.stdout.write(
                f"\r  Decode {100.0*(k+1)/n_total:5.1f}%  "
                f"[{k+1:>5}/{n_total}]  mIoU={run:.4f}  "
                f"{it_s:4.1f} it/s  ETA {eta/60:5.1f}min   "
            )
            sys.stdout.flush()
    sys.stdout.write("\r" + " " * 90 + "\r")
    sys.stdout.flush()

    mean_miou = float(np.mean([d["mIoU"] for d in all_ious]))
    per_class = {}
    for cls in T.MAP_CLASSES:
        vals = [d[cls] for d in all_ious if d.get(cls) is not None]
        per_class[cls] = round(float(np.mean(vals)), 4) if vals else None
    return mean_miou, per_class, len(all_ious)


def main():
    args = parse_args()

    # --- Config + Reproduzierbarkeit (gleiche Helfer wie train_linux) --------
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit(
            "[eval_full_val] seg_decoder_torch nicht importierbar -- ohne "
            "Decoder keine mIoU. (Gleiche Ursache wuerde validate_seg im "
            "Training deaktivieren.)"
        )

    # --- Modell bauen + Checkpoint laden (nur Weights, kein Optimizer) -------
    model_cfg    = cfg["model"]
    model_config = T.ModelConfig(**model_cfg)
    model        = T.BEVWorldModel(model_config).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))

    # --- Seg-Decoder (eingefroren) aus derselben Config-Quelle ---------------
    decode_cfg = cfg.get("decode_validation", {})
    seg_ckpt   = decode_cfg.get("seg_checkpoint")
    if not (seg_ckpt and Path(seg_ckpt).exists()):
        raise SystemExit(
            f"[eval_full_val] seg_checkpoint nicht gefunden: {seg_ckpt}\n"
            f"                (decode_validation.seg_checkpoint in der YAML pruefen.)"
        )
    seg_decoder = T.build_seg_decoder(seg_ckpt, device, verbose=True)

    # --- Val-Dataset (build_dataloaders liefert train/val/test; nur val genutzt)
    print("[eval_full_val] Baue DataLoader...")
    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    n_avail = len(dataset)

    # --- Sample-Auswahl: identische Logik wie build_seg_val_cache() ----------
    #     n_subset=None -> alle (Permutation aller Indizes -> sortiert -> [0..N-1]).
    if args.n_subset is None:
        indices = list(range(n_avail))
        mode = f"FULL-VAL ({n_avail} Samples)"
    else:
        rng     = np.random.default_rng(args.seed)
        n_take  = min(args.n_subset, n_avail)
        indices = sorted(rng.choice(n_avail, size=n_take, replace=False).tolist())
        mode = f"SUBSET (n={n_take}, seed={args.seed}) -- Sanity-Check"

    print(f"\n{'='*64}")
    print(f"  Full-Val-mIoU  |  Phase: {args.phase}  |  {mode}")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"{'='*64}\n")

    t0 = time.time()
    mean_miou, per_class, n_done = evaluate(
        model, seg_decoder, dataset, device, indices, z_mode=args.z_mode)
    dt = time.time() - t0

    # --- Ergebnis ------------------------------------------------------------
    result = {
        "checkpoint":  str(args.checkpoint),
        "config":      str(args.config),
        "phase":       args.phase,
        "n_samples":   n_done,
        "full_val":    args.n_subset is None,
        "z_mode":      args.z_mode,
        "mIoU":        round(mean_miou, 4),
        "per_class":   per_class,
        "eval_time_sec": round(dt, 1),
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(args.output) if args.output else \
        Path(args.checkpoint).parent / "miou_fullval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"{'='*64}")
    print(f"  FERTIG  |  {n_done} Samples in {dt/60:.1f} min")
    print(f"  mIoU (Full-Val): {mean_miou:.4f}")
    print(f"  per-Klasse:      " +
          "  ".join(f"{c}={per_class[c]}" for c in T.MAP_CLASSES))
    print(f"  geschrieben:     {out}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
