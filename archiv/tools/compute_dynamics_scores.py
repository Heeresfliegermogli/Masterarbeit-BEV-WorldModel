#!/usr/bin/env python3
# =============================================================================
# compute_dynamics_scores.py — Task 22.0b: Dynamik-Score je Fenster
# =============================================================================
# Score = ||x_t - x_{t-1}||_2 / sqrt(N) (RMS der Frame-Differenz im Latentraum)
# fuer jedes konsekutive Frame-Paar; das Fenster mit Target t erbt den Score
# seines Paars (t-1, t) — "wieviel Bewegung muss vorhergesagt werden".
# Enthaelt Ego- UND Objektdynamik (bewusst; ego-warp-kompensierte Variante
# als Option dokumentiert, Task 22.3). Ausgabe: JSON token->score je Split
# -> predictions/task22/dynamics_<split>.json (klein, ~1-2M).
#
# USAGE (Cluster, CPU reicht — Lesen ist der Flaschenhals):
#   python -u compute_dynamics_scores.py --config config_seg_dump_eval.yaml
# =============================================================================
import argparse
import json
import time
from pathlib import Path

import numpy as np

import train_linux as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", default="cell")
    args = ap.parse_args()

    cfg = T.load_config(args.config, args.phase)
    train_loader, val_loader, _ = T.build_dataloaders(cfg)
    out_dir = Path("predictions/task22")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, loader in (("val", val_loader), ("train", train_loader)):
        ds = loader.dataset
        scores, t0, n_pairs = {}, time.time(), 0
        for scene in ds._all_scenes:
            prev = None
            for fr in scene:
                cur = np.asarray(ds._load_latent(fr), dtype=np.float32)
                if prev is not None:
                    d = cur - prev
                    scores[fr["token"]] = float(np.sqrt(np.mean(d * d)))
                    n_pairs += 1
                    if n_pairs % 1000 == 0:
                        r = n_pairs / (time.time() - t0)
                        print(f"  [{split}] {n_pairs} Paare, {r:5.1f} P/s", flush=True)
                prev = cur
        path = out_dir / f"dynamics_{split}.json"
        path.write_text(json.dumps(scores))
        arr = np.array(list(scores.values()))
        print(f"[done {split}] {len(scores)} Paare -> {path} | "
              f"mean {arr.mean():.4f} p25 {np.percentile(arr,25):.4f} "
              f"median {np.median(arr):.4f} p75 {np.percentile(arr,75):.4f} "
              f"max {arr.max():.4f}", flush=True)


if __name__ == "__main__":
    main()
