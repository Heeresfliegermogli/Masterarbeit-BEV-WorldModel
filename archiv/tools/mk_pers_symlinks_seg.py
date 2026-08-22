#!/usr/bin/env python3
# =============================================================================
# mk_pers_symlinks_seg.py — Task 21.1c: Seg-Persistenz-Baseline als Symlinks
# =============================================================================
# Baut das Injection-Verzeichnis der Persistenz-Baseline OHNE Kopien:
#   Vorhersage fuer Frame i = reales Latent von Frame i-1   (i >= 3 je Szene)
#   Frames 0..2 je Szene    = reales Latent i               (Luecken-Konvention
#   wie 20.1: kontextlose Frames bei ALLEN Varianten real gefuellt)
# Szenen-Splitting identisch zu Code/bev_dataset.py: Timestamp-Sortierung +
# 2.0-s-Luecken-Schwelle (scene_gap_sec).
# WICHTIG: Die Symlink-ZIELE sind CONTAINER-Pfade (/seg_val_real/...) — auf dem
# Host erscheinen die Links "kaputt", im Docker (Mount seg_gate) sind sie gueltig.
# =============================================================================
import pickle
import sys
from pathlib import Path

PKL      = Path("/home/vima/bevfusion_decoder/nuscenes_infos_val.pkl")
REAL_DIR = Path("/home/vima/Desktop/Masterarbeit/Nuscenes/latents/seg/val")
OUT      = Path("/home/vima/det_latents_out/latents_seg_pers")
CONTAINER_REAL = "/seg_val_real"   # Mount im Container seg_gate
GAP_SEC  = 2.0

with open(PKL, "rb") as f:
    infos = sorted(pickle.load(f)["infos"], key=lambda x: x["timestamp"])
infos = [i for i in infos if (REAL_DIR / f"bev_latent_{i['token']}.npy").exists()]
print(f"{len(infos)} Frames mit Latent (erwartet 6019)")

threshold_us = GAP_SEC * 1e6
scenes, current = [], [infos[0]]
for prev, curr in zip(infos[:-1], infos[1:]):
    if curr["timestamp"] - prev["timestamp"] > threshold_us:
        scenes.append(current); current = [curr]
    else:
        current.append(curr)
scenes.append(current)
print(f"{len(scenes)} Szenen")

OUT.mkdir(parents=True, exist_ok=True)
n_pers = n_real = 0
for scene in scenes:
    for i, fr in enumerate(scene):
        src_tok = scene[i - 1]["token"] if i >= 3 else fr["token"]
        link = OUT / f"bev_latent_{fr['token']}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f"{CONTAINER_REAL}/bev_latent_{src_tok}.npy")
        if i >= 3: n_pers += 1
        else:      n_real += 1
print(f"{n_pers} Persistenz-Links + {n_real} Real-Fill = {n_pers + n_real}")
if n_pers + n_real != len(infos):
    sys.exit("FEHLER: Link-Anzahl != Frame-Anzahl")
