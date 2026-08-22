#!/usr/bin/env python3
# =============================================================================
# compute_23_panel.py — Task 23.3: Masken fuer das Panel GT|frozen|adaptiert
# =============================================================================
# Laeuft IM DOCKER (seg_gate). Waehlt 3 Val-Fenster mit hohem stop_line-Anteil
# (aus verschiedenen Szenen), dekodiert die WM-VORHERSAGE einmal mit dem
# frozen und einmal mit dem adaptierten Kopf, speichert GT+beide Masken als
# npz -> /output/panel23/ (Render lokal, Compute/Render getrennt wie Task 17).
# =============================================================================
import pickle
from pathlib import Path

import numpy as np
import torch

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

CFG = "configs/nuscenes/seg/fusion-bev256d2-lss.yaml"
CKPTS = {"frozen": "pretrained/bevfusion-seg.pth",
         "adapted": "/output/adapted/seg_adapted_full_best.pth"}
PRED = Path("/output/dump_seg_minimal")
GT = np.load("/output/gt_masks_seg/gt_masks_pipe_val.npz")
OUT = Path("/output/panel23"); OUT.mkdir(exist_ok=True)

# --- 3 Tokens: hoher stop_line-Anteil, verschiedene Szenen -------------------
with open("data/nuscenes/nuscenes_infos_val.pkl", "rb") as f:
    infos = sorted(pickle.load(f)["infos"], key=lambda x: x["timestamp"])
scenes, cur = [], [infos[0]]
for a, b in zip(infos[:-1], infos[1:]):
    (scenes.append(cur), cur := [b]) if b["timestamp"] - a["timestamp"] > 2e6 \
        else cur.append(b)
scenes.append(cur)
cand = []
for si, sc in enumerate(scenes):
    for i, fr in enumerate(sc):
        if i >= 3 and fr["token"] in GT.files:
            m = np.unpackbits(GT[fr["token"]]).reshape(6, 200, 200)
            cand.append((m[3].mean(), si, fr["token"]))
cand.sort(reverse=True)
picked, used = [], set()
for share, si, tok in cand:
    if si not in used:
        picked.append(tok); used.add(si)
    if len(picked) == 3:
        break
print("Tokens:", picked)

configs.load(CFG, recursive=True)
cfg = Config(recursive_eval(configs), filename=CFG)
device = torch.device("cuda")
models = {}
for name, ck in CKPTS.items():
    m = build_model(cfg.model)
    load_checkpoint(m, ck, map_location="cpu")
    m = m.to(device).eval()
    models[name] = m

with torch.no_grad():
    for tok in picked:
        x = np.load(PRED / f"bev_latent_{tok}.npy").astype(np.float32)
        x = torch.from_numpy(x if x.ndim == 4 else x[None]).to(device)
        out = {"gt": np.unpackbits(GT[tok]).reshape(6, 200, 200)}
        for name, m in models.items():
            feat = m.decoder["backbone"](x)
            feat = m.decoder["neck"](feat)
            probs = m.heads["map"](feat)          # eval-Branch: sigmoid
            out[name] = (probs[0].cpu().numpy() > 0.5).astype(np.uint8)
        np.savez_compressed(OUT / f"panel_{tok}.npz", **out)
        print(f"{tok}: gt stop_line {out['gt'][3].mean():.3f} | "
              f"frozen {out['frozen'][3].mean():.3f} | "
              f"adapted {out['adapted'][3].mean():.3f}")
print("FERTIG")
