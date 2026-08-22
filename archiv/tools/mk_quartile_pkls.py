#!/usr/bin/env python3
# =============================================================================
# mk_quartile_pkls.py — Task 22: Dynamik-Quartil-Splits der Val-Infos
# =============================================================================
# Teilt die 5743 Fenster-Targets (Frames mit >=3 Vorgaengern, Dynamik-Score
# vorhanden) nach Quartilen des Dynamik-Scores (RMS Latent-Frame-Differenz,
# predictions/task22/dynamics_val.json) und schreibt je Quartil eine
# ann_file-kompatible pkl -> /home/vima/det_latents_out/quartile_pkls/ (im
# Docker als /output/quartile_pkls sichtbar). Injection-Eval pro Quartil via
# --cfg-options data.test.ann_file=/output/quartile_pkls/dyn_qK.pkl.
# Kontextlose Frames (276) sind ausgeschlossen (kein Score, Real-Fill).
# =============================================================================
import json
import pickle
from pathlib import Path

import numpy as np

SRC_PKL = Path("/home/vima/Desktop/Masterarbeit/Nuscenes/complete/nuscenes_infos_val.pkl")
SCORES  = Path("predictions/task22/dynamics_val.json")
OUT     = Path("/home/vima/det_latents_out/quartile_pkls")

with open(SRC_PKL, "rb") as f:
    data = pickle.load(f)
scores = json.loads(SCORES.read_text())

# Fenster-Targets = Frames mit Score UND >=3 Vorgaengern: der Score existiert
# fuer jedes Paar (t-1,t), also ab Frame 1; Fenster brauchen Frame >=3. Wir
# rekonstruieren die Szenenstruktur wie ueberall (Timestamp + 2s-Luecke).
infos = sorted(data["infos"], key=lambda x: x["timestamp"])
scenes, cur = [], [infos[0]]
for a, b in zip(infos[:-1], infos[1:]):
    if b["timestamp"] - a["timestamp"] > 2e6:
        scenes.append(cur); cur = [b]
    else:
        cur.append(b)
scenes.append(cur)

targets = []
for sc in scenes:
    for i, fr in enumerate(sc):
        if i >= 3:
            assert fr["token"] in scores, fr["token"]
            targets.append((fr, scores[fr["token"]]))
print(f"{len(targets)} Fenster-Targets (erwartet 5743)")

vals = np.array([s for _, s in targets])
edges = np.percentile(vals, [25, 50, 75])
print(f"Quartilgrenzen: {edges.round(4).tolist()}")

OUT.mkdir(parents=True, exist_ok=True)
for q in range(4):
    lo = -np.inf if q == 0 else edges[q - 1]
    hi = np.inf if q == 3 else edges[q]
    sel = [fr for fr, s in targets if lo < s <= hi] if q > 0 else \
          [fr for fr, s in targets if s <= hi]
    out = dict(data)
    out["infos"] = sel
    p = OUT / f"dyn_q{q + 1}.pkl"
    with open(p, "wb") as f:
        pickle.dump(out, f)
    print(f"q{q + 1}: {len(sel)} Frames -> {p}")
