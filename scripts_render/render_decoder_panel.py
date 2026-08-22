#!/usr/bin/env python3
# =============================================================================
# render_decoder_panel.py — Panel GT | frozen | adaptierter Kopf
# =============================================================================
# Rendert die npz aus compute_23_panel.py (Docker) zu einem 3x3-Panel.
# Klassenfarben (Okabe-Ito-nah), Overlay mit Prioritaet duenner Klassen oben.
# =============================================================================
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG
apply_style(17.0)
import matplotlib.patches as mpatches
import numpy as np

SRC = Path("/home/vima/det_latents_out/panel23")
CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
           "carpark_area", "divider"]
# Zeichenreihenfolge: flaechig unten, duenn oben
ORDER = [0, 2, 4, 1, 5, 3]
COLORS = {0: (0.55, 0.55, 0.55), 2: (0.35, 0.70, 0.90), 4: (0.80, 0.60, 0.70),
          1: (0.00, 0.62, 0.45), 5: (0.94, 0.89, 0.26), 3: (0.84, 0.37, 0.00)}
LABELS = {0: "drivable", 2: "walkway", 4: "carpark", 1: "ped_cross",
          5: "divider", 3: "stop_line"}

def to_rgb(masks):
    img = np.ones((200, 200, 3)) * 0.12
    for k in ORDER:
        img[masks[k] > 0] = COLORS[k]
    return img

files = sorted(glob.glob(str(SRC / "panel_*.npz")))[:3]
fig, axes = plt.subplots(len(files), 3, figsize=FIG(10.5, 3.6 * len(files)))
cols = [("gt", "nuScenes-Annotation"), ("frozen", "ursprünglicher Kopf auf Vorhersage"),
        ("adapted", "adaptierter Kopf auf Vorhersage")]
for r, f in enumerate(files):
    d = np.load(f)
    for c, (key, title) in enumerate(cols):
        ax = axes[r, c] if len(files) > 1 else axes[c]
        ax.imshow(to_rgb(d[key]), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=S(11))
        if c == 0:
            ax.set_ylabel(f"Szene {chr(65 + r)}", fontsize=S(10))

handles = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in ORDER]
fig.legend(handles=handles, ncol=6, loc="lower center", fontsize=S(9),
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"23_decoder_panel.{ext}", dpi=200, bbox_inches="tight")
print("geschrieben:", out / "23_decoder_panel.png")
