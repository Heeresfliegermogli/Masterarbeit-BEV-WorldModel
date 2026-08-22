#!/usr/bin/env python3
# =============================================================================
# render_det_boxes_peds.py — Box-Panel Fussgaenger-Szenen (transponiert)
# =============================================================================
# Setzt die LiDAR-BEV-Renders aus tools/visualize.py (Docker, Injection-Hook)
# zusammen. Umbau 22.08. (Prof-Feedback "Panels zu klein"): TRANSPONIERT zu
# 4 Zeilen (GT | urspruenglicher Kopf + real | urspruenglicher Kopf +
# Vorhersage | adaptierter Kopf + Vorhersage) x 2 Spalten (Szenen A/B).
# Eigene hochformatige Float-Seite, Einbaubreite 12 cm, Hoehe <= 23 cm.
# =============================================================================
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, invert_lidar
apply_style(14.4)          # Einbaubreite 12 cm; 14.4 = 17*12/14.2 kompensiert
                           # den tight-Crop (Raster fuellt die 17-cm-Breite nicht)

BASE = Path("/home/vima/det_latents_out")
SCENES = [
    ("4a1972f8731b4cdea40fc69a38a735b1", "Szene A (Fußgänger)"),
    ("9a1174901ea1401a9972a44616214c9b", "Szene B (Fußgänger)"),
]
ROWS = [("viz_gt", "Ground Truth"), ("viz_oracle", "ursprünglicher Kopf\n+ reales Latent"),
        ("viz_model", "ursprünglicher Kopf\n+ Vorhersage"), ("viz_adapted", "adaptierter Kopf\n+ Vorhersage")]

def find(variant, token):
    hits = glob.glob(str(BASE / variant / "lidar" / f"*-{token}.png"))
    return hits[0] if hits else None

# 12 cm breit, ~22.5 cm hoch: 2 quadratische Panels/Zeile (~5.3 cm Kante)
# + Zeilenlabels links + Spaltentitel oben, 4 Zeilen.
fig, axes = plt.subplots(len(ROWS), len(SCENES), figsize=FIG(12, 22.5),
                         gridspec_kw=dict(wspace=0.04, hspace=0.06))
fig.patch.set_facecolor("white")
for r, (variant, rlabel) in enumerate(ROWS):
    for c, (tok, clabel) in enumerate(SCENES):
        ax = axes[r, c]
        p = find(variant, tok)
        if p is None:
            ax.text(0.5, 0.5, "fehlt", ha="center"); ax.axis("off"); continue
        img = invert_lidar(mpimg.imread(p))
        # zentraler Crop (ego-nah, wo die Dynamik sitzt)
        h, w = img.shape[:2]
        m = int(h * 0.12)
        ax.imshow(img[m:h-m, m:w-m])
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(clabel, fontsize=S(11), pad=4)
        if c == 0:
            ax.set_ylabel(rlabel, fontsize=S(10), rotation=90, labelpad=6)
# Kein Suptitle: die Aussage steht in der LaTeX-Caption; die
# Spaltenkoepfe sind die Legende des Panels (kollidierte sonst).
fig.tight_layout()
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"24_det_boxes_peds.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("geschrieben:", out / "24_det_boxes_peds.png")
