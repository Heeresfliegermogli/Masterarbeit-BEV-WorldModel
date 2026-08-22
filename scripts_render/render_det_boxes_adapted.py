#!/usr/bin/env python3
# =============================================================================
# render_det_boxes_adapted.py — Box-Panel GT | Orakel | frozen | adaptiert
# =============================================================================
# Setzt die LiDAR-BEV-Renders aus tools/visualize.py (Docker, Injection-Hook)
# zu einem 3x4-Panel zusammen — der visuelle Beweis des Korridor-Befunds:
# Persistenz laesst bewegte Boxen "kleben", das World-Model schiebt sie nach.
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
apply_style(17.0)

BASE = Path("/home/vima/det_latents_out")
TOKENS = [
    ("f3c9f3e85f384381bfced8c2a863376f", "Szene A"),
    ("755ef1d006e94b2a87523000781a7a99", "Szene B"),
    ("77c3d98fab3e4d3ea65747caaa74d605", "Szene C"),
]
COLS = [("viz_gt", "Ground Truth"), ("viz_oracle", "ursprünglicher Kopf\n+ reales Latent"),
        ("viz_model", "ursprünglicher Kopf\n+ Vorhersage"), ("viz_adapted", "adaptierter Kopf\n+ Vorhersage")]

def find(variant, token):
    hits = glob.glob(str(BASE / variant / "lidar" / f"*-{token}.png"))
    return hits[0] if hits else None

# Hochformat (Umbau 22.08. analog 5.17): 4 Zeilen Varianten x 3 Spalten Szenen
fig, axes = plt.subplots(len(COLS), len(TOKENS), figsize=FIG(12, 16.4),
                         gridspec_kw=dict(wspace=0.04, hspace=0.06))
fig.patch.set_facecolor("white")
for r, (variant, rlabel) in enumerate(COLS):
    for c, (tok, clabel) in enumerate(TOKENS):
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
            ax.set_ylabel(rlabel, fontsize=S(10))
# Kein Suptitle: die Aussage steht in der LaTeX-Caption; die
# Spaltenkoepfe sind die Legende des Panels (kollidierte sonst).
fig.tight_layout()
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"24_det_boxes_panel.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("geschrieben:", out / "24_det_boxes_panel.png")
