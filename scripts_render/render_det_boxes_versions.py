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
    ("4a1972f8731b4cdea40fc69a38a735b1", "Szene A"),
    ("df25b3ecaf994001ab5d5b92b9de89fd", "Szene B"),
    ("5bd85334fdf94fb99c7eaa45d5feba0d", "Szene C"),
    ("a2fada921a7d4141877f4a51328a21af", "Szene D"),
]
COLS = [("viz_gt", "Ground Truth"), ("viz_model", "ursprünglicher Kopf\n+ Vorhersage"),
        ("viz_adapted", "adaptierter Kopf v1\n+ Vorhersage"), ("viz_v2", "adaptierter Kopf v2\n+ Vorhersage")]

def find(variant, token):
    hits = glob.glob(str(BASE / variant / "lidar" / f"*-{token}.png"))
    return hits[0] if hits else None

fig, axes = plt.subplots(len(TOKENS), len(COLS), figsize=FIG(15, 15),
                         gridspec_kw=dict(wspace=0.04, hspace=0.06))
fig.patch.set_facecolor("white")
for r, (tok, rlabel) in enumerate(TOKENS):
    for c, (variant, clabel) in enumerate(COLS):
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
    fig.savefig(out / f"25_det_boxes_versions.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("geschrieben:", out / "25_det_boxes_versions.png")
