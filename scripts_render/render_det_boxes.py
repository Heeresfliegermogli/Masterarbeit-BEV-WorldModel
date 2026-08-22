#!/usr/bin/env python3
# =============================================================================
# render_det_boxes.py — Box-Panel GT | Orakel | World-Model | Persistenz
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
    ("4094ae4656fb4b8fb1906192b24a34cb", "Szene A"),
    ("3bf56ebb22b741339967a95a9fbe2081", "Szene B"),
    ("adce54dcf6404d37bd170ffc4c2a7836", "Szene C"),
]
COLS = [("viz_gt", "Ground Truth"), ("viz_oracle", "real\n(reales Latent)"),
        ("viz_model", "Weltmodell\n(Vorhersage)"), ("viz_pers", "Persistenz\n(kopiert)")]

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
    fig.savefig(out / f"20_det_boxes_panel.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("geschrieben:", out / "20_det_boxes_panel.png")
