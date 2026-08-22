#!/usr/bin/env python3
# =============================================================================
# render_vae_panel.py — Mittelwert-vs-Samples-Masken-Panel (lokal)
# =============================================================================
# Liest die npz aus eval_vae.py (--panel_npz) und rendert das Kern-Bild des
# Kapitels: Zeilen = Szenen, Spalten = [Real | Mittelwert (z=mean) | Sample 1-3].
# Farbkodierung wie 17_masks_overview: eine Farbe je Klasse, Overlay.
# Compute (Cluster) / Render (lokal) getrennt — bevwm-Env hat kein matplotlib.
#
# USAGE: python3 render_vae_panel.py predictions/task18/panel_0.01.npz
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG
apply_style(17.0)

# Okabe-Ito je Klasse (Reihenfolge = MAP_CLASSES aus der npz)
CLS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]

def mask_to_rgb(mask):
    """[n_cls,200,200] bool -> RGB-Overlay (spätere Klassen überdecken)."""
    img = np.ones(mask.shape[1:] + (3,), dtype=np.float32)  # weiss
    for c in range(mask.shape[0]):
        col = np.array([int(CLS_COLORS[c][i:i+2], 16) / 255 for i in (1, 3, 5)])
        img[mask[c]] = col
    return img

def main():
    npz_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "predictions/task18/panel_0.01.npz")
    d = np.load(npz_path, allow_pickle=True)
    real, mean, samples = d["real"], d["mean"], d["samples"]
    P = real.shape[0]
    # Spalte 2 = "mean"-Array des npz: beim FLOW-Kopf ist das die
    # deterministische Basisvorhersage (frozen Backbone), beim CVAE z=mu.
    _is_flow = "flow" in npz_path.stem
    _mean_lbl = "deterministische Basis" if _is_flow else "Mittelwert ($z=\\mu$)"
    cols = ["Real (Target)", _mean_lbl, "Sample 1", "Sample 2", "Sample 3"]

    fig, axes = plt.subplots(P, 5, figsize=FIG(13.5, 2.8 * P))
    if P == 1:
        axes = axes[None, :]
    for r in range(P):
        imgs = [real[r], mean[r]] + [samples[r, s] for s in range(3)]
        for c, (ax, m) in enumerate(zip(axes[r], imgs)):
            ax.imshow(mask_to_rgb(m), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=S(10))
            if c == 0:
                ax.set_ylabel(f"Szene {r+1}", fontsize=S(10))
    # Klassen-Legende
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=CLS_COLORS[i],
                          label=str(cl), markersize=9)
               for i, cl in enumerate(d["classes"])]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=S(8.5),
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout()

    tag = npz_path.stem.replace("panel_", "")
    out = Path("visualizations")
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"18_vae_panel_{tag}.{ext}", dpi=200, bbox_inches="tight")
    print("geschrieben:", out / f"18_vae_panel_{tag}.png", "+ .pdf")

if __name__ == "__main__":
    main()
