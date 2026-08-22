#!/usr/bin/env python3
# =============================================================================
# render_head_comparison.py — Kopf-Versionen im Vergleich (Det, Injection)
# =============================================================================
# Gruppierte Balken: frozen | v1 | v2 — je auf
# WM-Vorhersagen und realen Latents (Full-Val 6019). Kernaussagen:
# (1) beide adaptierten Koepfe heben Vorhersagen um ~+0.09 mAP (v1~v2,
#     Differenz < Rauschboden 0.0044); (2) NUR v2 (Real-Mix) haelt auch auf
#     realen Latents fast Orakel-Niveau -> Ein-Kopf-Betriebspunkt. Okabe-Ito.
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)
import numpy as np

GREY = "#999999"; BLUE = "#0072B2"; GREEN = "#009E73"; BLACK = "#000000"

HEADS = ["ursprünglicher Kopf", "adaptiert v1", "adaptiert v2 (mit realen Latents)"]
COLORS = [GREY, BLUE, GREEN]
DATA = {  # (mAP, NDS) je [pred, real]
    "ursprünglicher Kopf":  [(0.3438, 0.4761), (0.6858, 0.7146)],
    "adaptiert v1": [(0.4329, 0.5328), (0.6603, 0.6918)],
    "adaptiert v2 (mit realen Latents)": [(0.4292, 0.5333), (0.6710, 0.7040)],
}

fig, axes = plt.subplots(1, 2, figsize=FIG(11.5, 4.6))
for ax, mi, mlabel in [(axes[0], 0, "mAP"), (axes[1], 1, "NDS")]:
    xg = np.arange(2)   # [Vorhersagen, reale Latents]
    w = 0.26
    for j, h in enumerate(HEADS):
        vals = [DATA[h][0][mi], DATA[h][1][mi]]
        ax.bar(xg + (j - 1) * w, vals, width=w * 0.92, color=COLORS[j],
               zorder=3, edgecolor="white", label=h)
        for x, v in zip(xg + (j - 1) * w, vals):
            # hohe Balken: Wert IM Balken (sonst kollidiert er mit der
            # gestrichelten Orakel-Linie), sonst darueber
            if v > 0.6:
                ax.text(x, v - 0.045, f"{fmt(v)}", ha="center", color="white",
                        fontweight="bold", fontsize=S(8.2))
            else:
                ax.text(x, v + 0.014, f"{fmt(v)}", ha="center", fontsize=S(8.2))
    ax.axhline(DATA["ursprünglicher Kopf"][1][mi], color=BLACK, lw=1.0, ls="--",
               zorder=2)
    ax.text(-0.42, DATA["ursprünglicher Kopf"][1][mi] + 0.012, "Referenz: reales Latent",
            fontsize=S(8), ha="left", color=BLACK)
    ax.set_xticks(xg)
    ax.set_xticklabels(["auf Weltmodell-Vorhersagen", "auf realen Latents"], fontsize=S(9.5))
    ax.set_ylabel(mlabel)
    ax.set_ylim(0, 0.92)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)

fig.tight_layout(rect=[0, 0.07, 1, 1])
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=3, frameon=False,
           fontsize=S(8.5), bbox_to_anchor=(0.5, -0.01))
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"25_head_comparison.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "25_head_comparison.png")
