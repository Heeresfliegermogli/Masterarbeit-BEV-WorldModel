#!/usr/bin/env python3
# =============================================================================
# render_pareto.py — Pareto Qualitaet vs. Ressourcenbedarf (Seg)
# =============================================================================
# (Aufruf aus dem Projektroot: python3 scripts_render/render_pareto.py)
# Links: mIoU vs. Inferenzlatenz [ms]; rechts: mIoU vs. Peak-VRAM [MB].
# Markerflaeche ~ Parameterzahl. Minimal-Konfig = gefuellter Sweet-Spot,
# Pareto-Front (Persistenz -> minimal) dezent. Kein Titel (Caption in LaTeX).
# Werte verifiziert (09.08.): minimal/8L/CVAE = Full-Val 5743; Flow = 300-
# Subset (markiert). Quellen: task19/master_table.csv, Flow-Abschlussbericht,
# task26_ressourcen/ressourcen.csv.
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG
apply_style(17.0)

GREEN = "#009E73"; BLUE = "#0072B2"; ORANGE = "#E69F00"; GREY = "#999999"

# (Label, mIoU, Latenz[ms], VRAM[MB], Params[M], Farbe, subset300)
PTS = [
    ("Persistenz",        0.5326,  0.2,   1, 0.0, GREY,   False),
    ("Weltmodell minimal", 0.6946, 4.41, 147, 6.05, GREEN, False),
    ("CVAE-Kopf",         0.6883,  4.58, 149, 6.42, BLUE,   False),
    ("Weltmodell 8 Schichten",    0.6927,  7.10, 165, 9.21, BLUE,   False),
]

def area(params):
    return 60 + params * 55   # linear ~ Parameterzahl (0 -> sichtbarer Punkt)

fig, (axL, axR) = plt.subplots(1, 2, figsize=FIG(12.5, 5.2))

for ax, xi, xlabel, xmax in [(axL, 2, "inkrementelle Vorhersagelatenz [ms]", 9),
                             (axR, 3, "Spitzen-GPU-Speicher [MB]", 200)]:
    # Pareto-Front: Persistenz -> minimal (dezent)
    p0, pm = PTS[0], PTS[1]
    ax.plot([p0[xi], pm[xi]], [p0[1], pm[1]], "-", color=GREEN, lw=1.2,
            alpha=0.35, zorder=1)
    for lab, miou, lat, vram, par, col, sub in PTS:
        x = (lat, vram)[xi - 2]
        sweet = "minimal" in lab   # gefuellter Marker bleibt
        ax.scatter(x, miou, s=area(par), c=col, zorder=4,
                   edgecolors="black" if sweet else "white",
                   linewidths=1.6 if sweet else 1.0,
                   alpha=1.0 if sweet else 0.85,
                   marker="o")
        tag = lab + (" *" if sub else "")
        # gezielte Label-Offsets (die zwei ~4.5ms-Punkte trennen)
        if xi == 2:   # Latenz-Panel: 4.41 vs 4.58 ms -> Labels auseinanderziehen
            off = {"Weltmodell minimal": (-xmax*0.025, 0.030, "right", "bottom"),
                   "CVAE-Kopf":                 (xmax*0.045, -0.035, "left", "top"),
                   "Weltmodell 8 Schichten":            (0.0,        0.022, "center", "bottom"),
                   "Persistenz":                (xmax*0.02,  0.006, "left", "bottom")}
        else:         # VRAM-Panel: 147/149/165 MB draengen sich -> vertikal staffeln
            off = {"Weltmodell minimal": (-xmax*0.02, -0.030, "right", "top"),
                   "CVAE-Kopf":                 (xmax*0.02,   0.018, "left", "bottom"),
                   "Weltmodell 8 Schichten":            (xmax*0.015, -0.050, "left", "top"),
                   "Persistenz":                (xmax*0.02,  0.006, "left", "bottom")}
        ddx, ddy, ha, va = off.get(lab, (xmax*0.02, 0.006, "left", "bottom"))
        _lead = (xi == 2 and lab in ("Weltmodell minimal", "CVAE-Kopf"))
        ax.annotate(tag, (x, miou), xytext=(x + ddx, miou + ddy),
                    fontsize=S(8.3), va=va, ha=ha,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8,
                                    shrinkA=2, shrinkB=8) if _lead else None)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0.51, 0.745)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mIoU (Vollvalidierung, Proxy-Skala)")
    ax.grid(color="0.92", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axL.annotate("Fläche ~ Parameterzahl", (0.97, 0.03), xycoords="axes fraction",
             ha="right", fontsize=S(8), color="0.4")

fig.tight_layout()
out = Path("visualizations")
for ext in ("pdf", "png"):
    fig.savefig(out / f"pareto.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "pareto.pdf")
