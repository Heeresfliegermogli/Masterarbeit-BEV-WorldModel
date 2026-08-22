#!/usr/bin/env python3
# =============================================================================
# render_error_decomposition.py — Thesis-Figur: Drei-Komponenten-Fehlerzerlegung
# =============================================================================
# Wasserfall auf der GT-Metrik-Skala: perfekte Metrik ->
# Real-Latent-Referenz -> World-Model. Beschriftung NEUTRAL (reine
# Metrikabstaende, keine kausalen Fehleranteile — Sprachregelung 14.08.).
# Werte werden aus den Ergebnis-JSONs GELADEN und die Abstaende im
# Skript berechnet (Nachtrag 14.08.: vorher stand das Seed-Mittel 0.342 am
# WM-Balken; Thesis-Referenzwert ist Seed 42 = 0.3438 -> Label 0.344,
# Abstand 0.686-0.344 = 0.342). Okabe-Ito.
# =============================================================================
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)

BLACK = "#000000"; ORANGE = "#E69F00"; GREEN = "#009E73"; GREY = "#BBBBBB"

# Werte aus den Task-Artefakten (Referenz-Seed 42, wie im Thesis-Text)
seg = json.load(open("predictions/task21/gt_eval.json"))["gt_miou"]
det = json.load(open("predictions/task20/seed_noise.json"))["varianten"]

PANELS = [
    # (Titel, Metrik-Label, perfekt, referenz, modell, modell_label)
    ("Segmentierung (mIoU vs. Annotation)", "mIoU", 1.0,
     seg["orakel"]["mean"], seg["minimal"]["mean"], "Weltmodell\nminimal"),
    ("Detektion (mAP vs. Annotation)", "mAP", 1.0,
     det["orakel"]["mAP"], det["baseline"]["seed42"]["mAP"], "Weltmodell\n6-Term"),
]

fig, axes = plt.subplots(1, 2, figsize=FIG(12.5, 5.6))
for ax, (title, mlabel, perf, orak, wm, wmlab) in zip(axes, PANELS):
    perc = perf - orak          # Wahrnehmungs-Decke
    fore = orak - wm            # Forecasting-Rest
    tot = perc + fore
    # Wasserfall: 3 Saeulen
    ax.bar(0, perf, color=GREY, width=0.6, zorder=3, edgecolor="white")
    ax.bar(1, orak, color=BLACK, width=0.6, zorder=3, edgecolor="white")
    ax.bar(1, perc, bottom=orak, color=ORANGE, width=0.6, zorder=3,
           alpha=0.45, edgecolor="white", hatch="//")
    ax.bar(2, wm, color=GREEN, width=0.6, zorder=3, edgecolor="white")
    ax.bar(2, fore, bottom=wm, color=ORANGE, width=0.6, zorder=3,
           alpha=0.85, edgecolor="white")
    ax.text(0, perf + 0.015, f"{fmt(perf)}", ha="center", fontsize=S(10))
    ax.text(1, orak + 0.018, f"{fmt(orak)}", ha="center", fontsize=S(9),
            fontweight="bold")
    ax.text(2, wm - 0.055, f"{fmt(wm)}", ha="center", fontsize=S(9),
            color="white", fontweight="bold")
    ax.text(1, orak + perc / 2, f"{fmt(perc)}",
            ha="center", va="center", fontsize=S(9))
    # Forecasting-Label: bei duennem Segment (Seg) rechts daneben mit Pfeil
    if fore / tot < 0.2:
        ax.annotate(f"{fmt(fore)}",
                    xy=(2.33, wm + fore / 2), xytext=(2.5, wm + fore / 2),
                    ha="left", va="center", fontsize=S(8.5),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.9))
        ax.set_xlim(-0.65, 3.9)
    else:
        ax.text(2, wm + fore / 2, f"{fmt(fore)}",
                ha="center", va="center", fontsize=S(9))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["perfekte\nMetrik", "Referenz:\nreales Latent", wmlab],
                       fontsize=S(9))
    ax.set_ylabel(mlabel)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=S(11))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)


fig.tight_layout()
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"21_error_decomposition.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "21_error_decomposition.png")
