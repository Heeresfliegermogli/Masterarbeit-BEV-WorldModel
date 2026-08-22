#!/usr/bin/env python3
# =============================================================================
# render_thesis_synthesis.py — zwei Synthese-Figuren fuer die Thesis
# =============================================================================
# (Aufruf aus dem Projektroot: python3 scripts_render/render_thesis_synthesis.py)
# 1) korridor_overview: Seg+Det nebeneinander als horizontale Skala
#    Persistenz -> World-Model -> Orakel -> perfekte Metrik (Kapitel-Auftakt).
# 2) rueckholquote: Anteil des Forecasting-Gaps, den die Kopf-Adaptation
#    zurueckholt (Seg 28.7%, Det 26.1%) — der Synthese-Befund als Bild.
# Datenquellen: TASK20/21/23/24/25-Berichte (GT-Injection, Full-Val 6019).
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)

GREY = "#999999"; GREEN = "#009E73"; BLACK = "#000000"; ORANGE = "#E69F00"
BLUE = "#0072B2"
OUT = Path("visualizations")

# --- 1) Korridor-Doppelpanel -------------------------------------------------
ROWS = [
    ("Segmentierung\n(mIoU vs. Annotation)",
     [("Persistenz", 0.4647, GREY), ("Weltmodell", 0.5898, BLUE),
      ("Referenz:\nreales Latent", 0.6295, BLACK), ("perfekte\nMetrik", 1.0, "#DDDDDD")]),
    ("Detektion\n(mAP vs. Annotation)",
     # World-Model = Referenzwert Seed 42 (0.3438, wie Thesis-Text/20_det_korridor),
     # nicht das Seed-Mittel 0.3422 (Nachtrag 14.08.)
     [("Persistenz", 0.1696, GREY), ("Weltmodell", 0.3438, BLUE),
      ("Referenz:\nreales Latent", 0.6858, BLACK), ("perfekte\nMetrik", 1.0, "#DDDDDD")]),
]
fig, axes = plt.subplots(2, 1, figsize=FIG(11, 5.0), sharex=True)
for ax, (label, marks) in zip(axes, ROWS):
    ax.hlines(0, 0, 1, color="0.85", lw=6, zorder=1)
    # Korridor-Segmente einfaerben
    ax.hlines(0, marks[0][1], marks[1][1], color=GREEN, lw=6, zorder=2)
    ax.hlines(0, marks[1][1], marks[2][1], color=ORANGE, lw=6, zorder=2)
    for i, (name, v, col) in enumerate(marks):
        ax.plot(v, 0, "o", ms=13, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5)
        # Labels oberhalb der Bahn (unterhalb kollidieren sie mit den
        # Segment-Beschriftungen); liegen zwei Marker zu dicht beieinander,
        # rutscht das zweite eine Etage hoeher statt zu ueberlappen.
        _dx = min([abs(v - _pv) for _, _pv, _ in marks[:i]], default=9.9)
        ytxt = 1.15 if _dx < 0.08 else 0.55
        ax.annotate(f"{name}\n{fmt(v)}" if v < 1 else name, (v, 0),
                    xytext=(v, ytxt), ha="center", fontsize=S(8.7),
                    va="bottom" if ytxt > 0 else "top")
    ax.set_ylabel(label, fontsize=S(10), rotation=0, ha="right", va="center")
    ax.set_ylim(-1.25, 1.05)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[1].set_xlim(0, 1.04)
axes[1].set_xlabel("Metrik-Skala (echte nuScenes-Annotation, Injektionsprotokoll)")
axes[1].tick_params(axis="x", labelsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"korridor_overview.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("geschrieben:", OUT / "korridor_overview.png")

# --- 2) Rueckholquote --------------------------------------------------------
# Seg: Gap 0.6295-0.5898=0.0397, zurueckgeholt 0.6012-0.5898=0.0114 -> 28.7%
# Det: Gap 0.6858-0.3438=0.3420, zurueckgeholt 0.4329-0.3438=0.0891 -> 26.1%
#      (v2-Kopf: 0.0854 -> 25.0%, praktisch identisch)
vals = [0.0114 / 0.0397, 0.0891 / 0.3420]
fig, ax = plt.subplots(figsize=FIG(7.8, 5.2))
bars = ax.bar([0, 1], [v * 100 for v in vals], color=[GREEN, BLUE],
              width=0.52, zorder=3, edgecolor="white")
ax.text(0, vals[0] * 100 + 1.2, f"{vals[0]:.1%}\n(+0.011 mIoU\nvon 0.040 Abstand)",
        ha="center", fontsize=S(9.5))
ax.text(1, vals[1] * 100 + 1.2, f"{vals[1]:.1%}\n(+0.089 mAP\nvon 0.342 Abstand)",
        ha="center", fontsize=S(9.5))
ax.axhspan(25, 33.3, color="0.93", zorder=1)
ax.set_xlim(-0.62, 1.92)
ax.text(1.88, 29.1, "~1/4 bis 1/3", fontsize=S(8.5), color="0.45",
        ha="right", va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Segmentierung\n(Kopf-Adaptation)", "Detektion\n(Kopf-Adaptation v1)"],
                   fontsize=S(10))
ax.set_ylabel("zurückgeholter Anteil des Abstands\nReferenz (reales Latent) – Weltmodell  [%]")
ax.set_ylim(0, 40)
ax.grid(axis="y", color="0.9", zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rueckholquote.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", OUT / "rueckholquote.png")
