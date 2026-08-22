#!/usr/bin/env python3
# =============================================================================
# render_seg_gt_korridor.py — Seg-GT-Korridor + per-Klasse-Zerlegung
# =============================================================================
# Links: mIoU gegen ECHTE nuScenes-Map-GT (Injection, iou@max) — Persistenz ->
# Modelle -> Orakel (frozen BEVFusion-Seg = Decke). Rechts: per-Klasse-Balken
# aller Varianten. Kernaussagen: (1) Rangfolge der Pseudo-GT-Metrik bestaetigt
# (Minimal ~ 6-Term >> Persistenz); (2) Modell haelt 89-98% des Orakels je
# Klasse -> der Selten-Klassen-Einbruch ist ueberwiegend die DECODER-Decke,
# nicht World-Model-Fehler; groesster WM-Verlust bei duennen Strukturen
# (divider 88.8%). Okabe-Ito.
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

BLACK = "#000000"; GREY = "#999999"; BLUE = "#0072B2"; GREEN = "#009E73"

CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
           "carpark_area", "divider"]
CLS_SHORT = ["drivable", "ped_cross", "walkway", "stop_line", "carpark", "divider"]

DATA = {  # (Label, Farbe, mean, per-Klasse in CLASSES-Reihenfolge)
    "Persistenz": (GREY,  0.4647, [0.7627, 0.3421, 0.5105, 0.3102, 0.4757, 0.3868]),
    "Weltmodell minimal": (GREEN, 0.5898, [0.8341, 0.5618, 0.6281, 0.4746, 0.5637, 0.4761]),
    "Weltmodell 6-Term":  (BLUE,  0.5882, [0.8338, 0.5599, 0.6284, 0.4711, 0.5594, 0.4763]),
    "real":       (BLACK, 0.6295, [0.8535, 0.6074, 0.6734, 0.5114, 0.5951, 0.5360]),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG(13.5, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.9]})

# --- links: mean-Korridor ---------------------------------------------------
names = list(DATA.keys())
xs = range(len(names))
ax1.bar(xs, [DATA[n][1] for n in names], color=[DATA[n][0] for n in names],
        width=0.62, zorder=3, edgecolor="white")
for i, n in enumerate(names):
    ax1.text(i, DATA[n][1] + 0.012, f"{fmt(DATA[n][1])}", ha="center", fontsize=S(10))
ax1.set_xticks(list(xs))
ax1.set_xticklabels(names, fontsize=S(8), rotation=16, ha="right")
ax1.set_ylabel("mIoU vs. nuScenes-Annotation (iou@max)")
ax1.set_ylim(0, 0.72)
ax1.set_title("Mittel (6 Klassen)", fontsize=S(11))

# --- rechts: per-Klasse -----------------------------------------------------
w = 0.19
xc = np.arange(len(CLASSES))
for j, n in enumerate(names):
    ax2.bar(xc + (j - 1.5) * w, DATA[n][2], width=w * 0.92, color=DATA[n][0],
            zorder=3, edgecolor="white", label=n)
ax2.set_xticks(xc)
ax2.set_xticklabels(CLS_SHORT, fontsize=S(9))
ax2.set_ylabel("IoU vs. Annotation")
ax2.set_ylim(0, 0.95)
ax2.set_title("je Klasse", fontsize=S(11))


for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)

fig.tight_layout(rect=[0, 0.10, 1, 1])
_h, _l = ax2.get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=4, frameon=False,
           fontsize=S(8.5), bbox_to_anchor=(0.5, -0.02))
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"21_seg_gt_korridor.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "21_seg_gt_korridor.png")
