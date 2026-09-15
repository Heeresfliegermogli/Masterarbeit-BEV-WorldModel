#!/usr/bin/env python3
# =============================================================================
# render_beispiel_klassen.py — Beispiel-Render-Skript (Abb. 5.8 der Thesis)
# =============================================================================
# mIoU der Minimal-Konfiguration je Klasse auf der Vollvalidierung,
# gruppiert nach gross/statisch vs. duenn/klein. Zeigt den typischen
# Aufbau aller Thesis-Figuren: thesis_style.py (Fontkalibrierung,
# deutsche Dezimalkommas) + kleine Ergebnis-JSONs als Datenquelle.
# Aufruf aus dem Repo-Root: python3 scripts_render/render_beispiel_klassen.py
# =============================================================================
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)

OFF = 0.6946      # Gesamt-mIoU der Minimal-Konfiguration (Vollvalidierung)
OUT = Path("scripts_render")
BLACK = "#000000"; BLUE = "#0072B2"; VERM = "#D55E00"

pc = json.load(open("scripts_render/beispiel_seg_headline_vollvalidierung.json"))["per_class"]
order = [("drivable_area", "groß/statisch"), ("walkway", "groß/statisch"),
         ("carpark_area", "groß/statisch"), ("divider", "dünn/klein"),
         ("ped_crossing", "dünn/klein"), ("stop_line", "dünn/klein")]
cols = {"groß/statisch": BLUE, "dünn/klein": VERM}

fig, ax = plt.subplots(figsize=FIG(7.6, 4.4))
xs = range(len(order))
ax.bar(xs, [pc[c] for c, g in order], color=[cols[g] for c, g in order],
       width=0.62, zorder=3, edgecolor="white")
ax.axhline(OFF, color=BLACK, lw=1.2, ls="--")
for i, (c, g) in enumerate(order):
    ax.text(i, pc[c] + 0.008, f"{fmt(pc[c])}", ha="center", fontsize=S(9))
ax.set_xticks(list(xs))
ax.set_xticklabels([c for c, g in order], fontsize=S(9), rotation=15)
ax.set_ylabel("mIoU je Klasse (Vollvalidierung)")
ax.set_ylim(0.4, 0.95)
ax.legend(handles=[mpatches.Patch(color=BLUE, label="groß/statisch"),
                   mpatches.Patch(color=VERM, label="dünn/klein (IoU-randempfindlich)"),
                   plt.Line2D([0], [0], color=BLACK, ls="--", label=f"Gesamt ({fmt(OFF, 4)})")],
          fontsize=S(8.5), loc="upper right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="0.9", zorder=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"beispiel_miou_je_klasse.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", OUT / "beispiel_miou_je_klasse.png")
