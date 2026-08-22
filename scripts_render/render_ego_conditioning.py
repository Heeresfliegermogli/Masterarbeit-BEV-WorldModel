#!/usr/bin/env python3
# =============================================================================
# render_ego_conditioning.py — Ego-Conditioning-Ablation (Full-Val-mIoU)
# =============================================================================
# Balkendiagramm: off-Baseline vs. vier Ego-Conditioning-Injektionen
# (gate / FiLM-action / FiLM-state / token). Kernargument: NUR gate erhaelt
# die Baseline (n.s. zu off), alle anderen Injektionen schaden signifikant.
# Lokal rendern (bevwm-Env auf dem Cluster hat kein matplotlib).
# Okabe-Ito-Palette, konsistent mit den 16b/16c/17-Figuren.
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

OFF = 0.6946           # SmoothL1-Minimal-Headline (16b.9)
THR = 0.014            # Signifikanzschwelle mIoU (15.2)
PRED = Path("predictions/task16d")

# Okabe-Ito
BLACK = "#000000"; GREEN = "#009E73"; VERM = "#D55E00"; ORANGE = "#E69F00"

def load(label):
    return json.load(open(PRED / f"ego_{label}_fullval.json"))["mIoU"]

# Reihenfolge nach mIoU absteigend; Mechanismus-Gruppierung ueber Farbe.
bars = [
    ("off\n(Baseline)",      OFF,          BLACK,  "Referenz"),
    ("gate",                 load("gate"), GREEN,  "Gate-Bias"),
    ("action",               load("action"), VERM, "FiLM"),
    ("token",                load("token"), ORANGE, "Ego-Token"),
    ("state",                load("state"), VERM,  "FiLM"),
]

fig, ax = plt.subplots(figsize=FIG(8.2, 5.0))
xs = range(len(bars))
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]

# Signifikanzband um off
ax.axhspan(OFF - THR, OFF + THR, color="0.85", zorder=0,
           label=f"Streuungsband der Seed-Variabilität ($\\pm${THR})")
ax.axhline(OFF, color=BLACK, lw=1.2, ls="--", zorder=1)

b = ax.bar(xs, vals, color=cols, width=0.62, zorder=3,
           edgecolor="white", linewidth=1.5)

for i, (lab, v, c, mech) in enumerate(bars):
    d = v - OFF
    txt = f"{fmt(v, 4)}" if i == 0 else f"{fmt(v, 4)}\n({fmt(d, 4, sign=True)})"
    ax.text(i, v + 0.006, txt, ha="center", va="bottom",
            fontsize=S(9.5), color=BLACK)

ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in bars], fontsize=S(10))
ax.set_ylabel("mIoU (Vollvalidierung, 5743 Fenster)", fontsize=S(11))
ax.set_ylim(0.55, 0.72)

# Mechanismus-Annotation unter den Balken
for i, (lab, v, c, mech) in enumerate(bars):
    ax.text(i, 0.558, mech, ha="center", va="bottom", fontsize=S(8),
            color="0.35", style="italic")

ax.legend(loc="upper right", fontsize=S(9), framealpha=0.9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="0.9", zorder=0)
fig.tight_layout()

out = Path("visualizations")
out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"16d_ego_conditioning.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "16d_ego_conditioning.png", "+ .pdf")
