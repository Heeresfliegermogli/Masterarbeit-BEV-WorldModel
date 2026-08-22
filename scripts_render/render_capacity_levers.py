#!/usr/bin/env python3
# =============================================================================
# render_capacity_levers.py — letzte mIoU-Hebel (Task-Loss, Kapazitaet, Residual)
# =============================================================================
# Balken: off vs. Task-Loss-Dosisreihe (0.1/0.5/1.0), cap_l8, residual.
# Kernaussage: Task-Loss schadet monoton+dosisabhängig; Tiefe und Residual
# landen im Signifikanzband -> 0.69-Decke steht auf vier Beinen. Okabe-Ito.
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

OFF = 0.6946
THR = 0.014
PRED = Path("predictions/task16f")

BLACK = "#000000"; VERM = "#D55E00"; BLUE = "#0072B2"; GREEN = "#009E73"

def load(L):
    return json.load(open(PRED / f"{L}_fullval.json"))["mIoU"]

bars = [
    ("off\n(Baseline)",        OFF,             BLACK),
    ("hartes\nResidual",       load("residual"), GREEN),
    ("8 Layer\n(Kapazität)",   load("cap_l8"),  BLUE),
    ("$\\lambda_{task}$ 0.1",  load("tl_01"),   VERM),
    ("$\\lambda_{task}$ 0.5",  load("tl_05"),   VERM),
    ("$\\lambda_{task}$ 1.0",  load("tl_10"),   VERM),
]

fig, ax = plt.subplots(figsize=FIG(8.6, 5.0))
xs = range(len(bars))
ax.axhspan(OFF - THR, OFF + THR, color="0.85", zorder=0,
           label=f"Streuungsband der Seed-Variabilität ($\\pm${THR})")
ax.axhline(OFF, color=BLACK, lw=1.2, ls="--", zorder=1)
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars],
       width=0.62, zorder=3, edgecolor="white", linewidth=1.5)
for i, (lab, v, c) in enumerate(bars):
    txt = f"{fmt(v, 4)}" if i == 0 else f"{fmt(v, 4)}\n({fmt(v - OFF, 4, sign=True)})"
    ax.text(i, v + 0.003, txt, ha="center", va="bottom", fontsize=S(9), color=BLACK)
ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in bars], fontsize=S(9.5))
ax.set_ylabel("mIoU (Vollvalidierung, 5743 Fenster)", fontsize=S(11))
ax.set_ylim(0.65, 0.715)
ax.legend(loc="lower left", fontsize=S(9), framealpha=0.9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="0.9", zorder=0)
fig.tight_layout()
out = Path("visualizations"); out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"16f_miou_lever2.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "16f_miou_lever2.png", "+ .pdf")
