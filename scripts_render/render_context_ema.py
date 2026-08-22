#!/usr/bin/env python3
# =============================================================================
# render_context_ema.py — mIoU-Hebel 4-Frames + EMA (Full-Val)
# =============================================================================
# Balkendiagramm: off-Baseline vs. 4-Frames (nf4) vs. EMA (shadow). Kernaussage:
# BEIDE billigen Hebel liegen im Signifikanzband um off -> flach, kein mIoU-Gewinn
# -> deterministischer Plateau bei ~0.69 bestaetigt. Okabe-Ito, lokal rendern.
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
PRED = Path("predictions/task16e")

BLACK = "#000000"; BLUE = "#0072B2"; GREEN = "#009E73"; GREY = "#999999"

def load(p):
    return json.load(open(PRED / p))["mIoU"]

# Reihenfolge: Referenz, dann die zwei getesteten Hebel, dann Konsistenz-Check.
bars = [
    ("off\n(Baseline)",       OFF,                          BLACK, "Referenz"),
    ("nf4\n(4 Frames)",       load("nf4_fullval.json"),     BLUE,  "Kontext-Hebel"),
    ("ema\n(EMA shadow)",     load("ema_shadow_fullval.json"), GREEN, "Weight-Averaging"),
    ("ema\n(raw)",            load("ema_raw_fullval.json"),  GREY,  "Konsistenz-Check"),
]

fig, ax = plt.subplots(figsize=FIG(7.8, 5.0))
xs = range(len(bars))
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]

ax.axhspan(OFF - THR, OFF + THR, color="0.85", zorder=0,
           label=f"Streuungsband der Seed-Variabilität ($\\pm${THR})")
ax.axhline(OFF, color=BLACK, lw=1.2, ls="--", zorder=1)

ax.bar(xs, vals, color=cols, width=0.6, zorder=3, edgecolor="white", linewidth=1.5)

for i, (lab, v, c, mech) in enumerate(bars):
    d = v - OFF
    txt = f"{fmt(v, 4)}" if i == 0 else f"{fmt(v, 4)}\n({fmt(d, 4, sign=True)})"
    ax.text(i, v + 0.004, txt, ha="center", va="bottom", fontsize=S(9.5), color=BLACK)

ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in bars], fontsize=S(10))
ax.set_ylabel("mIoU (Vollvalidierung)", fontsize=S(11))
ax.set_ylim(0.60, 0.71)
ax.legend(loc="lower right", fontsize=S(9), framealpha=0.9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="0.9", zorder=0)
fig.tight_layout()

out = Path("visualizations"); out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"16e_miou_lever.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "16e_miou_lever.png", "+ .pdf")
