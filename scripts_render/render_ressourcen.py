#!/usr/bin/env python3
# =============================================================================
# render_ressourcen.py — Inferenzzeit-Balkenplot
# =============================================================================
# (Aufruf aus dem Projektroot: python3 scripts_render/render_ressourcen.py)
# Balken: mittlere Vorhersagezeit je Variante [ms], mit 0.5-s-Zyklus als
# Referenzlinie. Quelle: predictions/task26_ressourcen/ressourcen.json.
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

BLUE = "#0072B2"; GREEN = "#009E73"; ORANGE = "#E69F00"; GREY = "#999999"
BLACK = "#000000"
# Quelle seit 22.08.: konsolidierte ressourcen.csv (das Original-JSON wurde
# bei den Task-27-Wiederholungen zu ressourcen_rep{1..3}.json; die CSV
# tragt dieselben Werte inkl. bevfusion_encoder-Zeile).
import csv
with open("predictions/task26_ressourcen/ressourcen.csv") as f:
    d = {r["variante"]: r for r in csv.DictReader(f)}

BARS = [
    ("Persistenz", 0.0, GREY),
    ("Seg-WM\nminimal", float(d["seg_wm_minimal"]["t_forward_ms_mean"]), GREEN),
    ("CVAE-Kopf", float(d["cvae_head"]["t_forward_ms_mean"]), GREEN),
    ("Flow-Kopf\n(det. Forward)", float(d["flow_head"]["t_forward_ms_mean"]), GREEN),
    ("Flow-Kopf\n(+1 Euler)", float(d["flow_head"]["t_flow_total_ms"]), ORANGE),
    ("Seg-WM\n8 Layer", float(d["seg_wm_8layer"]["t_forward_ms_mean"]), BLUE),
    ("Det-WM\n6-Term", float(d["det_wm_6term"]["t_forward_ms_mean"]), BLUE),
    ("BEVFusion-\nEncoder", float(d["bevfusion_encoder"]["t_forward_ms_mean"]), BLACK),
]
fig, ax = plt.subplots(figsize=FIG(10.5, 5))
xs = range(len(BARS))
ax.bar(xs, [b[1] for b in BARS], color=[b[2] for b in BARS], width=0.66,
       zorder=3, edgecolor="white")
for i, (_, v, _c) in enumerate(BARS):
    ax.text(i, v + 3, f"{fmt(v, 1)}" if v > 0 else "~0", ha="center", fontsize=S(9))
ax.axhline(500, ls="--", color="0.35", lw=1.2, zorder=2)
ax.text(len(BARS) - 0.5, 508, "nuScenes-Zyklus 500 ms (2 Hz)", ha="right",
        fontsize=S(9), color="0.35")
ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in BARS], fontsize=S(8.5))
ax.set_ylabel("mittlere Vorhersagezeit [ms]  (Batch 1, fp16, TITAN RTX)")
ax.set_ylim(0, 560)
ax.grid(axis="y", color="0.9", zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"ressourcen_inferenzzeit.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "ressourcen_inferenzzeit.png")
