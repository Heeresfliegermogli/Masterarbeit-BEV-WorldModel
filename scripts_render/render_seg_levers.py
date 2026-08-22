#!/usr/bin/env python3
# =============================================================================
# render_seg_levers.py — (1) per-Klasse-Decken-Analyse, (2) Gesamt-Hebel-Figur
# =============================================================================
# (1) 19_per_class: Headline-mIoU je Klasse, gruppiert gross/statisch vs
#     duenn/klein -> visualisiert, WO der Restfehler der 0.69-Decke sitzt.
# (2) 19_all_levers: horizontale Balken ALLER getesteten Hebel vs off=0.6946
#     mit Signifikanzband -- die "Money-Figur" des Seg-Strangs.
# Okabe-Ito, lokal rendern. Datenquelle: master_table.csv + Headline-JSON.
# =============================================================================
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, de, fmt
apply_style(17.0)

OFF = 0.6946
THR = 0.014
OUT = Path("visualizations"); OUT.mkdir(exist_ok=True)
BLACK = "#000000"; BLUE = "#0072B2"; GREEN = "#009E73"; VERM = "#D55E00"
ORANGE = "#E69F00"; PURP = "#CC79A7"; GREY = "#999999"

# ---------------------------------------------------------------- Figur 1 ---
pc = json.load(open("predictions/task19/headline_minimal_fullval.json"))["per_class"]
order = [("drivable_area", "groß/statisch"), ("walkway", "groß/statisch"),
         ("carpark_area", "groß/statisch"), ("divider", "dünn/klein"),
         ("ped_crossing", "dünn/klein"), ("stop_line", "dünn/klein")]
cols = {"groß/statisch": BLUE, "dünn/klein": VERM}

fig, ax = plt.subplots(figsize=FIG(7.6, 4.4))
xs = range(len(order))
ax.bar(xs, [pc[c] for c, g in order], color=[cols[g] for c, g in order],
       width=0.62, zorder=3, edgecolor="white")
ax.axhline(OFF, color=BLACK, lw=1.2, ls="--", label=f"Gesamt-mIoU ({OFF})")
for i, (c, g) in enumerate(order):
    ax.text(i, pc[c] + 0.008, f"{fmt(pc[c])}", ha="center", fontsize=S(9))
ax.set_xticks(list(xs))
ax.set_xticklabels([c for c, g in order], fontsize=S(9), rotation=15)
ax.set_ylabel("mIoU je Klasse (Vollvalidierung)")
ax.set_ylim(0.4, 0.95)
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=BLUE, label="groß/statisch"),
                   mpatches.Patch(color=VERM, label="dünn/klein (IoU-randempfindlich)"),
                   plt.Line2D([0], [0], color=BLACK, ls="--", label=f"Gesamt ({OFF})")],
          fontsize=S(8.5), loc="upper right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="0.9", zorder=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"19_per_class.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", OUT / "19_per_class.png")

# ---------------------------------------------------------------- Figur 2 ---
GROUP_COL = {"Baseline": GREY, "Headline": BLACK, "16d Ego": ORANGE,
             "16e Kontext/EMA": GREEN, "16f Hebel": BLUE, "18 Generativ": PURP}
# Anzeige-Namen der Gruppen fuer die Legende (CSV-Schluessel bleiben unveraendert)
GROUP_LBL = {"Headline": "Referenzkonfiguration",
             "16d Ego": "Ego-Konditionierung", "16e Kontext/EMA": "Kontext/EMA",
             "16f Hebel": "Task-Loss/Kapazität/Residual", "18 Generativ": "Generativ"}
# Anzeige-Korrekturen einzelner Varianten-Labels (Nachtrag 14.08.):
# kein "= OFF"-Jargon, kein "F3" (kollidiert mit Forschungsfrage F3)
VAR_LBL = {"Ego FiLM state": "FiLM Zustand",
           "Ego FiLM action": "FiLM Aktion",
           "4 Input-Frames": "4 Eingabeframes",
           "Kapazitaet: 8 Layer": "Kapazität: 8 Schichten",
           "EMA (Shadow-Weights)": "EMA",
           "CVAE beta=0.001 (z=mean)": "CVAE beta=0.001 (z=Mittel)",
           "CVAE beta=0.01 (z=mean)": "CVAE beta=0.01 (z=Mittel)",
           "CVAE beta=0.1 (z=mean)": "CVAE beta=0.1 (z=Mittel)",
           "Persistenz ego-gewarpt": "Persistenz ego-kompensiert",
           "6-Term-Baseline (mse+5 Regler)":
               "6-Term-Baseline (MSE + 5 Terme)",
           "SmoothL1-Minimal (smooth_l1+std) = OFF":
               "SmoothL1-Minimal (smooth_l1+std)",
           "Flow Matching (mean = frozen Backbone)":
               "Flow Matching (deterministische Basis)",
           "Persistenz naiv (F3 kopieren)":
               "Persistenz naiv (letztes Latent kopieren)"}
rows = []
with open("predictions/task19/master_table.csv") as f:
    for r in csv.DictReader(f):
        if r["variante"].startswith("EMA-Lauf raw"):
            continue  # reiner Konsistenz-Check, keine eigene Variante
        rows.append((r["gruppe"], r["variante"], float(r["miou_fullval"])))
rows.sort(key=lambda r: r[2])

apply_style(14.0)          # 19_all_levers wird 14 cm breit eingebunden
fig2, ax2 = plt.subplots(figsize=FIG(9.2, 11.4))
ys = range(len(rows))
ax2.axvspan(OFF - THR, OFF + THR, color="0.88", zorder=0,
            label=f"Streuungsband der Seed-Variabilität ($\\pm${THR})")
ax2.axvline(OFF, color=BLACK, lw=1.2, ls="--", zorder=1)
ax2.barh(list(ys), [m for _, _, m in rows],
         color=[GROUP_COL[g] for g, _, _ in rows], zorder=3,
         height=0.62, edgecolor="white")
for y, (g, n, m) in zip(ys, rows):
    ax2.text(m + 0.003, y, f"{fmt(m, 4)}", va="center", fontsize=S(8))
ax2.set_yticks(list(ys))
ax2.set_yticklabels([VAR_LBL.get(n, de(n)) for _, n, _ in rows], fontsize=S(8.5))
ax2.set_xlabel("mIoU (Vollvalidierung)")
ax2.set_xlim(0.50, 0.735)
handles = [mpatches.Patch(color=c, label=GROUP_LBL.get(g, g)) for g, c in GROUP_COL.items()]
handles.append(mpatches.Patch(color="0.88", label="Seed-Variabilität ±0,014"))
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.grid(axis="x", color="0.9", zorder=0)
fig2.tight_layout(rect=[0, 0.06, 1, 1])
# Legende auf die Mitte des SICHTBAREN Blocks zentrieren: die tight-bbox
# der Achse enthaelt die langen y-Labels links -> so steht die Legende
# links wie rechts gleich weit ueber, statt einseitig auszureissen.
fig2.canvas.draw()
_bb = ax2.get_tightbbox(fig2.canvas.get_renderer()).transformed(
    fig2.transFigure.inverted())
_pos = ax2.get_position()
# Titel auf denselben Block zentrieren (x in Achsenkoordinaten)
_cx = ((_bb.x0 + _bb.x1) / 2 - _pos.x0) / (_pos.x1 - _pos.x0)
ax2.title.set_x(_cx)
fig2.legend(handles=handles, fontsize=S(8.5), loc="upper center",
            bbox_to_anchor=((_bb.x0 + _bb.x1) / 2, _pos.y0 - 0.075),
            bbox_transform=fig2.transFigure, ncol=3, frameon=False,
            columnspacing=1.4, handlelength=1.3, handletextpad=0.6)
for ext in ("png", "pdf"):
    fig2.savefig(OUT / f"19_all_levers.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", OUT / "19_all_levers.png")
