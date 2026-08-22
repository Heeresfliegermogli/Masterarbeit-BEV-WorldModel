#!/usr/bin/env python3
# =============================================================================
# render_legacy_figs.py — Neuaufbau der 4 Figuren ohne erhaltenes Quell-Skript
# =============================================================================
# 16c_rollout, 16d_ego_persistence, 17_miou_over_epochs, 17_stdgap_beforeafter
# entstanden urspruenglich aus Inline-Code; hier aus den Original-Daten
# (predictions/task16c/*.json, predictions/task17/miou_curves.json aus den
# Trainings-Logs von Job 123424) reproduziert — Titel OHNE Task-Referenzen,
# Dateinamen unveraendert. Okabe-Ito.
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

BLUE = "#0072B2"; ORANGE = "#D55E00"; GREY = "#999999"; GREEN = "#009E73"
OUT = Path("visualizations")

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("geschrieben:", OUT / f"{name}.png")

# --- 1) 16c_rollout ---------------------------------------------------------
rb = json.load(open("predictions/task16c/rollout_baseline.json"))["curve"]
rm = json.load(open("predictions/task16c/rollout_smoothl1_minimal.json"))["curve"]
ks = sorted(int(k) for k in rb)
fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG(13, 5))
a1.plot(ks, [rb[str(k)]["mIoU"] for k in ks], "o-", color=BLUE, lw=2.5, ms=8,
        label="Baseline (6 Terme)")
a1.plot(ks, [rm[str(k)]["mIoU"] for k in ks], "o-", color=ORANGE, lw=2.5, ms=8,
        label="SmoothL1-Minimal (2 Terme)")
a1.plot(ks, [rb[str(k)]["mIoU_pers"] for k in ks], "s--", color=GREY, lw=2,
        label="Persistenz (letztes Latent wiederholen)")
a1.set_xlabel("Rollout-Schritt k  (~0,5 s/Schritt)"); a1.set_ylabel("mIoU")
a1.set_title("Rollout-mIoU vs. Horizont")
a2.axhline(1.0, ls="--", color="0.6")
a2.text(1.05, 1.004, "ideal (std-Ratio = 1)", fontsize=S(8), color="0.55",
        va="bottom")
a2.plot(ks, [rb[str(k)]["std_ratio"] for k in ks], "o-", color=BLUE, lw=2.5,
        ms=8, label="Baseline (6 Terme)")
a2.plot(ks, [rm[str(k)]["std_ratio"] for k in ks], "o-", color=ORANGE, lw=2.5,
        ms=8, label="SmoothL1-Minimal (2 Terme)")
a2.set_xlabel("Rollout-Schritt k"); a2.set_ylabel("std-Ratio")
a2.set_title("std-Ratio über den Rollout")
for a in (a1, a2):
    a.set_xticks(ks); a.grid(color="0.92")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(rect=[0, 0.10, 1, 1])
_h, _l = a1.get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=3, frameon=False,
           fontsize=S(9), bbox_to_anchor=(0.5, -0.02))
save(fig, "16c_rollout")

# --- 2) 16d_ego_persistence -------------------------------------------------
ep = json.load(open("predictions/task16c/ego_pers.json"))["curve"]
fig, ax = plt.subplots(figsize=FIG(7.5, 5.2))
ax.plot(ks, [rm[str(k)]["mIoU"] for k in ks], "o-", color=ORANGE, lw=2.5, ms=8,
        label="Weltmodell (Smooth-L1 minimal)")
ax.plot(ks, [ep[str(k)]["pers_ego"] for k in ks], "s-", color=GREEN, lw=2.5,
        ms=8, label="Persistenz ego-kompensiert")
ax.plot(ks, [ep[str(k)]["pers_naiv"] for k in ks], "s--", color=GREY, lw=2,
        label="Persistenz naiv")
ax.set_xlabel("Rollout-Schritt k"); ax.set_ylabel("mIoU"); ax.set_xticks(ks)
ax.legend(fontsize=S(9)); ax.grid(color="0.92")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, "16d_ego_persistence")

# --- 3) 17_miou_over_epochs -------------------------------------------------
cur = json.load(open("predictions/task17/miou_curves.json"))
fig, ax = plt.subplots(figsize=FIG(9, 5.5))
for key, col, lab in [("baseline", BLUE, "Baseline (6 Terme)"),
                      ("smoothl1_minimal", ORANGE, "SmoothL1-Minimal (2 Terme)")]:
    pts = cur[key]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=2,
            ms=6, label=f"{lab} (final {fmt(pts[-1][1], 4)})")
ax.axhline(0.6823, ls="--", color="0.6")
ax.text(1, 0.6828, "bisherige Bestmarke 0.6823", fontsize=S(8.5), color="0.5")
ax.set_xlabel("Epoche"); ax.set_ylabel("mIoU (300-Subset-Decode, alle 3 Ep.)")
ax.legend(fontsize=S(9), loc="lower right"); ax.grid(color="0.92")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, "17_miou_over_epochs")

apply_style(10.0)          # 17_stdgap_beforeafter wird 10 cm breit eingebunden
# --- 4) 17_stdgap_beforeafter -----------------------------------------------
fig, ax = plt.subplots(figsize=FIG(6.5, 5.4))
ax.bar([0, 1], [0.909, 0.986], color=[BLUE, ORANGE], width=0.72, zorder=3)
ax.axhline(1.0, ls="--", color="0.6"); ax.text(1.32, 1.002, "ideal", fontsize=S(9), color="0.6")
ax.text(0, 0.912, "0.909\n(Abweichung 9,1 %)", ha="center", fontsize=S(9))
ax.text(1, 0.978, "0.986\n(Abweichung 1,4 %)", ha="center", va="top",
        color="white", fontweight="bold", fontsize=S(9))
ax.set_xticks([0, 1])
ax.set_xticklabels(["Baseline\n(MSE, 6 Terme)", "SmoothL1-Minimal\n(2 Terme)"])
ax.set_ylabel("std-Ratio  (1 = keine Abweichung)")
ax.set_ylim(0.85, 1.02)
ax.grid(axis="y", color="0.92", zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, "17_stdgap_beforeafter")
