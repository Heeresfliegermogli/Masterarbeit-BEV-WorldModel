#!/usr/bin/env python3
# =============================================================================
# synth_loss_sweeps.py — Synthese (lokal, kein Compute)
# =============================================================================
# Liest die 5 Sweep-CSVs (grad/ssim/mean/cos/recon), baut EINE Master-Tabelle
# und erzeugt die Thesis-Plots:
#   1) Dosis-Wirkungs: dmIoU + dstd-Ratio je Regler gegen das variierte Gewicht
#      (Delta vom jeweiligen Nullpunkt), mit 15.2-Signifikanzbaendern.
#   2) Trade-off: std-Ratio vs mIoU (alle Punkte), Baseline + Ideal markiert.
# Farben: Okabe-Ito (farbenblind-sicher per Konstruktion). Eine Achse je Panel.
# =============================================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG
apply_style(17.0)
from pathlib import Path

PRED = Path("predictions/task16b")
OUT  = Path("visualizations"); OUT.mkdir(exist_ok=True)

# 15.2-Rauschboden (Signifikanzschwellen)
THR_MIOU, THR_STD = 0.014, 0.009

# Okabe-Ito (ohne Gelb wg. Weisskontrast) -> 5 Regler
COL = {"lambda_grad": "#E69F00", "lambda_ssim": "#56B4E9",
       "lambda_mean": "#009E73", "lambda_cos": "#D55E00",
       "recon_loss":  "#CC79A7"}
LAB = {"lambda_grad": "grad", "lambda_ssim": "ssim", "lambda_mean": "mean",
       "lambda_cos": "cos", "recon_loss": "recon (mse/smooth_l1)"}

# --- Laden + Master-Tabelle ---------------------------------------------------
frames = []
for reg in ["lambda_grad", "lambda_ssim", "lambda_mean", "lambda_cos", "recon_loss"]:
    f = PRED / f"sweep_{reg}.csv"
    if not f.exists():
        print(f"[warn] fehlt: {f}"); continue
    d = pd.read_csv(f)[["regler", "value", "best_miou", "std_ratio", "best_epoch"]]
    frames.append(d)
master = pd.concat(frames, ignore_index=True)
master.to_csv(PRED / "synthesis_master.csv", index=False)
print("[synth] Master-Tabelle -> predictions/task16b/synthesis_master.csv")
print(master.to_string(index=False))

# --- numerische Regler fuer die Dosis-Wirkungs-Kurven -------------------------
NUM = ["lambda_grad", "lambda_ssim", "lambda_mean", "lambda_cos"]

def num_sub(reg):
    d = master[master.regler == reg].copy()
    d["w"] = d["value"].astype(float)
    d = d.sort_values("w")
    base_m = d[d.w == 0]["best_miou"].iloc[0]
    base_s = d[d.w == 0]["std_ratio"].iloc[0]
    d["dmiou"] = d["best_miou"] - base_m
    d["dstd"]  = d["std_ratio"] - base_s
    return d

# =============================================================================
# FIGUR 1: Dosis-Wirkungs (2 Panels)
# =============================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG(11, 4.4))
for ax, col, thr, ttl in [(a1, "dmiou", THR_MIOU, "$\\Delta$ mIoU vs. Nullpunkt"),
                          (a2, "dstd",  THR_STD,  "$\\Delta$ std-Ratio vs. Nullpunkt")]:
    ax.axhspan(-thr, thr, color="0.85", zorder=0)            # Rauschband
    ax.axhline(0, color="0.5", lw=0.8, zorder=1)
    for reg in NUM:
        d = num_sub(reg)
        ax.plot(d.w, d[col], "-o", color=COL[reg], lw=2, ms=6,
                label=LAB[reg], zorder=3)
    ax.set_xlabel("variiertes Gewicht $\\lambda$")
    ax.set_title(ttl, fontsize=S(11))
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
a1.set_ylabel("$\\Delta$ mIoU")
a2.set_ylabel("$\\Delta$ std-Ratio")
fig.tight_layout(rect=[0, 0.14, 1, 1])
_h, _l = a1.get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=4, frameon=False, fontsize=S(9),
           bbox_to_anchor=(0.5, 0.055))
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"16b9_dose_response.{ext}", dpi=150, bbox_inches="tight")
print(f"[synth] Plot -> {OUT}/16b9_dose_response.png / .pdf")

# =============================================================================
# FIGUR 2: Trade-off std-Ratio vs mIoU (alle Punkte)
# =============================================================================
fig2, ax = plt.subplots(figsize=FIG(7.2, 5.6))
# ERST scatten (setzt die Datengrenzen), DANN Grenzen fixieren + Ideal-Linie.
for reg in ["lambda_grad", "lambda_ssim", "lambda_mean", "lambda_cos", "recon_loss"]:
    d = master[master.regler == reg]
    ax.scatter(d.best_miou, d.std_ratio, s=70, color=COL[reg],
               label=LAB[reg], zorder=3, edgecolor="white", linewidth=0.8)
    for _, r in d.iterrows():
        ax.annotate(str(r["value"]), (r.best_miou, r.std_ratio),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=S(7), color="0.35")
ax.set_xlim(0.668, 0.690)
ax.set_ylim(0.875, 1.005)
ax.axhline(1.0, color="0.6", lw=1.0, ls="--", zorder=1)    # Ideal = keine Abweichung
ax.text(0.6685, 0.999, "ideal (std-Ratio = 1, keine Abweichung)", fontsize=S(8),
        color="0.5", va="top", ha="left")
ax.annotate("SmoothL1 = kleinste Abweichung\n(gleiche mIoU)", (0.6735, 0.9266),
            textcoords="offset points", xytext=(10, -4), fontsize=S(8),
            color="#CC79A7", va="center", ha="left")
ax.set_xlabel("mIoU  (höher = besser)")
ax.set_ylabel("std-Ratio  (näher an 1 = kleinere Abweichung)")
ax.grid(True, color="0.92", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=S(9), loc="lower left", title="Regler")
fig2.tight_layout()
for ext in ["png", "pdf"]:
    fig2.savefig(OUT / f"16b9_tradeoff.{ext}", dpi=150, bbox_inches="tight")
print(f"[synth] Plot -> {OUT}/16b9_tradeoff.png / .pdf")
print("[synth] fertig.")
