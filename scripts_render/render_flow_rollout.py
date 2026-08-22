#!/usr/bin/env python3
# =============================================================================
# render_flow_rollout.py — Rollout det vs. Flow-Sample (k=1..4)
# =============================================================================
# Zwei Panels: (links) mIoU(k) det vs. Sample vs. Persistenz; (rechts)
# std-Ratio(k) — zeigt die zwei Kernbefunde: kein Sample-Ueberholen UND
# det bleibt varianz-kalibriert (~1.0, Erbe des lambda_std-Terms), waehrend
# der Flow-Kalibrierungsfehler kompoundiert. Okabe-Ito, lokal rendern.
# =============================================================================
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG
apply_style(17.0)

BLACK = "#000000"; BLUE = "#0072B2"; VERM = "#D55E00"; GREY = "#999999"
PRED = Path("predictions/task18")

det = json.load(open(PRED / "rollout_det.json"))["curve"]
smp = json.load(open(PRED / "rollout_sample.json"))["curve"]
ks = [1, 2, 3, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG(11.5, 4.4))

# --- links: mIoU(k) ---
ax1.plot(ks, [det[str(k)]["mIoU"] for k in ks], "o-", color=BLACK, lw=2,
         label="deterministisch (Mittelwert)")
ax1.plot(ks, [smp[str(k)]["mIoU"] for k in ks], "s-", color=BLUE, lw=2,
         label="Flow-Sample (1 Trajektorie)")
ax1.plot(ks, [det[str(k)]["mIoU_pers"] for k in ks], "^--", color=GREY, lw=1.5,
         label="Persistenz (letztes Latent kopieren)")
ax1.set_xlabel("Rollout-Schritt k"); ax1.set_ylabel("mIoU")
ax1.set_xticks(ks); ax1.set_title("mIoU über den Horizont", fontsize=S(11))
ax1.grid(color="0.9")

# --- rechts: std-Ratio(k) ---
ax2.axhline(1.0, color=GREY, lw=1.2, ls=":", label="Ideal (= reale Varianz)")
ax2.plot(ks, [det[str(k)]["std_ratio"] for k in ks], "o-", color=BLACK, lw=2,
         label="deterministisch")
ax2.plot(ks, [smp[str(k)]["std_ratio"] for k in ks], "s-", color=VERM, lw=2,
         label="Flow-Sample")
ax2.set_xlabel("Rollout-Schritt k"); ax2.set_ylabel("std-Ratio (pred/real)")
ax2.set_xticks(ks); ax2.set_title("Varianz-Kalibrierung über den Horizont", fontsize=S(11))
ax2.grid(color="0.9")

for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.tight_layout(rect=[0, 0.15, 1, 1])
_h1, _l1 = ax1.get_legend_handles_labels()
_h2, _l2 = ax2.get_legend_handles_labels()
fig.legend(_h1, _l1, loc="lower left", bbox_to_anchor=(0.06, -0.01),
           frameon=False, fontsize=S(9))
fig.legend(_h2, _l2, loc="lower right", bbox_to_anchor=(0.98, -0.01),
           frameon=False, fontsize=S(9))
out = Path("visualizations"); out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"18_flow_rollout.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "18_flow_rollout.png", "+ .pdf")
