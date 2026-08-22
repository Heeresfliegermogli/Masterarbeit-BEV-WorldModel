#!/usr/bin/env python3
# =============================================================================
# render_flow_calibration.py — Skalen-Dosis-Wirkung der Kalibrierung
# =============================================================================
# x = Residual-Skala s (1.0 -> 0.5). Links: mIoU (Einzel-Sample, Best-of-8,
# mean-Linie) — zeigt den Crossover Best-of-8 > mean ab s<=0.8. Rechts:
# std-Ratio (->1.0) und Masken-Diversitaet (Trade-off). Okabe-Ito.
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

P = Path("predictions/task18")
SCALES = ["1.0", "0.8", "0.6", "0.5"]
d = {s: json.load(open(P / f"eval_flow30_sc{s}.json")) for s in SCALES}
xs = [float(s) for s in SCALES]
MEAN = d["1.0"]["miou_mean"]

BLACK = "#000000"; BLUE = "#0072B2"; GREEN = "#009E73"; VERM = "#D55E00"; GREY = "#999999"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG(11.5, 4.3))

ax1.axhline(MEAN, color=BLACK, ls="--", lw=1.3, label=f"deterministisch (mean, {MEAN})")
ax1.plot(xs, [d[s]["miou_best_of_k"] for s in SCALES], "o-", color=GREEN, lw=2,
         label="Best-of-8 (Coverage)")
ax1.plot(xs, [d[s]["miou_sample"] for s in SCALES], "s-", color=BLUE, lw=2,
         label="Einzel-Sample")
ax1.set_xlabel("Residual-Skala s"); ax1.set_ylabel("mIoU (Teilstichprobe mit 300 Fenstern)")
ax1.invert_xaxis()
ax1.grid(color="0.9")

ax2.axhline(1.0, color=GREY, ls=":", lw=1.3, label="Ideal std-Ratio = 1.0")
ax2.plot(xs, [d[s]["std_ratio_sample"] for s in SCALES], "o-", color=VERM, lw=2,
         label="std-Ratio Samples")
ax2b = ax2.twinx()
ax2b.plot(xs, [d[s]["diversity_mask"] for s in SCALES], "^--", color=BLUE, lw=1.6,
          label="Masken-Diversität (rechts)")
ax2.set_xlabel("Residual-Skala s"); ax2.set_ylabel("std-Ratio")
ax2b.set_ylabel("Masken-Diversität")
ax2.invert_xaxis()

h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()

for ax in (ax1, ax2):
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
    ax.grid(color="0.9")

fig.tight_layout(rect=[0, 0.16, 1, 1])
_h1, _l1 = ax1.get_legend_handles_labels()
fig.legend(_h1, _l1, loc="lower left", bbox_to_anchor=(0.06, -0.01),
           ncol=1, frameon=False, fontsize=S(8.5))
fig.legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(0.98, -0.01),
           ncol=1, frameon=False, fontsize=S(8.5))
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"18_flow_calibration.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "18_flow_calibration.png")
