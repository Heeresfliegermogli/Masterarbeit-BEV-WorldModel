#!/usr/bin/env python3
# =============================================================================
# render_det_korridor.py — Det-Korridor Persistenz -> Modelle -> Orakel
# =============================================================================
# Balken fuer mAP und NDS (Full-Val 6019, Injection-Protokoll). Kernaussagen:
# (1) beide World-Model-Konfigs VERDOPPELN die Persistenz-mAP; (2) SEIT 20.3
# (Seed-43-Replikat) ist der Abstand 6-Term > minimal BELEGT: Delta 0.018 mAP
# bei einer Seed-Streuung von nur 0.0044 (= 4x Rauschboden, Vorzeichen in
# BEIDEN Seeds gleich) -> "from six to two" traegt auf Det NICHT voll;
# (3) Real-Latent-Referenz = Decke des frozen TransFusion-Kopfes. Okabe-Ito.
# Nachtrag 14.08.: Balken/Label zeigen den REFERENZWERT (Seed 42, wie im
# Thesis-Text: 6-Term 0.3438 -> 0.344, minimal 0.3264 -> 0.326), nicht mehr
# das Seed-Mittel; die weissen Punkte zeigen weiterhin die Einzel-Seeds.
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)

BLACK = "#000000"; GREY = "#999999"; BLUE = "#0072B2"; GREEN = "#009E73"

# (Label, mAP-Werte, NDS-Werte, Farbe) — mehrere Werte = Seeds 42/43
rows = [
    ("Persistenz",        [0.1696],         [0.3908],         GREY),
    ("Weltmodell\nminimal", [0.3264, 0.3220], [0.4662, 0.4622], GREEN),
    ("Weltmodell\n6 Terme",  [0.3438, 0.3407], [0.4761, 0.4741], BLUE),
    ("Referenz:\nreales Latent", [0.6858],   [0.7146],         BLACK),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG(11.5, 4.6))
for ax, idx, name in [(ax1, 1, "mAP"), (ax2, 2, "NDS")]:
    xs = range(len(rows))
    refs = [r[idx][0] for r in rows]           # Referenz = Seed 42
    ax.bar(xs, refs, color=[r[3] for r in rows], width=0.62, zorder=3,
           edgecolor="white")
    for i, r in enumerate(rows):
        vals = r[idx]
        ax.text(i, max(vals) + 0.018, f"{fmt(refs[i])}", ha="center", fontsize=S(10))
        if len(vals) > 1:                        # Einzel-Seeds als Punkte
            ax.plot([i] * len(vals), vals, "o", ms=4.5, color="white",
                    markeredgecolor="0.2", markeredgewidth=0.8, zorder=5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=S(8))
    ax.set_ylabel(name)
    ax.set_ylim(0, 0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)

# Delta-Angabe als reiner Wert (Analyse gehoert in Text/Caption)
yb = 0.46
ax1.plot([1, 1, 2, 2], [0.375, yb, yb, 0.375], color="0.35", lw=1.0)
ax1.text(1.5, yb + 0.015, "+0,018 mAP", ha="center", fontsize=S(9), color="0.2")

fig.tight_layout()
out = Path("visualizations")
for ext in ("png", "pdf"):
    fig.savefig(out / f"20_det_korridor.{ext}", dpi=150, bbox_inches="tight")
print("geschrieben:", out / "20_det_korridor.png")
