#!/usr/bin/env python3
# =============================================================================
# filmstrip_render.py — Rollout-Filmstrip (RENDER, lokal)
# =============================================================================
# Umbau 11.08.: statt EINER 4x7-Figur zwei getrennte Dateien mit je 5 Spalten
# (letzter Kontextframe F3 + Rollout k=1..4), 2 Zeilen (Real oben, Praediktion
# unten). Grund: bei 17 cm Druckbreite waren 7 Spalten x 4 Zeilen nicht mehr
# erkennbar; F1/F2 tragen nichts zur Aussage bei. mIoU je Rollout-Schritt steht
# jetzt als Panel-Titel. Szenenauswahl unveraendert (Gate-alpha-Extreme).
# Ausgaben: 17_filmstrip_statisch.{pdf,png}, 17_filmstrip_dynamisch.{pdf,png}
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)
from pathlib import Path

NPZ = Path("predictions/task17/filmstrip.npz")
OUT = Path("visualizations"); OUT.mkdir(exist_ok=True)
d = np.load(NPZ)
n_input = int(d["n_input"]); K = int(d["k"])


def mask_rgb(mask):
    H, W = mask.shape[1], mask.shape[2]
    cmap = plt.get_cmap("tab10")
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    for c in range(mask.shape[0]):
        img[mask[c]] = (np.array(cmap(c % 10)[:3]) * 255).astype(np.uint8)
    return img


SEQS = [("static", "statisch", "17_filmstrip_statisch"),
        ("dynamic", "dynamisch", "17_filmstrip_dynamisch")]

for name, human, fname in SEQS:
    # BEIDE Filmstrips auf 17 cm: bei 5 Panels waeren 10 cm nur ~2 cm/Panel
    # -> Bildinhalt unlesbar (Empfehlung an die Thesis: >= 14 cm einbinden)
    apply_style(17.0)
    real = d[f"{name}_real"]              # [L, n_cls,200,200]
    pred = d[f"{name}_pred"]              # [K, n_cls,200,200]
    mi   = d[f"{name}_mious"]             # [K]
    ma   = float(d[f"{name}_mean_alpha"])

    ncols = 1 + K                          # F3 (letzter Kontext) + k=1..K
    fig, ax = plt.subplots(2, ncols, figsize=FIG(2.9 * ncols, 7.4))

    for c in range(ncols):
        f_real = n_input - 1 + c           # F3, F4, ... im echten Verlauf
        ax[0, c].imshow(mask_rgb(real[f_real]))
        ax[1, c].imshow(mask_rgb(real[f_real] if c == 0 else pred[c - 1]))
        head = "Kontext t" if c == 0 else f"k={c}"
        ax[0, c].set_title(head, fontsize=S(11))
        if c > 0:
            ax[1, c].set_title(f"mIoU {fmt(mi[c-1])}", fontsize=S(10.5),
                               color="#0072B2", pad=6)
        else:
            ax[1, c].set_title("Startpunkt", fontsize=S(10.5), color="0.45", pad=6)
        for r in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    ax[0, 0].set_ylabel("Real", fontsize=S(11), rotation=0, ha="right",
                        va="center", labelpad=18)
    ax[1, 0].set_ylabel(f"Modell\n(mean α={fmt(ma, 2)})", fontsize=S(11), rotation=0,
                        ha="right", va="center", labelpad=18)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"geschrieben: {OUT/fname}.pdf  (mean-alpha {ma:.2f}, "
          f"mIoU {[round(float(x),3) for x in mi]})")
