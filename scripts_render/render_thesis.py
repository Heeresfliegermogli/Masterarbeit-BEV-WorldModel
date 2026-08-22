#!/usr/bin/env python3
# render_thesis.py — Masken-Figuren (RENDER-Teil, lokal): thesis_masks.npz -> Figuren.
# Trennung Compute (Cluster, GPU-Decode) / Render (lokal, matplotlib), weil das
# bevwm-Env auf dem Cluster kein matplotlib hat.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, "scripts_render")
from thesis_style import apply_style, S, FIG, fmt
apply_style(17.0)
from pathlib import Path

NPZ = Path("predictions/task17/thesis_masks.npz")
OUT = Path("visualizations"); OUT.mkdir(exist_ok=True)
d = np.load(NPZ)
idx, mious, preds, reals, alphas = d["idx"], d["mious"], d["preds"], d["reals"], d["alphas"]
n, n_cls = preds.shape[0], preds.shape[1]


def mask_rgb(mask):
    H, W = mask.shape[1], mask.shape[2]
    cmap = plt.get_cmap("tab10")
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    for c in range(mask.shape[0]):
        img[mask[c]] = (np.array(cmap(c % 10)[:3]) * 255).astype(np.uint8)
    return img


def agree_rgb(p, r):
    H, W = p.shape[1], p.shape[2]
    out = np.full((H, W, 3), 245, dtype=np.uint8)
    active = np.any(r, axis=0) | np.any(p, axis=0)
    ag = np.all(p == r, axis=0)
    out[ag & active] = (90, 190, 90); out[~ag] = (215, 60, 60)
    return out


# --- Masken-Overview (n Zeilen x 3) ---
fig, ax = plt.subplots(n, 3, figsize=FIG(9, 3.0 * n))
if n == 1:
    ax = ax[None, :]
for r in range(n):
    for c, (img, ttl) in enumerate([(mask_rgb(preds[r]), "Prädiktion"),
                                    (mask_rgb(reals[r]), "Real"),
                                    (agree_rgb(preds[r], reals[r]), "Uebereinstimmung")]):
        ax[r, c].imshow(img); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(ttl, fontsize=S(11))
    ax[r, 0].set_ylabel(f"Bsp {int(idx[r])}\nmIoU={fmt(mious[r])}", fontsize=S(9),
                        rotation=0, ha="right", va="center", labelpad=30)
fig.tight_layout(rect=[0, 0, 1, 0.985])
for e in ["png", "pdf"]:
    fig.savefig(OUT / f"17_masks_overview.{e}", dpi=200, bbox_inches="tight")
print("masks ->", OUT / "17_masks_overview.png")

# --- Gate-Alpha: 3 repraesentative Beispiele (statisch -> dynamisch) ---
order = np.argsort([a.mean() for a in alphas])
sel = [order[0], order[len(order) // 2], order[-1]]     # min, median, max alpha
fig2, ax2 = plt.subplots(1, 3, figsize=FIG(11, 4.0))
for j, s in enumerate(sel):
    im = ax2[j].imshow(alphas[s], cmap="viridis", vmin=0, vmax=1)
    ax2[j].set_xticks([]); ax2[j].set_yticks([])
    _m = alphas[s].mean()
    _tag = "überw. kopieren" if _m < 0.4 else ("überw. vorhersagen" if _m > 0.55 else "gemischt")
    # dreizeilig: bei 17 cm Einbaubreite passt eine einzeilige Zeile nicht
    ax2[j].set_title(f"mIoU {fmt(mious[s])}\nmean α = {fmt(_m, 2)}\n{_tag}",
                     fontsize=S(10))
cb = fig2.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
cb.set_label("Gate-Alpha  (0 = kopieren, 1 = vorhersagen)")
for e in ["png", "pdf"]:
    fig2.savefig(OUT / f"17_gate_alpha.{e}", dpi=200, bbox_inches="tight")
print("gate ->", OUT / "17_gate_alpha.png")
print(f"[render] {n} Beispiele, {n_cls} Klassen | mIoU {mious.min():.3f}-{mious.max():.3f} "
      f"| alpha {min(a.mean() for a in alphas):.2f}-{max(a.mean() for a in alphas):.2f}")
