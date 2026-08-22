"""
visualize_predictions.py — BEV World Model: Prediction Visualisierung
======================================================================

Lädt pred_*.npy und real_*.npy aus einem Inference-Output-Ordner
und erstellt Heatmaps analog zum sanity_check.py aus Task 1.

AUFRUF:
    python visualize_predictions.py \
        --pred-dir predictions/phase2_check \
        --out-dir  visualizations/phase2 \
        --n-vis    4

OUTPUT:
    visualizations/phase2/
        heatmaps_overview.png    -- Pred | Real | |Diff| für alle Samples
        distribution.png         -- Histogramm + Channel-Aktivierung
        channel_grid.png         -- Einzelne Channels (0,42,128,200) Vergleich
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_pairs(pred_dir: Path, n_max: int):
    """Lädt pred/real Paare aus dem Verzeichnis."""
    pred_files = sorted(pred_dir.glob("pred_*.npy"))[:n_max]
    pairs = []
    for pf in pred_files:
        token = pf.stem.replace("pred_", "")
        rf = pred_dir / f"real_{token}.npy"
        if not rf.exists():
            print(f"  [Warnung] real_{token}.npy nicht gefunden — übersprungen")
            continue
        pred = np.load(str(pf)).astype("float32")   # [256, 128, 128]
        real = np.load(str(rf)).astype("float32")   # [256, 128, 128]
        pairs.append((token, pred, real))
    return pairs


# ---------------------------------------------------------------------------
# Plot 1: Heatmap Overview (Pred | Real | |Diff|)
# ---------------------------------------------------------------------------

def plot_heatmaps(pairs, out_dir: Path):
    """
    Pro Sample eine Zeile mit 3 Spalten:
        Predicted t+1 | Real t+1 | |Predicted - Real|

    Channel-Mittelung: [256, 128, 128] → .mean(axis=0) → [128, 128]
    """
    n = len(pairs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(
        "BEV World Model — Predicted vs. Real (channel-gemittelt)\n"
        "Phase 2 (Cell-Level, 3072 Tokens)",
        fontsize=13, y=1.01
    )

    col_titles = ["Predicted t+1", "Real t+1", "|Pred − Real|"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11,
                               color=["steelblue","tomato","darkorange"][col])

    for row, (token, pred, real) in enumerate(pairs):
        # Channel-Mittelung
        pred_mean = pred.mean(axis=0)   # [128, 128]
        real_mean = real.mean(axis=0)   # [128, 128]
        diff_mean = np.abs(pred_mean - real_mean)

        # Gemeinsame Farbskala für Pred und Real
        vmin = min(pred_mean.min(), real_mean.min())
        vmax = max(pred_mean.max(), real_mean.max())

        # MSE für dieses Sample
        mse = float(((pred - real)**2).mean())

        # Predicted
        im0 = axes[row][0].imshow(pred_mean, cmap="viridis",
                                   vmin=vmin, vmax=vmax, origin="upper")
        axes[row][0].axis("off")
        plt.colorbar(im0, ax=axes[row][0], fraction=0.046, pad=0.04)
        axes[row][0].set_ylabel(f"{token[:12]}…\nMSE={mse:.4f}",
                                 fontsize=8, rotation=0, labelpad=80, va="center")

        # Real
        im1 = axes[row][1].imshow(real_mean, cmap="viridis",
                                   vmin=vmin, vmax=vmax, origin="upper")
        axes[row][1].axis("off")
        plt.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)

        # Differenz (eigene Skala, hot colormap)
        im2 = axes[row][2].imshow(diff_mean, cmap="hot",
                                   vmin=0, origin="upper")
        axes[row][2].axis("off")
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = out_dir / "heatmaps_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# Plot 2: Verteilung (Histogramm + Channel-Aktivierung)
# ---------------------------------------------------------------------------

def plot_distribution(pairs, out_dir: Path):
    """
    Überlagert Pred- und Real-Histogramme + Channel-Aktivierung.
    Zeigt ob die Wertverteilungen übereinstimmen (Hauptindikator Decoder).
    """
    # Alle Werte zusammenfassen
    all_pred = np.concatenate([p.flatten() for _, p, _ in pairs])
    all_real = np.concatenate([r.flatten() for _, _, r in pairs])

    # Channel-Aktivierung: mittlere Aktivierung pro Channel über alle Samples
    ch_pred = np.stack([p for _, p, _ in pairs]).mean(axis=(0, 2, 3))  # [256]
    ch_real = np.stack([r for _, _, r in pairs]).mean(axis=(0, 2, 3))  # [256]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wertverteilung: Predicted vs. Real", fontsize=13)

    # --- Histogramm (log-Skala) ---
    bins = np.linspace(
        min(all_pred.min(), all_real.min()),
        min(max(all_pred.max(), all_real.max()), 20),   # cap bei 20 für Übersicht
        120
    )
    axes[0].hist(all_real, bins=bins, alpha=0.6, color="tomato",
                 label=f"Real   (mean={all_real.mean():.4f}, std={all_real.std():.4f})",
                 log=True)
    axes[0].hist(all_pred, bins=bins, alpha=0.6, color="steelblue",
                 label=f"Pred   (mean={all_pred.mean():.4f}, std={all_pred.std():.4f})",
                 log=True)
    axes[0].set_xlabel("Latent-Wert")
    axes[0].set_ylabel("Anzahl Pixel (log)")
    axes[0].set_title("Histogramm (alle Samples, log-Skala)")
    axes[0].legend(fontsize=8)

    # --- Channel-Aktivierung ---
    x = np.arange(256)
    axes[1].bar(x, ch_real, alpha=0.6, color="tomato",  label="Real")
    axes[1].bar(x, ch_pred, alpha=0.6, color="steelblue", label="Pred")
    axes[1].set_xlabel("Channel Index")
    axes[1].set_ylabel("Mittlere Aktivierung")
    axes[1].set_title("Mittlere Aktivierung pro Channel")
    axes[1].legend(fontsize=8)

    # --- Scatter: Pred-mean vs Real-mean pro Channel ---
    axes[2].scatter(ch_real, ch_pred, alpha=0.5, s=15, color="steelblue")
    mn = min(ch_real.min(), ch_pred.min())
    mx = max(ch_real.max(), ch_pred.max())
    axes[2].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="y=x (perfekt)")
    axes[2].set_xlabel("Real Channel-Mittelwert")
    axes[2].set_ylabel("Pred Channel-Mittelwert")
    axes[2].set_title("Channel-Korrelation (Pred vs. Real)")
    axes[2].legend(fontsize=8)

    # Pearson-Korrelation
    corr = float(np.corrcoef(ch_real, ch_pred)[0, 1])
    axes[2].text(0.05, 0.95, f"Pearson r = {corr:.4f}",
                 transform=axes[2].transAxes, fontsize=9,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    out = out_dir / "distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")
    print(f"  Pearson-Korrelation (Channel-Mittelwerte): {corr:.4f}")


# ---------------------------------------------------------------------------
# Plot 3: Einzelne Channels
# ---------------------------------------------------------------------------

def plot_channels(pairs, out_dir: Path, channels=(0, 42, 128, 200)):
    """
    Zeigt einzelne Channels um zu sehen ob Pred und Real
    dieselben Channels 'feuern'.
    """
    n_samples = min(len(pairs), 2)
    n_ch = len(channels)
    n_rows = n_samples * 2   # je Sample: Pred-Zeile + Real-Zeile

    fig, axes = plt.subplots(n_rows, n_ch,
                              figsize=(n_ch * 3, n_rows * 3),
                              squeeze=False)
    fig.suptitle("Einzelne Channels: Predicted vs. Real", fontsize=12, y=1.01)

    for ch_col, ch_idx in enumerate(channels):
        axes[0][ch_col].set_title(f"Channel {ch_idx}", fontsize=10)

    for sample_i in range(n_samples):
        token, pred, real = pairs[sample_i]
        row_pred = sample_i * 2
        row_real = sample_i * 2 + 1

        for ch_col, ch_idx in enumerate(channels):
            p_ch = pred[ch_idx]   # [128, 128]
            r_ch = real[ch_idx]   # [128, 128]
            vmin = min(p_ch.min(), r_ch.min())
            vmax = max(p_ch.max(), r_ch.max())

            im_p = axes[row_pred][ch_col].imshow(p_ch, cmap="viridis",
                                                   vmin=vmin, vmax=vmax)
            axes[row_pred][ch_col].axis("off")
            plt.colorbar(im_p, ax=axes[row_pred][ch_col], fraction=0.046, pad=0.04)

            im_r = axes[row_real][ch_col].imshow(r_ch, cmap="viridis",
                                                   vmin=vmin, vmax=vmax)
            axes[row_real][ch_col].axis("off")
            plt.colorbar(im_r, ax=axes[row_real][ch_col], fraction=0.046, pad=0.04)

        axes[row_pred][0].set_ylabel(f"Sample {sample_i+1}\nPred",
                                      fontsize=9, rotation=0,
                                      labelpad=45, va="center", color="steelblue")
        axes[row_real][0].set_ylabel(f"Sample {sample_i+1}\nReal",
                                      fontsize=9, rotation=0,
                                      labelpad=45, va="center", color="tomato")

    plt.tight_layout()
    out = out_dir / "channel_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# Zusammenfassung
# ---------------------------------------------------------------------------

def print_summary(pairs):
    print("\n" + "="*60)
    print("  ZUSAMMENFASSUNG")
    print("="*60)
    mses  = [float(((p-r)**2).mean()) for _, p, r in pairs]
    means = [(p.mean(), r.mean()) for _, p, r in pairs]
    stds  = [(p.std(),  r.std())  for _, p, r in pairs]

    print(f"\n  {'Token':36s}  {'MSE':>8}  {'Pred-mean':>10}  {'Real-mean':>10}  {'Δmean':>8}")
    print("  " + "-"*76)
    for i, (token, pred, real) in enumerate(pairs):
        pm, rm = means[i]
        print(f"  {token[:36]:36s}  {mses[i]:>8.5f}  {pm:>10.4f}  {rm:>10.4f}  {abs(pm-rm):>8.5f}")

    print(f"\n  Ø MSE:       {np.mean(mses):.6f}")
    print(f"  Ø Pred-mean: {np.mean([m[0] for m in means]):.4f}")
    print(f"  Ø Real-mean: {np.mean([m[1] for m in means]):.4f}")
    print(f"  Ø Pred-std:  {np.mean([s[0] for s in stds]):.4f}")
    print(f"  Ø Real-std:  {np.mean([s[1] for s in stds]):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualisiert pred/real BEV-Latents aus Inference-Output."
    )
    parser.add_argument("--pred-dir", required=True,
                        help="Ordner mit pred_*.npy und real_*.npy")
    parser.add_argument("--out-dir",  default="visualizations",
                        help="Ausgabe-Ordner für PNGs (default: visualizations)")
    parser.add_argument("--n-vis",    type=int, default=4,
                        help="Anzahl Samples visualisieren (default: 4)")
    parser.add_argument("--channels", nargs="+", type=int,
                        default=[0, 42, 128, 200],
                        help="Welche Channels für channel_grid.png (default: 0 42 128 200)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BEV World Model — Prediction Visualisierung")
    print(f"  Input:  {pred_dir.resolve()}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*60}\n")

    # Paare laden
    pairs = load_pairs(pred_dir, args.n_vis)
    if not pairs:
        print("[Fehler] Keine pred/real Paare gefunden!")
        sys.exit(1)
    print(f"  {len(pairs)} Paare geladen\n")

    print("Erstelle Plots...")
    plot_heatmaps(pairs, out_dir)
    plot_distribution(pairs, out_dir)
    plot_channels(pairs, out_dir, tuple(args.channels))
    print_summary(pairs)

    print(f"\n  Fertig! PNGs in: {out_dir.resolve()}/")
    print(f"    heatmaps_overview.png  — Pred | Real | |Diff|")
    print(f"    distribution.png       — Histogramm + Channel-Korrelation")
    print(f"    channel_grid.png       — Einzelne Channels\n")


if __name__ == "__main__":
    main()
