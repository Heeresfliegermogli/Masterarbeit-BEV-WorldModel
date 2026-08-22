# TASK 16b.7 — L1/MSE-Ablation (recon_loss: MSE vs SmoothL1) (Abschlussbericht)

**Datum:** 2026-07-17 bis 2026-07-18
**Chat:** Task 16b.7 (kategoriale Rekonstruktions-Ablation)
**Status:** ABGESCHLOSSEN (Job 123303 COMPLETED, ExitCode 0, 3h42min)
**Baut auf:** `TASK16B_6_ABSCHLUSSBERICHT.md` (Sweep-Apparat, Staging-Fix)

## Ziel

Kategoriale Ablation der Rekonstruktionsdistanz (TERM 1 des Loss): MSE gegen
SmoothL1/Huber, sonst identische Baseline-Lambdas (mse=1, cos=0.1, mean=0.1,
std=1.0, grad=0, ssim=0.1). KEIN Regler-Sweep, ein Entweder-Oder. SmoothL1
gewaehlt statt reinem L1, weil naeher am Literatur-Komparator DINO-Foresight
[NeurIPS 2025] (dort genuegt SmoothL1 allein).

## Code-Aenderung (verifiziert VOR Compute)

TERM 1 war fest `F.mse_loss`. Neuer Schalter `recon_loss: mse|l1|smooth_l1` in
`compute_loss`, durch `train_one_epoch`/`validate`/main als Keyword durchgereicht,
aus der Config gelesen (`train_cfg.get("recon_loss", "mse")`), in `run_summary.json`
mitpersistiert. Default `mse` -> rueckwaerts-kompatibel.
Verifikation: `py_compile`; Baseline-Aequivalenztest `recon_loss="mse"`
BIT-IDENTISCH zur alten reinen MSE-Berechnung (diff 0.00e+00); l1/smooth_l1
liefern andere, positive Werte; ungueltige Werte -> ValueError.
`mk_sweep_cfgs.py` um `recon_loss` als patchbaren Schluessel erweitert (kategorial,
patch_line setzt den String). Basis-Config um `recon_loss: mse` ergaenzt.
Alle Dateien nach jedem Schritt md5-synchron lokal<->Cluster gehalten.

## Ablauf

Per-Regler-Kopie `sbatch_sweep_recon.sh` (REGLER=recon_loss, VALUES=(mse
smooth_l1), 2 GPU, 1 Welle), PLAIN abgesetzt (kein CLI --export -- 16b.6-Fix).
Staging fehlerfrei; beide Laeufe parallel; die Loss-Header-Zeilen bestaetigten
im echten Training identische Lambdas, nur TERM-1-Distanz verschieden:
  mse       -> "1.0*MSE + 0.1*CosSim + ... + 1.0*Std ..."
  smooth_l1 -> "1.0*SMOOTH_L1 + 0.1*CosSim + ... + 1.0*Std ..."
Danach Inference (300 Val, --no-save) + Harvest.

## Ergebnisse

| recon_loss | mIoU   | std-Ratio | pred_std | mse     | cossim | Stop-Epoche |
|------------|--------|-----------|----------|---------|--------|-------------|
| mse        | 0.6759 | 0.9089    | 0.2407   | 0.02865 | 0.8538 | 17          |
| smooth_l1  | 0.6735 | **0.9266**| 0.2454   | 0.02947 | 0.8513 | 17          |

(real_std = 0.2649; CSV: `predictions/task16b/sweep_recon_loss.csv`)

Deltas (SmoothL1 vs MSE):
- **mIoU: -0.0024 (n.s.)** -- praktisch gleich (weit unter Schwelle 0.014).
- **std-Ratio: +0.0177 (SIGNIFIKANT)** -- ~2x die Schwelle 0.009. Bester
  std-Ratio-Wert der gesamten 16b-Kette (ueber cos=0's 0.9182).

Sanity: der mse-Lauf (0.6759 / 0.9089) repliziert den Baseline-Anker
(~0.6732 / 0.9086) -> der mse-Pfad ist durch den Patch unveraendert.

## Interpretation (der eigentliche Befund)

**SmoothL1 ist der erste Term, der den std-Gap SIGNIFIKANT schliesst, ohne mIoU
zu kosten.** Alle additiven Regler waren entweder flach (mean/ssim) oder
verschlechterten die std-Ratio bei hohen Werten (grad/cos). SmoothL1 dreht das
um. Mechanismus: MSE bestraft grosse Abweichungen quadratisch -> zieht die
Vorhersage zum bedingten Mittelwert -> schrumpft die Channel-std
(Regression-to-the-mean, der strukturelle std-Gap-Treiber). SmoothL1/Huber
bestraft den Ausreisser-Schwanz nur linear -> erhaelt mehr Varianz -> pred_std
steigt (0.2407 -> 0.2454, naeher an real 0.2649). Anschlussfaehig an
DINO-Foresight (SmoothL1 genuegt). Vorbehalt: n=1 Seed, Schwellen fuer n=3
kalibriert; der std-Ratio-Effekt (0.0177) ist aber robuster als die meisten
Sweep-Deltas (~2x Schwelle).

## Konsequenz fuer 16b.9

Die minimale hinreichende Loss-Menge ist damit NICHT einfach "mse+std", sondern
plausibel **"smooth_l1 + std"** -- gleiche mIoU, besserer std-Gap, ein Term
weniger als die Baseline. Der Headline-Vergleich in 16b.9 sollte SmoothL1
beruecksichtigen (Kandidaten: mse-minimal vs smooth_l1-minimal vs Baseline).

## Geaenderte/erzeugte Dateien

- `train_linux.py` — recon_loss-Schalter (TERM 1) + Durchreichung + run_summary.
- `mk_sweep_cfgs.py` — recon_loss als patchbarer Schluessel.
- `config_sweep_base_cluster_fp16.yaml` — `recon_loss: mse` ergaenzt.
- `sbatch_sweep_recon.sh` — Per-Regler-Kopie (recon_loss, mse/smooth_l1), plain.
- `configs/task16b/recon_loss_sweep/config_recon_loss_{mse,smooth_l1}.yaml`
- `checkpoints/task16b/recon_loss_sweep/recon_loss_{mse,smooth_l1}/phase2/` (Cluster)
- `predictions/task16b/sweep_recon_loss.csv`

## Naechster Schritt

Task 16b.9 — Synthese (jetzt inkl. SmoothL1): Gesamttabelle + Dosis-Wirkungs-
Plots + Trade-off std-Ratio vs mIoU + minimale hinreichende Loss-Menge. Headline
auf vollem Val (5743) ohne Early-Stop; Kandidaten mse-minimal / smooth_l1-minimal
/ Baseline (Nutzer-Entscheidung).
