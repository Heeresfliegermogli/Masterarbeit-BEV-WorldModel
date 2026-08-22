# TASK 16b.9 — Synthese + Headline (Abschluss der 16b-Sweep-Kette)

**Datum:** 2026-07-18 bis 2026-07-19
**Chat:** Task 16b.9 (Synthese + Full-Val-Headline)
**Status:** ABGESCHLOSSEN (Job 123424 headline COMPLETED, ExitCode 0, 12h44min)
**Baut auf:** 16b.4 (grad), 16b.5 (ssim), 16b.6 (mean/cos), 16b.7 (recon_loss)

## Ziel

Zusammenfassender Abschluss der 16b-Loss-Studie: (P1) Gesamttabelle +
Dosis-Wirkungs-/Trade-off-Plots + minimale hinreichende Loss-Menge; (P2) der
finale Headline-mIoU auf vollem Val (5743) OHNE Early-Stop fuer die beste
reduktive Konfig gegen die Baseline.

## Phase 1 — Synthese (lokal, synth_16b9.py)

Master-Tabelle aller 16 Sweep-Punkte -> predictions/task16b/synthesis_master.csv.
Plots (Okabe-Ito, farbenblind-sicher; PNG+PDF) -> visualizations/:
- 16b9_dose_response.* : dmIoU + dstd-Ratio je additivem Regler gegen das
  gedrehte Gewicht, mit 15.2-Signifikanzbaendern.
- 16b9_tradeoff.* : std-Ratio vs mIoU aller Punkte + Ideal-Linie (std=1).

Befund der Synthese (300-Proxy, n=1/Wert):
- grad: mIoU n.s. ↑ bis 0.3, std-Ratio signifikant ↓ (schlechter).
- ssim: mIoU ↓ mit λ (sig. bei 0.3), std-Ratio flach.
- mean: flach auf beiden Metriken.
- cos: mIoU n.s. ↑ zu 0.5, std-Ratio signifikant ↓; Nullpunkt beste std-Ratio.
- recon (16b.7): SmoothL1 = mIoU gleich, std-Ratio SIGNIFIKANT besser.
-> Im Trade-off ist SmoothL1 der EINZIGE Pareto-guenstige Punkt. Alle Konfigs
   bleiben << 1.0 (std-Gap strukturell -> motiviert Task 18/generativ).

**Minimale hinreichende Loss-Menge:** mse/std tragend; grad/ssim/mean/cos bei
λ=0 entbehrlich (|dmIoU| < 0.014). Mit 16b.7: die tragende Rekonstruktion ist
SmoothL1 statt MSE. -> Kandidat MINIMAL = smooth_l1 + std (2 Terme).

## Phase 2 — Headline (Cluster, Job 123424, 2 GPU parallel)

Zwei Laeufe, miou_early_stop=false, bis Epoche 50 (Val-Loss-Patience 10 griff
NICHT -> voller 50-Epochen-Lauf), danach eval_full_val.py auf ALLEN 5743
Val-Samples:
- BASELINE: mse=1, cos=0.1, mean=0.1, std=1, grad=0, ssim=0.1 (recon=mse) -- 6 Terme.
- SMOOTHL1-MINIMAL: smooth_l1=1, std=1, Rest 0 -- 2 Terme.
PLAIN abgesetzt (sbatch_headline.sh, kein --export). Staging fehlerfrei; beide
parallel auf a100-4; Full-Val-Eval je ~4 min.

### Ergebnis (der Headline)

| Konfig            | Terme | mIoU (Full-Val 5743) | vs. Ref 0.6823 | vs. Persist. 0.5329 |
|-------------------|-------|----------------------|----------------|---------------------|
| Baseline          | 6     | 0.6920               | +0.0097        | +0.1591             |
| SmoothL1-Minimal  | **2** | **0.6946**           | **+0.0123**    | +0.1617             |

**Delta SmoothL1-Minimal − Baseline = +0.0026** (< Schwelle 0.014 -> gleichauf,
SmoothL1 minimal vorn).

Full-Val vs. 300-Proxy: die Full-Val-Werte (0.692/0.695) liegen ~0.004-0.006
ueber den 300-Subset-Decode-Plateaus (~0.688/0.690) am Trainingsende -- der
Proxy unterschaetzt leicht (fixe seed-42-Auswahl vs. voller Split). Transparent
gemacht wie vom Arbeitsplan gefordert.

## Kernbefund (der berichtbare Erkenntnisbeitrag)

**"From six to two":** Die Minimal-Konfig mit nur `smooth_l1 + std` ERREICHT die
volle 6-Term-Baseline (+0.0026, n.s.). Das Weglassen von cos/mean/grad/ssim
kostet nichts. Vier von sechs Loss-Termen sind entbehrlich.

**Neue Bestmarke:** Der Headline-Lauf OHNE mIoU-Early-Stop findet den echten
konvergierten mIoU (0.6946), ueber der bisherigen Bestmarke 0.6823 (early-
stopped). SmoothL1-Minimal ist Gesamtsieger: wenigste Terme, hoechste mIoU,
UND bester std-Gap (16b.7).

**Deterministische Decke:** Trotz der Verbesserung bleibt die std-Ratio deutlich
< 1.0 (~0.9 mit MSE, ~0.93 mit SmoothL1) -- der std-Gap ist strukturell
(Regression-to-the-mean), nicht durch Loss-Tuning schliessbar. Das motiviert
Task 18 (probabilistisch/generativ), wenn der Gap kritisch wird.

Framing: LeWM ("from six to one"), DINO-Foresight (SmoothL1 genuegt) -- die
16b-Studie liefert das analoge reduktive Ergebnis fuer das Seg-BEV-World-Model.

## Empfohlene finale Loss-Konfig (fuer 16c und die Thesis)

`recon_loss: smooth_l1`, `lambda_mse(=TERM1)=1.0`, `lambda_std=1.0`, Rest 0.
Checkpoint: checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt
(Full-Val-mIoU 0.6946).

## Geaenderte/erzeugte Dateien

- synth_16b9.py (Synthese-Skript), predictions/task16b/synthesis_master.csv
- visualizations/16b9_dose_response.{png,pdf}, 16b9_tradeoff.{png,pdf}
- configs/task16b/headline/config_headline_{baseline,smoothl1_minimal}.yaml
- sbatch_headline.sh (PLAIN; 2 Laeufe + Full-Val-Eval)
- checkpoints/task16b/headline/{baseline,smoothl1_minimal}/phase2/
  {best_miou.pt, miou_fullval.json}
- predictions/task16b/... (Sweep-CSVs bereits vorhanden)

## Naechster Schritt

Task 16c (MUSS) — autoregressive Rollout-Evaluation k=1..4, KEIN Training.
Basis-Konfig = SmoothL1-Minimal-Checkpoint (0.6946). Liefert mIoU(k) + std-Ratio(k)
und den Drift-Befund (mit dem std-Gap-Befund aus 16b.9 die Grundlage fuer die
Diskussion deterministisch vs. generativ in der Thesis).
