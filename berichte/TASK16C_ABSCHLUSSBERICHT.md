# TASK 16c — Autoregressive Rollout-Evaluation (k=1..4) (Abschlussbericht)

**Datum:** 2026-07-20
**Chat:** Task 16c (Multi-Step-Rollout, kein Training)
**Status:** ABGESCHLOSSEN (Job 123442 COMPLETED, ExitCode 0, 26min)
**Baut auf:** 16b.9 (finale Konfigs + Checkpoints)

## Ziel

Multi-Step-/Drift-Analyse durch autoregressiven Rollout des trainierten
3->1-Modells (KEIN Retraining, KEINE Architekturaenderung). Horizont k=1..4
(~0.5-2.0s @ nuScenes-Keyframes 2Hz). Groesste identifizierte Angriffsflaeche
gegenueber Stand 2026 (DINO-Foresight etc. berichten Multi-Step-Rollout).

## Methode (rollout_eval.py, idempotent)

Rein zur Inferenzzeit: die eigene Praediktion wandert ins Kontextfenster
(inference.py-Muster: cur = cat([cur[:,1:], pred])). Nur szeneninterne,
konsekutive 7-Frame-Fenster (n_input=3 + K=4), nie ueber Szenengrenzen ->
**5467 Val-Fenster** (92 Val-Szenen). Metriken je Schritt, ueber alle Fenster:
- mIoU(k)      = IoU(decode(F_{3+k}_pred), decode(F_{3+k}_real)) (Seg-Decoder, wie 14/15/16b)
- std-Ratio(k) = mean_ch_std(F_{3+k}_pred) / mean_ch_std(F_{3+k}_real)
- Persistenz(k)= IoU(decode(F3_real), decode(F_{3+k}_real))  (triviale Referenz)
Beide 16b.9-Checkpoints (baseline, smoothl1_minimal) parallel, PLAIN abgesetzt,
KEIN /dev/shm-Staging (config_rollout.yaml stage_to_shm:false -> Val von BeeGFS).

## Ergebnisse

| k | Baseline mIoU | SmoothL1-Min mIoU | Persistenz | Baseline std-R | SmoothL1 std-R |
|---|---------------|-------------------|------------|----------------|----------------|
| 1 | 0.6926        | 0.6952            | 0.5326     | 0.9090         | 0.9859         |
| 2 | 0.5961        | 0.6000            | 0.4269     | 0.9298         | 0.9872         |
| 3 | 0.5241        | 0.5306            | 0.3776     | 0.9511         | 0.9997         |
| 4 | 0.4678        | 0.4769            | 0.3470     | 0.9685         | 1.0127         |

(n=5467 je Zelle. JSONs: predictions/task16c/rollout_{baseline,smoothl1_minimal}.json;
Plot: visualizations/16c_rollout.{png,pdf})

### Konsistenz-Check (bestanden)
- k=1-mIoU (0.6926/0.6952) ~ 16b.9-Headline-Full-Val (0.6920/0.6946) -> der
  Single-Step-Rollout reproduziert die Headline; Pipeline korrekt.
- Persistenz(k=1) = 0.5326 ~ projektbekannte Persistenz-Baseline 0.5329.

### Drift
- mIoU faellt k1->k4 um **~32% relativ** (Baseline -32.5%, SmoothL1 -31.4%);
  absolut von ~0.69 auf ~0.47 (~2s Horizont). Moderater Drift, erwarteter
  Exposure-Bias (das Modell sah im Training nie eigene, verrauschte Inputs).
- **Modell schlaegt die Persistenz bei JEDEM k** (beide Konfigs) -- selbst bei
  k=4 noch +0.12-0.13 mIoU ueber der trivialen Baseline. Das Modell bleibt ueber
  den ganzen Horizont nutzbar, faellt NIE unter "letzten Frame wiederholen".

### SmoothL1-Vorteil KOMPOUNDIERT
dmIoU (SmoothL1 - Baseline) je k: +0.0026, +0.0039, +0.0065, +0.0091 -> waechst
mit dem Horizont. Die 2-Term-Minimal-Konfig ist nicht nur gleichauf (16b.9),
sondern ueber den Rollout **leicht stabiler** als die 6-Term-Baseline.

### std-Gap-Kompoundierung (direkter Rueckbezug auf den Loss-Kern)
Die std-Ratio STEIGT mit k (autoregressive Rueckfuehrung blaeht die Varianz auf):
- Baseline (MSE): 0.9090 -> 0.9685 (naehert sich 1 von unten).
- SmoothL1-Minimal: 0.9859 -> 1.0127 (startet nahe 1, SCHIESST bei k=4 leicht
  UEBER 1). SmoothL1 haelt die Varianz ueber den gesamten Rollout deutlich naeher
  am Ideal -- der 16b.7-Vorteil (weniger mean-seeking) traegt in den Rollout.
Interpretation: `lambda_std` + SmoothL1 halten den std-Gap im Single-Step klein
UND kompoundieren im Rollout guenstig (kein Varianz-Kollaps; eher leichtes
Ueberschiessen).

## Go/No-Go fuer Task 18 (generativ)

Zwei Kriterien, beide berichtbar:
1. std-Gap (16b.9): bleibt im Single-Step strukturell < 1.0 (MSE ~0.91),
   mit SmoothL1 ~0.99 fast geschlossen.
2. Rollout-Drift (16c): moderat (~32% rel. bis k=4), aber das Modell bleibt
   ueber den ganzen Horizont deutlich ueber der Persistenz.
-> Das deterministische Modell ist fuer kurze Horizonte (k<=2, ~1s) stark und
   ueber k=4 hinaus nutzbar. Task 18 (probabilistisch/generativ) ist MOTIVIERT
   (schaerfere Langhorizont-Praediktion, Varianz-Kontrolle), aber NICHT erzwungen
   -- kein katastrophaler Drift, kein Unterschreiten der Persistenz. Klare,
   ehrliche Grundlage fuer die det-vs-generativ-Diskussion in der Thesis.

## Geaenderte/erzeugte Dateien

- rollout_eval.py (idempotent, wiederverwendbar fuer Task-18-Output)
- config_rollout.yaml (stage_to_shm:false), sbatch_rollout.sh (PLAIN, 2 GPU)
- predictions/task16c/rollout_{baseline,smoothl1_minimal}.json
- visualizations/16c_rollout.{png,pdf}

## Naechster Schritt

Der Seg-Kern-MUSS-Pfad (Task 12-16) ist damit abgeschlossen. Weiter: Task 17
(Thesis-Materialien/Visualisierungen konsolidieren) bzw. Task 18 (OPTIONAL
generativ), falls der std-Gap/Drift-Befund es rechtfertigt. 16b.10 (Optuna)
bleibt optionales Seitengleis (nach 16b.9 wenig Grenznutzen, s. dortige Notiz).
