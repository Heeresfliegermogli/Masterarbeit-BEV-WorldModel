# TASK 16d (Vorstufe) — Ego-gewarpte Persistenz-Baseline (Abschlussbericht)

**Datum:** 2026-07-21
**Chat:** Verbesserungs-Check / ego-warped persistence
**Status:** ABGESCHLOSSEN (voller Lauf 5467 Fenster, KEIN Retraining)
**Baut auf:** 16c (Rollout), Verbesserungsvorschlag #1 (Ego-Conditioning-Notiz)

## Anlass

Die 16c-Persistenz ist NAIV (F3 unveraendert kopiert) und enthaelt die triviale
"Szene bewegt sich mit dem Ego"-Komponente. Der +0.16-Vorsprung des Modells
darueber ueberschaetzt den Beitrag. Die Occupancy-Literatur nutzt die
EGO-GEWARPTE Persistenz: F3 mit der echten Ego-Bewegung ins Frame von F_{3+k}
warpen -> haertere, ehrlichere Referenz. Umsetzbar OHNE Retraining.

## Methode (ego_warp_persistence.py, idempotent)

- Ego-Pose je Frame aus der pkl (ego2global_translation + _rotation-Quaternion,
  Zuordnung ueber token). yaw = atan2(2(wz+xy), 1-2(y^2+z^2)).
- Relative 2D-Starrkoerper-Transform F3-Ego -> F_{3+k}-Ego; decode(F3)-Maske
  (200x200, output_scope +/-50m/0.5m) per grid_sample warpen (nearest).
- mIoU_pers_ego(k) = IoU(warp(decode(F3)), decode(F_{3+k})).
- **Konvention EMPIRISCH validiert** (Geometrie fehleranfaellig): 8-Kombinationen-
  Sweep (yaw-Vorzeichen x Achsentausch x Transform-Richtung) auf 80 Fenstern.
  Zwei Gates: (1) Selbst-Warp k=0 -> mIoU 1.0 (Resampling korrekt); (2) ego>=naiv
  fuer alle k. EINDEUTIGER Gewinner: yaw=+1, AXIS_SWAP=True, INVERT=False
  (Achsentausch korrigiert x/y<->Grid-Konvention der BEV-Maske). Alle anderen 7
  Kombinationen schlechter als naiv. Voller Lauf: Gate1=1.0, naiv reproduziert
  16c exakt (k1 0.5326).

## Ergebnisse (5467 Fenster)

| k | Modell (SmoothL1-Min) | naive Pers. | ego-gewarpte Pers. | Modell - ego | Modell - naiv |
|---|-----------------------|-------------|--------------------|--------------|----------------|
| 1 | 0.6952                | 0.5326      | 0.6286             | +0.0666      | +0.1626        |
| 2 | 0.6000                | 0.4269      | 0.5387             | +0.0613      | +0.1731        |
| 3 | 0.5306                | 0.3776      | 0.4826             | +0.0480      | +0.1530        |
| 4 | 0.4769                | 0.3470      | 0.4418             | +0.0351      | +0.1299        |

Figur: visualizations/16d_ego_persistence.{png,pdf}.
JSON: predictions/task16c/ego_pers.json.

## Interpretation (ehrliche Neueinordnung)

- Die ego-gewarpte Persistenz ist bei jedem k DEUTLICH staerker als die naive
  (k1 0.629 vs 0.533) -> die naive Referenz war zu leicht.
- Das Modell schlaegt BEIDE Baselines bei JEDEM k. Aber der Vorsprung gegenueber
  der HARTEN (ego-gewarpten) Referenz ist nur +0.067..+0.035 -- gegenueber der
  naiven war er +0.16..+0.13, also ~2.5-4x aufgeblaeht durch die triviale
  Ego-Bewegung.
- Der Modellvorsprung ueber die ehrliche Baseline SCHRUMPFT mit dem Horizont
  (+0.067 bei k=1 -> +0.035 bei k=4): das Modell erfasst echte Szenendynamik
  ueber die Ego-Bewegung hinaus, aber MODERAT und mit dem Rollout abnehmend.

**Konsequenz fuers Write-up:** Der Headline-Vorsprung ist gegen die naive
Persistenz zu berichten UND gegen die ego-gewarpte einzuordnen -- letztere ist
die ehrliche Referenz. Der Beitrag des Modells (Dynamik jenseits Ego-Motion) ist
real, aber bescheiden. Motiviert direkt Verbesserungsvorschlag #1 (explizites
Ego-Conditioning: die Ego-Bewegung NICHT implizit lernen lassen).

## Geaenderte/erzeugte Dateien

- ego_warp_persistence.py (idempotent; Konventions-Sweep im --smoke)
- predictions/task16c/ego_pers.json
- visualizations/16d_ego_persistence.{png,pdf}

## Naechster Schritt

Empfohlene Reihenfolge (aus dem Verbesserungs-Check): (2) volles Ego-Conditioning
#1 (Retrain, hoechster Modell-Hebel) ODER die billigen Ablationen #4 (3->4 Frames)
+ #6 (EMA). #2 (Rollout-Finetuning) NICHT noetig (Modell > beide Persistenzen bei
jedem k). #5 (std-Gap generativ) durch 16b.9 weitgehend entkraeftet.
