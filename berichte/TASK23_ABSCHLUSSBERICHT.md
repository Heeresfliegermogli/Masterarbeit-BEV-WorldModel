# Task 23 — Decoder-Adaptation Seg (Abschlussbericht, 02.08.)

Datum: 2026-08-02. Motivation: Task 21/22 zeigten die Decke in der
WAHRNEHMUNG (frozen Decoder), nicht im World-Model; gezieltes WM-Training
(22) konnte seltene Klassen nicht heben. Task 23 testet den komplementaeren
Hebel: den POST-FUSER-STACK (SECOND+FPN+SegHead, 9.3M Param.) auf
(WM-Vorhersage -> echte GT) nachtrainieren — der "Uebersetzer" lernt die
leicht geglaetteten Vorhersage-Latents, das World-Model bleibt unangetastet.

## Protokoll

- 23.0: Train-Split-Vorhersagen des 16b.9-minimal-Modells (27038 Fenster,
  OHNE Real-Fill; eval_dump_latents --split train --no_fill) -> BeeGFS
  dumps_task23/ (~212G); GT-Cache im Pipeline-Grid 200x200 (mk_gt_masks_seg
  --grid pipe, Geometrie gate-verifiziert).
- 23.1: Standalone-Trainer im Docker (adapt_seg_head.py) OHNE Sensor-
  Pipeline: nur decoder.backbone+neck+heads.map trainierbar, Original-
  Focal-Loss, AdamW 1e-4, batch 8, 8 Epochen (~13 min/Ep. TITAN RTX).
  Val-Proxy (feste 0.5-Schwelle, 1000 Val-Vorhersagen) monoton
  0.5801 -> 0.5917; stop_line +0.036.
- 23.2: UNVERAENDERTE Injection-Kette (tools/test.py --eval map), nur
  Checkpoint = seg_adapted_full_best.pth; Full-Val 6019, IoU@max.

## Ergebnisse (echte GT)

(a) ADAPTIERTER Kopf + Val-VORHERSAGEN vs frozen Kopf (Task 21):

| | mean | drivable | ped_cross | walkway | stop_line | carpark | divider |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen    | 0.5898 | 0.8341 | 0.5618 | 0.6281 | 0.4746 | 0.5637 | 0.4761 |
| adaptiert | 0.6012 | 0.8373 | 0.5694 | 0.6382 | 0.5071 | 0.5669 | 0.4884 |
| Delta     | +0.0114| +0.0032| +0.0076| +0.0101| **+0.0325** | +0.0032| +0.0123|

(b) ADAPTIERTER Kopf + REALE Latents vs frozen Orakel:
mean 0.6293 vs 0.6295 (identisch); stop_line sogar +0.0155 (0.5269),
divider +0.0021, dafuer carpark -0.0098, ped_cross -0.0045 — Umverteilung
zu duennen Klassen bei unveraenderter Decke.

## Befunde

BEFUND 1 — DIE DECKE IST TEILWEISE ADRESSIERBAR, ABER IM KOPF: +0.0114
mean-GT-mIoU auf identischen Vorhersagen; der Orakel-Abstand schrumpft von
0.0397 auf 0.0281 (~29% des Forecasting-Gaps zurueckgeholt), Retention
93.7% -> 95.5%.

BEFUND 2 — GENAU DER SELTEN-KLASSEN-EFFEKT, DEN TASK 22 NICHT LIEFERN
KONNTE: stop_line +0.0325, divider +0.0123 — der frozen Kopf verlor duenne
Strukturen aus den geglaetteten Vorhersagen; der adaptierte extrahiert sie
wieder. Lehre: das Klassenwissen gehoert in den UEBERSETZER (auf GT
trainiert), nicht in die Latent-Regression (Task-22-Befund 1: dort wirkt
es nur statistisch/kalibrierend).

BEFUND 3 — KEIN PREIS AUF REALEN LATENTS: mean der Decke unveraendert
(0.6293 vs 0.6295) -> der Gewinn in (a) ist echte Uebersetzungs-
Verbesserung, keine Decken-Verschiebung; ein Kopf fuer beide Faelle ist
praktikabel.

EINSCHRAENKUNGEN: 1 Seed (Adaptations-Training; Eval deterministisch);
Kopf sah nur minimal-Modell-Vorhersagen (Uebertragbarkeit auf andere
WM-Konfigs ungetestet); Ablation mode=head (nur SegHead) nicht gerechnet
(Option). Fuer Det (52% Forecasting-Anteil) als Future Work notiert —
dort ist das Rueckholpotential groesser.

## Artefakte

- Checkpoints: det_latents_out/adapted/seg_adapted_full_{best,last}.pth
  (Voll-Format, direkt injection-/visualisierungs-tauglich).
- Code: adapt_seg_head.py (+ --mode head Ablation), eval_dump_latents
  --split/--no_fill, mk_gt_masks_seg --grid pipe, run_23_adapt.sh.
- Logs/Evals: adapt23_full.log, eval23_adapted_{pred,real}.log.
- BeeGFS: dumps_task23/dump_train_minimal, gt_masks_seg/gt_masks_pipe_*.
- Offen/optional: Masken-Panel GT | frozen | adaptiert (gleiche Szenen),
  head-only-Ablation, Det-Analogon.

## Lektion

- BEVFusion-SegHead liefert im Eval-Branch bereits SIGMOID-Wahrschein-
  lichkeiten — Doppel-Sigmoid im Proxy ergab IoU=Klassen-Praevalenz
  (drivable 0.286); bei fremden Koepfen IMMER die Output-Semantik der
  forward-Branches pruefen.
