# Task 24 — Decoder-Adaptation Det (Abschlussbericht, 02.08.)

Analogon zu Task 23 fuer den Det-Strang (TransFusion). Post-Fuser-Stack
(SECOND+FPN+TransFusionHead) auf (WM-Vorhersage det_baseline -> GT-Boxen)
nachtrainiert; GT direkt aus infos-pkl (valid_flag, 9-dim inkl. Velocity),
Loss-Assembly = Original head(feat,metas)+head.loss(...), keine Sensor-
Pipeline. Train: 13519 Vorhersagen (stride 2, ohne Real-Fill), 6 Epochen
AdamW 5e-5, Loss 5.2 -> 2.94 (Docker det_extract, ~2.5h TITAN RTX).

## Ergebnisse (Injection, Full-Val 6019)

| | mAP | NDS |
|---|---:|---:|
| frozen Kopf + Vorhersagen (Task 20) | 0.3438 | 0.4761 |
| ADAPTIERT + Vorhersagen | **0.4329** | **0.5328** |
| Delta | **+0.0891** | +0.0567 |
| frozen Orakel (real) | 0.6858 | 0.7146 |
| ADAPTIERT + real | 0.6603 | 0.6918 |

BEFUND 1: +0.0891 mAP (~20x Det-Rauschboden 0.0044) = groesster Einzel-
gewinn des Projekts; ~26% des Forecasting-Gaps zurueckgeholt — fast exakt
die Seg-Rueckholquote (29%, Task 23): die Adaptations-Quote scheint
metrik-uebergreifend stabil.
BEFUND 2: Anders als Seg zahlt Det einen Preis auf realen Latents
(-0.0255 mAP) — der Kopf spezialisiert sich staerker auf die Vorhersage-
Verteilung -> Einsatz als DEDIZIERTER Forecast-Kopf (Zwei-Kopf-Betrieb),
nicht als Ersatz.
KERNSATZ (23+24 gemeinsam): Die Wahrnehmungs-Decke ist teilweise
adressierbar — im GT-trainierten Uebersetzer, nicht im World-Model; die
Rueckholquote (~1/4 bis 1/3 des Forecasting-Gaps) reproduziert sich in
beiden Straengen.

Einschraenkungen: 1 Seed, halbe Trainingsmenge (stride 2), Kopf nur auf
det_baseline-Vorhersagen trainiert, keine Epochen-Selektion (last=best).
Artefakte: adapt_det_head.py, adapted_det/det_adapted_{best,last}.pth,
eval24_adapted_{pred,real}.log, BeeGFS dumps_task24/; Dumper --stride.
Lektionen: sbatch-Naht bei head/sed-Konstruktion pruefen (Job 127214,
Train-Teil verschluckt); Stride-Dumps drucken keinen Fortschritt
((i+1)%200 mit stride 2 nie 0) -> Ueberwachung via Dateizahl.

## NACHTRAG 02.08. — Figuren
- 24_det_boxes_panel.*: GT | Orakel | frozen+Vorhersage | adaptiert+Vorhersage
  (3 Szenen, Auswahl nach maximaler Box-Pixel-Differenz aus 80 gemeinsam
  gerenderten Frames — in Caption ausweisen). Frozen verliert Fahrzeugreihen,
  adaptiert stellt sie wieder her.
- 24_det_boxes_peds.*: Fussgaenger-Spezialfall — frozen+Vorhersage findet in
  fussgaengerreichen Szenen NULL Fussgaenger, adaptiert erste zurueck
  (+69% Fussgaenger-Detektionsflaeche ueber die 80 Frames, aber weit unter
  Orakel; partielle Rueckgewinnung ehrlich ausweisen).
