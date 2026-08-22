# Task 25 — Verfeinerte Decoder-Adaptation Det (Abschlussbericht, 03.08.)

Verfeinerung des Task-24-Kopfs: (1) 12 Ep. + Cosine-LR statt 6 konstant,
(2) REAL-MIX 20% (2703 der 13519 Train-Token IMMER als reales Latent, GT
identisch), (3) Val-Loss-Selektion auf 500 Holdout-Vorhersagen (Protokoll
wie Task-20-WM-Selektion; Training adapt_det_head_v2.py, Best = Ep. 9,
VAL 3.544->3.227 mit Schwankungen — Selektion notwendig, v1 hatte keine).

## Ergebnisse (Injection, Full-Val 6019)

| Kopf | pred mAP/NDS | real mAP/NDS |
|---|---:|---:|
| frozen             | 0.3438/0.4761 | 0.6858/0.7146 (Orakel) |
| v1 (Task 24)       | 0.4329/0.5328 | 0.6603/0.6918 |
| v2 (Task 25, Ep.9) | 0.4292/0.5333 | 0.6710/0.7040 |

BEFUND 1: Vorhersage-Gewinn GEHALTEN (-0.0037 mAP zu v1 = unter dem
Det-Rauschboden 0.0044; NDS sogar +0.0005).
BEFUND 2: REAL-MALUS FAST HALBIERT: -0.0255 -> -0.0148 mAP (-0.0228 ->
-0.0106 NDS) — der Real-Mix wirkt wie erhofft.
EMPFEHLUNG: v2 als EIN-KOPF-Betriebspunkt — ein einziger Kopf, der auf
Vorhersagen +0.085 mAP ueber frozen liegt und auf realen Latents nur
~0.015 mAP unter dem Orakel. v1 bleibt als Spezialist archiviert (bester
pred-Wert 0.4329), frozen als Orakel-Referenz. Laenger+Cosine allein
brachte auf pred nichts Zusaetzliches (Verbesserung steckt im Mix +
der Selektion, nicht in der Dauer).
Einschraenkungen: 1 Seed; Mix-Anteil 20% nicht durchgesweept; Kopf kennt
nur det_baseline-Vorhersagen.
Artefakte: adapt_det_head_v2.py, adapted_det/det_adapted_v2_{best,last}.pth,
adapt25_train.log, eval25_{pred,real}.log, real_mix_tokens.txt (Seed 42).

## NACHTRAG 03.08. — Figuren + Master-Tabelle
- 25_head_comparison.*: frozen/v1/v2 auf Vorhersagen + realen Latents
  (mAP+NDS, Orakel-Linie) — die quantitative Kernfigur des Kapitels.
- 25_det_boxes_versions.*: Fussgaenger-Szenen, 4 Spalten GT|frozen|v1|v2 —
  v1 und v2 qualitativ auf Augenhoehe (v2 haelt die Rueckgewinnung in
  Szene A, in B nur v1 sichtbar; konsistent mit statistischem Gleichstand).
- predictions/task25/head_master_table.csv: konsolidierte Versions-Tabelle
  BEIDER Straenge (Seg frozen/23; Det frozen/v1/v2 x pred/real, mit Quellen).
