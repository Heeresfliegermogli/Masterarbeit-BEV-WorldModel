# Thesis-Gliederung (Entwurf 03.08.2026)

Erzaehlbogen: Ein schlankes World-Model im Feature-Raum eines eingefrorenen
Wahrnehmungsmodells — wie weit traegt es, wo liegt seine Decke, und wo sitzt
der Hebel dagegen. Jedes Kapitel nennt seine Quellen (Berichte/Figuren).

## 1. Einleitung
Motivation (Vorhersage als Kernfaehigkeit AD), Forschungsfragen:
(F1) Kann ein ~6M-Transformer BEV-Latents eines frozen Encoders praedizieren?
(F2) Welche Loss-Terme tragen? (F3) Wo liegt die Leistungsgrenze und wem
gehoert sie? (F4) Laesst sie sich verschieben? Beitragsliste (Minimal-Loss,
Injection-Messkette, Fehlerzerlegung, Decoder-Adaptation).

## 2. Grundlagen & Verwandte Arbeiten          [WORLDMODEL_REFERENZEN.md]
BEV-Wahrnehmung (BEVFusion), World-Models: 4 Linien (deterministisch: LAW,
DINO-Foresight; generativ: Vista, BEVWorld; occupancy: OccWorld; planning:
World4Drive) + JEPA-Strang (LeWM) und Frozen-vs-Co-trained-Debatte 2026
(frozen > finetuned Evidenz). Einordnung des eigenen Ansatzes: Feature-
Forecasting mit task-nativem BEV-Encoder.

## 3. Methode                                   [Kap. 12-14, 20.0/20.1]
Pipeline (frozen Encoder/Decoder, Latents 256x128x128 bzw. 256x180x180),
Modell (4-Layer Pre-LN, gated skip, 6.05M), Loss-Baukasten (6 Terme + recon-
Schalter), drei Messebenen: Proxy / Full-Val / INJECTION gegen echte GT
(Orakel-Gates als Kettenbeweis). Signifikanz: Seed-Studien (0.014 mIoU /
0.0044 mAP).

## 4. Loss-Studie: From Six to Two              [15.x, 16b-Kette, Fig. 16b9_*]
Sweeps aller Regler -> smooth_l1+std genuegt auf Seg (0.6946); SmoothL1
schliesst den std-Gap (Fig. 17_stdgap). ABER metrik-spezifisch: auf Det
tragen die 6 Terme (+0.018 mAP = 4x Rauschboden; Kap. 20.3). LeWM-Parallele.

## 5. Die 0.69-Decke: sechsfach kreuzvalidiert  [16c-16f, 19; Fig. 19_all_levers]
Rollout (16c), Ego-Conditioning (16d), Kontext/EMA (16e), Task-Loss/Tiefe/
Residual (16f) — alles flach/negativ. Per-Klasse-Zerlegung (19_per_class),
aleatorische Evidenz (decode(t) vs decode(t+1)=0.53).

## 6. Generative Koepfe                          [18 B1/B2/B2.1; Fig. 18_*]
CVAE quasi-deterministisch -> Flow Matching: echte Multimodalitaet (37x),
Best-of-K-Zerlegung, post-hoc Kalibrierung; det. Modell + lambda_std bleibt
bester Punkt-Praediktor (Rollout-Varianz-Argument).

## 7. GT-Verankerung & Transfer                  [20, 21; Fig. 20/21_korridor]
Det-Transfer: Modell VERDOPPELT Persistenz-mAP; Seg gegen echte GT:
Rangfolge validiert, Retention 89-98%. FEHLERZERLEGUNG (Kernfigur
21_error_decomposition): Seg 90% Wahrnehmung / 10% Forecasting; Det 48/52.

## 8. Gezielte Hebel am falschen und richtigen Ort  [22, 23, 24, 25]
22: Selten-Klassen-/Dynamik-Gewichtung im WM — kein GT-Gewinn, aber
std-Kalibrierung (Meta-Befund: Proxy kann irrefuehren). 23: Seg-Kopf-
Adaptation (+0.0114, stop_line +0.033). 24/25: Det-Kopf v1/v2 (+0.089 mAP,
Real-Mix schliesst Real-Malus; Ein-Kopf-Betriebspunkt). Rueckholquote
~1/4-1/3 in beiden Straengen. Figuren 23_decoder_panel, 24_det_boxes_*,
25_head_comparison, 25_det_boxes_versions.

## 9. Diskussion
Empfohlene Betriebspunkte (Seg: 16b.9+Kopf23; Det: baseline+v2), Grenzen
(1 Seed bei Adaptation, frozen-Paradigma, nuScenes-Groesse), Einordnung in
die Frozen-Debatte. Future Work: Klassen-Gewichtung im Kopf, volle
Datenmenge, TC-WM-Adapter, Encoder-Co-Training, Det-Rollout.

## 10. Fazit
Antworten auf F1-F4 in je 2 Saetzen; Kernsatz: "Der richtige Ort fuer
Supervision ist der GT-trainierte Uebersetzer, nicht die Latent-Regression."

Anhang: Master-Tabellen (task19/master_table.csv, task25/head_master_table.csv),
Rauschboeden, Reproduzierbarkeit (scripts_render/, archiv/).
