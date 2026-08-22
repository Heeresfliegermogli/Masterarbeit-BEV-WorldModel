# TASK 17 — Visualisierung & Thesis-Materialien (Abschlussbericht)

**Datum:** 2026-07-20
**Chat:** Task 17 (Thesis-Bildmaterial)
**Status:** ABGESCHLOSSEN (Kern-Abbildungssammlung konsolidiert + erzeugt)
**Baut auf:** 16b.4-16b.9 (Sweeps/Synthese), 16c (Rollout)

## Rahmen (ehrlich)

Task 17 war als BEGLEITENDES Auto-Logging ab Task 13 geplant -- das wurde nicht
aufgesetzt, ist am Kern-Ende aber gegenstandslos: die Laeufe sind durch, es wird
nicht neu trainiert, nur um Logging nachzuruesten. Stattdessen die EINMALIGEN
Thesis-Materialien aus vorhandenen Daten/Checkpoints konsolidiert + die fehlenden
qualitativen Bilder erzeugt. NICHT rekonstruierbar (transparent gemacht):
- std-Wert PRO Epoche: der std-Ratio wird nur zur Inferenzzeit (300-Subset)
  gemessen, nicht je Trainings-Epoche -> ersetzt durch den vorhandenen
  std-Ratio(k)-Rollout-Verlauf (16c) + std-Gap-Vorher/Nachher (unten).

## Erzeugte/konsolidierte Abbildungen (visualizations/)

| Datei | Inhalt | Caption-Kern |
|-------|--------|--------------|
| 16b9_tradeoff.{png,pdf}       | std-Ratio vs mIoU, alle 16 Sweep-Punkte | SmoothL1 einziger Pareto-guenstiger Punkt; alle << Ideal (std-Gap strukturell) |
| 16b9_dose_response.{png,pdf}  | dmIoU + dstd je Regler vs Gewicht | additive Regler entbehrlich (im 15.2-Rauschband) |
| 16c_rollout.{png,pdf}         | mIoU(k)/std-Ratio(k)/Persistenz, k=1..4 | moderater Drift, schlaegt Persistenz bei jedem k; SmoothL1 stabiler |
| 17_miou_over_epochs.{png,pdf} | mIoU-Verlauf, Headline ohne Early-Stop | beide Konfigs konvergieren > 0.6823; SmoothL1 durchgaengig >= Baseline |
| 17_stdgap_beforeafter.{png,pdf}| std-Ratio Baseline vs SmoothL1-Minimal | **KERNARGUMENT: std-Gap 9.1% -> 1.4% (Single-Step)** |
| 17_masks_overview.{png,pdf}   | pred vs real vs Agreement, 6 Val-Bsp | **STAERKSTES BILD: plausible BEV-Seg, mIoU 0.47-0.87** |
| 17_gate_alpha.{png,pdf}       | Gate-Alpha 3 Bsp (statisch->dynamisch) | Modell KOPIERT statische, SAGT dynamische Bereiche VORHER |
| 17_filmstrip.{png,pdf}        | 7-Frame-Rollout, real vs pred, statisch+dynamisch | **qualitativer 16c-Drift: statisch stabil (mIoU ~0.95), dynamisch driftet (0.31->0.18)** |

Architektur-Diagramme liegen bereits vor (Task-9/13-Material).

## 7-Frame-Rollout-Filmstrip (qualitative 16c-Ergaenzung)

Zwei automatisch gewaehlte 7-Frame-Szenensequenzen (per Gate-Alpha: statisch
mean-alpha 0.03 / dynamisch 0.79), je real F1..F7 oben vs. autoregressive
Praediktion unten (F1-F3 echter Kontext, F4-F7 Rollout mit mIoU je Schritt):
- STATISCH (Kreuzung, Fenster 1852): Modell kopiert -> mIoU 0.966/0.958/0.951/0.951,
  praktisch KEIN Drift ueber alle 4 Schritte. Struktur bleibt erhalten.
- DYNAMISCH (Gabelung, Fenster 4527): Modell sagt vorher -> mIoU 0.307/0.276/0.243/
  0.175, sichtbarer Drift/Verwaschung bis k=4.
-> macht den 16c-Befund (Drift + copy-vs-predict) in EINEM Bild sichtbar und
verbindet ihn mit der Gate-Interpretation. Datenbasis: predictions/task17/filmstrip.npz.

## Neuer qualitativer Befund (Gate-Alpha)

Der Gate-Alpha (Task-9b-Mechanismus, alpha in [0,1]: 0=Skip/kopieren,
1=Transformer/vorhersagen) ist raeumlich strukturiert UND korreliert mit der
Szenendynamik:
- statische Szenen -> niedriges alpha (~0.18), das Modell KOPIERT -> hohe mIoU (0.86).
- dynamische Szenen -> hohes alpha (~0.63), das Modell SAGT VORHER -> niedrigere mIoU (0.47).
Das validiert die Gate-Architektur (Task 9b) qualitativ: der copy-vs-predict-
Split wird gelernt und ist interpretierbar. Datenbasis: 6 seed-42-Val-Beispiele
(predictions/task17/thesis_masks.npz).

## Referenz-Positionierung (methodisches Geruest -- Komparator-Zahlen aus den
## Papers zu VERIFIZIEREN, hier NICHT erfunden)

Direkter mIoU-Leaderboard-Vergleich ist meist NICHT apples-to-apples (andere
Datensaetze/Metriken/Aufgaben). Daher methodische Einordnung:

| Referenz | Aufgabe / Setup | Eval-Protokoll | Positionierung zu dieser Arbeit |
|----------|-----------------|----------------|----------------------------------|
| **Diese Arbeit** | BEV-Seg-Latent-Praediktion (3->1), nuScenes | frozen-Seg-Decoder-mIoU (voller Val 5743) | mIoU **0.6946**, std-Ratio ~0.99, Minimal-Loss (smooth_l1+std), Rollout k=1..4 |
| DINO-Foresight [NeurIPS 2025] | Feature-Space-Forecasting, frozen DINO-Encoder, autoregressiver Rollout, SmoothL1 | frozen-Decoder auf gefrorenen Features | **engster methodischer Zwilling** -- legitimiert das frozen-Decoder-mIoU-Protokoll UND die SmoothL1-Wahl (16b.7) | 
| LeWM / LeWorldModel | Loss-Engineering ("from six to one") | -- | liefert das reduktive Framing; unser 16b.9-Befund ist das analoge "from six to two" |
| DINO-WM | World-Model auf DINO-Patch-Tokens (196/Frame) | -- | Token-Effizienz-Argument (Task-18-Vormerkung 16x16-Grid) |
| OccWorld | 3D-Occupancy-World-Model, VQVAE-Tokens, autoregressiv | Occupancy-Forecasting 1s/2s/3s | Multi-Step-Rollout-Standard (motiviert 16c) |
| BEVWorld | BEV-World-Model, multimodal | -- | deterministisches std-Limit-Argument (motiviert Task 18) |
| FIERY | BEV-Instance-Future-Prediction (Kamera->BEV), probabilistisch | BEV-Forecasting-Metriken | probabilistische Zukunft als Kontrast zum deterministischen Ansatz |
| LAW | Latent-World-Model, self-supervised Driving | -- | Latent-Praediktions-Paradigma |

-> To-do fuer die schriftliche Arbeit: je Referenz die berichtete Zahl + Metrik +
Datensatz aus dem Paper eintragen; klarstellen, WO ein direkter Vergleich zulaessig
ist (frozen-Decoder-Protokoll: DINO-Foresight) und wo nur qualitativ.

## Geaenderte/erzeugte Dateien

- visualize_thesis.py (Cluster-Compute: Decode -> .npz, kein matplotlib)
- render_thesis.py (lokal: .npz -> Masken/Gate-Figuren)
- synth_16b9.py (16b.9-Synthese-Plots, bereits vorhanden)
- predictions/task17/thesis_masks.npz
- visualizations/17_{miou_over_epochs,stdgap_beforeafter,masks_overview,gate_alpha}.{png,pdf}
- config_rollout.yaml (wiederverwendet, stage_to_shm:false)

## Naechster Schritt

Seg-Kern-MUSS-Pfad (Task 12-17) abgeschlossen. Offen: Task 18 (OPTIONAL generativ;
motiviert durch std-Gap-Rest + Rollout-Drift, aber nicht erzwungen), Task 19
(Seg-Strang-Zusammenfassung), 16b.10 (Optuna, optionales Seitengleis). Fuer die
schriftliche Arbeit: Referenz-Zahlen verifizieren, Captions uebernehmen.

---------------------------------------------------------------------------
## NACHTRAG 2026-07-25 — Komparator-Verifikation (To-do aus obiger Tabelle)

Alle 7 Referenzen gegen die Papers geprueft (arXiv-Abstracts/HTML-Volltexte).
Ergebnis: 5 verifiziert, 2 korrekturbeduerftig. Verifizierte Zahlen:

| Referenz | Datensatz/Task | Verifizierte Zahlen | Status |
|---|---|---|---|
| DINO-Foresight (arXiv 2412.11673) | **Cityscapes** (NICHT nuScenes), Seg-Forecasting | mIoU short 71.8 / mid 59.8 (ALL); Copy-Last 54.7/40.4; Oracle 77.0; SmoothL1 (beta=0.1) BESTAETIGT; mid-term autoregressiv | VERIFIZIERT; Venue "NeurIPS 2025" in unserer Tabelle NICHT belegt -> als arXiv zitieren bzw. Venue pruefen |
| LeWorldModel (arXiv 2603.19312) | JEPA from pixels, 2D/3D-Control | ~15M Params, 1 GPU; "reduziert tunable HYPERPARAMETER von sechs auf einen" | VERIFIZIERT — ABER PRAEZISIEREN: LeWM reduziert Hyperparameter 6->1, wir Loss-TERME 6->2; Analogie im Text sauber trennen |
| DINO-WM (arXiv 2411.04983) | Planung auf DINOv2-Patches, 6 Control-Umgebungen | 224x224-Inputs; Patch-Zahl im Paper NICHT beziffert | **KORREKTUR: unsere "196 Patches/Frame" ist unbelegt** — DINOv2 nutzt Patch-14 -> bei 224px waeren es 256; Zahl aus der Tabelle STREICHEN, ersetzen durch "DINOv2-Patch-Tokens (N im Paper nicht beziffert)". Kein mIoU-Eval. |
| OccWorld (arXiv 2311.16038) | Occ3D-nuScenes, 4D-Occupancy-Forecasting | mIoU 25.78 (1s) / 15.14 (2s) / 10.51 (3s), avg 17.14; IoU 34.63/25.07/20.18; Tokenizer-Rekonstruktion 66.38 mIoU; GPT-artig autoregressiv | VERIFIZIERT |
| BEVWorld (arXiv 2407.05679) | nuScenes + CARLA, multimodale Generierung | Diffusion NICHT-autoregressiv (alle Zukunfts-Tokens in einem Schritt); nuScenes FID 22.85 (1s) / 37.37 (3s); Lidar-Chamfer 0.44/0.73; Downstream-Detection +8.4 NDS / +13.4 mAP durch Pretraining | VERIFIZIERT |
| FIERY (arXiv 2104.10490) | nuScenes, BEV-Instanz-Future-Prediction | 2.0s, short range 30x30m: IoU 59.4 / VPQ 50.2 (Static-Baseline 47.9/43.1); probabilistisch via PRESENT-/FUTURE-Verteilung (diag. Gauss) + KL | VERIFIZIERT — ZUSATZBEFUND: FIERYs Mechanismus (Future-Vert. sieht Labels, KL zieht Present-Vert. heran) IST exakt unser B1-CVAE-Muster -> zusaetzlicher methodischer Anker fuer Task 18/B1 |
| LAW (arXiv 2406.08481) | nuScenes/NAVSIM (open-loop), CARLA (closed-loop) | Aux-Task "Zukunfts-Latent aus Features+Ego-Trajektorie" bestaetigt; SOTA-Claim; exakte L2-/Collision-Zahlen nicht im Abstract | QUALITATIV VERIFIZIERT — bei Zitat im Text Zahlen aus Paper-Tabellen ziehen |

QUERVERGLEICH (der eine legitime quantitative Anker): DINO-Foresight ist unser
Protokoll-Zwilling (frozen Features + frozen Decoder + SmoothL1), aber auf
Cityscapes-BILD-Features statt nuScenes-BEV — direkte mIoU-Werte sind daher
NICHT vergleichbar. Vergleichbar ist die STRUKTUR: deren Gewinn ueber Copy-Last
short-term +17.1 mIoU (71.8 vs 54.7) vs. unser +16.2 (0.6946 vs 0.5326) —
gleiche Groessenordnung, gleiche Charakteristik (Modell schliesst ~2/3 der
Luecke zum Oracle nicht — auch dort bleibt eine aleatorische Decke: Oracle 77.0
vs Forecast 71.8). Diese Parallele stuetzt die Decken-Interpretation (Task 19).

FUER DIE SCHRIFTLICHE ARBEIT: (1) "196 Patches" ueberall streichen (auch
WORLDMODEL_REFERENZEN.md korrigiert); (2) LeWM-Analogie praezisieren
(Hyperparameter vs Loss-Terme); (3) DINO-Foresight-Venue pruefen; (4) LAW-
Zahlen bei Bedarf aus dem Paper-PDF ziehen.
