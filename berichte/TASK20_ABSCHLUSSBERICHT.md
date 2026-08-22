# Task 20 — Det-Strang: Transfer-Studie (Abschlussbericht, Stand 30.07.)

Datum: 2026-07-26 bis 30. Aufbau als TRANSFER-STUDIE, nicht als Wiederholung
des Seg-Programms: getestet wird, ob die Seg-Befunde (16b-19) auf Det-Latents
generalisieren. VOLLSTAENDIG: der Seed-43-Rauschboden (20.3, unten) entscheidet
Befund 2 — "from six to two" gilt auf Detection NICHT (Delta +0.018 mAP fuer
die 6-Term-Loss, 4x Rauschboden).

## 20-Vorstufe: Det-Latent-Extraktion (26.07.)

Docker bevfusion:correct + latent_saver-Hook (Task-12-Rezept, det-Config
transfusion/secfpn/camera+lidar/swint_v0p075/convfuser + bevfusion-det.pth).
val 6019 + train 28130 Latents (1,256,180,180) fp16, ~16.6MB/Frame, gesamt
~560G -> BeeGFS lrt81-vima/latents_det/{val,train} (EIGENES Top-Level-Dir,
latents/ gehoert thlu). Train via Rolling-Sync (Puffer <25G). Gates: std-Faktor
Det/Seg 2.65 gemessen (Doku 2.6), Modell-Sanity mAP 0.686/NDS 0.715 =
Referenzwerte. Tempo ~4.2 f/s auf TITAN RTX.

## 20.0 Infrastruktur

latent_scale-Normalisierung (0.3641, gemessen 50+50 Samples: seg-std 0.2687 /
det-std 0.7379) beim Laden -> SmoothL1-Arbeitspunkt beta/sigma identisch zum
Seg-Strang; Default 1.0 = bit-identisch. config_det_base: grid 45 (6075
Tokens), Modell 6.057M (nur +3k durch groessere PE). Det-Val liefert 5743
Fenster = EXAKT die Seg-Zahl (gleiche Szenenstruktur). Det-Packs (train 467G +
val 100G) idempotent auf BeeGFS (latents_det_packed_fp16/).

## 20.1 Messmaschine: Docker-Injection statt Decoder-Nachbau

Der Det-Kopf (TransFusion) + nuScenes-devkit werden NICHT nachgebaut (Risiko,
Cluster-Installationen); stattdessen Latent-INJECTION-Hook im lokalen Docker
(Gegenstueck zum Saver, Env LOAD_BEV_LATENTS/LATENT_LOAD_DIR, Default aus):
tools/test.py bewertet injizierte Latents mit der validierten Original-Kette.
- ORAKEL-GATE A: reale Latents -> mAP 0.6858/NDS 0.7146 EXAKT reproduziert.
- GATE B: Normalisierungs-Roundtrip (x0.3641 -> fp16 -> /0.3641) -> identisch
  bis zur 4. Stelle -> Normalisierung metrisch verlustfrei.
- Cluster-Dumper eval_dump_latents.py: Vorhersagen als npy (Roh-Skala fp16,
  Format injection-kompatibel), 5743 Vorhersagen + 276 Real-Fill.
- LUECKEN-KONVENTION: devkit bewertet alle 6019 Samples; die ersten 3 Frames
  je Szene (276 = 4.6%) haben keinen Kontext und werden bei ALLEN Varianten
  inkl. Persistenz mit dem realen Latent gefuellt (hebt Absolutwerte leicht,
  kuerzt sich im Vergleich).
- PROTOKOLL-ABWEICHUNG zu Seg: Checkpoint-Selektion via Val-Loss (kein
  Cluster-seitiger Decode); Snapshots alle 10 Ep fuer nachgelagerte
  Metrik-Kurven.
- PERSISTENZ-BASELINE lokal via Symlinks (Vorhersage fuer Frame i = reales
  Latent i-1, i>=3).

## 20.2 Transfer-Test "from six to two" (Job 124921, 2 GPU, ~27h)

Zwei Trainings from scratch (50 Ep, Val-Loss-Patience, batch 8, kein OOM,
~30 min/Epoche = ~3x Seg wegen 2x Datenvolumen + 3.9x Attention; Loader-bound
data% ~55-65):

| Variante (Full-Val 6019, Injection)   | mAP    | NDS    |
|---------------------------------------|-------:|-------:|
| Persistenz                            | 0.1696 | 0.3908 |
| det_minimal (smooth_l1 + lambda_std)  | 0.3264 | 0.4662 |
| det_baseline (6-Term)                 | 0.3438 | 0.4761 |
| Orakel (frozen-Head-Decke)            | 0.6858 | 0.7146 |

BEFUND 1 — HAUPTERGEBNIS: Beide World-Model-Konfigs VERDOPPELN die
Persistenz-mAP (+0.157/+0.174). Das Modell lernt echte Objektdynamik; der
Det-Korridor (Persistenz 25% des Orakels) macht den Dynamik-Beitrag viel
sichtbarer als der Seg-Strang (dort Persistenz bereits 77% des Modells).
Die METHODIK generalisiert vollstaendig (gleiche Architektur, gleiche
Configs, ein Code-Pfad).

BEFUND 2 — "from six to two" gilt nur TEILWEISE: minimal erreicht ~95% der
6-Term-Baseline, liegt aber darunter (d = -0.0174 mAP / -0.0099 NDS) —
anders als Seg (+0.0026 n.s.). EINSCHRAENKUNG: Det-Rauschboden noch offen
(keine Seed-Studie; Detection-Metriken tendenziell rauschiger als mIoU).
Seed-43-Replikat beider Konfigs laeuft (Job 125355) -> Nachtrag. Ehrliche
Zwischenlesart: die Zusatzterme (cos/mean/ssim) DEUTEN auf Det einen kleinen
positiven Beitrag an, den sie auf Seg nicht hatten — plausibel wegen der
anderen Latent-Statistik (schwerere Tails, std-Faktor 2.65).

## 20.3 NACHTRAG Seed-43-Rauschboden (30.07., Job 125355) — BEFUND 2 ENTSCHIEDEN

Zweck: der Abstand minimal vs. 6-Term (-0.0174 mAP) war ohne Rauschboden nicht
interpretierbar. Beide Konfigs daher IDENTISCH nochmal from scratch mit Seed 43
(50 Ep, gleiche Configs bis auf training.seed), Dump + Injection wie in 20.1.

| Konfig                | Seed 42 mAP/NDS | Seed 43 mAP/NDS | Seed-Streuung |
|-----------------------|----------------:|----------------:|--------------:|
| det_minimal (2 Terme) |  0.3264/0.4662  |  0.3220/0.4622  | 0.0044/0.0040 |
| det_baseline (6 Term) |  0.3438/0.4761  |  0.3407/0.4741  | 0.0031/0.0020 |

DET-RAUSCHBODEN: max. Seed-Streuung 0.0044 mAP (0.0040 NDS) -> konservative
Schwelle 2x = 0.0088 mAP. Die Val-Losses der Seed-Paare waren praktisch
identisch (minimal 0.006987 vs. 0.006984; baseline 0.038817 vs. 0.038838),
d.h. das Rauschen entsteht NICHT im Training, sondern in der Uebersetzung
Latent -> Boxen (Detection-Metriken sind randempfindlich).

ENTSCHEIDUNG BEFUND 2: der 6-Term-Vorsprung ist BELEGT.
- Delta (baseline - minimal): Seed 42 +0.0174, Seed 43 +0.0187, Mittel +0.0180
  mAP (+0.0109 NDS) — gleiches Vorzeichen und gleiche Groesse in BEIDEN Seeds.
- +0.018 = 4x Rauschboden, 2x die konservative Schwelle -> signifikant.

FOLGERUNG (Kernsatz fuer die Thesis): "from six to two" ist KEIN allgemeines
Gesetz, sondern eine EIGENSCHAFT DER ZIELMETRIK. Auf Segmentierung sind
cos/mean/grad/ssim entbehrlich (16b.9: +0.0026 n.s.), auf Detection kosten
sie ~5% relativ an mAP, wenn man sie weglaesst. Plausible Ursache: die
Det-Latents haben schwerere Tails (std-Faktor 2.65) und die Boxen-Regression
haengt an feinen, lokalisierten Strukturen — genau dort greifen die
Struktur-Terme (grad/ssim) und die Richtungstreue (cos), waehrend die
flaechigen Seg-Masken davon nichts sehen. Die MINIMAL-Konfig bleibt fuer den
Seg-Strang die Empfehlung; fuer Detection ist die 6-Term-Loss vorzuziehen.
BEFUND 1 bleibt unberuehrt und wird durch Seed 43 bestaetigt (beide Seeds
verdoppeln die Persistenz-mAP).

Artefakte: predictions/task20/seed_noise.json (alle Zahlen + Schwellen),
Figur 20_det_korridor.* (jetzt mit Einzel-Seed-Punkten und Delta-Klammer),
Logs eval_det_{minimal,baseline}_s43.log, Checkpoints task20/det_*_s43/.

## Metrik-Kurzerklaerung (mAP/NDS)

nuScenes matched Boxen per ZENTRUMSABSTAND (0.5/1/2/4 m). AP = Flaeche unter
der Precision-Recall-Kurve; mAP = Mittel ueber 10 Klassen x 4 Toleranzen.
NDS = 0.5*mAP + 0.5*Qualitaet der Treffer (Translation/Groesse/Orientierung/
Geschwindigkeit/Attribut). Daher Persistenz mAP 0.17 aber NDS 0.39: bewegte
Objekte fallen aus den Toleranzen, die getroffenen statischen Boxen sind
aber praezise.

## Visualisierung (Figuren)

- 20_det_korridor.{png,pdf}: mAP/NDS-Balken Persistenz -> Modelle -> Orakel.
- 20_det_boxes_panel.{png,pdf}: 3 Szenen x [GT | Orakel | World-Model |
  Persistenz], LiDAR-BEV mit TransFusion-Boxen (Score >= 0.3) aus injizierten
  Latents (tools/visualize.py, gleicher Hook). Sichtbar: das Modell haelt
  die grossen/nahen Objekte, verliert aber kleine/ferne unter die Schwelle
  (konsistent mit mAP-Haelfte zum Orakel).

## Artefakte

- Daten: BeeGFS latents_det/{val,train} (roh), latents_det_packed_fp16/.
- Checkpoints: task20/{det_minimal,det_baseline}/phase2/ (+ Snapshots 10-40;
  _s43-Varianten nach Job 125355).
- Dumps/Evals: predictions/task20/dump_det_{minimal,baseline}/ (Cluster);
  lokal Logs eval_det_{minimal,baseline}.log, gateA/B, pers_baseline.
- Code: latent_scale (bev_dataset/-loader/train_linux), eval_dump_latents.py,
  Injection-Hook (bevfusion.py, Docker-Repo), render_20_boxes.py,
  render_20_korridor.py; Fix in Code/shm_staging.py (s.u.).
- Configs: config_det_{base,minimal,baseline}[_eval|_s43].yaml;
  sbatch_20_train[_s43].sh, sbatch_20_dump_{baseline,minimal}.sh.

## Lektionen

- shm_staging-Platzpruefung zaehlte bereits gestagte Splits doppelt — bei Seg
  (2x286G < 1TB) nie ausgeloest, bei Det (2x567G) sofort; Fix: gestagte
  Splits zaehlen nicht zum Bedarf (Job 124905 verloren, ~25min).
- Docker-Injection statt Metrik-Nachbau: 1 Tag statt 2-4, null Cluster-
  Installationen, Orakel-Gate als Beweis — Muster fuer alle kuenftigen
  "fremden Koepfe".
- 3x Epochenzeit bei Det ist loader-/bandbreitenbound (data% ~60), nicht
  GPU-bound; bei Serienlaeufen 1 Lauf/Node mit vollen CPUs.

## Offen

- Optional: Snapshot-Metrik-Kurve (Checkpoints 10-40 durch den Eval-Zyklus).
- Optional: Panel-Szenen mit schnellen Fahrzeugen fuer den "klebende
  Boxen"-Effekt der Persistenz.
