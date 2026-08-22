# Task 19 — Seg-Strang: Abschluss & Zusammenfassung

Datum: 2026-07-25. Konsolidierung des gesamten Seg-Strangs (Tasks 12-18).
Kein neues Compute; Quellen: vorhandene Full-Val-JSONs (predictions/task16*/
task18/), Master-Tabelle predictions/task19/master_table.csv (collect_task19.py).
Parallel laeuft der optionale Flow-Kalibrierungs-Nachlauf (B2.1, Job 124563);
sein Ergebnis wird als datierter Nachtrag ergaenzt und aendert die Kernaussagen
nicht.

## 1. Beste Konfiguration (Empfehlung fuer die Thesis)

**SmoothL1-Minimal**: recon_loss=smooth_l1 (Gewicht 1.0) + lambda_std=1.0,
alle anderen Terme 0. 4-Layer-Pre-LN-Transformer, d_model=256, 3 Frames,
gated skip, 6.05M Parameter. **Full-Val mIoU 0.6946** (5743 Fenster),
std-Gap single-step ~1.4%.

Begruendung: (a) erreicht die 6-Term-Baseline (0.6920) bei einem Drittel der
Regler ("from six to two", 16b.9); (b) SmoothL1 schliesst den std-Gap ohne
mIoU-Kosten (16b.7); (c) der lambda_std-Term liefert Varianz-Stabilitaet auch
im Rollout (std-Ratio 0.99-1.01 ueber k=1..4 — Befund 18/B2); (d) unabhaengige
Bestaetigung der Minimal-Loss-Philosophie durch LeWorldModel (arXiv 2603.19312:
2 Terme = Next-Embedding-Prediction + Gauss-Regularisierer).

## 2. Master-Ergebnistabelle (Full-Val, vs off=0.6946, Schwelle 0.014)

Vollstaendig als CSV: predictions/task19/master_table.csv. Kurzfassung:

| Gruppe    | Variante                          | mIoU   | d vs off | Verdikt |
|-----------|-----------------------------------|-------:|---------:|---------|
| Baseline  | Persistenz naiv                   | 0.5326 | --       | --      |
| Baseline  | Persistenz ego-gewarpt            | 0.6286 | --       | --      |
| Headline  | 6-Term-Baseline                   | 0.6920 | -0.0026  | n.s.    |
| Headline  | **SmoothL1-Minimal (off)**        | 0.6946 | --       | SIEGER  |
| 16d       | Ego gate / FiLM a / Token / FiLM s| 0.6910 / 0.6175 / 0.6164 / 0.6094 | -0.004 n.s. / je ~-0.08 | sig. schlechter (ausser gate) |
| 16e       | 4 Frames / EMA                    | 0.6915 / 0.6945 | -0.003 / -0.000 | n.s.  |
| 16f       | Task-Loss 0.1/0.5/1.0             | 0.6788 / 0.6717 / 0.6688 | -0.016..-0.026 | sig. schlechter |
| 16f       | 8 Layer / hartes Residual         | 0.6927 / 0.6936 | -0.002 / -0.001 | n.s. |
| 18        | CVAE b=0.001/0.01/0.1 (z=mean)    | 0.6875-0.6883 | ~-0.007 | n.s. (Guard ok) |
| 18        | Flow Matching (mean)              | 0.6946 | 0        | = Backbone |

KEIN getesteter Hebel schlaegt die Minimal-Konfig — sechs unabhaengige
Richtungen (Loss, Konditionierung, Kontext, Averaging, Task-Kopplung,
Kapazitaet/Struktur, Generativ) bestaetigen die Decke. Figur: 19_all_levers.*

## 3. Decken-Analyse: warum 0.69 die Grenze dieses Setups ist

KEIN MESSFEHLER — Audit: Same-Seed-Replikation 2x bestaetigt (0.0001-genau),
physikalisch plausible Baseline-Ordnung, monotone Dosis-Wirkungs-Effekte wo
Effekte existieren, Messkette empfindlich genug fuer Verschlechterungen.

URSACHEN (per-Klasse-Zerlegung der Headline, Figur 19_per_class.*):

| Klasse        | mIoU  | Charakter        |
|---------------|------:|------------------|
| drivable_area | 0.898 | gross, statisch  |
| walkway       | 0.734 | gross, statisch  |
| carpark_area  | 0.688 | mittel, statisch |
| divider       | 0.640 | duenn (Linien)   |
| ped_crossing  | 0.630 | klein, duenn     |
| stop_line     | 0.555 | sehr klein, duenn|

1. ALEATORISCHE UNSICHERHEIT: die Zukunft ist bei 2 Hz teilweise echt
   unvorhersagbar; dieser Fehleranteil ist im Problem, nicht im Modell.
   Anker: decode(real_t) vs decode(real_{t+1}) ueberlappt nur zu 0.53
   (naive Persistenz) — die Szene AENDERT sich stark zwischen Frames.
2. IoU-RANDEMPFINDLICHKEIT DUENNER STRUKTUREN: bei 1-2 Zellen breiten
   Linien kostet 1 Pixel Versatz 30-50% IoU; der Restfehler konzentriert
   sich exakt dort (stop_line 0.56 vs drivable 0.90).
3. DATEN-/FEATURE-DECKE: 28k Trainingssequenzen, eingefrorene BEVFusion-
   Latents. Mehr Kapazitaet lernt daraus nichts Neues (8 Layer: n.s.).

Feld-Einordnung: kurzfristige semantische Zukunftsvorhersage schlaegt ueberall
nur moderat ueber warped-copy-Baselines (unser +0.16 naiv / +0.07 ego-gewarpt
ist feldtypisch); echte Gewinne kommen aus Features/Daten (DINO-Foresight)
oder liegen in anderen Groessen (Diversitaet/Coverage — s. Task 18).

## 4. Diskussionspunkte (Thesis)

- std-Gap: von ~9.1% (MSE-Baseline) auf ~1.4% (SmoothL1) geschlossen — OHNE
  generative Methoden (17_stdgap_beforeafter). Der lambda_std-Term liefert
  zusaetzlich Rollout-Varianz-Stabilitaet (18/B2-Rollout: det std 0.99-1.01),
  d.h. die Stabilitaet, die generative Weltmodelle als Vorteil anfuehren.
- Generativ (Task 18): CVAE-Sonde funktional deterministisch (Negativ-
  Baseline); Flow Matching liefert echte Multimodalitaet (37x Diversitaet,
  Best-of-8 holt -0.033 des Sample-Malus zurueck = Coverage), Genauigkeit
  bleibt beim deterministischen Betriebspunkt (Perception-Distortion-
  Tradeoff + Kalibrierungsrest -0.011).
- Rollout (16c/16d-Vorstufe): Modell schlaegt beide Persistenzen bei jedem k;
  Vorsprung vs ego-gewarpt (+0.07..+0.035) = ehrlicher Dynamik-Beitrag.

## 5. Figuren-Inventar (final, alle visualizations/)

| Figur | Inhalt | Kapitel |
|---|---|---|
| 16b9_tradeoff / 16b9_dose_response | Loss-Engineering: Trade-off + Sweeps | Loss |
| 17_miou_over_epochs | Trainingsverlauf | Methode |
| 17_stdgap_beforeafter | std-Gap 9.1%->1.4% (Kernargument Loss) | Loss |
| 17_masks_overview | Qualitative Masken (staerkstes Bild) | Ergebnis |
| 17_gate_alpha | Gate kopiert statisch / sagt dynamisch | Architektur |
| 17_filmstrip | 7-Frame-Rollout qualitativ | Rollout |
| 16c_rollout | Rollout-Kurven beider Konfigs | Rollout |
| 16d_ego_persistence | ego-gewarpte Persistenz-Baseline | Baselines |
| 16d_ego_conditioning | Ego-Ablation (nur gate neutral) | Ablation |
| 16e_miou_lever / 16f_miou_lever2 | Kontext/EMA + Task-Loss/Kapazitaet/Residual | Ablation |
| 18_vae_panel_* / 18_flow_rollout | CVAE- vs Flow-Samples + Rollout-Kalibrierung | Generativ |
| 19_per_class | Decken-Zerlegung je Klasse | Diskussion |
| 19_all_levers | Gesamt-Hebel-Figur ("Money-Figur") | Zusammenfassung |

Konsistenz: alle Ablations-Figuren nutzen off=0.6946 + Signifikanzband 0.014,
Okabe-Ito-Palette, schwarzer off-Balken. Referenztabelle (Task 17) bleibt
methodisches Geruest — Komparator-Zahlen vor Abgabe gegen die Papers pruefen.

## 6. Entscheidung Det-Strang (Task 20)

EMPFEHLUNG: FUTURE WORK. Begruendung: 2-4 Wochen Aufwand (Latent-Extraktion
~34k Frames [512,180,180], neue Decode-/mAP-Infra, komplette Pipeline-
Wiederholung) fuer methodische Wiederholung ohne neue Erkenntnisart; die
dokumentierte Latent-Inkompatibilitaet (128^2 vs 180^2, andere Fuser-Gewichte,
std-Faktor ~2.6) ist als Begruendung im Text ausreichend. FINALE BESTAETIGUNG
DURCH DEN NUTZER STEHT AUS (haengt an Restzeit bis Abgabe).

## 7. Offene Punkte nach Task 19

- B2.1-Nachtrag (Job 124563: 30-Ep-Kopf + Skalen-Sweep) -> in TASK18_B2-
  Bericht und ggf. hier nachtragen.
- Komparator-Zahlen der Referenztabelle verifizieren (Task-17-Auftrag).
- Det-Strang-Entscheidung final bestaetigen (Abschnitt 6).

## Artefakte

- predictions/task19/master_table.csv (+ headline_*_fullval.json lokal)
- visualizations/19_per_class.*, 19_all_levers.*
- Scripts: collect_task19.py, render_19.py (beide idempotent, lokal)

---------------------------------------------------------------------------
## NACHTRAG 2026-07-25 — B2.1-Ergebnis eingetroffen (Job 124563)

Der parallel gelaufene Kalibrierungs-Nachlauf ist abgeschlossen (Details:
TASK18_B2_FLOW_ABSCHLUSSBERICHT.md, Nachtrag). Kurzfassung fuer den
Strang-Abschluss: post-hoc Residual-Skalierung s=0.5 bringt die Sample-
Varianz auf 1.11 und hebt Best-of-8 auf 0.6998 = +0.010 UEBER den
deterministischen Mittelwert (Coverage-Ergebnis, Orakel-Auswahl — kein
einsetzbarer Praediktor). Damit ist der offene Punkt "B2.1-Nachtrag" aus
Abschnitt 7 erledigt; an der besten Konfiguration (Abschnitt 1) und der
Decken-Aussage (Abschnitt 3) aendert sich nichts. Der Generativ-Absatz in
Abschnitt 4 ist um den Zusatz "Kalibrierung post-hoc loesbar, Coverage-
Gewinn belegt" zu lesen. Neue Figur: 18_flow_calibration.*

---------------------------------------------------------------------------
## NACHTRAG 2026-07-30 — Det-Strang durchgefuehrt: Geltungsbereich der
## Minimal-Loss-Empfehlung (Task 20 komplett, inkl. Seed-43-Rauschboden)

Abschnitt 6 ist UEBERHOLT: der Det-Strang wurde entgegen der Future-Work-
Empfehlung DOCH durchgefuehrt (Nutzer-Entscheid; Injection-Protokoll statt
Decoder-Nachbau machte ihn mit ~4 Tagen viel billiger als die dort
geschaetzten 2-4 Wochen). Ergebnisse: TASK20_ABSCHLUSSBERICHT.md.

KONSEQUENZ FUER DIESEN BERICHT — der Kernbefund bekommt eine explizite
GELTUNGSBEREICHS-GRENZE, aendert sich fuer Seg aber NICHT:

1. Abschnitt 1 (beste Konfiguration) gilt unveraendert FUER DEN SEG-STRANG:
   SmoothL1-Minimal, mIoU 0.6946. "from six to two" ist auf Seg belegt
   (+0.0026 n.s.).
2. NEU: "from six to two" ist KEINE allgemeine Regel, sondern eine
   EIGENSCHAFT DER ZIELMETRIK. Auf Detection (Task 20, Full-Val 6019,
   Injection in frozen TransFusion-Kopf) liegt die 6-Term-Baseline BELEGT
   ueber der Minimal-Konfig: Delta +0.0174 (Seed 42) / +0.0187 (Seed 43) /
   Mittel +0.0180 mAP bei einem gemessenen Seed-Rauschboden von nur 0.0044
   mAP (= 4x Rauschboden, gleiches Vorzeichen in beiden Seeds).
   Mechanistische Lesart: Det-Latents haben schwerere Tails (std-Faktor
   ~2.65) und die Boxen-Regression haengt an feinen, lokalisierten
   Strukturen — dort tragen grad/ssim/cos, die auf flaechigen Seg-Masken
   nichts beitragen.
3. EMPFEHLUNG (differenziert statt universell):
   | Zielmetrik | Loss-Konfig | Beleg |
   |---|---|---|
   | Segmentierung (mIoU) | SmoothL1-Minimal (2 Terme) | 16b.9: 0.6946, +0.0026 n.s. |
   | Detection (mAP/NDS)  | 6-Term-Baseline            | 20.3: +0.0180 mAP = 4x Rauschboden |
4. Zur "Money-Figur" 19_all_levers: die Aussage "kein Hebel schlaegt die
   Minimal-Konfig" gilt fuer den SEG-Strang; in der Thesis mit Fussnote auf
   Task 20.3 versehen, damit die beiden Kernsaetze nicht als Widerspruch
   gelesen werden.
5. Damit ist auch der offene Punkt "Det-Strang-Entscheidung final
   bestaetigen" (Abschnitt 7) erledigt.

Artefakte: predictions/task20/seed_noise.json, Figur 20_det_korridor.*
(Seed-Punkte + Delta-Klammer), TASK20_ABSCHLUSSBERICHT.md Abschnitt 20.3.

## NACHTRAG 22.08.2026 — Korrektur drivable_area-Rundung
Die Tabelle oben nennt drivable_area 0.898; das Primaer-Artefakt
predictions/task19/headline_minimal_fullval.json (per_class) traegt
0.8973 -> kaufmaennisch gerundet 0.897. Der Tabellenwert war ein
Uebertragungsfehler. Massgeblich: 0.897 (so auch Figur 19_per_class
seit der expliziten ROUND_HALF_UP-Rundung). Fliesstext der Thesis
entsprechend auf 0,897 korrigieren.
