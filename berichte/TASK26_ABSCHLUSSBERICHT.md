# Task 26 — Ressourcenbedarf (Abschlussbericht, 07.08.)

Schliesst Bearbeitungspunkt 4 der Aufgabenstellung (Speicherbedarf +
Inferenzzeit fehlten; Trainingsaufwand war verstreut). KEINE neuen
Trainings — nur Inferenz-Messungen + Log-Auswertung.

## Protokoll
- Hardware: NVIDIA TITAN RTX (24GB), Treiber 535.230.02, torch 2.1.2+cu121.
- Inferenzzeit: 1 Vorhersageschritt (3 Kontext-Latents -> 1 Latent), Batch 1,
  fp16-autocast wie im Betrieb; 30 Warmup + 300 Messiterationen,
  torch.cuda.synchronize() vor jeder Zeitnahme -> mean +- std.
- Det: latent_scale=0.3641-Pfad (Eingaben betriebsnah skaliert).
- Flow: steps=1 (Befund 18/B2: identisch zu steps=10) — Backbone-Forward UND
  +1 Euler-Schritt getrennt und gesamt.
- Speicher: torch.cuda.max_memory_allocated() nach Reset, Batch 1.
- BEVFusion-Encoder (Variante 7): Kamera+LiDAR->Latent, im Docker gemessen
  (env-gesteuerter Timing-Hook um Encoder+Fuser, 20 Warmup + 300 Iterationen,
  Hook nach Messung zurueckgebaut).
- GPU exklusiv (nvidia-smi geprueft, nur ~3GB Grundlast Anzeige); Taktverhalten
  nicht fixiert (Boost aktiv) — Absolutwerte daher leicht optimistisch, die
  Verhaeltnisse sind robust.

## Ergebnisse — statisch + Inferenz (Full-Tabelle: ressourcen.csv)

| Variante | Params | ckpt [MB] | Gew. fp16 [MB] | Peak-VRAM [MB] | Forward [ms] |
|---|---:|---:|---:|---:|---:|
| Seg-WM minimal (Betriebspunkt) | 6.054M | 69.4 | 11.5 | 147 | 4.41 +- 0.32 |
| Seg-WM 8 Layer (16f) | 9.213M | 105.6 | 17.6 | 165 | 7.10 +- 0.04 |
| Det-WM 6-Term (Betriebspunkt) | 6.057M | 69.4 | 11.6 | 249 | 11.45 +- 0.07 |
| Flow-Kopf, det. Forward | 12.572M | 97.8 | 24.0 | 266 | 4.48 +- 0.02 |
| Flow-Kopf, +1 Euler (gesamt) | " | " | " | " | 14.02 (Kopf +9.54) |
| CVAE-Kopf b=0.01 | 6.424M | ~73 | 12.3 | 149 | 4.58 +- 0.01 |
| Persistenz | 0 | 0 | 0 | 0 | ~0 |
| BEVFusion-Encoder (Kontext) | (SwinT+SparseConv+Fuser) | — | — | — | 172.78 +- 1.16 |

## Trainingsaufwand (aus Logs, trainingsaufwand.csv)

| Lauf | Hardware | s/Epoche | Epochen | GPU-h |
|---|---|---:|---:|---:|
| Seg-Baseline (Task 13) | TITAN RTX | 817 | 50 | 11.3 |
| Seg-WM minimal Headline (123424) | A100-80GB | 893 | 50 | 12.4 |
| Det-WM 6-Term (20.2, 124921) | A100-80GB | 1800 | 50 | 25.0 |
| Seg-Kopf-Adaptation (23) | TITAN RTX | 468 | 8 | 1.0 |
| Det-Kopf-Adaptation v1 (24) | TITAN RTX | 1500 | 6 | 2.5 |
| Det-Kopf-Adaptation v2 (25) | TITAN RTX | 1300 | 12 | 4.3 |

## Befund (3-5 Saetze)

Der World-Model-Vorhersageschritt kostet auf einer TITAN RTX 4.4 ms (Seg)
bzw. 11.5 ms (Det) — also unter 1% bzw. ~2.3% des 500-ms-nuScenes-Zyklus
(2 Hz); das Modell ist mit >40x Puffer echtzeitfaehig und entschaerft
Limitation 5.9 direkt. Der generative Flow-Kopf addiert ~9.5 ms je
Euler-Schritt (Gesamt 14 ms, steps=1 genuegt) — die generative Erweiterung
ist praktisch gratis. Die 8-Layer-Variante kostet +60% Rechenzeit (7.1 vs
4.4 ms) fuer statistisch nicht signifikanten Qualitaetsgewinn (16f) — der
Verzicht auf Tiefe ist auch ressourcenseitig gerechtfertigt. Zur Einordnung:
der eingefrorene BEVFusion-Encoder selbst braucht 172.8 ms — das World-Model
ist damit ein Aufschlag von ~2.5% (Seg) auf die Wahrnehmung, die es
weiterdenkt. Speicher ist unkritisch (Peak <270 MB, Gewichte fp16 <25 MB;
Embedded-tauglich).

## Artefakte
- predictions/task26_ressourcen/ressourcen.json (alle Rohwerte inkl. Batch-8-
  Durchsatz), ressourcen.csv (flache Tabelle), trainingsaufwand.csv
- visualizations/ressourcen_inferenzzeit.{png,pdf} (Balkenplot vs. 0.5-s-Zyklus)
- Skripte: archiv/tools/bench_resources.py (WM-Varianten),
  scripts_render/render_ressourcen.py (Plot); Encoder via temporaerem
  Timing-Hook in bevfusion.py (zurueckgebaut, Backup bevfusion_hostbackup.py).

## Lektion
- Docker-Bind-Mount /bevfusion: Edits via `docker exec python -c` persistieren
  NICHT zuverlaessig zwischen exec-Aufrufen -> die Host-Datei
  (~/bevfusion/bevfusion-main/...) direkt editieren, dann greift der Container.
- Encoder-Timing-Hook feuert erst nach dem Map-Rasterisierungs-Preprocessing
  des ersten Val-Samples (LoadBEVSegmentation) — kein Hang, nur Vorlauf.

## NACHTRAG 09.08. — Pareto-Figur (26.5)
Pareto Qualitaet vs. Ressourcenbedarf (Seg): die Minimal-Konfiguration ist
Pareto-optimal — hoechste Guete (0.6946, Full-Val/Proxy-Skala) bei geringster Latenz (4.4 ms) und
kleinstem VRAM (147 MB); mehr Tiefe (8 Layer, 7.1 ms) und der generative
Flow-Kopf (14 ms) kosten Rechenzeit ohne Punktvorhersage-Gewinn. Konsistenz-
Check: minimal/8-Layer/CVAE aus Full-Val 5743, Flow aus 300-Subset (in der
Figur mit * markiert). Figur visualizations/pareto.{pdf,png},
Skript scripts_render/render_pareto.py.

## NACHTRAG 10.08. — Typografie-Anhebung (6 Figuren)
Fuer den A4-Satz wurden die Beschriftungen von 17_filmstrip,
18_vae_panel_0.01 (+ beta-Anhang-Panels), 20_det_korridor,
21_seg_gt_korridor, 21_error_decomposition und pareto vergroessert:
gemeinsames Modul scripts_render/_thesis_style.py (Canvas x0.80 bei
Fonts x1.15 + angehobene rcParams -> effektiv ~1.44x groessere Schrift,
Seitenverhaeltnisse und Dateinamen unveraendert, bbox_inches='tight').
Wo die groessere Schrift Kollisionen erzeugte, wurden Tick-Labels
gestrafft (Details stehen in Caption/Text) und Titel umbrochen;
Messwerte/Farben nachweislich unveraendert (Zahlen-Diff alt vs. neu).

## NACHTRAG 11.08. — LiDAR-Panels invertiert (weisser Grund)
Die vier Detektions-Panels (20_det_boxes_panel, 24_det_boxes_panel,
24_det_boxes_peds, 25_det_boxes_versions) zeigen die Punktwolke jetzt
DUNKEL AUF WEISS (Druck-/Tonerfreundlich, konsistent zu den uebrigen
hell-grundigen Figuren). Umsetzung: invert_lidar() in
scripts_render/_thesis_style.py — selektive Inversion ueber die
Farbsaettigung: unbunte Pixel (Punktwolke/Hintergrund) werden invertiert
(Faktor 0.82, damit dichte Bereiche dunkelgrau statt tiefschwarz bleiben),
gesaettigte Pixel (Boxen) behalten ihre Farbcodierung und werden nur leicht
abgedunkelt (0.88) — GT/Vorhersage/Varianten-Zuordnung und Klassenfarben
sind unveraendert. Gleiche Dateinamen, gleiche Szenen (Auswahl nach
maximaler Differenz weiterhin dokumentiert), Typografie auf das Niveau des
Fonts-Auftrags gehoben, Titel/Spaltenlabels umbrochen. Kein Docker-Rerun
noetig (Inversion arbeitet auf den vorhandenen Renders).

## NACHTRAG 11.08. (2) — Typografie zentralisiert + Filmstrip umgebaut
- scripts_render/thesis_style.py ist jetzt die EINE Quelle der Typografie
  (apply_style/S/FIG + invert_lidar); ALLE Render-Skripte importieren sie,
  auch die bisher fehlenden (render_thesis_synthesis, render_legacy_figs,
  render_19). Kuenftige Figuren erben das Niveau automatisch.
- Neu auf Referenzniveau gerendert: korridor_overview, rueckholquote,
  17_stdgap_beforeafter, 16c_rollout, 16d_ego_persistence,
  17_miou_over_epochs, 19_all_levers (Legende jetzt AUSSERHALB der Achse,
  verdeckte vorher die unteren Balken), 18_vae_panel_flow_s10.
- FILMSTRIP UMGEBAUT: statt einer 4x7-Figur zwei Dateien
  17_filmstrip_statisch.pdf und 17_filmstrip_dynamisch.pdf mit je 5 Spalten
  (Kontext F3 + k=1..4) und 2 Zeilen; mIoU je Schritt als Panel-Titel.
  Panelbreite ~+45%. Szenenauswahl (Gate-alpha-Extreme) unveraendert.
  Die alte 17_filmstrip.{pdf,png} bleibt vorerst liegen (nicht geloescht).
- BONUS: In der Fehlerzerlegung wird der Det-WM-Wert jetzt als UNGERUNDETES
  Seed-Mittel (0.3438/0.3407) uebergeben -> Gap 0.3436 (statt 0.3436 aus
  gerundetem 0.3422; Anzeige bleibt 0.344). HINWEIS: die im Text genannten
  0.342 entsprechen dem Gap gegen den SEED-42-Wert 0.3438 (so auch in
  rueckholquote.pdf) — Figur (Seed-Mittel) und Text (Seed 42) verwenden
  also unterschiedliche Bezugswerte; bitte im Text EINE Variante waehlen.

## NACHTRAG 12.08. — Redaktioneller Figuren-Durchgang (Abgabefassung)
Umsetzung des Betreuer-Feedbacks ("einheitliche Schriftgroessen, keine
Ueberlappungen, Seitenbreite einhalten, Legenden ergaenzen"):
- TYPOGRAFIE ZENTRAL: scripts_render/thesis_style.py neu — apply_style(width_cm)
  setzt rcParams so, dass bei der EINBAUBREITE ueberall dieselbe gedruckte
  Punktgroesse ankommt (Basis 9 pt, Ticks 8, Titel 10; Untergrenze 7 pt).
  Alle Figuren haben 17 cm natuerliche Breite (FIG() normiert, Seitenverhaeltnis
  bleibt); 19_all_levers wird mit width_cm=14 vorgehalten. pdf.fonttype=42
  (TrueType/CIDFontType2 eingebettet, kein Type-3) — verifiziert.
- UMLAUTE: alle ASCII-Umlaute in Anzeige-Strings ersetzt (ae/oe/ue -> ä/ö/ü);
  Labels, die aus DATEN-Artefakten stammen (master_table.csv), werden nur fuer
  die Anzeige ueber thesis_style.de() normiert — die CSV bleibt unveraendert.
- TERMINOLOGIE: "Aleatorik-Evidenz ..." -> "Zeitliche Veränderung bei
  Δt = 0,5 s: mIoU(real t, real t+1) = 0,53" (21_error_decomposition).
- KOLLISIONEN behoben: korridor_overview (Titel/Segment-Labels),
  25_head_comparison (eine gemeinsame Legende unter beiden Panels, Werte in
  hohen Balken innen), 21_error_decomposition (eine gemeinsame Fussnote),
  17_stdgap_beforeafter (Wert vs. ideal-Linie), Box-Panels 20/24/25
  (Spaltenlabels umbrochen, Panelabstaende eng, Suptitle entfernt — die
  Aussage steht in der LaTeX-Caption, die Spaltenkoepfe sind die Legende).
- ABWEICHUNG VOM AUFTRAG (bewusst, bitte pruefen): 17_filmstrip_statisch wird
  wie der dynamische auf 17 cm vorgehalten. Bei 10 cm Einbaubreite waeren die
  5 Panels nur ~2 cm breit und der Bildinhalt unlesbar -> Empfehlung, BEIDE
  Filmstrips mit >= 14 cm einzubinden.
- Alle 34 PDFs neu erzeugt; pdftotext-Scan: keine ASCII-Umlaute, kein
  "Signifikanz", keine "0.69-Decke", kein "Aleatorik". Die alte
  17_filmstrip.pdf (vor dem Split) wurde nicht neu erzeugt und kann entfallen.

## NACHTRAG 12.08. (2) — Bild-fuer-Bild-Durchgang
Alle zehn Befunde des Durchgangs behoben:
1. korridor_overview: Segment-Labels unter die Bahn versetzt (kein Ueberlapp
   mit "World-Model 0.342/0.590").
2. 16b9_dose_response: Legende + Band-Hinweis unter die Figur ausgelagert.
3. 17_filmstrip_statisch: bleibt bei width_cm=17 (jetzt auch so eingebunden).
4. 17_gate_alpha: Panel-Titel dreizeilig (mIoU / mean α / Tendenz).
5. 18_flow_calibration: beide Legenden unterhalb der Panels.
6. 21_error_decomposition: Titel entfernt — GENERELL gilt jetzt: KEIN
   suptitle und keine beschreibenden Achsen-Titel in den finalen Renders
   (14 suptitle + 13 beschreibende ax-Titel entfernt); nur kurze
   Panel-/Spaltenlabels bleiben, die Bildunterschrift kommt aus LaTeX.
7. 23_decoder_panel: URSACHE der Mini-Schrift war ein nicht normiertes
   figsize (10.5 in statt 6.69 in) — der Wrapper hatte Ausdruecke wie
   "3.6 * len(files)" nicht erfasst; ebenso in render_thesis.py behoben.
8./9. 16c_rollout und 18_flow_rollout: Legenden unter die Panels.
10. 20_det_boxes_panel: Spaltenlabels umbrochen, Zeilenlabels gekuerzt.
ZUSATZ (PDF-Groesse): Ursache war NICHT die Raster-DPI, sondern die mit
pdf.fonttype=42 vollstaendig eingebettete Schrift (~700 kB je Figur).
scripts_render/optimize_pdfs.sh subsettet die Fonts per Ghostscript
(/prepress, EmbedAllFonts) mit Text-Verifikation je Datei:
16 MB -> 2,9 MB (35/35 Dateien, Text und Rasterbilder erhalten).
Raster-Panels zusaetzlich auf dpi=200 begrenzt.

## NACHTRAG 12.08. (3) — Bilddurchgang, zweite Runde
- korridor_overview: Werte-Labels jetzt IMMER oberhalb der Bahn; liegen zwei
  Marker naeher als 0.08 (Seg: 0.590/0.629), rutscht das zweite eine Etage
  hoeher statt zu ueberlappen. "Detection" -> "Detektion" (auch in
  rueckholquote und 21_error_decomposition).
- 21_seg_gt_korridor: suptitle-Zeilen entfernt, rechter Paneltitel auf
  "je Klasse" gekuerzt; die Klassen-Legende sitzt jetzt unter den Panels
  (verdeckte sonst die Balken).
- 21_error_decomposition: Fusstext-Block ersatzlos entfernt.
- GENERELLE REGEL durchgesetzt und per grep verifiziert: KEIN suptitle,
  KEIN figtext/fig.text in irgendeinem Render-Skript. In-Bild-Text nur noch
  Achsen, Werte, Legenden und neutrale Paneltitel ("Mittel (6 Klassen)",
  "je Klasse", "Rollout-mIoU vs. Horizont", "Kontext F3", "k=1" ...).
- Alle 35 PDFs neu gerendert und per optimize_pdfs.sh subgesettet (2,9 MB).

## NACHTRAG 13.08. — Messwiederholung (Task 27.5)
Drei unabhaengige Messserien (je 30 Warmup/300 Iter., identisches Protokoll):
Mittelwerte stabil (Spannweite ueber die Serien < 0.35 ms je Variante;
ressourcen_reps_stat.json). Die Task-26-Zahlen sind damit als Mittel dreier
Serien abgesichert; Taktverhalten (Boost) bleibt als Kontext dokumentiert.

## NACHTRAG 14.08. — Finale Terminologie-Korrektur (Abgaberunde)
Globale Sprachregelung in ALLEN Figuren durchgesetzt: "Orakel" ->
"Real-Latent-Referenz" (kurz "real"); alle wertenden Segment-/Analyse-
Beschriftungen entfernt (korridor_overview ohne "Dynamik-Beitrag"/
"Forecasting-Gap"; 20_det_korridor nur noch "+0,018 mAP";
21_error_decomposition nur reine Werte ohne %-Zuschreibungen — konsistent
zur Caption "Metrikabstaende, keine kausalen Fehleranteile").
18_vae_panel_flow_s10: Spalte 2 korrekt als "deterministische Basis"
beschriftet (zeigt den frozen-Backbone-Forward, nicht CVAE z=mean);
CVAE-Panels eingedeutscht ("Mittelwert (z=mu)"). pareto: Flow-Punkt
komplett entfernt (Qualitaet des Basispfads + Ressourcen des Kopfes waren
methodisch schief; Ressourcen stehen in Tab. 5.2), "Sweet-Spot"/"per
Konstruktion" gestrichen, Achsen auf 9 ms/200 MB gestrafft.
rueckholquote: in der Thesis gestrichen, Skript bleibt (kein Re-Render
noetig). Abnahme-Scan: kein Orakel/Forecasting-Gap/Dynamik-Beitrag/
Sweet-Spot/per Konstruktion/d. Fehlers/CVAE z=mean in den PDFs; keine
Prozente in 21_error_decomposition.

## NACHTRAG 14.08. (abends) — Nachtrag 1 des Finale-Korrektur-Auftrags
Zwei Zahlenkorrekturen + Label-Reste (Punkte 10-13):
(10) 21_error_decomposition: Det-Panel zeigte das SEED-MITTEL 0.342 am
WM-Balken (Gap dann 0.344) — wirkte wie vertauscht. Jetzt laedt das Skript
die Werte aus predictions/task21/gt_eval.json + task20/seed_noise.json
(Referenz-Seed 42) und berechnet die Abstaende: WM 0.344, Abstand 0.342;
Seg unveraendert korrekt (0.590/0.040). Nichts mehr hart kodiert.
(11) 20_det_korridor: Balken/Labels auf Referenzwerte Seed 42 umgestellt
(6-Term 0.344, minimal 0.326; NDS 0.476/0.466), Einzel-Seeds bleiben als
Punkte; Klammer weiterhin "+0,018 mAP" (berichteter Mittel-Delta).
korridor_overview konsistent mitgezogen (0.3422 -> 0.3438).
(12) 19_all_levers: "Headline" -> "Referenzkonfiguration", "= OFF"
gestrichen, "(F3 kopieren)" -> "(letzten Frame kopieren)" (F3-Kollision
mit Forschungsfrage), Band-Legende "Seed-Variabilitaet +-0,014"
(Anzeige-Mapping VAR_LBL, CSV-Schluessel unveraendert).
(13) 17_stdgap_beforeafter: "Gap x %" -> "Abweichung x %", y-Label
"1 = keine Abweichung"; Balken verbreitert (weisses In-Bar-Label lief
sonst ueber den Balkenrand). 16c_rollout-Paneltitel "std-Gap-
Kompoundierung" -> "std-Ratio ueber den Rollout" (gleiche Sprachregel).
Abnahme per pdftotext + PNG-Sichtung aller vier Figuren bestanden.

## NACHTRAG 14.08. (spaet) — Nachtrag 2 des Finale-Korrektur-Auftrags
Zwei Anhang-Nachzuegler + zwei eigene pdftotext-Funde:
(14) 16c_rollout: "std-Gap" war bereits in der vorigen Runde entfernt
(Paneltitel jetzt "std-Ratio ueber den Rollout") — verifiziert.
(15) 18_flow_rollout: "(F3 kopieren)" -> "(letzten Frame kopieren)".
Zusaetzlich (gleiche Sprachregeln, eigener Scan): 16b9_tradeoff
"std-Gap" -> "Abweichung" (3 Stellen inkl. y-Label); rueckholquote
(in Thesis gestrichen, Skript bereinigt) "Forecasting-Gap" ->
"Abstand Real-Latent-Referenz - World-Model", "Gap" -> "Abstand".
Voll-Abnahme ueber alle 35 PDFs BESTANDEN: kein Treffer fuer Orakel/
Forecasting-Gap/Dynamik-Beitrag/Sweet-Spot/per Konstruktion/d. Fehlers/
CVAE z=mean/std-Gap/F3 kopieren/= OFF/Headline/Signifikanz.

## NACHTRAG 17.08. — Mini-Korrekturen (RENDER_AUFTRAG_MINIKORREKTUREN)
(1) korridor_overview 0.344: war seit 14.08. bereits korrekt (Schreib-
session hatte alten Einbau) — verifiziert, kein Re-Render noetig.
(2) 16b9_dose_response: "gedrehtes Gewicht lambda" -> "variiertes
Gewicht lambda" (xlabel beider Panels). (3) Filmstrips: "Kontext F3"
-> "Kontext t" (F3-Kollision mit Forschungsfrage). Abnahme pdftotext:
kein 0.342/gedreht/Kontext F3; "variiertes Gewicht" 2x.

## NACHTRAG 18.08. — Finaler Abbildungs-Check (abbildungsaenderungen_final.pdf)
Matplotlib-Anteil umgesetzt: (5.4) 19_all_levers "letzten Frame kopieren"
-> "letzten Latent kopieren", "Flow Matching (mean = frozen Backbone)"
-> "(deterministische Basis)". (5.16) 25_head_comparison "frozen Kopf"
-> "ursprünglicher Kopf". (5.17) 25_det_boxes_versions +
(A.10/A.11) 24_det_boxes_panel/peds: "frozen Kopf" -> "ursprünglicher
Kopf", "ADAPTIERTER Kopf" -> "adaptierter Kopf" (einheitliche
Kleinschreibung). (A.1) 16c_rollout "letzten Frame wiederholen" ->
"letzten Latent wiederholen". (A.8) 18_flow_rollout "letzten Frame
kopieren" -> "letzten Latent kopieren". BEREITS ERLEDIGT aus frueheren
Runden: 5.1 (0.344), 5.2 (variiertes Gewicht), 5.5/5.6 (Kontext t).
NICHT unser Anteil: 2.2/4.1 (PowerPoint, Vincent), Captions (LaTeX,
Schreibsession). Abnahme: kein frozen Kopf/ADAPTIERTER/letzten Frame
in den 7 PDFs; alle optimiert.

## NACHTRAG 22.08. — Abb. 5.17 transponiert (RENDER_AUFTRAG_5_17_UMBAU)
Prof-Feedback "Panels zu klein": 24_det_boxes_peds von 2x4 (Szenen x
Varianten) auf 4x2 TRANSPONIERT (Zeilen GT / urspruenglicher Kopf+real /
urspruenglicher Kopf+Vorhersage / adaptierter Kopf+Vorhersage; Spalten
Szene A/B mit Titeln, Zeilenlabels rotiert links). Neue Einbaubreite
12 cm als eigene hochformatige Float-Seite. MESSUNG: tight 14.1x25.6cm
-> bei 12-cm-Einbau 12x21.8cm (<23-Limit), Panelkante ~5.2cm (vorher
~4), Druck-Font 1.01x9pt. KALIBRIER-DETAIL: FIG() normiert auf 17cm,
das hohe Raster fuellt die Breite aber nicht (tight-Crop auf 14.1cm)
-> apply_style(14.4)=17*12/14.2 kompensiert; einmal iteriert,
konvergiert. Geschwister-Panels (20/24/25) NICHT angefasst (Auftrag:
nur nach Rueckfrage).

## NACHTRAG 22.08. (2) — Finale Terminologie-Runde (RENDER_AUFTRAG_FINALE_ABBILDUNGEN)
11 Figuren: "Full-Val" -> "Vollvalidierung", "Real-Latent-Referenz" ->
"Referenz: reales Latent", "nuScenes-GT"/"vs. GT" -> "nuScenes-
Annotation"/"vs. Annotation" (inkl. Zeilen-/Paneltitel korridor_overview
+ error_decomposition), "WM"/"World-Model" -> "Weltmodell" (Labels/
Legenden/Pareto-Punkte), "Ego-Conditioning" -> "Ego-Konditionierung",
"frozen Kopf" (23_decoder_panel) -> "ursprünglicher Kopf", "WM-
VORHERSAGEN/REALEN Latents" -> "Weltmodell-Vorhersagen/realen Latents",
"Real-Mix" -> "mit realen Latents", "Peak-" -> "Spitzen-GPU-Speicher",
"mse+5 Regler" -> "MSE + 5 Terme", "Injection-Protokoll" ->
"Injektionsprotokoll". LAYOUT-FIX dabei: 21_seg_gt_korridor linke
xticklabels kollidierten mit den laengeren Weltmodell-Namen ->
einzeilig rotiert (16 Grad, S(8)). Nicht-Anfassen-Liste beachtet
(2.2/3.1/4.1/5.17/A.1/A.8/A.9/A.10 unveraendert; kein pauschales
Latent-Ersetzen). Abnahme-Scan ueber die 11 PDFs BESTANDEN (kein
Full-Val/nuScenes-GT/vs. GT/Real-Latent-Referenz/Ego-Conditioning/
frozen Kopf/WM-VORHERSAGEN/Real-Mix/Peak-GPU).

## NACHTRAG 22.08. (3) — Dezimalkommas zentral + 5.12 + A.9/A.10 hochkant
(1) DEZIMALKOMMAS: zentraler Patch in thesis_style.py auf
matplotlib.text.Text.set_text (Regex nur Punkt ZWISCHEN Ziffern) —
erwischt Ticks, Annotationen, Balkenwerte und Legenden ALLER Skripte
ohne Einzelaenderungen (Text.__init__ ruft set_text; Tick-Labels laufen
beim Draw darueber). 16c_rollout zusaetzlich "~0,5 s/Schritt" (Komma +
Leerzeichen). Alle 30 quantitativen Figuren neu gerendert.
(2) 20_det_korridor (5.12, Terminologie-Nachzuegler): World-Model ->
Weltmodell minimal / 6 Terme, Real-Latent-Referenz -> Referenz: reales
Latent. (3) A.9/A.10 (20_/24_det_boxes_panel) von 3x4-quer auf
4 Zeilen (Varianten) x 3 Spalten (Szenen) HOCHFORMAT transponiert
(analog 5.17; 14.4x18.9cm tight, apply_style(17) unveraendert);
A.9-Label "World-Model (Vorhersage)" -> "Weltmodell (Vorhersage)".
NEBENFIX: render_ressourcen.py las das bei Task 27 zu _rep{1..3}
umbenannte ressourcen.json -> auf konsolidierte ressourcen.csv
umgestellt (identische Werte inkl. Encoder-Zeile), Figur wieder
reproduzierbar + kommaisiert. ABNAHME ueber 30 PDFs: kein [0-9].[0-9],
kein World-Model/Real-Latent-Referenz/0.5s. Sichtproben 5.12 + A.9 ok.

## NACHTRAG 22.08. (4) — Explizite Rundung (RENDER_AUFTRAG_RUNDUNG)
Zentraler Helfer fmt() in thesis_style.py: Decimal(str(x)) +
ROUND_HALF_UP statt f"{:.3f}" (das auf dem Binaerwert rundet:
0.6295 binaer knapp darunter -> "0.629"). ALLE Wert-Label-Stellen in
14 Skripten auf fmt() umgestellt (3-/4-stellig, sign-Variante fuer
Deltas), Figuren neu gerendert. ERGEBNIS: korridor_overview +
21_seg_gt_korridor zeigen jetzt 0,630 (aus 0.6295). ABER 19_per_class:
Label bleibt KORREKT 0,897 — das Primaer-Artefakt headline_minimal_
fullval.json traegt 0.8973 (nicht ~0.8975 wie im Auftrag vermutet);
der 0.898-Wert in TASK19-Tabelle/Fliesstext war ein Uebertragungs-
fehler (Nachtrag im TASK19-Bericht; Fliesstext bitte auf 0,897
aendern). Terminologie: 18_flow_calibration "300er-Subset" ->
"Teilstichprobe mit 300 Fenstern". AUFGERAEUMT: veraltetes
17_filmstrip.{pdf,png} (Vorgaenger der statisch/dynamisch-Fassungen,
seit 11.08. von keinem Skript mehr erzeugt) -> archiv/
alt_visualisierungen/. Scan: alle 34 PDFs ohne Dezimalpunkte,
kein Subset; Stand der 16:33-Runde sonst unveraendert.

## NACHTRAG 22.08. (5) — Genus + Sprach-Restlabels (RUNDUNG-Nachtrag)
PFLICHT Genus Neutrum: "letzten Latent" -> "letztes Latent" in
16c_rollout, 18_flow_rollout, 19_all_levers (ueberschreibt die
fruehere Freigabe-Liste). OPTIONAL mitgenommen: 19_all_levers-Labels
eingedeutscht (FiLM Zustand/Aktion, 4 Eingabeframes, Kapazitaet:
8 Schichten, EMA, z=Mittel, ego-kompensiert; via VAR_LBL, CSV
unveraendert); 16d_ego_persistence "Weltmodell (Smooth-L1 minimal)" +
"Persistenz ego-kompensiert". 04_praediktor.png NICHT angefasst
(Reviewer-Freigabe). Alle 4 Figuren nach Thesis/fig/ kopiert,
tas_modern.pdf neu kompiliert. Abnahme pdftotext gruen.
