# CLAUDE.md — BEV World Model (Masterarbeit)

Stehender Kontext fuer Claude Code. Quelle: konsolidiert aus den
TASK*_ABSCHLUSSBERICHT.md + Arbeitsplan.py (Stand 2026-07-15, nach Task 16b.5).
Details IMMER in den Berichten (Ordner ./berichte/) nachschlagen, nicht raten.

## Projekt

Transformer-basiertes BEV-World-Model fuer autonomes Fahren (Masterarbeit,
UniBw Muenchen). Prognose des naechsten BEV-Latent-Frames aus 3 vorherigen
Frames; Latents stammen aus dem eingefrorenen BEVFusion-Seg-Encoder
(nuScenes, Latent-Shape [512, 128, 128]).

- Modell: ~6M Parameter, 4-Layer Pre-LayerNorm-Transformer
  (Code/: downsampling, embedding, transformer, output_head, upsampling_head)
- Loss: 6 gewichtete Terme — lambda_mse / cos / mean / std / grad / ssim
- Kernmetrik: mIoU via eingefrorenem Seg-Decoder; Nebenmetrik std-Ratio
  = mean(pred_std)/mean(real_std) aus inference_log.json (300 Val-Samples)
- Referenz: bestes mIoU 0.6823 vs. Persistenz-Baseline 0.5329
- Betreuer: Leon Pohl. Cluster-Kontakte: Thorsten (BeeGFS-Packs), Anton.

## Aktueller Stand (nach 16b.9, 16b-KETTE ABGESCHLOSSEN)

16b.9 FERTIG (Job 123424, Full-Val ohne Early-Stop). HEADLINE-mIoU (5743):
Baseline (6 Terme) 0.6920 | SmoothL1-Minimal (smooth_l1+std, 2 Terme) 0.6946.
Delta +0.0026 (n.s.) -> "FROM SIX TO TWO": Minimal-Konfig ERREICHT die Baseline,
cos/mean/grad/ssim entbehrlich. Beide > alte Bestmarke 0.6823 (Headline ohne
Early-Stop = echter konvergierter mIoU). SmoothL1-Minimal = Gesamtsieger
(wenigste Terme, hoechste mIoU, bester std-Gap aus 16b.7). std-Ratio bleibt <1.0
-> std-Gap strukturell -> Task 18.
FINALE LOSS-KONFIG (fuer 16c/Thesis): recon_loss=smooth_l1, TERM1=1.0, std=1.0,
Rest 0. Checkpoint: checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt.
Synthese: synth_16b9.py, visualizations/16b9_*.{png,pdf}, synthesis_master.csv.

## Frueherer Stand (nach 16b.6)

VIER Regler-Sweeps der 16b-Kette fertig, KEINER liefert belastbaren mIoU-Gewinn:
- lambda_mean (16b.6, Job 123172): FLACH auf beiden Metriken (alle Deltas n.s.).
  0-Punkt = Baseline -> Regler nicht noetig. CSV sweep_lambda_mean.csv.
- lambda_cos (16b.6, Job 123173): mIoU n.s. (leichter Aufwaertstrend zu 0.5,
  +0.0110 knapp unter Schwelle), std-Ratio signifikant+monoton SCHLECHTER
  (0.9182->0.8888). Nullpunkt cos=0 = BESTE std-Ratio (0.9182) bei baseline-
  gleichem mIoU. CSV sweep_lambda_cos.csv.
Tendenz minimale hinreichende Loss-Menge: mse + std tragend; grad/ssim/mean/cos
nahe 0. Endgueltig in 16b.9.
Anker-Replikate erneut sauber: mean=0.1 -> 0.6755/0.9086, cos=0.1 -> 0.6736/0.9075.

Frueheres Fundament (16b.4/16b.5), beide Struktur-Regler ohne mIoU-Gewinn:
- lambda_grad (16b.4, Jobs 122859+122997): kein signifikanter mIoU-Gewinn;
  std-Ratio verschlechtert sich signifikant+monoton bei hoeheren Werten.
  CSV: predictions/task16b/sweep_lambda_grad.csv
- lambda_ssim (16b.5, Job 123043, a100-3, COMPLETED 4h53min): mIoU sinkt
  MONOTON mit steigendem lambda_ssim (0 -> 0.6867, 0.1 -> 0.6732, 0.3 ->
  0.6704); bei 0.3 knapp signifikant SCHLECHTER. std-Ratio flach (kein Effekt).
  Bester Wert = SSIM AUS (0.6867 = hoechster 16b-Proxy bisher).
  CSV: predictions/task16b/sweep_lambda_ssim.csv

REPLIKAT-BEFUND (16b.5): Basis-Config traegt lambda_ssim=0.1 -> Sweep-Wert 0.1
war config-identisch zum 16b.4-Anker (Seed 42) und reproduzierte ihn auf
0.0001 (mIoU 0.6731->0.6732, std-Ratio 0.9085->0.9086). Folgerung: Same-Seed
ist faktisch deterministisch; die 0.014-Schwelle (15.2) ist reine Seed-Varianz,
kein Hardware-Rauschen. Cross-Value-Deltas <0.014 bleiben n.s.

FINALER-LOSS-TENDENZ (Zwischenstand): lambda_ssim=0 (streichen),
lambda_grad klein/0. Endgueltig erst in 16b.9 (Synthese).

16b.7 ABGESCHLOSSEN (18.07., Job 123303): recon_loss-Schalter (TERM 1) gebaut
(mse|l1|smooth_l1, in train_linux.py+mk_sweep_cfgs+Basis-Config, mse bit-identisch).
BEFUND: SmoothL1 vs MSE -> mIoU praktisch gleich (-0.0024 n.s.), std-Ratio
SIGNIFIKANT BESSER (0.9089->0.9266, +0.0177 = ~2x Schwelle, bester 16b-Wert!).
SmoothL1 = erster Term, der den std-Gap schliesst OHNE mIoU-Kosten (weniger
mean-seeking als MSE). CSV sweep_recon_loss.csv. -> Minimale Loss-Menge plausibel
smooth_l1+std statt mse+std.

16c FERTIG (20.07., Job 123442, rollout_eval.py, 5467 Val-Fenster, beide Konfigs).
Rollout k=1..4: mIoU faellt ~32% rel. (0.69->0.47), aber Modell schlaegt
Persistenz bei JEDEM k (k4: ~0.47 vs 0.35). k=1 ~ Headline (Konsistenz ok).
SmoothL1-Vorteil KOMPOUNDIERT (dmIoU +0.0026->+0.0091 ueber k). std-Ratio STEIGT
im Rollout (Varianz-Aufblaehung): Baseline 0.91->0.97, SmoothL1 0.99->1.01
(naeher am Ideal, schiesst k4 leicht drueber). Moderater Drift, NICHT katastrophal
-> Task 18 motiviert aber nicht erzwungen. JSONs predictions/task16c/,
Plot visualizations/16c_rollout.*. rollout_eval.py idempotent (Task-18-tauglich).

17 FERTIG (20.07.): Thesis-Bildmaterial konsolidiert+erzeugt. visualize_thesis.py
(Cluster-Decode -> .npz) + render_thesis.py (lokal, weil bevwm-Env KEIN matplotlib
hat -> Compute/Render getrennt). Figuren visualizations/: 16b9_tradeoff,
16b9_dose_response, 16c_rollout, 17_miou_over_epochs, 17_stdgap_beforeafter
(KERNARG: std-Gap 9.1%->1.4%), 17_masks_overview (staerkstes Bild), 17_gate_alpha
(Gate kopiert statisch / sagt dynamisch vorher), 17_filmstrip (7-Frame-Rollout:
statisch stabil mIoU~0.95, dynamisch driftet 0.31->0.18 -- qualitative 16c-Ergaenzung).
Referenztabelle als METHODISCHES Geruest (Komparator-Zahlen aus Papers zu
verifizieren, NICHT erfunden). Scripts: visualize_thesis.py + render_thesis.py,
filmstrip_compute.py + filmstrip_render.py (Compute Cluster / Render lokal).

16d-VORSTUFE FERTIG (21.07.): ego-gewarpte Persistenz-Baseline (ego_warp_persistence.py,
kein Retrain). nuScenes ego2global -> BEV-Maske warpen; Konvention per 8-Kombi-Sweep
+ 2 Gates validiert (yaw=+1, AXIS_SWAP=True, INVERT=False; Gate1 Selbst-Warp=1.0).
BEFUND (5467 Fenster): ego-gewarpt viel staerker als naiv (k1 0.629 vs 0.533).
Modell schlaegt BEIDE bei jedem k, aber Vorsprung vs ego-gewarpt nur +0.067..+0.035
(vs naiv +0.16..+0.13, ~3x aufgeblaeht durch Ego-Motion). Ehrliche Neueinordnung:
Modell-Dynamik-Beitrag real aber moderat, schrumpft mit Horizont. Figur
16d_ego_persistence.*, JSON predictions/task16c/ego_pers.json.
VERBESSERUNGS-CHECK (VERBESSERUNGSVORSCHLAEGE_MODELL.md, Stand nach 16b.6):
#3 Loss-Diaet ERLEDIGT (16b.9, sogar staerker). #2 Rollout-Finetuning NICHT
getriggert (Modell > beide Persistenzen). #5 std-Gap generativ ENTKRAEFTET (16b.7
gap 1.4%). #1 Ego-Conditioning = staerkster Rest (Retrain); ego-gewarpte Baseline
(oben) ist die kostenlose Vorstufe/Motivation. #4 (3->4 Frames)/#6 (EMA) billige Ablationen.

27 SEED-ABSICHERUNG FERTIG (14.08., TASK27_ABSCHLUSSBERICHT.md; alles
lokal). Kopf-Adaptation MULTI-SEED bestaetigt: Seg 0.6008+-0.0005 (4 Seeds,
jeder >= +0.0102 ueber frozen 0.5898); Det v2 pred 0.4314+-0.0021 (min.
+0.0854 ueber frozen), real 0.6720+-0.0008 (max. -0.0148 unter Orakel).
Kopf-Seed-Streuung 20-40x kleiner als Effekte -> 0.014-Schwelle (WM-Varianz)
hier nicht massgeblich. Flow-Pareto-Punkt = Full-Val 0.6946 per Konstruktion
(frozen Backbone). Best-of-K-Kurve (Coverage): 0.645->0.6875 (K=16).
Ressourcen 3x wiederholt (<0.5ms Spannweite). LEKTION: archiv/tools/ braucht
PYTHONPATH Root+Code/.

25 VERFEINERTE DET-ADAPTATION FERTIG (03.08., TASK25_ABSCHLUSSBERICHT.md).
v2 = 12 Ep. Cosine + 20% Real-Mix + Val-Loss-Selektion (Holdout 500, Best
Ep. 9; adapt_det_head_v2.py). ERGEBNIS: pred 0.4292/0.5333 (Gewinn
gehalten, -0.0037 < Rauschboden) und real 0.6710/0.7040 — Real-Malus fast
halbiert (-0.0255 -> -0.0148). EMPFEHLUNG: v2 = EIN-KOPF-Betriebspunkt
(+0.085 mAP auf Vorhersagen, nur -0.015 unter Orakel auf real); v1 als
pred-Spezialist archiviert. Mix+Selektion wirkten, Dauer allein nicht.

26 RESSOURCENBEDARF FERTIG (07.08., TASK26_ABSCHLUSSBERICHT.md; schliesst
Bearbeitungspunkt 4). Inferenz Batch 1 fp16 TITAN RTX (30 Warmup/300 Iter):
Seg-WM minimal 4.41ms | Det-WM 11.45ms | Flow +9.5ms/Euler (gesamt 14ms) |
8-Layer 7.1ms (+60% fuer n.s. Gewinn) | BEVFusion-Encoder 172.8ms. WM =
<1-2.3% des 500ms-Zyklus -> echtzeitfaehig >40x Puffer (entschaerft Lim 5.9),
WM-Aufschlag ~2.5% auf die Wahrnehmung. Peak-VRAM <270MB, Gewichte fp16
<25MB. Trainingsaufwand aus Logs: Headline ~12-25 GPU-h, Kopf-Adaptation
1-4.3 GPU-h. Artefakte predictions/task26_ressourcen/*, Figur
ressourcen_inferenzzeit.*, archiv/tools/bench_resources.py. LEKTION:
Docker-/bevfusion-Edits via exec persistieren nicht -> Host-Datei editieren.

24 DECODER-ADAPTATION DET FERTIG (02.08., Bericht TASK24_ABSCHLUSSBERICHT.md).
TransFusion-Analogon zu 23: Kopf auf 13519 det_baseline-Train-Vorhersagen
(stride 2) nachtrainiert (adapt_det_head.py, GT aus pkl, Original-Loss).
ERGEBNIS: adaptiert+Vorhersagen mAP 0.4329/NDS 0.5328 vs frozen
0.3438/0.4761 = +0.0891 mAP (~20x Rauschboden) — GROESSTER EINZELGEWINN
des Projekts, ~26% des Forecasting-Gaps (Seg: 29% -> Quote stabil).
ABER: auf realen Latents -0.0255 mAP (0.6603 vs 0.6858) -> dedizierter
Forecast-Kopf (Zwei-Kopf-Betrieb), kein Ersatz. KERNSATZ 23+24: Decke im
GT-trainierten Uebersetzer adressierbar (~1/4-1/3 des Gaps), nicht im WM.

23 DECODER-ADAPTATION SEG FERTIG (02.08., alles lokal; Bericht
TASK23_ABSCHLUSSBERICHT.md). Post-Fuser-Stack (SECOND+FPN+SegHead, 9.3M)
auf (Train-Vorhersagen 27038 x minimal-Modell -> echte GT) nachtrainiert
(adapt_seg_head.py im Docker, 8 Ep., ohne Sensor-Pipeline; Train-Dump
BeeGFS dumps_task23/, GT-Cache 200x200 gt_masks_pipe_*). ERGEBNIS
(Injection, Full-Val): adaptierter Kopf auf Vorhersagen 0.6012 vs frozen
0.5898 (+0.0114; stop_line +0.0325, divider +0.0123 = GENAU der Selten-
Klassen-Hub, den Task 22 im WM nicht liefern konnte); auf REALEN Latents
Decke unveraendert (0.6293 vs 0.6295) -> echter Uebersetzungs-Gewinn,
~29% des Forecasting-Gaps zurueckgeholt, Retention 93.7->95.5%.
KERNSATZ: Klassenwissen gehoert in den GT-trainierten UEBERSETZER, nicht
in die Latent-Regression. Checkpoint det_latents_out/adapted/
seg_adapted_full_best.pth (Voll-Format, auch fuer Visualisierung).
Einschraenkungen: 1 Seed, nur minimal-Vorhersagen gesehen; Det-Analogon
= Future Work (52% Forecasting-Anteil). LEKTION: SegHead-Eval-Branch
liefert bereits Sigmoid (Doppel-Sigmoid-Falle im Proxy).

22 SELTEN-KLASSEN-/DYNAMIK-HEBEL FERTIG (02.08., Jobs 125758/763/764 Sweep +
125773 Dumps + lokale GT-Kette; Bericht TASK22_ABSCHLUSSBERICHT.md).
5 Varianten auf 16b.9-Minimal: rare_w{2,4,8} (zellgewichtete SmoothL1 auf
Selten-Klassen-Regionen aus Map-GT), samp_rare, samp_dyn (Sampler).
PROXY: alle 5 ueber Anker 0.6735 (best rare_w2 0.6862, +0.0127 n.s.).
ECHTE GT: KEIN Gewinn — rare_w2 0.5850 / rare_w8 0.5835 / samp_dyn 0.5842
vs minimal 0.5898; kein per-Klasse-Hub (divider/stop_line alle leicht
negativ); samp_dyn in JEDEM Dynamik-Quartil unter minimal, Defizit waechst
mit Dynamik (q4 -0.008). VORBEHALT: early-stopped vs konvergiert (~-0.005
erklaerbar) -> bestenfalls neutral. ECHTER EFFEKT: Varianz-Kalibrierung —
std-Ratio 0.9266 -> 0.987..0.9975 monoton in w = staerkster std-Gap-Hebel
des Projekts (statistisch, nicht semantisch). META: Pseudo-GT kann am Rand
irrefuehren (+0.013 Proxy bei <=0 GT) -> Sweep-Gewinne < Schwelle immer
gegen GT pruefen. Betreuer-Frage beantwortet: unter frozen Decoder heben
Gewichtung/Balancierung die Selten-Klassen nicht (konsistent Task 21).
Empfehlung bleibt SmoothL1-Minimal; rare_w2 als dokumentierte Option fuer
std-Kalibrierung. LEKTIONEN: max 2 Trainings je 500G-cgroup (OOM-Deadlock
5er-Job, Freeze-Detektor via Log-mtime); inference.py IMMER mit --output je
Lauf; Quartil-/Subset-PKLs nur aus container-konsistentem Quell-PKL.
Dynamik-Quartile konfundieren Ego-Tempo mit Kartentyp (q3>q4) — nur
Innerhalb-Quartil-Deltas interpretieren. JSON predictions/task22/gt_eval22.json.

21 SEG-GT-VERANKERUNG FERTIG (31.07., Betreuer-Input; Bericht
TASK21_ABSCHLUSSBERICHT.md). Seg-mIoU erstmals gegen ECHTE nuScenes-Map-GT
(Injection wie 20.1, --eval map, IoU@max, Full-Val 6019): Orakel-Gate
bestanden (0.62946 vs Referenz 0.62947). ERGEBNIS (mean): Persistenz 0.4647 |
WM minimal 0.5898 | WM 6-Term 0.5882 | Orakel 0.6295. BEFUND 1: Rangfolge
der Pseudo-GT-Metrik VALIDIERT (minimal ~ 6-Term, d +0.0016; from-six-to-two
gilt auf Seg auch gegen GT). BEFUND 2: Selten-Klassen-Einbruch = ueberwiegend
DECODER-Decke — WM-Retention je Klasse 88.8-97.7% (mean 93.7%); groesster
WM-Verlust bei duennen Strukturen (divider 11%, ped_cross 7.5%); Headroom
mean nur 0.040. KORREKTUR: Seg-Latents sind (1,256,128,128) fp32 — 256
Kanaele, NICHT 512 (Kopfzeile oben veraltet). Betreuer-Vorschlag CE/Focal/
Balancing: nur via Task-Loss (16f-H1: schadet), gewichtete Latent-Loss oder
Sampler adressierbar; realistisches Ziel ~+0.01-0.02 GT-mIoU -> Task-22-
Entscheidung beim Nutzer. JSON predictions/task21/gt_eval.json, Figur
21_seg_gt_korridor.*, Container seg_gate (Latents ro-gemountet).

20.3 SEED-43-RAUSCHBODEN FERTIG (30.07., Job 125355 Training + Dumps 125425/26
nach BeeGFS + 2 lokale Injection-Evals) -> TASK 20 KOMPLETT, BEFUND 2 ENTSCHIEDEN.
Seed 42 / Seed 43 (mAP): minimal 0.3264 / 0.3220 | baseline 0.3438 / 0.3407.
DET-RAUSCHBODEN = max. Seed-Streuung 0.0044 mAP (0.0040 NDS), Schwelle 2x =
0.0088. Delta (6-Term minus minimal): +0.0174 (s42) / +0.0187 (s43) / Mittel
+0.0180 mAP, +0.0109 NDS = 4x Rauschboden, gleiches Vorzeichen in BEIDEN Seeds
-> SIGNIFIKANT. Val-Losses der Seed-Paare fast identisch (minimal 0.006987 vs
0.006984) -> Rauschen entsteht in Latent->Boxen, nicht im Training.
KERNSATZ: "from six to two" ist keine allgemeine Regel, sondern eine
EIGENSCHAFT DER ZIELMETRIK — auf Seg entbehrlich (+0.0026 n.s.), auf Det
kosten die Zusatzterme ~5% rel. mAP, wenn man sie weglaesst (schwerere
Latent-Tails, Boxen haengen an feinen lokalisierten Strukturen -> grad/ssim/cos
tragen). EMPFEHLUNG: Seg -> Minimal, Det -> 6-Term. Befund 1 (Modell
VERDOPPELT Persistenz) in beiden Seeds bestaetigt. JSON
predictions/task20/seed_noise.json, Figur 20_det_korridor.* (Einzel-Seed-Punkte
+ Delta-Klammer), Logs eval_det_*_s43.log.

20.1+20.2 DET-STRANG KERN FERTIG (27.-28.07., Bericht TASK20_ABSCHLUSSBERICHT.md).
MESSMASCHINE: Docker-Injection-Hook (LOAD_BEV_LATENTS, Gegenstueck zum Saver)
statt TransFusion-Nachbau — Gate A (reale Latents) reproduziert mAP 0.6858/NDS
0.7146 EXAKT, Gate B (Normalisierungs-Roundtrip) identisch -> Kette bewiesen,
null Cluster-Installationen. eval_dump_latents.py (Cluster) + Luecken-Konvention
(276 kontextlose Frames real gefuellt, alle Varianten gleich) + Selektion via
Val-Loss (Protokoll-Abweichung dokumentiert). ERGEBNISSE (Full-Val 6019):
Persistenz 0.1696/0.3908 | det_minimal (smooth_l1+std) 0.3264/0.4662 |
det_baseline (6-Term) 0.3438/0.4761 | Orakel 0.6858/0.7146. BEFUND 1: Modell
VERDOPPELT Persistenz-mAP -> Methodik generalisiert, Dynamik-Beitrag klar.
BEFUND 2: from-six-to-two nur TEILWEISE (minimal -0.0174 mAP unter 6-Term,
anders als Seg) — Rauschboden offen, Seed-43-Paar laeuft (Job 125355, Cron).
shm_staging-FIX: Platzpruefung zaehlte gestagte Splits doppelt (bei 2x567G
ausgeloest, Job 124905 verloren). Figuren 20_det_korridor.* +
20_det_boxes_panel.* (visualize.py + Injection, 3 Szenen x GT/Orakel/Modell/
Persistenz). Training: ~30min/Ep (3x Seg, loader-bound), batch 8 ok, 34h-Limit.
OFFEN: Seed-43-Nachtrag, dann finale Signifikanz-Aussage zu Befund 2.

20.0 DET-INFRASTRUKTUR FERTIG (27.07., lokal verifiziert + deployed).
latent_scale-Normalisierung in Code/bev_dataset (+Fabriken/train_linux
durchgereicht, Default 1.0 = bit-identisch): Det-Latents werden beim Laden
mit 0.3641 auf Seg-Skala gebracht (gemessen 50+50: seg-std 0.2687 / det-std
0.7379, Faktor 2.746) -> SmoothL1-Arbeitspunkt beta/sigma identisch zum
Seg-Strang; Decoder-Seite (20.1) muss 1/0.3641 zurueckrechnen.
config_det_base.yaml: grid 45 (6075 Tokens), latents_det-Pfade, packed-Ziele
latents_det_packed_fp16/ (Packs entstehen idempotent beim ersten Job),
decode_validation AUS bis 20.1. SMOKE gruen: Modell 6.057M baut/forwarded
45er-Grid (2,256,180,180), Det-Val liefert 5743 Fenster = EXAKT wie Seg
(gleiche Szenenstruktur, Protokolle direkt vergleichbar), std nach Skalierung
0.284, smooth_l1+std-Loss endlich. NAECHSTE SCHRITTE: 20.1 Det-Decoder
(det_decoder_torch: frozen TransFusion-Head, mAP/NDS, ORAKEL-GATE decode(real)
== mAP 0.686/NDS 0.715) -> 20.2 zwei Trainings (Sieger-Rezept + Kontrolle;
Kontroll-Konfig-Wahl OFFEN: 6-Term-Baseline vs mse+std).

20-VORSTUFE DET-LATENT-EXTRAKTION FERTIG (26.07., lokal Docker bevfusion:correct
+ latent_saver-Hook, Task-12-Rezept mit det-Config transfusion/secfpn/
camera+lidar/swint_v0p075/convfuser.yaml + bevfusion-det.pth, --eval bbox).
BEIDE Splits extrahiert (fp16 direkt), nach BeeGFS gesynct und verifiziert:
val 6019 + train 28130 Dateien unter lrt81-vima/latents_det/{val,train}
(EIGENES Top-Level-Dir — latents/ gehoert thlu, kein Schreibrecht!), je ~16.6MB,
gesamt ~560G. Shape (1,256,180,180) fp16 — 256 KANAELE (nicht 512; Arbeitsplan
hatte recht), std-Faktor Det/Seg 2.65 gemessen (Doku 2.6 bestaetigt).
Modell-Sanity: mitgelaufene Val-Eval mAP 0.686/NDS 0.715 = Referenzwerte.
Train via Rolling-Sync (rsync --remove-source-files, nur Dateien >2min alt,
Puffer blieb <25G). Lokale Val-Kopie liegt noch unter
/home/vima/det_latents_out/latents_val_synced (94G, loeschbar nach Freigabe).
Tempo-Referenz: ~4.2 Frames/s auf TITAN RTX (val 25min, train 2h).
TASK-20-ENTSCHEIDUNG OFFEN (Datengrundlage komplett; naechste Schritte waeren
pack_latents fuer det + Pipeline-Anpassung grid 45x45/2025 Tokens).

17-NACHTRAG KOMPARATOR-VERIFIKATION FERTIG (25.07., alle 7 Referenzen gegen
Papers geprueft, Nachtrag in TASK17_ABSCHLUSSBERICHT.md): 5 verifiziert mit
Zahlen (DINO-Foresight Cityscapes 71.8/59.8 + Copy-Last 54.7 + SmoothL1
bestaetigt; OccWorld 25.78/15.14/10.51; BEVWorld FID 22.85, non-autoregressiv;
FIERY IoU 59.4/VPQ 50.2 — deren Present/Future-Gauss+KL = exakt unser
B1-CVAE-Muster, neuer Anker; LeWM "six to one" bestaetigt). 2 KORREKTUREN:
DINO-WM "196 Patches" UNBELEGT (DINOv2 Patch-14 -> 256 bei 224px; ueberall
gestrichen, auch WORLDMODEL_REFERENZEN.md) + LeWM-Analogie praezisieren
(HYPERPARAMETER 6->1 vs unsere Loss-TERME 6->2). Quervergleich: DINO-Foresight
Copy-Last-Gap +17.1 vs unser +16.2 = gleiche Groessenordnung; deren
Oracle-Rest (77.0 vs 71.8) stuetzt die aleatorische Decken-These. Offen:
DINO-Foresight-Venue pruefen, LAW-Zahlen bei Bedarf aus PDF.

19 SEG-STRANG-ABSCHLUSS ERSTELLT (25.07., lokal, kein Compute). Master-Tabelle
predictions/task19/master_table.csv (20 Varianten, alle vs off=0.6946);
Figuren 19_all_levers.* (Money-Figur: kein Hebel schlaegt Minimal-Konfig,
6 Richtungen) + 19_per_class.* (Decken-Zerlegung: drivable 0.90 vs stop_line
0.56). Bericht TASK19_ABSCHLUSSBERICHT.md: beste Konfig (SmoothL1-Minimal,
Begruendung inkl. LeWM + Rollout-Varianz-Stabilitaet), Decken-Analyse
(aleatorisch + IoU-Randempfindlichkeit + Datengroesse, Messfehler-Audit),
Figuren-Inventar. Det-Strang-Empfehlung "Future Work" UEBERHOLT (Det-Strang
in Task 20 durchgefuehrt); Nachtrag 30.07. im TASK19-Bericht zieht die
Geltungsbereichs-Grenze: Empfehlung differenziert Seg -> SmoothL1-Minimal,
Det -> 6-Term (s. 20.3). 18/B2.1 KALIBRIERUNG FERTIG (25.07., Job 124563): post-hoc Residual-Skalierung
LOEST den Kalibrierungsrest — s=0.5: std 1.11, Einzel-Sample 0.6798, Best-of-8
0.6998 = +0.010 UEBER dem det. Mittelwert (Coverage-Ergebnis, Orakel-Auswahl,
KEIN einsetzbarer Praediktor); Diversitaet bleibt 0.012 (20x CVAE). 30 Ep allein
aendern fast nichts (wirksamer Hebel = Skalierung); s als interpretierbarer
Trade-off-Regler Diversitaet<->Genauigkeit ohne Retraining. Empfehlung s=0.5.
Nachtraege in TASK18_B2- und TASK19-Bericht, Figur 18_flow_calibration.*,
JSONs eval_flow30_sc*.json, Checkpoint task18/flow30/. TASK 18 KOMPLETT.

18/B2 FLOW MATCHING ABGESCHLOSSEN (25.07., Jobs 124417 Training / 124517
Best-of-K / 124521 Rollout). FM-Residual-Kopf (6.5M = Backbone-Paritaet) auf
frozen 16b.9-Backbone. BEFUNDE: (1) ECHTE Multimodalitaet: Masken-Diversitaet
0.0221 = 37x CVAE, im Panel strukturell plausibel. (2) Best-of-K-ZERLEGUNG des
Sample-Malus -0.044 = -0.033 Verteilungs-Preis (Best-of-8 holt zurueck ->
Coverage; Perception-Distortion-Tradeoff) + -0.011 Kalibrierungsfehler
(std-Ratio Samples 1.57 statt 1.0, nur 15 Ep Kopftraining). (3) ROLLOUT: Sample
ueberholt det NIRGENDS (k1-4, Luecke ~-0.045) — WEIL det varianz-kalibriert
bleibt (std 0.99-1.01 ueber k): der lambda_std-Term liefert die Rollout-
Stabilitaet, die generative Modelle sonst versprechen (Kern-Erkenntnis);
Flow-Kalibrierungsfehler kompoundiert stattdessen (1.58->1.90). (4) steps=1 ==
steps=10 (One-Step-Projektion ok). FAZIT: generativ lohnt fuer Multimodalitaet/
Coverage, nicht fuer Genauigkeit; det 16b.9 bleibt bester Betriebspunkt.
DECKEN-ERKLAERUNG (kein Messfehler, 6-fach kreuzvalidiert): 0.69 = aleatorische
Decke; per-Klasse Headline: drivable 0.90 / walkway 0.73 vs stop_line 0.56 /
ped_crossing 0.63 — Restfehler in duennen/dynamischen Strukturen (IoU-
Randempfindlichkeit + echte Unvorhersagbarkeit + 28k-Datengroesse); decode(real
t) vs decode(real t+1) nur 0.53. Bericht TASK18_B2_FLOW_ABSCHLUSSBERICHT.md,
Figuren 18_vae_panel_flow_s10.* + 18_flow_rollout.*. Code: Code/flow_head.py,
flow_sample(), FM-Zweige train_linux, allow_missing (checkpointing), --mode in
rollout_eval, --flow_steps/miou_best_of_k in eval_vae — alles Default-neutral.
OFFEN: Task 19 (Seg-Abschluss inkl. Decken-/per-Klasse-Analyse); optional
Kalibrierungs-Nachlauf Flow (laengeres Kopftraining, Min-SNR).

18/B1 CVAE ABGESCHLOSSEN (24.07., Jobs 124248 Training / 124362 Sample-Eval).
BEFUND: technisch gesund, funktional QUASI-DETERMINISTISCH. Guard bestanden
(z=mean Full-Val 0.6875-0.6883, alle n.s. vs off 0.6946), KEIN KL-Collapse
(0.2-0.4 nats stabil) — ABER: Samples NICHT schaerfer als Mittelwert
(Gradient-Ratio ~0.86 identisch), Masken-Diversitaet nur ~0.06% der Zellen,
konsistent ueber beta {0.001,0.01,0.1} -> strukturelle Grenze der 32-dim-z-
Sonde (FiLM setzt z-Info nicht in raeumliche Varianz um; Decoder ignoriert z
funktional). = Baseline-Ergebnis des generativen Kapitels: klassische
CVAE/SVG-Sonde reicht nicht; deckt sich mit VFMF/FlowWM (2026), die dafuer
Flow Matching nutzen. NAECHSTER SCHRITT B2: Flow-Matching-Residual-Head auf
frozen 16b.9-Backbone. Berichte: TASK18_B1_CVAE_ABSCHLUSSBERICHT.md +
TASK18_METHODIK_VAE_DIFFUSION.md (Mathe CVAE+Diffusion/FM) +
WORLDMODEL_REFERENZEN.md (Literaturliste inkl. SVG/FlowWM-Anker). Panels
visualizations/18_vae_panel_*. Code modular: vae_mode off|cvae (Default off
bit-identisch), --z_mode in eval_full_val, eval_vae.py (K-Sample-Metriken),
render_18_panel.py. LEKTION: KL>0 noetig aber nicht hinreichend — Output-
Diversitaet immer separat messen; Free-Bits gegen Anfangs-KL dimensionieren.

16f LETZTE mIoU-HEBEL ABGESCHLOSSEN (23.07., Jobs 124076/124077). ALLE DREI
Richtungen aus dem Literatur-Abgleich negativ/flach: (H1) Decoder-Task-Loss
(BCE durch frozen Decoder, Finetune auf 0.6946) SCHADET monoton+dosisabhaengig
(0.1->0.6788, 0.5->0.6717, 1.0->0.6688, alle sig. schlechter) -- Latent-
Regression ist das bessere Signal. (H2) Tiefe verdoppelt (8 Layer, 9.2M):
0.6927 n.s. = Decke nicht tiefenbedingt; Breite (d_model 384) NICHT testbar
ohne Umbau (d_model an Kanalzahl 256 gekoppelt, kein Input-Proj -> Absturz).
(H3) Hartes Residual statt Gate: 0.6936 n.s. = Gate aequivalent, nicht noetig.
FAZIT: 0.69-Decke steht auf VIER Beinen (Loss 16b / Konditionierung 16d /
Kontext+EMA 16e / Task-Kopplung+Tiefe+Residual 16f) -> mIoU-Strang ENDGUELTIG
abgeschlossen; weitere Gewinne nur via Paradigmenwechsel (Task 18) oder
Features. Neue Code-Pfade alle Default-neutral: forward_logits_grad,
lambda_task/task_loss_every, --init_weights, output_mode gated|residual.
Bericht TASK16F_MIOU_HEBEL2_ABSCHLUSSBERICHT.md, Figur 16f_miou_lever2.*,
JSONs predictions/task16f/. Literaturliste berichte/WORLDMODEL_REFERENZEN.md
(LeWorldModel arXiv 2603.19312 bestaetigt 16b.9-Minimal-Loss unabhaengig).

16e mIoU-HEBEL ABGESCHLOSSEN (22.07., Job 123807, 2 GPU parallel). BEIDE billigen
Hebel FLACH: 4-Frames (nf4, n_frames 3->4) 0.6915 (-0.0031 n.s.) | EMA-Weight-
Averaging 0.6945 (-0.0001 n.s.). Konsistenz-Check: raw best_miou des EMA-Laufs
0.6952 = reproduziert off (0.6946) -> same-seed-Determinismus erneut belegt.
KERNBEFUND: kein mIoU-Gewinn. Zusammen mit 16d (Ego flach) sind ALLE billigen/
mittleren mIoU-Hebel ausgeschoepft -> deterministischer Plateau bei ~0.69 = Decke
dieses ~6M-Setups (Arbeitshypothese bestaetigt). Rest-Headroom nur in Kapazitaet
(Architektur eingefroren) oder generativ (Task 18, dann fuer SCHAERFE nicht mIoU).
EMPFEHLUNG: mIoU-Strang abschliessen, beste Konfig bleibt 16b.9-Minimal (0.6946)
-> Task 19 (Seg-Abschluss) oder bewusste Task-18-Entscheidung. EMA in train_linux
(train_cfg.get("ema",False), Default AUS = bit-identisch). Bericht
TASK16E_MIOU_HEBEL_ABSCHLUSSBERICHT.md, Figur 16e_miou_lever.*, JSONs
predictions/task16e/. Configs config_nf4.yaml/config_ema.yaml, sbatch_16e.sh.

16d EGO-CONDITIONING ABGESCHLOSSEN (21.07., Jobs 123599 FiLM / 123621 gate+token
/ 123628 FiLM-Nachhol-Eval). ENTKRAEFTET: KEIN Ego-Conditioning schlaegt off
(0.6946). Full-Val (5743): gate 0.6910 (-0.0036 n.s. -> ERHAELT Baseline),
FiLM-action 0.6175, token 0.6164, FiLM-state 0.6094 (alle ~-0.08 signifikant
SCHLECHTER). KERNBEFUND: der INJEKTIONSMECHANISMUS entscheidet, nicht das Signal
-- gate (zero-init-Bias auf Gate-Logits, minimaler Eingriff) landet auf der
Baseline zurueck; FiLM/Ego-Token (Feature-Modulation) stoeren die Latent-Repr.
und schaden. Bei k=1 traegt das Ego-Delta keine Zusatzinfo ueber den 3-Frame-
Kontext hinaus -> #1 als mIoU-Hebel erledigt (bestenfalls neutral). Bericht
TASK16D_EGO_CONDITIONING_ABSCHLUSSBERICHT.md, Figur 16d_ego_conditioning.*,
JSONs predictions/task16d/. EVAL-LEKTION: Full-Val nie mit stage_to_shm=true auf
knappem --mem (286G-tmpfs zaehlt gegen Step-cgroup -> OOM); eval-Config
stage_to_shm=false (mmap von BeeGFS). srun-Helferskripte ins geteilte Home, nicht
node-lokales /tmp.

16d FiLM-VORLAUF (21.07., Job 123599): FiLM-Conditioning gebaut
(config.py ego_cond_mode=off|state|action|both; bev_world_model FiLM zero-init=
Identitaet -> off bit-identisch; bev_dataset liefert ego_delta [n_frames*4] aus
ego2global; train_linux+eval_full_val durchgereicht). Alle Gates gruen (FiLM-Id,
off-Equiv, echte Deltas 2-5m, Gradientenfluss). Training: state+action parallel
(GPU0/1), smooth_l1+std, ohne Early-Stop, dann eval_full_val. "off" = vorhandenes
16b.9-Headline 0.6946 (kein Retrain). sbatch_ego.sh (PLAIN), configs config_ego_*.yaml.
OFFEN nach dem Lauf: rollout_eval.py + ego_warp fuer konditionierte Modelle
(rollout_eval braucht Ego-Delta-Patch: per-Schritt-action = ego(idx(1+k)->idx(2+k))).

NAECHSTE SCHRITTE (Arbeitsplan):
1. Seg-Kern-MUSS (Task 12-17) ABGESCHLOSSEN. Offen/optional: volles Ego-Conditioning
   (#1, Retrain), 3->4 Frames (#4) + EMA (#6); Task 18 (generativ,
   motiviert durch std-Gap-Rest + Rollout-Drift, aber nicht erzwungen), Task 19
   (Seg-Strang-Zusammenfassung), 16b.10 (Optuna, wenig Nutzen nach 16b.9).
   Cluster-Jobs IMMER plain, NIE CLI --export (s.u.). (P1 lokal) Gesamttabelle + Dosis-Wirkungs-Plots + Trade-off
   std-Ratio vs mIoU + Minimal-Loss-Analyse. (P2 Cluster) ZWEI Headline-Laeufe
   auf vollem Val (5743): MINIMAL (mse+std, Rest 0) UND BASELINE, beide
   miou_early_stop=false bis Konvergenz, dann eval_full_val.py. 1 Job/2 GPU/parallel.
   (P3) Bericht: beste Konfig, det. Decke (mIoU-Plateau bei std-Gap -> Task 18).
   Konfigs: Baseline = mse1/cos0.1/mean0.1/std1/grad0/ssim0.1; Minimal = mse1/std1/Rest0.
   ALLE Sweep-Jobs PLAIN absetzen (in-file), NIE CLI --export (s.u.).
   URSACHE ENDGUELTIG BESTAETIGT (16.07., ~15 Diagnose-Jobs, Bisektion):
   *** NIE sbatch mit CLI --export absetzen -- es zerstoert das /dev/shm-Staging. ***
   Die Abstuerze (Jobs 123065/067/130/141/144/146/147/148/155/169/170/171,
   shutil.copy2 -> utime(dst) FileNotFoundError beim 236G-Schreiben, Zielverz.
   verschwindet OHNE unlink-Syscall = Mount/Namespace-Teardown) werden allein
   durch das sbatch-CLI-Flag --export ausgeloest. Bisektion:
   - Plain `sbatch script.sh` (ALLE Direktiven in der Datei): staged FEHLERFREI
     (Jobs 123143/145/167/168; auch 16b.4/16b.5 liefen so).
   - Identisches Skript + NUR `--export=SWEEP_REGLER=lambda_mean` auf der CLI
     (job-name/nodelist in der Datei): STAGING SCHEITERT (Job 123171). Ein
     einziges CLI-Flag macht den Unterschied.
   AUSGESCHLOSSEN (alle einzeln getestet): RAM/Knoten (a100-2 hatte ~1.8TB frei),
   cgroup (memory.max=max), Parallelitaet, Batch-vs-srun, das shm_staging-Skript,
   der Cleanup-Trap, import torch, mk_sweep_cfgs.
   MECHANISMUS (Vermutung, egal fuer den Fix): mit explizitem --export richtet
   SLURM Env/Session anders ein -> job_container/tmpfs-Verz. wird abgeraeumt.
   REGRESSION durch die 16b.6-Env-Parametrisierung (SWEEP_REGLER via --export)
   eingefuehrt; 16b.4/16b.5 nutzten Plain-Submit und liefen.
   FIX: REGLER/VALUES IN DER DATEI setzen (wie 16b.5), PLAIN absetzen. Fuer
   parallele Regler -> Per-Regler-Kopie des sbatch, NICHT --export.
   (Wegwerf-Diagnosedateien: sbatch_shmdiag.sh, sbatch_bisect.sh, sbatch_sweep_sampled.sh)
   (16b.9-Framing: LeWM-/DINO-Foresight; beste Konfig als Headline.)
2. 16c — MUSS: autoregressive Rollout-Evaluation k=1..4, kein Training.
3. 17 Thesis (Komparatoren: DINO-Foresight, DINO-WM, LeWorldModel);
   18 optional generativ; 19/20 Det-Strang nur als Stretch.

## Speicher-Disziplin Cluster (HARTE REGEL, 29.07. teuer gelernt)

*** GROSSE ARTEFAKTE NIE INS CLUSTER-/home. ***
/home ist eine GETEILTE 1.8T-Platte fuer alle Nutzer. Zwei Det-Dumps
(je ~95G) unter ~/Code_final/predictions/ haben sie auf 100% gefuellt
-> Dump-Jobs starben nach 45 min an ENOSPC (Exit-Code trotzdem 0, weil die
sbatch-Skripte kein set -e haben!) und der Cluster war fuer ALLE blockiert
(Meldung von Nils Schacknat). Aufgeraeumt: Home 318G -> 21G.
- Dumps/Latents/Packs IMMER nach BeeGFS:
  /mnt/beegfs/ssd/lrt81-students/lrt81-vima/ (1.3P frei; dumps_task20/,
  home_offload/, latents_det/, latents_det_packed_fp16/).
- Im Home ausgelagert per Symlink (Pfade unveraendert nutzbar):
  ~/pkl, ~/bevfusion-seg.pth, ~/Code_final/wandb -> home_offload/.
  Eine BeeGFS-Kopie von checkpoints/ liegt unter home_offload/checkpoints
  (verifiziert 212/212 .pt); das Home-Original wurde bewusst BEHALTEN
  (Stand 21G ist unauffaellig). miniconda3 bleibt im Home (Env auf BeeGFS
  = zaehe Imports).
- eval_dump_latents.py hat seit 29.07. eine PLATZ-VORPRUEFUNG (shutil.disk_usage
  vs. n_files x C x H x W x 2, Abbruch bei <105%) -> nie wieder ein halber Dump.
- Vor jedem grossen Schreibvorgang: df -h /home pruefen; du -sh ~ nach dem Job.

## Arbeitsweise (WICHTIG)

- Plan-first: Vorgehen + Entscheidungen erst vorschlagen, Freigabe abwarten,
  DANN Code/Jobs. Ein Subtask pro Arbeitspaket.
- Ein-Code-Pfad: EINE Codebasis fuer lokal + Cluster, Verhalten nur ueber
  Configs. Keine Parallel-Pipelines, keine Env-Sonderzweige im Code.
- Patch-Verifikation VOR Compute: diff zeigen; bash -n fuer sbatch,
  python -m py_compile fuer Python. Config-Patches nur ueber
  mk_sweep_cfgs.py / patch_line (Exact-1-Match-Guard, nie blind sed).
- Dokumentation: je Task ein TASKXX_ABSCHLUSSBERICHT.md; datierte Eintraege
  ergaenzen statt alte Schluesse ueberschreiben. Berichte/Kommentare auf
  DEUTSCH mit ASCII-Umlauten (ae/ue/oe, ss).
- ORDNERSTRUKTUR SEIT 03.08.: Root entruempelt — archiv/configs/ (Task-
  Configs), archiv/sbatch/ (alle Job-Skripte), archiv/tools/ (Einmal-Helfer
  wie mk_*, adapt_*, harvest_*), scripts_render/ (ALLE Figuren-Skripte;
  Aufruf aus dem Root: python3 scripts_render/<x>.py, README dort). Im Root
  nur aktive Pipeline + Basis-/Eval-Configs. Lokal und Cluster identisch.
- Datei-Konvention: Orchestrierung top-level im Projektroot (train_linux.py,
  inference.py, sweep_runner.py, sbatch_*, mk_*, harvest_*, eval_full_val.py);
  Module unter Code/ (Grossschreibung). Gilt fuer ALLE neuen Dateien.
- Lange Laeufe: tmux + tee + python -u.
- Destruktives (scancel, rm, Ueberschreiben von Checkpoints/CSVs) nie ohne
  explizite Bestaetigung.

## Wissenschaftliche Leitplanken

- Signifikanz (15.2, n=3 Seeds, konservativ): |dmIoU| > 0.014,
  |d std-Ratio| > 0.009. Kleinere Deltas = Rauschen, nicht interpretieren.
- Arbeitspunkt lambda_std = 1.0 (fix in allen Sweeps; 10.0 = Ueberregularisierung).
- std-Gap (~16-26%) ist strukturell (Regression-to-the-mean unter MSE),
  kein Bug; echte Loesungen (probabilistisch/diffusiv) out of scope.
- Sweeps: 300-Sample-Proxy + mIoU-Plateau-Early-Stopping
  (miou_patience 3, min_delta 0.014). Full-Val nur fuer 16b.9/Headline.
- Re-Evaluation historischer Checkpoints: Basis-Config + explizites
  --checkpoint Override.
- Seg- und Det-Latents sind INKOMPATIBEL (128x128 vs. 180x180, andere
  Fuser-Gewichte, std-Faktor ~2.6) — Seg-Decoder nie auf Det anwenden.

## Umgebung 1: lokaler Server (hier laeuft Claude Code)

- alre-server-u20, Ubuntu 20.04, 1x TITAN RTX 24GB.
- Python 3.8.10 SYSTEMWEIT, KEIN conda. torch 2.1.2+cu121.
- Projektroot: ~/Desktop/Masterarbeit/Code_final/ (Module: Code_final/Code/).
- Rolle: Entwicklung, Patches, kleine Tests; Zero-Cost-Fallback fuer Sweeps
  (Reaktivierung nur wenn Cluster-Queue >2-3 Tage blockiert, Task 16a).
- Docker bevfusion:correct fuer BEVFusion-Decoder-Arbeiten; nach jedem Start
  einmal `python setup.py develop` im Container. Startbefehl steht in den
  Berichten/Memory — nicht frei rekonstruieren.

## Umgebung 2: DGX-Cluster (Steuerung via ssh vom lokalen Server)

- Headnode-Alias: `head` (~/.ssh/config, Key-Auth passwortlos, ControlMaster
  aktiv). Voll: vima@tas-dgx-head.lrt.unibw-muenchen.de. Compute:
  tas-dgx-a100-[1-4], je 8x A100-80GB; Login nur auf den Headnode.
- Muster: ssh head 'befehl' — non-interaktiv, einzeln. sbatch/squeue/sacct/
  scancel NUR dort. Kein srun --pty durch Claude (interaktiv = manuell).
- ACHTUNG (16b.5): die nicht-interaktive Shell hat SLURM NICHT im PATH
  (Binaries: /opt/slurm/bin). Fuer squeue/scontrol/sbatch/sacct IMMER eine
  Login-Shell erzwingen: ssh head 'bash -lc "squeue -u vima"'.
  Ohne bash -lc: "squeue: command not found".
- Code auf dem Cluster: ~/Code_final/ (Deploy per rsync vom lokalen Server;
  Excludes: predictions/, checkpoints/, wandb/, __pycache__/).
- Conda-Env: bevwm (Python 3.10, numpy 1.26.4). Kein Partition/Account noetig.
- Daten: BeeGFS /mnt/beegfs/ssd/lrt81-students/lrt81-vima/latents/seg/{train,val}
  (roh) bzw. latents_packed_fp16/{train,val} (fp16-Packs; OHNE seg/-Ebene —
  anders als lokal!). BeeGFS-Lesen 0.30-0.54 GB/s = Flaschenhals, darum
  fp16-Packs + /dev/shm-Staging (job_shared, ~286G, Cleanup via EXIT-trap).

### SLURM-Gotchas (teuer erkauft — einhalten)

- --mem IMMER explizit setzen, sonst wartet der Job ewig auf einen komplett
  freien Knoten.
- gres KLEIN schreiben: --gres=gpu:a100:N.
- Job IMMER aus ~/Code_final submitten; vorher mkdir -p logs
  (--output=logs/... wird vor dem cd aufgeloest).
- /dev/shm nur auf Compute-Nodes pruefen (~1008G); Headnode hat nur 63G.
- Knoten-Check vor Submit:
  for n in 1 2 3 4; do scontrol show node tas-dgx-a100-$n \
    | grep -oE 'State=[^ ]+|CfgTRES=[^ ]+|AllocTRES=[^ ]+'; done
  (frei = CfgTRES minus AllocTRES gres/gpu; IDLE+DRAIN nimmt nichts an.)
- --time aus GEMESSENEN Epochenzeiten hochrechnen:
  n_wellen x ~30 Epochen x ~650 s (Konkurrenz-Puffer) + Staging + Inference
  + 20% — NICHT aus Faustformeln (Lektion Job 122859: Welle 2 am Limit gekillt).
- WANDB_MODE=offline auf dem Cluster.
- sbatch --export: KOMMA ist der Item-Trenner der Liste. Ein Wert mit Komma
  (SWEEP_VALUES=0,0.1,0.5) wird zerlegt (-> nur "0" kommt an, Job scheitert am
  Guard). Seit 16b.6 nimmt sbatch_sweep_parallel.sh Werte per DOPPELPUNKT
  (SWEEP_VALUES=0:0.1:0.5); Leerzeichen/Komma/Doppelpunkt werden intern zu
  Leerzeichen. Parallele Sweeps aus EINER Datei:
    sbatch --job-name=sweep_lambda_mean --export=ALL,SWEEP_REGLER=lambda_mean ...
    sbatch --job-name=sweep_lambda_cos  --export=ALL,SWEEP_REGLER=lambda_cos,SWEEP_VALUES=0:0.1:0.5 ...
  (nur REGLER noetig, wenn VALUES = Default 0 0.1 0.3, wie bei lambda_mean).
- Wellen-Modus sbatch_sweep_parallel.sh: 1 <= NGPU <= |Werte|,
  ceil(|Werte|/NGPU) Wellen, EIN /dev/shm-Staging ueber alle Wellen.
- Checkpoints tragen seit 16b.4 den vollen mIoU-Plateau-Zustand; Resume von
  Alt-Checkpoints faellt kontrolliert auf Defaults zurueck (.get()-Pfad).

## Sweep-Kette (16b-Standardablauf)

1. Configs: python mk_sweep_cfgs.py ... aus config_sweep_base_cluster_fp16.yaml
   (lambda_std=1.0 fix; genau EIN Regler + checkpoints.dir je Config).
2. Diff der erzeugten Configs zeigen -> Freigabe.
3. ssh head 'cd ~/Code_final && sbatch sbatch_sweep_parallel.sh ...'
4. Pollen: ssh head 'squeue -u vima'; Logs: ssh head 'tail -n 100 ~/Code_final/logs/...'
5. Im Job: Training je Wert -> inference.py --no-save (300 Samples) ->
   run_summary.json + inference_log.json je Wert.
6. harvest_sweep.py -> CSV unter predictions/task16b/.
7. Ergebnisse gegen Signifikanzschwellen einordnen; Bericht ergaenzen.

## HDF5 / Loader

- Linux/Server: swmr=True, num_workers=16 (Cluster-Sweeps; Diagnose 16b),
  prefetch_factor=4, pin_memory NICHT setzen (Default true; false kostet
  +93 s/Epoche).
- Windows (nur Referenz): swmr=False, num_workers=0, h5_path=null (.npy).

## Schluesselpfade

- Bestes Seg-WM-Checkpoint: checkpoints/task14/phase2/best_miou.pt
- 16b-Sweep-Checkpoints: checkpoints/task16b/<regler>_sweep/
- BEVFusion-Seg-Gewichte: ~/bevfusion-seg.pth (Cluster);
  /home/vima/bevfusion/bevfusion-main/pretrained/bevfusion-seg.pth (lokal)
- Sweep-Basis-Config: config_sweep_base_cluster_fp16.yaml (Cluster) /
  config_sweep_base_fp16.yaml (lokal, 16a-Fallback)

## Grenzen dieser Datei

Diese CLAUDE.md ist Zusammenfassung, nicht Quelle. Bei Widerspruch gewinnen
die TASK*_ABSCHLUSSBERICHT.md und Arbeitsplan.py. Fehlen diese Dateien lokal,
den Nutzer fragen statt Inhalte zu erfinden.
