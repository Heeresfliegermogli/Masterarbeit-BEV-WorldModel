"""
===========================================================================
BEV WORLD MODEL — MASTERARBEIT ARBEITSPLAN
(Seg-Fokus zuerst, Det als abgetrennter Stretch-Strang;
 periodische Decode-Validierung + iteratives Loss-Engineering)
===========================================================================

VERWENDUNG:
  Jeder TASK-Block ist so geschrieben, dass er als eigenständige Aufgabe in
  einen SEPARATEN Chat kopiert werden kann. Jeder Task nennt: ZIEL, VORGEHEN
  (konkrete Schritte), ERGEBNIS/DELIVERABLE und schließt mit einem
  TASKxx_ABSCHLUSSBERICHT.md ab. Die ARCHITEKTUR- und KONTEXT-Sektionen
  oben dienen als Referenz, die in jeden Chat mitkopiert werden sollte.

===========================================================================
GRUNDSATZ-ENTSCHEIDUNG: SEG ZUERST, DET DANACH
===========================================================================

Der GESAMTE Plan bis Task 19 bezieht sich AUSSCHLIESSLICH auf den
Segmentierungs-Strang (Seg). Der Detection-Strang (Det) ist ein
EIGENSTÄNDIGER, klar abgetrennter Stretch-Block am Ende (Task 20), der nur
angegangen wird, wenn der Seg-Strang vollständig abgeschlossen ist und Zeit
bleibt.

WARUM diese Trennung (aus Experimenten Task 10/11 belegt):
  - Seg und Det nutzen KOMPLETT GETRENNTE BEVFusion-Checkpoints, -Encoder
    und -Decoder. Seg: BEVSegHead. Det: TransFusionHead.
  - Die Latents sind NICHT austauschbar: ein Seg-Latent durch den
    Det-Decoder liefert 0 Boxen (Task 11), und umgekehrt. Bestätigtes
    Faktum, keine offene Frage.
  - Unterschiedliche Auflösung: Seg = [256, 128, 128], Det = [256, 180, 180].
  - Unterschiedliche std-Charakteristik: Seg ~0.263, Det ~0.690 (2.6x).
  - Unterschiedliche Metrik: Seg = mIoU, Det = mAP/NDS.
  -> Es sind effektiv ZWEI getrennte Pipelines, die sich nur die METHODIK
     teilen (gleiche Architektur-Idee, gleiche Loss-Philosophie), aber
     getrennte Configs, Trainingsläufe, Decoder und Berichte haben.

EINORDNUNG (für die schriftliche Arbeit):
  Dass BEVFusion zwei getrennte Latent-Räume hat, ist eine Eigenheit des
  Benchmark-Settings — jede Aufgabe wird einzeln für beste Scores optimiert.
  In realen Anwendungen würde EINE gemeinsame Repräsentation für beide
  Aufgaben genutzt. Dieser Kontrast ist ein guter Diskussionspunkt, aber
  KEIN Arbeitsziel dieser Arbeit (Fusion wird nicht umgesetzt).

===========================================================================
PROJEKT-ÜBERBLICK (SEG-STRANG)
===========================================================================

Ziel:
  Ein Transformer-basiertes World Model, das aus 3 aufeinanderfolgenden
  Seg-BEV-Latents (t-2, t-1, t) den nächsten Seg-Latent (t+1) vorhersagt.
  Die predicted Latents müssen BEVFusion-Seg-decoder-kompatibel sein
  (gültige Segmentierungsmasken erzeugen). KERNZIEL: Prediction-Schärfe
  maximieren (std-Gap minimieren) über Loss-Engineering.

Input:   3 x [1, 256, 128, 128]  (Seg-Latents aus bevfusion-seg.pth, eingefroren)
Output:  1 x [1, 256, 128, 128]  (predicted nächster Seg-Latent)

ZENTRALE ERGEBNISSE (ALTE Baseline, nur Val-Set ~5743 Latents):
  Val-Loss (Phase 2, Gated Skip):  0.036193  (Epoch 28)
  Pearson r:                       0.9975
  pred_std / real_std:             0.213 / 0.290  -> std-Gap 26%  <- ZIELGRÖSSE
  neck_CosSim:                     0.8285
  mIoU:                            0.564   <- HAUPT-METRIK
  ACHTUNG: Diese Zahlen stammen vom Val-Set. Nach dem Umstieg auf den vollen
  Datensatz wird eine NEUE Baseline gerechnet (Task 13) — sie ist der
  eigentliche Referenzpunkt für alle Loss-Experimente.

Loss (BASELINE — vier Terme):
  loss = MSE (l=1.0) + 0.1*CosSim + 0.1*Dist (mean+std) + 0.1*SSIM
  (SSIM channel-gemittelt -> schwach)

Loss (ZIEL der Experimente — aufgeteilt & konfigurierbar):
  loss = MSE (l=1.0) + l_cos*CosSim + l_mean*Dist_mean
       + l_std*Dist_std (Anti-Blur, hochgewichtet) + l_grad*Gradient (NEU)

Hardware & Zeit:
  Linux-Server, TITAN RTX 24GB. Training: ~1,5 h pro Lauf (50 Epochen).
  Wenige Weights, viel RAM -> der iterative Loss-Zyklus ist gut machbar
  (viele Konfigurationen in vertretbarer Zeit).
  Storage: gelöst, kein Engpass.

===========================================================================
DIAGNOSE: WARUM DER std-GAP ENTSTEHT (Motivation Loss-Engineering)
===========================================================================

KRÄFTE-UNGLEICHGEWICHT in der aktuellen Loss:
  Richtung Blur:    MSE (l=1.0, stark) · SSIM (l=0.1, wirkungslos) · Dist-mean (l=0.1)
  Richtung Schärfe: Dist-std (l=0.1) — EINZIGE Gegenkraft, 10x schwächer
  -> pred_std << real_std mathematisch vorgezeichnet (Regression-to-Mean).

std-Berechnung (Term 3, pro Channel über H,W):
  sigma_c = sqrt( (1/(N-1)) * Summe (x_i - mu_c)^2 ),  N = 128*128 = 16384
  loss_dist = MSE(mu_pred, mu_real) + MSE(sigma_pred, sigma_real)
  -> der sigma-Summand ist der direkte Hebel gegen den std-Gap.

===========================================================================
ARCHITEKTUR REFERENZ (Seg-World-Model, aktueller Stand)
===========================================================================

  [3x Seg-Latent]   3 x [256,128,128]
         |-------------------------------+ Gated Skip (Frame t, volle Auflösung)
         v                               |
  [1. Downsampling]   AvgPool2d(k=4,s=4) -> 3 x [256,32,32]  (grid_size=32)
  [2. Token Embedding] Flatten -> 3072 Tokens [256]
  [3. Pos. Encoding]   PE_x+PE_y (256 räumlich) + PE_t (256 zeitlich)
  [4. Transformer]     4 Layer, 8 Heads, Pre-LN, FFN 256->1024->256
  [5. Output Head]     letzte 1024 Tokens -> Linear+GELU -> [256,32,32]
  [6. Upsampling Head] ConvTranspose x2 (+GroupNorm+GELU) -> Conv2d -> [256,128,128]
  [7. Gated Skip]      alpha=Sigmoid(Conv1x1); pred = alpha*trans + (1-alpha)*skip
         v
  [Predicted Seg-Latent t+1]  [256,128,128]

PARAMETER (Phase 2, N=4, Gated Skip): ~6.05M
CONFIG: config_nuscenes_val.yaml (grid_size, epochs, patience, lambda_* etc.)

===========================================================================
ZWEI-EBENEN-VALIDIERUNG (Kernkonzept, gilt ab Task 14)
===========================================================================

Es gibt ZWEI verschiedene, sauber zu trennende Mechanismen:

  EBENE 1 — INTERN, AUTOMATISCH, in JEDEM Trainingslauf:
    Bei FESTER Loss-Gleichung wird periodisch (alle N Epochen) ein
    Val-Subset DEKODIERT und die echte Metrik (mIoU) gemessen. Der beste
    Checkpoint wird nach METRIK gewählt, nicht nach Loss.
    -> holt das Beste aus EINER gegebenen Loss heraus.
    (Hinweis: bereits existierendes Early-Stopping arbeitet bisher nach
     Val-LOSS mit patience=10. Ebene 1 ergänzt/ersetzt das durch mIoU.)

  EBENE 2 — EXTERN, MANUELL, ZWISCHEN den Trainingsläufen:
    Die Loss-Gleichung SELBST wird verändert (l_std, Gradient-Loss etc.),
    je ein neuer Lauf gestartet, Ergebnisse verglichen.
    -> findet die BESTE Loss-Gleichung über mehrere Läufe.
    Das ist der wissenschaftliche Kern (Task 15).

  Merksatz: Ebene 1 = bester Checkpoint aus EINER Loss.
            Ebene 2 = beste Loss-Gleichung über MEHRERE Läufe.

ZWEI ABBRUCHKRITERIEN (nicht verwechseln!):
  a) Abbruch EINES Laufs: existiert bereits (Early-Stopping, patience=10
     auf Val-Loss; künftig auf mIoU). Frage: "Wann ist DIESES Modell
     austrainiert?"
  b) Abbruch des ZYKLUS (Ebene 2): NEU zu definieren. Frage: "Wann höre
     ich auf, neue Loss-Konfigurationen zu probieren?" Vorschlag:
     max. 6-8 Konfigurationen, ODER Stopp wenn std-Ratio > 0.9 UND
     mIoU >= Baseline. (Siehe Task 15.)

===========================================================================
TASKS 1–11: ABGESCHLOSSEN
===========================================================================

  Task 1–6:  Infrastruktur, Daten, Embedding, Transformer, Heads, Inference
  Task 5:    Training -> Val-Loss=0.036193, Pearson r=0.9975
  Task 6:    Inference -> std-Gap=26%
  Task 9b:   (geplant/teilw.) Latent-Visualisierung (Gate-Alpha etc.)
  Task 9c:   BEVFusion Seg-Decoder-Test -> neck_CosSim=0.8285
  Task 9d:   Segmentierungsmasken -> mIoU=0.564
  Task 10:   Seg vs. Det Latent-Vergleich -> CosSim(channel)=0.880,
             CosSim(spatial)=0.160, Auflösung 128 vs 180, std 0.263 vs 0.690
  Task 11:   Det Option B (Seg-Latent durch Det-Decoder) -> 0 Boxen
             -> Seg und Det endgültig als getrennte Stränge bestätigt

===========================================================================
TASK 12 — EXTRAKTION ALLER SEG-LATENTS (VOLLER DATENSATZ)
Status: NÄCHSTER SCHRITT
Chat: eigenständig | Abschluss: TASK12_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Alle Seg-Latents über das vollständige nuScenes-Set extrahieren. Bisher
  nur ~5743 Latents (Val-Set). Jetzt: volles trainval (~40000 Keyframes).
  NUR SEG — keine Det-Latents in diesem Schritt.

VORGEHEN:
  1. BEVFusion-Inference mit bevfusion-seg.pth über das volle trainval-Set
     (im Docker-Container, torchpack dist-run wie gehabt)
     -> ~40000 Seg-Latents [256,128,128] als .npy
  2. Split nach nuScenes-Konvention definieren: 700 train / 150 val Szenen
     (NICHT der bisherige interne 70/15/15-Split auf nur Val!)
  3. HDF5-Cache neu aufbauen (build_hdf5_cache.py, swmr=True, num_workers=4)
  4. config aktualisieren: neue Pfade, neuer Split, h5_path auf vollen Cache
  5. Sanity-Check: Sample-Count, std bestätigen (~0.263), keine korrupten
     Frames, Shape [256,128,128] durchgehend

DELIVERABLE:
  - vollständiger Seg-Latent-Datensatz (.npy + HDF5-Cache)
  - aktualisierte config mit train/val-Split
  - Sanity-Report (Counts, std, Shapes)

AUFWAND: ~2-3 Tage (Inference läuft automatisch)

===========================================================================
TASK 13 — NEUE BASELINE AUF VOLLEM DATENSATZ
Status: AUSSTEHEND (direkt nach Task 12, VOR allen Loss-Experimenten)
Chat: eigenständig | Abschluss: TASK13_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Das BESTEHENDE Modell mit der UNVERÄNDERTEN Baseline-Loss auf dem vollen
  Datensatz trainieren und evaluieren. Das ist der gültige Referenzpunkt
  für alle Loss-Experimente (Task 15). KRITISCH: Ohne diese neue Baseline
  würde man später Loss-Varianten (auf vollen Daten) gegen die alte
  Val-Baseline vergleichen — ungültig, weil sich zwei Dinge gleichzeitig
  geändert hätten (Datenmenge UND Loss).

VORGEHEN:
  1. Training mit unveränderter 4-Term-Loss auf vollem Datensatz
     (config: epochs=50, patience=10, lambda_cos=0.1, lambda_dist=0.1)
  2. Standard-Metriken erheben und als BASELINE einfrieren:
     - Val-Loss, Pearson r
     - std_pred / std_real (std-Ratio)  <- zentrale Zielgröße
     - neck_CosSim, neck_MSE (Seg-Decoder)
     - mIoU pro Klasse + mean mIoU
  3. Vergleich der vollen-Daten-Baseline gegen die alte Val-Baseline
     (erwartbar: bessere mIoU durch mehr Trainingsdaten)

DELIVERABLE:
  - eingefrorene Baseline-Metriken (Tabelle) auf vollem Datensatz
  - bester Checkpoint dieser Baseline
  - kurze Einordnung: was hat der größere Datensatz gebracht?

AUFWAND: ~1 Tag (Training ~1,5 h + Auswertung)

===========================================================================
TASK 14 — PERIODISCHE DECODE-VALIDIERUNG IN DAS TRAINING EINBAUEN (EBENE 1)
Status: AUSSTEHEND (nach Task 13, vor den Loss-Experimenten)
Chat: eigenständig | Abschluss: TASK14_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Das Training so erweitern, dass es periodisch die predicted Latents durch
  den BEVFusion-Seg-Decoder schickt und die echte mIoU misst — und den
  besten Checkpoint nach mIoU (statt nach Val-Loss) auswählt. Das ist
  Ebene 1 (siehe Zwei-Ebenen-Validierung oben).

AUSGANGSLAGE (gut):
  - inference_decoder.py baut den Seg-Decoder (SECOND + SECONDFPN +
    BEVSegHead) bereits direkt aus dem .pth zusammen — Extraktion existiert.
  - train_linux.py hat bereits validate(), find_best_checkpoint, wandb-
    Logging und visualize_predictions — nur arbeitet validate() bisher rein
    im Latent-Raum (MSE), nicht mit Decodierung.

VORGEHEN:
  SCHRITT 1 — Decoder in der Trainingsumgebung verfügbar machen:
    Prüfen, ob mmdet3d + mmcv in der Trainingsumgebung lauffähig sind.
    -> Weg A (bevorzugt): ja -> Decoder direkt im Trainingsskript laden.
    -> Weg B (Fallback, GLEICHWERTIG): nein -> Decode als SEPARATER Schritt
       nach jedem Lauf (Training schreibt val-Predictions als .npy, ein
       Decode-Skript im Docker liest, dekodiert, gibt mIoU zurück).
    WICHTIG: Bei nur 1,5 h Trainingszeit ist Weg B kaum ein Nachteil — der
    Hauptnutzen von In-Loop-Decode (lange Läufe früh abbrechen) entfällt
    hier ohnehin. NICHT tagelang an mmdet3d-Installation festbeißen; wenn
    Weg A nicht zügig klappt, Weg B nehmen.

  SCHRITT 2 — Validierungs-Funktion erweitern:
    validate_seg(pred_latents) -> mIoU
    (BEVSegHead aus inference_decoder.py + mIoU-Logik aus Task 9d kapseln)

  SCHRITT 3 — festes Val-Subset definieren:
    ~200-500 Samples, fix (reproduzierbar), reicht als Trend-Indikator;
    nicht alle ~6000 val-Samples jede N-te Epoche dekodieren (zu teuer).

  SCHRITT 4 — Checkpoint-Auswahl umstellen:
    save_best nach mIoU statt nach Val-Loss (oder beide loggen). mIoU-Kurve
    über Epochen in wandb mitloggen.

  SCHRITT 5 — verifizieren:
    Einen kompletten Lauf fahren und prüfen, dass mIoU-Kurve, bester
    Checkpoint und Logging sauber funktionieren.

DELIVERABLE:
  - erweitertes train_linux.py mit Decode-Validierung (Weg A oder B)
  - validate_seg()-Funktion + festes Val-Subset
  - ein verifizierter Lauf mit mIoU-Kurve über Epochen
  - kurze Notiz: Weg A oder B gewählt, warum

AUFWAND: ~1-2 Tage (Seg; Decoder-Extraktion existiert großteils)

===========================================================================
TASK 15 — LOSS-ENGINEERING ALS KONTROLLIERTE DOSIS-WIRKUNGS-STUDIE
            (EBENE 2, KERN)
Status: 15.1-15.3 + 15.3B ABGESCHLOSSEN/laufend. Die REGLER-SWEEPS (frueher
        15.4-15.8) sind in Task 16a herausgeloest (eigenstaendige Durchfuehrung
        + Automatisierung, siehe dort). Task 15 umfasst damit: Loss-Apparatur
        (15.1), Rauschboden (15.2), lambda_std-Sweep (15.3), Pipeline-Umbau
        (15.3B). Die restlichen Regler laufen unter 16a.
Chat:   PRO SUBTASK ein eigener Chat | Abschluss je Subtask:
        TASK15.1_ABSCHLUSSBERICHT.md ... TASK15.3B.x_ABSCHLUSSBERICHT.md
        (Sweep-Berichte: TASK16a.x, siehe Task 16a)
===========================================================================

WISSENSCHAFTLICHER KERN (Reframing):
  Das HAUPTSTEUERELEMENT der Untersuchung ist der Loss-Term. Jeder Sweep-Lauf
  variiert GENAU EINE Groesse (ein lambda) und haelt alles andere fix -> eine
  Dosis-Wirkungs-Studie ("was passiert, wenn wir den Term hoch-/runterdrehen").
  Das Deliverable ist nicht nur die beste Konfig, sondern eine REPRODUZIERBARE
  Aussage ueber die Wirkung jedes Terms.

ZIEL & NORDSTERN (wichtige Korrektur ggü. alter Fassung):
  - mIoU = PRIMAERES Ergebnis (Entscheidungs- und Abbruchkriterium).
  - std-Ratio = MANIPULATIONS-CHECK (hat der Regler die Varianz bewegt?),
    NICHT mehr das Ziel. Ein Modell, das std-Ratio durch halluzinierte
    Hochfrequenz hebt, kann SCHLECHTERE mIoU haben (Super-Resolution-Trade-off).
  - Ein mIoU-Plateau bei persistierendem std-Gap ist ein GUELTIGES ERGEBNIS
    (gemessene deterministische Decke -> motiviert Task 18), kein Scheitern.

GEMEINSAME REGELN (galten fuer 15.3 und gelten UNVERAENDERT WEITER fuer alle
16a-Sweeps -- hier zentral dokumentiert, damit 16a nicht duplizieren muss):
  - eine unabhaengige Variable pro Lauf (ein lambda), alles andere fix
  - Aenderung NUR via YAML -> KEIN Code-Edit zwischen Laeufen
    (ein Code-Edit ist eine unkontrollierte Variable; genau dafuer ist 15.1 da)
  - identisch ueber alle Laeufe: Datensatz/Split, Optimizer, lr, weight_decay,
    epochs, n_frames, d_model/n_layers/n_heads
  - AvgPool(k=4) UND Architektur EINGEFROREN auf Baseline-Stand
    -> Architektur-Aenderungen sind ein CONFOUNDER fuer die Loss-Studie und
       gehoeren nach Task 18 (auch wenn AvgPool der groessere mIoU-Hebel waere)
  - Checkpoint-Wahl: mIoU-Gate (Task 14), nicht Val-Loss
  - dasselbe FESTE Decode-Val-Subset ueber alle Laeufe
  - pro Lauf geloggt: Config + Git-Commit + Seed (wandb erledigt das)
  - jeder Sweep verankert an zwei Punkten: lambda = 0 (Term AUS) UND
    lambda = Baseline-Wert
  - Zielgroessen-Hierarchie ueberall gleich:
        mIoU                 = primaeres Ergebnis
        std-Ratio            = Manipulations-Check
        MSE/Pearson/CosSim   = sekundaer, mechanistisch

---------------------------------------------------------------------------
15.1 — LOSS-UMBAU + KORREKTHEITS-CHECK  (chat-0, reine Apparatur)
       Bericht: TASK15.1_BERICHT.md
---------------------------------------------------------------------------
  WARUM ZUERST: hier entsteht KEIN wissenschaftliches Ergebnis, nur das
    Werkzeug. Erst bauen und beweisen, dass es nichts verfaelscht, dann
    einfrieren. Ohne bestandenen Check misst jeder spaetere Sweep den Bug
    statt das lambda -> Fundament der gesamten Studie.
  VORGEHEN:
    a) dist-Term aufteilen in loss_mean + loss_std, je eigenes lambda
       -> erschafft ueberhaupt erst den Regler fuer die std-Komponente
    b) Gradient-Term ergaenzen (Anti-Blur), eigenes lambda_grad, DEFAULT 0:
         pred_dx = pred[...,:,1:] - pred[...,:,:-1]
         pred_dy = pred[...,1:,:] - pred[...,:-1,:]
         loss_grad = MSE(pred_dx,targ_dx) + MSE(pred_dy,targ_dy)
    c) SSIM abschaltbar machen (lambda_ssim, 0 moeglich)
    d) ALLE lambda als YAML-Parameter:
         lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim
       -> jede spaetere Ablation laeuft NUR ueber die Config
    e) KORREKTHEITS-CHECK (Regressionstest): lambda so setzen, dass die ALTE
       Baseline-Loss exakt nachgebildet wird (lambda_mean = lambda_std =
       altes dist-lambda; lambda_grad = 0; SSIM wie Baseline) -> auf einem
       festen Batch NUMERISCH IDENTISCHER Loss-Wert wie Task 13.
  DELIVERABLE: umgebaute compute_loss(), Config-Schema, bestandener Check.
  AUFWAND: ~0,5 Tag, KEIN Training.

---------------------------------------------------------------------------
15.2 — RAUSCHBODEN  (Messung, kein Regler)
       Bericht: TASK15.2_BERICHT.md
---------------------------------------------------------------------------
  WARUM: ein lambda-Effekt ist erst ein BEFUND, wenn Delta groesser ist als
    die Lauf-zu-Lauf-Streuung bei IDENTISCHER Konfig. Sonst interpretiert man
    Rauschen. (Methodik aus LeWM Tab. 5 uebernommen: mehrere Seeds, mean +/- std.)
  VORGEHEN: verifizierte Baseline-Config 2-3x mit verschiedenen Seeds fahren;
    mIoU- und std-Ratio-Streuung als Fehlerbalken festhalten.
  DELIVERABLE: mean +/- std je Metrik -> Signifikanzschwelle fuer alle Sweeps.
  AUFWAND: ~1 Tag (2-3 Laeufe).

---------------------------------------------------------------------------
15.3 — SWEEP lambda_std  (KERN-REGLER)
       Bericht: TASK15.3_BERICHT.md
---------------------------------------------------------------------------
  unabhaengige Variable: lambda_std. Stufen: 0 . Baseline . 0.3 . 0.5 . 1.0.
  WARUM: direkter Hebel auf den std-Gap; die zentrale Dosis-Wirkungs-Kurve.
  ERWARTUNG (= das Ergebnis): std-Ratio steigt monoton, mIoU als umgekehrtes U
    -> Sweet-Spot plus empirischer Nachweis, dass weiteres Hochdrehen mIoU
    kostet (die Halluzinations-Grenze, an DIESEM Setup gemessen).
  DELIVERABLE: mIoU(lambda_std), std-Ratio(lambda_std), bestes lambda_std.
  AUFWAND: ~0,5 Tag pro Stufe.

---------------------------------------------------------------------------
15.3B — DATEN-PIPELINE: MEMMAP-PACKING + NODE-LOKALES STAGING (NACHTRAG,
         MOTIVIERT DURCH CLUSTER-MIGRATION -- EINGESCHOBEN ZWISCHEN 15.3
         UND 15.4)
         Berichte: TASK15.3B.1_BERICHT.md, TASK15.3B.2_BERICHT.md,
                   TASK15.3B.3_BERICHT.md
---------------------------------------------------------------------------
  STAND BEI EINFUEGUNG: 15.1-15.3 abgeschlossen (auf ALTER, ungepackter
    Pipeline). Dieser Block MUSS vor Fortsetzung von 15.4 durchlaufen sein
    -- 15.4 und alle folgenden Sweeps laufen unter der NEUEN, gepackten
    Pipeline.

  FORTSCHRITT (Stand laufender Umbau, in eigenem Chat):
    - 15.3B.1 CODE FERTIG + lokal verifiziert: pack_latents.py, packed_dir-
      Schalter (data.use_packed), Manifest, Timing-Instrumentierung. Innerer
      Check bit-identisch (max|Delta|=0.0 ueber 500 Frames, lokal UND auf
      BeeGFS). Packen auf beiden Umgebungen erfolgreich (lokal 6019 val in
      2.4min / 28130 train in 15.2min; Cluster analog auf BeeGFS).
    - 15.3B.1 AEUSSERER Anker laeuft auf dem Cluster (Job A, use_packed=true,
      Seed 42). mIoU-Verlauf bestaetigt bisher das erwartete Muster
      (0.57 -> 0.64 monoton). Formaler Abschluss sobald ein paar Decode-
      Punkte + saubere data%-Epochen vorliegen (NICHT bis Konvergenz noetig,
      da innerer Check die Bit-Gleichheit schon garantiert).
    - EMPIRISCHE KORREKTUR am urspruenglichen Speedup-Optimismus (WICHTIG,
      siehe 15.3B.3): Packing allein bringt auf dem Cluster WENIGER als der
      2.60x-Mini-Benchmark suggerierte, sobald mehrere I/O-Jobs auf DEMSELBEN
      Knoten laufen -- sie teilen sich die EINE BeeGFS-Netzwerkleitung des
      Knotens und bremsen sich gegenseitig (live beobachtet: zwei Jobs auf
      tas-dgx-a100-1, beide deutlich langsamer als lokal). data% blieb bei
      ~30% (I/O weiter spuerbarer Anteil). -> Packing ist notwendige
      VORBEDINGUNG fuer Staging (ein File statt zehntausender), aber NICHT
      allein die Loesung. Der eigentliche Hebel ist 15.3B.3 (node-lokales
      Staging), das damit vom "optionalen Nachgedanken" zum KERN-SCHRITT
      hochgestuft wird (siehe dort).

  ANLASS:
    Cluster-Migration deckte auf, dass das bestehende Zugriffsmuster (pro
    Sample 4 einzelne .npy-Reads, zufaellig ueber Zehntausende Dateien
    verteilt) auf BeeGFS deutlich staerker einbricht als lokal:
      - GPU-Auslastung nur 22-35% (lokal: 60-70%) trotz num_workers 4->16
      - mehr num_workers half kaum (weder lokal noch cluster-seitig)
      - RAM-Caching (--mem=600G) half nicht verlaesslich (Cgroup-Reclaim
        auf geteiltem Knoten, buff/cache blieb bei ~182G statt ~535G)
    Verifiziert per Mini-Benchmark (eine Szene, echter Read via .sum(),
    Summe-Check bestaetigt Korrektheit -- kein Wiederholung des HDF5-
    Fehlschlags, diesmal empirisch abgesichert statt nur behauptet):
      Cluster (BeeGFS, DGX-Knoten): 8-Worker 0.76s -> Memmap seriell 0.29s
         = 2.60x
      Lokal  (alre-server, NVMe):   4-Worker 0.32s -> Memmap seriell 0.23s
         = 1.39x
    Effekt real auf BEIDEN Umgebungen, auf dem Cluster deutlich staerker
    (Netzwerk-Metadaten-Rundlauf bei BeeGFS pro Datei-Open, lokal nur
    OS-Overhead ohne Netzwerk-Hop).

  DESIGN-ENTSCHEIDUNG (wichtig, verhindert Doppelpflege):
    EIN gemeinsamer Codepfad fuer lokal UND Cluster, kein separates System.
      - data.packed_dir (neu, optional) in derselben YAML wie
        train_sources/val_sources -- gesetzt: Laufzeit liest per Memmap;
        ungesetzt: exakt heutiges Verhalten (Fallback, kein Unterschied)
      - EIN Memmap PRO SPLIT (train/val), NICHT pro Szene und NICHT pro
        Fenster -- kein neuer Shuffle-Buffer-Sampler noetig, bestehendes
        shuffle=True bleibt unveraendert, Retraining-Frage dadurch
        praktisch hinfaellig (Sampling-REIHENFOLGE aendert sich nicht,
        nur wie Bytes geholt werden)
      - Begruendung gegen "pro Fenster packen": Sliding-Windows
        ueberlappen (Stride 1) -> ~4x Redundanz/Speicherbedarf, kaum
        weniger Opens
      - Begruendung gegen "pro Szene packen + Buffer": mehr Engineering
        (Shuffle-Buffer, Sampler-Umbau), Sampling-Semantik veraendert
        sich minimal -> Memmap pro Split ist einfacher UND naeher am
        Status quo
      - Preprocessing-Skript liest dieselbe Config (train_sources/
        val_sources) wie train_linux.py -- keine zweite, separat
        gepflegte Pfad-Definition

  --- 15.3B.1: MEMMAP-PACKING-UMBAU (Kern-Umbau) [CODE FERTIG, VERIFIZIERT] ---
    Bericht: TASK15.3B.1_ABSCHLUSSBERICHT.md
    STATUS: Code vollstaendig, lokal + Cluster gepackt, innerer Check
      bit-identisch bestanden. Aeusserer Anker laeuft (Job A). Punkte a-i
      unten sind UMGESETZT (Haken in Klammern).
    VORGEHEN:
      a) pack_latents.py: nimmt dieselbe YAML-Config, ruft
         _load_scenes_from_pkl() UNVERAENDERT auf (identische Frame-Menge/
         -Reihenfolge wie zur Laufzeit), zaehlt Gesamtframes, legt leeres
         Memmap [N,256,128,128] float32 per np.lib.format.open_memmap()
         an, schreibt jeden Frame einmalig (inkl. squeeze(axis=0),
         einmalig hier statt ~108.500x zur Laufzeit)
         EXPLIZITE ANFORDERUNG (nicht nur Nebeneffekt): Frames werden
         SZENENGEORDNET geschrieben (Szene fuer Szene, innerhalb der
         Szene zeitlich aufsteigend). Dann ist jedes Sliding-Window ein
         ZUSAMMENHAENGENDER Slice von 4 Zeilen = EIN sequenzieller
         32-64MB-Read statt 4 verstreuter Reads. Zufaellig bleibt nur
         die Fensterposition -- bei Reads dieser Groesse ist die
         Seek-Latenz gegenueber der Transferzeit zweitrangig (der
         BeeGFS-Engpass waren die vielen kleinen Opens, nicht die
         Zufaelligkeit an sich)
      b) KEIN separates Index-File: BEVLatentDataset berechnet die
         Zeilennummer jedes Frames zur Laufzeit selbst (laufender Zaehler
         beim Durchlaufen der Szenen) -- deterministisch bei gleicher
         Config, kein Artefakt, das mit den Daten auseinanderlaufen kann
      c) BEVDatasetConfig: neues optionales Feld packed_dir:
         Optional[str] = None (parallel zum alten h5_path, ohne dessen
         SWMR/Locking-Gepaeck)
      d) _load_latent(): wenn packed_dir gesetzt ->
         self._packed_array[row] statt np.load(); sonst exakt heutiger
         Pfad
      e) Speicherort: /mnt/beegfs/ssd/lrt81-students/lrt81-vima/
         latents_packed/{train,val}/packed.npy -- Original-Einzeldateien
         bleiben unangetastet als Referenz/Fallback stehen
      f) Zusatz-Check (guenstig, gleicher Umbau): pruefen ob
         train_linux.py batch.to(device, non_blocking=True) nutzt --
         pin_memory=True ist schon gesetzt, non_blocking erst macht den
         asynchronen Transfer tatsaechlich wirksam
      g) MANIFEST beim Packen schreiben (manifest.json neben packed.npy):
         split, dtype, shape [N,256,128,128], num_frames, config_hash,
         git_commit, source_root, scene_order_hash, created_at.
         Zweck: Audit-Artefakt gegen stille Config-/Reihenfolge-
         Mismatches (die Fehlerklasse, die man Monate spaeter nicht
         mehr findet) + Reproduzierbarkeits-Doku fuer die Thesis.
         WICHTIG: das Manifest wird NICHT als Index im Training benutzt
         (Zeilennummern werden weiterhin zur Laufzeit berechnet, siehe
         b) -- es dient nur der Pruefung/Dokumentation
      h) Memmap LAZY PRO WORKER oeffnen: das Memmap-Handle NICHT im
         Hauptprozess erzeugen und ueber die Fork-Grenze teilen, sondern
         in _load_latent() erst beim ersten Zugriff im jeweiligen
         Worker-Prozess oeffnen:
           if self._packed_array is None:
               self._packed_array = np.load(path, mmap_mode="r")
         Standard-Pattern gegen Fork/Spawn-Probleme bei num_workers > 0
      i) TIMING-INSTRUMENTIERUNG im Trainingsloop (einmalig einbauen,
         bleibt drin): pro Batch data_time (Warten auf den Loader) und
         compute_time (Forward/Backward) per time.perf_counter() messen,
         pro Epoche den Anteil data_time/(data_time+compute_time)
         loggen. Interpretation: hoch -> I/O weiter Engpass, niedrig ->
         GPU ist Engpass, Datenpipeline ausreichend. Das macht das
         Entscheidungs-Gate (15.3B.3) MESSBAR statt per nvidia-smi
         geschaetzt
    KORREKTHEITS-CHECK (zweistufig, Pflicht vor Uebernahme):
      - innere Ebene [BESTANDEN]: Stichprobe (~500 Tokens) _load_latent()
        alter vs. neuer Pfad, EXAKTE Gleichheit erwartet (reines Umkopieren,
        keine verlustbehaftete Transformation). Ergebnis: max|Delta|=0.0,
        500/500 exakt gleich, lokal UND auf BeeGFS. Laeuft automatisch als
        Teil jedes Pack-Laufs.
      - aeussere Ebene [LAEUFT, Job A]: Seed-42-Anker (lambda_std=0.1,
        Task-14-Baseline) unter neuer Pipeline, mIoU/std-Ratio gegen
        15.2-Rauschboden (|Delta mIoU|<0.014, |Delta std-Ratio|<0.009).
        WICHTIG: der Anker muss NICHT bis zur Konvergenz laufen. Da der
        innere Check bereits Bit-Gleichheit des Datenpfads beweist, dient
        der aeussere Anker nur als Bestaetigung, dass unter echter
        Trainingsdynamik nichts am Loader kaputt ist. Ein paar Decode-Punkte
        (Epoche 0/5/10), die dem lokalen mIoU-Verlauf folgen, plus saubere
        data%-Epochen genuegen -> danach Job killen (spart mehrere Stunden
        Cluster-Zeit bei ~30min/Epoche). Der KONVERGIERTE mIoU-Wert kann
        spaeter im Rahmen des ersten schnellen Staging-Laufs (15.3B.3)
        sauber dokumentiert werden -- zwei Fliegen, eine Klappe.
    DELIVERABLE: pack_latents.py, erweiterte BEVDatasetConfig/
      BEVLatentDataset, bestandener zweistufiger Check, packed_dir
      aktiviert in Cluster- UND lokaler Config
    AUFWAND: ~1-2 Tage Implementierung + Preprocessing-Laufzeit (vorher
      an Teildatensatz messen, z.B. nur val/95GB, dann hochrechnen --
      nicht raten)

  --- REIHENFOLGE-ENTSCHEIDUNG (NEU, aus dem laufenden Umbau) ---
    Urspruenglich stand 15.3B.2 (float16) vor 15.3B.3 (Staging). Das wird
    UMGEDREHT, aus zwei Gruenden:

    (A) SAUBERE ATTRIBUTION: float16 ist verlustbehaftet, Staging nicht.
        Schaltet man beide gleichzeitig ein und das mIoU weicht ab, ist
        nicht zuordenbar, ob float16 (erwartbar) oder ein Staging-Bug
        (nicht erwartbar) die Ursache ist. Getrennt und in dieser
        Reihenfolge (erst verlustfreies Staging, dann float16) bleibt jede
        Abweichung eindeutig einer Ursache zuordenbar.

    (B) STAGING IST DER EIGENTLICHE HEBEL, float16 primaer sein ENABLER:
        Der Cluster-Befund (data% ~30% trotz Packing, Netzwerk-Leitung als
        geteilter Engpass) zeigt, dass Packing allein das I/O-Problem nicht
        loest. Node-lokales Staging (RAM) nimmt das Netzwerk ganz aus dem
        Loop -- DAS ist der Durchbruch-Kandidat. float16 ist dann vor allem
        wichtig, WEIL es das Staging leichter macht (halbe Groesse -> halber
        --mem-Bedarf -> weniger OOM-Risiko -> mehr parallele Jobs pro Knoten
        moeglich), nicht primaer wegen der Bytes im Loop.

    NEUE REIHENFOLGE innerhalb 15.3B:
      15.3B.1 (Packing, fp32)  -> FERTIG/laeuft
      15.3B.3 (Staging, fp32)  -> ALS NAECHSTES (verlustfreier I/O-Beweis)
      15.3B.2 (float16)        -> DANACH (Groessenreduktion + eigener
                                  mIoU-Check, macht Staging leichtgewichtiger)
      [optional 15.3B.3b: Staging + float16 kombiniert, finaler Zustand]

    EMPIRISCHE VORAB-INDIKATION (dd-Mikrobenchmark, bewusst als grob
    markiert): /dev/shm-Read auf freiem DGX-Knoten ~4.8 GB/s (single-stream,
    also UNTERSCHAETZT -- tmpfs skaliert mit parallelen Workern, echtes
    Muster liest mit num_workers parallel). BeeGFS grob ~1-2 GB/s. Der
    Latenz-Vorteil (kein Netzwerk-Rundlauf pro Read) kommt obendrauf und
    wird von dd NICHT erfasst. -> Staging klar gerechtfertigt, aber die
    konkrete Speedup-Zahl bleibt dem echten data%-Vergleich vorbehalten
    (NICHT auf eine Zahl vorab festnageln).


  --- 15.3B.3: NODE-LOKALES STAGING (/dev/shm) [KERN-SCHRITT, HOCHGESTUFT] ---
    Bericht: TASK15.3B.3_ABSCHLUSSBERICHT.md
    STATUS-AENDERUNG: War urspruenglich "Vormerkung, nicht terminiert,
      optional". Durch den Cluster-Befund (Packing allein bringt data% nur
      auf ~30%, Netzwerk bleibt geteilter Engpass) HOCHGESTUFT zum
      Kern-Schritt und ALS NAECHSTES nach 15.3B.1 (siehe Reihenfolge-
      Entscheidung). Erster Durchlauf bewusst mit float32 (verlustfrei),
      um den reinen I/O-Effekt sauber von der float16-Frage zu trennen.

    ZWECK: Packing (15.3B.1) und float16 (15.3B.2) VERKLEINERN die I/O-Last,
      ELIMINIEREN sie aber nicht -- pro Epoche wird das gepackte File
      weiterhin komplett ueber BeeGFS (Netzwerk) gelesen, und mehrere Jobs
      auf einem Knoten teilen sich diese eine Leitung. Node-lokales Staging
      nimmt das Netzwerk-I/O ganz aus dem Trainings-Loop: einmalig pro
      Job-Start ins knoten-lokale RAM-Dateisystem kopieren, danach mit
      RAM-Speed lesen.

    NODE-LOKALE FAKTEN (heute verifiziert, ersetzen frueheres Raten):
      - /dev/shm auf den DGX-Rechenknoten: 1008G (~50% der 2TB RAM),
        node-lokal und fuer normale Nutzer schreibbar (kein root noetig,
        kein /raid-Rechte-Problem). ACHTUNG: auf dem Kopfknoten nur 63G --
        immer auf dem RECHENKNOTEN pruefen (srun ... df -h /dev/shm), nicht
        auf dem Headnode.
      - Passt bequem: train fp32 440G + val 95G = 535G in 1008G, mit ~470G
        Puffer. Mit float16 (15.3B.2) nur noch ~268G -> noch mehr Spielraum.
      - Alles in /dev/shm zaehlt gegen das SLURM-Cgroup---mem-Limit des Jobs
        -> --mem entsprechend hoch: fp32 ~600-700G, fp16 ~350G. Das ist auch
        die wahrscheinlichste ERKLAERUNG fuer das alte "RAM wird nicht voll"-
        Raetsel (--mem=600G brachte nur ~182G buff/cache): nicht der RAM
        fehlte, das Cgroup-Limit hinderte den Kernel am Cachen. tmpfs umgeht
        das, weil es explizit allokiert statt opportunistisch cacht.
      - PARALLELITAETS-BONUS (Klaerung der offenen Frage aus dem
        Zwischenbericht): /dev/shm ist ein GETEILTES Dateisystem des Knotens.
        Mehrere Jobs auf demselben Knoten koennen DIESELBE /dev/shm/
        packed.npy mitlesen -> nur EINE Kopie im RAM noetig, nicht pro Job
        eine. Und weil dann aus RAM statt ueber die eine Netzwerkleitung
        gelesen wird, faellt der geteilte Engpass weg, der paralleles
        Training bisher sabotierte. Das macht den Job-Ebenen-Parallelismus
        (mehrere Sweeps gleichzeitig, geplant fuer 15.4+) erst real
        nutzbar -- ist aber eigenstaendig zu testen (zwei GEPACKTE Jobs auf
        einem Knoten: verschlechtert sich data% ggue. Solo? offene Messung).

    HEBEL, nach Prioritaet:

      (1) /raid (node-lokaler NVMe-Scratch, RAID0 8x NVMe laut
          cluster_anleitung) -- theoretisch staerkster PERSISTENZ-Hebel
          (ueberlebt Job-Ende, kein Neu-Kopieren pro Job), ABER weiter
          blockiert.
          STAND (Erinnerung): /raid ist root:root, 755 -- nur Lesen.
            Anfrage an Anton/Thorsten; laut Rueckmeldung ist Anton NICHT
            der richtige Ansprechpartner, offen wer es ist. -> NICHT als
            gesetzt einplanen. Wenn doch: gepackt (fp16 ~268G bzw. fp32
            535G) einmalig nach /raid, dann NVMe-Speed OHNE Neu-Kopieren
            pro Job (Vorteil ggue. /dev/shm, das pro Job neu kopiert).
          -> Solange ungeklaert: Weg (2) ist der reale, aktive Plan.

      (2) tmpfs-Staging (/dev/shm) -- AKTIVER Weg (admin-frei, ohne
          /raid-Rechte, node-lokal auf 1008G verifiziert).
          VORGEHEN (konkret, fuer das sbatch-Skript):
            a) Am Job-Anfang node-lokale Groesse pruefen (df -h /dev/shm auf
               dem Rechenknoten) -- fail-fast, falls durch andere Nutzer zu
               voll (der Knoten ist Multi-Tenant, war schon mit ~51G von
               fremd belegt).
            b) packed.npy von BeeGFS nach /dev/shm/<eindeutig>/ kopieren,
               ABER mit Existenz-Check: liegt es schon dort (anderer Job auf
               demselben Knoten hat's kopiert), NICHT neu kopieren -> eine
               geteilte Kopie. Analog zur idempotenten Skip-Logik von
               pack_latents.py.
            c) packed_dir (bzw. neuer Schalter) auf den /dev/shm-Pfad zeigen
               lassen. Zwei Umsetzungsvarianten (Design-Entscheidung offen):
               - Config-Schalter stage_to_shm: true (analog use_packed,
                 eleganter, wiederverwendbar; train_linux.py kopiert selbst
                 und biegt packed_dir intern um), ODER
               - reiner cp-Schritt im sbatch-Skript + packed_dir manuell auf
                 /dev/shm (kein Python-Eingriff, schneller zusammengesetzt).
               Tendenz: Config-Schalter (konsistent mit use_packed-Muster).
            d) --mem hoch: fp32 ~600-700G, fp16 ~350G (siehe node-lokale
               Fakten oben). Der einmalige Kopier-Overhead (grob 5-15min bei
               BeeGFS-Speed) ist bei mehrstuendigen Laeufen vernachlaessigbar
               -- explizit vom Nutzer bestaetigt als irrelevant ggue. dem
               Durchsatzgewinn.
          NEBENEFFEKT: die 4x-Leseredundanz der ueberlappenden Sliding-
            Windows (Stride 1) wird irrelevant, sobald aus RAM gelesen wird.
          TRADEOFF (ehrlich vermerkt): hohe --mem-Anforderung kann die
            SLURM-Queue-Zeit erhoehen. Bei einem geteilten Knoten mit ~1TB
            /dev/shm real gegen den Durchsatzgewinn abwaegen, nicht blind
            maximieren -- fp16 (15.3B.2) entschaerft das (halber --mem).

      (3) DataLoader-Tuning ERNEUT pruefen -- NACH Packing/Staging, nicht
          davor.
          BEGRUENDUNG: num_workers 4->16 half bisher NICHT, weil die
            Metadaten-Latenz pro Datei-Open nicht parallelisierbar war
            (siehe Cluster-Diagnose). Nach Packing (ein File) bzw. Staging
            (RAM) faellt dieser Grund weg -> Worker skalieren dann wieder.
          KONKRET: num_workers erneut hochziehen, persistent_workers=True,
            prefetch_factor erhoehen, pin_memory=True + non_blocking=True
            (Letzteres ohnehin schon Zusatz-Check in 15.3B.1.f).

    NICHT LOHNEND (bewusst verworfen, hier nur zur Vollstaendigkeit):
      - Kompression (z.B. zstd): zwar sauberer als der gescheiterte
        HDF5+gzip-Versuch (Task 13), aber ueberfluessig, sobald (1) oder (2)
        greift -- der Engpass ist dann nicht mehr das Lesevolumen.
      - Multi-GPU/DDP: bei Daten-Hunger (GPU 22-35%) wuerde es die I/O-Last
        nur vervielfachen. Der sinnvolle Mehr-GPU-Hebel bleibt auf
        JOB-Ebene (unabhaengige Sweep-Laeufe parallel), nicht im Loop.

    ENTSCHEIDUNGS-GATE (gilt fuer alle drei Hebel): nach JEDEM umgesetzten
      Schritt (Packing / float16 / Staging) eine echte Epoche auf dem
      Cluster messen und GPU-Auslastung pruefen -- Messgroesse ist der
      data_time-Anteil aus der Timing-Instrumentierung (15.3B.1.i),
      nicht nur nvidia-smi. Sobald die GPU wieder der
      Engpass ist (Auslastung hoch, Epochenzeit deutlich unter der lokalen
      ~800s-Marke), ist das I/O-Problem geloest -> keine weiteren
      Staging-Schritte noetig. Dasselbe Zeit-Gate wie in 15.8 Punkt 5.

    DELIVERABLE: sbatch-Staging-Skript (/dev/shm, mit node-lokaler Groessen-
      pruefung + idempotentem Kopier-Check), ggf. Config-Schalter
      stage_to_shm, gemessene data%/Epochenzeit vorher (BeeGFS-gepackt,
      Baseline aus Job A) vs. nachher (/dev/shm), Entscheidung ob Staging
      produktiv genutzt wird. ERSTER LAUF: fp32 (verlustfrei), damit der
      I/O-Effekt sauber isoliert ist (mIoU MUSS exakt gleich Job A bleiben,
      da bit-identischer Datenpfad). Gleichzeitig Gelegenheit, den
      konvergierten mIoU-Anker-Wert schnell + sauber zu dokumentieren.
    AUFWAND: gering (~0.5 Tag Skripting). Hauptnutzen/Aufwand-Verhaeltnis
      sehr guenstig -- und laut Cluster-Befund der Schritt mit dem groessten
      erwarteten Effekt auf die Trainingszeit.

  --- 15.3B.2: FLOAT16-STORAGE-VALIDIERUNG (baut auf 15.3B.1 auf) ---
    Bericht: TASK15.3B.2_ABSCHLUSSBERICHT.md
    STATUS: NUR nach bestandenem 15.3B.1 UND nach 15.3B.3 (fp32-Staging-
      Beweis) -- siehe Reihenfolge-Entscheidung oben. Bewusst separater
      Schritt, NICHT Teil des Kern-Umbaus (verschiedene Fragen: 15.3B.1 =
      schneller lesen, 15.3B.2 = weniger Bytes UND verlustbehaftet, braucht
      eigene Verifikation). Code groesstenteils schon vorhanden (--dtype /
      packed_dtype: float16 in pack_latents.py implementiert, _load_latent
      castet immer auf float32 zurueck) -> 15.3B.2 ist primaer ein
      VALIDIERUNGS-Schritt, kein Umbau.
    VORGEHEN:
      a) pack_latents.py um --dtype Parameter erweitern (float32 Default,
         float16 fuer diesen Test) -- gleiches Skript, kein zweites
         System
      b) _load_latent(): IMMER .astype(np.float32) direkt nach dem Lesen,
         unabhaengig vom Storage-Dtype -- bei float32-Storage ein No-Op,
         bei float16-Storage der eigentliche Upcast. Verhindert, dass
         Modell/Loss/Decoder (alle float32-Annahme) einen stillen
         Dtype-Bug sehen
      c) Wertebereich vorab bekannt unproblematisch: Latents max ~15.6,
         min 0.0 (ReLU-aktiviert) -- komfortabel im float16-Bereich, kein
         Ueberlauf-Risiko
    VALIDIERUNG (zweistufig, analog 15.3B.1):
      - Rundtrip-Fehler auf Stichprobe: Original vs. float16->float32,
        max. absoluter Fehler dokumentieren
      - Seed-42-Anker MIT float16-Storage laufen lassen, mIoU/std-Ratio
        gegen 15.2-Rauschboden pruefen
      - ZUSAETZLICHER ANKER (nur hier, nicht in 15.3B.1): bestes
        lambda_std aus 15.3 einmal unter float16-Storage laufen lassen
        und gegen den 15.3-Altstand pruefen.
        BEGRUENDUNG: 15.4 baut auf diesem Uebergabepunkt auf -- wird er
        nicht validiert, entsteht ein methodischer Bruch zwischen 15.3
        (alte Pipeline) und 15.4 (neue Pipeline).
        WARUM NUR FUER float16: beim fp32-Packing (15.3B.1) beweist der
        innere Check exakte Bit-Gleichheit -> der Datenpfad ist
        identisch, ein zweiter Anker waere redundant. float16 ist
        dagegen verlustbehaftet -> der Uebergabepunkt braucht eine
        eigene Bestaetigung
    ENTSCHEIDUNG: Delta innerhalb Rauschboden -> float16 wird neuer
      Standard (halbiert 440GB->~220GB, weiter reduzierter I/O). Delta
      ausserhalb -> bei float32 bleiben, Befund dokumentieren
    DELIVERABLE: Rundtrip-Fehleranalyse, Anker-Vergleich float16 vs.
      float32, Entscheidung + Begruendung
    AUFWAND: ~0.5-1 Tag (Skript-Erweiterung trivial, Hauptkosten =
      erneuter Preprocessing- + Anker-Lauf)

  GESAMT-DELIVERABLE (15.3B): vereinheitlichte Daten-Pipeline (ein
    Codepfad, packed_dir/use_packed-Schalter, lokal UND Cluster),
    node-lokales /dev/shm-Staging als der eigentliche Geschwindigkeits-
    Hebel, gemessener (nicht geschaetzter) data%-Vergleich ueber alle
    Stufen, Entscheidung zu float16-Storage -- alles VOR Fortsetzung von
    15.4. Das I/O-Problem gilt als geloest, sobald der data%-Anteil so weit
    faellt, dass die GPU wieder der Engpass ist (Gate unten).
  AUFWAND (gesamt 15.3B): grob 3-4 Tage (exkl. Preprocessing-Wartezeit).
    15.3B.1 groesstenteils erledigt; verbleibend v.a. 15.3B.3 (Staging) +
    15.3B.2 (float16-Validierung), beide leichtgewichtig.
  REIHENFOLGE (final): 15.3B.1 -> 15.3B.3 (fp32-Staging) -> 15.3B.2
    (float16) -> optional 15.3B.3b (Staging+float16 kombiniert).

---------------------------------------------------------------------------
===========================================================================
Task 16a — LOKALER SWEEP-APPARAT (FALLBACK-PFAD)
Status: FALLBACK — Hauptausfuehrung der Sweeps: Task 16b (naechster Abschnitt)
===========================================================================
  ABGRENZUNG/HISTORIE: Die Loss-Regler-Sweeps waren urspruenglich als
    15.4-15.8 geplant, wurden als 16a.2-16a.8 herausgeloest und sind nach
    dem STRATEGIE-ENTSCHEID (unten) als 16b.4-16b.10 in den Cluster-
    Hauptpfad gewandert. In 16a verbleiben: die abgeschlossene Historie
    (16a.0 Automatisierung, 16a.1 Diagnose, 16a.1B Beschleunigung) und der
    einsatzbereite LOKALE FALLBACK-APPARAT.

  STRATEGIE-ENTSCHEID (2026-07-13, nach 16a.1B + 16b-Vorarbeiten;
  Jobs 122810/122811/122812, NOTIZ_16B_LOADER_DIAGNOSE.md):
    CLUSTER = HAUPTGLEIS fuer alle Sweeps. Begruendung:
    - Epochenzeit Cluster nach 1d+1b: ~1450s -> ~663s (2.1x); EINZELLAUF-
      GLEICHSTAND mit lokal (~685s) -- der fruehere 2x-Malus ist eliminiert.
    - PARALLELITAET bewiesen: N Trainings teilen EINE /dev/shm-Kopie
      (memmap read-only, shm_cleanup: job_shared), Konkurrenz <=5%,
      Durchsatz 2.85x bei 3 GPUs. 1 Sweep = 1 Mehr-GPU-Job.
    - Rest-Engpass (data% ~58) = collate/IPC-Grundkosten -- diagnostiziert,
      pin_memory=false FALSIFIZIERT (+93s/Epoche), bewusst akzeptiert.
    LOKAL bleibt Null-Kosten-Fallback (Ein-Code-Pfad; identische Dateien,
    Config steuert Umgebung).
    RUECKFALLKRITERIUM: Queue liefert >2-3 Tage keine passenden GPUs ODER
    cluster-seitiger Blocker -> betroffener Sweep laeuft LOKAL (Abschnitt
    "FALLBACK-BETRIEB" unten) mit IDENTISCHEN Werten/Schwellen/Kriterien.

  -------------------------------------------------------------------------
  16a.0 — AUTOMATISIERUNGS-SKRIPT (Weg 1: feste, begruendete Wertelisten)
  -------------------------------------------------------------------------
    BEWUSSTE SCOPE-GRENZE: Weg 1, NICHT adaptiv. Das Skript arbeitet eine
      VORGEGEBENE Liste von Lambda-Werten seriell ab -- es entscheidet NICHT
      selbst, wo als Naechstes gemessen wird (das waere Weg 2 / Optuna, siehe
      16b.10, bewusst optional/nachgelagert). Der Mensch gibt die Werte vor,
      liest das "Knie" nach dem groben Scan selbst ab und misst bei Bedarf
      manuell feiner nach. Begruendung: robust (~1 Tag Bauzeit), wissen-
      schaftlich sauber (feste, dokumentierte Werte statt black-box-Suche),
      volle Kontrolle beim Menschen an genau der Stelle, wo Urteil > Automatik.
    BASIS: erweitert die vorhandenen mk_lambda_cfg.py / mk_seed_cfg.py.
    FUNKTION: (1) je Lambda-Wert eine Config generieren (Line-Level-Regex-
      Patch mit Exact-1-Match-Guard, NICHT YAML-load/dump -- erhaelt
      Kommentare, bewaehrtes Muster aus 15.x). (2) Training seriell starten
      (tmux-safe). (3) Ergebnisse (mIoU-Verlauf, bestes mIoU, std-Ratio,
      MSE, Pearson, CosSim) je Lauf einsammeln und in EINE Tabelle/CSV
      schreiben. (4) Idempotent: bereits gelaufene Werte ueberspringen.
    NUTZT: mIoU-Plateau-Stopping (aus 15.3B.2) -> jeder Sweep-Lauf killt
      sich selbst am Plateau, statt stur 50 Epochen zu fahren. Das ist der
      Zeit-Hebel, der die Sweeps ueberhaupt seriell-lokal machbar macht.
    DELIVERABLE: sweep_runner.py (o.ae.), Ergebnis-CSV je Regler.
    AUFWAND: ~1 Tag Skripting (einmalig, dann fuer alle Regler wiederver-
      wendbar).

  -------------------------------------------------------------------------
  16a.1 — VORAB-DIAGNOSE (nur wo sie ZEIT SPART -- gezielt, nicht flaechig)
  -------------------------------------------------------------------------
    PRINZIP: Bevor ein teurer Sweep (~15-25h/Regler) gefahren wird, mit
      einer BILLIGEN Inferenz-Messung (~15 Min, nur Vorwaertslauf auf dem
      Val-Subset mit einem VORHANDENEN Checkpoint, KEIN Training) pruefen,
      ob der Regler ueberhaupt etwas zu tun hat. Dasselbe Diagnose-Prinzip,
      mit dem 15.2 den std-Gap (~16.5%) gemessen hat (vgl. RAUSCH_REPORT.md /
      persistence_baseline.py).
    ZEITVORTEIL (asymmetrisch, daher lohnend): Zeigt die Diagnose einen
      KLEINEN Gap -> ganzer Sweep vermeidbar bzw. auf 2-3 Bestaetigungs-
      Punkte reduzierbar (Ersparnis potenziell 10-20h). Zeigt sie einen
      GROSSEN Gap -> 15 Min "verloren", Sweep laeuft wie geplant (zeitlich
      neutral, wissenschaftlich sogar besser, weil Sweep-Motivation belegt).
      Man kann also nur gewinnen oder neutral rauskommen.

    ANWENDBARKEIT je Regler (ehrlich -- Diagnose NUR wo aussagekraeftig):
      * lambda_mean  -> JA. mean-Gap ist direkt messbar (Mittelwert der
          Vorhersagen vs. Ziel-Mittelwert), voellig analog zum std-Gap.
          Kleiner Gap (<~3-5%) sehr plausibel (MSE zieht den Mittelwert gut
          mit) -> dann Sweep skippen/minimieren. GROESSTER erwarteter
          Zeitsparer. DIESE Diagnose zuerst.
      * lambda_grad / lambda_ssim -> NEIN (bzw. nur schwach). "Blur/Schaerfe"
          ist zwar messbar (mittlere Gradienten-Magnitude pred vs. real),
          aber die Uebersetzung "Schaerfe-Wert -> mIoU-Effekt" ist NICHT
          direkt (Regler kann helfen ODER Rauschen verstaerken). -> direkt
          scannen, keine Vorab-Diagnose.
      * lambda_cos -> NEIN. Schon aktiv (0.1), geringer Effekt erwartet,
          Feinjustierung nur empirisch bestimmbar.

    ZWEITER NUTZEN (Reihenfolge-Optimierung fuer Deadline-Risiko): Die
      Diagnose + Hypothesen (jetzt 16b.4) legen die REIHENFOLGE der Sweeps fest --
      vielversprechende Regler (lambda_grad) ZUERST, wahrscheinlich flache
      (lambda_cos, evtl. lambda_mean) ZULETZT. Falls die Zeit ausgeht, sind
      die wichtigen Ergebnisse schon da, die unwichtigen guten Gewissens
      kuerzbar ("im Rauschen, grob bestaetigt").
    DELIVERABLE: kurze Diagnose-Notiz (mean-Gap-Wert + Entscheidung
      voll/mini/skip fuer lambda_mean).
    AUFWAND: ~15-30 Min.

  -------------------------------------------------------------------------
  16a.1B — TRAININGS-BESCHLEUNIGUNG LOKAL (EINSCHUB vor dem ersten Sweep,
            analog zur 15.3B-Konvention fuer nachtraegliche Einschuebe)
  -------------------------------------------------------------------------
    ANLASS/TIMING: Recherche-Task (12.07.2026, Bericht "Training Acceleration
      for a Small BEV Transformer World Model", im Projekt als PDF) hat
      konkrete Compute-seitige Hebel identifiziert. WARUM JETZT und nicht
      spaeter: vor uns liegen noch grob 13-19 Sweep-Laeufe a 3-5h = 40-95h
      GPU-Zeit. Konservativ 1.5x Speedup spart 15-30h -- gegen ~2 Tage Umbau
      + EINEN neuen Anker-Lauf (~4h). Nach mehreren gefahrenen Sweeps kippt
      die Rechnung (jeder Dynamik-Wechsel entwertet gelaufene Laeufe).
      -> Fenster ist JETZT, zwischen 16a.1 und 16a.2.
    HARDWARE-REALITAET (TITAN RTX = Turing sm_75, aus dem Bericht):
      KEIN FlashAttention (braucht Ampere), KEIN TF32, KEINE bf16-Tensor-
      Cores. fp16+GradScaler (aktiv) ist bereits die richtige Wahl. Viele
      Standard-Empfehlungen (compile-Benchmarks, Flash) sind A100-Zahlen
      und NICHT uebertragbar -- Erwartungen entsprechend daempfen.

    STUFE 0 — PROFILING ZUERST (Pflicht, ~0.5 Tag):
      torch.profiler mit record_function-Bloecken um h2d/forward/ssim_loss/
      backward/optim; CUDA-Events fuer Wanduhr. OHNE diese Attribution ist
      jede Priorisierung geraten -- insbesondere ob SSIM auf [256,128,128]
      oder der ConvTranspose-Head dominiert. Entscheidet, ob Stufe-2-SSIM-
      Massnahme ueberhaupt lohnt.
    STUFE 1 — NUMERISCH NEUTRAL (~1 Tag, KEIN neuer Anker noetig):
      (a) channels_last fuer Modell+Input (Conv-lastiger Upsampling-Head,
          8-35% auf Conv-Anteilen laut PyTorch-Doku)
      (b) AdamW(fused=True), zero_grad(set_to_none=True),
          cudnn.benchmark=True (Shapes konstant), keine .item()/.cpu()-
          Syncs im heissen Loop
      (c) naive Attention -> F.scaled_dot_product_attention (auf Turing nur
          Memory-Efficient-Backend, moderater Gewinn; auf A100 nimmt
          DERSELBE Code automatisch Flash -> Cluster-Bonus gratis)
      (d) fp16 ueber den PCIe-Bus, Upcast+AvgPool erst AUF der GPU
          (halbiert H2D-Volumen; zentral fuer den A100-Fall)
    STUFE 2 — DYNAMIK-AENDERND (~1-2 Tage, NEUER ANKER PFLICHT):
      (e) batch_size 8->32 mit WURZEL-LR-Skalierung (x2, Adam-Regel nach
          Malladi et al. 2205.10287 -- NICHT linear, das gilt fuer SGD) +
          kurzer Warmup. Groesster Quick-Hebel bei unterfuellter 24GB-Karte.
      (f) NUR falls Profil SSIM >20% zeigt: SSIM auf 32x32 statt 128x128
          rechnen (downgesampelt) oder S3IM. Sonst lassen.
      -> danach EIN neuer lambda_grad=0-Anker mit schneller Config; alle
         Folge-Sweeps konsistent damit. Der bereits gelaufene 16a.0-Anchor
         (lambda_grad=0, alte Config) dient als Plausibilitaets-Referenz.
    STUFE 3 — OPTIONAL/RISKANT (~0.5 Tag Versuch, Abbruch erlaubt):
      (g) torch.compile(model) Default-Modus (funktioniert unter Py3.8/
          torch2.1.2; TORCH_LOGS="graph_breaks,recompiles" pruefen). Auf
          Turing kleiner Gewinn erwartet; bei Problemen fallenlassen.
    BEWUSST NICHT (mit Grund):
      - Activation Checkpointing: tauscht Compute gegen Speicher -- wir sind
        NICHT speicherlimitiert, wuerde nur verlangsamen.
      - NVIDIA DALI: glaenzt bei JPEG-Decoding/Augmentation; unsere Daten
        sind fertige fp16-Memmaps, kein Decoding -> eigener GPU-Upcast
        (Stufe 1d) ist zielgerichteter.
      - 16x16-Grid (768 statt 3072 Tokens): staerkster Architektur-Hebel
        laut Literatur (LeWM bis ~200x weniger Tokens, DINO-WM 196 Patches),
        ABER die 32x32-Cell-Attention IST der Untersuchungsgegenstand der
        Arbeit und Architektur ist fuer die Loss-Studie eingefroren
        (Confounder-Regel, siehe Task 15) -> VORGEMERKT als Task-18-
        Experiment, hier tabu.
    ERFOLGSKRITERIUM: Epochenzeit-Vergleich alt/neu auf identischem
      Anker-Setup; mIoU des neuen Ankers im 15.2-Rauschband (Stufe 1)
      bzw. dokumentierter neuer Referenzpunkt (Stufe 2).
    DELIVERABLE: Profiling-Notiz (wohin geht die Epochenzeit), umgebaute
      train_linux.py/Config, neuer Anker-Lauf, Speedup-Zahl vorher/nachher.
    AUFWAND: ~2-3 Tage gesamt (inkl. Anker); Stufen einzeln abbrechbar.


  -------------------------------------------------------------------------
  16a-FALLBACK-BETRIEB (nur bei Cluster-Ausfall; sonst ruht dieser Pfad)
  -------------------------------------------------------------------------
    APPARAT (gewartet, einsatzbereit): sweep_runner.py (16a.0; train ->
      inference -> optional Full-Val -> CSV, seriell in tmux, idempotent)
      + config_sweep_base_fp16.yaml (6-Term, lambda_std=1.0) auf
      alre-server-u20 (TITAN RTX; ~3-5h/Wert nach 16a.1B).
    AUSFUEHRUNG: identische Wertelisten/Hypothesen/Schwellen wie 16b.4-16b.7
      (dort spezifiziert) -- nur Mechanik seriell-lokal statt parallel.
      Ergebnisse sind ueber die Nullpunkt-Bruecke (16b.4) an Cluster-Laeufe
      angeschlossen und austauschbar; Berichte weiterhin als TASK16B_X_...,
      mit Vermerk "Ausfuehrung lokal (Fallback)".


===========================================================================
Task 16b — CLUSTER-SWEEPS (HAUPTPFAD)
Status: HAUPTGLEIS (Strategie-Entscheid 2026-07-13; Vorarbeiten ~50% fertig)
Chat: je Teiltask eigenstaendig | Berichte: TASK16B[_X]_ABSCHLUSSBERICHT.md
===========================================================================
  ZIEL: alle Loss-Regler-Sweeps (vormals 16a.2-16a.8) auf dem DGX-Cluster
    ausfuehren -- 1 Sweep = 1 Mehr-GPU-Job (Werteanzahl = GPU-Anzahl),
    EIN geteiltes /dev/shm-Staging fuer alle Prozesse des Jobs.

  STAND VORARBEITEN (2026-07-13, Details NOTIZ_16B_LOADER_DIAGNOSE.md +
  TASK16A_1B_ABSCHLUSSBERICHT.md §7/§12):
    - 1d/1b deployed; Epochenzeit verifiziert ~663s (Job 122810, W16).
    - Loader-Diagnose (122811): pin_memory=false falsifiziert (+93s/Epoche
      -- Lehrstueck Metrik-Verschiebung: data% "besser", Zeit schlechter);
      num_workers 16 = -31s, mitgenommen. Rest-Engpass collate/IPC,
      bewusst akzeptiert (invasive Umbauten NICHT empfohlen).
    - Parallel-Test (122812): 3 Trainings, EIN Staging (job_shared),
      6x Skip-Copy verifiziert, Konkurrenz <=5%, Durchsatz 2.85x.
    - Code-Stand: pin_memory-Flag (bev_dataloader/train_linux),
      job_shared-Strategie (shm_staging), sbatch-Templates
      (sbatch_parallel_test.sh, sbatch_timing_check_3ep.sh).
    - RISIKO bleibt die QUEUE (fremde Langlaeufer-Jobs); Gegenmittel:
      fp16/400-600G-Fussabdruck + Fallback 16a.

  -------------------------------------------------------------------------
  16b.1 — SWEEP-BASIS-CONFIG CLUSTER  (BLOCKER fuer alles Weitere)
  -------------------------------------------------------------------------
    config_sweep_base_cluster_fp16.yaml: SECHS-Term-Loss-Block
      (lambda_std: 1.0 fix = 15.3-Arbeitspunkt, Rest Baseline; analog
      lokaler config_sweep_base_fp16.yaml), BeeGFS-Pfade, use_packed +
      stage_to_shm, shm_cleanup: job_shared, num_workers: 16,
      checkpoint-dir-Schema je Lambda-Wert.
    ACHTUNG: ALLE bisherigen Cluster-Configs tragen das 3-Term-Schema ->
      lambda_grad dort NICHT setzbar. Lambda-Configs entstehen per
      patch_line (mk_lambda_cfg-Muster, diff-Verifikation vor Compute).
    AUFWAND: ~0.5 Tag.

  -------------------------------------------------------------------------
  16b.2 — PARALLELER SWEEP-JOB (sbatch)
  -------------------------------------------------------------------------
    sbatch_sweep_parallel.sh aus sbatch_parallel_test.sh (Template):
      Pre-Staging sequenziell -> N Trainings (CUDA_VISIBLE_DEVICES=0..N-1,
      je eigenes Log) -> wait -> Inference-Schritt (16b.3) -> trap-Cleanup.
    RESSOURCEN-STARTPUNKT (5 Werte): --gres=gpu:a100:5, --cpus-per-task=96,
      --mem=600G, --time=06:00:00 (Plateau-Stopping variabel: ~15 Ep. x
      ~695s + Inference + Puffer).
    TF32-ENTSCHEIDUNG (Ampere-Feature, numerik-relevant!): falls aktiviert
      (torch.set_float32_matmul_precision("high")), dann VOR dem 16b.4-
      Nullpunkt und dokumentiert -- das Bruecken-Kriterium prueft die
      Vertraeglichkeit mit. Default: AUS lassen (Konservativ-Pfad).
    ALTERNATIVE bei fragmentierter Queue: SLURM-Array (1 Task = 1 Wert,
      je eigene Job-ID) -- Kosten: JEDER Task staged eigenstaendig
      (~7 min + 286G je Knoten); nur waehlen, wenn kein Knoten N GPUs
      am Stueck hergibt.
    AUFWAND: ~0.5 Tag.

  -------------------------------------------------------------------------
  16b.3 — AUSWERTE-KETTE
  -------------------------------------------------------------------------
    (a) --no-save-Patch fuer inference.py (16a.1-Uebergabe): keine
        pred_/real_-npy-Dumps (~10 GB/Wert), nur inference_log.json.
    (b) inference.py DIREKT IM JOB je Wert (300 Val-Samples, rohe .npy von
        BeeGFS, wenige Minuten) -> std-Ratio nach 15.3-Methodik.
    (c) CSV-Harvest (sweep_runner-Logik: run_summary.json +
        inference_log.json) lokal nach rsync ODER als Jobende-Schritt.
    (d) Full-Val des jeweiligen Siegers via eval_full_val.py auf dem
        Cluster (dataloader-basiert, packed + staging).
    DELIVERABLE 16b.1-3: TASK16B_ABSCHLUSSBERICHT.md (Operationalisierung).
    AUFWAND: ~0.5-1 Tag.

  16b-OPT — GPU-TEILUNG fuer noch mehr Durchsatz (OPTIONAL, uebernommen):
    MIG (bis 7 Instanzen/A100, braucht Admin/Thorsten+Anton; ~3-12%
    Kontention lt. Benchmarks; 1 Prozess = 1 Instanz) oder MPS (ohne Admin,
    keine Isolation). Erst relevant, wenn GPU-Anzahl der Engpass wird --
    aktuell reicht 1 Job = N ganze GPUs.

  -------------------------------------------------------------------------
  16b.4 — SWEEP lambda_grad  (ERSTER Regler; vormals 16a.2)
  -------------------------------------------------------------------------
    HYPOTHESE: MSE-Loss erzeugt glatte/verwaschene Vorhersagen (MSE->Blur).
      Ein Gradient-Term bestraft Abweichungen der raeumlichen Ableitungen ->
      bringt Kanten-/Strukturschaerfe zurueck (zielt auf RAUSCH_REPORT
      Ursache 2+3, raeumliche Hochfrequenz). Bewegt mIoU ueber den std-Term
      hinaus?
    GEGENHYPOTHESE (macht das Muster interessant): zu grosses lambda_grad
      verstaerkt Hochfrequenz-RAUSCHEN -> mIoU faellt wieder. Erwartet also
      ein OPTIMUM/Knie bei kleinen Werten mit Abfall danach -- anders als
      lambda_std (das monoton saettigte).
    WERTELISTE (begruendet, klein wegen erwartetem Optimum bei kleinen
      Werten): 0 . 0.05 . 0.1 . 0.3 . 1.0. Basis: bestes lambda_std aus 15.3
      FIX. Knie danach ggf. manuell verfeinern.
    DELIVERABLE: mIoU(lambda_grad), std-Ratio(lambda_grad), Kurvenform.
    NULLPUNKT-ROLLEN (lambda_grad=0, ERSTER Cluster-Sweep-Lauf ueberhaupt):
      Sweep-Anker + Stufe-1-Verifikation (16a.1B §8) + HARDWARE-BRUECKE
      lokal->Cluster. Kriterium: mIoU ~0.68 +/- 0.014, std-Ratio ~0.915
      +/- 0.009 (15.2-Schwellen). AUSSERHALB DES BANDES: STOPP, Ursache
      klaeren, ggf. Fallback 16a -- KEINE weiteren Werte interpretieren.
    AUFWAND: EIN 5-GPU-Job (~3.5h Wanduhr); Fallback lokal ~15-25h seriell.

  -------------------------------------------------------------------------
  16b.5 — SWEEP lambda_ssim  (STRUKTUR-Regler, Redundanz zu grad; vormals 16a.3)
  -------------------------------------------------------------------------
    HYPOTHESE: SSIM misst lokale strukturelle Aehnlichkeit (Luminanz/
      Kontrast/Struktur) -> hilft bei lokaler Strukturtreue. Schon aktiv
      (0.1). VORSICHT REDUNDANZ: ueberlappt konzeptionell mit lambda_grad
      (beide zielen auf Struktur/Schaerfe) -> kombinierter Effekt evtl. NICHT
      additiv. Beim Auswerten explizit gegen den grad-Effekt halten.
    WERTELISTE: 0 . 0.1 . 0.3. Basis: beste Konfig aus 15.3 + 16b.4.
    DELIVERABLE: mIoU/std-Ratio(lambda_ssim), Redundanz-Einschaetzung vs. grad.
    AUFWAND: EIN 3-GPU-Job (~3.5h); Fallback lokal ~9-15h seriell.

  -------------------------------------------------------------------------
  16b.6 — SWEEP lambda_mean + lambda_cos  (VERTEILUNGS-/RICHTUNGS-Regler; vormals 16a.4)
  -------------------------------------------------------------------------
    Beide voraussichtlich FLACH (geringer Effekt) -> ZULETZT, ggf. verkuerzt.
    lambda_mean: NUR vollen Sweep, wenn 16a.1-Diagnose grossen mean-Gap
      zeigt. Sonst 2-3 Bestaetigungs-Punkte (0 . 0.1 . 0.3) oder dokumentierter
      Skip. HYPOTHESE: Verteilungs-Term analog lambda_std, aber fuer den
      Mittelwert -- MSE zieht den Mittelwert vermutlich schon gut mit, daher
      wenig zu holen.
    lambda_cos: schon aktiv (0.1). MSE bestraft Betrag, Cosinus die RICHTUNG
      der Feature-Vektoren. Erwartung flach/robust (cos + MSE ziehen meist
      gleich). WERTELISTE grob: 0 . 0.1 . 0.5.
    DELIVERABLE: mIoU/std-Ratio je Regler; Notwendigkeits-Nachweis (dass
      cos/mean ueberhaupt gebraucht werden -- der Lauf mit lambda=0 belegt es).
    AUFWAND: 1-2 Jobs (je Regler ~3 GPUs, ~3.5h); durch Diagnose/Verkuerzung
      potenziell deutlich weniger. Fallback lokal ~3-5h pro Wert.

  -------------------------------------------------------------------------
  16b.7 — L1 vs. MSE + NOTWENDIGKEITS-ABLATIONEN  (kategorial; vormals 16a.5)
  -------------------------------------------------------------------------
    VORGEHEN: ein Lauf MSE, ein Lauf L1 (sonst beste Konfig aus 16b.4-16b.6).
      L1 ist weniger mittelwert-suchend -> Entweder-Oder, KEIN Regler.
    DELIVERABLE: L1-vs-MSE-Vergleich (mIoU/std-Ratio).
    AUFWAND: EIN 2-GPU-Job (~3.5h); Fallback lokal ~6-10h.

  -------------------------------------------------------------------------
  16b.8 — (OPTIONAL) SLICED-WASSERSTEIN-REGLER  (vormals 16a.6)
  -------------------------------------------------------------------------
    STATUS: NUR falls 16b.4-16b.6 die Frage offenlassen (std-Ratio bleibt
      niedrig ODER mIoU-Effekt unklar). Bewusst ans Ende, nicht als Einstieg.
    IDEE: LeWMs Projektions-MECHANISMUS (M zufaellige 1D-Richtungen,
      Cramer-Wold), aber Marginale auf die ECHTE Latent-Verteilung matchen
      statt N(0,I) -> sliced-Wasserstein(pred, real) als Verteilungs-Term.
      Matcht die GANZE Verteilung (Form/Schweife), nicht nur mean+std.
      WICHTIG: NICHT SIGReg (zieht auf N(0,I), zerstoert bei eingefrorenem
      Decoder die Decode-Treue/mIoU).
    CAVEAT: gradienten-verrauscht, NEBEN MSE (nicht statt). Deterministischer
      Predictor trifft nur die Batch-Verteilung, nicht Per-Sample-Schaerfe ->
      staerker als reines std-Matching, aber KEIN Ersatz fuer Task 18 (generativ).
    AUFWAND: ~1-1.5 Tage (Implementierung + Lauf).

  -------------------------------------------------------------------------
  16b.9 — SYNTHESE  (zusammenfassender Sweep-Abschluss; vormals 16a.7)
  -------------------------------------------------------------------------
    VORGEHEN: alle Laeufe (lokal + was 16b liefert) in EINER Tabelle (mIoU,
      std-Ratio, MSE, Pearson, CosSim -- je mit Fehlerbalken aus 15.2).
      Dosis-Wirkungs-Plots (jede Metrik gegen das gedrehte Gewicht, analog
      LeWM Fig. 16) + Trade-off-Kurve std-Ratio vs. mIoU.
    ZUSAETZLICH (Framing-Ergebnis, Literatur-Abgleich 07/2026): explizite
      Ausweisung der MINIMALEN HINREICHENDEN LOSS-MENGE -- welche der sechs
      Terme sind laut den lambda=0-Laeufen (16b.4-16b.6) und 16b.7
      entbehrlich (|Delta mIoU| < Rauschboden ~0.014 aus 15.2)? Reduktiver
      Befund im Geiste von LeWM ("from six to one") und der DINO-Foresight-
      Loss-Ablation (SmoothL1 allein genuegt dort) -- verwandelt die Sweeps
      von reinem Tuning in einen berichtbaren Erkenntnis-Beitrag.
    ENTSCHEIDUNG: beste Loss-Konfig + Befund, ob die deterministische Decke
      erreicht ist (mIoU-Plateau bei persistierendem std-Gap -> Task 18/
      generativ). Achtung Wissenschaftlichkeit: Sweeps nutzen mIoU-Plateau-
      Early-Stopping (verkuerzt), der finale Headline-mIoU (voller Val-Split)
      laeuft OHNE Early-Stop bis Konvergenz -- diesen Unterschied im Bericht
      transparent machen.
    DELIVERABLE: beste Konfig, Plots, Empfehlung -> speist Task 18.
    AUFWAND: ~1 Tag.

  -------------------------------------------------------------------------
  16b.10 — (OPTIONAL, NACH 16b.4-16b.9) AUTOMATISIERTE LAMBDA-OPTIMIERUNG  (vormals 16a.8)
  -------------------------------------------------------------------------
    STATUS: der frueher als 15.8 gefuehrte Optuna-Ansatz (Weg 2, adaptiv).
      Bleibt bewusst OPTIONAL und NACHGELAGERT -- Weg 1 (16a.0, feste Listen)
      ist das Kern-Vorgehen. Optuna nur, falls nach den manuellen Sweeps eine
      feinere/mehrdimensionale Optimierung gewuenscht ist. Scope-Constraints
      (auf Vorschlag des Professors, mit expliziten Grenzen) siehe Detailblock
      weiter unten (unveraendert uebernommen).

  ABBRUCHKRITERIUM DES SWEEP-ZYKLUS:
    Stopp, wenn mIoU ueber die Sweeps plateaut (kein Delta > Rauschboden aus
    15.2) ODER nach Durchlauf der geplanten Sub-Sweeps. std-Ratio ist NICHT
    das Abbruchkriterium (nur Manipulations-Check).

  GESAMT-DELIVERABLE (16b):
    - config_sweep_base_cluster_fp16.yaml + lambda-Configs (patch_line)
    - sbatch_sweep_parallel.sh (1 Job = 1 Regler, N GPUs) + Auswerte-Kette
    - sweep_runner.py bleibt als Fallback-Werkzeug gewartet (16a)
    - Vorab-Diagnose-Notiz (mean-Gap)
    - je Regler ein Bericht/CSV mit Metrik-Vergleich gegen Baseline
    - Dosis-Wirkungs-Plots + Trade-off-Kurve (16b.9)
    - Empfehlung: beste Loss-Konfiguration -> Task 18
  AUFWAND (gesamt 16b): 16b.1-3 ~1.5-2 Tage Operationalisierung; danach je
    Regler ~0.5 Tage Wanduhr (Job) + Auswertung -> grob 1 Woche inkl.
    Synthese. Ohne 16b.8/16b.10. Fallback-Pfad 16a: 1-2 Wochen seriell.

---------------------------------------------------------------------------
16b.10 (DETAIL) — (OPTIONAL, NACH 16b.4-16b.9) AUTOMATISIERTE
        LAMBDA-OPTIMIERUNG (OPTUNA)  [frueher 15.8, dann 16a.8]
       Bericht: TASK16B_10_ABSCHLUSSBERICHT.md
---------------------------------------------------------------------------
  ANLASS: Vorschlag eines Professors (Rueckmeldung ueber Anton/Cluster-Team,
    04.07.2026) -- automatisiert die beste Lambda-Kombination suchen lassen,
    statt nur manuell zu sweepen.
  STATUS: OPTIONAL. Nur ansetzen, wenn 15.1-15.7 abgeschlossen sind UND nach
    dem Cluster-Umzug noch Zeitbudget uebrig ist. Kein Ersatz fuer die
    manuellen Sweeps, sondern eine moegliche Ergaenzung danach.

  MECHANISMUS: Optuna (Bayesian-artige Suche). objective(trial)-Funktion:
    (a) von Optuna vorgeschlagene Lambda-Werte uebernehmen,
    (b) daraus eine temporaere YAML bauen (PyYAML-Dump, Struktur existiert
        schon),
    (c) train_linux.py damit anstossen,
    (d) resultierende mIoU aus der Decode-Validierung (Task 14) auslesen
        und an Optuna zurueckgeben.
    Visualisierung ueber Optunas eingebaute Plots (Optimierungsverlauf,
    Parameter-Wichtigkeit, Parallel-Coordinate) plus Anbindung an das
    bestehende wandb (WeightsAndBiasesCallback) -- kein neues Dashboard
    noetig.

  BESCHRAENKUNGEN (mit Begruendung -- gelten verbindlich, falls 16b.10
  angegangen wird):

    1. NUR NACH 16b.4-16b.9, nie davor oder statt.
       Begruendung: die manuellen Sweeps liefern erst die Information,
       WELCHE Lambdas ueberhaupt Wirkung zeigen und WO ungefaehr die
       Sweet-Spots liegen. Ohne diese Vorarbeit gaebe es keine begruendeten
       Suchbereiche, nur Raten.

    2. SUCHRAUM AUF 2-3 EMPIRISCH WIRKSAME LAMBDAS BESCHRAENKEN (vermutlich
       lambda_std, lambda_grad), mit ENGEN Suchbereichen um die in 15.3/15.4
       gefundenen Sweet-Spots -- NICHT alle 6 Lambdas gemeinsam suchen.
       Begruendung: eine gemeinsame 6-Parameter-Suche waere strukturell
       exakt PLDMs O(n^6)-Grid-Problem, das LeWM explizit kritisiert und das
       der gesamte Ein-Regler-pro-Lauf-Aufbau von Task 15 bewusst vermeidet.
       Ein optimierter Punkt ueber alle 6 Dimensionen wuerde die
       interpretierbaren Dosis-Wirkungs-Kurven aus 15.3/15.4 (der eigentliche
       wissenschaftliche Beitrag) nicht ersetzen koennen -- er zeigt "was am
       besten funktioniert", nicht "warum" oder "wie empfindlich".

    3. PRUNING ZWINGEND (z.B. Optuna MedianPruner oder Hyperband).
       Begruendung: ohne Pruning ist der Aufwand nicht vertretbar -- naiv,
       jeder Trial ein voller Lauf bei lokaler Geschwindigkeit (~800s/Epoche,
       ~40 Epochen bis Konvergenz), waeren 20-30 Trials seriell 8-11 Tage
       durchgehende GPU-Zeit. Pruning bricht schwache Trials frueh ab und
       senkt die mittlere Trial-Kosten auf etwa 1/3 bis 1/2 eines vollen
       Laufs.

    4. FINALES ERGEBNIS GEGEN DEN RAUSCHBODEN AUS 15.2 VALIDIEREN.
       Begruendung: bei vielen Trials gegen denselben Val-Split besteht ein
       reales Risiko, dass die "beste" gefundene Kombination Zufallsrauschen
       des Val-Sets trifft statt echter Verbesserung. Nur eine Verbesserung
       klar oberhalb des 15.2-Rauschbodens zaehlt als echter Befund.

    5. ZEIT-/MACHBARKEITS-GATE: vor Terminierung die reale Epochenzeit auf
       der Ziel-Hardware messen, BEVOR ein Trial-Budget festgelegt wird.
       Begruendung (aktualisiert durch 15.3B-Befunde): die Laufzeit-Situation
       ist inzwischen geklaert -- lokal ~780s/Epoche (GPU-limitiert, data%
       nur ~16-21%), Cluster ~1370s/Epoche trotz fp16+Staging (I/O- bzw.
       Hardware-Overkill-limitiert, data% ~74%). Fuer eine Optuna-Budget-
       Hochrechnung also die ZUTREFFENDE Zahl nehmen: bei seriell-lokal
       ~780s, bei Cluster-Array die 1370s ABER parallelisiert ueber mehrere
       GPUs. Pruning (Punkt 3) bleibt zwingend.
       NACHTRAG (16a.1B/16b-Vorarbeiten, 07/2026): Cluster-Epochenzeit ist
       inzwischen ~663s (nicht 1370s), und N Trials teilen EINE /dev/shm-
       Kopie (job_shared) -> Szenario (b) ist deutlich realistischer
       geworden; Zahlen vor Budget-Festlegung dennoch frisch messen.

  AUFWANDSSCHAETZUNG (grobe Groessenordnung, mit obigem Vorbehalt zu 5.):
    Implementierung (objective-Funktion, YAML-Generierung, mIoU-Auslese):
      ~1-2 Tage.
    Rechenzeit -- zwei Szenarien, je nach Cluster-I/O-Ergebnis:
      (a) naiv, jeder Trial voll, lokale Geschwindigkeit: 8-11 Tage serielle
          GPU-Zeit -- NICHT vertretbar, Ausschlussgrund fuer diese Variante.
      (b) mit Pruning + Cluster-Parallelitaet (4x DGX, sofern das I/O-Problem
          dort geloest ist): geschaetzt 6-15 Stunden Wandzeit fuer 20-25
          Trials. Diese Schaetzung ist NICHT belastbar, bevor Punkt 5
          (Zeit-Gate) tatsaechlich gemessen wurde.

  DELIVERABLE (falls durchgefuehrt): Optuna-Study mit Optimierungs-Plots,
    gemeinsam optimierte beste Lambda-Kombination, Vergleich gegen die
    einzeln in 15.3/15.4 gefundenen besten Werte, Validierung der
    Verbesserung gegen den Rauschboden aus 15.2.

===========================================================================
Task 16c — ROLLOUT-EVALUATION (MULTI-STEP, AUTOREGRESSIV)
Status: MUSS — nach 16b.9 (beste Konfig steht), VOR Task 18
Chat: eigenstaendig | Abschluss: TASK16C_ABSCHLUSSBERICHT.md
===========================================================================

  EINORDNUNG (warum dieser Task — Ergebnis des Literatur-Abgleichs 07/2026):
    Die reine 1-Frame-Praediktion ist die groesste identifizierte
    Angriffsflaeche der Arbeit gegenueber dem Stand 2026: das Feld erwartet
    Multi-Step-/Drift-Analysen (Occupancy-Forecasting berichtet standard-
    maessig 1s/2s/3s; DINO-Foresight [NeurIPS 2025] evaluiert short/mid-term
    via autoregressivem Rollout eines SINGLE-STEP-Modells — exakt das
    hiesige Setup). Eine Rollout-Kurve ist der billigste Task mit dem
    hoechsten wissenschaftlichen Grenznutzen im Restplan.

  KERNIDEE — KEIN RETRAINING, KEINE ARCHITEKTURAENDERUNG:
    Das Modell bleibt exakt wie trainiert (3 Frames -> 1 Frame). Der
    Rollout passiert rein zur INFERENZZEIT: die eigene Praediktion wird
    ins Kontextfenster zurueckgeschoben:
      Schritt 1: [F1, F2, F3]                -> F4_pred
      Schritt 2: [F2, F3, F4_pred]           -> F5_pred
      Schritt 3: [F3, F4_pred, F5_pred]      -> F6_pred
      Schritt 4: [F4_pred, F5_pred, F6_pred] -> F7_pred
    Horizont: k=1..4 Schritte = ~0.5-2.0s (nuScenes-Keyframes @2Hz) —
    damit im Bereich der 2026 ueblichen Berichtshorizonte.

  VORGEHEN:
    (1) rollout_eval.py auf Basis von inference.py: Schleife ueber
        k = 1..4, Praediktionen puffern und als Input recyceln.
    (2) SZENEN-ALIGNMENT: nur Sequenzen mit >= 3+k konsekutiven Keyframes
        DERSELBEN nuScenes-Szene verwenden (Szenengrenzen nie
        ueberschreiten). Bei ~40 Keyframes/Szene unkritisch — fuer k=4
        braucht es 7 konsekutive Frames.
    (3) METRIKEN je Schritt k (via bestehender Decode-Validierung /
        eval_full_val.py-Pfad): mIoU(k) und std-Ratio(k).
    (4) PERSISTENCE-REFERENZ ueber denselben Horizont: letzten ECHTEN
        Frame k-mal kopieren -> mIoU_pers(k). Erst diese Referenzkurve
        macht den Drift interpretierbar (Kernfrage: ab welchem k faellt
        das Modell unter die triviale Baseline — falls ueberhaupt?).
    (5) std-Gap-KOMPOUNDIERUNG explizit ausweisen: der std-Ratio(k)-
        Verlauf zeigt, ob das Moment-Matching (lambda_std) den Drift
        verlangsamt — direkter Rueckbezug auf den Loss-Engineering-Kern
        der Arbeit.

  BASIS-KONFIG: beste Loss-Konfig aus 16b.9.
    KONDITIONAL-KLAUSEL (16b.10): Falls 16b.10/Optuna SPAETER eine Konfig
    mit Delta mIoU > Rauschboden (15.2, ~0.014) liefert, Rollout mit
    dieser Konfig WIEDERHOLEN — Skript idempotent, kein Retraining,
    ~0.5 Tage. 16b.10 blockiert 16c also NICHT (Muss haengt nicht an Kann).

  INTERPRETATION (beide Ausgaenge sind berichtbar — kein Ergebnisrisiko):
    - MILDER Drift (mIoU-Abfall bis k=4 < ~10% relativ): starkes
      Positiv-Ergebnis, Modell ist rollout-stabil.
    - STARKER Drift: ehrlicher Negativ-Befund, motiviert Task 18
      (generativ) — ZWEITES Entscheidungskriterium neben dem std-Gap.
    - ERWARTUNG (Exposure Bias): das Modell hat im Training nie eigene,
      leicht verrauschte Praediktionen als Input gesehen -> ein Abfall
      ueber k ist NORMAL und der MESSGEGENSTAND, kein Bug.

  DELIVERABLE:
    - rollout_eval.py (idempotent, wiederverwendbar fuer Task-18-Output)
    - mIoU(k)- und std-Ratio(k)-Kurven inkl. Persistence-Referenzkurve
    - Drift-Quantifizierung + go/no-go-Empfehlung an Task 18
    - TASK16C_ABSCHLUSSBERICHT.md

  AUFWAND: ~1-2 Tage (Skript + Inferenz-Laeufe; KEIN Training).

===========================================================================
TASK 17 — VISUALISIERUNG & THESIS-MATERIALIEN (BEGLEITEND)
Status: LAUFEND ab Task 13 — NICHT erst am Ende!
Chat: eigenständig (Setup) + begleitend | Abschluss: TASK17_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Sicherstellen, dass jeder Trainingslauf die für die Arbeit nötigen
  Artefakte automatisch speichert — damit am Ende KEIN Lauf wiederholt
  werden muss, nur um an Bilder zu kommen.

WICHTIG (Lehre aus Erfahrung):
  Visualisierung NICHT aufschieben. VOR dem Loss-Zyklus (Task 15) festlegen,
  welche Artefakte jeder Lauf automatisch erzeugt. Sonst fehlen später Daten.

ARTEFAKTE, DIE JEDER LAUF SPEICHERN SOLL:
  - Gate-Alpha-Heatmap (kopieren vs. vorhersagen, Task 9b)
  - pred vs. real Segmentierungsmasken (stärkstes Bild)
  - std-Werte pro Epoch (für std-Gap-Verlauf)
  - mIoU-Kurve über Epochen (aus Ebene-1-Validierung)
  - neck Feature-Map Vergleich pred vs. real

EINMALIGE THESIS-MATERIALIEN (am Ende):
  - Architektur-Diagramme (vorhanden)
  - std-Gap Vorher/Nachher (Baseline vs. beste Loss-Konfig) — Kernargument
  - Trade-off-Plot std-Ratio vs. mIoU über alle Loss-Iterationen
  - Vergleich mit Referenzen (LAW, BEVWorld, FIERY, OccWorld,
    DINO-Foresight, DINO-WM, LeWM — DINO-Foresight [NeurIPS 2025] als
    engster methodischer Zwilling legitimiert das frozen-Decoder-mIoU-
    Evaluationsprotokoll dieser Arbeit)

DELIVERABLE:
  - Logging-Erweiterung: alle obigen Artefakte automatisch pro Lauf
  - finale Abbildungs-Sammlung für die schriftliche Arbeit

AUFWAND: ~1 Tag Setup + laufend

===========================================================================
TASK 18 — (OPTIONAL) GENERATIVER SCHRITT (SEG)
Status: OPTIONAL — nur falls Loss-Engineering (Task 15/16a) den std-Gap nicht
        ausreichend schließt ODER die Rollout-Evaluation (Task 16c) starken
        Drift zeigt (zweites Entscheidungskriterium).
        ZEITBOX: max. 1-2 Wochen — läuft es nicht in der Box, wandert es in
        "Future Work" (die deterministische std-Matching-Lösung trägt die
        Arbeit auch allein).
Chat: eigenständig | Abschluss: TASK18_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Falls deterministisches Loss-Engineering an die Schärfe-Grenze stößt
  (Literatur BEVWorld/OccWorld: deterministische Regression hat ein
  fundamentales std-Limit), Architektur-/Generativ-Maßnahmen testen.

OPTIONEN (risikoarm -> ambitioniert):
  - Residual-Prediction: Modell sagt (t+1 - t) statt t+1 absolut
  - Output-Head-Kapazität erhöhen (Conv-Verfeinerung)
  - VAE-Kopf: mean + logvar, Sampling, KL-Term
  - Diffusion-Refinement auf dem deterministischen Output
  - NEU VORGEMERKT (aus Beschleunigungs-Recherche 12.07., waehrend 16a
    bewusst tabu -- Architektur eingefroren): 16x16-Token-Grid statt 32x32
    (768 statt 3072 Tokens, ~16x weniger Attention-FLOPs). Literatur-
    Evidenz stark: LeWM bis ~200x weniger Tokens bei erhaltener Task-
    Qualitaet, DINO-WM 196 Patches/Frame, OccWorld komprimierte VQVAE-
    Tokens. Test: mIoU(16x16) vs mIoU(32x32) -- falls im Rauschband,
    dauerhaft schnellere Architektur fuer alle Folge-Experimente.

DELIVERABLE:
  - getestete Variante(n) mit Metrik-Vergleich gegen beste Task-15-Konfig
  - qualitativer Schärfe-Vergleich der Masken

AUFWAND: hoch (3-5 Tage), höchster Impact, höheres Risiko

===========================================================================
TASK 19 — SEG-STRANG ABSCHLUSS & ZUSAMMENFASSUNG
Status: AUSSTEHEND (Abschluss des Seg-Strangs)
Chat: eigenständig | Abschluss: TASK19_ABSCHLUSSBERICHT.md
===========================================================================

ZIEL:
  Den Seg-Strang sauber abschließen: beste Konfiguration, alle Metriken,
  alle Materialien gebündelt. Ab hier ist der Seg-Teil der Arbeit komplett.

INHALT:
  - finale beste Loss-Konfiguration + Begründung
  - vollständige Metrik-Tabelle (Baseline -> beste Konfig)
  - alle Thesis-Abbildungen final
  - Diskussion: std-Gap-Reduktion erreicht? mIoU-Effekt? Limit?
  - ENTSCHEIDUNG: bleibt Zeit für den Det-Strang (Task 20)?

===========================================================================
TASK 20 — (STRETCH) DETECTION-STRANG
Status: STRETCH — NUR wenn Seg-Strang (Tasks 12-18) abgeschlossen + Zeit
Chat: mehrere eigenständige Chats (analog Seg) | Abschluss: je Teil-Bericht
===========================================================================

GRUNDIDEE:
  Der Det-Strang wiederholt die METHODIK des Seg-Strangs (gleiche
  Architektur-Idee, gleiches Zwei-Ebenen-Validierungs- und Loss-Engineering-
  Vorgehen), aber auf Det-Latents mit Det-spezifischen Anpassungen. KEIN
  gemeinsames Modell mit Seg — ein komplett eigener, paralleler Strang.

WAS GEGENÜBER SEG ANGEPASST WERDEN MUSS (konkrete Liste):
  1. AUFLÖSUNG: Det-Latents sind [256, 180, 180], nicht 128x128.
     -> grid_size in der config ändern: AvgPool(k=4) auf 180 ergibt 45x45
        (nicht 32x32). Damit ändern sich abgeleitet:
        - Tokenzahl pro Frame: 45*45 = 2025 (nicht 1024)
        - gesamt: 3 * 2025 = 6075 Tokens (nicht 3072)
        - Positional Encoding: nn.Embedding(45, ...) statt (32, ...)
        - Output-Head-Reshape: zurück auf 45x45 statt 32x32
     HINWEIS: grid_size ist bereits ein config-Parameter. Wenn der Seg-Code
     diese Werte konsequent aus grid_size ableitet (statt 32/128 hart zu
     codieren), ist dieser Punkt nur eine Zahl. SONST: betroffene Stellen in
     downsampling.py / embedding.py / output_head.py anpassen.
  2. EINGABE-LATENTS: Det-Latents aus bevfusion-det.pth extrahieren
     (eigener Task analog Task 12, ~40000 Keyframes, [256,180,180]).
  3. DECODER: TransFusionHead statt BEVSegHead. inference_det.py (analog
     inference_decoder.py) als importierbare Funktion bauen.
  4. METRIK: mAP/NDS statt mIoU. NEU zu implementieren via nuscenes-devkit
     (mAP über Klassen/Distanz-Schwellen; NDS kombiniert mAP mit
     Translation/Scale/Orientation/Velocity/Attribute-Fehlern).
     -> validate_det(pred_latents) -> mAP, NDS (für Ebene-1-Validierung)
  5. LOSS-GEWICHTE: eigener lambda-Satz. Det-Latents haben höhere std (0.69)
     -> andere optimale Balance als Seg, separat tunen.

TEIL-TASKS (analog Seg, je eigener Chat + Bericht):
  19a Det-Latent-Extraktion (analog Task 12)
  19b Architektur an 180x180 anpassen + Baseline-Training (analog 13)
  19c mAP/NDS implementieren + Decode-Validierung (analog 14, größerer
      Aufwand wegen devkit)
  19d Loss-Engineering Det (analog 15, eigene lambda)
  19e Visualisierung + Abschluss (analog 17/19)

AUFWAND: groß — eigenständiger zweiter Strang. Nur bei ausreichend Zeit.

===========================================================================
REIHENFOLGE & ABHÄNGIGKEITEN
===========================================================================

  Tasks 1–11 (abgeschlossen)
         |
         v
   Task 12  <- NÄCHSTER SCHRITT
  (alle Seg-Latents extrahieren, voller Datensatz)
         |
         v
   Task 13
  (neue Baseline auf vollem Datensatz -> Referenzpunkt einfrieren)
         |
         +-----------------------------+
         v                             v
   Task 14                       Task 17 (Setup)
  (Decode-Validierung Ebene 1)  (Visualisierungs-Artefakte festlegen)
         |                             |
         v                             | (läuft begleitend mit)
  +----> Task 15 (Ebene 2: Loss-Gleichung modifizieren, je eigener Chat)
  |       |
  |       v
  |    Training mit fester Loss (~1,5 h)
  |       |  intern: Ebene 1 (Task 14) -> bester Checkpoint nach mIoU
  |       v
  |    Decode + Metriken (mIoU, std-Ratio)
  |       |
  +-------+ Zyklus wiederholen (Sweeps 16b: Cluster PARALLEL, 1 Job = 1 Regler)
         |   (16a = lokaler Fallback via sweep_runner, falls Queue blockiert)
         v
   Task 16c (MUSS: Rollout-Evaluation, autoregressiv k=1..4, KEIN Training;
         |   Basis: beste Konfig aus 16b.9. 16b.10/Optuna ist optionales
         |   Seitengleis und blockiert 16c NICHT — liefert 16b.10 spaeter eine
         |   signifikant bessere Konfig: Rollout wiederholen, ~0.5 Tage)
         v
   Task 18 (optional: generativ — Entscheidung aus ZWEI Kriterien:
         |   std-Gap-Befund aus 16b.9 UND Drift-Befund aus 16c)
         |
         v
   Task 19 (Seg-Strang Abschluss + Entscheidung über Det)
         |
         v
   Task 20 (STRETCH: Det-Strang, nur bei Zeit — analog, mit Anpassungen)

PRIORISIERUNG:
  Muss (Seg-Kern):  Task 12 · 13 · 14 · 15 · 16b (Cluster-Sweeps) · 16c
  Soll:             Task 16a (lokaler Fallback-Apparat, gewartet) · Task 19
                    (sauberer Seg-Abschluss)
  Kann:             Task 18 (generativ) · Task 20 (kompletter Det-Strang)

KONFIGURIERBARKEIT (für späteren Det-Strang, im Seg-Code beachten):
  Beim Bauen/Erweitern des Seg-Codes die auflösungsabhängigen Werte
  konsequent aus grid_size ableiten (nicht 32/128 hart codieren). Decoder
  und Metrik als austauschbare Bausteine behandeln. Kostet im Seg-Strang
  fast nichts, macht Det später zu "neue config + neue Metrik" statt
  Code-Umbau. Falls eine Stelle zu verschachtelt ist: bewusst Seg-spezifisch
  lassen UND in Task 20 Punkt 1 vermerken (nicht vergessen!).

OFFENE PUNKTE:
  - Weg A vs. B für Decode-Validierung (Task 14, Schritt 1) — pragmatisch
    entscheiden, nicht an mmdet3d festbeißen
  - genaue Abbruchschwelle des Loss-Zyklus (Task 15) — bei Bedarf justieren

===========================================================================
NACHTRAG 2026-07-30: TASK 21 — SEG-GT-VERANKERUNG (Betreuer-Input Leon Pohl)
===========================================================================

MOTIVATION: Unsere Seg-Kernmetrik ist selbstreferenziell — mIoU vergleicht
decode(Vorhersage) gegen decode(reales Latent t+1), also gegen Pseudo-GT aus
dem eingefrorenen Decoder statt gegen die ECHTEN nuScenes-Kartenmasken.
Der Det-Strang (Task 20) misst dagegen gegen echte GT-Boxen. Task 21 zieht
den Seg-Strang auf dasselbe Niveau: mIoU gegen nuScenes-Map-GT via
Injection-Protokoll (tools/test.py --eval map, bevfusion-seg.pth) — gleiche
Messmaschine wie 20.1, nur Seg-Config statt Det-Config.

  21.0 ORAKEL-GATE (Kettenbeweis, lokal, ~1 h)
       Reale Seg-Val-Latents in den Docker injizieren -> muss BEVFusions
       eigenes Val-mIoU exakt reproduzieren (Analogon Gate A).
       Vorher pruefen: Injection-Hook shape-agnostisch (256,180,180 vs
       512,128,128)? Latent-Dateinamen-Konvention Seg == Det?
  21.1 DUMPS (Cluster, ~1 h GPU; IMMER nach BeeGFS, NIE /home)
       eval_dump_latents.py mit Seg-Config (generisch, latent_scale=1.0,
       Rueckskalierung entfaellt): SmoothL1-Minimal (16b.9-Checkpoint)
       + 6-Term-Baseline. Persistenz = Symlink-Konstruktion lokal (gratis).
       Luecken-Konvention wie 20.1 (kontextlose Frames real fuellen).
  21.2 INJECTION-EVALS (lokal Docker, 4 Laeufe x ~30-45 min, sequenziell)
       Orakel / Minimal / 6-Term / Persistenz -> mIoU vs nuScenes-GT.
  21.3 AUSWERTUNG + BERICHT (lokal)
       Korridor-Figur analog 20_det_korridor (Persistenz -> Modelle ->
       Orakel=Decoder-Decke), Vergleich der Rangfolge mit der Pseudo-GT-
       Metrik, per-Klasse-GT-Zerlegung (stuetzt/relativiert die
       aleatorische-Decken-These), TASK21_ABSCHLUSSBERICHT.md,
       Nachtraege in TASK19-Bericht + CLAUDE.md.

  ERWARTUNG: Absolutwerte deutlich unter 0.69 (Decke = Decoder-eigenes
  GT-mIoU); Kernfrage ist die RANGFOLGE (Minimal vs 6-Term vs Persistenz)
  unter echter GT — bleibt sie, ist die Pseudo-GT-Metrik validiert; kippt
  sie, ist das ein 20.3-analoger Befund (Metrik-Eigenschaft).
  SPEICHER: Dumps ~100G/Variante auf BeeGFS, lokal nur temporaer + nach
  Eval loeschen (Freigabe einholen).

===========================================================================
NACHTRAG 2026-07-31: TASK 22 — SELTEN-KLASSEN- & DYNAMIK-HEBEL
(Betreuer-Input CE/Weighted-CE/Focal/Balancing + Nutzer-Idee Dynamik)
===========================================================================

MOTIVATION: Task 21 zeigt Headroom mean 0.040 GT-mIoU (divider 0.060,
ped_crossing 0.046) — Selten-Klassen-Einbruch = ueberwiegend Decoder-Decke,
aber der WM-eigene Verlust sitzt bei duennen Strukturen. Task-Loss durch
den frozen Decoder ist literatur- (VFMF/DiT-WAM: keiner koppelt) und
empirisch (16f-H1) entkraeftet -> Hebel wirken auf LOSS-GEWICHTUNG und
SAMPLING, nicht auf Task-Kopplung. SOTA-Anker: TC-WM 2605.25620
(task-zentrierte Priorisierung frozen Features), CBGS (BEVFusion-Det).
Realistisches Ziel: +0.01-0.02 mean-GT-mIoU, Retention divider/ped_cross
-> ~95%; Wert = sauberes Ablations-Kapitel mit GT-Messkette als Richter.

  22.0 INFRA (lokal Docker + 1 Cluster-Pass, alles Default-neutral)
       a) Selten-Klassen-Gewichtskarten: nuScenes-Map-GT direkt im
          Latent-Grid rastern (128x128, +-51.2m/0.8m, 6 Klassen, uint8,
          val 6019 + train 28130; VALIDIERUNGS-GATE: eigene Rasterung vs
          pipeline gt_masks_bev auf N Frames). Sync als kompakte npz nach
          BeeGFS (~130M).
       b) Dynamik-Scores je Fenster: ||x_t - x_{t-1}||-basiert aus den
          fp16-Packs (1 Cluster-Pass); optional ego-warp-kompensiert
          (Code aus 16d-Vorstufe) -> Objekt- vs Ego-Dynamik getrennt.
          token->score als JSON/npy.
  22.1 SWEEP A — gewichtete Latent-Loss: SmoothL1-Zellgewicht
       1+(w-1)*rare_mask, w in {2,4,8}, multi-scale mitskaliert.
       Config rare_weight (Default 1.0 = bit-identisch).
  22.2 SWEEP B — Balanced Sampler nach Selten-Klassen-Pixelanteil
       (CBGS-Analogon, WeightedRandomSampler, Config sampler_mode=rare).
  22.3 SWEEP C — Dynamik-Balancing (Nutzer-Idee): Oversampling
       dynamikreicher Fenster (sampler_mode=dynamics, alpha-Regler);
       Alternative Loss-Gewichtung je Fenster dokumentieren.
  22.4 KONTROLLE (optional): Focal-Task-Loss kleines lambda (16f-H1-
       Nachtest unter Gewichtung).
  EVAL: 300-Proxy im Sweep; Sieger -> GT-Injection-Eval (Task-21-Kette),
       Zielmetrik per-Klasse-Retention + NEU dynamik-stratifizierte Eval
       (mIoU nach Dynamik-Quartilen, quantifiziert den Filmstrip-Befund).
  SPEICHER: alles Grosse nach BeeGFS; Gewichtskarten/Scores sind klein.

  NACHTRAG 01.08. (aus externer KI-Bewertung uebernommen, kuratiert):
  22.4 OPTIONAL (nur bei Signal aus 22.1-22.3): Occupancy-CHANGE-Masken —
       Zellgewichte aus GT-AENDERUNGS-Regionen (Maske_t != Maske_t+1),
       Bewegung auf ZELLebene statt Fensterebene; Infrastruktur (GT-Masken
       je Frame + cell_weight-Mechanik) vorhanden, ~0 Zusatzaufwand.
  EVAL-ERWEITERUNG (billig, eval-only, Thesis-Analyse): zusaetzlich
       DISTANZ-Ringe (ego-nah/mittel/fern) und fuer Det GROESSEN-Klassen
       stratifizieren. THESIS-ELEMENT: Drei-Komponenten-Zerlegung
       Decoder-Decke / Aleatorik / Forecasting-Fehler als eigene Figur
       (21_error_decomposition, Zahlen aus Task 19/20/21 vorhanden).

===========================================================================
NACHTRAG 2026-08-02: TASK 23 — DECODER-ADAPTATION (SEG, parallel zum
Schreiben; Motivation: Task-21/22-Befund "Decke sitzt in der Wahrnehmung")
===========================================================================

IDEE: Der frozen Seg-Kopf ist fuer REALE Latents optimiert; WM-Vorhersagen
sind leicht geglaettet. Post-Fuser-Stack (SECOND+FPN+SegHead) auf
(Train-Vorhersage -> echte GT) nachtrainieren, Messung mit UNVERAENDERTER
Injection-Kette (nur Checkpoint getauscht). NUR SEG; Det-Analogon ggf.
Future Work (dort 52% Forecasting-Anteil = mehr Rueckholpotential).

  23.0 Train-Dump minimal-Modell, OHNE Real-Fill (~26k Fenster, ~215G,
       BeeGFS dumps_task23/; eval_dump_latents --split train --no_fill).
       + GT-Cache im PIPELINE-Grid 200x200 (mk_gt_masks_seg --grid pipe).
  23.1 Standalone-Trainer im Docker (KEINE Bilder/LiDAR): BEVFusion laden,
       alles bis Fuser frozen, decoder.backbone+neck+heads.map trainieren,
       Original-Focal-Loss, batch 8, ~8-10 Epochen; Val-Proxy je Epoche.
       Ablation: head_only (Backbone/Neck frozen).
  23.2 Adaptierte Gewichte in Voll-Checkpoint einsetzen -> Injection-Evals:
       (a) adaptiert + Val-VORHERSAGEN (Headline vs 0.5898),
       (b) adaptiert + REALE Val-Latents (neue Orakel-Referenz vs 0.6295).
       Fokus per-Klasse divider/stop_line.
  23.3 Bericht TASK23 + Figur; Erwartung ehrlich: Teil des 0.04-Gaps,
       +0.01-0.02 mean realistisch; auch ein Neutral-Ergebnis haertet 21/22.

===========================================================================
NACHTRAG 2026-08-02: TASK 24 — DECODER-ADAPTATION DET (Analogon zu 23)
===========================================================================
MOTIVATION: Det-Fehler = 52% Forecasting (21_error_decomposition); Task 23
holte bei Seg ~29% des Gaps via Kopf-Adaptation zurueck -> Det-Potential
grob +0.05-0.09 mAP. NUR mit User-Freigaben; Speicher-Disziplin beachten.
  24.0 Dumps (Job 127214, BeeGFS dumps_task24/): Train-Vorhersagen
       det_baseline STRIDE 2 (~13.5k Fenster, ~215G — voll waere 420G und
       passt lokal nicht) + Val-Dump neu (lokale Kopie war geloescht).
  24.1 TransFusion-Head-Adaptation im det_extract-Docker: Injection-Hook
       um Trainings-Modus erweitern ODER forward_single-Trainingszweig
       direkt nutzen (GT-Boxen-Loss aus mmdet3d-Maschinerie, NICHT
       nachbauen); nur Head (+ ggf. decoder) trainierbar.
  24.2 Injection-Evals: adaptiert+Val-Vorhersagen (vs 0.3438/0.4761),
       adaptiert+real (vs Orakel 0.6858/0.7146).
  24.3 Bericht/Figur. Lokale Sequenz: erst Train-Dump (215G) fuer 24.1,
       nach Training loeschen, dann Val-Dump (100G) + reale Val-Latents
       (94G von BeeGFS latents_det/val) fuer 24.2.
===========================================================================
NACHTRAG 2026-08-03: TASK 25 — VERFEINERTE DET-ADAPTATION + PROJEKTSTATUS
===========================================================================
TASK 25 (FERTIG, TASK25_ABSCHLUSSBERICHT.md): v2-Kopf = 12 Ep. Cosine +
20% Real-Mix + Val-Loss-Selektion (Holdout 500). Ergebnis: pred-Gewinn
gehalten (0.4292 mAP), Real-Malus fast halbiert (0.6710 vs Orakel 0.6858)
-> v2 = EIN-KOPF-Betriebspunkt. Hebel "Klassen-Gewichtung im Kopf" und
"volle Datenmenge" bewusst als Future Work belassen (Saettigung/Aufwand).

PROJEKTSTATUS 03.08.2026: EXPERIMENTALPROGRAMM ABGESCHLOSSEN.
Alle Taskberichte vorhanden (1-6, 9C/9D, 10-25 inkl. 15.x/16x/18-B1/B2;
Tasks 7/8 im Gesamt-/Zwischenbericht-Bestand der Fruehphase, 16b.8 und
16b.10 bewusst uebersprungen). Alle Figuren task-referenz-frei und aus
Skripten reproduzierbar (render_legacy_figs.py ersetzt frueheren
Inline-Code). NAECHSTER SCHRITT: THESIS-SCHREIBEN.
Empfohlene Betriebspunkte: Seg = WM 16b.9-Minimal + adaptierter Kopf
(Task 23); Det = WM det_baseline + v2-Kopf (Task 25).

===========================================================================
NACHTRAG 2026-08-13: TASK 27 — SEED-ABSICHERUNG & KLEINE ZUSATZLAEUFE
(externe Experimentier-Vorschlaege; alles LOKAL, Cluster unberuehrt)
===========================================================================
  27.1 (Prio 1) SEG-KOPFADAPTATION MULTI-SEED: 3 Kopf-Trainings Seeds
       43/44/45 (WM + Vorhersagen FIX, nur Kopf-Seed variiert; adapt_seg_head
       --seed) + je Full-Val-Injection. Kernfrage: ist +0.0114 (0.5898 ->
       0.6012) unter Kopf-Seed-Variation reproduzierbar? Mittelwert +
       Streuung berichten. Seed 42 = Task-23-Wert.
  27.2 (Prio 2) DET-KOPFADAPTATION MULTI-SEED: 2 zusaetzliche v2-Trainings
       (Seeds 43/44, adapt_det_head_v2 --seed) + je 2 Injection-Evals
       (pred/real getrennt). Platztausch-Logistik wie Task 25.
  27.3 (Prio 3) FLOW AUF FULL-VAL: ERLEDIGT OHNE COMPUTE — der det. Flow-
       Forward ist der FROZEN 16b.9-Backbone, Full-Val per Konstruktion
       0.6946; Pareto-Punkt entsprechend gesetzt, Subset-Stern entfernt
       (0.6896 war reiner Subset-Effekt desselben Forwards).
  27.4 (Prio 4) BEST-OF-K-KURVE Flow: K in {1,2,4,8,16}, 300er-Subset,
       eval_vae; als Coverage-Analyse labeln (nicht einsetzbare Vorhersage).
  27.5 (Prio 5) RESSOURCENMESSUNG 3x wiederholt -> Median/Mittel +- Streuung
       in ressourcen.csv/Bericht nachtragen.
  NICHT erneut experimentiert (lt. Vorschlag + eigene Lage): Fehlerzerlegung
  (nur Umformulierung als Metrikabstaende), weitere Generativ-Laeufe.

===========================================================================
ENDE DES ARBEITSPLANS
===========================================================================
"""

if __name__ == "__main__":
    print(__doc__)
