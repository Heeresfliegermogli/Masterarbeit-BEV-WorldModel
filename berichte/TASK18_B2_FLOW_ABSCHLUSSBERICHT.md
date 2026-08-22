# Task 18/B2 — Flow-Matching-Residual-Kopf: Abschlussbericht

Datum: 2026-07-24/25. Jobs: 124417 (Training, 1 GPU, ~3.5h; 124413 verworfen,
Ladepfad-Bug), 124362/124517 (Sample-Evals inkl. Best-of-K), 124521 (Rollout
det vs. Sample). Methodik/Mathematik: TASK18_METHODIK_VAE_DIFFUSION.md §3.
B1-Vorlaeufer: TASK18_B1_CVAE_ABSCHLUSSBERICHT.md.

## Fragestellung

Liefert raeumlich aufgeloeste generative Modellierung (Flow Matching, der
2026-Standard: VFMF/FlowWM) die Multimodalitaet und Schaerfe, an der die
globale CVAE-Sonde (B1) gescheitert ist — und zu welchem Preis?

## Aufbau (modular, Default aus = bit-identisch)

flow_head=flow baut ein Geschwindigkeitsfeld v_theta (8 Residual-Conv-Bloecke,
Breite 224, tau-FiLM, zero-init Output): 6.518.208 Params = bewusste
KAPAZITAETS-PARITAET zum 6.054.144-Backbone ("gleiche Power, anderes
Objective"). Backbone = EINGEFRORENER 16b.9-Headline-Checkpoint (0.6946);
gelernt wird nur die Residuum-Verteilung p(r|c), r = x_real - x_det:
L_FM = ||v_theta(r_tau, tau, c) - (r - eps)||^2, Sampling per Euler-ODE.
forward() bleibt unveraendert (Flow nur ueber explizite Methoden) -> der
mIoU-Guard ist per Konstruktion das Backbone. Training 15 Ep (FM-Val-Loss
0.43 -> 0.26, konvergent); Selektion per Flow-Sample-mIoU. Gates gruen
(off-Aequivalenz bit-genau, Freeze verifiziert, Sampler seed-reproduzierbar).
LEKTION: --init_weights brauchte allow_missing (neue flow.*-Keys fehlen im
alten Checkpoint; strikter Load crashte Job 124413).

## Ergebnisse

### Single-Step Sample-Eval (300 Subset, K=8) — Vergleich mit B1

| Metrik                  | CVAE (b=0.01) | Flow s=10 | Flow s=1 |
|-------------------------|--------------:|----------:|---------:|
| Schaerfe-Ratio Sample   | 0.853         | 4.21      | 4.32     |
| std-Ratio Sample        | 0.980         | 1.57      | 1.59     |
| Diversitaet Masken      | 0.0006        | **0.0221**| 0.0220   |
| mIoU mean (Backbone)    | 0.6858        | 0.6896    | 0.6896   |
| mIoU Sample (1 zufaellig)| 0.6856       | 0.6455    | 0.6450   |
| mIoU Best-of-8          | 0.6873        | **0.6783**| 0.6776   |

### Rollout k=1..4 (1000 seed-42-Fenster, Job 124521)

| k | det mIoU | Sample mIoU | det std | Sample std | Persistenz |
|---|---------:|------------:|--------:|-----------:|-----------:|
| 1 | 0.6928   | 0.6496      | 0.986   | 1.581      | 0.5207     |
| 2 | 0.5974   | 0.5498      | 0.988   | 1.760      | 0.4118     |
| 3 | 0.5256   | 0.4773      | 0.999   | 1.846      | 0.3652     |
| 4 | 0.4698   | 0.4202      | 1.013   | 1.901      | 0.3333     |

Figuren: 18_vae_panel_flow_s10.* (Panel mit sichtbarer Sample-Diversitaet),
18_flow_rollout.* (Rollout-Kurven).

## KERNBEFUNDE

1. **ECHTE MULTIMODALITAET — der Mechanismus funktioniert.** Masken-
   Diversitaet 0.0221 = 37x CVAE; im Panel strukturell plausibel (Seitenarme/
   Walkway-Segmente variieren ueber Samples), kein blosses Pixelrauschen.
   Raeumlich aufgeloester Transport liefert, was das globale 32-dim-z nicht
   konnte — die VFMF/FlowWM-These bestaetigt sich im BEV-Latent-Setting.
2. **ZERLEGUNG DES SAMPLE-MALUS (Best-of-K):** -0.044 (Einzel-Sample vs mean)
   = -0.033 VERTEILUNGS-PREIS (durch Best-of-8 rueckholbar -> die Verteilung
   DECKT die echte Zukunft; Perception-Distortion-Tradeoff, Blau/Michaeli
   2018: der bedingte Mittelwert ist fuer mIoU-artige Metriken strukturell
   optimal) + -0.011 KALIBRIERUNGSFEHLER (std-Ratio 1.57 statt ~1.0; auch das
   beste Sample traegt den Rausch-Boden). Beim CVAE hebt Best-of-8 nichts
   (+0.002) — ohne Diversitaet gibt es nichts auszuwaehlen.
3. **ROLLOUT: das Literatur-Argument zieht in UNSEREM Setting nicht — und
   der Grund ist eine eigene Erkenntnis.** Die Sample-Trajektorie ueberholt
   den rekursiven Mittelwert nirgends (Luecke konstant ~ -0.045; rel.
   Degradation det -32% vs Sample -35%). Ursache: unser deterministisches
   Backbone bleibt ueber den ganzen Rollout VARIANZ-KALIBRIERT (std 0.99-1.01)
   — das ERBE DES lambda_std-TERMS (16b). Der Varianz-Kollaps, dessen
   Reparatur generative Rollouts in der Literatur motiviert (BEVWorld), ist
   bei uns bereits deterministisch und billig behoben. Der Flow-
   Kalibrierungsfehler kompoundiert stattdessen (1.58 -> 1.90).
4. **1-Schritt-Sampling genuegt:** steps=1 == steps=10 in allen Metriken —
   das Feld ist quasi tau-unabhaengig; die FlowWM-artige One-Step-Projektion
   funktioniert out of the box (Sampling-Kosten ~1 Zusatz-Forward).
5. Schaerfe-Ratio 4.2 ist als "Ueberschiessen" zu lesen, nicht als Gewinn:
   ueberschuessige Hochfrequenz-Energie im Latent (Unterkalibrierung nach nur
   15 Ep Kopftraining), die der Decoder groesstenteils wegfiltert.

## GESAMTFAZIT TASK 18 (B1+B2)

- CVAE (klassische Sonde): technisch gesund, funktional quasi-deterministisch
  -> Baseline-Negativ.
- Flow Matching (moderner Standard): echte, plausible Multimodalitaet mit
  quantifizierter Zerlegung des Genauigkeits-Preises; Kalibrierung nach
  kurzem Training unvollstaendig (laengeres Kopftraining/Min-SNR-Weighting =
  benannte Future Work).
- WICHTIGSTE UEBERGREIFENDE ERKENNTNIS: Der deterministische 16b.9-
  Betriebspunkt (smooth_l1 + lambda_std) ist auch im Rollout der genaueste
  UND varianz-stabile — der std-Term liefert die Rollout-Stabilitaet, die
  generative Modelle sonst exklusiv versprechen. Generativ lohnt in diesem
  Setting fuer MULTIMODALITAET (Coverage, Diversitaet, Unsicherheits-
  quantifizierung), nicht fuer Genauigkeit.
- Konsistent mit der 16f-Deckenanalyse: mIoU ~0.69 = aleatorische Decke des
  Setups (per-Klasse: drivable 0.90 vs stop_line 0.56 — Restfehler sitzt in
  duennen/dynamischen Strukturen); Details -> Task-19-Abschluss.

## Artefakte

- Checkpoint: checkpoints/task18/flow/phase2/best_miou.pt (nur flow.* trainiert)
- JSONs: predictions/task18/eval_flow_s{10,1}.json (+_v2 mit Best-of-K),
  eval_vae_*_v2.json (B1-Nachmessung), rollout_{det,sample}.json
- Figuren: visualizations/18_vae_panel_flow_s10.*, 18_flow_rollout.*
- Code: Code/flow_head.py (FlowHead + Euler-Sampler); flow_head/flow_dim/
  flow_layers (Code/config.py); flow_sample() (Code/bev_world_model.py);
  FM-Zweige in train_linux.py (Freeze, Train/Val, flow_val_steps);
  allow_missing (Code/checkpointing.py); --flow_steps (eval_vae.py);
  --mode det|sample (rollout_eval.py). ALLES Default-neutral.
- Configs: config_flow.yaml (+_eval); sbatch_18_flow.sh, sbatch_18_bestofk.sh,
  sbatch_18_rollout.sh.

## Lektionen

- Best-of-K IMMER mitmessen, wenn Verteilungen bewertet werden — ohne die
  Zerlegung waere der Sample-Malus als "Flow ist schlechter" fehlgedeutet
  worden (er ist zu 3/4 der legitime Verteilungs-Preis der Metrik).
- Kalibrierung (std-Ratio der Samples) ist DIE Qualitaetsgroesse eines
  generativen Kopfs — Diversitaet allein genuegt nicht.
- Neue Module + alte Checkpoints: Ladepfade brauchen explizites
  allow_missing-Design (strict bleibt Default).

---------------------------------------------------------------------------
## NACHTRAG 2026-07-25 — B2.1: Kalibrierungs-Nachlauf (Job 124563)

AUFBAU: (1) FM-Kopf 30 statt 15 Epochen (from scratch, sonst identisch;
FM-Val-Loss 0.262 -> 0.252). (2) POST-HOC RESIDUAL-SKALIERUNG als
Kalibrier-Regler (Temperature-Analogon): x = x_det + s * r_hat, Sweep
s in {1.0, 0.8, 0.6, 0.5} (scale-Param in flow_sample / --flow_scale in
eval_vae.py). Eval wie gehabt: K=8, steps=1, 300er-Subset.

ERGEBNIS:

| s   | std-Ratio | Diversitaet | mIoU Sample | Best-of-8 | vs mean 0.6896 |
|-----|----------:|------------:|------------:|----------:|----------------|
| 1.0 | 1.535     | 0.0223      | 0.6451      | 0.6776    | -0.012         |
| 0.8 | 1.342     | 0.0182      | 0.6624      | 0.6900    | +-0            |
| 0.6 | 1.176     | 0.0141      | 0.6748      | 0.6974    | +0.008         |
| 0.5 | 1.106     | 0.0120      | 0.6798      | 0.6998    | **+0.010**     |

Figur: 18_flow_calibration.* (Dosis-Wirkung ueber s).

BEFUNDE:
1. KALIBRIERUNGSREST GESCHLOSSEN UND UMGEKEHRT: ab s<=0.8 gilt
   Best-of-8 >= mean; bei s=0.5 liegt Best-of-8 +0.010 UEBER dem
   deterministischen Mittelwert — die kalibrierte Verteilung DECKT die echte
   Zukunft besser ab als die gehedgte Punktschaetzung (das theoretisch
   erwartete Coverage-Ergebnis). Diversitaet bleibt substanziell
   (0.012 = 20x CVAE).
2. EHRLICHE EINORDNUNG: Best-of-8 ist eine COVERAGE-Metrik mit Orakel-
   Auswahl (bestes Sample per GT-Vergleich gewaehlt) — KEIN einsetzbarer
   Praediktor. Aussage: "die Verteilung enthaelt bessere Zukuenfte als der
   Mittelwert", nicht "das Modell sagt besser vorher". Fuer Downstream-
   Nutzung (Planung ueber mehrere Zukuenfte, Risiko-Bewertung) ist genau
   diese Eigenschaft der Mehrwert.
3. TRADE-OFF-REGLER: s steuert monoton Varianz-Kalibrierung gegen
   Diversitaet (Dosis-Wirkungs-Kurve) — ein interpretierbarer, post-hoc
   einstellbarer Arbeitspunkt ohne Retraining. EMPFEHLUNG: s=0.5
   (std 1.11, bestes Einzel-Sample- UND Best-of-8-mIoU); s~0.45 traefe
   std=1.0 exakt (Feintuning bei Bedarf).
4. Die 30 Ep allein aendern wenig (s=1.0-Zeile ~= 15-Ep-Werte: 0.6451 vs
   0.6450) — der WIRKSAME Hebel ist die Skalierung; das Residuum-Feld war
   schon nach 15 Ep im Wesentlichen gelernt, nur zu energiereich.
   (Konsistent: Schaerfe-Ratio bleibt auch bei s=0.5 mit 2.26 ueber 1.0 —
   die Samples tragen weiterhin mehr Hochfrequenz als real; std ist die
   kalibrierte Groesse, Gradient-Energie nicht vollstaendig.)

REVISION DER B2-KERNAUSSAGE (Punkt 2 des Hauptberichts): Der -0.011-
"Kalibrierungsrest" ist KEIN offenes Problem mehr, sondern per post-hoc
Skalierung geschlossen; der Gesamtbefund verschiebt sich von "Kalibrierung
unvollstaendig (Future Work)" zu "kalibrierbar ohne Retraining; Coverage-
Gewinn +0.010 belegt". Die Genauigkeits-Rangfolge fuer den EINSATZ als
Praediktor bleibt unveraendert (deterministisch > Einzel-Sample).

Artefakte-Ergaenzung: checkpoints/task18/flow30/phase2/best_miou.pt;
predictions/task18/eval_flow30_sc{1.0,0.8,0.6,0.5}.json;
visualizations/18_flow_calibration.*; config_flow30.yaml,
sbatch_18_flow30.sh, render_18_calibration.py; scale-Param in
Code/bev_world_model.py (flow_sample) + --flow_scale (eval_vae.py).
