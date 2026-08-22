# Task 18/B1 — CVAE-Kopf: Abschlussbericht

Datum: 2026-07-24. Jobs: 124248 (Training beta-Sweep, 3 GPU, ~13.5h) und
124362 (Sample-Eval, 1 GPU, 3.5min). Methodik/Mathematik:
TASK18_METHODIK_VAE_DIFFUSION.md. Literatur: WORLDMODEL_REFERENZEN.md.

## Fragestellung

Bringt eine stochastische latente Variable (CVAE, SVG-LP-Muster) auf dem
deterministischen 16b.9-Fundament schaerfere und vielfaeltigere Vorhersagen —
d.h. loest sie das Regression-to-the-Mean-Problem (Blur/std-Gap), ohne den
mIoU zu kosten?

## Aufbau (modular, Default aus = bit-identisch)

vae_mode=cvae baut Prior p(z|h), Posterior q(z|h,x) und z-FiLM (zero-init)
auf das bestehende Modell (+370k Params, +6%); z_dim=32. Loss = smooth_l1 +
std + beta*KL mit Annealing (5 Ep) und Free-Bits (0.05/Dim). Training from
scratch (Warm-Start beguenstigt Collapse), beta-Sweep {0.001, 0.01, 0.1},
Headline-Regime. Inferenz: z=mu_p ("mean", deterministisch) oder z~p(z|h)
("sample"). Alle Gates vor Compute gruen (off-Aequivalenz bit-genau,
Gradientenfluss Prior/Posterior/FiLM, Free-Bits-Designfehler 0.5->0.05 im
Smoke-Test gefunden und behoben).

## Ergebnisse

### Trainingsphase (Job 124248, Full-Val 5743, z=mean)

| beta  | mIoU (z=mean) | d vs off=0.6946 | KL-Ende [nats] | Collapse? |
|-------|--------------:|----------------:|---------------:|-----------|
| 0.001 | 0.6875        | -0.0071 (n.s.)  | 0.43           | nein      |
| 0.01  | 0.6883        | -0.0063 (n.s.)  | 0.26           | nein      |
| 0.1   | 0.6878        | -0.0068 (n.s.)  | 0.22           | nein      |

mIoU-GUARD BESTANDEN: die Stochastik kostet keinen signifikanten mIoU.
KEIN Posterior-Collapse: KL stabil 0.2-0.4 nats ueber das ganze Training.

### Sample-Eval (Job 124362, 300 Subset-Samples, K=8 Prior-Samples)

| beta  | Schaerfe mean | Schaerfe Sample | Divers. Latent | Divers. Masken | mIoU mean | mIoU Sample |
|-------|--------------:|----------------:|---------------:|---------------:|----------:|------------:|
| 0.001 | 0.857         | 0.856           | 0.008          | 0.0006         | 0.6838    | 0.6839      |
| 0.01  | 0.853         | 0.853           | 0.007          | 0.0006         | 0.6858    | 0.6856      |
| 0.1   | 0.861         | 0.864           | 0.005          | 0.0005         | 0.6859    | 0.6857      |

(Schaerfe = Gradient-Magnitude-Ratio pred/real, 1.0 = real-scharf.
Divers. Masken = Anteil Zellen, in denen die K decodierten Masken uneins sind.)

## KERNBEFUND: technisch gesund, funktional quasi-deterministisch

1. SAMPLES SIND NICHT SCHAERFER: Gradient-Ratio Sample == Mean (~0.86 bei
   allen beta). Die Blur-Luecke zur Realitaet (~14% fehlende Gradient-Energie)
   bleibt vollstaendig bestehen.
2. KEINE FUNKTIONALE DIVERSITAET: die 8 Samples sind in nur ~0.06% der
   Maskenzellen uneins; im Panel (18_vae_panel_*.png) sind Mittelwert und
   Samples mit blossem Auge ununterscheidbar. Latent-Streuung ueber Samples
   nur 0.5-0.8% der realen Kanal-std.
3. Konsistent ueber ALLE beta (3 Groessenordnungen) -> kein Tuning-Problem,
   sondern eine strukturelle Grenze der Sonde.

MECHANIK-ERKLAERUNG: Der KL zeigt, dass z ~0.3 nats Information TRAEGT — aber
die zero-init-FiLM-Injektion (ein globaler Kanal-Scale/Shift auf die Tokens)
setzt diese Information kaum in raeumlich lokalisierte Output-Varianz um. Auf
dem starken deterministischen Fundament lernt der Decoder, z im Wesentlichen
zu ignorieren: die Rekonstruktion ist ohne z-Nutzung fast genauso gut, und
der smooth_l1-Anteil bestraft jede z-getriebene Abweichung vom bedingten
Mittelwert. Free-Bits verhindern den KL-Kollaps (q!=p bleibt), aber nicht die
funktionale Ignoranz im Decoder — ein bekanntes CVAE-Verhalten auf starken
Konditionierungen (vgl. SVG-Diskussion). Ein 32-dim globales z ist zudem
strukturell ungeeignet, PRO-OBJEKT-Mehrdeutigkeit (dieses Auto biegt ab,
jenes nicht) auszudruecken — dafuer braeuchte es raeumlich aufgeloeste
Stochastik.

## Einordnung + Konsequenz

- Der B1-Befund ist das BASELINE-ERGEBNIS des generativen Kapitels: die
  klassische stochastische Sonde (CVAE/SVG-Linie, minimaler Eingriff)
  REICHT NICHT, um die deterministische Blur-Decke zu durchbrechen.
- Das deckt sich mit der Feldentwicklung: die aktuellen Feature-Space-
  World-Models (VFMF Dez 25, FlowWM Jun 26) adressieren exakt dieses Problem
  ("collapse multimodal futures into a single blurry mean") und setzen dafuer
  FLOW MATCHING statt VAE-Latents ein — raeumlich aufgeloester Transport
  statt globalem Konditionierungsvektor.
- NAECHSTER SCHRITT (B2): Flow-Matching-Residual-Head auf dem eingefrorenen
  16b.9-Backbone (Plan folgt separat). Diffusion (DDPM) wird als verwandter
  Bezugspunkt im Methodik-Doc behandelt, aber nicht separat implementiert
  (FM = einfacheres Objective, schnelleres Sampling, direktere 2026-Anker).

## Artefakte

- Checkpoints: checkpoints/task18/vae_{0.001,0.01,0.1}/phase2/best_miou.pt
- JSONs: predictions/task18/{vae_*_fullval,eval_vae_*}.json, Panels
  predictions/task18/panel_*.npz
- Figuren: visualizations/18_vae_panel_{0.001,0.01,0.1}.{png,pdf}
- Code: vae_mode/vae_z_dim (Code/config.py), CVAE-Module+Forward
  (Code/bev_world_model.py), KL+Annealing+Free-Bits (train_linux.py),
  --z_mode (eval_full_val.py), eval_vae.py, render_18_panel.py
- Configs: config_vae_{0.001,0.01,0.1}.yaml (+_eval-Varianten, stage_to_shm
  false); sbatch_18_vae.sh, sbatch_18_eval.sh

## Lektionen

- Free-Bits-Schwelle IMMER gegen den erwarteten Anfangs-KL dimensionieren
  (0.5/Dim = 16 nats Freibetrag haette den Prior nie trainieren lassen).
- KL > 0 ist NOTWENDIG, aber nicht HINREICHEND fuer funktionale Stochastik —
  die Output-Diversitaet muss separat gemessen werden (eval_vae.py), sonst
  taeuscht ein gesund aussehender KL echte Generativitaet vor.
- 3-GPU-Jobs auf 48 CPUs sind loader-bound (data% ~65, +80% Epochenzeit);
  kuenftig --cpus-per-task 64-96 oder num_workers 12.
