# Task 18 — Methodik: CVAE und Diffusion/Flow Matching auf BEV-Latents

Datum: 2026-07-24. Methodisches Begleitdokument zum generativen Strang:
erklaert Ziel, Vorgehen und Mathematik BEIDER Instrumente bezogen auf unsere
Anwendung (Prognose des naechsten BEV-Latents x_next aus 3 Kontext-Frames).
CVAE (B1) ist implementiert und trainiert; Diffusion/Flow Matching (B2) ist
der geplante naechste Schritt und in den aktuellen Feature-Space-World-Models
(VFMF Dez 25, FlowWM Jun 26) de-facto Standard. Referenzen:
WORLDMODEL_REFERENZEN.md (Abschnitt "Methodische Anker Task 18/B1").

Notation: c = Kontext (3 Input-Frames bzw. deren Transformer-Repraesentation),
x = Ziel-Latent x_next [256,128,128], z = latente Zufallsvariable (dim 32).

---------------------------------------------------------------------------
## 1. Das gemeinsame Problem: Regression-to-the-Mean

Unser deterministisches Modell minimiert eine Distanz (SmoothL1) zwischen
Vorhersage f(c) und Ziel x. Der Minimierer einer solchen punktweisen Distanz
ist (naeherungsweise) der BEDINGTE ERWARTUNGSWERT:

    f*(c) = E[x | c]

Ist die Zukunft mehrdeutig — mehrere plausible x bei gleichem c (Fussgaenger
geht/steht, Auto biegt ab/faehrt weiter) — dann ist E[x|c] der DURCHSCHNITT
dieser Moeglichkeiten: raeumlich glatt, kontrastarm, mit zu geringer Varianz.
Genau das messen wir seit Task 15 als std-Gap; FlowWM (2026) nennt es
"collapse multimodal futures into a single blurry mean". Ein generatives
Modell ersetzt die Punktschaetzung durch die VERTEILUNG p(x|c): jedes Sample
ist eine in sich konsistente, scharfe Zukunft; der Mittelwert ueber viele
Samples reproduziert den unscharfen Durchschnitt.

---------------------------------------------------------------------------
## 2. CVAE (B1, implementiert)

### 2.1 Idee in unserer Anwendung

Eine kleine Zufallsvariable z (32-dim) kodiert, WELCHE der moeglichen
Zukuenfte eintritt. Der bestehende deterministische Praediktor bleibt das
Rueckgrat; z wird per FiLM in die Tokens vor dem Output-Head injiziert
(SVG-LP-Muster: gelernter Prior + stochastisches Latent auf deterministischem
Frame-Predictor). z komprimiert NICHT die Szene (das taete die FlowWM-Kritik
an VAE-World-Models treffen), sondern nur die Rest-Mehrdeutigkeit.

Drei Netze (alle klein, zusammen +370k Parameter = +6%):
  - Prior          p(z|c)   = N(mu_p(h), diag(sigma_p^2(h)))
  - Posterior      q(z|c,x) = N(mu_q(h,x), diag(sigma_q^2(h,x)))   [nur Training]
  - Decoder        p(x|c,z) = unser Head, FiLM-konditioniert auf z
mit h = Mean-Pool der Transformer-Tokens [256] und x-Zusammenfassung ueber
das vorhandene Downsampling (AvgPool -> Mean-Pool [256], kein neuer Encoder).

### 2.2 Mathematik: ELBO

Die Log-Likelihood log p(x|c) ist intraktabel. Mit einer Hilfsverteilung
q(z|c,x) gilt die Evidence Lower Bound (Jensen-Ungleichung):

    log p(x|c) >= E_{z~q}[ log p(x|c,z) ]  -  KL( q(z|c,x) || p(z|c) )
                  \_______________________/     \________________________/
                   Rekonstruktion                Regularisierung

Wir maximieren die ELBO, d.h. minimieren:

    L = L_recon( f(c,z), x )  +  beta * KL( q || p ),     z ~ q(z|c,x)

  - L_recon: unser bewaehrter smooth_l1 + lambda_std*std-Term (Surrogat fuer
    -log p(x|c,z); eine Gauss-Likelihood ergaebe MSE — SmoothL1 ist die
    robuste Variante, 16b.7-Befund).
  - beta: KL-Gewicht (beta-VAE, Higgins 2017). Sweep {0.001, 0.01, 0.1}.

REPARAMETRISIERUNG (Kingma/Welling 2014) macht das Sampling differenzierbar:

    z = mu_q + sigma_q * eps,   eps ~ N(0, I)

Gradienten fliessen durch mu_q, sigma_q in den Posterior (und via FiLM/Decoder
in den Rest).

KL zweier diagonaler Gauss-Verteilungen, geschlossen pro Dimension i:

    KL_i = 1/2 * [ log(sigma_p,i^2 / sigma_q,i^2)
                   + (sigma_q,i^2 + (mu_q,i - mu_p,i)^2) / sigma_p,i^2  -  1 ]
    KL   = sum_i KL_i     [nats]

### 2.3 Collapse-Gegenmittel

POSTERIOR-COLLAPSE = das Modell ignoriert z (KL -> 0, q == p), weil die
Rekonstruktion auch ohne z gut genug ist — Hauptrisiko auf unserem starken
deterministischen Fundament. Gegenmittel (beide implementiert):
  - KL-ANNEALING (Bowman 2016): beta(epoch) = beta * min(1, epoch/5) — die
    Rekonstruktion darf z erst "entdecken", bevor der KL-Druck greift.
  - FREE BITS (Kingma 2016): KL_i wird erst OBERHALB einer Schwelle bestraft:
    KL_pen = sum_i max(KL_i, lambda_fb), lambda_fb = 0.05/Dim. Unterhalb kein
    Gradient -> z darf "gratis" ~1.6 nats Information tragen.
    (Urspruenglich 0.5/Dim geplant — haette 16 nats Freibetrag bedeutet und
    den Prior lahmgelegt; im Smoke-Test entdeckt und korrigiert.)

### 2.4 Training vs. Inferenz + Ergebnisse Trainingsphase

  Training:  z ~ q(z|c,x)      (Posterior sieht das echte Ziel)
  Inferenz:  z = mu_p(c)       -> deterministische "mean"-Vorhersage
             z ~ p(z|c)        -> stochastisches Sample (eine Zukunft)

Ergebnisse (Job 124248, from scratch, Val-Loss-Patience Ep. 37-40):
  - mIoU-Guard (Full-Val, z=mean, vs off=0.6946): 0.6875/0.6883/0.6878 fuer
    beta 0.001/0.01/0.1 — alle n.s. -> Stochastik kostet keinen mIoU.
  - KL-Endwerte 0.43/0.26/0.22 nats: stabil, KEIN Collapse; z traegt eine
    kompakte Restunsicherheit (~0.3 nats), nicht die Szene.
Offen (eval_vae.py, Job 124362): Schaerfe (Gradient-Magnitude-Ratio),
Masken-Diversitaet ueber K=8 Prior-Samples, Panel Mittelwert vs. Samples.

---------------------------------------------------------------------------
## 3. Diffusion und Flow Matching (B2, geplant — heutiger Standard)

### 3.1 Idee in unserer Anwendung

Statt z + Decoder wird die BEDINGTE VERTEILUNG p(x|c) direkt als Transport
von Rauschen zu Daten gelernt. In unserer Anwendung: ein Netz erzeugt das
naechste BEV-Latent (oder das Residuum zum letzten Frame) aus Gauss-Rauschen,
konditioniert auf den Transformer-Kontext c. Vorbilder: BEVWorld (Diffusion
auf BEV-Latents), VFMF/FlowWM (Flow Matching im VFM-Feature-Raum).

### 3.2 Diffusion (DDPM, Ho et al. 2020)

VORWAERTSPROZESS (fix, kein Lernen): Daten x_0 werden ueber Pseudo-Zeit
tau = 1..T schrittweise verrauscht; geschlossen:

    x_tau = sqrt(abar_tau) * x_0 + sqrt(1 - abar_tau) * eps,   eps ~ N(0,I)

mit Rauschplan abar_tau (monoton fallend, abar_T ~ 0 -> reines Rauschen).

RUECKWAERTSPROZESS (gelernt): ein Netz eps_theta(x_tau, tau, c) schaetzt das
addierte Rauschen. Trainingsziel (vereinfachte ELBO):

    L_DDPM = E_{x_0, tau, eps} || eps - eps_theta(x_tau, tau, c) ||^2

SAMPLING: von x_T ~ N(0,I) iterativ entrauschen (T bzw. wenige DDIM-Schritte),
jede Trajektorie = ein scharfes, konsistentes Sample aus p(x|c).

### 3.3 Flow Matching (Lipman et al. 2023; VFMF/FlowWM-Variante)

Statt stochastischem Entrauschen wird ein GESCHWINDIGKEITSFELD gelernt, das
eine einfache Prior-Verteilung DETERMINISTISCH in die Datenverteilung
transportiert. Lineare Interpolation (Rectified Flow / OT-Pfad):

    x_tau = (1 - tau) * x_noise + tau * x_data,     tau ~ U[0,1]
    Zielgeschwindigkeit:  u = x_data - x_noise      (konstant je Paar)

    L_FM = E || v_theta(x_tau, tau, c) - u ||^2

SAMPLING: ODE  dx/dtau = v_theta(x, tau, c)  von tau=0 (Rauschen) nach tau=1
(Daten) integrieren — dank gerader Pfade genuegen WENIGE Schritte (FlowWM:
"one-step projection"). Vorteile gegenueber DDPM: einfacheres Ziel, kein
Rauschplan-Tuning, schnelleres Sampling; gegenueber CVAE: kein Posterior/
Collapse-Problem, ausdrucksstaerkere Multimodalitaet, nachweislich scharfe
Samples (VFMF/FlowWM). Preis: Sampling ist iterativ (Mehrfach-Forward statt
ein Forward), Training ein neues Objective statt eines Zusatz-Terms.

### 3.4 Konkreter B2-Zuschnitt (Vorschlag, noch nicht gebaut)

  - Ort: 32x32-Grid nach Downsampling (Kosten!) oder Residuum x - x_last
    auf 128x128; Konditionierung: Transformer-Tokens (Cross-Attention oder
    Concat), Architektur: kleines U-Net/ConvNet v_theta.
  - Modular wie B1: config-Schalter (z.B. gen_mode: off|cvae|flow), Default
    off = bit-identischer Altpfad; Guards wie gehabt.
  - Bewertung identisch zu B1 (mIoU-Guard mit deterministischem 1-Schritt-
    Mean-Sample, Schaerfe, Diversitaet, Panel) -> direkte CVAE-vs-FM-Tabelle.

---------------------------------------------------------------------------
## 4. Gegenueberstellung in unserem Kontext

| Kriterium              | CVAE (B1)                  | Diffusion/FM (B2)          |
|------------------------|----------------------------|----------------------------|
| Trainingsaufwand       | +1 Loss-Term, 1 Lauf       | neues Objective, eigenes Netz |
| Inferenz (1 Sample)    | 1 Forward                  | N ODE-/Denoise-Schritte    |
| Multimodalitaet        | via 32-dim z (begrenzt)    | voll (Feld ueber x-Raum)   |
| Hauptrisiko            | Posterior-Collapse         | Kosten/Komplexitaet        |
| Literatur-Rolle 2026   | klassische Baseline (SVG)  | Standard (VFMF/FlowWM/BEVWorld) |
| Unsere Rolle           | minimale stochastische Sonde | Eskalation, falls B1-Diversitaet zu klein |

Entscheidungslogik: zeigt der B1-Sample-Eval echte Schaerfe-/Diversitaets-
Gewinne, traegt das Kapitel mit CVAE + FM als Future Work. Bleibt die
Diversitaet marginal (kleiner KL ~0.3 nats als Warnsignal), ist B2 die
literaturgestuetzte Eskalation innerhalb der Zeitbox.
