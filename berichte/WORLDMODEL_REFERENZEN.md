# World-Model-Referenzen (Literaturliste, Stand 2026-07-23)

Gesammelt aus den Recherchen zu Task 16d/16e-Nachgang (mIoU-Hebel) und
Task-18-Vorbereitung. Komparatoren der Thesis (Task 17) markiert mit [K].
Alle Links zuletzt geprueft 2026-07-23.

## Direkte Komparatoren / naechste Verwandte

| Modell | Kernidee | Link |
|---|---|---|
| [K] DINO-Foresight (Dez 2024) | Zukunfts-Vorhersage eingefrorener DINOv2-VFM-Features per Masked Transformer; SmoothL1 im Feature-Raum; Seg/Depth/Normals via plug-and-play Heads. Direktester Verwandter unseres Setups (frozen features + Regression + Seg-Eval). | https://arxiv.org/abs/2412.11673 |
| [K] DINO-WM (2024) | Weltmodell auf eingefrorenen DINOv2-Patch-Features (Patch-Zahl im Paper nicht beziffert; 224px/Patch-14 -> 256), ViT-Predictor, aktionskonditioniert; Planung im Latent-Raum ohne Decoding. Evidenz fuer Token-Effizienz. | https://arxiv.org/abs/2411.04983 |
| [K] LeWorldModel (Maer 2026) | JEPA from pixels, ~15M Params, 1 GPU; Loss = MSE-Prediction + SIGReg (Gauss-Matching der Embeddings, lambda 0.1); kein EMA, kein Pretrained Encoder. VOLLTEXT-VERIFIZIERT (29.07.): Praediktor = TRANSFORMER (6 Layer, 16 Heads, ~10M, AdaLN-Action-Conditioning), DETERMINISTISCH — KEIN Diffusions-/FM-Anteil. Damit architektonisch UND im Zwei-Term-Rezept nahezu unser Zwilling; einziger struktureller Unterschied: Latent-Raum wird mitgelernt statt frozen/dekodierbar. PAPER-DETAILS (PDF v3, 29.07.): Autoren Maes/Le Lidec/Scieur/LeCun/Balestriero; Encoder ViT-tiny ~5M (patch 14, dim 192), Zustand = EIN [CLS]-Vektor/Frame (daher 48x schnelleres Planen, ~200x weniger Tokens als DINO-WM — alte Notiz-Zahl bezog sich hierauf); Praediktor autoregressiv ueber N-Frame-Historie mit kausaler Maske; AdaLN ZERO-INIT (= unser Identitaets-Trick); SIGReg = M=1024 Zufallsprojektionen + Epps-Pulley-Normalitaetstest, via Cramer-Wold -> isotrope Gauss (staerker als unser std-Matching: ganze Verteilungsform vs. nur per-Kanal-std). PRAEZISIERUNG der Zwillings-Analogie: Praediktor+Loss-Rezept = Zwilling; Zustandsrepraesentation maximal verschieden (192-dim Globalvektor vs. unsere raeumliche 256x128x128-Karte). | https://arxiv.org/abs/2603.19312 |

## Aktuelle Welle (letzte ~8 Monate)

| Modell | Kernidee | Link |
|---|---|---|
| VFMF (Dez 2025) | DINO-Foresight-Nachfolger: GENERATIV statt deterministisch -- autoregressives Flow Matching im VFM-Feature-Raum (kompaktes Diffusions-Latent); decodiert zu Seg/Depth/Normals/RGB. Moderner Referenzpunkt fuer Task 18. | https://arxiv.org/abs/2512.11225 |
| DeltaWorld (Apr 2026) | "A frame is worth one token": Delta-Tokens kodieren nur die Frame-Differenz; mehrere Zukuenfte in einem Forward; ~35x weniger Params als Cosmos. Radikale Residual-/Effizienz-Variante. | https://arxiv.org/abs/2604.04913 |
| DriveFuture (Mai 2026) | Zukunfts-Latents als Konditionierung der Downstream-Aufgabe (Planung); Training refined Vorhersage gegen GT-Zukunft via Cross-Attention. NAVSIM-SOTA. Stuetzt Task-Kopplung (Decoder-Task-Loss). | https://arxiv.org/abs/2605.09701 |
| ResWorld / Implicit Residual WM (Okt 2025, ICLR26) | 4D-Occupancy-Forecasting ueber TEMPORALE RESIDUEN der Szenen-Repraesentation -- Kapazitaet fokussiert auf Dynamik, Statisches wird kopiert (explizite Version unserer Gate-Logik). | https://arxiv.org/abs/2510.16729 |
| FUTURIST (Jan 2025, ICCV25) | Multimodaler Masked-Visual-Sequence-Transformer fuer semantische Zukunftsvorhersage; Masked-Modeling-Objective; SOTA Seg-Forecasting kurz/mittel. | https://arxiv.org/abs/2501.08303 |

## Etablierte Bezugspunkte (aelter, fuer Einordnung/Related Work)

| Modell | Kernidee | Link |
|---|---|---|
| BEVWorld (2024) | Multimodaler BEV-Latent-Raum (Kamera+Lidar-Tokenizer); Zukunftssequenz per Latent-DIFFUSION in einem Schritt (vermeidet autoregressive Fehlerakkumulation). | https://arxiv.org/abs/2407.05679 |
| OccWorld (2023) | 3D-Occupancy-Weltmodell; DISKRETE Tokens (VQ-VAE artig) + autoregressive Klassifikation statt Regression (kein Regression-to-the-mean). | https://arxiv.org/abs/2311.16038 |
| LAW / Latent World Model (2024) | Weltmodell (naechstes Latent aus Feature+Aktion) als selbstueberwachte HILFSAUFGABE, gemeinsam mit End-to-End-Fahraufgabe optimiert -- Task-Kopplung als Gewinnquelle. | https://arxiv.org/abs/2406.08481 |
| GAIA-1 (2023) | Generatives Weltmodell (Wayve), ~9B Params; diskrete Tokens + autoregressiver Transformer + Video-Diffusion-Decoder. Skalen-Referenz. | https://arxiv.org/abs/2309.17080 |
| Survey: World Models for AD (2025, laufend aktualisiert) | Systematischer Ueberblick (BEV-, Occupancy-, Punktwolken-, Bild-basiert; generativ vs. praediktiv). Guter Einstieg fuers Related-Work-Kapitel. | https://arxiv.org/abs/2501.11260 |
| Awesome-World-Model (GitHub-Sammlung, laufend) | Kuratierte Paper-Liste World Models AD/Robotik -- Nachschlagequelle fuer Neuerscheinungen. | https://github.com/LMD0311/Awesome-World-Model |

## Kernaussagen fuer unsere Arbeit (Kurzsynthese)

1. NIEMAND der aktuellen Welle holt mIoU-Gewinne durch weitere Trainings-Tweaks
   bei fixer deterministischer Regression -- die Richtung, die wir in 16b-16e
   ausgeschoepft haben. LeWorldModel bestaetigt unsere Minimal-Loss-Linie
   (2 Terme + Verteilungs-Regularisierer) unabhaengig.
2. Gewinne kommen aus drei Richtungen: (a) Objective-Wechsel (generativ:
   VFMF/BEVWorld; masked: FUTURIST; diskret: OccWorld), (b) Task-Kopplung
   (LAW, DriveFuture) -> bei uns: Decoder-Task-Loss, (c) reichere Features/
   Skala (DINO-Foresight, GAIA-1) -> bei uns per Design fixiert.
3. Residual-/Delta-Dynamik (ResWorld, DeltaWorld) ist die explizite Version
   unserer Gate-Struktur -- billige Ablation, Erwartung moderat.

## Methodische Anker Task 18/B1 (CVAE-Mechanik, ergaenzt 2026-07-24)

| Baustein (bei uns) | Arbeit | Link |
|---|---|---|
| CVAE-Grundgeruest (Prior/Recognition/Decoder) | Sohn et al. 2015, Conditional VAE | https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models |
| Reparametrisierung | Kingma & Welling 2014, VAE | https://arxiv.org/abs/1312.6114 |
| Gelernter Prior + z in det. Frame-Predictor (NAECHSTER VERWANDTER) | Denton & Fergus 2018, SVG-LP | https://arxiv.org/abs/1802.07687 |
| Stochastische Videovorhersage (verwandt) | Babaeizadeh et al. 2018, SV2P | https://arxiv.org/abs/1710.11252 |
| beta-Gewichtung des KL | Higgins et al. 2017, beta-VAE | https://openreview.net/forum?id=Sy2fzU9gl |
| KL-Annealing | Bowman et al. 2016 | https://arxiv.org/abs/1511.06349 |
| Free-Bits | Kingma et al. 2016 (IAF) | https://arxiv.org/abs/1606.04934 |
| FiLM-Injektion | Perez et al. 2018, FiLM | https://arxiv.org/abs/1709.07871 |

Framing: unser B1 = SVG-LP-Struktur auf BEV-Latents (klassische stochastische
Linie), bewusst abgegrenzt von der aktuellen generativen Welle (VFMF Flow
Matching, BEVWorld Diffusion, OccWorld diskret).

---------------------------------------------------------------------------
## NACHTRAG 2026-07-31 — Frozen- vs. Co-trained-Encoder (Recherche letzte
## ~6 Monate, Anlass: Betreuer-Diskussion Selten-Klassen-Decke / Task 21)

Kernfrage: trainieren aktuelle World Models Encoder/Decoder mit? Antwort:
das Feld ist ZWEIGETEILT — beide Lager sind 2026 aktiv, und im Strang MIT
dekodierbarer Metrik-Kette bleibt frozen der Standard.

| Lager | Arbeit (Datum) | Encoder/Decoder | Link |
|---|---|---|---|
| Frozen | VFMF: Forecasting VFM Features (12/2025) | Encoder frozen (VFM), Flow Matching im Feature-Raum | https://arxiv.org/abs/2512.11225 |
| Frozen | Latent Video Prediction Learns Better World Models (05/2026) | frozen V-JEPA-2 + leichte Probe SCHLAEGT voll finetuntes VideoMAE | https://arxiv.org/abs/2605.15618 |
| Frozen | Diffusion Transformer World-Action Model (06/2026) | Encoder UND Decoder frozen (SD-VAE); Benchmark 6 frozen Encoder, V-JEPA2 vorn | https://arxiv.org/abs/2606.12987 |
| Frozen | Generalist Forecasting with Frozen Video Models (07/2025) | Latent-Diffusion im frozen Backbone-Raum + Task-Readouts | https://arxiv.org/abs/2507.13942 |
| Mittelweg | TC-WM: Back to Parsimonious Latents (05/2026) | frozen Foundation-Features -> GELERNTES kompaktes Task-Latent obendrauf | https://arxiv.org/abs/2605.25620 |
| Co-trained | LeWorldModel (03/2026) | Encoder mittrainiert (SIGReg gegen Kollaps), KEIN Decoder | https://arxiv.org/abs/2603.19312 |
| Co-trained | Drive-JEPA (01/2026) | V-JEPA-Video-Backbone + Trajektorien-Distillation, End-to-End-Driving | https://arxiv.org/html/2601.22032 |
| Co-trained | AD-L-JEPA / AD-LiST-JEPA (02/2026, CVPR26) | selbst-ueberwachte LiDAR-JEPA, Encoder gelernt | https://arxiv.org/abs/2602.12540 |

EINORDNUNG FUER UNS: (1) Die JEPA-Linie trainiert Encoder mit — aber genau
diese Linie hat KEINEN Decoder und misst via Planning/Probing, nicht via
Task-Metrik. (2) Wo eine dekodierbare Kette existiert (Feature-Forecasting),
ist frozen 2026 weiter Standard, mit EXPLIZITER Evidenz frozen > finetuned
(2605.15618). (3) Der moderne Mittelweg (TC-WM) lernt ein kompaktes Latent
AUF frozen Features — wuerde bei uns aber den Injection-Messweg brechen
(privater Latentraum). Unser Setup = Lager 1 mit task-nativem BEV-Encoder;
die Task-21-Decke ist der dokumentierte Preis dieser Wahl.

---------------------------------------------------------------------------
## NACHTRAG 2026-08-01 — Kuratierte Uebernahme aus externer KI-Bewertung
## (Dokument des Nutzers; jede Referenz VOR Aufnahme gegen arXiv verifiziert)

UEBERNOMMEN (verifiziert, mit Rolle in unserer Argumentation):
| Arbeit | Link | Rolle bei uns |
|---|---|---|
| LAW: Enhancing End-to-End AD with Latent World Model (ICLR 2025) | https://arxiv.org/abs/2406.08481 | NAECHSTER VERWANDTER der deterministischen Linie: self-supervised Future-Latent-Prediction als Hilfsziel; bei uns Forecasting als EIGENSTAENDIGE Aufgabe mit frozen Task-Metrik-Kette |
| World4Drive (ICCV 2025) | https://arxiv.org/abs/2507.00603 | Anker der PLANNING-Linie: intention-aware Multi-Traj-Bewertung im Latentraum; Kontrast: wir bewerten Wahrnehmungs-Metriken, nicht Trajektorien |
| Vista (2024) | https://arxiv.org/abs/2405.17398 | Anker der GENERATIVEN SIMULATOR-Linie (High-Fidelity-Video); Abgrenzung "realistische Zukunft" vs unser "decision-relevantes Latent-Forecasting" |

BEREITS IN DER LISTE: BEVWorld, OccWorld (Kategorien-Anker unveraendert).
NICHT UEBERNOMMEN: LCDrive / FutureX / MindDrive (im externen Dokument ohne
arXiv-IDs, per Suche nicht verifizierbar -> potentiell halluziniert);
PreWorld 2502.07309 (occupancy, plausibel aber ungeprueft — nur bei Bedarf
mit Vollverifikation). DriveDreamer/Copilot4D: generative Simulator-Linie,
fuer unsere Argumentation durch Vista abgedeckt.

SONSTIGE VERWERTUNG des Dokuments (Task-22-Kontext): (a) Drei-Komponenten-
Zerlegung Decoder-Decke/Aleatorik/Forecasting als explizites Thesis-Element
(Figur 21_error_decomposition); (b) Occupancy-CHANGE-Masken als optionale
22.4-Loss-Variante (Zellgewichte aus GT-Aenderungsregionen — Infrastruktur
vorhanden); (c) Distanz-/Groessen-stratifizierte Evals als billige Analyse-
Erweiterung. Rollout-Curriculum/Uncertainty-Head: bereits geprueft bzw.
durch Task 18 beantwortet, nicht erneut aufgegriffen.
