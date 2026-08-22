# Task 16f — Letzte mIoU-Hebel: Decoder-Task-Loss, Kapazitaet, Residual

Datum: 2026-07-23. Jobs 124076 (tl16f: Task-Loss-Finetune, 3 GPU, ~4h) und
124077 (cap16f: Kapazitaet+Residual from scratch, 3 GPU, ~11h).
Motivation aus dem Literatur-Abgleich (WORLDMODEL_REFERENZEN.md): die aktuelle
World-Model-Welle holt Gewinne aus (a) Task-Kopplung (LAW/DriveFuture),
(b) Skala, (c) Residual-/Delta-Struktur (ResWorld/DeltaWorld) -- H1/H2/H3
testen genau diese drei Richtungen in unserem Setup.

## Ergebnis (Full-Val, 5743 Fenster, vs. off=0.6946, Schwelle 0.014)

| Variante   | Hebel                              | mIoU   | d vs off | Signifikanz |
|------------|------------------------------------|-------:|---------:|-------------|
| off        | Referenz (16b.9-Minimal)           | 0.6946 |  --      | --          |
| residual   | hartes Residual statt Gate (H3)    | 0.6936 | -0.0010  | **n.s.**    |
| cap_l8     | n_layers 4->8, 9.2M Params (H2)    | 0.6927 | -0.0019  | **n.s.**    |
| tl_01      | Decoder-Task-Loss 0.1 (H1)         | 0.6788 | -0.0158  | schlechter  |
| tl_05      | Decoder-Task-Loss 0.5 (H1)         | 0.6717 | -0.0229  | schlechter  |
| tl_10      | Decoder-Task-Loss 1.0 (H1)         | 0.6688 | -0.0258  | schlechter  |
| cap_d384   | d_model 384 (H2b)                  | --     | --       | ABGESTUERZT |

Figur: visualizations/16f_miou_lever2.{png,pdf}.

## KERNBEFUNDE

**H1 (Task-Kopplung): Decoder-Task-Loss SCHADET monoton und dosisabhaengig.**
BCE(decode(pred), decode(real)) durch den eingefrorenen Decoder (15-Ep-Finetune
auf dem 0.6946-Checkpoint, lr 2e-5) senkt den mIoU um -0.016..-0.026, streng
monoton in lambda_task. Die Dosis-Monotonie zeigt: der Term selbst ist das
Problem, nicht das Finetune-Regime. Interpretation: der BCE-Gradient zieht das
Latent in decoder-gefaellige, aber von der echten Latent-Verteilung wegfuehrende
Richtungen -- die Latent-Regression (smooth_l1+std) ist bereits das bessere
Trainingssignal. Die LAW/DriveFuture-Analogie traegt hier NICHT: dort wird die
Downstream-Aufgabe MITGELERNT, hier ist der Decoder fixiert.

**H2 (Kapazitaet/Tiefe): doppelte Tiefe bringt nichts.** 8 Layer (9.2M Params,
~1.5x Trainingszeit) landen mit 0.6927 exakt im Rauschband der 6M-Baseline.
Die 0.69-Decke ist im getesteten Rahmen NICHT tiefenbedingt.
EINSCHRAENKUNG: die Breiten-Variante (d_model 384) ist NICHT testbar ohne
Architektur-Umbau -- d_model ist an die Latent-Kanalzahl 256 gekoppelt (die
Zell-Vektoren gehen ohne Eingangs-Projektion als Tokens ins Modell; Absturz
mat1/mat2 24576x256 vs 384x384). Fuer einen echten Breiten-Test braeuchte es
eine Linear-Projektion 256->d_model->256 (Future Work; Erwartung nach dem
Tiefen-Ergebnis: flach).

**H3 (Residual): hartes Residual = gelerntes Gate.** output_mode=residual
(pred = letzter Frame + Head-Delta, Gate umgangen) erreicht 0.6936 (n.s.).
Das Gate ist also AEQUIVALENT aber nicht notwendig -- konsistent mit
ResWorld/DeltaWorld (Residual-Struktur genuegt), und ein schoenes
Vereinfachungs-Argument: die Architektur haette auch ohne Gate funktioniert.

## GESAMTFAZIT: die 0.69-Decke steht jetzt auf VIER Beinen

| Richtung          | Task  | Ergebnis                      |
|-------------------|-------|-------------------------------|
| Loss-Engineering  | 16b   | Minimal-Loss = Baseline       |
| Konditionierung   | 16d   | Ego bestenfalls neutral       |
| Kontext/Averaging | 16e   | 4 Frames, EMA flach           |
| Task-Kopplung     | 16f   | Task-Loss schadet             |
| Kapazitaet (Tiefe)| 16f   | 8 Layer flach                 |
| Output-Struktur   | 16f   | Residual = Gate (aequivalent) |

Der deterministische mIoU ~0.69 ist damit die abschliessend belegte Decke
dieses Setups (frozen BEVFusion-Latents, deterministische Regression, k=1).
Weitere mIoU-Gewinne erfordern einen Paradigmenwechsel (Objective: generativ/
masked/diskret -> Task 18; oder reichere Features -> per Design fixiert).

## Artefakte

- JSONs: predictions/task16f/{tl_01,tl_05,tl_10,cap_l8,residual}_fullval.json
- Figur: visualizations/16f_miou_lever2.{png,pdf} (render_16f.py)
- Code: forward_logits_grad in Code/seg_decoder_torch.py (Gradientenfluss durch
  frozen Decoder); Task-Loss-Zweig + lambda_task/task_loss_every + --init_weights
  in train_linux.py; output_mode (gated|residual) in Code/config.py +
  Code/bev_world_model.py. Alles Default-neutral (lambda_task=0 / gated =
  bit-identischer Altpfad, lokal verifiziert inkl. Persistenz-Gate 0.0e+00).
- Configs: config_taskloss_{01,05,10}.yaml, config_cap_l8.yaml,
  config_cap_d384.yaml (defekt, s.o.), config_residual.yaml.
- sbatch: sbatch_16f_taskloss.sh, sbatch_16f_cap.sh (beide plain).

## Lektionen

- d_model ist KEIN freier Regler: Token-Dim = Latent-Kanalzahl (256), jede
  Breiten-Aenderung braucht Eingangs-/Ausgangs-Projektionen.
- Smoke-Tests muessen den FORWARD mit realistischen Shapes pruefen, nicht nur
  den Modellbau (der d384-Absturz waere lokal abfangbar gewesen).
