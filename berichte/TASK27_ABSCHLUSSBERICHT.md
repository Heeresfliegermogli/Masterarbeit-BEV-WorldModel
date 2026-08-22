# Task 27 — Seed-Absicherung & Zusatzlaeufe (Abschlussbericht, 14.08.)

Umsetzung der externen Experimentier-Vorschlaege (5 Punkte, alles LOKAL).

## 27.1 Seg-Kopfadaptation Multi-Seed (Prio 1) — BESTAETIGT
Kopf-Trainings Seeds 42-45 (WM + Vorhersagen FIX; adapt_seg_head --seed),
je Full-Val-GT-Injection:
| Seed | 42 | 43 | 44 | 45 | Mittel +- Streuung |
|---|---|---|---|---|---|
| GT-mIoU | 0.6012 | 0.6011 | 0.6000 | 0.6011 | 0.6008 +- 0.0005 |
Frozen-Basis 0.5898 -> Gewinn in JEDEM Seed >= +0.0102; Kopf-Seed-Streuung
(0.0005) ist ~20x kleiner als der Effekt. Die Sorge "Effekt < globale
0.014-Schwelle" ist ausgeraeumt: 0.014 misst WM-Trainings-Varianz, die hier
nicht variiert. Thesis-Formulierung: "+0.0110 +- 0.0005 (4 Seeds)".
JSON predictions/task27/seg_head_seeds.json.

## 27.2 Det-Kopfadaptation v2 Multi-Seed (Prio 2) — BESTAETIGT
v2-Trainings Seeds 42-44 (12 Ep. Cosine + Real-Mix + Val-Selektion,
adapt_det_head_v2 --seed; Seed variiert Init/Shuffle/Holdout+Mix):
| Seed | pred mAP/NDS | real mAP/NDS |
|---|---|---|
| 42 | 0.4292/0.5333 | 0.6710/0.7040 |
| 43 | 0.4309/0.5308 | 0.6729/0.7035 |
| 44 | 0.4342/0.5342 | 0.6722/0.7032 |
| Mittel | 0.4314 +- 0.0021 | 0.6720 +- 0.0008 |
pred-Gewinn min. +0.0854 ueber frozen (0.3438) = ~40x Kopf-Seed-Streuung;
Real-Malus max. -0.0148 unter Orakel (0.6858), stabil in jedem Seed.
JSON predictions/task27/det_head_seeds.json.

## 27.3 Flow auf Full-Val (Prio 3) — OHNE COMPUTE ERLEDIGT
Der deterministische Flow-Forward IST der frozen 16b.9-Backbone; Full-Val
per Konstruktion 0.6946. Pareto-Punkt entsprechend gesetzt, Subset-Stern
entfernt (0.6896 war Subset-Effekt desselben Forwards, 300er-Protokoll).

## 27.4 Best-of-K-Kurve Flow (Prio 4) — COVERAGE-ANALYSE
300er-Subset, steps=1, Seed 42 (Einzel-Sample je K neu gesampelt):
| K | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| best-of-K mIoU | 0.6450 | 0.6588 | 0.6697 | 0.6811 | 0.6875 |
Monotone Saettigungskurve: K=16 erreicht ~0.6875 und naehert sich dem
det. Mittelwert-Niveau (0.6896, 300er-Skala); Interpretation NUR als
Stichprobenabdeckung der gelernten Verteilung (Orakel-Auswahl, kein
einsetzbarer Praediktor). JSONs predictions/task27/bestofk_k*.json.

## 27.5 Ressourcenmessung 3x (Prio 5) — ABGESICHERT
Spannweite der Serien-Mittel < 0.5 ms je Variante
(ressourcen_reps_stat.json); Task-26-Nachtrag geschrieben.

## Lektionen
- Archivierte Tools (archiv/tools/) brauchen beim Aufruf aus dem Root
  PYTHONPATH auf Root+Code/ (train_linux-Import) — erster Best-of-K-Lauf
  scheiterte daran; Ergebnis-Checks je Lauf fingen es ab.
- Kopf-Seed-Streuung ist auf beiden Straengen um Groessenordnungen kleiner
  als die WM-Seed-Streuung — Adaptations-Effekte brauchen KEINE 0.014-Skala.

## Artefakte
predictions/task27/{seg_head_seeds,det_head_seeds}.json, bestofk_k*.json;
Checkpoints det_latents_out/adapted*/...s43/s44...; --seed-Flags in
adapt_seg_head.py/adapt_det_head_v2.py; config_27_flow_local.yaml.
