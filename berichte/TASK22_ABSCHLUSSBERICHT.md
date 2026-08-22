# Task 22 — Selten-Klassen- & Dynamik-Hebel (Abschlussbericht, 02.08.)

Datum: 2026-07-31 bis 08-02. Ausloeser: Betreuer-Vorschlag (CE/Weighted-CE/
Focal bzw. Daten-Balancierung gegen den Selten-Klassen-Einbruch) + Nutzer-Idee
Dynamik-Balancierung. Rahmen: Task 21 hatte den adressierbaren Headroom auf
~0.040 GT-mIoU beziffert (Rest = Decoder-Decke); Task-Loss durch den frozen
Decoder war durch 16f-H1 bereits entkraeftet -> getestet wurden LOSS-GEWICHTUNG
und SAMPLING.

## Varianten (alle auf 16b.9-Minimal-Basis, Sweep-Protokoll, Seed 42)

- rare_w{2,4,8}: SmoothL1-Zellgewicht 1+(w-1)*rare_mask (stop_line/
  ped_crossing/divider aus nuScenes-Map-GT, 128x128 im Latent-Grid;
  Geometrie-Gate pixelidentisch zur Pipeline). Val-Loss ungewichtet.
- samp_rare: WeightedRandomSampler ~ (Selten-Klassen-Pixelanteil)^0.5.
- samp_dyn: WeightedRandomSampler ~ (Dynamik-Score/Median)^0.5;
  Score = RMS der Latent-Frame-Differenz (enthaelt Ego- + Objektdynamik).

## Ergebnisse

### Proxy (300 Samples, Anker = 16b.7 smooth_l1 0.6735, Schwelle 0.014)

| Variante | Proxy-mIoU | Delta | std-Ratio (Anker 0.9266) |
|---|---:|---:|---:|
| rare_w2 | 0.6862 | +0.0127 | 0.9873 |
| rare_w8 | 0.6861 | +0.0126 | 0.9975 |
| rare_w4 | 0.6829 | +0.0094 | 0.9911 |
| samp_dyn | 0.6807 | +0.0072 | 0.9902 |
| samp_rare | 0.6785 | +0.0050 | 0.9876 |

Alle 5 ueber dem Anker (einzeln n.s., einheitliches Vorzeichen; rare_w =
reiner Loss-Effekt bei identischer Datenreihenfolge). std-Ratio: ALLE
Varianten schliessen den std-Gap fast vollstaendig, dosisabhaengig-monoton
in w (0.9975 bei w=8) — die Selten-Klassen-Regionen sind die kontraststarken
Strukturen, an denen Mean-Seeking die Varianz frisst.

### Echte GT (Injection, Full-Val 6019; Referenz minimal KONVERGIERT 0.5898)

| Variante | mean | divider | stop_line | ped_cross |
|---|---:|---:|---:|---:|
| minimal (konvergiert, Task 21) | 0.5898 | 0.4761 | 0.4746 | 0.5618 |
| rare_w2 | 0.5850 | 0.4692 | 0.4701 | 0.5587 |
| rare_w8 | 0.5835 | 0.4684 | 0.4699 | 0.5583 |
| samp_dyn | 0.5842 | 0.4719 | 0.4684 | 0.5557 |

### Dynamik-stratifiziert (GT, Quartile der Latent-Dynamik, je ~1436)

| Quartil | minimal | samp_dyn (Delta) | rare_w2 (Delta) |
|---|---:|---:|---:|
| q1 statisch | 0.5755 | -0.0041 | -0.0006 |
| q2 | 0.5762 | -0.0045 | -0.0030 |
| q3 | 0.6159 | -0.0063 | -0.0064 |
| q4 dynamisch | 0.5899 | -0.0082 | -0.0111 |

Nebenbefund: die Quartil-Absolutwerte sind NICHT monoton in der Dynamik
(q3 am hoechsten) — der Roh-Score konfundiert Ego-Tempo mit Kartentyp
(Highways = schnell UND kartenseitig einfach; statische Szenen = Kreuzungen).
Fuer Vergleiche zaehlen nur die Deltas innerhalb eines Quartils.

## Befunde

BEFUND 1 — PROXY-GEWINN UEBERSETZT SICH NICHT AUF GT: Auf echter GT liegen
alle drei geprueften Varianten leicht UNTER der konvergierten minimal-Basis;
kein gezielter Hub bei divider/stop_line/ped_cross (alle per-Klasse-Deltas
leicht negativ, Dosis w2 vs w8 irrelevant). VORBEHALT Protokoll: Varianten
early-gestoppt (Ep. 21-24) vs. Referenz konvergiert — die Groessenordnung
des Defizits (-0.005..-0.006 mean) passt zum reinen Konvergenz-Rueckstand
(Pseudo-GT-Defizit -0.008). Ehrliche Lesart: bestenfalls NEUTRAL auf GT,
sicher KEIN gezielter Selten-Klassen-Gewinn. (Sauber trennbar durch GT-Eval
des early-gestoppten 16b.7-Ankers — optional, ~1h.)

BEFUND 2 — DYNAMIK-SAMPLING HILFT NICHT, AUCH NICHT IN DYNAMISCHEN SZENEN:
samp_dyn ist in JEDEM Quartil unter minimal, und das Defizit WAECHST mit der
Dynamik (q1 -0.004 -> q4 -0.008). Die Nutzer-/Betreuer-Hypothese "mehr
dynamische Trainingsszenen -> besser bei Dynamik" ist damit auf GT-Ebene
entkraeftet (im Rahmen des Protokoll-Vorbehalts).

BEFUND 3 — DER ECHTE EFFEKT DER GEWICHTUNG IST VARIANZ-KALIBRIERUNG:
std-Ratio 0.9266 -> 0.9873..0.9975 (monoton in w) bei praktisch
unveraendertem GT-mIoU. Die zellgewichtete Loss ist damit der staerkste
bekannte std-Gap-Hebel des Projekts (staerker als der SmoothL1-Wechsel
16b.7), wirkt aber statistisch, nicht semantisch. (Messprotokoll-Hinweis:
Anker-std-Ratio aus Cluster-, Varianten aus lokaler Inference — gleiches
300er-Subset/Seed, deterministisch.)

BEFUND 4 (META) — DIE PSEUDO-GT-METRIK KANN AM RAND IRREFUEHREN: +0.013
Proxy bei <=0 GT zeigt: Naehe zu decode(real t+1) ist nicht dasselbe wie
Naehe zur Realitaet. Die GT-Verankerung (Task 21) war methodisch notwendig;
Sweep-Gewinne unterhalb der Schwelle sind kuenftig IMMER gegen GT zu pruefen.

FAZIT: Die Betreuer-Vorschlaege sind vollstaendig und ehrlich beantwortet:
unter einem frozen Decoder heben weder Klassen-Gewichtung noch Balancierung
die Selten-Klassen-Leistung — konsistent mit Task 21 (Headroom sitzt in der
Wahrnehmung). Als Empfehlung bleibt SmoothL1-Minimal; die zellgewichtete
Loss (w=2) ist eine dokumentierte OPTION, wenn Varianz-Kalibrierung ohne
Kalibrierungs-Nachlauf gewuenscht ist. Der mIoU-Strang bleibt geschlossen.

## Pannen/Lektionen dieser Task

- 5 parallele Trainings @500G-cgroup: Kernel-OOM killte 4 Prozesse (SLURM:
  "4 oom_kill events"), 3 Nachbarn deadlockten auf tote Loader-Worker,
  76 min unentdeckt -> Freeze-Detektor (Log-mtime) in die Monitoring-Crons
  aufgenommen; Neustart als 2x2-GPU-Jobs (16e-Muster) lief fehlerfrei.
- In-Job-Inference ohne --output: alle Varianten ueberschrieben dasselbe
  _base/inference_log.json -> std-Ratios lokal nachgemessen; kuenftig
  --output je Lauf setzen.
- Quartil-PKLs aus fremdem Quell-PKL erbten falsche absolute Datenpfade
  (/dataset/...) -> IMMER aus dem container-konsistenten PKL bauen; die
  Sekunden-schnellen "Erfolge" flogen erst durch den Ergebnis-Check auf.

## Artefakte

- predictions/task22/: gt_eval22.json, dynamics_{train,val}.json,
  inf_<label>/inference_log.json (std-Ratios), Quartil-PKLs (det_latents_out).
- BeeGFS: gt_masks_seg/{train,val}.npz, dumps_task22/dump_22_*.
- Code: mk_gt_masks_seg.py (+Gate), compute_dynamics_scores.py,
  mk_quartile_pkls.py, run_22_gt_evals.sh, run_22_quartile_evals.sh,
  Task-22-Zweige in bev_dataset/bev_dataloader/train_linux (Default AUS).
- Configs: config_22_*.yaml, sbatch_22_sweep[_a|_b].sh, sbatch_22_dump.sh.
- Checkpoints: task22/<label>/phase2/ (Cluster + lokal best_miou.pt).
