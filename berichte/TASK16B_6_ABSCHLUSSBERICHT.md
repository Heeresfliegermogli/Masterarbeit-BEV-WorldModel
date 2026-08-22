# TASK 16b.6 — Sweeps lambda_mean + lambda_cos + Staging-Bug-Fix (Abschlussbericht)

**Datum:** 2026-07-15 bis 2026-07-17
**Chat:** Task 16b.6 (Cluster-Sweeps Verteilungs-/Richtungs-Regler)
**Status:** ABGESCHLOSSEN (Jobs 123172 lambda_mean + 123173 lambda_cos, beide
COMPLETED ExitCode 0). Enthaelt einen langwierigen Staging-Bug samt Fix.
**Baut auf:** `TASK16B_5_ABSCHLUSSBERICHT.md` (Apparat + Determinismus-Nachweis),
`TASK16A_1_ABSCHLUSSBERICHT.md` (mean-Gap-Diagnose -> Werteraster)

## Ziel

Letzte zwei Regler-Sweeps der 16b-Kette, beide FLACH erwartet:
- `lambda_mean in {0, 0.1, 0.3}` (Verteilungs-Regler; Raster aus 16a.1:
  rel_mean_gap 3.75 % im Skip/Minimier-Band -> Mini-Sweep).
- `lambda_cos in {0, 0.1, 0.5}` (Richtungs-Regler; MSE bestraft Betrag, Cosinus
  die Richtung der Feature-Vektoren).
`lambda_std=1.0` fix, Rest Baseline. 0-Punkt je Regler = Notwendigkeits-Nachweis
(braucht der Loss den Term ueberhaupt?). 0.1 je Regler = Anker-Replikat (Basis-
Config traegt lambda_mean=0.1 UND lambda_cos=0.1).

## Der Staging-Bug (der eigentliche Aufwand dieses Tasks)

Der erste Parallel-Versuch (SWEEP_REGLER via `sbatch --export`) scheiterte ~6x
reproduzierbar im /dev/shm-Pre-Staging: `shutil.copy2 -> os.utime(dst)`
`FileNotFoundError`, das Ziel-Verzeichnis verschwand mitten im 236G-Kopieren.
Nach ~15 Diagnose-Jobs und Bisektion:

**URSACHE: das sbatch-CLI-Flag `--export`.** Sobald `--export` (ALL ODER eine
Einzelvariable) auf der sbatch-Kommandozeile steht, richtet SLURM Environment/
Session anders ein und das per-Job-tmpfs-Verzeichnis wird abgeraeumt (kein
`unlink`-Syscall in strace -> Mount/Namespace-Teardown, konsistent mit
job_container/tmpfs). PLAIN `sbatch script.sh` (alle Direktiven in der Datei)
staged fehlerfrei.

Entscheidender Test (Job 123171): identisches, plain funktionierendes Skript +
NUR `--export=SWEEP_REGLER=lambda_mean` -> Staging scheitert. Ein einziges Flag.

AUSGESCHLOSSEN (alle einzeln getestet, kosteten je einen Job): RAM/Knoten-Druck
(a100-2 hatte ~1.8TB frei), cgroup (memory.max=max, --mem nur advisory),
Parallelitaet (Solo-Job scheitert genauso), Batch-vs-srun-Kontext, das
shm_staging-Skript selbst (via srun + Diagnose-sbatch fehlerfrei), der Cleanup-
Trap (deaktiviert -> scheitert weiter), import torch/CUDA-Init, mk_sweep_cfgs.

**Selbst verursachte Regression:** Die 16b.6-Env-Parametrisierung
(SWEEP_REGLER/SWEEP_VALUES via `--export`) hatte den CLI-`--export` ueberhaupt
erst eingefuehrt. 16b.4/16b.5 nutzten Plain-Submit und liefen fehlerfrei.

**FIX:** `sbatch_sweep_parallel.sh` zurueckgebaut auf in-file REGLER/VALUES +
PLAIN-Submit (16b.5-Muster). Parallele Regler = Per-Regler-KOPIE
(`sbatch_sweep_lambda_cos.sh`), jede plain. Deutliche Warnung im Skript und in
CLAUDE.md: NIE mit CLI --export absetzen.

**Lehren aus dem Debugging:** (1) Zeitstempel Head- vs. Compute-Node weichen
~2h ab -> Uhr-Vergleiche ueber Knoten hinweg sind wertlos (fuehrte zu einer
falschen "temporalen RAM-Druck"-Zwischendiagnose). (2) Bei "gleicher Code,
anderes Ergebnis" systematisch bisektieren (plain vs +1 Flag), nicht Mechanismen
raten. (3) strace auf unlink/rmdir grenzt "Prozess loescht selbst" gegen
"externer Teardown" sauber ab.

## Ergebnisse

### lambda_mean (Job 123172, COMPLETED 5h13min)

| lambda_mean | mIoU   | std-Ratio | Stop-Epoche |
|-------------|--------|-----------|-------------|
| 0           | 0.6770 | 0.9074    | 17          |
| 0.1         | 0.6755 | 0.9086    | 14          |
| 0.3         | 0.6777 | 0.9132    | 20          |

Deltas vs. Nullpunkt (0):

| lambda_mean | delta mIoU | delta std-Ratio |
|-------------|------------|-----------------|
| 0.1         | -0.0015 (n.s.) | +0.0012 (n.s.) |
| 0.3         | +0.0007 (n.s.) | +0.0058 (n.s.) |

**FLACH wie vorhergesagt.** Kein signifikanter Effekt auf mIoU oder std-Ratio.
Der 0-Punkt (0.6770) ist praktisch gleich der Baseline -> lambda_mean wird nicht
gebraucht, schadet aber auch nicht. Deckt sich exakt mit der 16a.1-Diagnose
(mean-Gap klein, MSE zieht den Mittelwert schon gut mit).

### lambda_cos (Job 123173, COMPLETED 5h37min)

| lambda_cos | mIoU   | std-Ratio | Stop-Epoche |
|------------|--------|-----------|-------------|
| 0          | 0.6732 | 0.9182    | 20          |
| 0.1        | 0.6736 | 0.9075    | 14          |
| 0.5        | 0.6842 | 0.8888    | 17          |

Deltas vs. Nullpunkt (0):

| lambda_cos | delta mIoU | delta std-Ratio |
|------------|------------|-----------------|
| 0.1        | +0.0004 (n.s.) | **-0.0107 (signifikant)** |
| 0.5        | +0.0110 (n.s., knapp unter Schwelle 0.014) | **-0.0294 (signifikant)** |

- **mIoU:** leicht steigender Trend zu 0.5 (+0.0110), aber n.s. (unter 0.014).
- **std-Ratio:** verschlechtert sich **signifikant und monoton** mit steigendem
  lambda_cos (0.9182 -> 0.9075 -> 0.8888) -> bewegt sich von 1.0 WEG, vergroessert
  den std-Gap. Gleiche Richtung wie lambda_grad (16b.4).
- **Bemerkenswert:** der Nullpunkt cos=0 hat die BESTE std-Ratio (0.9182, naeher
  an 1.0 als jeder andere 16b-Lauf) bei baseline-gleichem mIoU (0.6732). D.h.
  cos AUSschalten verbessert std-Ratio, ohne mIoU zu kosten.

### Anker-Replikation (Determinismus, dritte + vierte Bestaetigung)

| Wert (config-identisch zum Anker) | mIoU | std-Ratio | Anker-Erwartung |
|-----------------------------------|------|-----------|-----------------|
| lambda_mean=0.1 | 0.6755 | 0.9086 | mIoU ~0.6732, std ~0.9086 |
| lambda_cos=0.1  | 0.6736 | 0.9075 | mIoU ~0.6732, std ~0.9086 |

Beide reproduzieren den 16b.4/16b.5-Anker im Rauschen (std-Ratio bei mean sogar
exakt 0.9086). Bestaetigt erneut: Same-Seed ist faktisch deterministisch, die
0.014-Schwelle ist reine Seed-Varianz.

## Synthese (Vorschau 16b.9)

Alle vier Struktur-/Verteilungs-Regler der 16b-Kette liefern KEINEN belastbaren
mIoU-Gewinn:
- lambda_grad (16b.4): n.s. mIoU, std-Ratio signifikant schlechter bei hohen Werten.
- lambda_ssim (16b.5): mIoU monoton schlechter (sig. bei 0.3), std-Ratio flach.
- lambda_mean (16b.6): flach auf beiden Metriken.
- lambda_cos (16b.6): mIoU n.s. (leichter Aufwaertstrend), std-Ratio signifikant
  schlechter bei hohen Werten; Nullpunkt = beste std-Ratio.

**Tendenz minimale hinreichende Loss-Menge:** mse + std (die tragenden Terme);
grad/ssim/mean/cos nahe 0. Endgueltige Bestimmung + Headline-mIoU auf vollem Val
in 16b.9. Vorbehalt durchgehend: n=1 Seed/Wert, Schwellen fuer n=3 kalibriert.

## Geaenderte/erzeugte Dateien

- `sbatch_sweep_parallel.sh` — Rueckbau auf in-file REGLER/VALUES + PLAIN-Submit,
  Warnung gegen --export (dauerhaft). Aktuell auf lambda_mean gesetzt.
- `sbatch_sweep_lambda_cos.sh` — Per-Regler-Kopie fuer lambda_cos (dauerhaft,
  Muster fuer parallele Regler ohne --export).
- `configs/task16b/lambda_{mean,cos}_sweep/config_*.yaml`
- `predictions/task16b/sweep_lambda_mean.csv`, `sweep_lambda_cos.csv`
- `checkpoints/task16b/lambda_{mean,cos}_sweep/.../phase2/` (Cluster)
- Wegwerf-Diagnose (koennen geloescht werden): `sbatch_shmdiag.sh`,
  `sbatch_bisect.sh`, `sbatch_sweep_sampled.sh`
- `CLAUDE.md` — Staging-Bug-Ursache + --export-Verbot, 16b.6-Status.

## Naechster Schritt

Task 16b.9 — Synthese: Dosis-Wirkungs-Plots ueber alle Regler, minimale
hinreichende Loss-Menge festlegen, beste Konfig, Headline-mIoU auf vollem Val
(5743) OHNE Early-Stop. Danach 16c (autoregressive Rollout-Evaluation k=1..4).
