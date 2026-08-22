# TASK 16b.5 — Sweep lambda_ssim + Same-Seed-Replikat des 16b.4-Ankers (Abschlussbericht)

**Datum:** 2026-07-14 bis 2026-07-15
**Chat:** Task 16b.5 (Cluster-Sweep lambda_ssim, Hauptgleis)
**Status:** ABGESCHLOSSEN (Job 123043 COMPLETED, ExitCode 0; alle 3 Werte
trainiert + evaluiert, CSV geerntet)
**Baut auf:** `TASK16B_4_ABSCHLUSSBERICHT.md` (Wellen-Modus + Resume-Patch,
lambda_grad-Sweep), `TASK16B_2_3_ABSCHLUSSBERICHT.md` (Sweep-Apparat)

## Ziel

Zweiter Regler-Sweep der 16b-Kette: `lambda_ssim in {0, 0.1, 0.3}`,
`lambda_std=1.0` fix, Rest Baseline. Zweck laut Arbeitsplan: **Redundanz-Check
gegen `lambda_grad`** — beide sind Struktur-Regler, und nach dem 16b.4-Befund
(kein belastbarer mIoU-Gewinn durch lambda_grad) ist die Frage, ob SSIM
leistet, was der Gradienten-Term nicht geliefert hat.

## Ausgangslage / Design-Besonderheit

Die Sweep-Basis-Config (`config_sweep_base_cluster_fp16.yaml`) traegt SSIM
bereits in der Baseline: `lambda_ssim: 0.1`. Der Sweep-Wert `0.1` erzeugt damit
eine Config, die **semantisch byte-identisch** zum 16b.4-Nullpunkt
(`lambda_grad=0`, Seed 42, dieselben sechs Lambdas) ist — nur das
checkpoints.dir unterscheidet sich.

Diese Kollision wurde bewusst als **Same-Seed-Replikat** genutzt statt sie zu
vermeiden: da drei Werte in derselben einen Welle exakt so lange laufen wie
zwei, kostet der Replikatwert keine zusaetzliche Wall-Clock. Er liefert
erstmals eine direkte Schaetzung der Hardware-/Nichtdeterminismus-Streuung
(bisher nur die n=3-Seed-Streuung aus 15.2 bekannt). Erwartung vor dem Lauf:
mIoU ~0.6731, std-Ratio ~0.9085 (die 16b.4-Ankerwerte).

## Ablauf

### 1. Vorbereitung (lokal, verifiziert vor Compute)

- Configs erzeugt: `mk_sweep_cfgs.py --regler lambda_ssim --values 0 0.1 0.3`
  aus der Cluster-Basis. Diff-Kontrolle: jede Per-Lambda-Config unterscheidet
  sich von der Basis in **genau zwei Zeilen** (Regler + checkpoints.dir).
- `sbatch_sweep_parallel.sh` gepatcht (vier `<<< EDIT`-Zeilen +
  Ressourcen): `--job-name=sweep_lambda_ssim`, `--gres=gpu:a100:3`
  (3 Werte -> 1 Welle), `--cpus-per-task=66`, `--mem=500G`,
  `--time=13:00:00`, `REGLER=lambda_ssim`, `VALUES=(0 0.1 0.3)`.
  Verifikation: `bash -n` (Syntax sauber).
- **Zeitbudget aus GEMESSENEN Zeiten** (Lektion 16b.4): 16b.4/Welle 1 lief
  6h21min fuer 23 Epochen = ~995 s/Epoche unter 4-Wege-Konkurrenz. Die 650 s
  aus der Loader-Diagnose gelten nur fuer Einzellaeufe -> nicht verwendet.
  Budget: 35 Ep. (Obergrenze) x 1000 s + Staging + 3x Inference/Harvest,
  +20 % -> 13 h.
- **Rollback-Guard vor rsync-Deploy:** `md5sum` von `train_linux.py`,
  `checkpointing.py`, `inference.py`, `harvest_sweep.py`, `shm_staging.py`
  lokal vs. Cluster verglichen -> bitgleich. Der 16b.4b-Resume-Patch war auf
  beiden Seiten vorhanden; der Deploy konnte ihn nicht zurueckrollen.

### 2. Job 123043 — 3-Werte-Sweep in 1 Welle (a100-3, 3 GPUs)

Knoten-Check vor Submit: a100-3 mit 5 freien GPUs, a100-4 mit 7 -> sofort
schedulbar. Submit aus `~/Code_final`. Job startete unmittelbar auf a100-3.

**Ergebnis:** COMPLETED, ExitCode 0:0, Elapsed **4h53min36s** — deutlich unter
dem 13h-Budget, weil a100-3 kaum Konkurrenz trug (im Gegensatz zur 4-Wege-Last
in 16b.4). Staging (286 GB -> /dev/shm, 1081 GB frei), Training aller drei
Werte parallel, serielle Inference (300 Val-Samples, `--no-save`), CSV-Harvest
und /dev/shm-Cleanup liefen vollstaendig durch. Kein Zeitlimit-Problem, kein
Resume noetig.

### 3. SLURM-PATH-Fund (nebenbei, in CLAUDE.md nachgezogen)

Die nicht-interaktive SSH-Shell (`ssh head 'befehl'`) hat SLURM **nicht** im
PATH (`squeue: command not found`); die Binaries liegen unter `/opt/slurm/bin`.
Loesung: Login-Shell erzwingen — `ssh head 'bash -lc "squeue -u vima"'`. In
CLAUDE.md unter den SLURM-Gotchas ergaenzt.

## Ergebnisse

| lambda_ssim | mIoU   | std-Ratio | pred_std | mse     | cossim | Stop-Epoche | Stop-Grund   |
|-------------|--------|-----------|----------|---------|--------|-------------|--------------|
| 0           | 0.6867 | 0.9046    | 0.2396   | 0.02826 | 0.8554 | 17          | miou_plateau |
| 0.1         | 0.6732 | 0.9086    | 0.2406   | 0.02866 | 0.8538 | 17          | miou_plateau |
| 0.3         | 0.6704 | 0.9035    | 0.2393   | 0.02981 | 0.8478 | 23          | miou_plateau |

(real_std = 0.2649 fuer alle; CSV: `predictions/task16b/sweep_lambda_ssim.csv`)

### Replikat-Validierung (der methodische Kern)

`lambda_ssim=0.1` (config-identisch zum 16b.4-Anker) reproduzierte den Anker
**auf 0.0001 in beiden Metriken**:

| Metrik    | 16b.4-Anker (lambda_grad=0) | 16b.5-Replikat (lambda_ssim=0.1) | Delta   |
|-----------|-----------------------------|----------------------------------|---------|
| mIoU      | 0.6731                      | 0.6732                           | +0.0001 |
| std-Ratio | 0.9085                      | 0.9086                           | +0.0001 |

**Interpretation:** Die Hardware-/Nichtdeterminismus-Streuung ist praktisch
null — um Groessenordnungen kleiner als die Seed-Schwelle 0.014 (15.2). Zwei
Folgerungen:
1. Same-Seed-Laeufe sind faktisch deterministisch. Die 0.014-Schwelle misst
   damit reine **Seed-Varianz**, nicht Mess-/Hardware-Rauschen.
2. Die n=1-Punktschaetzer pro Sweep-Wert sind fuer den jeweiligen Seed
   verlaesslich; die verbleibende Unsicherheit ist ausschliesslich die
   Seed-Streuung. Cross-Value-Deltas unter 0.014 bleiben damit n.s.

### Kurvenform (Vorbehalt: n=1 Seed/Wert, Schwelle fuer n=3 kalibriert)

Deltas gegen den Nullpunkt `lambda_ssim=0`:

| lambda_ssim | delta mIoU vs. 0 | delta std-Ratio vs. 0 |
|-------------|------------------|------------------------|
| 0.1         | -0.0135 (n.s., knapp unter Schwelle) | +0.0040 (n.s.) |
| 0.3         | **-0.0163 (signifikant, knapp)**     | -0.0011 (n.s.) |

- **mIoU sinkt monoton** mit steigendem lambda_ssim (0.6867 -> 0.6732 ->
  0.6704). Bei 0.3 ueberschreitet der Ruecklauf knapp die Signifikanzschwelle:
  SSIM **verschlechtert** die Kernmetrik. Bester Wert ist `lambda_ssim=0`
  (SSIM aus).
- **std-Ratio ist flach** — alle drei innerhalb von 0.005, kein Effekt. SSIM
  ist damit auch kein Hebel gegen den std-Gap.
- **Interne Konsistenz-Probe:** das +0.0135 von 0.1->0 (SSIM abschalten) deckt
  sich fast exakt mit dem Sprung von der 16b.4-Baseline (0.6731, SSIM=0.1) auf
  hier-0 (0.6867) — zweimal derselbe Effekt "SSIM aus", ueber zwei Jobs.

### Redundanz-Check gegen lambda_grad (Sweep-Ziel)

Beide Struktur-Regler liefern **keinen** mIoU-Gewinn:
- `lambda_grad` (16b.4): bestenfalls neutral (n.s. Aufwaertstrend bis 0.3),
  aber **signifikanter std-Ratio-Nachteil** bei hoeheren Werten.
- `lambda_ssim` (16b.5): **schadet** der mIoU leicht und monoton (signifikant
  bei 0.3), **kein** std-Ratio-Effekt.

**Einordnung fuer den finalen Loss:** Die Daten sprechen dafuer, **SSIM auf 0
zu setzen** (Regler streichen) und `lambda_grad` klein (0.05-0.1) oder 0 zu
halten. Beides vereinfacht den Loss, ohne mIoU zu kosten — im Gegenteil, SSIM=0
lag in diesem Sweep bei mIoU 0.6867, dem hoechsten Proxy-Wert der gesamten
16b-Kette bisher (ueber dem lambda_grad=0.3-Bestwert 0.6858 aus 16b.4).

## Lektionen

1. **Config-Kollisionen als kostenlose Replikate nutzen.** Dass die Baseline
   SSIM=0.1 traegt, machte den Sweep-Wert 0.1 zum Gratis-Replikat des
   16b.4-Ankers. Statt ihn zu streichen (2-GPU-Job) wurde er behalten (3-GPU-
   Job, gleiche Wall-Clock) und lieferte den bisher fehlenden Hardware-
   Rausch-Nachweis. Vor kuenftigen Sweeps pruefen, ob ein geplanter Wert
   zufaellig die Baseline trifft — solche Punkte sind Validierungsgeschenke.
2. **Same-Seed ist deterministisch, die 0.014-Schwelle ist reine Seed-Varianz.**
   Bestaetigt empirisch, dass die Signifikanzschwelle nicht durch Mess- oder
   Hardware-Rauschen aufgeblaeht ist. Fuer belastbare Kurven bleibt ein
   Seed-Sweep (n=3) noetig; ein Hardware-Replikat ersetzt ihn nicht.
3. **SLURM braucht eine Login-Shell ueber ssh.** `ssh head 'squeue'` scheitert
   (leerer PATH); `ssh head 'bash -lc "squeue"'` funktioniert. Dauerhaft in
   CLAUDE.md.
4. **Zeitbudget aus gemessenen Epochenzeiten trug** — 13h-Budget bei realen
   4h53min; kein Kill wie in 16b.4. Konkurrenzlast schwankt aber stark
   (4h53 solo vs. 6h21 fuer eine Welle unter 4-Wege-Last), das Budget muss die
   Worst-Case-Last abdecken, nicht die beobachtete.

## Geaenderte/erzeugte Dateien

- `configs/task16b/lambda_ssim_sweep/config_lambda_ssim_{0,0.1,0.3}.yaml`
- `sbatch_sweep_parallel.sh` — 16b.5-Parameter (job-name/gres/mem/time/REGLER/
  VALUES; Wellen-Modus + Resume-Patch unveraendert aus 16b.4 uebernommen)
- `checkpoints/task16b/lambda_ssim_sweep/lambda_ssim_{...}/phase2/best_miou.pt`
  (auf dem Cluster)
- `predictions/task16b/lambda_ssim_sweep/lambda_ssim_{...}/inference_log.json`
  (auf dem Cluster)
- `predictions/task16b/sweep_lambda_ssim.csv` (lokal geholt)
- `CLAUDE.md` — SLURM-PATH-Gotcha + Statuszeile 16b.5

## Naechster Schritt

Task 16b.6 (`lambda_mean` + `lambda_cos`, flach erwartet). Nutzt denselben
wellenfaehigen `sbatch_sweep_parallel.sh` und `--resume`-Patch ohne weitere
Infrastruktur-Aenderung. Offen: ob beide Regler in einem Job (zwei Sweeps
nacheinander) oder getrennt laufen, und ob wegen der flachen Erwartung ein
verkuerztes Werteraster genuegt.
