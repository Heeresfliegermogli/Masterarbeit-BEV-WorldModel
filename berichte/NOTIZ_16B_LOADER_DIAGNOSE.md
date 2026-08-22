# NOTIZ 16b — CLUSTER: LOADER-DIAGNOSE + PARALLELITAETS-TEST
(vorgezogen aus 16a.1B-Kontext; Jobs 122811 + 122812)

**Datum:** 2026-07-13 | **Knoten:** A100-SXM4-80GB | **Fragen:** (1) Ist der
pin_memory-Thread der Rest-Engpass hinter data% ~58 nach 1d? (2) Koennen
mehrere Prozesse EINE /dev/shm-Kopie teilen, und was kostet das?

## Ergebnis (je 1 Epoche, decode aus, identischer Knoten)

| Zelle | pin | Worker | Epoche | data% | data_s | compute_s |
|---|---|---:|---:|---:|---:|---:|
| Referenz (Job 122810) | an | 8 | ~695 s | 57-58 | 333 | 240 |
| W16 | an | 16 | 663.6 s | 54.2 | 295 | 250 |
| P0 | aus | 8 | 788.5 s | 45.8 | 310 | 366 |
| P0W16 | aus | 16 | 783.1 s | 45.7 | 306 | 363 |

## Urteil

1. **pin-Thread-These FALSIFIZIERT (zweifach):** (a) ohne pin haette
   data_wait kollabieren muessen — fiel nur 333->310 s; (b) Epoche +93 s,
   weil pageable h2d synchron/langsam wird (compute 240->366 s; +37 ms/Step
   = 268 MB @ ~7 GB/s pageable vs. ~25 GB/s pinned — Rechnung geht auf).
2. **Worker-These ebenfalls raus:** 16 Worker nur -31 s (~4.5%); P0W16 ~ P0.
3. **Metrik-Lehrstueck:** P0 hat das "beste" data% (45.8) und die
   schlechteste Epochenzeit — Wartezeit wandert nur die Messgrenze entlang.
   Es zaehlt die Zeit-Spalte.
4. Rest-Block ~90 ms/Step = Grundkosten collate + multiprocessing-IPC bei
   268-MB-Batches; skaliert weder mit Workern noch entfaellt er ohne pin.
   Nur invasiv angreifbar (Custom-Prefetcher, pinned Ring-Buffer) —
   NICHT empfohlen (Restpotenzial ~1.15x gegen Umbaurisiko in Sweep-Phase).

## Konsequenz fuer die 16b-Cluster-Basis-Config

- pin_memory: AN lassen (Default; Zeile nicht setzen). Der Diagnose-Flag
  (training.pin_memory, Default true) bleibt im Code — dokumentiert und
  reproduzierbar, Verhalten unveraendert.
- num_workers: 16 (+ --cpus-per-task=24 im sbatch) -> ~663 s/Epoche.
- Damit Cluster-Stand: 1370-1580 s (vor 1d) -> ~663 s = **~2.1x**;
  Einzellauf ~= lokal, Haupthebel bleibt PARALLELITAET.
- Ende der billigen Hebel erreicht; Decke: compute ~250 s + Val ~120 s.

## Parallelitaets-Test: 3 GPUs, EIN geteiltes Staging (Job 122812)

Mechanik: shm_cleanup: "job_shared" (neue Registry-Strategie in shm_staging.py,
1 Zeile) -> gleiches Job-Verzeichnis /dev/shm/<user>_<jobid>, aber KEIN
Prozess-Ende-Cleanup (cleanup()-Guard); sbatch-trap raeumt am JOB-Ende.
Staging laeuft einmal SEQUENZIELL vor dem Parallel-Start (Skip-Check nicht
race-sicher bei Kaltstart); die Prozesse treffen dann den Skip. memmap
read-only -> tmpfs-Seiten werden geteilt, EINE RAM-Kopie fuer alle
(3x separat = 858 GB haette nicht in /dev/shm gepasst -> Sharing ist ab
3 Prozessen Pflicht, nicht Kuer).

| 3x identischer W16-Lauf, parallel | Zeit | data% |
|---|---:|---:|
| Solo-Referenz (122811) | 663.6 s | 54.2 |
| GPU0 / GPU1 / GPU2 | 693.1 / 696.7 / 699.6 s | 56.2 / 55.7 / 54.7 |

**Urteil: funktioniert.** 6x Skip-Copy verifiziert, fail=0, Loss identisch
(numerisch neutral). Konkurrenz-Kosten +4.5-5.4% -- OBERGRENZE, da auf dem
Knoten parallel drei fremde Jobs liefen (CPU-/Bandbreiten-Fremdlast, die
die Solo-Referenz evtl. nicht hatte). Wanduhr: 3 Epochen-Aequivalente in
12:32 min (~2.85x Durchsatz). Identischer Seed = identische Zugriffsmuster
= realistisches Sweep-Szenario (dort variiert nur lambda).

Hochrechnung lambda-Sweep: ~15 Epochen bis Plateau x ~695 s ~ 2.9 h/Lauf;
5 Werte parallel (1 Job, 5 GPUs) ~ 3.5 h Wanduhr statt ~15 h seriell.
sbatch_parallel_test.sh dient als Template fuer den Sweep-Job.

## Betriebs-Learnings

- **train_linux.py raeumt sein /dev/shm-Staging am Prozessende SELBST auf**
  (Auto-Cleanup "Ansatz A", vgl. Config-Kommentar) -> in Multi-Lauf-Jobs
  staged JEDER Lauf neu (~420 s extra pro Zelle). Fuer 16b-Mehrfachlaeufe:
  Staging vorziehen oder Cleanup-Verhalten konfigurierbar machen.
  (profile_train.py raeumt dagegen NICHT auf -> dort griff der Skip.)
- Ohne die Matrix waere pin_memory=false als plausible "Optimierung"
  deploybar gewesen -> haette +93 s/Epoche GEKOSTET. Falsifikation vor
  Deployment hat sich in einem einzigen Job bezahlt gemacht.
- Loss ueber alle Zellen identisch (0.062950) -> Flags numerisch neutral.

## STRATEGIE-ENTSCHEID (2026-07-13): Cluster wird HAUPTGLEIS

**Entscheid:** Alle weiteren Sweeps (16a.2, 16a.3, 16a.4) laufen auf dem
Cluster; lokal (TITAN) bleibt als FALLBACK/Debug-Gleis einsatzbereit
(Ein-Code-Pfad, null Zusatzkosten). Begruendung: Einzellauf-Gleichstand
(~663 s vs. ~685 s lokal) + 2.85x Durchsatz bewiesen + TITAN wird frei.
**Rueckfallkriterium:** gibt die Queue laenger keine GPUs her, laeuft der
betreffende Sweep unveraendert lokal (sweep_runner.py existiert weiter).

**Konsequenz Task-Reihenfolge:** 16b war "parallele Wette, blockiert nicht"
— jetzt BLOCKEN die 16b-Kernbausteine 16a.2. Durch die Vorarbeiten (1d-
Deployment, Diagnose, job_shared, Parallel-Template) ist ~die Haelfte von
16b erledigt. Restbausteine (Reihenfolge = Abarbeitung im 16b-Chat):

1. **config_sweep_base_cluster_fp16.yaml** — Sweep-Basis mit SECHS-Term-
   Loss-Block (lambda_std: 1.0 fix, Rest Baseline; wie lokale Sweep-Basis),
   BeeGFS-Pfaden, use_packed/stage_to_shm, shm_cleanup: job_shared,
   num_workers: 16, eigenem checkpoint-dir-Schema. ACHTUNG: alle bisherigen
   Cluster-Configs tragen das 3-Term-Schema -> lambda_grad dort NICHT setzbar.
   Die lambda-Configs entstehen daraus per patch_line (mk_lambda_cfg-Muster).
2. **sbatch_sweep_parallel.sh** — aus sbatch_parallel_test.sh (Template):
   Pre-Staging + N Prozesse (CUDA_VISIBLE_DEVICES) + wait + trap.
   lambda_grad-Sweep hat GENAU 5 Werte (0/0.05/0.1/0.3/1.0) = EIN
   5-GPU-Job = ein kompletter Sweep. Ressourcen-Vorschlag: gres a100:5,
   cpus-per-task=96 (5x17), mem=600G (286 shm + 5x Worker-Puffer),
   time=06:00:00 (Plateau-Stopping variabel, ~15 Ep. x 695 s + Puffer).
3. **Auswerte-Kette:** Empfehlung: inference.py DIREKT IM JOB nach den
   Trainings (liest rohe .npy von BeeGFS, ~300 Samples = wenige Minuten;
   VORHER den aus 16a.1 vorgemerkten --no-save-Patch einbauen, sonst
   ~10 GB npy-Dumps pro Wert ins Home). run_summary.json +
   inference_log.json liegen dann je Wert im Cluster-Home; CSV-Harvest
   (sweep_runner-Logik) lokal nach rsync ODER als Mini-Schritt am Job-Ende.
   Full-Val des Siegers (eval_full_val.py) laeuft jetzt ebenfalls auf dem
   Cluster (dataloader-basiert, packed+staging).

**Anker-/Vergleichbarkeits-Vermerk:** Der lambda_grad=0-Nullpunkt (erster
Cluster-Sweep-Lauf) uebernimmt DREI Rollen: Sweep-Anker, Stufe-1-
Verifikation (16a.1B §8) UND Hardware-Bruecke lokal->Cluster. Kriterium:
mIoU im 15.2-Band um den lokalen Arbeitspunkt (~0.68 +/- 0.014, std-Ratio
~0.915 +/- 0.009). Faellt er ins Band, sind alle Sweep-Deltas cluster-intern
konsistent UND an die lokale 15.x-Historie angeschlossen; faellt er raus,
STOPP und Ursache klaeren, bevor weitere Werte laufen.

## Artefakte

- Configs: config_diag_{W16,P0,P0W16}.yaml + config_par_gpu{0,1,2}.yaml
  (Wegwerf), Job-Logs logs/loader_diag_122811.out + logs/par3_122812.out
  (+ par_122812_gpu{0,1,2}.log), sbatch_loader_diag.sh,
  sbatch_parallel_test.sh (= Template fuer den parallelen Sweep-Job)
- Code: pin_memory-Flag in Code/bev_dataloader.py + train_linux.py;
  job_shared-Strategie in Code/shm_staging.py (alles Default-neutral,
  lokal wie Cluster eingespielt, Ein-Code-Pfad intakt)
- checkpoints/task16b_diag_* und task16b_par_gpu* -> loeschbar
