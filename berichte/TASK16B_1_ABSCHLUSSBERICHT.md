# TASK 16b.1 — Sweep-Basis-Config Cluster (Abschlussbericht)

**Datum:** 2026-07-13
**Chat:** Task 16b.1 (Cluster-Operationalisierung, Hauptgleis)
**Status:** ABGESCHLOSSEN

## Ziel

`config_sweep_base_cluster_fp16.yaml` erstellen — Blocker fuer den gesamten
16b-Pfad (16b.2 sbatch, 16b.3 Auswerte-Kette, 16b.4ff Sweeps). Muss den
6-Term-Loss-Block (15.1-Schema) mit der bewaehrten Cluster-Umgebung
(BeeGFS, fp16-Packs, /dev/shm-Staging) kombinieren.

## Ausgangslage / Problem

Alle bisherigen Cluster-Configs trugen das ALTE 3-Term-Loss-Schema
(`lambda_cos`/`lambda_dist`/`lambda_ssim`). `train_linux.py` liest Lambdas
per `train_cfg.get(key, default)` — ein fehlender Key faellt STILL auf den
Default zurueck. Ohne Umstellung waere `lambda_std` unbemerkt auf 0.1 statt
dem 15.3-Arbeitspunkt 1.0 gelaufen, und `lambda_grad` waere als Sweep-
Regler gar nicht patchbar gewesen.

## Vorgehen

Config als Kreuzung zweier bestehender Dateien gebaut:
- **Loss-Block** aus `config_sweep_base_fp16.yaml` (16a, lokal): voller
  6-Term-Block, `lambda_std: 1.0` fix, Rest Baseline.
- **Umgebung** aus `config_nuscenes_full_cluster_fp16.yaml` (15.3B):
  BeeGFS-Pfade, `use_packed` + `stage_to_shm`, mIoU-Plateau-Stopping,
  `decode_every: 3`.

Drei gezielte Aenderungen gegenueber dem 15.3B-Cluster-Anker, aus der
16b-Loader-Diagnose (Jobs 122811/122812, `NOTIZ_16B_LOADER_DIAGNOSE.md`):

| Aenderung | Alt | Neu | Begruendung |
|---|---|---|---|
| `shm_cleanup` | `job_local` | `job_shared` | N Sweep-Prozesse teilen EINE /dev/shm-Kopie (memmap read-only) |
| `num_workers` | 8 | 16 | -31 s/Epoche (Diagnose) |
| `pin_memory` | (Default true) | unveraendert | `false` waere +93 s/Epoche gewesen — bewusst NICHT gesetzt |

Zusaetzlich: `checkpoints.dir` + Inference-Pfade auf `task16b/_base`-
Platzhalter umgestellt (isoliert von Task-15-Checkpoints); Cluster-Pfad-
Konvention beachtet (`latents_packed_fp16/{train,val}` ohne `seg/`-Ebene,
anders als lokal).

## Verifikation

1. **YAML-Validitaet + Loss-Werte:** alle 6 Lambdas korrekt, `lambda_std=1.0`,
   `num_workers=16`, `shm_cleanup=job_shared`, `pin_memory` nicht gesetzt.
2. **patch_line-Kompatibilitaet:** `dir` + alle 6 `lambda_*`-Keys kommen
   je GENAU 1x als Top-Level-Key vor (Match-Garantie fuer spaetere
   Per-Lambda-Configs).
3. **Diff gegen `config_nuscenes_full_cluster_fp16.yaml`** (auf dem
   Cluster gefahren): zeigt ausschliesslich die drei tabellierten
   Aenderungen + Loss-Block-Ersatz + Checkpoint/Inference-Pfade. Keine
   unbeabsichtigten Abweichungen.
4. **Empirische Bestaetigung des `job_shared`-Mechanismus** (Job 122812,
   bereits vor 16b.1 gelaufen, hier als Nachweis herangezogen): 1x
   sequenzielles Staging (~286.5 GB), danach 3x Skip-Copy ueber parallele
   Prozesse verifiziert, `fail=0`, Loss ueber alle drei Laeufe identisch
   (0.062950) — die Mechanik, auf der diese Config aufbaut, ist am echten
   Cluster bereits bewiesen, nicht nur spezifiziert.

## Ergebnis

`config_sweep_base_cluster_fp16.yaml` liegt auf dem Cluster unter
`~/Code_final/`, ist patch_line-bereit und freigegeben fuer 16b.2. Kein
Compute committed (reine Config-Erstellung + Diff-Verifikation).

## Offen / uebergeben an 16b.2 (separater Chat)

- `sbatch_sweep_parallel.sh` aus `sbatch_parallel_test.sh` (Template)
  ableiten: Pre-Staging sequenziell -> N Trainings (`CUDA_VISIBLE_DEVICES`)
  -> `wait` -> Inference-Schritt -> trap-Cleanup.
- Ressourcen-Startpunkt (5 Werte, erster Sweep `lambda_grad`):
  `--gres=gpu:a100:5`, `--cpus-per-task=96`, `--mem=600G`,
  `--time=06:00:00`.
- TF32-Entscheidung vor dem 16b.4-Nullpunkt treffen und dokumentieren
  (Default: AUS, Konservativ-Pfad).
- Per-Lambda-Configs aus dieser Basis per `patch_line` erzeugen
  (`mk_lambda_cfg`-Muster), Kontrolle via `diff`.

## Artefakte

- `config_sweep_base_cluster_fp16.yaml` (Cluster: `~/Code_final/`)
- Diese Notiz: `TASK16B_1_ABSCHLUSSBERICHT.md`
