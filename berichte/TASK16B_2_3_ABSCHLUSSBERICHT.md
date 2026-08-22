# TASK 16b.2-3 — Paralleler Sweep-Job + Auswerte-Kette (Abschlussbericht)

**Datum:** 2026-07-13
**Chat:** Task 16b.2-3 (Cluster-Operationalisierung, Hauptgleis)
**Status:** ABGESCHLOSSEN (Code deployed + verifiziert; Live-Feuerprobe = 16b.4)
**Baut auf:** `TASK16B_1_ABSCHLUSSBERICHT.md` (Basis-Config), `NOTIZ_16B_LOADER_DIAGNOSE.md` (job_shared-Nachweis, Job 122812)

## Ziel

Den parallelen Sweep-Job (16b.2) und die Auswerte-Kette (16b.3)
operationalisieren: EIN Sweep = EIN Mehr-GPU-Job (GPU-Zahl = Werteanzahl),
EIN geteiltes `/dev/shm`-Staging, danach Inference + CSV je Wert. Ergebnis ist
ein wiederverwendbares sbatch-Template fuer alle Regler-Sweeps (16b.4-16b.7).

## Ausgangslage

16b.1 lieferte `config_sweep_base_cluster_fp16.yaml`. Job 122812 hatte die
`job_shared`-Mechanik bereits bewiesen (6x Skip-Copy, Konkurrenz <=5%, 2.85x
Durchsatz), nutzte zum Vorstagen aber einen **Inline-Heredoc-Hack** und lief mit
Wegwerf-Configs. Fuer den produktiven Sweep fehlten vier Bausteine:
(a) ein benannter Pre-Stage-Einstieg (`shm_staging.py` hatte nur `--cleanup`),
(b) ein generischer Sweep-sbatch, (c) die Config-Vorab-Generierung mit
diff-Verifikation, (d) die Vermeidung der ~10 GB npy-Dumps pro Inference-Wert.

## Vorgehen / Aenderungen

**1. `shm_staging.py` — `--stage`-Modus (Patch).** Standalone-Einstieg
symmetrisch zu `--cleanup`, XOR-Guard (genau eine Aktion pro Aufruf). Ruft
`stage_and_rewrite(cfg)` und verwirft das umgebogene cfg (jeder
Trainingsprozess biegt seine eigene In-Memory-Config selbst um). Ersetzt den
122812-Heredoc-Hack durch einen ordentlichen, diffbaren Tuergriff.
Default-neutral: aendert kein bestehendes Verhalten.

**2. `mk_sweep_cfgs.py` — Config-Generator (neu).** Verallgemeinert
`mk_lambda_cfg.py` (nur `lambda_std`, ein Wert) auf jeden Regler + Werteliste.
Patcht per `patch_line` (Exact-1-Match-Guard) GENAU ZWEI Zeilen je Config
(`<regler>` + `checkpoints.dir`), fuehrt NICHTS aus. Diese Trennung erlaubt die
vom Arbeitsplan geforderte diff-Verifikation VOR dem Compute. Konvention:
`config_<regler>_<L>.yaml`, `checkpoints.dir = <sweep-root>/<regler>_<L>`.

**3. `inference.py` — `--no-save` (Patch).** Gated ausschliesslich die zwei
`np.save`-Aufrufe. `pred_std`/`real_std` (fuer std-Ratio) stammen aus
`compute_metrics`, NICHT aus den npy -> der CSV-Harvest bleibt intakt,
`inference_log.json` unveraendert. Default-neutral (ohne Flag byte-identisch).
Spart ~10 GB/Wert.

**4. `harvest_sweep.py` — CSV-Ernte (neu, 16b.3c).** `harvest()` 1:1 aus
`sweep_runner.py` -> die Cluster-CSV ist spaltenidentisch zu den lokalen
16a-Sweeps (direkt vergleichbar). Liest `run_summary.json` +
`inference_log.json` (+ optional `fullval.json`) je Wert. Auto-Discover der
Werte aus vorhandenen Ordnern moeglich. Kein Compute. Laeuft als Jobende-Schritt
ODER lokal nach `rsync`.

**5. `sbatch_sweep_parallel.sh` — Sweep-Job (neu, 16b.2).** Abgeleitet aus
`sbatch_parallel_test.sh`: uebernimmt den bewaehrten conda-Block (Kandidaten-Loop
+ `import torch`-Check, VOR `set -u`), den House-Style mit explizitem
`|| { exit 1; }` (kein `set -e`) und den trap. Kein Partition/Account (wie
122812). Ablauf: Configs erzeugen -> `--stage` (einmal sequenziell) ->
N Trainings parallel (`CUDA_VISIBLE_DEVICES=0..N-1`) -> `wait` mit
Fehlschlag-Sammlung -> Inference `--no-save` -> Harvest. Editierbar sind nur
vier Zeilen (job-name, `--gres`, `REGLER`, `VALUES`). Zwei Deploy-Guards
pruefen, dass `Code/shm_staging.py` `job_shared` UND `--stage` traegt.

**Struktur-Konvention.** `shm_staging.py` ist ein Modul und liegt in `Code/`;
`inference.py`, `mk_*`, `harvest_*`, `sbatch_*` sind top-level. Der
Standalone-Aufruf `python Code/shm_staging.py --stage` ist zulaessig, weil das
Modul nur `yaml` + stdlib importiert (keine Geschwister-Module).

**Behobener Deploy-Fehler.** Der Projekt-Snapshot trug in `inference.py`
`Path(__file__).parent / "code"` (klein) statt `"Code"`; beim Deploy wurde das
kurz reintroduced und per `sed` auf beiden Maschinen korrigiert (`parse ok`).

## Verifikation (2026-07-13, Cluster `tas-dgx-head`)

- **Config-Generierung:** 5 Configs erzeugt; `diff` Basis vs.
  `config_lambda_grad_0.1.yaml` = GENAU 2 geaenderte Zeilen (`lambda_grad` +
  `dir`), Rest byte-identisch. Whitespace-Alignment der gepatchten Zeile
  normalisiert (kosmetisch, YAML-neutral, Inline-Kommentar erhalten) — exakt das
  Verhalten des bewaehrten `mk_lambda_cfg`.
- **sbatch:** `bash -n` ok.
- **Deploy-Guards:** `job_shared` UND `--stage` in `Code/shm_staging.py`
  vorhanden.
- **Syntax:** `py_compile` aller vier Python-Dateien ok.
- **`job_shared`-Mechanik selbst:** bereits in Job 122812 bewiesen (6x
  Skip-Copy, `fail=0`, Loss identisch, 2.85x) — die Grundlage ist am echten
  Cluster belegt, nicht nur spezifiziert.

**Bewusst NOCH NICHT gefahren (= 16b.4):** der erste echte GPU-Job. Der
`lambda_grad=0`-Nullpunkt uebernimmt DREI Rollen (Sweep-Anker,
Stufe-1-Verifikation, Hardware-Bruecke lokal->Cluster) und validiert die
gesamte Kette live. Kein Compute in 16b.2-3 committed.

## Ressourcen-Startpunkt (16b.4)

`--gres=gpu:a100:5`, `--cpus-per-task=96` (5x~19; bei exklusivem Knoten 120 =
5x24 sauberer), `--mem=600G`, `--time=06:00:00`.

## TF32-Entscheidung

Default **AUS** (Konservativ-Pfad). `torch.set_float32_matmul_precision` wird
NICHT gesetzt -> Numerik unveraendert gegenueber der lokalen 15.x-Historie, der
`lambda_grad=0`-Nullpunkt ist damit direkt an den lokalen Arbeitspunkt
anschlussfaehig. Falls spaeter `"high"` gewuenscht: VOR dem 16b.4-Nullpunkt
setzen und dokumentieren; das Bruecken-Kriterium (15.2-Band) prueft die
Vertraeglichkeit dann mit.

## Offen / uebergeben an 16b.4

- `mkdir -p ~/Code_final/logs` auf dem Cluster; Job IMMER aus `~/Code_final`
  submitten (`--output=logs/...` wird vor dem `cd` aufgeloest).
- Ersten Job absetzen (`sbatch sbatch_sweep_parallel.sh`).
- Nullpunkt-Kriterium: mIoU ~0.68 +/- 0.014, std-Ratio ~0.915 +/- 0.009
  (15.2-Schwellen). AUSSERHALB DES BANDES: STOPP, Ursache klaeren, ggf.
  Fallback 16a — keine weiteren Werte interpretieren.
- Skip-Kontrolle im Log: bei vorgezogenem `--stage` muessen ALLE Laeufe
  skippen (Grep-Zeile im sbatch am Trainings-Ende).

## Artefakte

- `sbatch_sweep_parallel.sh` (neu, top-level)
- `mk_sweep_cfgs.py` (neu, top-level)
- `harvest_sweep.py` (neu, top-level)
- `Code/shm_staging.py` (Patch: `--stage`-Modus)
- `inference.py` (Patch: `--no-save`; `"code"`->`"Code"` korrigiert)
- `configs/task16b/lambda_grad_sweep/config_lambda_grad_{0,0.05,0.1,0.3,1.0}.yaml`
  (Dry-Run, diff-verifiziert)
- Dieser Bericht: `TASK16B_ABSCHLUSSBERICHT.md`
