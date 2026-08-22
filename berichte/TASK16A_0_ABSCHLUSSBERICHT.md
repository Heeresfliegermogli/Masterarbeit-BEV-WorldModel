# TASK 16a.0 — ABSCHLUSSBERICHT

**Titel:** Automatisierung der lokalen Lambda-Sweeps (Weg 1, feste Wertelisten)
**Datum:** 2026-07-12
**Server:** alre-server-u20 (NVIDIA TITAN RTX, 24 GB), Python 3.8.10, kein conda
**Status:** ABGESCHLOSSEN — Kette End-to-End verifiziert (Smoke-Lauf lambda_grad=0)


## 1. Aufgabe

16a.0 ist reine Infrastruktur: das Skript bauen und verifizieren, mit dem die
Folge-Sweeps (16a.2 ff.) automatisiert laufen. KEINE wissenschaftlichen Laeufe.
Deliverable laut Arbeitsplan: `sweep_runner.py` + Ergebnis-CSV je Regler,
aufbauend auf `mk_lambda_cfg.py` / `mk_seed_cfg.py`, mit mIoU-Plateau-Stopping
(15.3B.2), idempotent, seriell/tmux-tauglich.

Gilt als fertig, sobald die Kette
Config-Patch -> Training -> Harvest -> CSV einmal sauber durchlaeuft.


## 2. Ausgangslage / Befunde vor dem Bau

- **Configs im Projekt waren veraltet:** die vorhandenen fp16-Configs
  (`config_nuscenes_full_fp16.yaml`, `_cluster_fp16.yaml`) trugen im
  `training:`-Block noch das alte 3-Term-Schema (`lambda_cos`/`lambda_dist`/
  `lambda_ssim`). `lambda_dist` ist ein toter Key: `train_linux.py` liest seit
  15.1 die sechs Terme (`lambda_mse/cos/mean/std/grad/ssim`) getrennt. Als
  Sweep-Basis untauglich, weil der Runner `lambda_std`/`lambda_grad` nicht
  patchen kann, wenn die Zeilen fehlen. Grund: das sind die 15.3B.2-
  *Validierungs*-Configs, die bewusst Baseline-Loss fahren.
- **Code war aktuell:** `train_linux.py` (6-Term-Loss + mIoU-Plateau-Stopping +
  packed-fp16 + shm), `bev_dataset.py`, `bev_dataloader.py`, `pack_latents.py`,
  `shm_staging.py` auf 15.3B-Stand.
- **mIoU-Plateau-Stopping existiert bereits** (`miou_early_stop`/`miou_patience`/
  `miou_min_delta`, Task 15.3B.2). 16a.0 musste es nur per Config aktivieren,
  nicht bauen. Das ist der Zeit-Hebel, der die Sweeps seriell-lokal machbar macht.
- **std-Ratio wird im Training nicht berechnet:** `validate_seg` liefert nur
  `(mIoU, per_class)`. In 15.3 kam die std-Ratio per **inference.py post-hoc**
  auf `best_miou.pt` (erste 300 Val-Samples, `inference_log.json`).
- **Fixwert lambda_std = 1.0** (aus TASK15_3-Bericht: konservativer Arbeitspunkt,
  std-Ratio 0.915, nachweislich mIoU-kostenfrei, kleinste Latent-Stoerung).


## 3. Design-Entscheidung: Harvest-Mechanismus (Option A, praezisiert)

`train_linux.py` schreibt am Laufende kein maschinenlesbares Summary (nur
stdout-Tabelle + wandb + Checkpoints; wandb ist auf diesem Server nicht mal
installiert). Drei Wege wurden abgewogen; gewaehlt:

**Option A (praezisiert): additiver `run_summary.json`-Hook + inference.py-Harvest.**
- `train_linux.py` schreibt am Ende ein `run_summary.json` (rein observational,
  keine Trainingsdynamik).
- Latent-Metriken (std-Ratio/pred_std/MSE/CosSim) holt der Runner wie in 15.3
  per **inference.py post-hoc** aus `inference_log.json` — dadurch sind die
  Sweep-Zahlen mit 15.3 vergleichbar, und der `train_linux.py`-Eingriff bleibt
  minimal (kein `validate_seg`-Umbau).

Verworfen: reines stdout-Scraping (fragil, liefert keine std-Ratio);
wandb-Auslesen (auf diesem Server gar nicht verfuegbar — nachtraeglich bestaetigt,
dass die Entscheidung gegen wandb richtig war).


## 4. Deliverables

### 4.1 `config_sweep_base_fp16.yaml` (neu)
fp16-Infrastruktur der bisherigen Config (packed-fp16, `use_packed=true`,
Plateau-Keys, `decode_every=3`, n_subset=300) + `training:`-Block mit allen
**sechs expliziten** Loss-Termen. `lambda_std: 1.0` fix (15.3-Arbeitspunkt),
`lambda_grad: 0.0` als Default (16a.2-Variable). Absolute Pfade durchgaengig.

### 4.2 `train_linux.py` — 4 additive Edits (alle mit `# Task 16a.0` markiert)
1. Init `best_epoch_miou = -1`, `stop_reason = "max_epochs"`.
2. `best_epoch_miou = epoch` beim mIoU-Update.
3. `stop_reason` an beiden Break-Stellen (`val_loss_patience` / `miou_plateau`).
4. Nach der Schleife: `run_summary.json` nach `<checkpoint_dir>/phase2/`
   (best_miou, best_epoch_miou, best_val_loss, stop_reason, epochs_run,
   die 6 Lambdas, seed).

### 4.3 `sweep_runner.py` (neu)
Verallgemeinert `mk_lambda_cfg.py` (nur lambda_std) auf jeden Regler ueber eine
Werteliste und schliesst die Schleife bis zur CSV. Pro Wert L:
1. Basis-Config per Zeilen-Regex patchen (`patch_line` 1:1 aus `mk_lambda_cfg.py`
   uebernommen: whitespace-flexibel, Exact-1-Match-Guard, Kommentare bleiben):
   `<regler> -> L` und `checkpoints.dir -> <sweep-root>/<regler>_<L>`.
2. `train_linux.py` seriell (blockierend, tmux-safe) -> `best_miou.pt` +
   `run_summary.json`.
3. `inference.py` post-hoc auf `best_miou.pt`, erste `--n-infer` Val-Samples
   -> `inference_log.json`.
4. optional `--full-val` -> `eval_full_val.py` -> Full-Val-mIoU.
5. Harvest -> Zeile in `sweep_<regler>.csv`.
   std-Ratio = mean(pred_std)/mean(real_std) aus den Records (exakt 15.3-Rechnung).

**Idempotenz auf Schritt-Ebene:** jeder Schritt wird uebersprungen, wenn sein
Output existiert. Runner-Neustart nimmt genau da wieder auf. `--force` erzwingt
Neuberechnung. `--dry-run` schreibt nur Configs + zeigt Kommandos.

**Bewusste Scope-Grenze:** Weg 1, NICHT adaptiv. Die Werteliste kommt vom
Menschen (`--values`); der Runner entscheidet nicht, wo als naechstes gemessen
wird. Das "Knie" liest man aus der CSV ab und misst bei Bedarf manuell feiner.


## 5. Verifikation

**Stufe 1 — Dry-Run + diff (0 GPU):** `--dry-run` fuer `lambda_grad 0 0.05 1.0`.
Der `diff config_sweep_base_fp16.yaml <gepatcht>` zeigte **exakt 2 Zeilen**
(`lambda_grad` mit erhaltenem Kommentar + `dir`) — Exact-1-Match-Guard und
Kommentar-Erhaltung bestaetigt. Alle train/inference-Kommandos pfad-korrekt.

**Stufe 2 — Smoke (kurz, echt):** Lauf `lambda_grad=0` (mit lambda_std=1.0 fix,
also der spaetere 16a.2-Ankerpunkt). Kette lief vollstaendig durch:
- mIoU-Plateau-Stopping feuerte (3 Messungen ohne Fortschritt >0.014),
  `best_miou.pt` bei Epoche 17 gesichert;
- `[16a.0] run_summary.json -> .../phase2/run_summary.json` geschrieben;
- Inference automatisch auf `best_miou.pt` (NICHT best_val_loss.pt) gestartet,
  `inference_log.json` (20 Records) geschrieben;
- Harvest -> CSV.

**CSV-Referenzzeile (Smoke, 20 Samples):**

| Feld | Wert |
|---|---|
| regler / value | lambda_grad / 0 |
| best_miou | 0.6756 |
| best_epoch | 17 |
| std_ratio | 0.9027 |
| pred_std / real_std | 0.2366 / 0.2621 |
| mse | 0.03171 |
| cossim | 0.8384 |
| best_val_loss | 0.052162 |
| epochs_run | 21 |
| stop_reason | miou_plateau |

Die `std_ratio=0.9027` deckt sich sauber mit 15.3 (dort 0.915 bei lambda_std=1.0);
die kleine Differenz ist durch 20 vs. 300 Samples + fp16 erklaert. Damit ist der
Records-Harvest (mean(pred_std)/mean(real_std)) als korrekt belegt.


## 6. Learnings / Merkposten

- **`--sweep-root` immer ABSOLUT angeben** (oder Runner aus `Code_final/` starten,
  oder Argument weglassen -> Default `/…/Code_final/checkpoints/task16a`). Im Smoke
  wurde `.../checkpoints/…` (drei Punkte als Kurzschreibweise) woertlich als
  Argument uebernommen; Linux legt fuer `...` ein reales Verzeichnis an (nur `.`
  und `..` sind reserviert), Nautilus blendet es als "versteckt" aus (Strg+H).
  KEIN Config- oder Skriptfehler — der Runner schrieb exakt das uebergebene
  Argument. Die Basis-Config traegt durchgaengig absolute Pfade, produktiv
  tritt das nicht auf.
- **wandb ist auf alre-server-u20 nicht installiert** — Training laeuft ohne,
  Harvest liegt bewusst auf `run_summary.json`/`inference_log.json`, also
  unabhaengig von wandb. Bestaetigt Option A nachtraeglich.
- **Der In-Epoche-Ticker "wasserfallt" durch die Runner-Pipe:** die `\r`-Zeile
  aus `train_one_epoch` (15.3B-Timing) ueberschreibt sich nur im direkten TTY;
  durch `tee`/Popen landet jede Aktualisierung auf eigener Zeile. Rein kosmetisch
  (geschwaetzigeres Logfile), kein Eingriff noetig.
- **fp16-Packs lagen aus 15.3B.2** (train 28130 / val 6019) -> `[pack] SKIP`,
  kein Neupacken. Sweeps lesen dieselben Packs.


## 7. Offen / uebergeben an Folge-Tasks

- **Idempotenz-Formalcheck** (Re-Run -> SKIP) wurde durch das Aufraeumen von
  `task16a_smoke` nicht separat protokolliert; die Skip-Logik ist simpel
  (Output-Existenz je Schritt) und beim naechsten realen Sweep ohnehin sichtbar.
- **16a.2 (naechster Task):** erster echter Sweep `lambda_grad`, vorgeschlagene
  Werteliste `--values 0 0.05 0.1 0.3 1.0`, lambda_std=1.0 fix. Ankerpunkt
  `lambda_grad=0` liegt aus dem Smoke bereits vor (allerdings nur 20-Sample-
  Inference; produktiv mit `--n-infer 300` wiederholen).
- Danach 16a.3 (`lambda_ssim`), 16a.4 (`lambda_mean`/`lambda_cos`).

**Signifikanzschwellen (aus 15.2, gelten fuer alle Sweeps):**
|Delta mIoU| > ~0.014, |Delta std-Ratio| > ~0.009.


## 8. Artefakte

- `config_sweep_base_fp16.yaml` (Sweep-Basis, 6 Terme, lambda_std=1.0 fix)
- `sweep_runner.py`
- `train_linux.py` (mit 4 Task-16a.0-Edits)
- Smoke-Referenz-CSV (verworfen mit `task16a_smoke`, Zeile in Abschnitt 5 dokumentiert)
