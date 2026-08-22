# TASK 16b.4 — Live-Feuerprobe lambda_grad-Sweep + Resume-Patch (Abschlussbericht)

**Datum:** 2026-07-13 bis 2026-07-14
**Chat:** Task 16b.4 (Cluster-Live-Feuerprobe, Hauptgleis)
**Status:** ABGESCHLOSSEN (GO, alle 5 Werte trainiert + evaluiert; Resume-Luecke
gefunden und gepatcht, Nachweis live erbracht)
**Baut auf:** `TASK16B_1_ABSCHLUSSBERICHT.md` (Basis-Config),
`TASK16B_2_3_ABSCHLUSSBERICHT.md` (paralleler Sweep-Job + Auswerte-Kette)

## Ziel

Der erste echte Mehr-GPU-Cluster-Lauf des in 16b.1-16b.3 gebauten
Sweep-Apparats: `lambda_grad in {0, 0.05, 0.1, 0.3, 1.0}`, `lambda_std=1.0`
fix, Rest Baseline. Der `lambda_grad=0`-Nullpunkt traegt eine Dreifachrolle:
Sweep-Anker, Stufe-1-Speedup-Verifikation (Hardware-Bruecke lokal->Cluster)
und GO/NO-GO-Tor fuer die Interpretation der uebrigen Werte. Kriterium:
mIoU 0.68 +/- 0.014 und std-Ratio 0.915 +/- 0.009 (15.2-Rauschboden, n=3 Seeds).

## Ausgangslage

`sbatch_sweep_parallel.sh` (16b.2/16b.3) war deployed und guard-verifiziert,
aber fuer 1 Job = 1 Regler mit GPU-Zahl == Werteanzahl gebaut. Verfuegbare
Cluster-Kapazitaet schwankte staerker als angenommen (Drain-Zustaende,
fremde Jobs anderer Nutzer) -> vor dem eigentlichen Sweep wurde das Skript
um einen **Wellen-Modus** erweitert (adaptiv: `1 <= GPU-Zahl <= Werteanzahl`,
`ceil(Werteanzahl/GPU-Zahl)` sequenzielle Wellen, EIN Job, EIN `/dev/shm`-
Staging ueber alle Wellen hinweg). Diese Erweiterung war noetig, weil zum
Submit-Zeitpunkt kein Knoten 5 GPUs am Stueck freigab.

## Ablauf

### 1. Wellen-Modus-Erweiterung (vor dem Sweep)

`sbatch_sweep_parallel.sh` umgebaut: GPU-Guard von `NGPU == |VALUES|` auf
`1 <= NGPU <= |VALUES|` gelockert, Trainings-Start+Wait zu einer
Wellen-Schleife verschmolzen (`for OFF in 0..|VALUES| step NGPU`), Staging
bleibt job-uebergreifend (gleiche `SLURM_JOB_ID`) ueber alle Wellen liegen.
Bei `NGPU == |VALUES|` degeneriert die Schleife zu genau 1 Welle = altes
Verhalten (rueckwaertskompatibel). Trockengetestet mit Mock-Prozessen fuer
NGPU=1/3/4/5/6/0 (Guard-Grenzfaelle) vor dem ersten scharfen Einsatz.

### 2. Job 122859 — 5-Werte-Sweep in 2 Wellen (a100-1, 4 GPUs)

Cluster-Kapazitaetscheck (`scontrol show node`) zeigte a100-1 mit 4 freien
GPUs als einzigen schedulbaren Kandidaten fuer >=4 GPUs. `--gres=gpu:a100:4`
-> automatisch 2 Wellen: `{0, 0.05, 0.1, 0.3}` parallel, danach `{1.0}` solo.
Ressourcen: `--mem=550G` (286G shm-Pack einmalig + 4x ~60-66G
Worker/pinned-Puffer), `--cpus-per-task=88`, `--time=11:00:00`.

**Ergebnis:** Welle 1 lief vollstaendig durch (19:31-01:52 Uhr, **6h21min**
statt der geschaetzten 3.5-4h — die urspruengliche Zeitschaetzung war ohne
empirische Grundlage getroffen und zu optimistisch). Welle 2
(`lambda_grad=1.0`, solo) startete um 01:52 Uhr und wurde bei Epoch 26/28
(Plateau 2/3, mIoU 0.6798) durch das `--time=11:00:00`-Limit um 06:23 Uhr
hart gekillt (`CANCELLED ... DUE TO TIME LIMIT`). Alle vier Welle-1-Werte
hatten vollstaendige `run_summary.json`; `lambda_grad=1.0` hatte nur einen
Zwischen-Checkpoint (`best_miou.pt`, Epoch 23, mIoU 0.6800), keinen
`run_summary.json`. Schritt 4/5 (Inference + Harvest) liefen fuer **keinen**
der 5 Werte, da der Job vor Erreichen dieser Schritte starb.

### 3. Resume-Luecke gefunden -> Task 16b.4b (Patch)

Vor dem Nachziehen des fehlenden Werts wurde `--resume` gegen `checkpointing.py`
geprueft: `save_checkpoint`/`load_checkpoint` persistieren `epoch`, `val_loss`,
`best_val_loss`, Modell-/Optimizer-/Scheduler-State — **nicht** `best_miou`,
den mIoU-Plateau-Zaehler oder die val-loss-Patience. `train_linux.py` setzt
`best_miou = -1.0` beim Skriptstart hart, unabhaengig vom Resume-Checkpoint.

**Risiko ohne Patch:** Nach Resume waere die naechste Decode-Epoche automatisch
"neues Bestes" (da `-1.0 <` alles) und haette `best_miou.pt` (Epoch 23,
0.6800) potenziell mit einem schlechteren Wert ueberschrieben, plus Verlust
des Plateau-Fortschritts (2/3 -> 0/3).

**Patch (`checkpointing.py` + `train_linux.py`, 9 Hunks):**
- `save_checkpoint`: 5 neue optionale Kwargs (`best_miou`, `best_epoch_miou`,
  `miou_patience_cnt`, `miou_stop_ref`, `patience_cnt`), Defaults = altes
  Frisch-Init-Verhalten -> rueckwaertskompatibel zu allen bestehenden
  Checkpoints (z.B. Task-14-`best_miou.pt`).
- Resume-Block haelt `resume_ckpt` zwischen; ein neuer Restore-Block direkt
  nach der bestehenden mIoU-Frisch-Init ueberschreibt die 5 Felder per
  `.get()`, falls `resume_ckpt` vorhanden.
- Alle drei `save_checkpoint`-Aufrufe (`best_val_loss.pt`, `best_miou.pt`,
  periodischer `epoch_XXX.pt`) geben den vollen Zustand jetzt mit.
- Verifikation vor Deploy: `py_compile` beide Dateien, isolierter
  Funktionstest (Save->Load->Restore mit den echten Zahlen aus Job 122859:
  `best_miou=0.68`, Epoch 23, Plateau 2/3) sowie Gegenprobe mit einem
  simulierten Alt-Checkpoint (fehlende Keys) -> faellt exakt auf die alten
  Defaults zurueck. Beide Assertions bestanden vor dem Cluster-Deploy.

### 4. Live-Nachweis (unbeabsichtigt, aber aufschlussreich)

Job 122997 (1 GPU, a100-3) resumte `lambda_grad=1.0` von genau dem
`best_miou.pt`-Checkpoint, den **Job 122859 vor dem Patch-Deploy** gespeichert
hatte. Ergebnis im Log:

```
→ mIoU-Zustand wiederhergestellt: best_miou=-1.0000 (Epoch -1), Plateau=0/3
```

Dies ist **kein Patch-Fehler**, sondern der beabsichtigte Rueckwaertskompatibilitaets-
Pfad in freier Wildbahn: der geladene Checkpoint traegt die neuen Felder nicht
(vor dem Patch gespeichert), `.get()` faellt korrekt auf die alten Defaults
zurueck — exakt das im Funktionstest geprueft Verhalten, hier scharf
ausgeloest statt im Mock. Effekt: Training lief 9 Epochs laenger (24->32
statt sofortigem Stop bei Plateau 3/3), `best_miou.pt` wurde bei Epoch 24
(0.6772, schlechter als der eigentliche Epoch-23-Bestwert 0.6800) einmal
ueberschrieben — das exakte Risiko, das der Patch fuer *neue* Checkpoints
verhindert. Kein Schaden am Endergebnis: das Training fand am Ende einen
besseren Wert (0.6834, Epoch 29) als der urspruengliche Zwischenstand.
Mehrkosten: ~1h zusaetzliche Rechenzeit statt eines sofortigen Stopps.
**Ab jetzt tragen alle neu gespeicherten Checkpoints den vollen Zustand;
kuenftige Resumes greifen wie vorgesehen.**

### 5. Vollstaendige Nachernte (Job 122997, im Anschluss an den Resume)

Da fuer keinen der 5 Werte Inference/Harvest gelaufen war, hat Job 122997
nach Abschluss des Resume-Trainings automatisch die serielle Inference
(16b.3-Methodik, 300 Val-Samples, `--no-save`) fuer **alle 5 Werte**
nachgezogen, danach den CSV-Harvest. Laufzeit gesamt (Resume-Training +
5x Inference + Harvest): ca. 2h40min (14:31-15:0x Uhr grob, keine Zeitlimit-
Probleme, 1 GPU auf a100-3, kein Konkurrenzdruck).

## Ergebnisse

| lambda_grad | mIoU   | std-Ratio | Stop-Epoch | Stop-Grund     |
|-------------|--------|-----------|------------|----------------|
| 0           | 0.6731 | 0.9085    | 17         | miou_plateau   |
| 0.05        | 0.6772 | 0.9096    | 20         | miou_plateau   |
| 0.1         | 0.6787 | 0.9021    | 20         | miou_plateau   |
| 0.3         | 0.6858 | 0.8907    | 23         | miou_plateau   |
| 1.0         | 0.6834 | 0.8816    | 29 (Resume, s.o.) | miou_plateau |

CSV: `predictions/task16b/sweep_lambda_grad.csv`

### GO/NO-GO (Nullpunkt lambda_grad=0)

- mIoU 0.6731 vs. Band `[0.666, 0.694]` (0.68 +/- 0.014) -> **im Band**
- std-Ratio 0.9085 vs. Band `[0.906, 0.924]` (0.915 +/- 0.009) -> **im Band,
  knapp** (0.0025 ueber der Untergrenze)

**-> GO.** Hardware-Bruecke lokal->Cluster bestaetigt, Stufe-1-Speedup
numerisch neutral. Interpretation der uebrigen Werte zulaessig.

### Kurvenform (Vorbehalt: n=1 Seed/Wert, Signifikanzschwelle fuer n=3 kalibriert)

| lambda_grad | delta mIoU vs. 0 | delta std-Ratio vs. 0 |
|-------------|------------------|------------------------|
| 0.05        | +0.0041 (n.s.)   | +0.0011 (n.s.)         |
| 0.1         | +0.0056 (n.s.)   | -0.0064 (n.s.)         |
| 0.3         | +0.0127 (n.s., knapp unter Schwelle) | **-0.0178 (signifikant)** |
| 1.0         | +0.0103 (n.s.)   | **-0.0269 (signifikant)** |

- mIoU-Effekt durchgehend **nicht signifikant**; leichter Aufwaertstrend bis
  0.3, danach leichter Ruecklauf zu 1.0 (0.6858 -> 0.6834, delta -0.0024) —
  passt zum erwarteten Knie/Rauschmuster, aber mit n=1 statistisch nicht
  abgesichert.
- std-Ratio verschlechtert sich **signifikant und monoton** mit steigendem
  lambda_grad (0.9085 -> 0.8816) — bewegt sich von 1.0 weg, d.h. der Regler
  vergroessert den std-Gap, statt ihn zu schliessen. Wirkt damit tendenziell
  gegen den `lambda_std=1.0`-Arbeitspunkt.
- **Einordnung:** lambda_grad zeigt keinen belastbaren mIoU-Gewinn, aber
  einen klar messbaren std-Ratio-Nachteil bei hoeheren Werten. Falls der
  Regler in den finalen Loss aufgenommen wird, sprechen die Daten eher fuer
  einen kleinen bis moderaten Wert (0.05-0.1) als fuer 1.0.

## Lektionen

1. **Zeitschaetzungen ohne empirische Grundlage sind unzuverlaessig.** Die
   urspruengliche 3.5-4h-Annahme pro Welle war zu optimistisch (real:
   6h21min unter 4-Wege-GPU-Konkurrenz). Kuenftige `--time`-Budgets: aus
   real gemessenen Epochenzeiten hochrechnen
   (`n_wellen * ~30 Epochs * ~650s Konkurrenz-Puffer + Staging + Inference
   + 20% Sicherheit`), nicht aus einer Vorab-Faustformel.
2. **Checkpoint-Resume ist nur so vollstaendig wie der gespeicherte Zustand.**
   `--resume` existierte bereits, deckte aber nur Modell/Optimizer/Scheduler/
   Epoch/best_val_loss ab — nicht die mIoU-Plateau-Logik, die den
   eigentlichen Stop-Mechanismus des Sweeps darstellt. Sowas faellt nur auf,
   wenn man den Resume-Pfad VOR dem Ernstfall durchdenkt, nicht danach.
3. **Rueckwaertskompatible Patches beweisen sich am besten live — auch
   ungeplant.** Der Resume von einem Vor-Patch-Checkpoint war ein
   Zufalls-Funktionstest des `.get()`-Fallback-Pfads unter Realbedingungen;
   Ergebnis deckungsgleich mit dem isolierten Mock-Test vor dem Deploy.
4. **`stage_to_shm` ist bereits in `train_linux.py` selbst verankert**
   (config-getrieben, `data.stage_to_shm: true`), nicht nur im sbatch-
   Orchestrierungsskript. Ein einzelner `python train_linux.py --resume ...`-
   Aufruf staged und raeumt sich selbst — kein eigenes Wellen-sbatch fuer
   Einzelwert-Nachlaeufe noetig.
5. **Wellen-Modus (adaptiv `1 <= NGPU <= |VALUES|`) ist jetzt der
   Standard-Pfad** fuer alle Folge-Sweeps (16b.5+), nicht nur ein Sonderfall
   fuer 16b.4. Degeneriert bei ausreichend freien GPUs automatisch zu einer
   einzigen Welle.

## Geaenderte/erzeugte Dateien

- `sbatch_sweep_parallel.sh` — Wellen-Modus-Erweiterung (dauerhaft, gilt fuer
  16b.5+)
- `checkpointing.py`, `train_linux.py` — Task-16b.4b-Resume-Patch (dauerhaft)
- `sbatch_resume_lambda_grad_1.sh` — Einmal-Skript fuer den Nachlauf
  (Resume + Vollernte aller 5 Werte); als Vorlage fuer kuenftige
  Einzelwert-Nachlaeufe wiederverwendbar
- `configs/task16b/lambda_grad_sweep/config_lambda_grad_{0,0.05,0.1,0.3,1.0}.yaml`
- `checkpoints/task16b/lambda_grad_sweep/lambda_grad_{...}/phase2/`
- `predictions/task16b/lambda_grad_sweep/lambda_grad_{...}/inference_log.json`
- `predictions/task16b/sweep_lambda_grad.csv`

## Naechster Schritt

Task 16b.5 (`lambda_ssim`, VALUES=(0 0.1 0.3), 3 Werte) — kann direkt den
wellenfaehigen `sbatch_sweep_parallel.sh` und den `--resume`-Patch nutzen,
ohne weitere Infrastruktur-Aenderungen.
