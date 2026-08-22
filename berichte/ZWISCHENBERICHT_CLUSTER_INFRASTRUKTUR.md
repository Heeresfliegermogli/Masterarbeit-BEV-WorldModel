# Zwischenbericht — Cluster-Infrastruktur (DGX A100, LRT8)

**Zeitraum:** Juli 2026
**Zweck:** Referenzdokument für alle bisher geschaffenen Strukturen, betriebliche Erkenntnisse und offene Punkte der Cluster-Migration. Ergänzt `ZWISCHENBERICHT_TASK15_CLUSTER.md` (dort: konzeptionelle Einordnung + Kurzfassung); hier: vollständige technische Referenz.

---

## 1. Zugang

- Account über Thorsten Lüttel (LDAP), Nutzer `vima`, Gruppen `tas-students` + `dgx-user`.
- Login ausschließlich über den Headnode: `ssh vima@tas-dgx-head.lrt.unibw-muenchen.de` — nur aus dem UniBw-Netz oder per VPN. Direkter Login auf die DGX-Knoten selbst ist nicht möglich, nur über SLURM.
- Erstpasswort-Reset lief zunächst nicht über die Uni-Mail-gekoppelte Erwartung — über Rücksprache mit Thorsten geklärt.

## 2. Hardware & System — mit Korrekturen gegenüber der offiziellen Anleitung

| Aspekt | Anleitung sagt | Tatsächlich verifiziert |
|---|---|---|
| GPUs pro Knoten | Diagramm zeigt 4 | **8×** A100-SXM4-80GB pro Knoten (`scontrol show node`: `Gres=gpu:a100:8`) → 32 GPUs cluster-weit über 4 Knoten |
| GRES-Bezeichner | Beispiel nutzt `gpu:A100:1` | tatsächlich **klein**: `gpu:a100:1` — Großschreibung führt zu `Unable to allocate resources` |
| `/raid` (Node-Scratch) | "very fast scratch volume… only for temp usage" (impliziert nutzbar) | existiert (28TB, `md1`-RAID0), aber **`root:root`, 755** — kein Schreibzugriff für reguläre Nutzer. Nutzung ungeklärt (kein Ansprechpartner über Anton lösbar, siehe Offene Punkte) |

Weitere Eckdaten: 256 CPU-Kerne, ~2TB RAM pro Knoten; Ubuntu 24.04, Treiber 595.71.05, CUDA 13.2 (Treiber-Maximum, PyTorch-Wheels für ältere CUDA-Versionen laufen problemlos darunter); Partition `gpu` (Default), alle 4 Knoten (`tas-dgx-a100-[1-4]`) dort zusammengefasst.

## 3. Geschaffene Verzeichnisstruktur

| Inhalt | Pfad | Bemerkung |
|---|---|---|
| Code | `/home/vima/Code_final/` | rsync vom lokalen Server, bewusst ohne `predictions/`, `checkpoints/` (voll), `wandb/`, `__pycache__/` |
| Checkpoints | `/home/vima/Code_final/checkpoints/{task14,task15}/` | selektiv: nur `best_*.pt`, historische/verworfene Läufe (`task13_SANITYCHECK*`, `Finished/`) bewusst ausgelassen |
| PKL-Metadaten | `/home/vima/pkl/nuscenes_infos_{train,val}.pkl` | im Home, **nicht** auf beegfs (Schreibrechte auf `.../students/`-Ebene fehlen) |
| Latents (440G/95G) | `/mnt/beegfs/ssd/lrt81-students/lrt81-vima/latents/seg/{train,val}` | von Thorsten vorbereitet; `~/beegfs` ist ein Symlink dorthin — Achtung: `readlink -f ~/beegfs` zeigt den korrekten Pfad, der reale Ordnername ist `lrt81-students` (ein Bindestrich-Wort), nicht `lrt81/students/` |
| Seg-Decoder-Checkpoint | `/home/vima/bevfusion-seg.pth` | für Decode-Validierung (Task 14) |
| Miniconda + Env | `/home/vima/miniconda3`, Env `bevwm` | Python 3.10 |

## 4. Python-Environment (`bevwm`)

`torch==2.1.2+cu121`, `numpy==1.26.4` (**explizit gepinnt** — 2.x bricht die Torch↔NumPy-Interop, `_ARRAY_API not found`), `pyyaml`, `tqdm`, `h5py`, `wandb`, `pytorch-msssim` (für reproduzierbares `lambda_ssim`-Verhalten, sonst stiller Fallback auf MSE).

## 5. SLURM — Betriebswissen & Fallstricke

Diese Liste ist der eigentliche Ertrag der Migration im Kleinen — Punkte, die beim nächsten Job direkt Zeit sparen:

1. **`--mem` immer explizit setzen.** Ohne Angabe reserviert SLURM offenbar den kompletten Knotenspeicher (~2TB) als Default — der Job wartet dann effektiv auf einen komplett freien Knoten, unabhängig davon, wie viele GPUs frei sind. Ursache mehrerer scheinbar unerklärlicher Wartezeiten.
2. **GRES-Typ `a100` klein schreiben**, nicht `A100` (s. Tabelle oben).
3. **`tmux`-Sessions sind node-lokal.** Eine auf dem Headnode gestartete Session ist auf einem DGX-Knoten nicht sichtbar (`tmux ls` → `error connecting`, eigener Tmux-Server pro Host). Immer: erst `tmux new -s <name>` auf dem Headnode, danach `srun ... --pty bash` **innerhalb** dieser Session — nicht umgekehrt.
4. **`/tmp` ist ebenfalls node-lokal.** Nur `/home` ist laut Anleitung überall gleich gemountet — bestätigt, alles andere (auch `/tmp`) ist es nicht. Test-Skripte im Home ablegen, nicht in `/tmp`, wenn sie über Headnode/DGX hinweg erreichbar sein sollen.
5. **`REASON=(Resources)` in `squeue` heißt "wartet AUF Ressourcen"**, nicht "hat Ressourcen bekommen" — leicht misszuverstehen, kein Fehler, reine Terminologie.
6. **Beobachtete Fairshare-Dynamik:** `FAIRSHARE=0` auch als komplett neuer Nutzer beobachtet (`sprio -u vima`), Ursache nicht abschließend geklärt (evtl. clusterweite QOS-Konfiguration). `REASON` wechselt zwischen `(Priority)` und `(Resources)` je nach Cluster-Gesamtlast — bei aktiver Mehrfachnutzung durch andere (ein anderer Nutzer mit 3 gleichzeitigen Jobs beobachtet) können auch kurze Jobs mehrere Minuten bis Stunden warten.
7. **`conda tos accept`** für die Default-Channels ist vor der ersten `conda create` nötig (neuere Conda-Version, in der Anleitung nicht erwähnt).
8. **Nicht-interaktive Kontexte (`| tee`) unterdrücken wandb-Prompts** — führt zu `UsageError: No API key configured` statt der erwarteten (1)/(2)/(3)-Abfrage. Fix: `export WANDB_MODE=offline` explizit vor dem Trainingslauf setzen, unabhängig davon, ob die Ausgabe gepiped wird.

## 6. I/O-Diagnose auf dem Cluster — Kernbefund

Erwartung vor der Migration: BeeGFS über Infiniband HDR-200 sollte der lokalen Einzelplatte klar überlegen sein. Tatsächlicher Befund: das Gegenteil, jedenfalls mit dem ursprünglichen Zugriffsmuster.

| Metrik | Lokal (alre-server) | Cluster (BeeGFS, ursprünglich) |
|---|---|---|
| GPU-Auslastung | 60–70% | 22–35% |
| Wirkung von mehr `num_workers` | keine (Platte am Anschlag) | kaum (vermutlich Netzwerk-Metadaten-Rundlauf, nicht Bandbreite) |

**Ausgeschlossene Fixes** (jeweils mit Messung, nicht nur vermutet):
- Mehr `num_workers` (4→16): keine signifikante Verbesserung.
- RAM-Caching via `--mem=600G`: `free -h` zeigte nur `buff/cache=182G` statt der erwarteten ~535G trotz reichlich freiem RAM — vermutlich Cgroup-Reclaim auf dem geteilten Multi-Tenant-Knoten, nicht kontrollierbar.
- `/raid`-Node-Scratch: keine Schreibrechte (s. Abschnitt 2).

**Verifizierter Fix:** Memmap-Packing (viele Frames in eine zusammenhängende Datei statt tausender Einzeldateien), per Mini-Benchmark mit Summen-Check abgesichert (identische Werte über alle Varianten, kein Wiederholen des früheren HDF5-Fehlschlags):

| Umgebung | Referenz (parallelisiert) | Memmap (seriell) | Faktor |
|---|---|---|---|
| Cluster (BeeGFS) | 8 Worker: 0,76s | 0,29s | **2,60×** |
| Lokal (NVMe) | 4 Worker: 0,32s | 0,23s | **1,39×** |

→ Umgesetzt als **Task 15.3B** im Arbeitsplan (zwischen 15.3 und 15.4 eingeschoben, da 15.1–15.3 bereits auf der alten Pipeline abgeschlossen sind).

## 7. Aktueller Stand

- Pipeline läuft Ende-zu-Ende auf dem Cluster (Smoke-Test über echten Trainingslauf bestanden — Environment, Pfade, GPU-Zugriff funktionieren), aber mit unzureichendem Durchsatz für produktive Sweeps.
- Multi-GPU/DDP bewusst **nicht** als Lösung gewählt — bei 22–35% GPU-Auslastung ist Datenhunger, nicht Rechenkapazität, der Engpass; mehr GPUs würden die I/O-Last nur vervielfachen. Der sinnvolle Parallelitäts-Hebel liegt stattdessen auf Job-Ebene (mehrere unabhängige Sweep-Läufe gleichzeitig auf verschiedenen freien GPUs) — für die Fortsetzung von 15.3+ vorgemerkt.

## 8. Offene Punkte (nicht vergessen)

- **Szenenzahl-Diskrepanz ungeklärt:** Cluster meldete beim Smoke-Test 364/364 Train- bzw. 92/92 Val-Szenen statt der lokal gewohnten 700/150 (Sequenzzahlen mit 27038/5743 lagen dabei überraschend nah an den lokalen 28130/6019). Ursache nie verifiziert — vorgeschlagener Check (Frame-Anzahl direkt aus der PKL zählen, lokal gegen Cluster vergleichen) wurde nie durchgeführt, weil die I/O-Untersuchung dazwischenkam. Vor dem nächsten großen Sweep-Lauf nachholen.
- **`inference.py` hat vermutlich denselben `code`/`Code`-Case-Bug** wie `smoke_test.py` (dort gefixt), aber nie verifiziert oder korrigiert — noch nicht produktiv genutzt auf dem Cluster.
- **`/raid`-Schreibrechte** weiterhin ungeklärt; laut Rückmeldung ist Anton dafür nicht der richtige Ansprechpartner — offen, wer es ist.
- **Task 15.3B** (Memmap-Packing-Umbau) steht als nächster Schritt in eigenem Chat an, vor Fortsetzung von Task 15.4.
