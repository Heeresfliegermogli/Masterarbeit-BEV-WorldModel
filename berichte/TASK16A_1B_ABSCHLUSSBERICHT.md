# TASK 16a.1B — ABSCHLUSSBERICHT

**Titel:** Trainings-Beschleunigung lokal (Einschub vor dem ersten Sweep)
**Datum:** 2026-07-12
**Server:** alre-server-u20 (NVIDIA TITAN RTX, 24 GB), Python 3.8.10, PyTorch 2.1.2
**Status:** ABGESCHLOSSEN — lokal 1.19x (Step-Harness); Cluster: Epochenzeit
halbiert (Train-Anteil 1146 -> 574 s, Steady-State-Lauf Job 122810)


## 1. Aufgabe

Vor den 16a-Sweeps (~13-19 Laeufe a 3-5h) den lokalen Trainings-Step
beschleunigen. Grundlage: Recherche-Bericht "Training Acceleration for a Small
BEV Transformer World Model" (12.07.2026, im Projekt als PDF). Hardware-
Realitaet TITAN RTX (Turing sm_75): kein TF32, keine bf16-Tensor-Cores;
fp16+GradScaler (aktiv) ist bereits die richtige Wahl. Vorgehen nach Plan:
Stufe 0 (Profiling, Pflicht) -> Stufe 1 (numerisch neutral) -> Stufe 2/3 nur
bei belegtem Nutzen.


## 2. Stufe 0 — Profiling (profile_train.py, temporaer)

### 2.1 Methodik
Eigenstaendiges Wegwerf-Skript `profile_train.py` (analog diagnose_mean_gap.py
in 16a.1): nutzt die kanonischen Interfaces (BEVWorldModel, compute_loss,
build_dataloaders, set_seed), bildet den heissen Loop aus train_one_epoch 1:1
nach (autocast + GradScaler + clip max_norm=1.0), fasst train_linux.py fuer die
Messung NICHT an. Per-Phase-Wanduhr (h2d/forward/loss/backward/optim) via
CUDA-Events; data_wait (Loader-Warten) separat per perf_counter; SSIM-Kosten
bewusst als A/B (lambda_ssim an vs. 0.0, volles Forward+Backward-Delta) statt
record_function um den Forward-Call (der den Backward-Anteil verschluckt
haette); torch.profiler-Op-Tabelle zur Head-vs-SSIM-Trennung.

### 2.2 Baseline-Befund (bs=8, n=8 aktive Steps, IST-Zustand)

| Phase | ms/Step | % |
|---|---:|---:|
| h2d | 42.03 | 24.4 |
| forward | 31.89 | 18.5 |
| loss | 9.23 | 5.4 |
| backward | 75.48 | 43.8 |
| optim | 13.78 | 8.0 |
| **Step gesamt** | **172.5** | |

Kernbefunde:
- **data_wait ~0.1 ms/Step** -> der Loader ist lokal NICHT der Engpass
  (deckt sich mit 15.3B §5: lokal data% 16-21%, GPU-bound).
- **Stufe 1c (SDPA) war bereits erledigt:** nn.MultiheadAttention mit
  need_weights=False dispatcht in torch 2.1 auf den fused SDPA-Pfad
  (Flash-/Mem-Efficient-Kernel in der Op-Tabelle sichtbar). Auf A100 nimmt
  derselbe Code automatisch die schnelleren Kernel — Cluster-Bonus gratis,
  kein Umbau noetig.
- backward dominiert und ist attention-lastig (bereits optimaler Pfad) ->
  niedrige Decke fuer weitere Stufe-1-Massnahmen absehbar.
- h2d 42 ms: der Loader upcastete fp16->fp32 VOR dem PCIe-Transfer ->
  doppeltes Busvolumen. Das wurde der Stufe-1-Haupthebel (1d).


## 3. Umbauten (Stufe 1, alle in der einen Codebasis)

### 3.1 — 1d: fp16 ueber den Bus, Upcast erst auf der GPU
`bev_dataset._load_latent` gibt den Latent im STORAGE-dtype zurueck (fp16
bleibt fp16; np.array(row) statt astype("float32")). Der Upcast passiert an
den h2d-Punkten GPU-seitig (`.to(device, non_blocking=True).float()`).
Fuenf Schnittstellen:
- train_linux.py: Train-Loop, Validate-Loop, Sample-Viz — GPU-Upcast.
- train_linux.py Val-Subset-Cache: `.cpu().float()` — der Seg-Decoder
  (mIoU-Referenz) laeuft bewusst fp32; der Decode-/Metrik-Pfad bleibt
  dadurch bit-identisch. Decode-Val bezieht daraus bereits fp32.
- eval_full_val.py: inp GPU-Upcast; tgt `.float()` (speist den Decoder ->
  schuetzt die Headline-mIoU).
- inference.py UNBERUEHRT (liest rohe .npy, Modus A liefert weiter fp32).

Neutralitaet: fp16->fp32 ist ein exaktes Widening — Modell und Decoder sehen
in allen Pfaden bit-identisches fp32 wie zuvor; nur der Ort des Upcasts und
das Busvolumen aendern sich. Beleg im Profil: Memcpy HtoD 335 -> 165 ms
(Summe ueber 8 Steps) — exakt halbiert. Kein neuer Anker noetig.
Rueckwaertskompatibel: auf fp32-Packs ist np.array(row) fp32 und .float()
ein No-op.

### 3.2 — 1b: fused AdamW + cudnn.benchmark
- `AdamW(..., fused=(device.type=="cuda"))`: buendelt die ~1000
  aten::add_-Kernel/Step in einen; kompatibel mit GradScaler (torch 2.1.2).
- `set_seed`: cudnn.benchmark=True / deterministic=False. Shapes sind
  konstant -> einmaliges Autotune im ersten Step, danach Gewinn.
  TRANSPARENZ-VERMERK: Bit-Reproduzierbarkeit DESSELBEN Seeds entfaellt;
  die 15.2-Signifikanzschwellen basieren auf Seed-ZU-Seed-Varianz und
  bleiben unberuehrt.


## 4. Ergebnis lokal (bs=8, n=25 aktive Steps, nach 1d+1b)

| Phase | vorher (n=8) | nachher (n=25) |
|---|---:|---:|
| h2d | 42.03 | 22.40 |
| forward | 31.89 | 32.61 |
| loss | 9.23 | 12.64 |
| backward | 75.48 | 73.69 |
| optim | 13.78 | 3.67 |
| **Step gesamt** | **172.5** | **145.2** |

**Speedup 1.19x im Step-Harness.** Compute-Kern (fwd+loss+bwd) stabil
~117-119 ms ueber alle Laeufe — der echte Boden der TITAN. Projektion
Train-Anteil/Epoche: 3380 x 145 ms ~ 490 s (vorher ~585 s); die reale
Epochenzeit liefert der erste 16a.2-Lauf (run_summary.json).

Messmethodik-Korrektur (dokumentiert): der 1d-Zwischenlauf bei n=8 zeigte
einen flachen Step (h2d-Gewinn scheinbar im optim-Fenster "verschluckt") —
das war n=8-Rauschen der Fensterzuordnung, KEINE Ueberlappung; auf dem
seriellen CUDA-Stream ist die h2d-Ersparnis real, wie n=25 zeigt.


## 5. bs32-Probe — Stufe 2 GESTRICHEN (schliesst 15.3B-§3-Uebergabe)

Die in 15.3B aufgesetzte, nie durchgekommene batch_size-8->32-Diagnose wurde
lokal per profile_train nachgeholt (config_probe_bs32.yaml, n=12), beide
Laeufe mit 1d+1b:

| pro Sample | bs=8 | bs=32 |
|---|---:|---:|
| h2d | 2.80 ms | 2.80 ms |
| forward | 4.08 ms | 4.10 ms |
| loss | 1.58 ms | 1.35 ms |
| backward | 9.21 ms | 9.00 ms |
| **gesamt** | **18.15 ms** | **17.39 ms** |

Alle Phasen pro Sample flach (h2d/forward bis auf die zweite Nachkommastelle
identisch) -> die TITAN ist bei bs=8 bereits GESAETTIGT — die 3072 Tokens/
Sample machen die effektive Batch-Dimension laengst gross. Gewinn ~4.2%,
dagegen stuenden neuer Anker, Wurzel-LR-Reskalierung und Verlust der direkten
Vergleichbarkeit zum 15.3-Arbeitspunkt. **Entscheidung: Stufe 2 (bs->32)
entfaellt.** (bs=32 passt in 24 GB — belegt —, lohnt nur nicht.)


## 6. Weitere Streichungen (begruendet)

- **1a channels_last:** Conv-Anteil ~8% des Steps (ConvolutionBackward
  ~11 ms), realistischer Gewinn 1-3 ms; dagegen 5D->4D-Reshape-Umbau
  (memory_format propagiert nicht automatisch durch den [B,3,256,128,128]-
  Pfad) mitten vor der Sweep-Kampagne. Gestrichen.
- **Stufe 2f SSIM@32:** SSIM-A/B bei n=8 nicht belastbar (11.9% vs. 20.6%
  fuer identischen Code zwischen zwei Laeufen). Zusaetzlich: Massnahme waere
  DYNAMIKAENDERND (Loss aendert sich -> Anker) und wuerde den kommenden
  lambda_ssim-Sweep (16a.3) konfundieren. Gestrichen.
- **Stufe 3 torch.compile:** laut Plan optional/riskant, auf Turing kleiner
  Gewinn erwartet; Sweep-Start hat Vorrang. Gestrichen.


## 7. Cluster-Deployment + 1d-A/B-Messung (Job 122803, A100, 1 Knoten)

Deployment als reiner Datei-Sync (Ein-Code-Pfad-Prinzip): bev_dataset.py
(-> Code/), train_linux.py, eval_full_val.py, profile_train.py per scp,
vorher Backup nach backup_pre_16a1b/. Messdesign: A/B in EINEM sbatch-Job
(`sbatch_profile_1d_ab.sh`) auf identischem Knoten — Lauf A mit alter
bev_dataset.py (Backup; .float() ist auf fp32-Rueckgabe No-op -> exakt alter
Pfad), Lauf B mit neuer; 1b in BEIDEN aktiv -> isoliert den 1d-Effekt.
Staging lief einmalig in A (236 GB in 596 s = 0.40 GB/s + 50.5 GB in 79 s —
deckt die dokumentierten BeeGFS-Raten), B skippte idempotent; EXIT-trap
(Datei-Restore + /dev/shm-Cleanup) griff, Log fehlerfrei, env verifiziert
([env]-Zeile: bevwm-Python).

### Ergebnis (je 25 aktive Steps, bs=8)

| Kennzahl (Cluster, A100) | A: vor 1d | B: nach 1d | Delta |
|---|---:|---:|---|
| data_wait (Loader-Warten) ms/Step | 148.80 | 20.91 | **/ 7.1** |
| h2d ms/Step | 24.36 | 14.09 | / 1.7 |
| Memcpy HtoD (Op-Tabelle, avg) ms | 12.00 | 6.57 | / 1.8 (halbiert) |
| Step GPU-seitig (sync-zu-sync) ms | 101.86 | 89.25 | -12% |
| **Effektiver Zyklus (Step + Warten) ms** | **250.7** | **110.2** | **2.27x** |

Der Compute-Kern ist zwischen A und B stabil (forward 18.1/19.1, backward
46.4/45.1 ms) — das Delta liegt REIN im Datenpfad, das Experiment ist sauber.
1d wirkt auf dem Cluster genau dort, wo 15.3B den Engpass dokumentiert hat:
der Worker-astype entfaellt, stack/pin/h2d bewegen halbe Bytes, der Loader
kommt fast hinterher (data_wait 149 -> 21 ms).

### Steady-State-Bestaetigung (3-Epochen-Lauf, Job 122810)

Regulaerer train_linux.py-Lauf (config_timing_check_3ep.yaml: epochs=3,
eigenes checkpoint-dir, sonst identisch zur Cluster-fp16-Config; BEIDE Loader
aktiv, persistent_workers) — der Realbetriebs-Check zur Kurzlauf-Projektion:

| pro Epoche | vor 1d (15.3B-Telemetrie) | nach 1d+1b (E0-E2) | Delta |
|---|---:|---:|---|
| data_time | 847 s | 333 s | / 2.5 |
| compute_time | 299 s | 240 s | -20% (1b sichtbar) |
| **Train-Anteil** | **1146 s** | **574 s** | **2.0x** |
| data% | 74-76 | 57.0-58.1 | |
| volle Epoche (inkl. Val) | ~1370-1580 s | 691.6-700.3 s | ~2.1x |

Der Kurzlauf-Vorsichtsvermerk hat sich BESTAETIGT: data_wait im Steady-State
~98.6 ms/Step (333 s / 3380) statt 20.9 ms im Profiling — der parallele
val-Loader (persistent_workers) und das Prefetch-Gleichgewicht ueber 3380
Steps kosten real. Messhinweis: die data/compute-Aufteilung ist eine
CPU-seitige Naeherung (async-CUDA verschiebt Anteile ueber die Grenze);
die Bruttozahlen (Train-Anteil, Epochenzeit) sind die belastbaren Groessen.
Nebenbefunde: mIoU 0.6104 nach 3 Epochen (gestauchte Cosine-LR, Baseline-
Lambdas) plausibel-gesund, KEIN Ankervergleich; shm-trap raeumte sauber;
BeeGFS-Staging diesmal 0.67 GB/s (Lastvariabilitaet, vgl. 0.40 im A/B-Job).

### Einordnung und Konsequenz

- **Ergebnis der Gleis-Frage: Einzellauf-GLEICHSTAND statt Umkehr.** Die
  Kurzlauf-Projektion (~372 s, "Cluster schneller als lokal") hielt im
  Steady-State nicht; real: Cluster 574 s Train-Anteil vs. lokal ~490 s
  (projiziert aus 145.2 ms x 3380). Der 15.3B-§6-Malus ("Cluster pro
  Einzellauf ~2x langsamer") ist damit ELIMINIERT, aber nicht gedreht.
  Strategische Konsequenz fuer 16b: **Cluster-Parallelitaet gibt es jetzt
  ohne Einzellauf-Strafe** — mehrere Sweep-Werte gleichzeitig bei ~lokaler
  Einzellauf-Geschwindigkeit.
- A100-Compute vs. TITAN: 89.05 vs. 145.0 ms = nur 1.63x — konsistent mit
  "A100 ist Hardware-Overkill fuer das 6M-Modell" (15.3B). Der Gewinn kam
  nicht aus der GPU, sondern aus dem Datenpfad.
- Rest-Engpass sichtbar: data% ~58 auch nach 1d. Kandidat: der einzelne
  pin_memory-Thread (268 MB/Batch pinnen) bzw. collate — optionaler
  16b-Hebel, kein Blocker.
- Report-Header der A/B-Laeufe zeigt lambda_std=0.1 statt 1.0: die
  .get-Defaults griffen, weil die Cluster-Config noch das 3-Term-Schema
  traegt — fuer alle Timing-Messungen irrelevant.


## 8. Verifikation (Stufe-1-Erfolgskriterium) — uebergeben an 16a.2

Plan-Kriterium: mIoU im 15.2-Rauschband. KEIN separater Anker-Lauf: der
**lambda_grad=0-Nullpunkt von 16a.2 IST exakt dieser Anker** (Sweep-Basis-
Config, neue Geschwindigkeit). Pruefkriterien beim ersten Sweep-Lauf:
mIoU im Band des Arbeitspunkts (~0.68 +/- 0.014), std-Ratio ~0.915 +/- 0.009.
Damit liefert der erste 16a.2-Lauf Neutralitaets-Bestaetigung UND reale
Epochenzeit in einem.


## 9. Learnings / Merkposten

- **n=8 ist fuer Step-Deltas zu verrauscht** (optim-Fenster-Sprung 14->33 ms,
  SSIM-A/B 11.9<->20.6%). Fuer Vorher/Nachher-Aussagen n>=25 aktive Steps;
  nur Step-Total und Op-Tabellen-Aggregate (Memcpy, add_) sind bei n=8 stabil.
- **profile_train --out immer explizit setzen** — statischer Default hat den
  bs8-Report mit dem bs32-Report ueberschrieben (aus Konsole rekonstruiert,
  keine Datenverluste). Header-Zeile war zudem ein statischer String
  ("benchmark=False" auch nach 1b) — inzwischen dynamisch aus
  torch.backends gelesen.
- **SDPA-Check vor SDPA-Umbau:** nn.MultiheadAttention(need_weights=False)
  nutzt in torch 2.1 bereits den fused Pfad — Op-Tabelle pruefen erspart
  einen geplanten Umbau (1c entfiel ersatzlos).
- **fp16->fp32 ist exaktes Widening** — Verlagern des Upcasts (CPU->GPU) ist
  bit-neutral und braucht keinen Anker; nur fp32->fp16 waere verlustbehaftet.
- Decoder-Pfade sind die fp32-Wachposten: ueberall wo seg_decoder Targets
  sieht (Val-Subset-Cache, eval_full_val tgt) expliziter .float()-Guard.
- scp-Konvention (15.3B §7) gilt weiter: von der Quellmaschine, ueberschreibt
  ohne Rueckfrage -> Cluster-Backup VOR dem Sync.


## 10. Offen / uebergeben

- **An 16a.2 (separater Chat):** (a) lambda_grad=0-Nullpunkt dient als
  Stufe-1-Verifikationsanker (Kriterien §8); (b) der in 16a.1 vorgemerkte
  `--no-save`-Patch fuer inference.py (npy-Dumps ~10 GB x Sweep-Werte
  vermeiden; Runner braucht nur inference_log.json) ist dort VOR dem
  Runner-Start einzubauen.
- **An 16b:** (a) GLEIS-NEUBEWERTUNG (Steady-State bestaetigt, Job 122810):
  Einzellauf-Gleichstand Cluster~lokal (574 vs. ~490 s Train-Anteil, volle
  Epoche ~695 s) — der 2x-Malus aus 15.3B ist eliminiert, Cluster-
  Parallelitaet kommt jetzt OHNE Einzellauf-Strafe -> 16b-Gleis gestaerkt.
  Der Telemetrie-Bestaetigungspunkt ist damit ERLEDIGT. Optionaler
  Rest-Hebel: data% ~58 (Kandidat pin_memory/collate). (b) Cluster-Sweep-
  Basis-Config (6-Term-Schema + BeeGFS-Pfade + stage_to_shm) — die
  vorhandene config_nuscenes_full_cluster_fp16.yaml traegt noch das
  3-Term-Schema (fuer Anker ok, fuer den Runner unpatchbar);
  (c) sweep_runner -> SLURM-Array.
- profile_train.py verbleibt als dokumentiert-temporaeres Artefakt neben dem
  Code (Reproduzierbarkeit der Profiling-Notiz), ist NICHT Teil der Pipeline.


## 11. Artefakte

- `profile_train.py` (temporaer; CUDA-Event-Phasen, SSIM-A/B, Op-Tabelle;
  robust gegen 3-Term-Configs)
- `sbatch_profile_1d_ab.sh` (Cluster-A/B in einem Job, doppelter EXIT-trap)
- Profil-Reports: `profile_report_bs8_nach_1d_1b.md` (rekonstruiert),
  `profile_report_bs32_probe.md`, Baseline + 1d-Zwischenstand in den
  Chat-Logs; Cluster-A/B: `predictions/task16a_1b_cluster/{A_vor_1d,
  B_nach_1d}/profile_report.md` + Job-Log `logs/profile_1d_ab_122803.out`
  (lokal gespiegelt)
- Steady-State-Check: `sbatch_timing_check_3ep.sh`,
  `config_timing_check_3ep.yaml` (Wegwerf, 2-Zeilen-diff zur Cluster-fp16-
  Config), Job-Log `logs/timing_check_122810.out`, run_summary.json +
  Checkpoints unter `checkpoints/task16a_1b_timing_check/` (loeschbar)
- Geaenderte Pipeline-Dateien: `bev_dataset.py`, `train_linux.py`,
  `eval_full_val.py` (1d), `train_linux.py` (1b)
- `config_probe_bs32.yaml` (Wegwerf-Probe, loeschbar)
- Diese Notiz: `TASK16A_1B_ABSCHLUSSBERICHT.md`


## 12. NACHTRAG (2026-07-13): Strategie-Entscheid Cluster-Hauptgleis

Nach Abschluss dieser Task wurden die 16b-Vorarbeiten (Loader-Diagnose Job
122811, Parallelitaets-Test Job 122812, geteiltes /dev/shm-Staging via
"job_shared") direkt angeschlossen — Ergebnisse in
`NOTIZ_16B_LOADER_DIAGNOSE.md`. Daraus gefallener Entscheid:

- **Alle Sweeps (16a.2/3/4) laufen auf dem CLUSTER** (Einzellauf-Gleichstand
  + 2.85x-Durchsatz bewiesen); lokal bleibt Fallback-/Debug-Gleis.
- 16a.2 wurde entgegen §10 NICHT lokal gestartet; die dortigen Uebergaben
  gelten sinngemaess mit Ausfuehrungsort Cluster. Der lambda_grad=0-
  Nullpunkt uebernimmt zusaetzlich die Rolle der **Hardware-Bruecke**
  lokal->Cluster (Kriterien aus §8 unveraendert; faellt er aus dem Band:
  STOPP vor weiteren Werten).
- 16b wechselt von "parallele Wette" zu **Voraussetzung fuer 16a.2**;
  Restbausteine (6-Term-Sweep-Basis-Config, Parallel-Sweep-sbatch,
  Auswerte-Kette inkl. --no-save-Patch) sind in der Notiz spezifiziert.

Die Ergebnisse und Schlussfolgerungen der Abschnitte 1-11 bleiben
unveraendert gueltig.
