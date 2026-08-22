# Zwischenbericht — Loss-Engineering-Konzeption & Cluster-Migration

**Zeitraum:** Juli 2026
**Status:** Konzeption Task 15 abgeschlossen, Cluster-Pipeline noch nicht produktionsreif (Task 15B aussteht)

---

## 1. Kontext

Zwei parallele Arbeitsstränge in diesem Zeitraum: (a) die konzeptionelle Ausarbeitung von Task 15 (Loss-Engineering) unter Einordnung des LeWorldModel-Papers, und (b) die Migration des Trainings auf den DGX-A100-Cluster, die einen unerwarteten, aber lehrreichen Umweg über eine grundlegende Dateninfrastruktur-Überarbeitung nahm.

---

## 2. Konzeption: LeWorldModel-Einordnung & Task-15-Struktur

**Zentrale Einordnung:** LeWorldModels Kerntechnik (SIGReg) ist auf unser Setup **nicht übertragbar** — sie bekämpft *Representation Collapse* bei gemeinsam trainiertem Encoder+Predictor; unser eingefrorener Encoder kann gar nicht kollabieren. Unser std-Gap ist stattdessen *Regression-to-the-Mean* (MSE-Minimierer = bedingter Erwartungswert unter aleatorischer Unsicherheit) — ein anderes Phänomen, das SIGReg falsch adressieren würde (zieht Richtung N(0,I), zerstört Decode-Kompatibilität).

**Was dennoch übernommen wurde:**
- Seed-Varianz-Methodik (mehrere Seeds, Fehlerbalken) → Task 15.2, Rauschboden
- Dosis-Wirkungs-Kurven-Darstellung (Metrik gegen λ) → Task 15.7
- Projektions-Mechanismus (Cramér-Wold) als *optionale* Erweiterung, aber auf die reale Latent-Verteilung gerichtet statt auf N(0,I) → Task 15.6 (Sliced-Wasserstein, optional)

**Task 15 wurde neu strukturiert** in 15.1–15.8, mit einer wichtigen Korrektur: **mIoU ist der Nordstern**, std-Ratio nur noch Manipulations-Check (nicht mehr Zielgröße — ein Modell kann std-Ratio durch Halluzination heben und dabei schlechtere mIoU erzielen). Kernprinzip: ein Regler pro Lauf, YAML-only-Änderungen, AvgPool/Architektur eingefroren über den gesamten Sweep.

| Term | Achse | Literatur-Bezug |
|---|---|---|
| lambda_mse | Anker (Magnitude) | Feld-Standard (LeWM, DINO-WM, PLDM, LAW) |
| lambda_cos | punktweise Richtung | eigene Ergänzung, kein Signaturterm im Feld |
| lambda_mean / lambda_std | Verteilung (1./2. Moment) | verwandt VICReg/SIGReg, aber gegen Regression-to-Mean statt Collapse, Ziel = echte Verteilung statt Prior |
| lambda_grad | räumliche Hochfrequenz | Gradient Difference Loss (Mathieu et al. 2016) |
| lambda_ssim | lokale Struktur | vermutlich verzichtbar (RAUSCH_REPORT, Notwendigkeits-Check in 15.5) |

**Feld-Verortung:** LAW/World4Drive dekodieren ihre Latent-Vorhersage nie (Planning-Metrik statt Rekonstruktions-Treue); BEVWorld dekodiert, erkauft Schärfe aber generativ (Diffusion). Unsere Kombination — deterministische Regression **und** Decode-Pflicht — ist im Feld unüblich; das erklärt strukturell, warum der std-Gap bei uns sichtbar wird, wo er anderswo verdeckt bleibt.

**Ergänzung 15.8 (optional):** automatisierte Lambda-Optimierung (Optuna), auf Wunsch eines Professors — mit fünf expliziten Leitplanken (nur nach 15.1–15.7, Suchraum auf 2–3 wirksame Lambdas beschränkt, Pruning zwingend, Ergebnis gegen 15.2-Rauschboden validieren, Zeit-Gate vor Terminierung), um nicht in PLDMs O(n⁶)-Falle zurückzufallen.

---

## 3. Cluster-Migration: Durchführung & zentrale Erkenntnis

**Infrastruktur stand:** Account, Code-Sync (Latents lagen bereits auf BeeGFS, nur Code/PKLs/Checkpoints übertragen), Miniconda-Environment (PyTorch 2.1.2+cu121), SLURM-Grundlagen. Wiederkehrende Stolpersteine dabei, kurz notiert: `--mem` muss immer explizit gesetzt werden (sonst Reservierung des kompletten Knotenspeichers → faktische Blockade); GRES-Typ ist `a100` (klein), nicht `A100`; NumPy muss auf `1.26.4` gepinnt werden (2.x bricht die Torch-NumPy-Interop); ein Code/code-Case-Mismatch in `smoke_test.py`/`inference.py`.

**Die zentrale, wichtigere Erkenntnis:** Das I/O-Problem, das die Migration überhaupt motivierte, **verschärfte sich auf dem Cluster**, statt sich zu lösen. GPU-Auslastung 22–35% (lokal: 60–70%), obwohl BeeGFS über Infiniband HDR-200 dem Papier nach deutlich potenter ist als die lokale Platte.

**Systematisch ausgeschlossene Fixes:**
- **Mehr `num_workers`** (4→16): kaum Wirkung — anders als lokal (dort Disk-Durchsatz-Limit) hier vermutlich Netzwerk-Metadaten-Rundlauf pro Datei-Open, der durch mehr Parallelität nicht sinkt.
- **RAM-Caching** (`--mem=600G`, Page-Cache): kein Effekt — `free -h` zeigte nur ~182GB Cache statt der erwarteten ~535GB trotz reichlich freiem RAM; vermutlich Cgroup-Reclaim auf dem geteilten Multi-Tenant-Knoten.
- **`/raid`-Node-Scratch**: nicht nutzbar — Schreibrechte fehlen (`root:root`, 755), nur Lesezugriff.

**Verifizierter Fix:** Memmap-Packing (viele Frames in eine zusammenhängende Datei statt tausender Einzeldateien). Per Mini-Benchmark mit Korrektheits-Check (identische Summen-Kontrolle über alle Varianten) sauber gemessen, nicht nur behauptet — explizit gegen eine Wiederholung des früheren HDF5-Fehlschlags abgesichert:

| Umgebung | Referenz (parallelisiert) | Memmap (seriell) | Faktor |
|---|---|---|---|
| Cluster (BeeGFS) | 8 Worker: 0,76s | 0,29s | **2,60×** |
| Lokal (NVMe) | 4 Worker: 0,32s | 0,23s | **1,39×** |

Der Unterschied zwischen beiden Faktoren bestätigt die Diagnose: der Effekt ist umso größer, je teurer der Netzwerk-Rundlauf pro Open ist.

---

## 4. Resultierende Entscheidungen (→ Task 15B, neu in den Arbeitsplan aufgenommen)

- **Ein gemeinsamer Codepfad** für lokal und Cluster (`data.packed_dir` als optionaler Schalter), keine zwei getrennt gepflegten Pipelines.
- **Ein Memmap pro Split** (train/val), nicht pro Szene und nicht pro Fenster — vermeidet sowohl unnötige Redundanz (Sliding-Windows überlappen, Stride 1) als auch die Notwendigkeit eines neuen Shuffle-Buffer-Samplers. Bestehendes `shuffle=True` bleibt unverändert → Retraining-Frage für bereits gelaufene 15.2/15.3-Läufe entschärft sich weitgehend.
- **Zweistufiger Korrektheits-Check** vor Übernahme: Stichproben-Gleichheit + Seed-42-Anker-Lauf gegen den 15.2-Rauschboden — dasselbe Verifikationsmuster wie bei 15.1.
- **Float16-Storage** als eigener Folge-Subtask (15B.2), bedingt auf 15B.1, mit Rundtrip-Fehleranalyse und eigenem Anker-Vergleich — nicht vorschnell übernommen, sondern separat empirisch geprüft.
- **Multi-GPU/DDP explizit verworfen** für dieses Problem: bei 22–35% GPU-Auslastung ist Daten-Hunger, nicht Rechenkapazität, der Engpass — mehr GPUs würden die I/O-Last nur vervielfachen. Der sinnvolle "Mehr-GPU"-Hebel liegt stattdessen auf **Job-Ebene**: unabhängige Sweep-Läufe (15.3/15.4, verschiedene Seeds/Lambdas) parallel auf mehreren freien GPUs starten, statt seriell — vorgemerkt für die Fortsetzung der Sweeps.

---

## 5. Aktueller Stand & nächste Schritte

- **Lokal:** unverändert funktionsfähig, Umbau dort mangels akutem Bedarf nachrangig, aber im vereinheitlichten Design ohne Zusatzkosten möglich.
- **Cluster:** Pipeline läuft Ende-zu-Ende (Smoke-Test bestanden), aber mit unzureichendem Durchsatz für produktive Sweeps.
- **Nächster Schritt:** Task 15B (Memmap-Packing-Umbau) in eigenem Chat, mit 15B.1 (Kern-Umbau) vor 15B.2 (Float16-Validierung). Erst nach bestandenem Korrektheits-Check Fortsetzung von Task 15.3+ auf dem Cluster unter der neuen, einheitlichen Pipeline.
