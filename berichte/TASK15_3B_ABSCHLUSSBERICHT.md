# TASK 15.3B — ABSCHLUSSBERICHT (kombiniert: 15.3B.1 + 15.3B.3 + 15.3B.2)

**Titel:** Daten-Pipeline-Umbau: Memmap-Packing, node-lokales /dev/shm-Staging, float16-Storage
**Datum:** 2026-07-12
**Umgebungen:** alre-server-u20 (TITAN RTX 24GB, Python 3.8.10, kein conda) + DGX-Cluster tas-dgx-head (4 Knoten x 8x A100-80GB, conda "bevwm", BeeGFS)
**Status:** ABGESCHLOSSEN — alle drei Unterschritte verifiziert; strategischer Kernbefund dokumentiert


## 1. Aufgabe und Reihenfolge-Entscheidung

15.3B war ein Infrastruktur-Einschub zwischen 15.3 und den Regler-Sweeps
(inzwischen Task 16a): die Daten-Pipeline schneller/kleiner machen, motiviert
durch die Cluster-Migration (urspruenglich 22-35% GPU-Auslastung im
.npy-Modus ueber BeeGFS).

Geplante Reihenfolge war 15.3B.1 (Packing) -> 15.3B.2 (float16) -> 15.3B.3
(Staging). Wurde bewusst UMGEDREHT zu .1 -> .3 -> .2, aus zwei Gruenden:
(a) saubere Attribution — Staging ist verlustfrei, float16 nicht; getrennt
und in dieser Reihenfolge ist jede mIoU-Abweichung eindeutig zuordenbar;
(b) der Cluster-Befund (data% ~89% TROTZ Packing) zeigte, dass Staging der
eigentliche Hebel ist und float16 primaer dessen Enabler (halber --mem).


## 2. 15.3B.1 — Memmap-Packing [BESTANDEN]

**Bau:** `pack_latents.py` packt alle .npy-Latents eines Splits in EIN
memmap-taugliches `packed.npy` + `manifest.json` (config_hash,
scene_order_hash, verify-Ergebnis). Idempotent (Skip wenn vorhanden,
`--force` zum Neubau). Frame-Reihenfolge kommt aus einer echten
BEVLatentDataset-Instanz (`iter_pack_order()`) — eine Quelle der Wahrheit,
kein separates Ordering. `bev_dataset.py` bekam den packed-Zweig
(lazy per-Worker-Open, fork-sicher, Frame-Zahl-Guard), `bev_dataloader.py`/
`train_linux.py` den `data.use_packed`-Schalter mit `ensure_packed()` beim
Start.

**Innerer Check (Pflicht, automatisch je Pack-Lauf):** 500-Frame-Stichprobe
.npy vs. packed. float32: **bit-identisch, max|Delta| = 0.000e+00, 500/500
exakt** — lokal UND auf BeeGFS. (float16: "Rundungsfehler erwartet",
siehe §5.)

**Pack-Kosten (einmalig):** lokal val 6019 Frames/101GB in 2.4min, train
28130 Frames/471.9GB in 15.2min (NVMe). Cluster analog auf BeeGFS.

**Aeusserer Anker (Job A, Cluster, use_packed=true, Seed 42):** mIoU
0.5698 (E0) -> 0.6406 (E4) -> 0.6623 (E9), Val-Loss monoton fallend bis
E12; durch 12h-Zeitlimit bei ~E13 beendet — ausreichend, da der innere
Check die Datengleichheit bereits garantiert und der Verlauf exakt dem
lokalen Muster folgt. **Damit gilt: gepackte Pipeline == ungepackte
Pipeline.**

**Timing-Nebenbefund Job A:** data% ~89% (E5-E12: 88.7-90.4), Epochenzeit
~3000-3400s — die A100 wartet fast nur. Das war die empirische
Rechtfertigung, 15.3B.3 vom "optionalen Nachgedanken" zum Kern-Schritt
hochzustufen. Der geplante Job B (Timing-Vergleich unpacked) wurde nach
OOM-Neustarts und gegenseitiger Knoten-Interferenz verworfen — seine eine
saubere Epoche (1489s, data% 74.9 im .npy-Modus, waehrend Job A idle war)
diente als grober Referenzpunkt, ein formaler A/B-Vergleich auf geteiltem
Knoten ist ohnehin nicht sauber messbar.


## 3. 15.3B.3 — node-lokales /dev/shm-Staging [BESTANDEN, mit ernuechterndem Befund]

**Bau:** `shm_staging.py` (neues Modul): `stage_and_rewrite(cfg)` kopiert
packed.npy+manifest einmalig BeeGFS -> /dev/shm/<user>_<jobid>/ und biegt
die `*_packed_dir`-Keys der Config in-place um; Groessen-Vorabpruefung
(fail-fast bei vollem Knoten); idempotenter Skip. Cleanup-Strategie
MODULAR ueber `CLEANUP_STRATEGIES`-Registry — implementiert Ansatz A
("job_local": jeder Job loescht nur sein eigenes Verzeichnis), Ansatz B/C
(geteilte Kopie mit Refcount / manuell) als Stubs fuer 16b vorbereitet.
Cleanup dreifach abgesichert: train_linux.py am normalen Ende, sbatch-trap
bei JEDEM Skript-Exit (greift auch bei scancel/OOM/Timeout — **verifiziert:
nach OOM-Kill von Job 122757 hat der trap sauber geloescht**), Sicherheitsnetz
im Code (loescht nie ausserhalb /dev/shm). Config: `data.stage_to_shm`,
`data.shm_cleanup`.

**Node-lokale Fakten (gemessen, ersetzen frueheres Raten):**
- /dev/shm auf RECHENKNOTEN: **1008G** (~50% der 2TB RAM). Auf dem
  KOPFKNOTEN nur 63G — Groessen-Checks IMMER auf dem Rechenknoten machen.
- Alles in /dev/shm zaehlt gegen das SLURM-Cgroup---mem-Limit -> fp32
  braucht --mem ~700G, fp16 ~400G. Das erklaert rueckwirkend auch das alte
  "RAM wird nicht voll"-Raetsel (Cgroup-Limit verhinderte Kernel-Caching;
  tmpfs allokiert explizit statt opportunistisch).
- **BeeGFS-Lesedurchsatz direkt gemessen** (der Staging-Kopiervorgang selbst):
  471.9GB in 870-976s = **0.48-0.54 GB/s**, val 101GB mit 0.30-0.57 GB/s.
  Das ist der harte Beleg des Engpasses — langsamer als eine einzelne SATA-SSD.
- dd-Mikrobenchmark /dev/shm-Read auf freiem Knoten: ~4.8 GB/s single-stream
  (Unterschaetzung, tmpfs skaliert mit parallelen Workern).

**Ergebnis:** data% fiel von ~89% (BeeGFS) auf **~74%**, Epochenzeit von
~3100s auf **~1370s** (~2.3x). Deutlich, aber NICHT der erhoffte
Groessenordnungssprung. Anschliessender Test num_workers 4->12: **kein
Effekt** (data% blieb ~74%, PyTorch warnt ohnehin bei >8) -> CPU-Parallelitaet
ist nicht der Engpass. batch_size-8->32-Diagnose wurde aufgesetzt, der Lauf
kam wegen Queue-Blockade nie durch (siehe §7); die Frage ist an 16a.1B
(Beschleunigungs-Einschub) uebergeben.

**Interpretation (Kernbefund):** Der verbleibende data%-Sockel ist
strukturell — die A100 erledigt Forward/Backward des 6M-Modells so schnell,
dass der DataLoader (egal wie schnell die Quelle) nicht nachkommt. Siehe §6.


## 4. Beifang: mIoU-Plateau-Early-Stopping [GEBAUT, GETESTET, SCHARF]

Zusaetzliches Kill-Kriterium neben Val-Loss-Patience (ersetzt sie nicht):
stoppt, wenn `miou_patience` Decode-Messungen in Folge keinen Fortschritt
> `miou_min_delta` (=0.014, 15.2-Rauschboden) ueber einer SEPARATEN
Referenz `miou_stop_ref` bringen. Der separate Referenzwert ist essenziell:
Vergleich gegen `best_miou` wuerde durch 0.001-Mini-Verbesserungen den
Zaehler ewig zuruecksetzen. `best_miou.pt` bleibt immer der BESTE
Checkpoint, nie der vom Abbruchzeitpunkt. Greift nur bei aktivem Decode.

Simulationstest ergab: patience=2 + strenge 0.014-Schwelle killt zu
aggressiv (unter-Rauschboden-Schritte in Folge = Kill trotz Aufwaertstrend)
-> **Default patience=3**, config-steuerbar (`miou_early_stop`/
`miou_patience`/`miou_min_delta`). `decode_every` von 5 auf 3 gesenkt
(feinere Plateau-Aufloesung; Decode-Mehrkosten < gesparte Plateau-Epochen).
Erste echte Plateau-Meldung im fp16-Lauf beobachtet ("1/3 ... ueber 0.6461"
bei E8) — Verhalten wie designed.

**Realtest bestanden (Job 122767, erster kompletter Selbst-Stopp):**
Sequenz E11 (0.6699, Plateau 1/3) -> E14 (0.6750, 2/3 — Delta zur Referenz
0.6612 war 0.0138, haarscharf unter der 0.014-Schwelle) -> E17 (0.6743,
3/3 -> Stopp). Drei Design-Aspekte im Feld bestaetigt: (a) `best_miou.pt`
wurde waehrend des Zaehler-Hochlaufs korrekt WEITER aktualisiert (E11, E14)
— der gesicherte Checkpoint ist der beste, nicht der letzte; (b) der
separate `miou_stop_ref` verhinderte, dass die Mini-Verbesserungen den
Zaehler zuruecksetzen (mit Vergleich gegen best_miou waere der Lauf bis E50
gekrochen); (c) Val-Loss fiel bei E17 NOCH (neuer best_val_loss!) — die
Val-Loss-Patience haette nicht gestoppt, nur das mIoU-Kriterium griff.
Ersparnis ~17-33 Epochen (~7-13h Cluster-Zeit) fuer ~0.005 mIoU unter dem
theoretischen Optimum — unterhalb des Rauschbodens, wissenschaftlich
kostenlos. Doppel-Cleanup verifiziert: train_linux.py loeschte
/dev/shm/vima_122767, der trap meldete danach idempotent "Nichts zu
loeschen".

Wissenschafts-Vermerk fuer die Arbeit:
Sweeps laufen MIT Plateau-Stopping (verkuerzt), der finale Headline-mIoU
OHNE (voller Lauf) — Unterschied im Bericht transparent machen.


## 5. 15.3B.2 — float16-Storage [BESTANDEN: verlustfrei genug]

**Mechanik:** float16 existiert NUR als Speicherformat. `_load_latent`
castet beim Lesen sofort auf float32 zurueck — Modell, Loss und Seg-Decoder
sehen nie ein fp16 (BEVFusion-Kompatibilitaet damit trivial gegeben).
Roundtrip-Fehler auf realistischen Latents (mean~0.14, std~0.27):
max|Delta| = 2.4e-04. Nebenbei-Fix: bei float32-Storage ersetzt
`np.array(row)` das unnoetige `.astype("float32")` (reine Kopie statt
dtype-Durchlauf, bit-identisch verifiziert); bei fp16 bleibt der Upcast.
**Falle dokumentiert:** fp16-Packs brauchen EIGENE `packed_dir`-Pfade
(`latents_packed_fp16/`) — sonst sieht der Skip-Check den fp32-Pack als
"schon da" und der fp16-Pack entsteht nie.

**mIoU-Vergleich (der eigentliche 15.3B.2-Check), Rauschboden ±0.014:**

| Decode-Punkt | fp32-Anker | fp16 lokal | fp16 Cluster |
|---|---|---|---|
| E0 | 0.5698 | 0.5698 | (0.5698-aequiv.) |
| E2 | — | 0.6209 | 0.6203 |
| E4/E5 | 0.6406 (E4) | 0.6461 (E5) | 0.6430 (E5) |
| E8/E9 | 0.6623 (E9) | 0.6586 (E8) | 0.6612 (E8) |
| konvergiert | ~0.68 (E34-37, Task 14) | (abgebrochen E9) | **0.6750 (E14, Selbst-Stopp E17)** |

Der Cluster-Lauf (Job 122767) lief bis zum mIoU-Plateau-Selbst-Stopp durch:
bestes mIoU **0.6750 @ E14** — mitten im Rauschband des Anker-Ziels
(0.6798 +/- 0.014 -> Band 0.666-0.694). Damit ist die fp16-Validierung mit
konvergiertem Endwert abgeschlossen, nicht nur ueber den Verlauf.

Differenzen ~0.004 — **weit unter dem Rauschboden**. fp16 ist verlustfrei
genug; halbe Groesse (train 440->220GB, val 95->48GB) und halber
--mem-Bedarf (~700->400G) sind geschenkt. Epochenzeit lokal fiel von ~880s
auf **~780s** (kombinierter Effekt fp16-Bytes + astype-Fix, nicht einzeln
attribuiert). Lokal data% nur 16-21% -> lokal ist die GPU der Engpass,
das Cluster-I/O-Problem existiert dort nicht.


## 6. Strategischer Kernbefund: Cluster != schnellere Einzellaeufe

Alle Massnahmen zusammen (Packing + /dev/shm + fp16) bringen den Cluster
auf ~1370-1580s/Epoche bei data% 74-76% — der lokale Server schafft
~780s bei data% 16-21%. Die wandb-Summary des finalen Laufs macht es
greifbar: compute_time 299s vs. data_time 847s pro Epoche — die A100
RECHNET 5 Minuten und WARTET 14. **Der Cluster ist pro Einzellauf ~2x LANGSAMER,
trotz staerkerer GPU.** Ursache: 6M-Parameter-Modell auf A100 ist
Hardware-Overkill; die GPU wartet strukturell auf Daten. Literatur-Deckung:
LeWorldModel (~15M Parameter) trainiert bewusst auf einer einzelnen L40S
in 10 Epochen — kleine World Models gehoeren auf kleine GPUs.

**Konsequenz (in den Arbeitsplan eingeflossen):** Zwei-Gleis-Strategie.
Lokal (16a) = verlaessliches UND schnelleres Gleis fuer Einzellaeufe;
Cluster (16b) = reiner Parallelitaets-Kandidat (viele Sweeps gleichzeitig,
/dev/shm als geteilte Node-Kopie macht das erst moeglich), plus neu
vorgemerkt MIG/MPS-GPU-Teilung. Einzellauf-Beschleunigung laeuft als
16a.1B-Einschub (Profiling, channels_last, SDPA, batch/LR) weiter.


## 7. Betriebs-Lektionen (Cluster)

- **--mem IMMER explizit setzen.** Default = ganzer Knoten -> Job wartet ewig.
- **persistent_workers verdoppelt die Worker** (Train- UND Val-Loader leben
  gleichzeitig): num_workers=8 heisst 16 Prozesse. Das killte Jobs
  122729/122730 bei --mem=64G per OOM (Job A ausgerechnet in der
  Val-Phase). Fix: num_workers=4 + --mem=200G fuer normale Laeufe.
- --mem-Staffel Staging: fp32 700G (Job 122757 wurde auf teilbelegtem
  Knoten trotzdem ge-OOM-t), fp16 400G — kam durch die Queue.
- **Queue-Realitaet:** 4 Knoten (nicht 8 — 8 ist GPUs/Knoten), oft
  tagelang durch andere Nutzer belegt; PD(Resources/Priority) ueber
  Stunden ist normal. Grosse --mem-Anforderungen warten laenger; fp16
  entschaerft genau das.
- `srun --jobid=<id>` in einen LAUFENDEN Job einklinken ist riskant
  (Ressourcen-Kollision, haette Job A fast gekillt) — Diagnose-Kommandos
  als eigenen Mini-srun-Job auf anderem Knoten fahren.
- Ctrl-C auf `tail -f` killt nur die Ansicht, nie den Job.
- scp ueberschreibt ohne Rueckfrage; laeuft immer von der Maschine aus,
  auf der die QUELLE liegt.

## 8. Config-Schema-Fund (nachtraeglich, unschaedlich fuer 15.3B)

Die fp16-Configs trugen im training:-Block noch das alte 3-Term-Schema
(`lambda_cos/lambda_dist/lambda_ssim`); `lambda_dist` ist seit 15.1 ein
toter Key, `lambda_mean/std/grad` fehlten und fielen auf Defaults. **Fuer
die 15.3B-Laeufe folgenlos**, weil die Defaults exakt die Baseline-Loss
ergeben (Log-Zeile bestaetigt: `1.0*MSE + 0.1*CosSim + 0.1*Mean + 0.1*Std
+ 0.0*Grad + 0.1*SSIM` — identisch zum fp32-Anker, Vergleich gueltig).
Fuer die Sweeps untauglich (Runner kann fehlende Zeilen nicht patchen) —
in 16a.0 durch eine bereinigte 6-Term-Basis-Config behoben.


## 9. Offene Punkte

- 364-vs-700-Train-Szenen-Diskrepanz: weiterhin ungeklaert, konsistent
  lokal+Cluster, nicht blockierend.
- /raid (node-lokales NVMe, wuerde Kopieren pro Job eruebrigen): weiter
  root-only, Ansprechpartner unklar.
- shm-Cleanup Ansatz B/C (geteilte Kopie fuer parallele Jobs): Stubs
  vorhanden, Aktivierung bei 16b-Parallelitaetstest.
- inference.py: vermuteter code/Code-Case-Sensitivity-Bug (fuer std-Ratio-
  Harvest relevant) — in 16a.0 adressiert.
- [ERLEDIGT waehrend Berichterstellung] fp16-Cluster-Lauf (Job 122767):
  sauber per mIoU-Plateau selbst gestoppt (E17), bestes mIoU 0.6750 (E14)
  im Rauschband des Ankers -> 16b Schritt (1) damit abgehakt (fp16+Staging
  mit --mem=400G kam durch die Queue und lief fehlerfrei End-to-End).


## 10. Deliverables (Dateien)

Code: `pack_latents.py` (neu), `shm_staging.py` (neu), `bev_dataset.py`
(packed-Zweig + astype-Fix), `bev_dataloader.py` (packed_dir-Threading),
`train_linux.py` (use_packed/ensure_packed, stage_to_shm-Hooks,
mIoU-Plateau-Stopping).
Configs: `config_nuscenes_full_cluster.yaml` (fp32-Anker),
`_cluster_nopack.yaml` (Timing-B, verworfen), `_cluster_shm.yaml`
(fp32-Staging), `_cluster_fp16.yaml` (fp16+Staging, --mem 400G),
`config_nuscenes_full_fp16.yaml` (lokal fp16 + Plateau-Stopping).
SLURM: `sbatch_anchor.sh`, `sbatch_timing_compare.sh`, `sbatch_shm_stage.sh`
(mit Cleanup-trap), `sbatch_fp16_cluster.sh`.
Checkpoints: `task15_3B1_anchor/`, `task15_3B3_shm/`, `task15_3B2_fp16/`
(lokal), `task15_3B2_fp16_cluster/`.
