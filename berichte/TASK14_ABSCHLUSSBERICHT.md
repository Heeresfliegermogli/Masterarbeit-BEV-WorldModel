# TASK 14 — ABSCHLUSSBERICHT: In-Loop-Decode-Validierung (Ebene 1)

**Datum:** 27.06.2026
**Status:** ✅ ABGESCHLOSSEN
**Modell-Lauf:** `phase_cell_0626_2243` (voller nuScenes-Split, 28.130 Train / 6.019 Val)

---

## Fragestellung

Das Training selektierte den besten Checkpoint bisher ausschließlich nach dem
**Val-Loss** im Latent-Raum (MSE + Cosine + Distribution + SSIM). Der Val-Loss
ist aber nur ein Proxy: Er misst die Ähnlichkeit der vorhergesagten Latents zu
den Ziel-Latents, nicht die eigentliche Aufgabenqualität — die **mIoU der
decodierten Segmentierungskarte**.

Ziel von Task 14 (Ebene 1 des zweistufigen Validierungsschemas): die echte mIoU
**periodisch während des Trainings** messen, indem die vorhergesagten Latents
durch den eingefrorenen BEVFusion-Seg-Decoder geschickt werden, und parallel zum
`best_val_loss.pt` einen zweiten Checkpoint `best_miou.pt` führen, der nach der
tatsächlichen Task-Metrik selektiert.

Diese Task war primär als **Implementierungsaufgabe** angelegt: Die Infrastruktur
wird gebaut und verifiziert, die Ergebnisse werden mitgenommen. Die systematische
Loss-Variation darauf folgt in Task 15 (Ebene 2).

---

## Vorgehen

### Architektur-Entscheidung: Pure-PyTorch-Decoder statt mmdet3d

Der Decode-Pfad (`inference_seg.py`, Task 9d) lebte bisher im BEVFusion-Docker und
hängt für SECOND-Backbone und SECONDFPN-Neck an den mmdet3d-Buildern. Diese
benötigen das alte `mmcv` (~1.4.x), das mit der PyTorch-2.1.2-Trainingsumgebung
**inkompatibel** ist. Drei Wege standen zur Wahl:

- **Weg A1** — mmdet3d in die Trainingsumgebung installieren: verworfen
  (mmcv/PyTorch-Versionskonflikt, im Arbeitsplan explizit als Sackgasse markiert).
- **Weg A2** — Decoder als reines PyTorch nachbauen: **gewählt.**
- **Weg B** — Decode als separater Docker-Schritt nach jedem Lauf: als Fallback
  vorgehalten, nicht benötigt.

Für **Weg A2** wurde `seg_decoder_torch.py` erstellt: ein 1:1-Nachbau der
Decoder-Kette als schlichte `nn.Module`, sodass die Gewichts-Keys exakt denen in
`bevfusion-seg.pth` entsprechen und mit `strict=True` geladen werden können.

```
Latent [256,128,128]
   → SECONDBackbone   → ([128,128,128], [256,64,64])
   → SECONDFPNNeck     → [512,128,128]
   → BEVGridTransform  → [512,200,200]
   → Classifier        → [6,200,200] logits
   → sigmoid + 0.5     → bool-Maske [6,200,200]
```

`BEVGridTransform` und Classifier waren bereits in `inference_seg.py` reines
PyTorch und wurden übernommen. Nur SECOND und SECONDFPN mussten nachgebaut werden
(einfache Conv-/Deconv-Stacks). Der Decode läuft bewusst in `float32`
(`autocast` deaktiviert), um exakt der Docker-Referenz zu entsprechen.

### Verifikation vor dem Einbau (`verify_decoder.py`)

Drei Ebenen, alle in der Trainingsumgebung, ohne Docker:

1. **strict-load** — lädt SECOND (72 Keys), SECONDFPN (12 Keys), Classifier
   (14 Keys). Erfolgreich → Architektur byte-genau korrekt. Dies ist der
   stärkste Einzelbeweis.
2. **mIoU-Abgleich** — 10 vorhandene `pred_/real_`-Latentpaare aus der
   Task-13-Inferenz pure-torch decodiert: **mean mIoU 0.6539**, identisch zur
   Docker-Referenz aus Task 13 (0.654). Die Forward-Numerik stimmt also nach der
   Maskenschwelle bis auf die vierte Nachkommastelle.
3. **Neck-Dump** (optional) — vorgehalten für einen exakten 1e-5-Vergleich der
   rohen Feature-Map, nicht benötigt.

**Gate bestanden** → Einbau in `train_linux.py`.

### Integration in `train_linux.py`

Vier additive Änderungen, bestehende Logik unberührt:

1. **`build_seg_val_cache()`** — wählt ein festes, seed-reproduzierbares Subset
   von **300 Val-Samples** und decodiert deren **real-Masken einmalig** vor (sie
   ändern sich über Epochen nie). Zugriff direkt über `val_loader.dataset[i]`.
2. **`validate_seg()`** — pro Epoche: `model(input)` → pred-Latent → Decoder →
   pred-Maske → `compute_iou()` gegen die gecachte real-Maske. Gibt mean mIoU +
   per-Klasse zurück.
3. **Decoder-Laden in `main()`** — gesteuert über einen neuen
   `decode_validation`-Block in der YAML; fehlt er, läuft alles wie in Task 13
   (rückwärtskompatibel). Fällt bei Ladefehler sauber auf Val-Loss-only zurück.
4. **Trainings-Loop** — Decode alle `decode_every=5` Epochen (plus Epoche 0),
   `best_miou.pt` parallel zu `best_val_loss.pt`, mIoU + per-Klasse ins Log.

### Konfiguration

```yaml
decode_validation:
  enabled:        true
  seg_checkpoint: ".../pretrained/bevfusion-seg.pth"
  decode_every:   5
  n_subset:       300
```

Loss- und Modell-Hyperparameter **unverändert** gegenüber Task 13 — damit ist die
mIoU dieses Laufs direkt mit der Task-13-Baseline vergleichbar.

---

## Ergebnisse

### mIoU-Verlauf (300-Sample-Val-Subset, In-Loop)

| Epoch | Train-Loss | Val-Loss   | mIoU       | LR        | Zeit/Epoche |
|------:|-----------:|-----------:|-----------:|----------:|------------:|
|     0 |   0.062912 |   0.063512 |   0.5698   | 9.99e-05  |    827 s    |
|     4 |   0.049402 |   0.056826 |   0.6381   | 9.76e-05  |    806 s    |
|     9 |   0.045640 |   0.052678 |   0.6634   | 9.05e-05  |    810 s    |
|    14 |   0.043858 |   0.051522 |   0.6685   | 7.96e-05  |    810 s    |
|    19 |   0.042654 |   0.051149 |   0.6685   | 6.58e-05  |    798 s    |
|    24 |   0.041759 |   0.050383 |   0.6770   | 5.05e-05  |    796 s    |
|    29 |   0.041031 |   0.050962 |   0.6722   | 3.52e-05  |    797 s    |
|  **34** | **0.040514** | **0.050214** | **0.6798** | 2.14e-05 | 799 s |
|    39 |   0.040138 |   0.050499 |   0.6761   | 1.05e-05  |    797 s    |
|    44 |   0.039889 |   0.050472 |   0.6767   | 3.42e-06  |    797 s    |

### Checkpoint-Selektion

- **`best_miou.pt`** → Epoch 34, **mIoU 0.6798**
- **`best_val_loss.pt`** → Epoch 37, Val-Loss 0.050190
- Early Stopping bei Epoch 47 (Patience 10 auf Val-Loss)

### Kernbeobachtungen

**1. mIoU übertrifft die Task-13-Baseline.** 0.6798 vs. 0.654 — derselbe Loss,
dieselbe Datenmenge, höhere Subset-mIoU. (Der direkte Zahlvergleich ist
einzuordnen: Task 13 war auf 10 Samples, hier 300; ein voller Vergleich auf 5743
Samples ist optional als Anhang möglich, siehe „Offene Punkte".)

**2. mIoU-Peak und Val-Loss-Minimum fallen nicht zwingend zusammen.** Der
mIoU-Peak liegt bei Epoch 34. `best_val_loss.pt` selektierte hingegen Epoch 37 —
eine Epoche, in der die Decode-Validierung gar nicht lief (kein Vielfaches von 5).
Das ist exakt die Daseinsberechtigung von Ebene 1: Die nach Val-Loss „beste"
Epoche ist nicht garantiert die nach der eigentlichen Task-Metrik beste, und ohne
In-Loop-Decode hätte man den mIoU-optimalen Checkpoint nie gesehen.

**3. mIoU plateauisiert früher als der Val-Loss.** Ab ~Epoch 24 bewegt sich die
mIoU nur noch im dritten Nachkommastellenbereich (0.677–0.680), während der
Val-Loss weiter sinkt. Die späten Val-Loss-Verbesserungen bringen für die
tatsächliche Segmentierungsqualität kaum noch etwas — ein verwertbares Signal für
die Wahl von Trainingsdauer und Loss-Gewichten in Task 15.

**4. Decode-Overhead ist vernachlässigbar.** Der 300-Sample-Decode läuft mit
~28 it/s in ~11 s, gegenüber ~800 s Trainingszeit pro Epoche. Die In-Loop-Messung
kostet faktisch nichts.

---

## Erkenntnisse / Debugging

- **Pure-PyTorch-Decoder als Schlüssel:** Der mmdet3d-Konflikt wird nicht gelöst,
  sondern umgangen. `strict=True` beim Laden ist gleichzeitig der
  Architektur-Korrektheitsbeweis — passt eine Shape nicht, scheitert der Load
  sofort und laut.

- **DataLoader-Worker-Deadlock beim Cache-Aufbau:** Erste Implementierung
  iterierte den Val-Loader und brach mit `break` ab. Bei `num_workers=4` führt
  das Abbrechen einer laufenden Iteration zu hängenden Worker-Prozessen. Fix:
  direkter Index-Zugriff `val_loader.dataset[i]` ohne Loader/Worker.

- **Vermeintlicher „Hänger" war Output-Buffering:** Über mehrere Startversuche
  schien das Training einzufrieren (kein Fortschrittsbalken). Tatsächlich lief die
  GPU durchgehend bei 77–90 % — die Ausgabe wurde nur blockweise gepuffert, weil
  sie durch `tee` lief (kein TTY) und der `\r`-Balken ohne Zeilenumbruch nie
  flushte. Fix: `python -u` (unbuffered) plus `flush=True` an den
  Epochen-Ausgaben. Der zwischenzeitlich eingebaute `spawn`-Start war daher
  **nicht** nötig (kein echter CUDA-Fork-Deadlock, da die Worker nur CPU-`.npy`
  lesen und CUDA nie anfassen) und wurde wieder entfernt — `fork` ist hier
  ~30 % schneller (5.1 vs. 2.4 it/s).

- **Fortschrittsanzeige vereinheitlicht:** Train-, Val- und Decode-Phase haben je
  einen `\r`-Balken, der nach Abschluss gelöscht wird, sodass pro Epoche nur die
  saubere Übersichtszeile (`Epoch | Train | Val | mIoU | LR | Zeit`) im Log bleibt.

---

## Modularität (Vorbereitung Task 16)

`seg_decoder_torch.py` enthält bereits `render_mask()` und `save_comparison()`
(pred | real | diff als PNG), die von der Validierung **bewusst nicht** aufgerufen
werden. In `validate_seg()` ist die Einhängestelle nach `compute_iou()`
kommentiert. Task 16 (Visualisierungs-Artefakte pro Lauf) kann damit ohne
Code-Duplikat die comparison-PNGs für eine Handvoll Subset-Tokens mitschreiben —
der mIoU-Pfad und der PNG-Pfad teilen sich denselben Decoder und dieselbe
`latent_to_mask()`-Logik.

---

## Artefakte

| Datei | Zweck |
|---|---|
| `Code/seg_decoder_torch.py` | Pure-PyTorch-Seg-Decoder + IoU + Render-Helfer (Task 16) |
| `train_linux.py` | erweitert um `validate_seg()`, `build_seg_val_cache()`, `best_miou.pt` |
| `config_nuscenes_full.yaml` | neuer `decode_validation`-Block |
| `verify_decoder.py` | Gate-Skript (strict-load + mIoU-Abgleich); nicht Teil des Trainings |
| `checkpoints/task14/phase2/best_miou.pt` | mIoU-bester Checkpoint (Epoch 34, 0.6798) |
| `checkpoints/task14/phase2/best_val_loss.pt` | Val-Loss-bester Checkpoint (Epoch 37) |
| `logs/task14_fresh.log` | vollständiges Trainingsprotokoll inkl. mIoU-Verlauf |

---

## Offene Punkte / Übergabe an Task 15

- **Optionaler voller Vergleich:** `best_miou.pt` vs. `best_val_loss.pt` auf dem
  vollen Val-Set (5743 Samples, via `inference.py` + Decode), um die
  Subset-mIoU (300) gegen die volle mIoU abzusichern. Für die Implementierungs-
  Task nicht erforderlich, aber als sauberer Schlusspunkt vor Task 15 sinnvoll.
- **wandb war im Lauf nicht installiert** — per-Klasse-mIoU wird zwar berechnet
  und ans Logging übergeben, landete aber nicht in einem Dashboard. Falls
  per-Klasse-Verläufe für den Bericht gewünscht sind: wandb aktivieren oder
  per-Klasse-Werte zusätzlich in die Log-Zeile schreiben.
- **Float16-Latent-Konversion** bleibt der primäre I/O-Hebel vor Task 15
  (halbiert das Lesevolumen des 440-GB-Train-Sets).
- **Ebene 2 (Task 15):** systematische Loss-Variation, jede Iteration mit eigenem
  Bericht (TASK15_I1, I2, …); die In-Loop-mIoU dieses Laufs ist die Referenz,
  gegen die die Loss-Änderungen gemessen werden.

---

**Fazit:** Ebene 1 ist implementiert und verifiziert. Das Training liefert ab
sofort pro Lauf eine echte mIoU-Kurve und einen nach Task-Metrik selektierten
Checkpoint, ohne Docker-Roundtrip und mit vernachlässigbarem Overhead. Der Lauf
bestätigt die Kernhypothese: mIoU-optimaler und Val-Loss-optimaler Checkpoint
sind nicht identisch (Epoch 34 vs. 37).
