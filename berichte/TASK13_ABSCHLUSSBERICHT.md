# TASK 13 — ABSCHLUSSBERICHT: Neue Baseline auf dem vollen nuScenes-Datensatz

**Datum:** 23.06.2026
**Status:** ✅ ABGESCHLOSSEN

---

## Fragestellung

Training und Auswertung des BEV-World-Models auf der **vollen Datengrundlage**
aus Task 12 (offizieller nuScenes-Split: 28.130 Train- / 6.019 Val-Keyframes),
als Ersatz für die bisherige Val-only-Teilmenge mit internem 70/15/15-Split.

Zentrale Bedingung: **Loss-Funktion und Modell-Hyperparameter bleiben unverändert**
gegenüber der alten Baseline. Nur die Datenmenge ändert sich. Nur so ist der
Vergleich "alte Baseline vs. neue Baseline" sauber und die resultierenden
Metriken als eingefrorener Referenzpunkt für die nachfolgenden Loss-Experimente
(Task 15) verwendbar.

---

## Vorgehen

### Code-Anpassungen

1. **`make_dataloaders_explicit()`** (`bev_dataloader.py`) um optionale
   HDF5-Cache-Parameter pro Split erweitert (`train_h5_path`, `val_h5_path`,
   `test_h5_path`). Anders als bei `make_dataloaders_split()` (eine Quelle,
   intern gesplittet → ein gemeinsamer Cache) liegen hier bereits getrennte
   Train-/Val-PKLs vor → entsprechend je Split ein eigener Cache. Parameter
   sind optional (`None` = `.npy`-Direktzugriff), damit rückwärtskompatibel.

2. **`train_linux.py`**: Der `explicit`-Zweig in `build_dataloaders()` liest die
   neuen Pfade aus der YAML und reicht sie durch. Zusätzlich `prefetch_factor`
   aus der Config steuerbar gemacht (konditional nur bei `num_workers > 0`).

3. **Bugfix `validate()`** (`train_linux.py`): Die Funktion akkumulierte
   `total_ssim` nicht und gab nur 4 statt 5 Werte zurück, während der Aufruf in
   `main()` bereits 5 erwartete (`val_loss, val_mse, val_cos, val_dist, val_ssim`).
   Vorbestehender Bug, der erst beim ersten vollständigen Epochen-Abschluss zum
   `ValueError: not enough values to unpack` führte. Behoben.

4. **tqdm-Fortschrittsbalken** in `train_one_epoch()` und `validate()` ergänzt
   (mit Fallback ohne Crash, falls tqdm fehlt). Zeigt Live-Loss, Batch-Zähler
   und `it/s`. Diagnostisch entscheidend (siehe "Erkenntnisse").

### HDF5-Caches gebaut (Task-13, Schritt 2 — später verworfen)

Mit dem unveränderten `build_hdf5_cache.py` aus Task 1, zwei getrennte Dateien
aus den **`complete/`-PKLs** (korrekter Stand aus Task 12):

| Cache | Tokens | Unkomprimiert | HDF5 (gzip-4) | Kompression | Build-Zeit |
|---|---|---|---|---|---|
| Val   | 6.019  | 94,05 GB  | 27,80 GB  | 3,4× | 1.223 s |
| Train | 28.130 | 439,53 GB | 129,62 GB | 3,4× | 5.593 s |

Beide automatisch via `verify_cache()` geprüft: vollständig, korrekte
Latent-Shape `(256, 128, 128)`.

> **Hinweis zur Pfad-Falle:** Unter `Nuscenes/` lag noch eine veraltete
> `nuscenes_infos_val.pkl` (alter Radar-Fork-Stand ohne `location`-Feld). Der
> erste Val-Cache wurde versehentlich daraus gebaut und neu erstellt, nachdem
> die alte PKL entfernt und alle Pfade auf `complete/` umgestellt wurden.

### Training

```bash
python train_linux.py --config config_nuscenes_full.yaml --phase cell
```

- Modus: `explicit`, **`.npy`-Direktzugriff** (kein HDF5 — siehe Erkenntnisse)
- 50 Epochen geplant, `patience=10`, Loss = MSE + 0.1·CosSim + 0.1·Dist + 0.1·SSIM
- Unverändert: `lr=1e-4`, `weight_decay=1e-4`, `d_model=256`, `n_layers=4`,
  `n_heads=8`, 3 Input-Frames, Phase `cell` (3.072 Tokens/Forward)
- batch_size 8 (Test gegen bewährte 4 — ohne Effekt, siehe unten)

**Verlauf:** Loss sank glatt und monoton. **Early Stopping bei Epoche 47**
(kein Val-Fortschritt seit Epoche 37). Konstant ~817 s/Epoche.

- **Bester Val-Loss: 0.050189** (Epoche 37)
- Train-Loss am Ende ~0.0399 → moderate Train/Val-Lücke, kein starkes Overfitting
- Checkpoint: `checkpoints/task13_baseline_full/phase2/best_val_loss.pt`

### Inference (voller Val-Split, 5.743 Sequenzen)

```bash
python inference.py --config config_nuscenes_full.yaml --phase cell \
  --checkpoint .../task13_baseline_full/phase2/best_val_loss.pt
```

### Seg-Decoder-Auswertung (BEVFusion, Docker)

10 repräsentative pred/real-Paare durch den BEVFusion-Seg-Decoder
(`bevfusion-seg.pth`, backbone + neck + map_head) → mIoU pro Klasse +
Visualisierungen. Container `bevfusion:correct`, `inference_seg.py` unverändert
aus Task 9d.

---

## Ergebnisse

### Latent-Metriken (5.743 Val-Sequenzen)

| Metrik | Alte Baseline (Val-only) | **Task 13 (voll)** | Δ |
|---|---|---|---|
| n_samples         | ~700    | **5.743**  | +720 % |
| mean cosine sim   | ~0,829  | **0,8549** | +0,026 ↑ |
| mean MSE          | ~0,036  | **0,0251** | −0,011 ↑ |
| pred_std (Ø)      | ~0,213  | 0,2093     | ≈ |
| real_std (Ø)      | ~0,290  | 0,2513     | — |
| **std-Ratio**     | ~0,73   | **0,833**  | **+0,10 ↑↑** |

**Kernresultat:** Die std-Ratio (pred_std / real_std) — der zentrale Indikator
für das bekannte "Regression-to-the-Mean"-Problem (zu glatte, varianzarme
Vorhersagen) — steigt deutlich von ~0,73 auf **0,833**. Das Modell lernt die
Latent-Verteilung spürbar besser, wenn es auf 7× mehr und vielfältigeren Szenen
trainiert, **ohne jede Änderung an der Loss-Funktion**. Auch cosine sim und MSE
verbessern sich, und das bei der ehrlicheren Aufgabe (Validierung auf komplett
ungesehenen Szenen statt in-distribution auf demselben kleinen Subset).

### Segmentierungs-Metriken (10 Samples, BEVFusion-Seg-Decoder)

| Klasse | mIoU | |
|---|---|---|
| drivable_area | **0,906** | dominante Klasse, durchgehend sehr gut |
| carpark_area  | 0,760 | gut |
| walkway       | 0,738 | gut |
| divider       | 0,621 | solide |
| ped_crossing  | 0,501 | mittel — kleine, seltene Klasse |
| stop_line     | 0,427 | schwächste — sehr dünne Strukturen |
| **mean mIoU** | **0,654** | |

Die Verteilung folgt dem erwarteten Muster: große, flächige Klassen werden sehr
gut rekonstruiert, feine/seltene Strukturen (stop_line, ped_crossing) schwächer.
Das ist charakteristisch für ein World Model, das auf komprimierten Latents
statt auf Pixelebene arbeitet — kein Regressionsfehler.

### Qualitative Visualisierung

`comparison_000681a0…png` (mIoU 0,951, Schnellstraße): pred und real nahezu
identisch, klare Fahrspur-/Divider-Geometrie. `comparison_0018da40…png`
(mIoU 0,822, Kreuzung): Grundstruktur gut getroffen, feine Strukturen
(Zebrastreifen, Haltelinien) leicht unschärfer als Real. Bestätigt die
quantitativen mIoU-Zahlen visuell.

---

## Erkenntnisse zur Daten-Pipeline (wichtig für Task 14/15)

Diese Punkte kosteten in Task 13 erheblich Zeit und sind für alle weiteren Läufe
auf voller Datenmenge relevant:

### 1. HDF5 + gzip ist hier kontraproduktiv — `.npy` ist ~2× schneller

Direkter Sanity-Check-Vergleich (jeweils 4 Epochen, identische Daten/Loss):

| Modus | batch_size | Ø Zeit/Epoche | bester Val-Loss |
|---|---|---|---|
| **.npy** | 4 | **~804 s** | 0,0576 |
| HDF5 (gzip-4) | 8 | ~1.656 s | 0,0582 |

Die gzip-Dekompression bei **jedem einzelnen Latent-Read** kostet so viel
CPU-Zeit, dass sie den Vorteil des einzelnen Datei-Handles mehr als auffrisst.
Die Loss-Werte sind praktisch identisch (beide Modi lernen gleich gut). →
**Entscheidung: `.npy`-Direktzugriff für den Produktionslauf.** Die HDF5-Caches
(157 GB) bleiben erhalten, werden aber nicht genutzt.

### 2. Der Bottleneck ist Disk-I/O, nicht GPU-Compute

batch_size 4 vs. 8 ergab praktisch identische Epochenzeit (~811 s vs. ~813 s).
Größere Batches verarbeiten mehr pro Schritt, aber die GPU wartet ohnehin auf
Daten. Pro Epoche werden ~108.000 Einzeldateien à ~16 MB gelesen (~1,7 TB
Lesevolumen). Die GPU-Auslastung schwankte entsprechend (17–46 %).

### 3. Page-Cache hilft nicht

Der Train-Datensatz (440 GB) passt nicht in die 126 GB RAM, daher wird er pro
Epoche größtenteils neu von Disk gelesen. Epoche 2+ waren nicht schneller als
Epoche 1.

### 4. Offener Optimierungshebel: float16-Latents

Latents als float16 statt float32 speichern würde das Lesevolumen **halbieren**
(~16 MB → ~8 MB pro Datei) und damit direkt die Ladezeit senken. Da ohnehin mit
AMP/autocast in float16 trainiert wird, ist der Präzisionsverlust minimal. Wäre
ein einmaliger Konvertierungslauf. **Empfehlung:** vor Task 15 umsetzen (dort
viele Läufe → größter Nutzen), als bewusste, durchgängig dokumentierte Änderung
der Datengrundlage — nicht mitten in einer Versuchsreihe wechseln.

### 5. Prozess-Hygiene: Zombie-Worker nach Abbruch

Nach einem abgebrochenen/gecrashten Lauf bleiben die DataLoader-Worker-Prozesse
(`num_workers=4`) als verwaiste Prozesse aktiv und lesen weiter aus den Daten —
sie konkurrieren mit dem nächsten Lauf um CPU und Disk und verlangsamen ihn
massiv. Das war die Ursache mehrerer scheinbarer "Hänger". **Vor jedem Neustart:**

```bash
pkill -9 -f "train_linux.py"
sleep 2
ps aux | grep "train_linux.py" | grep -v grep   # muss leer sein
```

### 6. Fehlender Fortschritts-Indikator verschleierte den echten Zustand

Ohne tqdm gab `train_one_epoch()` erst nach kompletter Epoche (Training + voller
Validierung) eine Zeile aus. "Keine Bewegung" bedeutete daher nicht "hängt",
sondern "Epoche noch nicht fertig". Der tqdm-Einbau war diagnostisch
entscheidend, um langsam-aber-arbeitend von echtem Deadlock zu unterscheiden.

---

## Status & Übergabe an Task 15

Die Task-13-Metriken sind der **eingefrorene Referenzpunkt** für die
Loss-Experimente:

- **Latent:** cosine sim 0,8549 · std-Ratio 0,833 · MSE 0,0251 (5.743 Samples)
- **Seg:** mean mIoU 0,654 (10 Samples)
- **Checkpoint:** `checkpoints/task13_baseline_full/phase2/best_val_loss.pt`
- **Setup:** `.npy`, explicit-Split, batch_size 8, 47 Epochen (Early Stop)

Jede Loss-Modifikation in Task 15 wird gegen genau diese Zahlen gemessen.
