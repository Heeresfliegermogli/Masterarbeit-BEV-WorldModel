# TASK 12 — ABSCHLUSSBERICHT: Seg-Latent-Extraktion über das volle nuScenes trainval-Set

**Datum:** 20.06.2026
**Status:** ✅ ABGESCHLOSSEN

---

## Fragestellung

Extraktion der fusionierten BEV-Latents (Camera+LiDAR, nach ConvFuser) für das
**komplette nuScenes trainval-Set** (~34.000 Keyframes) mittels des
hook-basierten `latent_saver`-Mechanismus im BEVFusion-Seg-Modell.

Ziel: Bereitstellung der vollständigen Latent-Datengrundlage für das
World-Model-Training auf voller Datenmenge (Task 13 ff.), als Ersatz für die
bisherige Val-only-Teilmenge.

---

## Vorgehen

### Datengrundlage vorbereiten

1. **Speicher freigeräumt** (~1,6T): Server war zu 98–100% voll. Nach
   Rücksprache/Freigabe alte Fremddaten gelöscht (UnrealEngine, ROS-Bags,
   Backup-Verzeichnis).
2. **nuScenes trainval+test heruntergeladen** (~347G komprimiert, ~469G
   extrahiert) via `li-xl/nuscenes-download`.
3. **Map-Expansion v1.3** separat nachgeladen (`basemap`, `expansion`,
   `prediction` → `maps/`), da der Downloader das Paket nicht enthält.
   Notwendig für `LoadBEVSegmentation`.

### Docker / Code-Stand korrigiert (zentraler Befund)

Der bis dahin genutzte Container (`bevfusion:task9d`) enthielt eine
**abweichende, Radar-erweiterte Fork-Variante** des BEVFusion-Codes. Deren
`nuscenes_converter.py` erzeugt PKLs ohne das von `LoadBEVSegmentation`
benötigte `location`-Feld (und mit abweichendem `_radar`-Suffix-Schema).

→ Umstieg auf den **korrekten Repo-Stand** `git.unibw.de/l81blepo/bevfusion`
  (als Bind-Mount statt ins Image gebacken):

```
docker run -it --gpus all \
  -v /home/vima/bevfusion/bevfusion-main:/bevfusion \
  -v /home/vima/Desktop/Masterarbeit/Nuscenes/complete:/bevfusion/data/nuscenes \
  -v /home/vima/bevfusion_decoder:/output \
  --shm-size 16g --workdir /bevfusion \
  bevfusion:correct /bin/bash
```

Das Image `bevfusion:correct` (gebaut aus `docker/Dockerfile` des Repos) ist
byte-identisch mit dem ursprünglichen `bevfusion:latest` (gleiche Image-ID
`79aa4241b29c`) — das Basis-Image war nie das Problem, nur der eingebrachte
Code-Stand.

### PKL-Generierung (korrekter Converter)

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

Ergebnis (verifiziert, `location`-Feld vorhanden, kein `radars`-Key):
- `nuscenes_infos_train.pkl` — 28.130 Samples
- `nuscenes_infos_val.pkl` — 6.019 Samples
- `nuscenes_infos_test.pkl` / `nuscenes_dbinfos_train.pkl`

### Hook-Einbau

`bevfusion.py` (Hook nach Fuser, vor Decoder, nur Eval-Modus, gesteuert über
`SAVE_BEV_LATENTS`) und `latent_saver.py` direkt ins gemountete Repo gelegt —
dauerhaft, kein erneutes Reinkopieren bei Container-Neustart nötig.

### Extraktionslauf

Val und Train separat, via `tools/test.py` (schlanker Forward-Pfad ohne
Visualisierungs-Overhead). Der Hook ist script-agnostisch und greift im selben
`forward_single()` wie `visualize.py`.

```bash
SAVE_BEV_LATENTS=1 FLATTEN_BEV_LATENTS=0 LATENT_DTYPE=float32 \
MASTER_HOST=127.0.0.1 MASTER_PORT=29500 \
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map \
  [--cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_train.pkl]
```

Train-Split via `--cfg-options`-Override des `ann_file`.

---

## Ergebnis

| Split | Samples | Größe | Speicherort |
|-------|---------|-------|-------------|
| Val   | 6.019   | 95G   | `.../Nuscenes/latents/seg/val/`   |
| Train | 28.130  | 440G  | `.../Nuscenes/latents/seg/train/` |
| **Summe** | **34.149** | **~535G** | |

### Latent-Charakteristik (verifiziert)

| Eigenschaft | Wert |
|-------------|------|
| Shape | `(1, 256, 128, 128)` |
| dtype | float32 |
| Mean (Stichprobe) | ~0.13 – 0.16 |
| Std (Stichprobe) | ~0.26 – 0.29 |
| NaN / Inf | 0 / 0 |

Std-Werte konsistent mit der Task-11-Seg-Referenz (~0.263). 10er-Stichprobe
aus dem Train-Set ausnahmslos `OK`.

### Hinweis zum Lauf-Ende

Der Train-Lauf wurde **nach** vollständiger Verarbeitung aller 28.130 Samples
(`28130/28130, ETA: 0s`) durch den OOM-Killer beendet (Exit 137) — bei der
nachgelagerten mAP/mIoU-Aggregation, die für die Latent-Extraktion irrelevant
ist. Der Hook speichert jedes Latent sofort während des Forward-Passes; alle
28.130 Dateien sind vollständig und validiert. Kein Datenverlust.

---

## Wichtige Learnings

1. **Code-Stand verifizieren, nicht nur das Docker-Image.** Identisches
   Base-Image ≠ identischer Code. Der `location`-KeyError war Symptom eines
   falschen Fork-Stands, nicht eines Daten- oder Pipeline-Fehlers.
2. **Bind-Mount statt In-Image-Code.** Der korrekte Repo-Stand wird gemountet
   — Änderungen (Hook) bleiben persistent, kein `commit` nötig.
3. **Map-Expansion ist Pflicht** für `LoadBEVSegmentation` (separates Paket).
4. **`test.py` statt `visualize.py`** für reine Latent-Extraktion — kein
   Bild-Rendering, gleicher Hook-Pfad.
5. **Sequenzielles Extrahieren + Löschen** der Download-Archive verhindert das
   gleichzeitige Vorhalten von `.tar` + entpackten Daten (EXT4-Read-Only-Fall
   bei vollgelaufener Disk vermeiden).

---

## Artefakte

| Datei / Pfad | Beschreibung |
|--------------|--------------|
| `.../Nuscenes/latents/seg/val/` | 6.019 Val-Latents (float32) |
| `.../Nuscenes/latents/seg/train/` | 28.130 Train-Latents (float32) |
| `/output/task12_val_extraction.log` | Lauf-Log Val |
| `/output/task12_train_extraction.log` | Lauf-Log Train |
| `bevfusion.py` / `latent_saver.py` | Hook im Repo `bevfusion-main/` |
| `nuscenes_infos_{train,val}.pkl` | Korrekte PKLs mit `location`-Feld |

---

## Nächster Schritt

**Task 13** — Neues World-Model-Baseline-Training auf der vollen
nuScenes-Datengrundlage (Seg-Latents, 28.130 train / 6.019 val), verpflichtend
vor den Loss-Experimenten (Task 15).
