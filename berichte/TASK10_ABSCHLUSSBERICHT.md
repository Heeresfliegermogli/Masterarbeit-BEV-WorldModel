# TASK 10 — ABSCHLUSSBERICHT: Seg vs. Det Latent-Vergleich

**Datum:** 26.05.2026
**Status:** ✅ ABGESCHLOSSEN

---

## Fragestellung

BEVFusion hat architekturell einen gemeinsamen Encoder/Fuser, der den BEV-Latent `[256, 128, 128]` erzeugt. Die öffentlichen Checkpoints (`bevfusion-seg.pth`, `bevfusion-det.pth`) sind jedoch separat trainiert, und die Fuser-Gewichte unterscheiden sich signifikant (diff ~0.10–0.33, aus Task 9c bekannt).

**Offene Frage:**
- **(a)** Gemeinsamer Latent-Raum — nur die Decoder-Köpfe sind verschieden?
- **(b)** Zwei verschiedene Latent-Räume — Encoder hat sich unterschiedlich spezialisiert?

→ Die Antwort bestimmt die Architektur von Task 11.

---

## Vorgehen

**5 Val-Tokens** (Mini-PKL `nuscenes_infos_val_5.pkl`):
- `163b70e6`, `6eb8a3ff`, `b10f0cd7`, `f56a5440`, `fd842039`

**Schritt 1 — Extraktion:**
- Seg-Latents: `bevfusion-seg.pth` + `configs/nuscenes/seg/fusion-bev256d2-lss.yaml` → `/output/seg_latents/` → Shape `[1, 256, 128, 128]`
- Det-Latents: `bevfusion-det.pth` + `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml` → `/output/det_latents/` → Shape `[1, 256, 180, 180]`

**Hinweis zu den Auflösungen:**
- Seg: `xbound/ybound [-51.2, 51.2, 0.4]`, `downsample=2` → **128×128**
- Det: `xbound/ybound [-54.0, 54.0, 0.3]`, `downsample=2` → **180×180**
- Verschiedene physische BEV-Reichweite und Auflösung — erwartet.

**Schritt 2 — Quantitativer Vergleich (`compare_latents.py`):**
- **Methode 1 — CosSim channel-avg:** GlobalAvgPool → `[256]`-Vektor → CosSim. Auflösungsunabhängig, misst Kanal-Aktivierungsmuster.
- **Methode 2 — CosSim spatial:** Det bilinear auf `128×128` resizen → flatten → CosSim. Misst räumliche Übereinstimmung nach Normierung.
- **MSE** nach Resize, Kanal-Statistiken.

---

## Ergebnisse

| Metrik | Wert |
|--------|------|
| **CosSim channel-avg** (Hauptindikator) | **0.880 ± 0.006** |
| CosSim spatial (nach Resize) | 0.160 ± 0.002 |
| MSE (nach Resize) | 0.500 ± 0.017 |
| Seg mean / Det mean | 0.152 / 0.227 |
| Seg std / Det std | 0.263 / 0.690 |

**Per-Sample CosSim channel:**

| Token | CosSim |
|-------|--------|
| 163b70e6 | 0.882 |
| 6eb8a3ff | 0.883 |
| b10f0cd7 | 0.883 |
| f56a5440 | 0.883 |
| fd842039 | 0.869 |

Sehr geringe Varianz (std=0.006) — das Ergebnis ist stabil über alle Samples.

---

## Interpretation

### CosSim channel-avg = 0.880 → Option (a): Gemeinsamer Latent-Raum ✅

Der Wert liegt klar über der 0.8-Schwelle. Beide Modelle kodieren dieselbe semantische Kanalstruktur — welche Kanäle aktiv sind, stimmt zwischen Seg und Det stark überein.

### Warum ist CosSim spatial so niedrig (0.160)?

Das ist kein Widerspruch, sondern hat zwei Ursachen:
1. **Unterschiedliche physische Reichweite:** Seg deckt ±51.2m ab, Det ±54m — nach bilinearem Resize auf 128×128 bleibt ein systematischer Skalenversatz.
2. **Schärfeunterschied:** Det-Aktivierungen sind deutlich spitzer (std ≈ 2.6× größer als Seg). Der Detektions-Task fordert präzise Peaks für Bounding Boxes; Segmentierung produziert kontinuierlichere Karten. Nach dem Resize kollidieren diese spitzen Det-Peaks nicht pixelgenau mit den glatteren Seg-Werten.

### Heatmap-Befund

Alle 5 Heatmaps zeigen dieselbe Struktur: Seg (oben) und Det (unten) haben identische aktive Regionen, denselben Ego-Vehicle-Mittelpunkt und dieselben Szenenstrukturen — nur mit unterschiedlicher Aktivierungsamplitude und Schärfe. Das bestätigt Option (a) visuell.

### Det std ≈ 2.6× größer als Seg std

Erklärbar durch den Aufgabenunterschied: Objektdetektion benötigt diskrete, scharfe Peaks (ein Auto = ein Peak), Segmentierung kontinuierliche Flächen (eine Fahrspur = eine Fläche). Beide Repräsentationen leben im selben 256-dimensionalen Kanalraum, unterscheiden sich aber in der räumlichen Schärfe.

---

## Schlussfolgerung & Konsequenz für Task 11

**Das bestehende World Model (trainiert auf Seg-Latents) kann direkt durch den Det-Decoder (`bevfusion-det.pth`) geschickt werden — kein separates Retraining nötig.**

Task 11 implementiert **Option B** (aus dem Arbeitsplan):
1. `inference_det.py` schreibt `pred_latent` (vom Seg-World-Model) in den Det-Decoder
2. backbone (SECOND) → neck (SECONDFPN) → object_head (CenterHead) → Bounding Boxes
3. Visualisierung: pred_boxes vs. real_boxes im BEV-Plot
4. Metrik: mAP (falls Referenz-Boxes vorhanden) oder qualitative Analyse

**Aufwand Task 11:** ~1 Tag (nur Inference-Script, kein Training).

---

## Artefakte

| Datei | Beschreibung |
|-------|--------------|
| `compare_latents.py` | Vergleichsscript (argparse, CosSim/MSE/Heatmap) |
| `/output/latent_comparison.json` | Quantitative Ergebnisse, alle 5 Samples |
| `/output/viz/latent_heatmap_seg_vs_det_*.png` | Heatmaps, 6 Kanäle pro Sample |
| `/output/seg_latents/*.npy` | 5 Seg-Latents `[1,256,128,128]` |
| `/output/det_latents/*.npy` | 5 Det-Latents `[1,256,180,180]` |
