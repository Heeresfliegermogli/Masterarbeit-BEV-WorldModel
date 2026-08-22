# Task 9d — Segmentierungsmasken-Visualisierung: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)
**Status:** ✅ ABGESCHLOSSEN

---

## Überblick

Task 9d liefert den zentralen qualitativen Beweis der Thesis: Die vom World
Model vorhergesagten BEV-Latents erzeugen im BEVFusion-Decoder kohärente,
strukturell korrekte Segmentierungsmasken — vergleichbar mit denen der echten
Latents.

**Ergebnis: mean mIoU = 0.564 über 10 Val-Samples**

---

## Erkenntnisse aus dem ersten Task-9d-Versuch

Der erste Versuch (fragmentierte Masken) hatte drei konkrete Bugs:

1. **Falsche Klassen hardcodiert** — `vehicle`, `road`, `bike_lane` statt der
   echten nuScenes-Klassen aus der Config (`drivable_area`, `ped_crossing`,
   `walkway`, `stop_line`, `carpark_area`, `divider`). Der letzte Conv-Layer
   des Classifiers hat `out_channels=len(classes)` — ein Mismatch führt zu
   inkompatiblen Gewichten.

2. **Map-Head nie geladen** — `head_cfg` war definiert aber `load_state_dict`
   wurde nie aufgerufen. Der Classifier lief mit zufälligen Gewichten.

3. **BEVGridTransform fehlte** — dieser Transform remappt den Neck-Output von
   `[512,128,128]` auf `[512,200,200]` bevor der Classifier greift. Ohne ihn
   bekommt der Classifier falsch dimensionierte Features.

---

## Finaler Ansatz

Direkte Gewichts-Extraktion aus `bevfusion-seg.pth` ohne volle BEVFusion-
Config-Pipeline. Kein Dataset, keine Sensordaten, kein torchpack erforderlich.

### Pipeline

```
pred_latent.npy  [256, 128, 128]
        ↓
SECOND backbone              → [128,128,128] + [256,64,64]
        ↓
SECONDFPN neck               → [512, 128, 128]
        ↓
BEVGridTransform             → [512, 200, 200]
  input_scope:  [[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]]
  output_scope: [[-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]]
        ↓
Conv-Classifier (3x Conv2d)  → [6, 200, 200] logits
        ↓
Sigmoid + Threshold (0.5)    → [6, 200, 200] bool
        ↓
pred_mask → PNG + IoU
```

### Gewichts-Laden (verifiziert)

```
decoder.backbone.*  → 72 keys, strict=True, missing=[]
decoder.neck.*      → 12 keys, strict=True, missing=[]
heads.map.classifier.* → 14 keys, strict=True, missing=[]
```

Sanity-Check: letzter Conv-Layer hat `out_channels=6` ✅

### Wichtige Architektur-Entdeckung

Analyse des BEVFusion-Quellcodes (`bevfusion.py`) ergab: Es gibt nur
**einen einzigen gemeinsamen Latent** — nicht zwei separate für Seg und Det.
Der Fuser erzeugt ein `x [256,128,128]` das parallel durch alle konfigurierten
Heads läuft. Die öffentlichen Checkpoints sind jedoch getrennt trainiert:

```
bevfusion-seg.pth → heads: {'map'}    (nur Kartensegmentierung)
bevfusion-det.pth → heads: {'object'} (nur 3D Objektdetektion)
```

Fuser-Gewichts-Vergleich bestätigt inkompatible Latent-Räume:
```
fuser.0.weight: diff=0.1039
fuser.1.weight: diff=0.3277
fuser.1.bias:   diff=0.1665
```

→ Unsere `pred_*.npy` / `real_*.npy` sind Seg-Latents und können nur mit
  dem Seg-Decoder sinnvoll dekodiert werden.

---

## Ergebnisse (10 Val-Samples)

### IoU pro Sample (pred vs. decoded real)

| Token    | mIoU  | drivable | ped_cross | walkway | stop_line | carpark | divider |
|----------|-------|----------|-----------|---------|-----------|---------|---------|
| 01a7d01c | 0.360 | 0.86     | 0.00      | 0.53    | 0.02      | --      | 0.39    |
| 1dfecb81 | 0.437 | 0.87     | --        | 0.55    | 0.00      | --      | 0.33    |
| 2140329a | 0.449 | 0.86     | 0.22      | 0.53    | 0.25      | --      | 0.38    |
| 296fcfbf | 0.628 | 0.90     | --        | 0.56    | --        | --      | 0.42    |
| 3bf56ebb | 0.748 | 0.94     | --        | 0.73    | --        | --      | 0.58    |
| 5bd85334 | 0.755 | 0.95     | --        | 0.74    | --        | --      | 0.58    |
| 61f89208 | 0.660 | 0.92     | --        | 0.67    | --        | --      | 0.39    |
| a2fada92 | 0.452 | 0.87     | --        | 0.57    | 0.00      | --      | 0.37    |
| b06a8151 | 0.686 | 0.91     | --        | 0.66    | --        | --      | 0.49    |
| f7d75d25 | 0.469 | 0.89     | --        | 0.55    | 0.00      | --      | 0.43    |

`--` = Klasse im Sample nicht vorhanden (union=0, aus mIoU-Berechnung ausgeschlossen)

### Zusammenfassung

```
mean mIoU:       0.5644

Per-Klassen-IoU (pred vs. decoded real):
  drivable_area   0.897  ██████████████████████████
  ped_crossing    0.111  ███
  walkway         0.608  ██████████████████
  stop_line       0.053  █
  carpark_area    0.000
  divider         0.437  █████████████
```

### Qualitative Beobachtungen

- **drivable_area** (hellblau): sehr hohe Übereinstimmung, pred und real
  nahezu identisch — das World Model lernt Fahrbahnstruktur zuverlässig
- **walkway** (rot): Konturen stimmen, leichte Randverschiebungen
- **divider** (lila): erkennbar, gut übereinstimmend
- **ped_crossing / stop_line** (pink/orange): niedrige IoU durch seltenes
  Auftreten in diesen Szenen — das World Model halluziniert leicht bei
  seltenen Klassen, was die IoU=0.00 erklärt
- **carpark_area**: in keinem der 10 Samples vorhanden
- Pred-Masken sind leicht breiter/weicher als real → das World Model
  überschätzt Ausdehnung leicht (konsistent mit bekanntem std-Gap von 26%)
- Diff-Bilder zeigen hauptsächlich grün (korrekt aktiv) mit roten Streifen
  an Rändern → **Boundary-Fehler**, kein fundamentaler Strukturfehler

### Einordnung

Das Ergebnis (mIoU=0.564) liegt im oberen Bereich der Erwartung aus Task 9c
(neck_CosSim=0.8285 → mIoU-Erwartung 0.3–0.6). Die Masken sind kohärent und
strukturell korrekt — keine Fragmentierung wie im ersten Versuch.

---

## Container & Umgebung

```
Image:      bevfusion:task9d
Checkpoint: /home/bevfusion/pretrained/bevfusion-seg.pth
Latents:    /workspace/decoder/pred_*.npy + real_*.npy (10 Samples)
Output:     /workspace/decoder/viz/
```

---

## Artefakte

```
/workspace/decoder/inference_seg.py          ← Hauptscript
/workspace/decoder/viz/pred_<token>.png      ← 10 pred Masken
/workspace/decoder/viz/real_<token>.png      ← 10 real Masken
/workspace/decoder/viz/comparison_<token>.png ← 10 Side-by-Side (pred|real|diff)
/workspace/decoder/viz/seg_results.json      ← mIoU + IoU pro Klasse
```

---

## Nächste Schritte

- **Task 9e (neu):** Seg-Latent vs. Det-Latent Vergleich — Analyse ob beide
  Latent-Räume strukturell ähnlich sind und ob ein kombiniertes World Model
  (Seg+Det gleichzeitig) realistisch wäre
- **Task 9b:** Latent-Space Visualisierungen (Heatmaps, Gate Alpha)
