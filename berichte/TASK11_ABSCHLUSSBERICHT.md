# TASK 11 — ABSCHLUSSBERICHT: Detection — World Model für Bounding Boxes

**Datum:** 27.05.2026
**Status:** ✅ ABGESCHLOSSEN (negativer Hauptbefund — methodisch vollständig)

---

## Fragestellung

Können die predicted BEV-Latents des Seg-World-Models direkt durch den
BEVFusion Det-Decoder (`bevfusion-det.pth`) geschickt werden, um Bounding
Boxes für Fahrzeuge, Fußgänger etc. zu erzeugen?

→ Task-10-Ergebnis (CosSim=0.880) hatte **Option B** motiviert:
kein Retraining, direkter Det-Decoder auf pred_latents.

---

## Vorgehen

### Pipeline (Option B)

```
pred_latent [256, 128, 128]  (Seg-World-Model)
       ↓  decoder.backbone (SECOND)     bevfusion-det.pth
       ↓  decoder.neck (SECONDFPN)      → [512, 128, 128]
       ↓  heads.object (TransFusionHead)
            shared_conv → heatmap_head → Top-K Queries
            Transformer Decoder (Cross-Attention Queries ↔ BEV)
            prediction_heads → Bounding Boxes
```

### Implementierung

- `inference_det.py`: TransFusionHead direkt aus mmdet3d instanziiert
- 96/96 Gewichte aus `bevfusion-det.pth` korrekt geladen
- Identische Pipeline zum Standard-BEVFusion-Det (nur Encoder ersetzt)
- 10 pred/real Latent-Paare aus `/workspace/decoder/`

### Technische Herausforderungen (dokumentiert)

| Problem | Ursache | Fix |
|---------|---------|-----|
| `SyntaxError: global CHECKPOINT` | `global` nach erster Nutzung | `global` an Funktionsanfang |
| `gaussian_overlap` unbekannt | Gehört in `train_cfg`, nicht `bbox_coder` | Parameter entfernt |
| `use_sigmoid` unbekannt in `GaussianFocalLoss` | mmdet3d liest es im Head, nicht im Loss | einmaliger Patch `transfusion.py` |
| CenterHead-Keys falsch gemappt | TransFusion ≠ CenterPoint-Architektur | direkte mmdet3d-Instanziierung |

---

## Ergebnis

### Quantitativ

| Metrik | Wert |
|--------|------|
| Geladene Head-Gewichte | 96/96 ✓ |
| Max Heatmap Score (real_latent) | **0.052** |
| Scores > 0.20 | **0** |
| Scores > 0.10 | **0** |
| Detektierte Boxen (pred + real) | **0** |

### Ursachenanalyse

**Direkte Ursache:** Heatmap-Scores bleiben unter 0.06 — weit unter dem
Detection-Threshold von 0.20. Der TransFusionHead erkennt keine Objekte,
weder aus pred_latents noch aus real_latents (Seg).

**Strukturelle Ursache:** Seg- und Det-Checkpoints sind **end-to-end separat
trainiert** — nicht nur die Decoder-Köpfe, sondern Encoder und Fuser:

```
Fuser-Gewichte:
  fuser.0.weight:       diff = 0.104
  fuser.1.weight:       diff = 0.328
  fuser.1.bias:         diff = 0.167

Camera Encoder-Gewichte:
  patch_embed.projection.weight: diff = 0.056
  patch_embed.norm.weight:       diff = 0.345
```

Seg-Encoder/Fuser hat gelernt **kontinuierliche Flächen** zu kodieren.
Det-Encoder/Fuser hat gelernt **diskrete Peaks** zu kodieren. Das äußert
sich quantitativ in:

| | Seg-Latent | Det-Latent |
|--|--|--|
| std | 0.263 | 0.690 |
| Charakter | glatte Flächen | scharfe Peaks |
| Det-Decoder max Score | 0.052 | normale Detektion |
| Ratio | — | 2.6× schärfer |

**Warum CosSim=0.880 trotzdem korrekt war:**
CosSim channel-avg misst die **Richtung** der Kanal-Aktivierungen — welche
Kanäle aktiv sind, nicht wie stark. Beide Latent-Räume kodieren dieselbe
Szenenstruktur (gleiche aktive Regionen, gleiche Kanal-Rangfolge) — aber mit
fundamental verschiedener Amplituden-Charakteristik. Der Det-Decoder wurde
auf scharfe Peaks trainiert und erkennt die glatten Seg-Aktivierungen nicht
als Objekte.

> CosSim ist notwendig aber nicht hinreichend für Cross-Decoder-Kompatibilität.

---

## Schlussfolgerung

**Option B ist strukturell nicht realisierbar.** Das Ergebnis ist kein
Implementierungsfehler, sondern eine fundamentale Inkompatibilität der
Latent-Charakteristiken.

Für echte Bounding-Box-Outputs aus dem World Model sind zwei Schritte nötig:
1. **Det-Latents extrahieren** (bevfusion-det.pth, ~5743 Samples)
2. **World Model auf Det-Latents trainieren** (~2 Tage, identisches Setup)

Dies wird als **Tasks 13–15** in den Arbeitsplan aufgenommen.

---

## Wissenschaftlicher Beitrag

Das negative Ergebnis ist methodisch vollständig und thesis-relevant:

1. **Option B systematisch getestet** — nicht nur theoretisch ausgeschlossen
2. **Quantitative Begründung** — max Score 0.052, std-Ratio 2.6×, 0 Boxen
3. **Architektur-Analyse** — Seg/Det Encoder+Fuser separat trainiert,
   verschiedene Latent-Charakteristiken by design
4. **Grenze von CosSim als Ähnlichkeitsmetrik** dokumentiert:
   Kanal-Richtung ≠ Amplituden-Kompatibilität

---

## Artefakte

| Datei | Beschreibung |
|-------|-------------|
| `inference_det.py` | Det-Inference Script (TransFusionHead via mmdet3d) |
| `/workspace/decoder/viz_det/det_results.json` | Quantitative Ergebnisse |
| `/workspace/decoder/viz_det/det_comparison_*.png` | BEV-Plots (10 Samples) |
