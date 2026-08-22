# Task 9c — BEVFusion Decoder-Test: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)
**Status:** ✅ ABGESCHLOSSEN

---

## Überblick

Task 9c ist das zentrale Entscheidungsgate des Projekts: Sind die predicted Latents
des World Models decoder-kompatibel — d.h. erzeugen sie im BEVFusion-Decoder
Feature Maps die strukturell den echten Latents entsprechen?

**Ergebnis: JA — Decoder ist kompatibel. mean neck_CosSim = 0.8285**

---

## Setup

### Docker-Container (neu aufgesetzt)

Da BEVFusion auf dem neuen Linux-System nicht installiert war, wurde ein
vollständiger Docker-Container aufgebaut:

```
Base Image:    nvidia/cuda:11.3.1-devel-ubuntu20.04
Python:        3.8 (Miniforge)
PyTorch:       1.10.1+cu113 (via pip, nicht conda)
mmcv:          1.4.0 + mmcv-full
mmdet:         2.20.0
nuscenes-devkit, mpi4py, numba
```

**Bekannte Probleme und Fixes:**
- Anaconda ToS → Miniforge als Ersatz
- Conda Channel-Konflikt PyTorch → pip install torch+cu113
- Timezone-Block → ENV TZ + tzdata vorab installiert
- `feature_decorator_ext` nicht kompiliert → manuell mit ninja gebaut
- `int` vs `int64_t` in feature_decorator.cpp → sed-Patch
- `flash_attn` fehlt → Import auskommentiert
- `numba` + `numpy` Inkompatibilität → numba==0.56.4 + numpy==1.23.5

### Checkpoints

Alle Pretrained Weights via `tools/download_pretrained.sh` geladen.
Für den Decoder-Test wird **bevfusion-seg.pth** verwendet (Seg-Checkpoint,
identisch mit dem für Task 1 verwendeten Checkpoint).

---

## Methode

### Warum Seg- statt Det-Checkpoint?

Die Latents wurden mit `bevfusion-seg.pth` extrahiert. Ein Vergleich der
Fuser-Gewichte zwischen Seg- und Det-Checkpoint ergab signifikante Unterschiede:

```
fuser.0.weight:       diff = 0.104
fuser.1.bias:         diff = 0.167
fuser.1.running_mean: diff = 1.540
fuser.1.running_var:  diff = 12.89
```

Die Det-Decoder-Gewichte erwarten eine andere Latent-Verteilung → Seg-Checkpoint
ist die einzig korrekte Wahl.

### Decoder-Pipeline

```
pred_latent [256, 128, 128]
        ↓
decoder["backbone"] (SECOND)     → [128, 128, 128] + [256, 64, 64]
        ↓
decoder["neck"] (SECONDFPN)      → Neck Feature Maps [512, 128, 128]
        ↓
Metrik: CosSim(pred_neck, real_neck)
```

Die Gewichte wurden direkt aus dem Checkpoint extrahiert (ohne volle
BEVFusion Config-Pipeline) um Config-Parsing-Probleme zu umgehen.

---

## Ergebnisse (10 Smoke-Test Samples)

| Token | neck_MSE | neck_CosSim |
|-------|----------|-------------|
| 01a7d01c | 0.037197 | 0.8016 |
| 1dfecb81 | 0.035090 | 0.8051 |
| 2140329a | 0.036924 | 0.8003 |
| 296fcfbf | 0.030777 | 0.8315 |
| 3bf56ebb | 0.029329 | 0.8438 |
| 5bd85334 | 0.030016 | 0.8455 |
| 61f89208 | 0.028015 | 0.8548 |
| a2fada92 | 0.033570 | 0.8216 |
| b06a8151 | 0.030270 | 0.8418 |
| f7d75d25 | 0.029144 | 0.8393 |

**mean neck_MSE:    0.032033**
**mean neck_CosSim: 0.8285** ← Hauptmetrik

### Interpretation

| Schwellenwert | Bedeutung |
|---|---|
| CosSim > 0.9 | Decoder vollständig kompatibel |
| **CosSim > 0.7** | **Decoder grundsätzlich funktionsfähig ✅** |
| CosSim < 0.5 | Decoder inkompatibel |

Mit CosSim = 0.8285 liegt das Modell klar im funktionsfähigen Bereich.
Der neck_MSE von 0.032 ist konsistent mit dem Val-Loss von 0.036 aus dem Training.

### Einordnung des std-Gaps

Das bekannte std-Gap (pred_std=0.213 vs real_std=0.290, ~26%) schlägt sich
in CosSim < 1.0 nieder — das Modell ist etwas "glatter" als die Realität.
Die grobe räumliche Struktur der BEV-Features wird jedoch korrekt vorhergesagt,
was für Decoder-Kompatibilität ausreicht.

---

## Entscheidung (Arbeitsplan-Gate)

**Decoder OK → Task 8 wird zu optionaler Ablation**

Task 8a (Loss-Verbesserungen) und Task 7 (mehr Daten) sind nicht notwendig.
Die Pipeline ist wissenschaftlich valide dokumentiert.

---

## Nächste Schritte

- **Task 9d (neu):** Segmentierungsmasken-Visualisierung
  pred_latent → Decoder → Map-Head → Seg-Masken PNG (vehicle, road, etc.)
  Vergleich pred_mask vs. real_mask via pixel-wise IoU pro Klasse
- **Task 9b:** Visualisierungen (Latent-Space Heatmaps, Gate Alpha)
- **Task 8b (optional):** Ablation-Matrix für Thesis

---

## Artefakte

```
/workspace/decoder/inference_decoder.py       ← Decoder-Test Script
/workspace/decoder/decoder_test_results.json  ← Ergebnisse (10 Samples)
```
