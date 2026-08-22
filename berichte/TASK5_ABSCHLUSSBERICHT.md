# Task 5 — Training Loop, Architektur-Updates & Erste Ergebnisse: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)
**Status:** ✅ ABGESCHLOSSEN

---

## Überblick

Task 5 umfasst mehr als ursprünglich geplant: Neben dem eigentlichen Training Loop
wurden während dieser Phase mehrere Architektur-Updates durchgeführt, zwei vollständige
Trainingsläufe (Phase 1 + Phase 2) abgeschlossen, erste Predictions generiert und
visualisiert sowie der Grundstein für Task 6–9 gelegt.

---

## Projektstruktur (aktueller Stand)

```
Code_final/
│
│  ── Daten-Pipeline (Task 1) ──────────────────────────────
├── Code/bev_dataset.py          # Dataset-Klasse
├── Code/bev_dataloader.py       # DataLoader + Train/Val/Test Split
├── Code/build_hdf5_cache.py     # HDF5-Cache Builder
│
│  ── Modell-Module (Tasks 2–4, inkl. Updates Task 5) ──────
├── Code/config.py               # ModelConfig dataclass
├── Code/downsampling.py         # DownsamplingModule
├── Code/embedding.py            # FrameLevelEmbedding, CellLevelEmbedding
├── Code/transformer.py          # TransformerBlock, TransformerEncoder
├── Code/output_head.py          # FrameLevelOutputHead, CellLevelOutputHead
├── Code/upsampling_head.py      # UpsamplingHead (GroupNorm)
├── Code/bev_world_model.py      # BEVWorldModel + Gated Skip Connection ← UPDATE
├── Code/checkpointing.py        # save/load Checkpoint
│
│  ── Training & Inference (Task 5/6) ──────────────────────
├── train_linux.py               # Training Loop Linux (Hauptskript) ← UPDATE
├── train.py                     # Training Loop Windows
├── inference.py                 # Inference Pipeline
├── config_nuscenes_val.yaml     # Aktive Konfiguration
├── config.yaml                  # Basis-Konfiguration
├── visualize_predictions.py     # Visualisierung pred/real Latents
│
│  ── Tests & Validierung ───────────────────────────────────
└── smoke_test.py                # 8/8 Tests bestanden
```

---

## Architektur-Updates in Task 5

### Update 1: Additive → Gated Skip Connection

**Motivation:** Die erste Visualisierung der Predictions zeigte deutliche
Unterschiede zwischen predicted und realen Latents im Differenz-Plot.
Das Problem wurde auf die additive Skip Connection zurückgeführt:

```python
# Vorher — additiv:
pred = transformer_out + skip   # Modell muss exakt Δ(t→t+1) lernen
                                 # kämpft gegen skip bei großer Bewegung
```

Bei bewegten Objekten (Fahrzeugen) und großen Δ zwischen t und t+1 kämpfte
die additive Skip Connection gegen den Transformer-Output an — das Modell
konnte nicht effizient unterscheiden wo es predicten und wo es kopieren soll.

**Lösung — Gated Skip Connection:**

```python
# Nachher — gated:
alpha = Sigmoid(Conv1x1(transformer_out))   # [B, 256, 128, 128] ∈ [0,1]
pred  = alpha * transformer_out + (1-alpha) * skip

# α ≈ 0: skip dominiert → statische Bereiche (Straße, Gebäude) werden kopiert
# α ≈ 1: transformer dominiert → Bewegung wird predictet
```

Das Modell lernt selbst pro Channel und räumlicher Position zu entscheiden
wann es Frame t kopieren und wann es wirklich predicten soll.

**Impact:**
```
Phase 2 ohne Gated Skip:  Val-Loss = 0.045385  (Epoch 35)
Phase 2 mit Gated Skip:   Val-Loss = 0.036193  (Epoch 28)
Verbesserung: 20.2% — schnellere Konvergenz und besseres Minimum
```

**Neue Parameter:** +65.792 (Conv1x1 256→256 + Bias)
**Gesamt Phase 2:** ~6.05M Parameter

### Update 2: Loss-Erweiterung

```
Vorher:  loss = MSE_multiscale + 0.1*CosSim + 0.1*Dist
Nachher: loss = MSE_multiscale + 0.1*CosSim + 0.1*Dist + 0.1*SSIM
```

Der SSIM-Term (Structural Similarity) wurde hinzugefügt um hochfrequente
Details zu verbessern. In Tests zeigte sich jedoch dass `lambda_ssim=0.1`
zu stark gewichtet ist und den Val-Loss durch den SSIM-Offset erhöht
(0.062 statt 0.036). SSIM mit `lambda_ssim=0.01` als schwache Regularisierung
ist für Task 8 (Ablation) geplant.

### Update 3: BatchNorm → GroupNorm im UpsamplingHead

BatchNorm normalisiert über den Batch und zwingt Latents auf mean≈0, std≈1 —
das kämpft gegen loss_dist an. GroupNorm(32) normalisiert innerhalb jedes
Samples und respektiert die echte BEVFusion-Latent-Verteilung.

---

## Training-Ergebnisse

### Daten

```
Dataset:        nuScenes Val-Set (Full), 92 Szenen
Split (seed=42): 64 Train / 13 Val / 15 Test
Sequenzen:      3746 Train / 844 Val / 1153 Test
Latents:        ~6000 .npy Dateien (je 16MB, [256, 128, 128])
HDF5-Cache:     bev_latents_val.h5 (~30GB, gzip level 1)
```

**Wichtige Erkenntnis zu HDF5 vs. .npy:**

| Modus | Zeit/Epoch | Anmerkung |
|---|---|---|
| Windows .npy | ~480s | I/O Bottleneck |
| Linux HDF5 (swmr=True) | ~232s | Dekompressionsoverhead |
| Linux .npy (num_workers=4) | ~99s | **Optimal** — Linux RAM-Cache |

Bei 257GB RAM des Linux-Systems werden alle ~94GB Latents nach Epoch 1
vollständig im RAM-Cache gehalten → kein SSD-Zugriff mehr. HDF5 lohnt sich
erst beim vollen Train-Set (~300GB > 257GB RAM).

### Phase 1 (Frame-Level, 3 Tokens)

```
Konfiguration:  batch_size=16, num_workers=4, lr=1e-4, epochs=50
Hardware:       NVIDIA TITAN RTX (24GB VRAM), Linux Ubuntu
Zeit/Epoch:     ~99s
Gesamt:         ~1.4 Stunden

Ergebnis:
  Bester Val-Loss: 0.059146 (Epoch 49, kein Early Stopping)
  dist (avg):      ~0.0004-0.0006  ← sehr niedrig ✓
  Verhalten:       Plateau ab Epoch ~15, kaum weitere Verbesserung
```

**Interpretation:** Phase 1 ist architektonisch begrenzt — 3 Tokens (GlobalAvgPool)
bedeutet alle räumliche Information wird auf einen einzigen Vektor komprimiert.
Das ist by design: Phase 1 ist ein schneller Sanity-Check, kein wissenschaftlicher
Hauptbeitrag.

### Phase 2 (Cell-Level, 3072 Tokens) — Hauptergebnis

```
Konfiguration:  batch_size=4, num_workers=4, lr=1e-4, epochs=50
Hardware:       NVIDIA TITAN RTX (24GB VRAM), Linux Ubuntu
Zeit/Epoch:     ~100s
Gesamt:         ~1.3 Stunden (Early Stopping bei Epoch 45)

Bestes Ergebnis (mit Gated Skip Connection):
  Val-Loss:  0.036193  (Epoch 28)
  dist:      ~0.003-0.004
  CosSim:    ~0.73

Vergleich Phase 1 vs. Phase 2:
  Phase 1:   0.059146  (3 Tokens,    Frame-Level)
  Phase 2:   0.036193  (3072 Tokens, Cell-Level)
  Verbesserung: 38.9%
```

**Interpretation:** Phase 2 zeigt die wissenschaftliche Kernaussage der Thesis:
spatio-temporales Reasoning auf Cell-Ebene (3072 Tokens) ermöglicht deutlich
präzisere Predictions als Frame-Level Aggregation (3 Tokens).

---

## Erste Prediction-Visualisierung (Zwischencheck)

Nach dem Training wurden 10 Samples durch `inference.py` generiert und mit
`visualize_predictions.py` visualisiert.

### Metriken auf Test-Samples

```
MSE:       0.0566 ± 0.0002
CosSim:    0.733
dist:      0.00087
Pearson r: 0.9975  (Channel-Korrelation Pred vs. Real)
```

### Qualitative Beobachtungen

**Positiv:**
- Grobe räumliche Struktur korrekt: Straßenverläufe, BEV-Grid-Muster,
  Kurven erkennbar und weitgehend übereinstimmend
- `distribution.png`: Pred- und Real-Histogramme nahezu identisch überlagert
- Channel-Korrelation Pearson r=0.9975 — nahezu perfekte statistische Übereinstimmung
- `dist=0.00087` — Channel-Statistiken extrem ähnlich → Grünes Licht für BEVFusion-Decoder

**Offen / Verbesserungsbedarf:**
- `heatmaps_overview.png`: Differenz-Plot zeigt flächendeckendes Rauschen
  (Werte bis ~0.25-0.30), besonders entlang von Straßenrändern und
  an Positionen bewegter Objekte
- Hochfrequente Details nicht korrekt rekonstruiert
- `dist=0.003-0.004` im Training (höher als erwartet, war 0.0005 ohne Gated Skip)
  — durch Gate-Modul veränderte Channel-Statistiken, noch nicht vollständig
  eingeschwungen

### Warum das Rauschen erwartet ist

Das Differenz-Bild zeigt räumlich strukturiertes Rauschen — kein zufälliges.
Fehler konzentrieren sich auf:
1. Positionen bewegter Objekte (Fahrzeuge): t+1 ≠ t, Δ ist groß
2. Straßenränder: feine Kantendetektion schwierig bei 4× Downsampling

Bei nur ~64 Train-Szenen (nuScenes Val-Set, nicht das eigentliche Train-Set)
ist das der erwartete Stand. Das volle Train-Set (700 Szenen) wird das
Rauschen deutlich reduzieren.

---

## Technische Infrastruktur

### Windows vs. Linux

| Aspekt | Windows | Linux (Ubuntu) |
|---|---|---|
| GPU-Erkennung | RTX 3090 (nach CUDA-Install) | TITAN RTX (direkt) |
| DataLoader | num_workers=0 (stabil) | num_workers=4 |
| HDF5 | langsamer als .npy | .npy mit RAM-Cache schneller |
| Empfehlung | Entwicklung/Smoke-Tests | Training |

### PyTorch-Kompatibilität (Python 3.8, Linux)

Auf dem Linux-System läuft Python 3.8, was einige Anpassungen erforderte:

```bash
pip3 install torch==2.1.2 torchvision==0.16.2 --index-url .../cu121 --no-deps
pip3 install typing-extensions==4.7.1 --ignore-requires-python

# API-Fixes für PyTorch 2.1.2:
torch.cuda.amp.GradScaler(enabled=...)   # statt torch.amp.GradScaler("cuda",...)
torch.cuda.amp.autocast(enabled=...)     # statt torch.amp.autocast("cuda",...)
```

Diese Fixes wurden in `train_linux.py` permanent eingebaut — `train.py`
(Windows) verwendet die neue API.

---

## Offene Punkte & Erkannte Probleme

| Problem | Ursache | Lösung (geplant) |
|---|---|---|
| Rauschen im Differenz-Plot | Zu wenig Train-Daten (64 Szenen) | Volles Train-Set (700 Szenen), Task 8 |
| dist=0.003 (höher als erwartet) | Gated Skip verändert Statistiken | Mehr Epochs / lambda_dist=0.2, Task 8 |
| SSIM-Term zu stark (λ=0.1) | Offset erhöht Val-Loss Skala | λ=0.01 als Ablation, Task 8 |
| Nur Val-Set trainiert | Kein Zugang zu Train-Set | Server-Training, Task 7 |
| wandb nicht installiert | Fehlende Installation | `pip install wandb` auf Server |

---

## Checkpoints

```
checkpoints/phase1/best_val_loss.pt   Val-Loss=0.059146  Phase 1 (Frame)
checkpoints/phase2/best_val_loss.pt   Val-Loss=0.036193  Phase 2 (Cell) ← Hauptergebnis
```

---

## Abhängigkeiten

```
torch==2.1.2+cu121   (Linux Python 3.8)
torchvision==0.16.2
h5py==3.11.0
pyyaml
matplotlib
pytorch-msssim       (für SSIM Loss)
numpy
```
