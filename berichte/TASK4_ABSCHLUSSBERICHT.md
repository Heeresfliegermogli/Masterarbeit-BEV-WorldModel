# Task 4 — Output Head, Upsampling & BEVWorldModel: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)  
**Status:** ✅ Vollständig abgeschlossen  

---

## Projektstruktur

```
Code_final/
│
│  ── Daten-Pipeline (Task 1) ──────────────────────────────
├── bev_dataset.py          # Dataset-Klasse
├── bev_dataloader.py       # DataLoader + Train/Val/Test Split
├── build_hdf5_cache.py     # BEVFusion Inference → HDF5
│
│  ── Modell-Module (Tasks 2–4) ────────────────────────────
├── config.py               # ModelConfig dataclass
├── downsampling.py         # DownsamplingModule
├── embedding.py            # FrameLevelEmbedding, CellLevelEmbedding
├── transformer.py          # TransformerBlock, TransformerEncoder
├── output_head.py          # FrameLevelOutputHead, CellLevelOutputHead  ← NEU
├── upsampling_head.py      # UpsamplingHead                             ← NEU
├── bev_world_model.py      # BEVWorldModel (komplettes Modell)          ← NEU
│
│  ── Training & Checkpoints (Task 5) ──────────────────────
├── checkpointing.py        # save/load Checkpoint
│
│  ── Tests ────────────────────────────────────────────────
├── test_transformer.py     # Tests Task 3
└── test_output.py          # Tests Task 4 (6/6 grün)                   ← NEU
```

**Wichtiger Hinweis zu Imports:**  
Die Dateien `embedding.py`, `transformer.py` und `checkpointing.py` hatten noch
alte Imports aus einer `model/`-Unterordner-Struktur. Diese wurden auf die
flache Struktur korrigiert:
```python
# alt:  from model.config import ModelConfig
# neu:  from config import ModelConfig
```

---

## Parameter-Übersicht (gemessen, Task 4 Selbst-Test)

### Phase 1 — Frame-Level

| Modul | Parameter | Shape Input → Output |
|---|---|---|
| `DownsamplingModule` | **0** | `[B, 3, 256, 128, 128]` → `[B, 3, 256, 32, 32]` |
| `FrameLevelEmbedding` | **66.560** | `[B, 3, 256, 32, 32]` → `[B, 3, 256]` |
| `TransformerEncoder` (N=4) | **3.159.040** | `[B, 3, 256]` → `[B, 3, 256]` |
| `FrameLevelOutputHead` | **65.792** | `[B, 3, 256]` → `[B, 256, 32, 32]` |
| `UpsamplingHead` | **2.688.768** | `[B, 256, 32, 32]` → `[B, 256, 128, 128]` |
| **Gesamt Phase 1** | **5.980.160** | `[B, 3, 256, 128, 128]` → `[B, 256, 128, 128]` |

### Phase 2 — Cell-Level

| Modul | Parameter | Shape Input → Output |
|---|---|---|
| `DownsamplingModule` | **0** | `[B, 3, 256, 128, 128]` → `[B, 3, 256, 32, 32]` |
| `CellLevelEmbedding` | **74.752** | `[B, 3, 256, 32, 32]` → `[B, 3072, 256]` |
| `TransformerEncoder` (N=4) | **3.159.040** | `[B, 3072, 256]` → `[B, 3072, 256]` |
| `CellLevelOutputHead` | **65.792** | `[B, 3072, 256]` → `[B, 256, 32, 32]` |
| `UpsamplingHead` | **2.688.768** | `[B, 256, 32, 32]` → `[B, 256, 128, 128]` |
| **Gesamt Phase 2** | **5.988.352** | `[B, 3, 256, 128, 128]` → `[B, 256, 128, 128]` |

---

## Implementierte Module

### `output_head.py`

Enthält beide Output-Heads — eine pro Phase. Beide liefern denselben Output
`[B, 256, 32, 32]`, damit der `UpsamplingHead` danach phase-agnostisch bleibt.

---

**`FrameLevelOutputHead`** — Phase 1:

```
[B, 3, 256]         Transformer Output (3 Frame-Tokens)
    ↓  [:, -1, :]   letzter Token = Frame t (hat t-2, t-1 via Attention gesehen)
[B, 256]
    ↓  Linear(256→256) + GELU
[B, 256]
    ↓  unsqueeze(-1).unsqueeze(-1)
[B, 256, 1, 1]
    ↓  Bilinear Upsample → (32, 32)   parameterfrei, stabil
[B, 256, 32, 32]
```

Warum nur der letzte Token?  
Der Transformer hat kein Causal Masking — jeder Token hat alle anderen gesehen.
Token t ist die kompakteste Zusammenfassung des aktuellen Szenen-Zustands
und damit die natürliche Basis für die t+1 Prediction.

Warum Bilinear statt ConvTranspose?  
Von `[1,1]` auf `[32,32]` ist Faktor 32 in einem Schritt — zu groß für einen
stabilen ConvTranspose. Bilinear ist parameterfrei und gibt dem `UpsamplingHead`
danach eine saubere Startbasis.

---

**`CellLevelOutputHead`** — Phase 2:

```
[B, 3072, 256]       Transformer Output (3072 Cell-Tokens)
    ↓  [:, -1024:, :] letzte 1024 Tokens = Frame t
[B, 1024, 256]
    ↓  Linear(256→256) + GELU
[B, 1024, 256]
    ↓  permute(0, 2, 1)
[B, 256, 1024]
    ↓  reshape(B, 256, 32, 32)
[B, 256, 32, 32]
```

Token-Aufteilung im Transformer Output:
```
Tokens    0..1023  → Frame t-2
Tokens 1024..2047  → Frame t-1
Tokens 2048..3071  → Frame t   ← [:, -1024:, :]
```

Warum `[:, -1024:, :]` statt `[:, 2048:3072, :]`?  
Ablation-Sicherheit: bei `n_frames=2` gibt es nur 2048 Tokens, bei `n_frames=4`
sind es 4096. "Letzte 1024 Tokens" = immer Frame t, unabhängig von `n_frames`.

Warum kein Bilinear Upsample wie in Phase 1?  
In Phase 2 hat jeder Token durch `PE_xy` eine genaue räumliche Position.
Das Reshape von `[B, 1024, 256]` → `[B, 256, 32, 32]` ist verlustfrei —
wir ordnen nur um. Das ist der wissenschaftliche Kern von Phase 2:
echtes spatio-temporales Reasoning ohne Informationsverlust durch Pooling.

---

### `upsampling_head.py`

**`UpsamplingHead`** — identisch für Phase 1 und Phase 2:

```
[B, 256,  32,  32]
    ↓  ConvTranspose2d(256→256, k=4, stride=2, pad=1) + BatchNorm + GELU
[B, 256,  64,  64]
    ↓  ConvTranspose2d(256→256, k=4, stride=2, pad=1) + BatchNorm + GELU
[B, 256, 128, 128]
    ↓  Conv2d(256→256, k=3, pad=1)   — Refinement, keine Größenänderung
[B, 256, 128, 128]
```

Warum zwei ConvTranspose statt einem?  
Faktor 4 in einem Schritt (k=8, stride=4) erzeugt Checkerboard-Artefakte und
ist schwer zu optimieren. Zwei Schritte à Faktor 2 (k=4, stride=2, pad=1) sind
der Decoder-Standard (U-Net, DCGAN) und exakt berechenbar:
```
Output = (Input - 1) * stride - 2*padding + kernel
       = (32 - 1) * 2 - 2 + 4 = 64  ✓
       = (64 - 1) * 2 - 2 + 4 = 128 ✓
```

Warum BatchNorm nach ConvTranspose?  
ConvTranspose kann intern sehr unterschiedliche Aktivierungsskalen erzeugen.
BatchNorm normalisiert auf Mittelwert≈0, Std≈1 — stabilisiert das Training.

Warum kein Sigmoid/Tanh am Ende?  
BEVFusion-Latents haben keinen festen Wertebereich. Ein Clipping würde den
MSE-Loss verfälschen — lineare Ausgabe ist korrekt.

---

### `bev_world_model.py`

**`BEVWorldModel`** — orchestriert alle 5 Stufen:

```python
def forward(self, x):   # x: [B, 3, 256, 128, 128]
    x = self.downsample(x)    # [B, 3, 256, 32, 32]
    x = self.embedding(x)     # [B, 3, 256]  oder  [B, 3072, 256]
    x = self.transformer(x)   # gleiche Shape
    x = self.output_head(x)   # [B, 256, 32, 32]
    x = self.upsample(x)      # [B, 256, 128, 128]
    return x
```

Phase wechseln — nur eine Zeile in der Config:
```python
cfg = ModelConfig(phase="frame")   # Phase 1
cfg = ModelConfig(phase="cell")    # Phase 2
model = BEVWorldModel(cfg)
```

`__init__` wählt automatisch die richtigen Klassen:
```python
if config.phase == "frame":
    self.embedding   = FrameLevelEmbedding(config)
    self.output_head = FrameLevelOutputHead(config)
else:
    self.embedding   = CellLevelEmbedding(config)
    self.output_head = CellLevelOutputHead(config)
# downsample, transformer, upsample: immer identisch
```

Hilfsmethode `count_parameters()` gibt Parameter pro Modul als dict zurück —
direkt verwendbar für die Thesis-Tabelle.

---

## Test-Ergebnisse

### `test_output.py` — 6/6 Tests bestanden

```
TEST 1 — FrameLevelOutputHead: Shape
  Input:  [4, 3, 256]
  Output: [4, 256, 32, 32]
  ✓ Shape korrekt: [4, 256, 32, 32]
  ✓ Kein NaN / Inf

TEST 2 — FrameLevelOutputHead: Gradient-Flow
  ✓ Gradient-Flow durch alle 2 Parameter-Tensoren
  ✓ Parameter gesamt: 65,792

TEST 3 — CellLevelOutputHead: Shape
  Input:  [4, 3072, 256]  (= 3 Frames × 32² Zellen)
  Output: [4, 256, 32, 32]
  ✓ Shape korrekt: [4, 256, 32, 32]
  ✓ Kein NaN / Inf

TEST 4 — CellLevelOutputHead: Gradient-Flow + Ablation n_frames=2
  ✓ Gradient-Flow durch alle 2 Parameter-Tensoren
  ✓ Ablation n_frames=2: [4, 2048, 256] → [4, 256, 32, 32]

TEST 5 — UpsamplingHead: Shape (Phase 1 und Phase 2)
  Phase 1: [4, 256, 32, 32] → [4, 256, 128, 128]  ✓
  Phase 2: [4, 256, 32, 32] → [4, 256, 128, 128]  ✓
  ✓ Parameter gesamt: 2,688,768

TEST 6 — End-to-End: Output Head → Upsampling Head
  Phase 1: [4, 256, 128, 128]  ✓  Gradient-Flow durch alle Module
  Phase 2: [4, 256, 128, 128]  ✓  Gradient-Flow durch alle Module

✅  Alle 6 Tests bestanden.
```

### `bev_world_model.py` Selbst-Test — bestanden

```
Phase 'frame': [2, 3, 256, 128, 128] → [2, 256, 128, 128]  ✓  5.980.160 Parameter
Phase 'cell':  [2, 3, 256, 128, 128] → [2, 256, 128, 128]  ✓  5.988.352 Parameter
✅  Selbst-Test bestanden.
```

---

## Design-Entscheidungen

**Warum output_head.py und upsampling_head.py getrennt?**  
Klare Verantwortlichkeiten: `output_head.py` übersetzt Tokens → Grid,
`upsampling_head.py` skaliert Grid → Originalauflösung. Wenn der Upsampler
ausgetauscht wird (z.B. für Ablation mit mehr Schichten), bleibt `output_head.py`
unberührt.

**Warum BEVWorldModel in einer eigenen Datei?**  
`BEVWorldModel` hat eine andere Aufgabe als die Einzelmodule — es orchestriert,
nicht transformiert. Saubere Trennung erleichtert Task 5 (Training Loop):
`train.py` importiert nur `BEVWorldModel` und `ModelConfig`, ohne die internen
Module zu kennen.

**Warum flache Ordnerstruktur?**  
~12 Dateien, ein Trainingsskript, ein GPU-Server — eine `model/`-Unterordner-
Struktur würde alle Imports verlängern ohne Mehrwert. Im Forschungskontext
ist Einfachheit wichtiger als Package-Struktur.

---

## Verwendung in Task 5 (Training Loop)

```python
# train.py — minimaler Imports-Block
from config          import ModelConfig
from bev_world_model import BEVWorldModel

# Modell instanziieren
cfg   = ModelConfig(phase="frame")        # oder "cell" für Phase 2
model = BEVWorldModel(cfg).to(device)

# Forward Pass + Loss
pred   = model(batch["input"])            # [B, 256, 128, 128]
target = batch["target"]                  # [B, 256, 128, 128]
loss   = F.mse_loss(pred, target)
```

---

## Offene Punkte / Vorarbeiten für folgende Tasks

| Task | Was fehlt noch |
|---|---|
| Task 5 | `train.py`: Training Loop, Loss, Optimizer, Scheduler, wandb Logging |
| Task 6 | `config.phase = "cell"` setzen, VRAM prüfen, Training starten |
| Task 7 | Ablation: `n_layers`, `n_frames`, Phase 1 vs 2, PE an/aus, Loss-Varianten |
| Task 8 | Evaluation auf Test-Set, Visualisierungen, Thesis-Materialien |

---

## Abhängigkeiten

```
torch       # nn.Module, ConvTranspose2d, F.interpolate
dataclasses # stdlib — ModelConfig
```

Alle anderen Imports (`config`, `downsampling`, `embedding`, `transformer`,
`output_head`, `upsampling_head`) sind projekt-interne Module — kein pip nötig.
