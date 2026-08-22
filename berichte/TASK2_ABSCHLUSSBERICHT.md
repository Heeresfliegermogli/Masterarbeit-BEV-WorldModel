# Task 2 — Downsampling & Token Embedding: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)  
**Status:** ✅ Vollständig abgeschlossen  

---

## Projektstruktur

```
Task2/
├── model/
│   ├── __init__.py
│   ├── config.py            # ModelConfig dataclass — zentrale Konfiguration
│   ├── downsampling.py      # DownsamplingModule (Phase 1 + 2 identisch)
│   └── embedding.py         # FrameLevelEmbedding (Phase 1) + CellLevelEmbedding (Phase 2)
├── test_downsampling.py     # 6 Unit Tests für DownsamplingModule
└── test_embedding.py        # 7 + 8 Unit Tests für beide Embedding-Varianten
```

---

## Parameter-Übersicht

| Modul | Parameter | Shape Input → Output |
|---|---|---|
| `DownsamplingModule` | **0** | `[B, T, 256, 128, 128]` → `[B, T, 256, 32, 32]` |
| `FrameLevelEmbedding` | **66.560** | `[B, 3, 256, 32, 32]` → `[B, 3, 256]` |
| `CellLevelEmbedding` | **74.752** | `[B, 3, 256, 32, 32]` → `[B, 3072, 256]` |

---

## Token-Vergleich Phase 1 vs. Phase 2

| | Phase 1 (Frame-Level) | Phase 2 (Cell-Level) |
|---|---|---|
| Tokens pro Sample | 3 | 3072 |
| Attention-Einträge | 3² = 9 | 3072² = 9.437.184 |
| Räumliche Info | verloren (GlobalAvgPool) | erhalten (pro Zelle) |
| Batch-Size | 16 | 4 (VRAM-Limit) |
| Wissenschaftlicher Beitrag | Baseline | **Kernbeitrag** |

---

## Implementierte Module

### `model/config.py`

**`ModelConfig`** — Zentrale Konfiguration als Python `@dataclass`:

```python
@dataclass
class ModelConfig:
    d_model:   int   = 256      # Token-Dimension (= BEVFusion Channels)
    n_frames:  int   = 3        # Anzahl Input-Frames
    grid_size: int   = 32       # Räumliche Auflösung nach Downsampling
    n_heads:   int   = 8        # Attention-Heads (d_head = 256/8 = 32)
    d_ff:      int   = 1024     # FFN innere Dimension (4 × d_model)
    n_layers:  int   = 4        # Transformer-Tiefe (Ablation: 2, 4, 6)
    dropout:   float = 0.1      # Dropout-Rate
    phase:     str   = "frame"  # "frame" (Phase 1) oder "cell" (Phase 2)
```

Properties: `d_head` (= `d_model // n_heads`), `n_tokens` (phase-abhängig: 3 oder 3072), `n_spatial_tokens` (= `grid_size²` = 1024).

`__post_init__` validiert: `d_model % n_heads == 0`, `phase ∈ {"frame", "cell"}`, `n_frames ≥ 2`, `dropout ∈ [0, 1)`.

### `model/downsampling.py`

**`DownsamplingModule`** — Räumliches Downsampling via `nn.AvgPool2d`:

- `kernel_size=4, stride=4`: 128 → 32, kein Padding, kein Overlap
- 0 lernbare Parameter — reine Mittelung, keine Transformation der BEVFusion-Latents
- Interner `view`-Trick: faltet `[B, T, C, H, W]` → `[B*T, C, H, W]` für AvgPool, danach zurück
- Reduktion Attention-Matrix (Phase 2): 49.152² → 3.072² = **256× kleiner**

### `model/embedding.py`

**`FrameLevelEmbedding`** — Phase 1, 3 Tokens:
- `nn.AdaptiveAvgPool2d(1)`: `[C, 32, 32]` → `[C]` pro Frame (räumliche Info komprimiert)
- `nn.Linear(256, 256)`: gelernte Umprojektion in Transformer-Raum
- `nn.Embedding(n_frames, d_model)`: Temporal PE — gelernter Vektor pro Frame-Position

**`CellLevelEmbedding`** — Phase 2, 3072 Tokens:
- Flatten: `[C, 32, 32]` → `[1024, C]` pro Frame (räumliche Struktur erhalten)
- `nn.Linear(256, 256)`: Projektion
- `nn.Embedding(grid_size, d_model//2)` × 2: faktorisiertes räumliches PE (PE_x ⊕ PE_y)
- `nn.Embedding(n_frames, d_model)`: Temporal PE
- Token-Reihenfolge: `[0..1023]` = Frame t-2, `[1024..2047]` = Frame t-1, `[2048..3071]` = Frame t

---

## Modularität & Phasen-Umschaltung

Alle Module sind **phase-agnostisch** oder explizit für beide Phasen ausgelegt. Der gesamte Downstream-Code (Transformer, Training Loop) muss nie geändert werden — nur zwei Zeilen in `BEVWorldModel.__init__()` (Task 4):

```python
# Phase 1 → Phase 2: nur diese zwei Zeilen tauschen
self.embedding   = FrameLevelEmbedding(config)   # → CellLevelEmbedding(config)
self.output_head = FrameLevelOutputHead()         # → CellLevelOutputHead()
```

`DownsamplingModule` und `TransformerEncoder` (Task 3) sind **identisch für beide Phasen**.

---

## Hauptskript-Integration (Vorschau Task 4/5)

Das spätere Hauptskript `train.py` instanziiert alle Module über `ModelConfig`:

```python
from model.config      import ModelConfig
from model.downsampling import DownsamplingModule
from model.embedding    import FrameLevelEmbedding, CellLevelEmbedding

# Umschalten via config — kein weiterer Code ändert sich:
cfg = ModelConfig(phase="frame")   # Phase 1
cfg = ModelConfig(phase="cell")    # Phase 2

# Instanziierung:
downsample = DownsamplingModule()
embedding  = FrameLevelEmbedding(cfg) if cfg.phase == "frame" else CellLevelEmbedding(cfg)
```

Aufruf des Hauptskripts (geplant):
```powershell
python train.py --config config.yaml --phase frame   # Phase 1
python train.py --config config.yaml --phase cell    # Phase 2
```

---

## Design-Entscheidungen

**Warum `AvgPool` statt `MaxPool` oder Strided Conv für Downsampling?**
AvgPool erhält die globale Aktivierungsstärke einer Region ohne lernbare Parameter einzuführen. MaxPool verliert schwache aber vorhandene Aktivierungen. Strided Conv würde die BEVFusion-Latents transformieren — wir wollen sie möglichst unverändert lassen, nur komprimieren.

**Warum `GlobalAvgPool` in Phase 1 statt Flatten?**
Flatten von `[256, 32, 32]` → `[262.144]` würde `Linear(262144, 256)` benötigen: 67 Mio. Parameter nur für das Embedding. `GlobalAvgPool` + `Linear(256, 256)` hat 66.560 Parameter. Trade-off: räumliche Info geht verloren — das ist der bewusste Kompromiss von Phase 1, den Phase 2 durch Cell-Tokens aufhebt.

**Warum PE_x ⊕ PE_y statt voller 2D-Tabelle in Phase 2?**
Eine 2D-Tabelle hätte `32×32 × 256 = 262.144` Parameter. Faktorisiert: `2 × 32 × 128 = 8.192` Parameter — Faktor 32 sparsamer, trotzdem vollständig. Zusätzlich kann das Modell x- und y-Richtung getrennt lernen.

**Warum `H_out` aus dem Ergebnis lesen statt hardcoden?**
`_, _, H_out, W_out = x.shape` statt `x.view(B, T, C, 32, 32)` — das Downsampling-Modul funktioniert damit für beliebige `kernel_size`/`stride`-Konfigurationen ohne Änderung. Wichtig für Ablation Studies.

**Warum `view()` statt `reshape()`?**
`view()` schlägt fehl bei nicht-kontigem Speicher-Layout — das ist Absicht. Es macht Speicher-Layout-Fehler sofort sichtbar statt sie still zu korrigieren. Der DataLoader garantiert kontigues Layout (`torch.from_numpy` auf C-order Arrays).

---

## Test-Ergebnisse (lokal, PyTorch)

### `test_downsampling.py` — 6/6 Tests bestanden
```
[OK] Test 1 — Output Shape: torch.Size([4, 3, 256, 32, 32])
[OK] Test 2 — Batch-Isolation
[OK] Test 3 — Frame-Isolation
[OK] Test 4 — Pooling-Korrektheit: 13.500 == 13.500
[OK] Test 5 — Parameter-Count: 0
[OK] Test 6 — Kein NaN / Inf
```

### `test_embedding.py` — 15/15 Tests bestanden
```
[OK] Frame Test 1–7: Shape, Token-Count, Gradient, NaN, Batch, PE_t, Params ✓
[OK] Cell  Test 1–8: Shape, Token-Count, Gradient, NaN, Batch, PE_xy, Boundaries, Params ✓
```

---

## Verwendung in Folge-Tasks

```python
# Minimaler Import für Task 3 (Transformer) und Task 4 (BEVWorldModel):
from model.config       import ModelConfig
from model.downsampling import DownsamplingModule
from model.embedding    import FrameLevelEmbedding, CellLevelEmbedding

cfg = ModelConfig(phase="frame")   # oder "cell"

downsample = DownsamplingModule()
embedding  = FrameLevelEmbedding(cfg) if cfg.phase == "frame" else CellLevelEmbedding(cfg)

# Ein Batch durch die Task-2-Pipeline:
# batch["input"]: [B, 3, 256, 128, 128]  ← aus Task 1 DataLoader
x = batch["input"]
x = downsample(x)   # [B, 3, 256, 32, 32]
x = embedding(x)    # [B, 3, 256]  oder  [B, 3072, 256]
# → bereit für Transformer (Task 3)
```

---

## Abhängigkeiten

```
torch        # nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Embedding
dataclasses  # stdlib — für ModelConfig
```
