# Task 3 — Transformer-Kern: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)  
**Status:** ✅ Vollständig abgeschlossen  

---

## Projektstruktur

```
Task2/                              ← Tasks 1–3 leben gemeinsam hier
├── model/
│   ├── __init__.py
│   ├── config.py                   # ModelConfig dataclass (Task 2)
│   ├── downsampling.py             # DownsamplingModule (Task 2)
│   ├── embedding.py                # FrameLevelEmbedding + CellLevelEmbedding (Task 2)
│   ├── transformer.py              # TransformerBlock + TransformerEncoder (Task 3) ← NEU
│   └── checkpointing.py            # save/load Checkpoint-Logik (Vorarbeit Task 5) ← NEU
├── test_downsampling.py
├── test_embedding.py
├── test_transformer.py             # 8 Unit Tests für Transformer (Task 3) ← NEU
└── training_sketch.py              # Checkpoint-Logik im Kontext (Vorarbeit Task 5) ← NEU
```

---

## Parameter-Übersicht

| Modul | Parameter | Shape Input → Output |
|---|---|---|
| `TransformerBlock` (N=1) | **789.760** | `[B, seq_len, 256]` → `[B, seq_len, 256]` |
| `TransformerEncoder` (N=2) | **1.579.520** | `[B, seq_len, 256]` → `[B, seq_len, 256]` |
| `TransformerEncoder` (N=4) | **3.159.040** | `[B, seq_len, 256]` → `[B, seq_len, 256]` |
| `TransformerEncoder` (N=6) | **4.738.560** | `[B, seq_len, 256]` → `[B, seq_len, 256]` |

`seq_len` ist phase-agnostisch: 3 (Phase 1) oder 3072 (Phase 2) — identischer Code.

### Aufschlüsselung pro Block (d_model=256, n_heads=8, d_ff=1024)

| Komponente | Parameter |
|---|---|
| `nn.MultiheadAttention` (Q, K, V + Output-Proj.) | 263.168 |
| `LayerNorm` × 2 | 1.024 |
| FFN `Linear(256→1024)` + `Linear(1024→256)` | 525.312 |
| Dropout | 0 |
| **Gesamt pro Block** | **789.504** |

---

## Implementierte Module

### `model/transformer.py`

**`TransformerBlock`** — ein einzelner Block in Pre-LayerNorm Variante:

```
x = x + Dropout(Attention(LayerNorm(x)))   ← Attention-Zweig
x = x + Dropout(FFN(LayerNorm(x)))         ← FFN-Zweig
```

Komponenten im Detail:
- `nn.LayerNorm(d_model)` × 2: Pre-LN vor Attention und vor FFN
- `nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)`: kein Masking, alle Tokens sehen alle Tokens
- `nn.Dropout(0.1)` × 2: auf Attention-Output und FFN-Output
- FFN: `Linear(256→1024)` + `GELU` + `Dropout(0.1)` + `Linear(1024→256)`
- Hilfsmethode `get_attention_weights()`: gibt `[B, n_heads, seq_len, seq_len]` zurück — nur für Visualisierung/Tests, nicht im Training-Forward-Pass

**`TransformerEncoder`** — Stack aus N Blöcken:

- `nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])`: jeder Block hat eigene, ungeteilte Parameter
- `forward()`: einfache for-Schleife über alle Blöcke, Shape ändert sich nie
- `get_all_attention_weights()`: gibt Attention-Gewichte aller Blöcke zurück (für Ablation/Thesis)

### `model/checkpointing.py` (Vorarbeit Task 5)

Wird in Task 5 vom Training Loop importiert. Enthält:

**`save_checkpoint()`** — speichert vollständigen Trainings-Snapshot:
```python
{
    "model_state_dict":     ...,   # alle Weights + Buffer
    "optimizer_state_dict": ...,   # AdamW Momentum + Variance
    "scheduler_state_dict": ...,   # CosineAnnealingLR Position
    "epoch":                int,
    "val_loss":             float,
    "best_val_loss":        float,
    "config":               dict,  # ModelConfig als plain dict
    "phase":                str,   # "frame" oder "cell"
    "timestamp":            str,
}
```

**`load_checkpoint()`** — lädt Checkpoint mit Phasen-Sicherheitscheck:
- Prüft ob `checkpoint["phase"] == model.config.phase` bevor Weights geladen werden
- `strict=True`: alle Parameter müssen übereinstimmen, kein silent mismatch
- `optimizer` und `scheduler` optional — falls `None`, nur Modell laden (für Inference)

**`find_best_checkpoint()`** — gibt Pfad zu `checkpoints/phase{1|2}/best_val_loss.pt` zurück.

---

## Checkpoint-Dateistruktur (nach Training Task 5)

```
checkpoints/
├── phase1/
│   ├── best_val_loss.pt      ← bestes Val-Loss Modell (wird überschrieben)
│   ├── epoch_010.pt          ← periodischer Snapshot
│   └── epoch_020.pt
└── phase2/
    ├── best_val_loss.pt
    └── epoch_010.pt
```

Phase 1 und Phase 2 Checkpoints sind **nicht kompatibel** — das Embedding-Modul hat in Phase 2 andere Shapes (`pe_x [32,128]`, `pe_y [32,128]` zusätzlich). `load_checkpoint()` fängt Mismatch-Versuche mit klarer Fehlermeldung ab.

---

## Phasen-Agnostizität des Transformers

Das ist der zentrale Designpunkt von Task 3: `TransformerEncoder` "sieht" nur `[B, seq_len, d_model]`. Er hat keinerlei Annahmen über `seq_len`:

| | Phase 1 | Phase 2 |
|---|---|---|
| seq_len | 3 | 3072 |
| Transformer-Code | identisch | identisch |
| Attention-Matrix | 3×3 = 9 Einträge | 3072×3072 = 9.437.184 Einträge |
| VRAM (N=4, A100) | ~4 GB | ~16 GB |
| Batch-Size | 16 | 4 |

---

## Design-Entscheidungen

**Warum Pre-LayerNorm statt Post-LayerNorm?**
Post-LN (originales Paper): `x = LayerNorm(x + Attention(x))` — instabiler in der Praxis, neigt zu explodierenden Gradienten in tiefen Netzen. Pre-LN: `x = x + Attention(LayerNorm(x))` — normalisiert vor der Rechnung, Gradienten stabiler. Moderner Standard seit GPT-2. Für 6 Wochen Training auf limitiertem VRAM wichtiger als letztes Prozent Performance.

**Warum kein Masking?**
Sprach-Modelle maskieren zukünftige Tokens (t darf t+1 nicht sehen). Bei uns sind alle Input-Frames (t-2, t-1, t) gleichberechtigt bekannt — wir predicten t+1 als Output, nicht als Token in der Sequenz. Kein Masking nötig, alle Tokens dürfen alle anderen sehen.

**Warum `need_weights=False` im Training-Forward-Pass?**
`nn.MultiheadAttention` kann optional die Attention-Gewichte zurückgeben — das kostet aber Speicher und Zeit. Im Training brauchen wir sie nicht. `get_attention_weights()` ist eine separate Methode die explizit aufgerufen wird — nur für Tests und Visualisierungen.

**Warum `nn.ModuleList` statt Python-Liste?**
Eine Python-Liste registriert Module nicht bei PyTorch. Die Parameter würden nicht in `model.parameters()` auftauchen → nicht trainiert. `nn.ModuleList` registriert alle enthaltenen Module korrekt.

**Warum `get_all_attention_weights()` im Encoder?**
Task 8 (Evaluation) soll Attention-Muster visualisieren — welcher Block achtet auf was. Diese Methode gibt die Gewichte aller N Blöcke zurück ohne den Forward-Pass zu verändern. Wird nie im Training aufgerufen.

---

## Test-Ergebnisse (lokal, PyTorch 2.10.0+cpu)

### `test_transformer.py` — 8/8 Tests bestanden

```
============================================================
BEV World Model — Task 3: Transformer Unit Tests
============================================================
PyTorch Version: 2.10.0+cpu
CUDA verfügbar:  False

--- Test 1: Output-Shape Phase 1 (seq_len=3) ---
  [OK] Shape korrekt: torch.Size([4, 3, 256])

--- Test 2: Output-Shape Phase 2 (seq_len=3072) ---
  Input:  torch.Size([2, 3072, 256])
  [OK] Shape korrekt: torch.Size([2, 3072, 256])

--- Test 3: Gradient-Flow ---
  [OK] Alle 3.159.040 Parameter haben Gradienten

--- Test 4: Kein NaN / Inf ---
  [OK] Kein NaN/Inf bei normalem Input
  [OK] Kein NaN/Inf bei extremem Input (×10)

--- Test 5: Parameter-Count ---
  n_layers=2:  1.579.520 Parameter  (~1.58M)
  n_layers=4:  3.159.040 Parameter  (~3.16M)
  n_layers=6:  4.738.560 Parameter  (~4.74M)
  [OK] Parameter-Count ausgegeben

--- Test 6: Attention-Weight Visualisierung (Phase 1) ---
  Attention-Tensor Shape: torch.Size([1, 8, 3, 3])
    [B=1, n_heads=8, seq_len=3, seq_len=3]

  Attention-Matrix (gemittelt über 8 Heads):
          t-2    t-1    t
  t-2  0.376  0.361  0.263
  t-1  0.406  0.261  0.333
  t    0.361  0.296  0.342
  [OK] Zeilensummen ≈ 1.0 ✓  [1.0, 1.0, 1.0]

--- Test 7: VRAM-Messung ---
  [SKIP] Kein GPU verfügbar — VRAM-Test übersprungen

--- Test 8: Training vs. Eval Modus ---
  [OK] Eval-Modus: Outputs identisch bei gleichem Input
  [OK] Train-Modus: Outputs variieren (Dropout aktiv)

============================================================
Alle Tests abgeschlossen.
============================================================
```

**Hinweis Attention-Matrix (Test 6):** Zufällig initialisierte Weights → alle Werte nahe 0.33 (gleichmäßige Aufmerksamkeit), kein gelerntes Muster. Nach Training werden deutlichere Strukturen erwartet — vermutlich starke Gewichtung von t-1 durch t, da zeitlich nächster Frame.

**Hinweis Test 7:** VRAM-Test übersprungen da kein GPU auf lokalem Rechner. Auf A100 erwartet: Phase 1 ~4 GB, Phase 2 ~16 GB (bei N=4).

---

## Verwendung in Folge-Tasks

```python
# Minimaler Import für Task 4 (BEVWorldModel):
from model.config      import ModelConfig
from model.transformer import TransformerEncoder

cfg     = ModelConfig(phase="frame")   # oder "cell"
encoder = TransformerEncoder(cfg)

# Ein Batch durch Tasks 1–3:
# x: [B, 3, 256, 128, 128]  ← aus Task 1 DataLoader
x = downsample(x)    # [B, 3, 256, 32, 32]   — Task 2
x = embedding(x)     # [B, 3, 256]            — Task 2 (Phase 1)
x = encoder(x)       # [B, 3, 256]            — Task 3
# → bereit für Output Head + Upsampling (Task 4)

# Checkpoint laden (nach Training, Task 5):
from model.checkpointing import load_checkpoint
checkpoint = load_checkpoint("checkpoints/phase1/best_val_loss.pt", model)
```

---

## Offene Punkte / Vorarbeiten für folgende Tasks

| Task | Was fehlt noch |
|---|---|
| Task 4 | `FrameLevelOutputHead`, `CellLevelOutputHead`, `UpsamplingHead`, `BEVWorldModel` |
| Task 5 | Training Loop, Loss, Optimizer, Scheduler — nutzt `checkpointing.py` direkt |
| Task 7 | `get_all_attention_weights()` für Ablation-Visualisierungen nutzen |
| Task 8 | VRAM-Messung auf A100 nachholen (Test 7 war SKIP) |

---

## Abhängigkeiten

```
torch       # nn.MultiheadAttention, nn.LayerNorm, nn.Linear, nn.ModuleList, nn.Dropout
dataclasses # stdlib — für asdict() in checkpointing.py
pathlib     # stdlib — für Checkpoint-Pfade
```
