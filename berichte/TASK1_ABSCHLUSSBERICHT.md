# Task 1 — Infrastruktur & Daten-Pipeline: Abschlussbericht

**Projekt:** BEV World Model (Masterarbeit)  
**Status:** ✅ Vollständig abgeschlossen  

---

## Projektstruktur

```
Task1/
├── data/
│   ├── bev_dataset.py          # Dataset-Klasse + Config
│   ├── bev_dataloader.py       # DataLoader-Factory (split + explicit)
│   └── build_hdf5_cache.py     # Optionaler HDF5-Cache-Builder
├── sanity_check.py             # Daten-Sanity-Check mit Visualisierung
├── check_missing.py            # Hilfsskript: fehlende .npy finden
└── sanity_check_output/
    ├── val/
    │   ├── sequences_val.png
    │   └── distribution_val.png
    └── train/
        ├── sequences_train.png
        └── distribution_train.png
```

---

## Datensatz (nuScenes Mini)

| Split | Szenen | Frames | Sequenzen (n=3 Input) |
|-------|--------|--------|----------------------|
| Train | 8      | 304    | 277                  |
| Val   | 2      | 81     | 75                   |

- **Aufnahmefrequenz:** 2 Hz (Keyframes)
- **Szenenlänge:** ~39–41 Frames pro Szene (~20s)
- **Fehlende Latents Train:** 19/323 (automatisch übersprungen)
- **Übersprungene Szenen:** 1 (nur 3 Frames nach Filterung — zu kurz für 4er-Fenster)

---

## Latent-Statistiken

| Metrik | Val    | Train  |
|--------|--------|--------|
| min    | 0.000  | 0.000  |
| max    | 11.406 | 15.648 |
| mean   | 0.141  | 0.147  |
| std    | 0.276  | 0.279  |

**Wichtige Erkenntnisse:**
- Shape aus `.npy`: `(1, 256, 128, 128)` — Batch-Dim wird via `squeeze(axis=0)` entfernt
- ReLU-aktiviert: min=0.0, keine negativen Werte
- Sehr sparse: Mean 0.14 bei Max ~15 → großer Peak bei 0 im Histogramm
- 256/256 Channels aktiv (keine toten Channels)
- Train/Val aus gleicher Verteilung — kein Distribution Shift

**Implikation für Training:** Die starke Sparsity bedeutet dass MSE-Loss alleine
dazu neigt "alles auf 0 setzen" als Strategie. Der Cosine-Similarity-Zusatzterm
(λ=0.1, siehe Task 5) ist deshalb wichtig.

---

## Implementierte Module

### `data/bev_dataset.py`

**`BEVDatasetConfig`** — Zentrale Konfiguration:
```python
@dataclass
class BEVDatasetConfig:
    sources: List[Tuple[str, str]]  # [(pkl_path, latent_dir), ...]
    n_input_frames: int = 3
    scene_gap_sec: float = 2.0
    h5_path: Optional[str] = None   # None = .npy Modus, Pfad = HDF5 Modus
```

**`BEVLatentDataset`** — PyTorch Dataset:
- Liest PKL(s), erkennt Szenen anhand Zeitlücken (>2s = neue Szene)
- Baut Sliding-Windows der Länge `n_input_frames + 1`
- Sequenzen werden **nie** über Szenengrenzen gebildet
- `scene_indices` Parameter für Train/Val/Test-Split ohne Code-Duplikation
- Filtert automatisch Frames ohne vorhandene `.npy`-Datei
- Zwei Lademodi: `.npy` direkt oder HDF5-Cache (thread-safe via `threading.local()`)
- `__getitem__` gibt zurück:
  ```python
  {
      "input":        Tensor [n_input_frames, 256, 128, 128],
      "target":       Tensor [256, 128, 128],
      "input_tokens": List[str],
      "target_token": str,
  }
  ```

### `data/bev_dataloader.py`

**Modus 1 — `make_dataloaders_split()`:** Eine PKL, Split auf Szenen-Ebene.
```python
loaders = make_dataloaders_split(
    sources=[("infos.pkl", "/Latents")],
    train_ratio=0.70, val_ratio=0.15,
    batch_size=16, seed=42,
)
# -> {"train": DataLoader, "val": DataLoader, "test": DataLoader|None}
```

**Modus 2 — `make_dataloaders_explicit()`:** Separate PKLs pro Split (aktueller Fall).
```python
loaders = make_dataloaders_explicit(
    train_sources=[("infos_train.pkl", "/Latents/Train")],
    val_sources  =[("infos_val.pkl",   "/Latents/Val")],
    batch_size=16,
)
```

### `data/build_hdf5_cache.py`

Einmaliger Konverter `.npy` → HDF5. Für Mini nicht nötig, relevant ab vollem
nuScenes (~28.000 Frames). Aufruf:
```powershell
python data/build_hdf5_cache.py `
    --pkl "...nuscenes_infos_train.pkl" `
    --npy "...Latents/Train" `
    --out "...Cache/bev_latents_train.h5"
```

### `sanity_check.py`

Vollständiger Daten-Check mit 4 Stufen:
```powershell
# Val + Train:
python sanity_check.py `
    --val-pkl   ".../nuscenes_infos_val.pkl" `
    --val-npy   ".../Latents/Val" `
    --train-pkl ".../nuscenes_infos_train.pkl" `
    --train-npy ".../Latents/Train"

# Nur Val:
python sanity_check.py `
    --val-pkl ".../nuscenes_infos_val.pkl" `
    --val-npy ".../Latents/Val"
```

| Check | Was geprüft wird |
|-------|-----------------|
| 1 | Dataset-Instanziierung, Shapes `[3,256,128,128]` + `[256,128,128]`, dtype float32 |
| 2 | Wertebereich min/max/mean/std, NaN, Inf |
| 3 | Keine Sequenz überspannt Szenengrenzen |
| 4 | Heatmap-Visualisierung + Histogramm/Channel-Aktivierung als PNG |

---

## Design-Entscheidungen

**Warum Split auf Szenen-Ebene?**
Sliding-Windows überlappen sich. Würde man Sequenzen direkt splitten,
könnten Frame t+1 im Val und Frame t im Train auftauchen → Data Leakage.
Split auf ganzen Szenen verhindert das vollständig.

**Warum `scene_indices` statt eigener Subset-Klasse?**
Ursprünglich gab es eine separate `_SceneSubsetDataset`-Klasse im DataLoader —
doppelter Code mit identischer Logik. Gelöst durch `scene_indices`-Parameter
direkt in `BEVLatentDataset`. Eine Klasse, drei Rollen: alle Szenen / nur Train / nur Val.

**Warum fehlende `.npy` filtern statt abbrechen?**
BEVFusion-Extraktion kann für einzelne Frames fehlschlagen. 19/323 fehlende
Frames im Train-Set sind ~6% — zu wenig um die PKL neu zu generieren,
zu viele um sie zu ignorieren. Automatisches Filtern mit Info-Ausgabe
ist der sauberste Kompromiss.

**HDF5 optional halten:**
Für Mini (81+304 Frames) ist direktes `.npy`-Laden schnell genug.
HDF5 wird relevant ab vollem nuScenes (~28k Frames, 4+ DataLoader-Worker).
Umschalten: nur `h5_path` in `BEVDatasetConfig` setzen.

---

## Verwendung in Folge-Tasks

```python
# Minimaler Import für alle weiteren Tasks:
from data.bev_dataloader import make_dataloaders_explicit

loaders = make_dataloaders_explicit(
    train_sources=[("C:/.../nuscenes_infos_train.pkl", "C:/.../Latents/Train")],
    val_sources  =[("C:/.../nuscenes_infos_val.pkl",   "C:/.../Latents/Val")],
    batch_size=16,        # Phase 1 (Frame-Level)
    # batch_size=4,       # Phase 2 (Cell-Level, VRAM-Limit)
    n_input_frames=3,
    num_workers=4,
    seed=42,
)

train_loader = loaders["train"]
val_loader   = loaders["val"]

# Ein Batch:
batch = next(iter(train_loader))
# batch["input"]  -> [16, 3, 256, 128, 128]
# batch["target"] -> [16, 256, 128, 128]
```

---

## Abhängigkeiten

```
numpy
torch
matplotlib
pickle (stdlib)
threading (stdlib)
h5py        # nur für build_hdf5_cache.py (optional)
```
