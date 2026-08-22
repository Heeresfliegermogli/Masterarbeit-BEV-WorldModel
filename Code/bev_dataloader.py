"""
bev_dataloader.py -- BEV World Model: DataLoader-Factory
====================================================================

Erstellt Train/Val/Test DataLoader mit zwei Modi:

  MODUS 1 -- split (Mini / Entwicklung):
    Eine oder mehrere PKLs werden auf Szenen-Ebene deterministisch
    in Train/Val/Test aufgeteilt. BEVLatentDataset wird mit den
    jeweiligen scene_indices instanziiert -- kein doppelter Code.

    Verwendung:
        loaders = make_dataloaders_split(
            sources=[("infos_val.pkl", "/Latents/Val")],
            train_ratio=0.70, val_ratio=0.15,
            batch_size=16, seed=42,
        )

  MODUS 2 -- explicit (Full nuScenes / Produktion):
    Separate PKLs pro Split. DataLoader werden direkt aus den
    jeweiligen BEVDatasetConfig-Objekten gebaut -- kein Split noetig.
    Optional: je Split ein eigener HDF5-Cache (train_h5_path/val_h5_path/
    test_h5_path) -- empfohlen fuer volle nuScenes-Datenmengen, bei denen
    .npy-Direktzugriff auf zehntausende Einzeldateien zu langsam ist.

    Verwendung (.npy direkt):
        loaders = make_dataloaders_explicit(
            train_sources=[("infos_train.pkl", "/Latents/Train")],
            val_sources  =[("infos_val.pkl",   "/Latents/Val")],
            test_sources =[("infos_test.pkl",  "/Latents/Test")],
            batch_size=16,
        )

    Verwendung (HDF5-Cache, empfohlen fuer vollen Datensatz):
        loaders = make_dataloaders_explicit(
            train_sources=[("infos_train.pkl", "/Latents/Train")],
            val_sources  =[("infos_val.pkl",   "/Latents/Val")],
            batch_size=16,
            train_h5_path="/Cache/bev_latents_train.h5",
            val_h5_path  ="/Cache/bev_latents_val.h5",
        )

Beide Funktionen geben zurueck:
    {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from bev_dataset import BEVDatasetConfig, BEVLatentDataset


# ---------------------------------------------------------------------------
# Sampling-Gewichte fuer den Train-Loader
# ---------------------------------------------------------------------------

def _build_train_sampler(ds, mode, alpha, gt_masks_path, dynamics_path, seed):
    """WeightedRandomSampler ueber die Train-Fenster.
    mode=rare:     w ~ (Selten-Klassen-Pixelanteil des Targets + 1e-3)^alpha
    mode=dynamics: w ~ (Dynamik-Score des Target-Paars / Median)^alpha
    replacement=True, num_samples=len(ds) -> Epochenlaenge unveraendert."""
    import json

    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    tokens = [w[-1]["token"] for w in ds._windows]
    if mode == "rare":
        assert gt_masks_path, "sampler_mode=rare braucht gt_masks_train_path"
        npz = np.load(gt_masks_path)
        idx = [BEVLatentDataset._GT_CLASSES.index(c) for c in ds.config.rare_classes]
        shares = np.array([
            np.unpackbits(npz[t]).reshape(6, 128, 128)[idx].any(axis=0).mean()
            for t in tokens], dtype=np.float64)
        weights = (shares + 1e-3) ** alpha
    elif mode == "dynamics":
        assert dynamics_path, "sampler_mode=dynamics braucht dynamics_scores_path"
        with open(dynamics_path) as f:
            scores = json.load(f)
        s = np.array([scores[t] for t in tokens], dtype=np.float64)
        weights = (s / np.median(s)) ** alpha
    else:
        raise ValueError(f"sampler_mode muss off|rare|dynamics sein, war '{mode}'")

    weights = weights / weights.mean()
    print(f"[sampler:{mode}] alpha={alpha}, n={len(weights)}, "
          f"w-min/median/max = {weights.min():.3f}/{np.median(weights):.3f}/"
          f"{weights.max():.3f}")
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True,
                                 generator=g)


# ---------------------------------------------------------------------------
# Hilfsfunktion: Szenen deterministisch shufflen und splitten
# ---------------------------------------------------------------------------

def _split_scene_indices(
    n_scenes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Teilt n_scenes Szenen-Indizes deterministisch in Train/Val/Test auf.

    Split passiert auf Szenen-Ebene: alle Sequenzen einer Szene landen
    immer im selben Split -- kein Leakage durch ueberlappende Fenster.

    Nutzt random.Random(seed) -- isolierter RNG, kein globaler State.
    """
    assert train_ratio + val_ratio <= 1.0

    indices = list(range(n_scenes))
    random.Random(seed).shuffle(indices)

    n_train = max(1, int(n_scenes * train_ratio))
    n_val   = max(1, int(n_scenes * val_ratio))
    n_test  = max(0, n_scenes - n_train - n_val)

    # Rundungsfehler abfangen
    if n_train + n_val + n_test != n_scenes:
        n_train = n_scenes - n_val - n_test

    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# Interner DataLoader-Builder
# ---------------------------------------------------------------------------

def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    prefetch_factor: int = 4,
    pin_memory: bool = True,   # Loader-Diagnose: der EINE pin-Thread ist Rest-Engpass-
                               # Kandidat (data% ~58 nach 1d); false nur fuer
                               # Cluster-Diagnose-Configs, Default unveraendert.
    sampler=None,              # WeightedRandomSampler (train)
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    # prefetch_factor darf NUR gesetzt werden, wenn num_workers > 0,
    # sonst wirft PyTorch einen ValueError. Bei num_workers=0 (Windows /
    # .npy-Debug) faellt der DataLoader auf synchrones Laden zurueck.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# MODUS 1 -- split
# ---------------------------------------------------------------------------

def make_dataloaders_split(
    sources: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 16,
    n_input_frames: int = 3,
    scene_gap_sec: float = 2.0,
    num_workers: int = 4,
    seed: int = 42,
    transform=None,
    h5_path: str = None,
    packed_dir: str = None,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    latent_scale: float = 1.0,   # multiplikative Normalisierung (Det->Seg-Skala)
) -> Dict[str, Optional[DataLoader]]:
    """
    Laedt alle Szenen aus 'sources', shuffled deterministisch und teilt
    auf Szenen-Ebene in Train/Val/Test. BEVLatentDataset wird pro Split
    mit den passenden scene_indices instanziiert.

    Args:
        sources:        [(pkl_path, latent_dir), ...]
        train_ratio:    Anteil Trainingsszenen (Standard: 0.70)
        val_ratio:      Anteil Validierungsszenen (Standard: 0.15)
        batch_size:     Batch-Groesse fuer alle DataLoader
        n_input_frames: Anzahl Input-Frames pro Sequenz
        scene_gap_sec:  Schwellwert fuer Szenentrennung in Sekunden
        num_workers:    DataLoader Worker-Prozesse
        seed:           RNG-Seed fuer Split und DataLoader
        transform:      Optionale Transformation fuer Input-Tensoren

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
    """
    # Config einmalig erstellen -- wird fuer alle drei Splits geteilt.
    # BEVLatentDataset laedt intern alle Szenen aus den sources und
    # filtert dann per scene_indices auf den gewuenschten Split.
    cfg = BEVDatasetConfig(
        sources=sources,
        n_input_frames=n_input_frames,
        scene_gap_sec=scene_gap_sec,
        h5_path=h5_path,
        packed_dir=packed_dir,
        latent_scale=latent_scale,
    )

    # Anzahl Szenen bestimmen ohne Dataset zu instanziieren:
    # Dazu kurz eine Dummy-Instanz mit scene_indices=[] -- laedt nur Metadaten.
    # Alternativ: einmal instanziieren und n_scenes aus _all_scenes lesen.
    _probe = BEVLatentDataset(cfg, scene_indices=[])
    n_scenes = len(_probe._all_scenes)

    train_idx, val_idx, test_idx = _split_scene_indices(
        n_scenes, train_ratio, val_ratio, seed
    )

    print(f"[make_dataloaders_split] {n_scenes} Szenen, seed={seed}:")
    print(f"  Train: {len(train_idx)} Szene(n), Indizes {sorted(train_idx)}")
    print(f"  Val:   {len(val_idx)} Szene(n), Indizes {sorted(val_idx)}")
    print(f"  Test:  {len(test_idx)} Szene(n), Indizes {sorted(test_idx)}")

    train_ds = BEVLatentDataset(cfg, scene_indices=train_idx, transform=transform)
    val_ds   = BEVLatentDataset(cfg, scene_indices=val_idx,   transform=transform)

    loaders: Dict[str, Optional[DataLoader]] = {
        "train": _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory),
        "val":   _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory),
        "test":  None,
    }

    if test_idx:
        test_ds = BEVLatentDataset(cfg, scene_indices=test_idx, transform=transform)
        loaders["test"] = _make_loader(test_ds, batch_size, shuffle=False, num_workers=num_workers, seed=seed, prefetch_factor=prefetch_factor, pin_memory=pin_memory)

    return loaders


# ---------------------------------------------------------------------------
# MODUS 2 -- explicit
# ---------------------------------------------------------------------------

def make_dataloaders_explicit(
    train_sources: List[Tuple[str, str]],
    val_sources:   List[Tuple[str, str]],
    test_sources:  Optional[List[Tuple[str, str]]] = None,
    batch_size: int = 16,
    n_input_frames: int = 3,
    scene_gap_sec: float = 2.0,
    num_workers: int = 4,
    seed: int = 42,
    transform=None,
    train_h5_path: str = None,
    val_h5_path: str = None,
    test_h5_path: str = None,
    train_packed_dir: str = None,
    val_packed_dir: str = None,
    test_packed_dir: str = None,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    latent_scale: float = 1.0,   # Normalisierung beim Laden (1.0 = aus)
    # --- Sampler-/Gewichtungs-Optionen (alle Default AUS = bit-identisch) -------------------------
    gt_masks_train_path: str = None,   # npz -> train-Samples mit cell_weight
    rare_classes: tuple = ("stop_line", "ped_crossing", "divider"),
    rare_weight: float = 1.0,
    sampler_mode: str = "off",         # off | rare | dynamics
    sampler_alpha: float = 0.5,        # Exponent der Sampling-Gewichte
    dynamics_scores_path: str = None,  # JSON token->score (train)
) -> Dict[str, Optional[DataLoader]]:
    """
    Baut DataLoader aus explizit getrennten PKL-Dateien.
    Kein weiterer Split noetig -- nuScenes liefert bereits getrennte
    train/val/test Info-Dateien.

    Geeignet fuer: Full nuScenes (700 Train + 150 Val + 150 Test Szenen).

    Args:
        train_sources:  [(pkl_path, latent_dir), ...] fuer Training
        val_sources:    [(pkl_path, latent_dir), ...] fuer Validierung
        test_sources:   [(pkl_path, latent_dir), ...] fuer Test (optional)
        batch_size:     Batch-Groesse
        n_input_frames: Anzahl Input-Frames pro Sequenz
        scene_gap_sec:  Schwellwert fuer Szenentrennung in Sekunden
        num_workers:    DataLoader Worker-Prozesse
        seed:           RNG-Seed fuer DataLoader-Generator
        transform:      Optionale Transformation fuer Input-Tensoren
        train_h5_path:  Optionaler Pfad zum HDF5-Cache fuer Train
                        (erstellt mit build_hdf5_cache.py).
                        None -> Modus A: direkt aus .npy laden (train_sources)
        val_h5_path:    Optionaler Pfad zum HDF5-Cache fuer Val.
        test_h5_path:   Optionaler Pfad zum HDF5-Cache fuer Test.

        HINWEIS: Anders als bei make_dataloaders_split() (EIN gemeinsamer
        h5_path fuer alle Splits, da dort eine einzige Quelle intern
        gesplittet wird) gibt es hier JE SPLIT einen eigenen h5_path --
        Train und Val sind bereits getrennte PKLs/Latent-Ordner und damit
        typischerweise auch getrennte HDF5-Dateien (z.B.
        bev_latents_train.h5 / bev_latents_val.h5).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader | None}
    """
    def _build(sources, shuffle, h5_path, packed_dir, is_train=False):
        cfg = BEVDatasetConfig(
            sources=sources,
            n_input_frames=n_input_frames,
            scene_gap_sec=scene_gap_sec,
            h5_path=h5_path,
            packed_dir=packed_dir,
            latent_scale=latent_scale,
            # Gewichtskarten NUR im Train-Split (Val-Loss bleibt
            # ungewichtet -> Early-Stopping/Selektion sweep-vergleichbar).
            gt_masks_path=(gt_masks_train_path if is_train else None),
            rare_classes=tuple(rare_classes),
            rare_weight=rare_weight,
        )
        ds = BEVLatentDataset(cfg, transform=transform)
        sampler = None
        if is_train and sampler_mode != "off":
            sampler = _build_train_sampler(ds, sampler_mode, sampler_alpha,
                                           gt_masks_train_path,
                                           dynamics_scores_path, seed)
            shuffle = False   # Sampler und shuffle schliessen sich aus
        return _make_loader(ds, batch_size, shuffle=shuffle, num_workers=num_workers,
                            seed=seed, prefetch_factor=prefetch_factor,
                            pin_memory=pin_memory, sampler=sampler)

    loaders: Dict[str, Optional[DataLoader]] = {
        "train": _build(train_sources, shuffle=True,  h5_path=train_h5_path, packed_dir=train_packed_dir, is_train=True),
        "val":   _build(val_sources,   shuffle=False, h5_path=val_h5_path,   packed_dir=val_packed_dir),
        "test":  _build(test_sources,  shuffle=False, h5_path=test_h5_path,  packed_dir=test_packed_dir) if test_sources else None,
    }

    print(f"[make_dataloaders_explicit] Train={len(loaders['train'])} Batches, "
          f"Val={len(loaders['val'])} Batches"
          + (f", Test={len(loaders['test'])} Batches" if loaders["test"] else ""))

    return loaders
