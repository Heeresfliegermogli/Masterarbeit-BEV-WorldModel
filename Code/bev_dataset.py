"""
bev_dataset.py -- BEV World Model: Dataset-Klasse
===================================================================

Laedt vorberechnete BEV-Latents und gibt Sequenzen der Laenge
n_input_frames+1 zurueck: n Input-Frames + 1 Target-Frame.

Zwei Lademodi (ueber BEVDatasetConfig.h5_path steuerbar):

  MODUS A -- .npy (Standard, kein h5_path):
    Jeder __getitem__-Aufruf oeffnet die .npy-Dateien direkt.
    Gut fuer: kleine Datensaetze, Entwicklung, wenn kein HDF5-Cache
    vorhanden ist.

  MODUS B -- HDF5 (h5_path angegeben):
    Latents werden aus einer einzigen HDF5-Datei gelesen die vorher
    mit build_hdf5_cache.py erstellt wurde. Pro Worker-Prozess wird
    eine eigene h5py.File-Instanz geoeffnet (thread-safe).
    Gut fuer: volles nuScenes, Training auf HDD/NFS, viele DataLoader-Worker.

Verwendung:
    # Modus A -- direkt aus .npy
    cfg = BEVDatasetConfig(
        sources=[("infos_val.pkl", "/Latents/Val")]
    )

    # Modus B -- aus HDF5-Cache
    cfg = BEVDatasetConfig(
        sources=[("infos_val.pkl", "/Latents/Val")],
        h5_path="/Cache/bev_latents_val.h5",
    )

    dataset = BEVLatentDataset(cfg)
    dataset = BEVLatentDataset(cfg, scene_indices=[0, 2, 5])
"""

import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Config-Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BEVDatasetConfig:
    """
    Konfiguration fuer BEVLatentDataset.

    sources:         Liste von (pkl_path, latent_dir) Paaren.
                     Jedes Paar beschreibt eine PKL und den zugehoerigen
                     Ordner mit .npy BEV-Latents. Mehrere Paare werden
                     zu einem einzigen Dataset gepoolt.

    n_input_frames:  Anzahl Input-Frames pro Sequenz (Standard: 3).
                     Target ist immer der unmittelbar folgende Frame.

    scene_gap_sec:   Zeitluecke in Sekunden ab der ein neuer Szenenstart
                     angenommen wird. Bei nuScenes 2Hz reicht 2.0s.

    h5_path:         Optionaler Pfad zu einer HDF5-Cache-Datei
                     (erstellt mit build_hdf5_cache.py).
                     None  -> Modus A: direkt aus .npy laden
                     Pfad  -> Modus B: aus HDF5-Cache laden

    packed_dir:      Optionaler Pfad zu einem Memmap-Pack-Verzeichnis
                     (erstellt mit pack_latents.py). Erwartet darin die
                     Datei 'packed.npy' -- ein einziges Memmap [N,C,H,W]
                     mit allen Frames dieses Splits in Szenen-Reihenfolge.
                     None  -> Modus A/B wie oben
                     Pfad  -> Modus C: ein Read pro Frame aus dem Memmap,
                              statt tausender Einzel-Opens.
                     Hat Vorrang vor h5_path, falls beides gesetzt.
    """
    sources:        List[Tuple[str, str]]   # [(pkl_path, latent_dir), ...]
    n_input_frames: int   = 3
    scene_gap_sec:  float = 2.0
    h5_path:        Optional[str] = None    # None = .npy Modus
    packed_dir:     Optional[str] = None    # None = kein Memmap-Pack (Modus C)
    latent_scale:   float = 1.0             # multiplikative Normalisierung
                                            # beim Laden (Det: 0.3641 -> Seg-Skala,
                                            # damit der SmoothL1-Arbeitspunkt beta/sigma
                                            # identisch bleibt). 1.0 = bit-identisch.
    # --- raeumlich gewichtete Latent-Loss (Default AUS) ----------
    gt_masks_path:  Optional[str] = None    # npz aus mk_gt_masks_seg.py
                                            # (Keys=Token, packbits 6x128x128).
                                            # None = kein cell_weight im Sample
                                            # -> bit-identisch zu vorher.
    rare_classes:   Tuple[str, ...] = ("stop_line", "ped_crossing", "divider")
    rare_weight:    float = 1.0             # Zellgewicht 1+(w-1)*rare_mask;
                                            # 1.0 = Gewichtskarte konstant 1.


# ---------------------------------------------------------------------------
# Hilfsfunktion: eine PKL einlesen und in Szenen teilen
# ---------------------------------------------------------------------------

def _load_scenes_from_pkl(
    pkl_path: str,
    latent_dir: str,
    scene_gap_threshold_sec: float = 2.0,
) -> List[List[Dict]]:
    """
    Liest eine BEVFusion-Info-PKL, sortiert Frames nach Timestamp und
    trennt sie anhand grosser Zeitluecken in einzelne Szenen.

    Jedes Frame-Dict bekommt '_latent_dir' eingetragen fuer den .npy-Lookup
    in Modus A. In Modus B wird dieses Feld ignoriert -- der Lookup
    laeuft dann ueber den token-Key im HDF5.
    """
    pkl_path   = Path(pkl_path)
    latent_dir = Path(latent_dir)

    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL nicht gefunden: {pkl_path}")
    if not latent_dir.exists():
        raise FileNotFoundError(f"Latent-Verzeichnis nicht gefunden: {latent_dir}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    infos: List[Dict] = sorted(data["infos"], key=lambda x: x["timestamp"])

    # Fehlende .npy-Dateien herausfiltern -- PKL kann mehr Eintraege haben
    # als tatsaechlich vorhandene Latents (z.B. wenn BEVFusion-Extraktion
    # fuer einzelne Frames gefailed ist). Fehlende Frames werden uebersprungen.
    n_before = len(infos)
    infos = [i for i in infos
             if (latent_dir / ("bev_latent_" + i["token"] + ".npy")).exists()]
    n_missing = n_before - len(infos)
    if n_missing > 0:
        print(f"  [Info] {n_missing}/{n_before} Frames ohne .npy uebersprungen "
              f"({latent_dir.name})")

    for info in infos:
        info["_latent_dir"] = str(latent_dir)

    threshold_us = scene_gap_threshold_sec * 1e6
    scenes: List[List[Dict]] = []
    current: List[Dict] = [infos[0]]

    for prev, curr in zip(infos[:-1], infos[1:]):
        if curr["timestamp"] - prev["timestamp"] > threshold_us:
            scenes.append(current)
            current = [curr]
        else:
            current.append(curr)

    scenes.append(current)
    return scenes


# ---------------------------------------------------------------------------
# HDF5-Handle-Cache -- ein Handle pro Worker-Thread (thread-safe)
# ---------------------------------------------------------------------------

class _H5HandleCache:
    """
    Verwaltet je einen h5py.File-Handle pro Worker-Thread.

    Warum noetig:
        h5py.File-Objekte sind NICHT thread-safe wenn mehrere Threads
        gleichzeitig lesen. PyTorch DataLoader startet pro Worker einen
        eigenen Prozess/Thread. Jeder Worker braucht deshalb seine eigene
        h5py.File-Instanz.

        Loesung: threading.local() -- jeder Thread bekommt seinen eigenen
        Slot. Beim ersten Zugriff eines Threads wird die Datei geoeffnet
        und im Thread-lokalen Slot gespeichert.
    """

    def __init__(self, h5_path: str):
        self._path  = h5_path
        self._local = threading.local()

    def get(self):
        """Gibt den h5py.File-Handle fuer den aktuellen Thread zurueck."""
        if not hasattr(self._local, "handle") or self._local.handle is None:
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "h5py nicht installiert. Bitte: pip install h5py\n"
                    "Oder verwende Modus A (kein h5_path in BEVDatasetConfig)."
                )
            # swmr=True: Single-Writer-Multiple-Reader -- erlaubt gleichzeitiges
            # Lesen aus mehreren Prozessen ohne Locks
            self._local.handle = h5py.File(self._path, "r", swmr=True)
        return self._local.handle


# ---------------------------------------------------------------------------
# Dataset-Klasse
# ---------------------------------------------------------------------------

class BEVLatentDataset(Dataset):
    """
    PyTorch Dataset fuer BEV World Model Training.

    Erstellt Sliding-Window-Sequenzen:
        Input:  [t-n, ..., t]  -> Tensor [n_input_frames, 256, 128, 128]
        Target: [t+1]          -> Tensor [256, 128, 128]

    Sequenzen werden NIE ueber Szenengrenzen oder PKL-Grenzen gebildet.

    Args:
        config:        BEVDatasetConfig mit Pfaden, Parametern und
                       optionalem h5_path fuer den HDF5-Modus.
        scene_indices: Optional -- nur Sequenzen dieser Szenen verwenden.
                       None = alle Szenen. Gesetzt von make_dataloaders_split()
                       fuer Train/Val/Test-Splits.
        transform:     Optionale Transformation auf den Input-Tensor.
    """

    def __init__(
        self,
        config: BEVDatasetConfig,
        scene_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.config    = config
        self.transform = transform

        # HDF5-Handle-Cache initialisieren falls h5_path angegeben
        self._h5: Optional[_H5HandleCache] = None
        if config.h5_path is not None:
            h5_path = Path(config.h5_path)
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5-Cache nicht gefunden: {h5_path}\n"
                    f"Bitte zuerst build_hdf5_cache.py ausfuehren."
                )
            self._h5 = _H5HandleCache(str(h5_path))

        # Memmap-Pack initialisieren falls packed_dir angegeben (Modus C).
        # Handle wird LAZY beim ersten _load_latent im jeweiligen Worker-
        # Prozess geoeffnet (siehe _load_latent) -- unter fork bekommt so
        # jeder Worker sein eigenes Mapping. Hier nur Pfad festhalten +
        # Existenz pruefen (fail-fast), noch NICHT oeffnen.
        self._latent_scale: float = float(getattr(config, "latent_scale", 1.0))
        self._packed_dir: Optional[str] = config.packed_dir
        self._packed_path: Optional[Path] = None
        self._packed_array = None
        self._packed_is_f32 = True     # wird beim Lazy-Open korrekt gesetzt
        if config.packed_dir is not None:
            self._packed_path = Path(config.packed_dir) / "packed.npy"
            if not self._packed_path.exists():
                raise FileNotFoundError(
                    f"Memmap-Pack nicht gefunden: {self._packed_path}\n"
                    f"Bitte zuerst pack_latents.py fuer diesen Split ausfuehren."
                )

        window_size = config.n_input_frames + 1

        # Alle PKL-Quellen laden
        self._all_scenes: List[List[Dict]] = []
        self._source_info: List[Dict] = []

        for pkl_path, latent_dir in config.sources:
            scenes = _load_scenes_from_pkl(
                pkl_path, latent_dir, config.scene_gap_sec
            )
            self._all_scenes.extend(scenes)
            self._source_info.append({
                "pkl":           pkl_path,
                "latent_dir":    latent_dir,
                "n_scenes":      len(scenes),
                "scene_lengths": [len(s) for s in scenes],
            })

        # Globale Pack-Zeilennummer je Frame zuweisen: laufender Zaehler ueber
        # ALLE Szenen in Konstruktions-Reihenfolge -- IDENTISCH zur Schreib-
        # Reihenfolge von pack_latents.py (das dieselbe Dataset-Instanz nutzt).
        # Bewusst VOR der scene_indices-Filterung: die Zeile eines Frames ist
        # damit global stabil, unabhaengig davon welcher Split gerade aktiv ist
        # (alle Splits lesen aus demselben packed.npy). Kein separates
        # Index-File -- die Zuordnung ist deterministisch aus der Config.
        # Wird immer gesetzt (auch ohne packed_dir), damit pack_latents.py die
        # Reihenfolge im .npy-Modus auslesen kann; als reiner int-Zaehler
        # ansonsten harmlos.
        row = 0
        for scene in self._all_scenes:
            for info in scene:
                info["_packed_row"] = row
                row += 1
        self._n_packed_frames = row

        # Aktive Szenen bestimmen
        if scene_indices is None:
            active_scenes = self._all_scenes
        else:
            if scene_indices and max(scene_indices) >= len(self._all_scenes):
                raise IndexError(
                    f"scene_indices enthaelt Index {max(scene_indices)}, "
                    f"aber es gibt nur {len(self._all_scenes)} Szenen."
                )
            active_scenes = [self._all_scenes[i] for i in scene_indices]

        # Sliding-Window-Sequenzen erzeugen
        self._windows: List[List[Dict]] = []

        for scene in active_scenes:
            n_frames  = len(scene)
            n_windows = n_frames - window_size + 1

            if n_windows <= 0:
                print(
                    f"[BEVLatentDataset] Warnung: Szene mit {n_frames} Frames "
                    f"zu kurz fuer Fenstergroesse {window_size}. Uebersprungen."
                )
                continue

            for i in range(n_windows):
                self._windows.append(scene[i : i + window_size])

        if config.packed_dir:
            mode = f"packed-memmap ({self._packed_path})"
        elif config.h5_path:
            mode = f"HDF5 ({config.h5_path})"
        else:
            mode = ".npy"
        self._print_summary(scene_indices, mode)

    # ------------------------------------------------------------------
    # Latent laden -- Modus A (.npy) oder Modus B (HDF5)
    # ------------------------------------------------------------------

    def _load_latent(self, frame: Dict) -> np.ndarray:
        """Laedt einen einzelnen Latent-Frame als float32 numpy Array [256,128,128].

        BEVFusion speichert Latents mit Batch-Dimension: (1, 256, 128, 128).
        squeeze(axis=0) entfernt diese Dimension deterministisch.

        Vorrang: packed_dir (Modus C) > h5_path (Modus B) > .npy (Modus A).
        """
        if self._packed_dir is not None:
            # Modus C: ein Read aus dem Memmap ueber die globale Zeilennummer.
            # Lazy Open PRO WORKER-PROZESS: self._packed_array startet als None
            # (auch nach fork), der erste Zugriff im Worker mappt die Datei.
            if self._packed_array is None:
                self._packed_array = np.load(self._packed_path, mmap_mode="r")
                # Guard: grober Mismatch (falsche Frame-Anzahl) -> harter Abbruch,
                # statt still falsche Zeilen zu lesen. Faengt NICHT eine
                # Umsortierung bei gleicher Anzahl -- dagegen schuetzt, dass
                # pack_latents.py dieselbe Dataset-Instanz zur Reihenfolge nutzt.
                if self._packed_array.shape[0] != self._n_packed_frames:
                    raise RuntimeError(
                        f"Pack-Mismatch: {self._packed_path} hat "
                        f"{self._packed_array.shape[0]} Zeilen, Dataset erwartet "
                        f"{self._n_packed_frames}. Config/Pack laufen auseinander "
                        f"-- packed.npy mit aktueller Config neu erzeugen."
                    )
                # Storage-dtype einmalig merken (Perf: nicht pro Frame pruefen).
                self._packed_is_f32 = (self._packed_array.dtype == np.float32)
            # Wir brauchen IMMER eine In-RAM-Kopie (kein Memmap-View ans Modell,
            # sonst haelt jeder Batch die ganze Datei gemappt).
            #  - der Latent wird im STORAGE-dtype zurueckgegeben
            #    (fp16 bleibt fp16). Der Upcast fp16->fp32 wird bewusst NICHT mehr
            #    hier gemacht, sondern erst GPU-seitig am jeweiligen h2d-Punkt
            #    (train_linux.py / eval_full_val.py) -- so gehen nur halb so viele
            #    Bytes ueber den PCIe-Bus (h2d war 24% der Step-Zeit, Stufe-0-Profil).
            #    Numerisch neutral: fp16->fp32 ist ein exaktes Widening, nur verlagert.
            #  - np.array() liefert IMMER eine In-RAM-Kopie (kein Memmap-View ans
            #    Modell), unabhaengig vom dtype.
            row = self._packed_array[frame["_packed_row"]]
            arr = np.array(row)                      # native dtype (fp16 ODER fp32), reine Kopie
            if self._latent_scale != 1.0:            # Normalisierung (dtype-erhaltend)
                arr = arr * arr.dtype.type(self._latent_scale)
            return arr

        if self._h5 is not None:
            # Modus B: HDF5-Lookup ueber Token-String
            handle = self._h5.get()
            arr = handle["latents"][frame["token"]][:]
        else:
            # Modus A: direkt aus .npy
            path = Path(frame["_latent_dir"]) / f"bev_latent_{frame['token']}.npy"
            arr = np.load(path).astype("float32")

        # BEVFusion speichert (1, 256, 128, 128) -- Batch-Dim entfernen
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)   # -> [256, 128, 128]

        if self._latent_scale != 1.0:   # Normalisierung (Modi A/B)
            arr = arr * arr.dtype.type(self._latent_scale)
        return arr

    # ------------------------------------------------------------------
    # Pack-Reihenfolge (Single Source of Truth fuer pack_latents.py)
    # ------------------------------------------------------------------

    def iter_pack_order(self):
        """Liefert (row, frame) fuer ALLE Frames in globaler Pack-Reihenfolge.

        Exakt die Menge und Reihenfolge, die _packed_row definiert. pack_latents.py
        nutzt diesen Iterator zum Schreiben des Memmaps -- dadurch gibt es nur EINE
        Implementierung der Frame-Reihenfolge (die des Datasets), Pack und Laufzeit
        koennen nicht auseinanderlaufen. Unabhaengig von scene_indices (laeuft ueber
        _all_scenes, nicht ueber die aktiven Fenster).
        """
        for scene in self._all_scenes:
            for frame in scene:
                yield frame["_packed_row"], frame

    # ------------------------------------------------------------------
    # Dataset-Interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _rel_ego_delta(fi: Dict, fj: Dict, scale: float = 10.0):
        """Relative Ego-Bewegung fi->fj in fi's Ego-Koordinaten:
        [dx/scale, dy/scale, cos(dyaw), sin(dyaw)]. Ego-Conditioning.
        Fehlt eine Pose -> Nullvektor (rueckwaerts-kompatibel)."""
        ti = fi.get("ego2global_translation"); ri = fi.get("ego2global_rotation")
        tj = fj.get("ego2global_translation"); rj = fj.get("ego2global_rotation")
        if ti is None or tj is None or ri is None or rj is None:
            return [0.0, 0.0, 1.0, 0.0]
        def yaw(q):  # q=[w,x,y,z]
            w, x, y, z = q
            return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yi = float(yaw(ri)); dyaw = float(yaw(rj)) - yi
        dxg, dyg = tj[0] - ti[0], tj[1] - ti[1]              # global
        c, s = np.cos(-yi), np.sin(-yi)                      # in fi-Ego-Frame drehen
        px = c * dxg - s * dyg
        py = s * dxg + c * dyg
        return [px / scale, py / scale, float(np.cos(dyaw)), float(np.sin(dyaw))]

    def __getitem__(self, idx: int) -> Dict:
        window       = self._windows[idx]
        input_frames = window[:-1]
        target_frame = window[-1]

        input_tensor = torch.from_numpy(
            np.stack([self._load_latent(f) for f in input_frames], axis=0)
        )
        target_tensor = torch.from_numpy(self._load_latent(target_frame))

        if self.transform is not None:
            input_tensor = self.transform(input_tensor)

        # Ego-Deltas entlang der Sequenz: [d12, d23, ..., d(n-1)n, d(n->target)]
        # -> flach [ (n_input) * 4 ]. Modell schneidet je nach ego_cond_mode:
        #    action = letzte 4 (F_last->target), state = erste (n_input-1)*4.
        chain = list(input_frames) + [target_frame]
        ego = []
        for a, b in zip(chain[:-1], chain[1:]):
            ego.extend(self._rel_ego_delta(a, b))
        ego_delta = torch.tensor(ego, dtype=torch.float32)   # [n_input * 4]

        sample = {
            "input":        input_tensor,   # [n_input_frames, 256, 128, 128]
            "target":       target_tensor,  # [256, 128, 128]
            "ego_delta":    ego_delta,      # [n_input*4] 
            "input_tokens": [f["token"] for f in input_frames],
            "target_token": target_frame["token"],
        }
        # Zellgewichte aus der Map-GT des TARGET-Frames.
        if self.config.gt_masks_path is not None:
            sample["cell_weight"] = torch.from_numpy(
                self._cell_weight(target_frame["token"]))   # [128,128] float32
        return sample

    # Klassen-Reihenfolge von mk_gt_masks_seg.py (= Seg-Config)
    _GT_CLASSES = ("drivable_area", "ped_crossing", "walkway", "stop_line",
                   "carpark_area", "divider")

    def _cell_weight(self, token: str) -> np.ndarray:
        """1+(w-1)*rare_mask aus der gepackten 6-Klassen-GT-Maske.
        npz wird pro Worker lazy geoeffnet (fork-sicher)."""
        if not hasattr(self, "_gt_masks_npz"):
            self._gt_masks_npz = np.load(self.config.gt_masks_path)
        packed = self._gt_masks_npz[token]
        masks = np.unpackbits(packed).reshape(6, 128, 128)
        idx = [self._GT_CLASSES.index(c) for c in self.config.rare_classes]
        rare = masks[idx].any(axis=0).astype(np.float32)
        w = float(self.config.rare_weight)
        return (1.0 + (w - 1.0) * rare).astype(np.float32)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def _print_summary(self, scene_indices, mode: str) -> None:
        total  = len(self._all_scenes)
        active = len(scene_indices) if scene_indices is not None else total
        print(
            f"[BEVLatentDataset] {len(self._source_info)} Quelle(n) | "
            f"{active}/{total} Szene(n) | "
            f"{len(self)} Sequenzen | "
            f"Modus: {mode}"
        )
