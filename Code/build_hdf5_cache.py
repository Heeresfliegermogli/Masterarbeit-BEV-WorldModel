"""
build_hdf5_cache.py -- BEV World Model: HDF5-Cache Builder
======================================================================

Konvertiert vorberechnete BEV-Latents (.npy Dateien) in eine einzige
HDF5-Datei. Das HDF5-Format erlaubt schnellen wahlfreien Zugriff auf
einzelne Frames ohne alle Latents im RAM halten zu muessen.

HDF5-Struktur:
    latents/<token>   float32 [256, 128, 128]   -- ein Dataset pro Frame
    meta/tokens       bytes-Array aller Tokens   -- fuer Vollstaendigkeitscheck

Aufruf (Val-Set):
    python data/build_hdf5_cache.py \\
        --pkl   nuscenes_infos_val.pkl \\
        --npy   /data/Latents/Val \\
        --out   /data/Cache/bev_latents_val.h5

Aufruf (mehrere Quellen in einem Schritt):
    python data/build_hdf5_cache.py \\
        --pkl   infos_train.pkl  infos_val.pkl \\
        --npy   /Latents/Train   /Latents/Val \\
        --out   /Cache/bev_latents_all.h5

Warum HDF5 statt einzelner .npy Dateien?
    - Ein einziger open()-Aufruf statt tausender np.load()-Aufrufe
    - Betriebssystem-Cache greift besser bei sequentiellem Zugriff
    - Kompression reduziert Speicherbedarf (gzip ~30-50% bei BEV-Latents)
    - Portabel: eine Datei weitergeben statt tausende Einzelfiles
    - h5py-Zugriff ist thread-safe mit swmr=True (wichtig fuer DataLoader
      mit num_workers > 0)
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# h5py wird lazy importiert in build_cache() und verify_cache()


# ---------------------------------------------------------------------------
# Hilfsfunktion: alle Frame-Tokens aus PKL(s) sammeln
# ---------------------------------------------------------------------------

def _collect_tokens(
    pkl_paths: List[str],
    npy_dirs: List[str],
) -> List[Tuple[str, Path]]:
    """
    Liest alle PKLs und gibt eine Liste von (token, npy_path) Paaren zurueck.
    Tokens werden dedupliziert -- falls dieselbe PKL versehentlich zweimal
    angegeben wird, taucht jeder Token nur einmal auf.

    Args:
        pkl_paths: Liste von Pfaden zu nuscenes_infos_*.pkl Dateien
        npy_dirs:  Liste von Ordnern mit .npy Latents, parallel zu pkl_paths

    Returns:
        Liste von (token, npy_path) Paaren, sortiert nach token
    """
    seen   = set()
    result = []

    for pkl_path, npy_dir in zip(pkl_paths, npy_dirs):
        npy_dir = Path(npy_dir)

        with open(pkl_path, "rb") as f:
            infos = pickle.load(f)["infos"]

        for info in infos:
            token = info["token"]
            if token in seen:
                continue
            seen.add(token)

            npy_path = npy_dir / f"bev_latent_{token}.npy"
            if not npy_path.exists():
                print(f"  [Warnung] .npy nicht gefunden: {npy_path} -- uebersprungen")
                continue

            result.append((token, npy_path))

    return sorted(result, key=lambda x: x[0])  # deterministisch sortiert


# ---------------------------------------------------------------------------
# Hauptfunktion: HDF5-Cache bauen
# ---------------------------------------------------------------------------

def build_cache(
    pkl_paths:    List[str],
    npy_dirs:     List[str],
    out_path:     str,
    compression:  str = "gzip",
    comp_level:   int = 4,
    chunk_shape:  Tuple[int, int, int] = (256, 128, 128),
    overwrite:    bool = False,
) -> None:
    """
    Baut eine HDF5-Cache-Datei aus .npy BEV-Latents.

    Args:
        pkl_paths:   Liste der PKL-Dateien (definiert welche Tokens erwartet werden)
        npy_dirs:    Liste der .npy-Ordner, parallel zu pkl_paths
        out_path:    Zielpfad fuer die .h5 Datei
        compression: HDF5-Kompressionsalgorithmus ('gzip', 'lzf', None)
                     gzip: beste Kompression, etwas langsamer
                     lzf:  schneller, etwas schlechtere Kompression
                     None: kein Overhead, maximale I/O-Geschwindigkeit
        comp_level:  Kompressionsstufe fuer gzip (1=schnell, 9=klein)
        chunk_shape: HDF5-Chunk-Groesse. Ein Chunk = ein Frame [256,128,128].
                     Wichtig: Chunk-Groesse bestimmt minimale Lesemenge.
                     Bei Frame-weisem Zugriff optimal = ein Frame pro Chunk.
        overwrite:   Bestehende Datei ueberschreiben?
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if overwrite:
            print(f"[build_cache] Ueberschreibe: {out_path}")
            out_path.unlink()
        else:
            raise FileExistsError(
                f"{out_path} existiert bereits. --overwrite um zu ueberschreiben."
            )

    # Alle Tokens sammeln
    print("[build_cache] Sammle Tokens aus PKL(s)...")
    token_paths = _collect_tokens(pkl_paths, npy_dirs)
    n = len(token_paths)
    print(f"[build_cache] {n} Frames gefunden. Starte Konvertierung...")
    print(f"  Kompression: {compression or 'keine'} (level {comp_level})")
    print(f"  Chunk-Shape: {chunk_shape}")
    print(f"  Ziel:        {out_path}")
    print()

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py nicht installiert. Bitte: pip install h5py")

    t_start = time.time()
    compress_kwargs = {}
    if compression:
        compress_kwargs = {"compression": compression, "compression_opts": comp_level}

    with h5py.File(out_path, "w") as f:

        # Gruppe fuer Latents
        grp = f.create_group("latents")

        # Gruppe fuer Metadaten
        meta = f.create_group("meta")
        meta.attrs["n_frames"]    = n
        meta.attrs["latent_shape"] = list(chunk_shape)
        meta.attrs["dtype"]       = "float32"
        meta.attrs["source_pkls"] = str(pkl_paths)

        # Alle Tokens als Byte-Array speichern (Vollstaendigkeitscheck)
        token_bytes = [t.encode("utf-8") for t, _ in token_paths]
        meta.create_dataset(
            "tokens",
            data=np.array(token_bytes, dtype=h5py.special_dtype(vlen=bytes)),
        )

        # Frames schreiben
        for i, (token, npy_path) in enumerate(token_paths):
            arr = np.load(npy_path).astype("float32")

            # BEVFusion speichert (1, 256, 128, 128) -- Batch-Dim entfernen
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(axis=0)   # -> [256, 128, 128]

            if arr.shape != chunk_shape:
                raise ValueError(
                    f"Unerwartete Shape {arr.shape} fuer Token {token}. "
                    f"Erwartet: {chunk_shape}"
                )

            grp.create_dataset(
                token,
                data=arr,
                chunks=chunk_shape,
                **compress_kwargs,
            )

            # Fortschritt
            if (i + 1) % 10 == 0 or i == n - 1:
                elapsed = time.time() - t_start
                fps     = (i + 1) / elapsed
                eta     = (n - i - 1) / fps if fps > 0 else 0
                mb_done = (i + 1) * 16.0   # ~16 MB pro Frame (unkomprimiert)
                print(
                    f"  [{i+1:4d}/{n}]  "
                    f"{elapsed:5.1f}s  "
                    f"{fps:.1f} frames/s  "
                    f"ETA {eta:.0f}s  "
                    f"~{mb_done/1024:.1f} GB verarbeitet"
                )

    # Abschlussbericht
    elapsed   = time.time() - t_start
    h5_size   = out_path.stat().st_size / 1e9
    raw_size  = n * 16.0 / 1024   # GB unkomprimiert
    ratio     = raw_size / h5_size if h5_size > 0 else 1.0

    print()
    print(f"[build_cache] Fertig in {elapsed:.1f}s")
    print(f"  Frames:          {n}")
    print(f"  Unkomprimiert:   {raw_size:.2f} GB")
    print(f"  HDF5-Groesse:    {h5_size:.2f} GB")
    print(f"  Kompressionsrate:{ratio:.1f}x")
    print(f"  Gespeichert:     {out_path}")


# ---------------------------------------------------------------------------
# Vollstaendigkeitscheck: HDF5 gegen PKL pruefen
# ---------------------------------------------------------------------------

def verify_cache(h5_path: str, pkl_paths: List[str], npy_dirs: List[str]) -> bool:
    """
    Prueft ob die HDF5-Datei alle erwarteten Tokens enthaelt.
    Liest dazu die Metadaten aus dem HDF5 und vergleicht mit den PKLs.

    Returns:
        True wenn vollstaendig, False wenn Tokens fehlen.
    """
    print(f"[verify_cache] Pruefe: {h5_path}")

    expected_tokens = set(t for t, _ in _collect_tokens(pkl_paths, npy_dirs))

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py nicht installiert. Bitte: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        cached_tokens = set(f["latents"].keys())
        n_frames      = f["meta"].attrs["n_frames"]
        latent_shape  = tuple(f["meta"].attrs["latent_shape"])

    print(f"  Erwartet:  {len(expected_tokens)} Tokens")
    print(f"  Im Cache:  {len(cached_tokens)} Tokens")
    print(f"  Latent-Shape: {latent_shape}")

    missing = expected_tokens - cached_tokens
    extra   = cached_tokens - expected_tokens

    if missing:
        print(f"  [FEHLER] {len(missing)} Tokens fehlen im Cache!")
        for t in sorted(missing)[:5]:
            print(f"    {t}")
        return False

    if extra:
        print(f"  [Warnung] {len(extra)} unerwartete Tokens im Cache (aus anderer PKL?)")

    print(f"  [OK] Cache vollstaendig.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Konvertiert BEV-Latent .npy Dateien in einen HDF5-Cache."
    )
    parser.add_argument(
        "--pkl", nargs="+", required=True,
        help="Pfad(e) zu nuscenes_infos_*.pkl (Leerzeichen-getrennt fuer mehrere)"
    )
    parser.add_argument(
        "--npy", nargs="+", required=True,
        help="Pfad(e) zu .npy Latent-Ordnern (parallel zu --pkl)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Zielpfad fuer die .h5 Datei (z.B. /data/Cache/bev_latents_val.h5)"
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf", "none"],
        help="Kompressionsalgorithmus (Standard: gzip)"
    )
    parser.add_argument(
        "--level", type=int, default=4,
        help="Kompressionsstufe fuer gzip 1-9 (Standard: 4)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Bestehende .h5 Datei ueberschreiben"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Nur Vollstaendigkeitscheck, kein Neuaufbau"
    )

    args = parser.parse_args()

    if len(args.pkl) != len(args.npy):
        parser.error("Anzahl --pkl und --npy muss gleich sein.")

    compression = None if args.compression == "none" else args.compression

    if args.verify_only:
        ok = verify_cache(args.out, args.pkl, args.npy)
        exit(0 if ok else 1)

    build_cache(
        pkl_paths=args.pkl,
        npy_dirs=args.npy,
        out_path=args.out,
        compression=compression,
        comp_level=args.level,
        overwrite=args.overwrite,
    )

    # Automatisch pruefen nach dem Bauen
    verify_cache(args.out, args.pkl, args.npy)
