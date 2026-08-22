"""
pack_latents.py -- BEV World Model: Memmap-Packing
=================================================================

Packt die vorberechneten .npy-BEV-Latents eines Splits in EIN
zusammenhaengendes Memmap-File ({packed_dir}/packed.npy). Zur Laufzeit
liest BEVLatentDataset dann Zeile fuer Zeile aus dieser einen Datei
statt tausender Einzel-Opens -- auf BeeGFS gemessen ~2.6x schneller
(siehe ZWISCHENBERICHT_TASK15_CLUSTER.md).

KERNPRINZIP (Single Source of Truth):
    Die Schreib-Reihenfolge kommt NICHT aus einer eigenen PKL-Logik hier,
    sondern aus einer echten BEVLatentDataset-Instanz (packed_dir=None,
    scene_indices=None). Genau dieselbe Instanz weist zur Laufzeit die
    Zeilennummern zu (_packed_row). Dadurch gibt es nur EINE Implementierung
    der Frame-Reihenfolge -> Pack und Laufzeit koennen nicht auseinanderlaufen.

MODI:
    explicit-Config:  --split {train,val,test}
        sources  <- data.<split>_sources
        out_dir  <- data.<split>_packed_dir
    split-Config:     --split all
        sources  <- data.sources
        out_dir  <- data.packed_dir   (EIN Pack fuer alle internen Splits)

VERWENDUNG:
    # float32 (Standard, verlustfrei):
    python -u pack_latents.py --config config_nuscenes_full.yaml --split val
    python -u pack_latents.py --config config_nuscenes_full.yaml --split train

    # float16:
    python -u pack_latents.py --config config_nuscenes_full.yaml --split val --dtype float16

HINWEIS:
    Zuerst an val (klein) laufen lassen, Laufzeit + Groesse messen, dann
    auf train hochrechnen -- nicht raten. Die Original-.npy bleiben
    unangetastet als Referenz/Fallback stehen.
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from bev_dataset import BEVDatasetConfig, BEVLatentDataset


DTYPES = {"float32": np.float32, "float16": np.float16}


# =============================================================================
# Manifest-Helper -- Audit-Artefakt, KEIN Laufzeit-Index.
# Zeilennummern bleiben weiterhin deterministisch zur Laufzeit berechnet
# (BEVLatentDataset._packed_row) -- das Manifest dient nur der Pruefung
# gegen stille Config-/Reihenfolge-Mismatches und der Reproduzierbarkeits-
# Doku fuer die Thesis.
# =============================================================================

def _config_hash(cfg: dict) -> str:
    """Hash ueber den INHALT des geladenen cfg-Dicts (nicht ueber die Datei --
    erkennt inhaltliche Aenderungen der wirksamen Config, ignoriert reine
    Formatierung/Kommentare in der YAML). sort_keys sorgt fuer Determinismus
    unabhaengig von der Key-Reihenfolge im Dict.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_commit() -> Optional[str]:
    """Best-effort Git-Commit-Hash. Weder Server noch Cluster nutzen derzeit
    Git-Versionierung fuer diesen Code -- kein Repo gefunden ist der
    ERWARTETE Normalfall hier, kein Fehler. Gibt None zurueck statt zu
    werfen; der Manifest-Wert wird dann explizit 'null' (JSON), nicht
    stillschweigend weggelassen.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _scene_order_hash(ds) -> str:
    """Hash ueber die Token-Sequenz in Pack-Reihenfolge (iter_pack_order).
    Zwei Packs mit identischem Hash haben garantiert dieselbe Frame-
    Reihenfolge -- Aenderungen an Sortierung/Filterung/Szenen-Trennung
    schlagen sofort sichtbar durch.
    """
    hasher = hashlib.sha256()
    for _row, frame in ds.iter_pack_order():
        hasher.update(frame["token"].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _source_root(sources) -> str:
    """Best-effort gemeinsamer Wurzelpfad aller latent_dirs eines Splits,
    nur fuer die Manifest-Doku (kein funktionaler Gebrauch). Fallback auf
    die vollstaendige Liste, falls kein gemeinsamer Pfad existiert (z.B.
    verschiedene Laufwerke).
    """
    import os
    latent_dirs = [str(Path(ld).resolve()) for _pkl, ld in sources]
    if len(latent_dirs) == 1:
        return latent_dirs[0]
    try:
        return os.path.commonpath(latent_dirs)
    except ValueError:
        return "; ".join(latent_dirs)


def _sources_and_outdir(data_cfg: dict, split: str):
    """Ermittelt (sources, out_dir) aus dem data-Block je nach Modus/Split."""
    mode = data_cfg.get("mode", "split")

    def _as_tuples(key):
        return [(s["pkl_path"], s["latent_dir"]) for s in data_cfg.get(key, [])]

    if mode == "explicit":
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"explicit-Config: --split muss train/val/test sein, nicht '{split}'."
            )
        sources = _as_tuples(f"{split}_sources")
        out_dir = data_cfg.get(f"{split}_packed_dir")
    elif mode == "split":
        if split != "all":
            raise ValueError(
                "split-Config: --split muss 'all' sein (EIN gemeinsamer Pack fuer "
                "die intern gesplittete Quelle)."
            )
        sources = _as_tuples("sources")
        out_dir = data_cfg.get("packed_dir")
    else:
        raise ValueError(f"data.mode muss 'explicit' oder 'split' sein, nicht '{mode}'.")

    if not sources:
        raise ValueError(f"Keine sources fuer split='{split}' in der Config gefunden.")
    if not out_dir:
        raise ValueError(
            f"Kein packed_dir fuer split='{split}' in der Config gesetzt "
            f"(erwartet: {'%s_packed_dir' % split if mode == 'explicit' else 'packed_dir'})."
        )
    return sources, out_dir


def pack_split(cfg: dict, split: str, dtype_str: str = "float32", force: bool = False,
               verify_samples: int = 500, log_every: int = 2000):
    """Packt EINEN Split aus einem bereits geladenen cfg-Dict.

    Von train_linux.py (Auto-Pack beim Start) und vom CLI-Wrapper pack()
    gleichermassen genutzt -- eine Implementierung, kein zweiter Codepfad.
    """
    if dtype_str not in DTYPES:
        raise ValueError(f"packed_dtype muss float32 oder float16 sein, nicht '{dtype_str}'.")
    store_dtype = DTYPES[dtype_str]

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    n_input_frames = model_cfg.get("n_frames", 3)
    scene_gap_sec = data_cfg.get("scene_gap_sec", 2.0)

    sources, out_dir = _sources_and_outdir(data_cfg, split)
    out_dir = Path(out_dir)
    out_path = out_dir / "packed.npy"

    print(f"[pack] Split  : {split}  (mode={data_cfg.get('mode', 'split')})", flush=True)
    print(f"[pack] dtype  : {dtype_str}", flush=True)
    print(f"[pack] Ziel   : {out_path}", flush=True)

    manifest_path = out_dir / "manifest.json"

    # Idempotenz: existiert der Pack schon, Schritt ueberspringen (billig, VOR
    # dem Dataset-Aufbau). Nur Header lesen (mmap_mode='r' liest keine Daten).
    # Der Laufzeit-Guard (shape[0] == N) faengt einen zur aktuellen Config
    # unpassenden Alt-Pack beim Training ohnehin ab. --force erzwingt Neu-Bau.
    if out_path.exists() and not force:
        existing = np.load(str(out_path), mmap_mode="r")
        print(f"[pack] SKIP: existiert bereits "
              f"(shape={tuple(existing.shape)}, dtype={existing.dtype}).", flush=True)
        print(f"[pack]       Neu erstellen mit --force.", flush=True)
        del existing
        if not manifest_path.exists():
            print(f"[pack] WARNUNG: {manifest_path.name} fehlt neben bestehendem "
                  f"packed.npy (vermutlich vor Einfuehrung des Manifests gepackt). "
                  f"Mit --force neu erzeugen, falls das Audit-Artefakt gebraucht wird.",
                  flush=True)
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-Instanz OHNE packed_dir -> liest .npy, weist _packed_row zu.
    ds_cfg = BEVDatasetConfig(
        sources=sources,
        n_input_frames=n_input_frames,
        scene_gap_sec=scene_gap_sec,
        h5_path=None,
        packed_dir=None,          # bewusst: hier wird GESCHRIEBEN, nicht gelesen
    )
    ds = BEVLatentDataset(ds_cfg, scene_indices=None)

    n_frames = ds._n_packed_frames
    if n_frames == 0:
        raise RuntimeError("Dataset hat 0 Frames -- nichts zu packen.")

    # Shape/dtype am ersten Frame bestimmen (nicht hart [256,128,128] annehmen).
    _, first_frame = next(ds.iter_pack_order())
    sample = ds._load_latent(first_frame)          # [C,H,W] float32
    frame_shape = sample.shape
    print(f"[pack] Frames : {n_frames}", flush=True)
    print(f"[pack] Shape  : {frame_shape} pro Frame", flush=True)

    total_bytes = n_frames * int(np.prod(frame_shape)) * np.dtype(store_dtype).itemsize
    print(f"[pack] Groesse: ~{total_bytes / 1e9:.1f} GB", flush=True)

    mm = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=store_dtype,
        shape=(n_frames, *frame_shape),
    )

    written = 0
    seen_rows = set()
    t0 = time.time()
    for row, frame in ds.iter_pack_order():
        arr = ds._load_latent(frame)               # .npy-Pfad, [C,H,W] float32
        if arr.shape != frame_shape:
            raise RuntimeError(
                f"Inkonsistente Frame-Shape in Zeile {row}: {arr.shape} != {frame_shape}"
            )
        mm[row] = arr.astype(store_dtype)
        seen_rows.add(row)
        written += 1
        if written % log_every == 0:
            dt = time.time() - t0
            rate = written / dt
            eta = (n_frames - written) / rate if rate > 0 else float("nan")
            print(f"[pack]   {written}/{n_frames}  ({rate:.0f} f/s, ETA {eta/60:.1f} min)",
                  flush=True)

    mm.flush()
    del mm  # Memmap schliessen, bevor wir zur Verifikation neu oeffnen

    # Vollstaendigkeit: jede Zeile 0..N-1 genau einmal geschrieben?
    if len(seen_rows) != n_frames or min(seen_rows) != 0 or max(seen_rows) != n_frames - 1:
        raise RuntimeError(
            f"Zeilen-Luecke: {len(seen_rows)} eindeutige Zeilen geschrieben, "
            f"erwartet {n_frames} lueckenlos 0..{n_frames - 1}."
        )
    print(f"[pack] Fertig : {written} Frames in {(time.time() - t0)/60:.1f} min", flush=True)

    # ------------------------------------------------------------------
    # Innerer Korrektheits-Check (Stichprobe): .npy-Pfad vs. packed-Pfad
    # ------------------------------------------------------------------
    print(f"[verify] Stichprobe {verify_samples} Frames (.npy vs packed)...", flush=True)
    packed_arr = np.load(str(out_path), mmap_mode="r")
    rng = np.random.default_rng(0)
    all_rows = [row for row, _ in ds.iter_pack_order()]
    frames_by_row = {row: frame for row, frame in ds.iter_pack_order()}
    check_rows = rng.choice(all_rows, size=min(verify_samples, n_frames), replace=False)

    max_abs = 0.0
    n_exact = 0
    for row in check_rows:
        frame = frames_by_row[int(row)]
        ref = ds._load_latent(frame)                          # .npy -> float32
        got = packed_arr[int(row)].astype("float32")          # packed -> float32
        diff = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        max_abs = max(max_abs, diff)
        if diff == 0.0:
            n_exact += 1

    print(f"[verify] max |Delta| = {max_abs:.3e}", flush=True)
    print(f"[verify] exakt gleich: {n_exact}/{len(check_rows)}", flush=True)
    if store_dtype == np.float32:
        if max_abs != 0.0:
            raise RuntimeError(
                f"float32-Pack ist NICHT bit-identisch (max|Delta|={max_abs:.3e}) "
                f"-- das darf bei verlustfreiem Umkopieren nicht passieren."
            )
        print("[verify] float32: bit-identisch. OK.", flush=True)
    else:
        print(f"[verify] float16: Rundungsfehler erwartet (max|Delta|={max_abs:.3e}). "
              f"Aeusserer Anker-Lauf entscheidet.", flush=True)

    # ------------------------------------------------------------------
    # Manifest schreiben -- NUR nach bestandenem Verify,
    # d.h. manifest.json existiert nur fuer einen validierten Pack.
    # Audit-/Doku-Artefakt: wird NICHT als Laufzeit-Index gelesen (siehe
    # Modulkopf) -- rein informativ + zur Erkennung stiller Mismatches.
    # ------------------------------------------------------------------
    print(f"[manifest] Scene-Order-Hash berechnen...", flush=True)
    manifest = {
        "split":            split,
        "dtype":            dtype_str,
        "shape":            [n_frames, *frame_shape],
        "num_frames":       n_frames,
        "config_hash":      _config_hash(cfg),
        "git_commit":       _git_commit(),          # None (JSON: null) falls kein Git-Repo
        "source_root":      _source_root(sources),
        "scene_order_hash": _scene_order_hash(ds),
        "created_at":       datetime.now(timezone.utc).isoformat(),
        # Zusatzfelder ueber die Arbeitsplan-Mindestmenge hinaus, gleicher
        # Aufwand, erhoehen den Audit-Wert:
        "verify_samples":   int(len(check_rows)),
        "verify_max_abs_diff": max_abs,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] Geschrieben: {manifest_path}", flush=True)
    if manifest["git_commit"] is None:
        print(f"[manifest]   git_commit=null (kein Git-Repo -- Server/Cluster "
              f"laufen derzeit ohne Versionierung, erwartet)", flush=True)

    print(f"\n[pack] packed_dir fuer die Config: {out_dir}", flush=True)
    return out_path


def pack(config_path: str, split: str, dtype_str: str = "float32", force: bool = False,
         verify_samples: int = 500):
    """CLI-Wrapper: laedt die YAML und packt einen Split (Standalone-Nutzung)."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return pack_split(cfg, split, dtype_str, force=force, verify_samples=verify_samples)


def _configured_splits(data_cfg: dict):
    """Welche Splits sind fuer den aktuellen mode zu packen?"""
    mode = data_cfg.get("mode", "split")
    if mode == "explicit":
        # test nur, wenn sowohl sources als auch packed_dir gesetzt sind
        splits = ["train", "val"]
        if data_cfg.get("test_sources") and data_cfg.get("test_packed_dir"):
            splits.append("test")
        return splits
    return ["all"]


def ensure_packed(cfg: dict, dtype_str: str = "float32", force: bool = False,
                  verify_samples: int = 500):
    """Packt alle fuer den aktuellen data.mode noetigen Splits, falls noch nicht
    vorhanden (idempotent -- Skip-Check greift pro Split). Vom Trainingsstart
    aufgerufen, damit Packen ein abschaltbarer Teil des train-Zyklus ist statt
    einer separaten Pipeline. Existiert der Pack schon, ist der Aufruf quasi
    kostenlos (nur Header-Read, kein Dataset-Aufbau).
    """
    data_cfg = cfg["data"]
    for split in _configured_splits(data_cfg):
        pack_split(cfg, split, dtype_str, force=force, verify_samples=verify_samples)


def main():
    ap = argparse.ArgumentParser(description="BEV-Latents in Memmap-Container packen.")
    ap.add_argument("--config", required=True, help="Pfad zur YAML-Config")
    ap.add_argument("--split", required=True,
                    help="explicit: train/val/test  |  split-Config: all")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES.keys()),
                    help="Storage-dtype (float32=verlustfrei, float16=halber Speicher)")
    ap.add_argument("--force", action="store_true",
                    help="Vorhandenes packed.npy ueberschreiben statt ueberspringen")
    ap.add_argument("--verify_samples", type=int, default=500,
                    help="Anzahl Frames fuer den inneren Korrektheits-Check")
    args = ap.parse_args()
    pack(args.config, args.split, args.dtype, force=args.force,
         verify_samples=args.verify_samples)


if __name__ == "__main__":
    main()
