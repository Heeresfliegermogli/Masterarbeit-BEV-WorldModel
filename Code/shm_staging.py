"""
shm_staging.py -- BEV World Model: node-lokales /dev/shm-Staging
==============================================================================

Kopiert die gepackten Memmap-Dateien (packed.npy + manifest.json) einmalig pro
Job-Start von BeeGFS (Netzwerk) ins knoten-lokale RAM-Dateisystem /dev/shm und
biegt die packed_dir-Pfade der Config dorthin um. Danach liest der DataLoader
mit RAM-Speed statt ueber die geteilte BeeGFS-Netzwerkleitung.

WARUM (Cluster-Befund Anker-Lauf):
    Trotz Memmap-Packing lag der data_time-Anteil bei ~89% (Epochenzeit ~3100s
    vs. lokal ~860s) -- die A100 langweilt sich, weil BeeGFS die Daten nicht
    schnell genug liefert. /dev/shm nimmt das Netzwerk ganz aus dem Loop.

INTEGRATION (ein Schalter, analog use_packed):
    data.stage_to_shm: true   -> train_linux.py ruft stage_and_rewrite() auf;
                                 Dateien werden nach /dev/shm kopiert (idempotent),
                                 die *_packed_dir-Keys der Config zeigen danach
                                 auf /dev/shm. Erfordert data.use_packed: true.
    data.stage_to_shm: false/weg -> kein Staging, Config unveraendert.

CLEANUP-STRATEGIE (bewusst modular -- Ansatz A jetzt, B/C spaeter erweiterbar):
    Die Strategie ist ueber CLEANUP_STRATEGIES + data.shm_cleanup austauschbar,
    ohne die Kopier-/Umbiege-Logik anzufassen:
      "job_local" (Ansatz A, DEFAULT): jeder Job kopiert in ein EIGENES
          Verzeichnis /dev/shm/<user>_<jobid>/ und loescht NUR dieses beim
          Job-Ende (trap im sbatch-Skript ruft cleanup() auf). Konfliktfrei,
          kein Teilen -> bei mehreren parallelen fp32-Jobs auf einem Knoten
          passt es evtl. nicht in /dev/shm (2x535GB > 1008GB); mit fp16 schon.
      "shared_refcount" (Ansatz B, NICHT implementiert): geteiltes Verzeichnis
          mit Referenzzaehlung, nur der letzte Job loescht.
      "shared_manual" (Ansatz C, NICHT implementiert): geteiltes Verzeichnis,
          kein Auto-Cleanup, separates Aufraeum-Kommando.
    Zum Aktivieren von B/C spaeter: neue Strategie-Funktion registrieren und
    data.shm_cleanup setzen -- train_linux.py und der Rest bleiben unveraendert.

VERWENDUNG (durch train_linux.py, nicht direkt):
    import shm_staging
    shm_staging.stage_and_rewrite(cfg)          # biegt cfg["data"][*_packed_dir] um
    ...
    shm_staging.cleanup(cfg)                     # am Job-Ende (auch via sbatch-trap)

STANDALONE-STAGING (Pre-Stage vor Parallel-Laeufen):
    python -u shm_staging.py --stage --config <cfg.yaml>   # kopiert einmal nach /dev/shm

STANDALONE-CLEANUP (fuer sbatch-trap / manuelles Aufraeumen):
    python -u shm_staging.py --cleanup --config <cfg.yaml>
    python -u shm_staging.py --cleanup --job-dir /dev/shm/vima_123456   # gezielt
"""

import argparse
import getpass
import os
import shutil
import time
from pathlib import Path

import yaml


SHM_ROOT = Path("/dev/shm")

# Dateien, die pro Split gestaged werden (packed.npy Pflicht, manifest.json
# best-effort -- fuer die Laufzeit nicht noetig, aber praktisch mitzunehmen).
_STAGE_FILES = ["packed.npy", "manifest.json"]


# ---------------------------------------------------------------------------
# Pfad-Konventionen (eine Stelle -- Cleanup und Staging muessen konsistent sein)
# ---------------------------------------------------------------------------

def _job_tag() -> str:
    """Eindeutiger Tag fuer dieses Job/diese Session.

    Auf dem Cluster liefert SLURM_JOB_ID einen pro-Job eindeutigen Wert. Ohne
    SLURM (lokal/interaktiv) Fallback auf PID -- dann ist die Kopie an den
    Prozess gebunden, was fuer Ansatz A genau richtig ist.
    """
    return os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"


def _job_local_dir() -> Path:
    """Ansatz A: job-eigenes /dev/shm-Verzeichnis /dev/shm/<user>_<tag>/."""
    return SHM_ROOT / f"{getpass.getuser()}_{_job_tag()}"


# Cleanup-Strategien geben das/die zu loeschende(n) Verzeichnis(se) zurueck.
# Neue Strategien (B/C) hier registrieren -- der Rest des Moduls ruft nur
# _target_root(strategy) und die Cleanup-Logik generisch auf.
CLEANUP_STRATEGIES = {
    "job_local": _job_local_dir,
    # Parallel-Laeufe: mehrere Prozesse im SELBEN Job teilen die eine
    # /dev/shm-Kopie (gleiche SLURM_JOB_ID -> gleicher Pfad; memmap-Seiten
    # werden read-only geteilt, EINE RAM-Kopie fuer alle). Der cleanup()-Guard
    # unten behandelt jede Nicht-job_local-Strategie als No-Op -> KEIN
    # Prozess-Ende-Cleanup; der sbatch-trap raeumt am JOB-Ende. Staging muss
    # VOR dem Parallel-Start einmal sequenziell laufen (Skip-Check ist nicht
    # race-sicher bei gleichzeitigem Kaltstart).
    "job_shared": _job_local_dir,
    # "shared_refcount": _shared_dir,   # Ansatz B -- spaeter
    # "shared_manual":   _shared_dir,   # Ansatz C -- spaeter
}


def _target_root(strategy: str) -> Path:
    if strategy not in CLEANUP_STRATEGIES:
        raise ValueError(
            f"Unbekannte shm_cleanup-Strategie '{strategy}'. "
            f"Verfuegbar: {sorted(CLEANUP_STRATEGIES)}."
        )
    return CLEANUP_STRATEGIES[strategy]()


# ---------------------------------------------------------------------------
# Konfig-Helfer: welche packed_dirs sind zu stagen?
# ---------------------------------------------------------------------------

def _packed_dir_keys(data_cfg: dict):
    """Liefert die im aktuellen mode relevanten *_packed_dir-Config-Keys."""
    mode = data_cfg.get("mode", "split")
    if mode == "explicit":
        keys = ["train_packed_dir", "val_packed_dir"]
        if data_cfg.get("test_packed_dir"):
            keys.append("test_packed_dir")
        return keys
    return ["packed_dir"]


def _free_bytes(path: Path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


# ---------------------------------------------------------------------------
# Kernoperation: kopieren + Config-Pfade umbiegen
# ---------------------------------------------------------------------------

def stage_and_rewrite(cfg: dict) -> dict:
    """Staged die packed.npy-Dateien nach /dev/shm und biegt cfg IN-PLACE um.

    Idempotent: existiert eine gueltige Kopie (packed.npy vorhanden, Groesse
    passt), wird NICHT neu kopiert -- so kann ein zweiter Job auf demselben
    Knoten (bei geteilten Strategien) bzw. ein Restart dieselbe Kopie nutzen.
    Bei Ansatz A ist das Zielverzeichnis job-eigen, d.h. praktisch immer frisch.

    Gibt cfg zurueck (dieselbe Instanz, IN-PLACE geaendert).
    """
    if not cfg["data"].get("stage_to_shm", False):
        return cfg

    if not cfg["data"].get("use_packed", False):
        raise ValueError(
            "data.stage_to_shm=true erfordert data.use_packed=true "
            "(es werden die gepackten Dateien gestaged, nicht die .npy-Einzeldateien)."
        )

    data_cfg = cfg["data"]
    strategy = data_cfg.get("shm_cleanup", "job_local")
    target_root = _target_root(strategy)

    if not SHM_ROOT.exists():
        raise RuntimeError(f"{SHM_ROOT} existiert nicht -- kein tmpfs auf diesem Knoten?")

    keys = _packed_dir_keys(data_cfg)

    # --- Groessen-Vorabpruefung (fail-fast, bevor irgendwas kopiert wird) ---
    # /dev/shm ist node-lokal und Multi-Tenant -> vorher pruefen ob genug frei.
    need = 0
    srcs = {}
    for key in keys:
        src_dir = Path(data_cfg[key])
        src_file = src_dir / "packed.npy"
        if not src_file.exists():
            raise FileNotFoundError(
                f"stage_to_shm: {src_file} nicht gefunden (use_packed haette es "
                f"erzeugen sollen -- lief ensure_packed vor dem Staging?)."
            )
        size = src_file.stat().st_size
        # Fix: bereits vollstaendig gestagte Splits (job_shared-
        # Pre-Staging im sbatch) zaehlen NICHT zum Platzbedarf -- sonst
        # scheitert der spaetere Skip-Copy-Fall an der Vorabpruefung, sobald
        # 2x Bedarf > /dev/shm (Det-Packs 567G; bei Seg 286G nie getriggert).
        dst_probe = target_root / src_dir.name / "packed.npy"
        if not (dst_probe.exists() and dst_probe.stat().st_size == size):
            need += size
        srcs[key] = (src_dir, src_file, size)

    free = _free_bytes(SHM_ROOT)
    print(f"[shm] Strategie: {strategy} | Ziel: {target_root}", flush=True)
    print(f"[shm] Benoetigt: ~{need/1e9:.1f} GB | frei in {SHM_ROOT}: "
          f"~{free/1e9:.1f} GB", flush=True)
    # 5% Sicherheitspuffer, damit wir den Knoten nicht bis auf 0 fuellen
    if need > free * 0.95:
        raise RuntimeError(
            f"stage_to_shm: nicht genug Platz in {SHM_ROOT} "
            f"(brauche ~{need/1e9:.1f} GB, frei ~{free/1e9:.1f} GB). "
            f"Knoten evtl. durch andere Nutzer belegt. Optionen: float16-Storage "
            f"(halbe Groesse), spaeter erneut, oder anderen Knoten anfordern."
        )

    # --- Kopieren (pro Split), Config-Key auf /dev/shm umbiegen ---
    for key in keys:
        src_dir, src_file, size = srcs[key]
        # Zielunterordner je Split, damit mehrere packed.npy nicht kollidieren
        dst_dir = target_root / src_dir.name
        dst_file = dst_dir / "packed.npy"

        if dst_file.exists() and dst_file.stat().st_size == size:
            print(f"[shm] {key}: bereits gestaged ({dst_file}), Skip-Copy.", flush=True)
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fname in _STAGE_FILES:
                s = src_dir / fname
                if s.exists():
                    t0 = time.time()
                    shutil.copy2(str(s), str(dst_dir / fname))
                    if fname == "packed.npy":
                        dt = time.time() - t0
                        rate = size / dt / 1e9 if dt > 0 else float("nan")
                        print(f"[shm] {key}: {fname} kopiert "
                              f"({size/1e9:.1f} GB in {dt:.0f}s, {rate:.2f} GB/s)",
                              flush=True)

        # Config-Pfad umbiegen: Laufzeit liest jetzt aus /dev/shm
        data_cfg[key] = str(dst_dir)

    print(f"[shm] Staging fertig. packed_dir-Keys zeigen auf {target_root}.", flush=True)
    return cfg


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup(cfg: dict = None, strategy: str = None, job_dir: str = None):
    """Loescht die gestagten /dev/shm-Daten dieses Jobs.

    Aufrufwege:
      - aus train_linux.py am normalen Job-Ende: cleanup(cfg)
      - aus dem sbatch-trap (auch bei scancel/Timeout/OOM): via
        `python shm_staging.py --cleanup --config <cfg>` ODER
        `--job-dir <pfad>` (falls die Config dort nicht verfuegbar ist)

    Ansatz A ("job_local"): loescht das job-eigene Verzeichnis -> konfliktfrei,
    weil kein anderer Job dieselbe Kopie nutzt. Bei kuenftigen geteilten
    Strategien (B/C) wuerde hier stattdessen die Refcount-/No-Op-Logik der
    jeweiligen Strategie greifen (dann ueber die Strategie-Registry geloest).
    """
    if job_dir:
        target = Path(job_dir)
    else:
        if strategy is None:
            strategy = (cfg or {}).get("data", {}).get("shm_cleanup", "job_local") \
                if cfg else "job_local"
        # Fuer geteilte Strategien waere hier die Entscheidung "darf ich loeschen?"
        # zu treffen. Ansatz A: immer ja (job-eigen).
        if strategy != "job_local":
            print(f"[shm cleanup] Strategie '{strategy}' hat keine Auto-Cleanup-Logik "
                  f"(nur 'job_local' implementiert) -- nichts geloescht.", flush=True)
            return
        target = _job_local_dir()

    if target.exists():
        # Sicherheitsnetz: nur unterhalb von /dev/shm loeschen, nie woanders
        try:
            target.resolve().relative_to(SHM_ROOT.resolve())
        except ValueError:
            print(f"[shm cleanup] ABBRUCH: {target} liegt nicht unter {SHM_ROOT} "
                  f"-- aus Sicherheit nichts geloescht.", flush=True)
            return
        shutil.rmtree(str(target), ignore_errors=True)
        print(f"[shm cleanup] Geloescht: {target}", flush=True)
    else:
        print(f"[shm cleanup] Nichts zu loeschen ({target} existiert nicht).", flush=True)


def main():
    ap = argparse.ArgumentParser(description="/dev/shm-Staging")
    ap.add_argument("--stage", action="store_true",
                    help="Packed-Dateien EINMALIG nach /dev/shm kopieren "
                         "(Pre-Stage vor Parallel-Laeufen). Braucht --config.")
    ap.add_argument("--cleanup", action="store_true", help="Gestagte Daten loeschen")
    ap.add_argument("--config", type=str, default=None,
                    help="Config, aus der Pfade/Cleanup-Strategie gelesen werden")
    ap.add_argument("--job-dir", type=str, default=None,
                    help="Konkretes /dev/shm-Verzeichnis (ueberschreibt Strategie; nur --cleanup)")
    args = ap.parse_args()

    # Genau EINE Aktion pro Aufruf (XOR) -- schuetzt vor versehentlichem Doppel.
    if args.stage == args.cleanup:
        ap.error("Genau EINES von --stage oder --cleanup angeben.")

    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if args.stage:
        # einmaliges, sequenzielles Vorstagen VOR dem Parallel-Start.
        # Befuellt /dev/shm/<user>_<jobid>/ (gleiche SLURM_JOB_ID wie die
        # nachfolgenden Trainings -> die treffen danach den Skip-Copy, race-frei).
        # Das zurueckgegebene (umgebogene) cfg wird verworfen: jeder
        # Trainingsprozess biegt seine EIGENE In-Memory-Config selbst um.
        if cfg is None:
            ap.error("--stage braucht --config (Pfade + Ziel-Strategie).")
        stage_and_rewrite(cfg)
    else:
        cleanup(cfg=cfg, job_dir=args.job_dir)


if __name__ == "__main__":
    main()
