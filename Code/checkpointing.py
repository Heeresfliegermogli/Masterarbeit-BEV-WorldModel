"""
checkpointing.py — BEV World Model: Checkpoint-Verwaltung
==========================================================

Speichert und lädt Modell-Weights getrennt nach Phase.

WARUM EIN EIGENES MODUL?
    Das Speichern von Weights klingt trivial — ist aber fehleranfällig.
    Häufige Fehler:
        - Nur model.state_dict() speichern, aber optimizer vergessen
          → Training kann nicht fortgesetzt werden (Momentum verloren)
        - Keine Config mitspeichern
          → Beim Laden weiß man nicht mehr welche Architektur dahintersteckt
        - Phase 1 Checkpoint in Phase 2 Modell laden
          → Shape-Mismatch Fehler, schwer zu debuggen

    Dieses Modul macht es richtig — ein Checkpoint enthält ALLES
    was man braucht um Training fortzusetzen oder Ergebnisse zu reproduzieren.

DATEISTRUKTUR:
    checkpoints/
    ├── phase1/
    │   ├── best_val_loss.pt      ← bestes Modell (nach Val-Loss)
    │   ├── epoch_010.pt          ← periodische Snapshots
    │   └── epoch_020.pt
    └── phase2/
        ├── best_val_loss.pt
        └── epoch_010.pt

WAS IN EINEM CHECKPOINT STECKT:
    {
        "model_state_dict":     OrderedDict aller Parameter + Buffer
        "optimizer_state_dict": AdamW Momentum/Variance für alle Parameter
        "scheduler_state_dict": CosineAnnealingLR aktueller Schritt
        "epoch":                int — welche Epoch wurde gerade abgeschlossen
        "val_loss":             float — Val-Loss dieser Epoch
        "best_val_loss":        float — bestes Val-Loss bisher
        "config":               dict — alle ModelConfig Parameter
        "phase":                str — "frame" oder "cell"
        "timestamp":            str — wann gespeichert (für Logs)
    }
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from config import ModelConfig


# ---------------------------------------------------------------------------
# Speichern
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    config: ModelConfig,
    checkpoint_dir: str,
    filename: str,
    best_miou: float = -1.0,            # mIoU-Plateau-Resume
    best_epoch_miou: int = -1,
    miou_patience_cnt: int = 0,
    miou_stop_ref: float = -1.0,
    patience_cnt: int = 0,
) -> Path:
    """
    Speichert einen vollständigen Trainings-Checkpoint.

    Ein Checkpoint ist mehr als nur die Weights — er ist ein vollständiger
    Snapshot des Trainingszustands. Mit ihm kann man:
        1. Training genau an dieser Stelle fortsetzen (epoch, optimizer)
        2. Das Modell für Inference laden (model_state_dict, config)
        3. Nachvollziehen wann und wie gut das Modell war (val_loss, timestamp)

    Args:
        model:          Das BEVWorldModel (oder jedes nn.Module)
        optimizer:      AdamW Instanz — enthält Momentum + Variance
        scheduler:      CosineAnnealingLR — enthält aktuellen LR-Schritt
        epoch:          Aktuelle Epoch (0-basiert)
        val_loss:       Val-Loss dieser Epoch
        best_val_loss:  Bestes Val-Loss über alle Epochs bisher
        config:         ModelConfig — wird als dict gespeichert
        checkpoint_dir: Basisverzeichnis (z.B. "checkpoints")
        filename:       Dateiname (z.B. "best_val_loss.pt" oder "epoch_010.pt")

    Returns:
        Path zur gespeicherten Datei
    """
    # Verzeichnis anlegen: checkpoints/phase1/ oder checkpoints/phase2/
    save_dir = Path(checkpoint_dir) / f"phase{1 if config.phase == 'frame' else 2}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    checkpoint = {
        # --- Modell ---
        # state_dict() gibt ein OrderedDict: {"layer.weight": Tensor, ...}
        # Es enthält Parameter (trainierbar) UND Buffer (nicht trainierbar,
        # z.B. BatchNorm running_mean). Beides wird für korrektes Laden benötigt.
        "model_state_dict": model.state_dict(),

        # --- Optimizer ---
        # AdamW speichert für jeden Parameter:
        #   - exp_avg:   gewichteter Mittelwert der Gradienten (Momentum)
        #   - exp_avg_sq: gewichteter Mittelwert der quadrierten Gradienten
        #   - step:      wie oft dieser Parameter schon geupdated wurde
        # Ohne das: Training kann zwar fortgesetzt werden, aber die ersten
        # Epochs nach dem Laden sind "kalt" — als ob von vorne angefangen.
        "optimizer_state_dict": optimizer.state_dict(),

        # --- Scheduler ---
        # CosineAnnealingLR weiß wo im Cosinus-Zyklus es gerade ist.
        # Ohne das: LR springt beim Laden zurück zum Anfang.
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,

        # --- Trainingsfortschritt ---
        "epoch":          epoch,
        "val_loss":       val_loss,
        "best_val_loss":  best_val_loss,

        # --- mIoU-Plateau-Zustand ---
        # Fehlt in Checkpoints von VOR diesem Patch -> Aufrufer nutzt beim
        # Laden .get()-Defaults (= altes Frisch-Init-Verhalten). Kein Bruch
        # bestehender Checkpoints (z.B. best_miou.pt).
        "best_miou":         best_miou,
        "best_epoch_miou":   best_epoch_miou,
        "miou_patience_cnt": miou_patience_cnt,
        "miou_stop_ref":     miou_stop_ref,
        "patience_cnt":      patience_cnt,

        # --- Architektur-Info ---
        # Wichtig: config als plain dict speichern, nicht als dataclass-Objekt.
        # dataclass-Objekte könnten beim Laden Probleme machen wenn die Klasse
        # sich geändert hat. Ein dict ist immer lesbar.
        "config":  asdict(config),
        "phase":   config.phase,

        # --- Reproduzierbarkeit ---
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Gespeichert: {save_path}  "
          f"(epoch={epoch}, val_loss={val_loss:.6f})")
    return save_path


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
    allow_missing: bool = False,   # 18/B2: strict=False-Laden fuer Modelle mit
                                   # NEUEN Zusatzmodulen (z.B. flow.*), die im
                                   # alten Checkpoint fehlen duerfen. Unexpected
                                   # Keys bleiben trotzdem ein Fehler.
) -> dict:
    """
    Lädt einen Checkpoint und stellt den Trainingszustand wieder her.

    SICHERHEITSCHECK:
        Bevor Weights geladen werden, prüft die Funktion ob die Phase
        des Checkpoints zur Phase des Modells passt. Ein Phase-1-Checkpoint
        in ein Phase-2-Modell laden würde einen kryptischen Shape-Mismatch
        Fehler erzeugen — dieser Check gibt eine klare Fehlermeldung.

    Args:
        checkpoint_path: Pfad zur .pt Datei
        model:           Instanziiertes BEVWorldModel (muss zur Phase passen!)
        optimizer:       Optional — wenn None, wird nur das Modell geladen
                         (nützlich für Inference oder Evaluation)
        scheduler:       Optional — analog zu optimizer
        device:          "cpu" oder "cuda" — wohin die Tensoren geladen werden

    Returns:
        Das vollständige Checkpoint-Dict (für Logging, epoch-Fortsetzung, etc.)

    Raises:
        ValueError:  Wenn Phase des Checkpoints nicht zur Modell-Phase passt
        FileNotFoundError: Wenn checkpoint_path nicht existiert
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")

    # map_location: Checkpoint auf das gewünschte Device laden.
    # Wichtig wenn der Checkpoint auf GPU gespeichert aber auf CPU geladen wird
    # (z.B. für Evaluation auf einem Rechner ohne GPU).
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Phasen-Check ---
    # Das ist die wichtigste Sicherheitsüberprüfung.
    # Wenn wir hier nicht prüfen, gibt es später einen Shape-Mismatch Error
    # der schwer zu debuggen ist ("expected shape [32, 128] but got [3, 256]")
    saved_phase = checkpoint.get("phase", "unknown")
    model_config = getattr(model, "config", None)

    if model_config is not None and saved_phase != model_config.phase:
        raise ValueError(
            f"Phasen-Mismatch!\n"
            f"  Checkpoint: phase='{saved_phase}'\n"
            f"  Modell:     phase='{model_config.phase}'\n"
            f"  Ein Phase-1-Checkpoint kann nicht in ein Phase-2-Modell geladen werden.\n"
            f"  Stelle sicher dass du das richtige Checkpoint-Verzeichnis verwendest."
        )

    # --- Modell-Weights laden ---
    # strict=True (Standard): ALLE Parameter müssen übereinstimmen.
    # Wenn ein Parameter fehlt oder ein Extra-Parameter vorhanden ist → Fehler.
    # Das ist das gewünschte Verhalten — silent mismatches sind gefährlich.
    # AUSNAHME (18/B2, allow_missing=True): Modelle mit neuen Zusatzmodulen
    # (flow.*) laden alte Checkpoints mit strict=False; fehlende Keys behalten
    # ihre frische Initialisierung. Unexpected Keys sind WEITER ein Fehler.
    if allow_missing:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(
                f"[Checkpoint] Unexpected Keys beim allow_missing-Laden: "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if missing:
            print(f"[Checkpoint] allow_missing: {len(missing)} Keys frisch "
                  f"initialisiert (z.B. {missing[0]})")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # --- Optimizer + Scheduler wiederherstellen (nur für Training-Fortsetzung) ---
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch     = checkpoint.get("epoch", 0)
    val_loss  = checkpoint.get("val_loss", float("inf"))
    timestamp = checkpoint.get("timestamp", "unbekannt")

    print(f"[Checkpoint] Geladen: {checkpoint_path}")
    print(f"  Phase:     {saved_phase}")
    print(f"  Epoch:     {epoch}")
    print(f"  Val-Loss:  {val_loss:.6f}")
    print(f"  Gespeichert: {timestamp}")

    return checkpoint


# ---------------------------------------------------------------------------
# Hilfsfunktion: besten Checkpoint finden
# ---------------------------------------------------------------------------

def find_best_checkpoint(checkpoint_dir: str, phase: str) -> Optional[Path]:
    """
    Gibt den Pfad zum besten Val-Loss Checkpoint zurück, falls vorhanden.

    Nützlich beim Starten von Evaluation oder Ablation-Runs:
    Man muss den genauen Dateinamen nicht kennen.

    Args:
        checkpoint_dir: Basisverzeichnis (z.B. "checkpoints")
        phase:          "frame" oder "cell"

    Returns:
        Path zur best_val_loss.pt, oder None wenn nicht vorhanden
    """
    phase_num  = 1 if phase == "frame" else 2
    best_path  = Path(checkpoint_dir) / f"phase{phase_num}" / "best_val_loss.pt"
    return best_path if best_path.exists() else None
