"""
train.py — BEV World Model: Hauptskript Training (Task 5)
=========================================================

Dieses Skript ist der "Dirigent" — es importiert alle Subskripte aus code/
und orchestriert den gesamten Trainingsprozess.

VERZEICHNISSTRUKTUR:
    bev_worldmodel/
    ├── train.py          ← dieses Skript
    ├── config.yaml       ← alle Hyperparameter
    └── code/
        ├── config.py
        ├── bev_world_model.py
        ├── bev_dataloader.py
        ├── checkpointing.py
        └── ...

AUFRUF:
    # Phase 1 (Frame-Level, 3 Tokens, schnell)
    python train.py --config config.yaml --phase frame

    # Phase 2 (Cell-Level, 3072 Tokens, wissenschaftlicher Kern)
    python train.py --config config.yaml --phase cell

    # Training fortsetzen ab einem Checkpoint
    python train.py --config config.yaml --phase frame \\
        --resume checkpoints/phase1/epoch_020.pt

WARUM sys.path.insert?
    train.py liegt eine Ebene ÜBER code/. Python findet Module nur im
    aktuellen Verzeichnis oder in installierten Paketen — nicht automatisch
    in Unterordnern. sys.path.insert(0, "code") sagt Python:
    "Schau auch in code/ nach Modulen". Danach funktioniert
    `from config import ModelConfig` obwohl config.py in code/ liegt.
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
import os
import time
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# --- code/ dem Python-Suchpfad hinzufügen ---
# __file__ = .../bev_worldmodel/train.py
# Path(__file__).parent = .../bev_worldmodel/
# / "code" = .../bev_worldmodel/code/
CODE_DIR = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE_DIR))

# Jetzt können wir alle Subskripte direkt importieren
from config          import ModelConfig
from bev_world_model import BEVWorldModel
# bev_dataloader wird lazy in build_dataloaders() importiert (smoke_test-kompatibel)
from checkpointing   import save_checkpoint, load_checkpoint, find_best_checkpoint

# wandb ist optional — wenn nicht installiert, wird ein Dummy verwendet
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warnung] wandb nicht installiert. Logging wird übersprungen.")

# matplotlib für Visualisierungen (optional)
try:
    import matplotlib
    matplotlib.use("Agg")   # kein Display nötig (Server ohne Monitor)
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[Warnung] matplotlib nicht installiert. Visualisierungen werden übersprungen.")


# =============================================================================
# BAUSTEIN 1 — Reproduzierbarkeit
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Setzt ALLE Zufallsgeneratoren auf denselben Seed.

    Warum alle?
        - random:           Python built-in (z.B. DataLoader-Shuffling)
        - numpy:            NumPy Operationen
        - torch:            CPU-Tensoren
        - torch.cuda:       GPU-Tensoren (alle GPUs)
        - deterministic:    cuDNN wählt immer denselben Algorithmus

    Ohne das: Zwei identische Runs mit denselben Hyperparametern
    erzeugen unterschiedliche Ergebnisse — das macht Ablation Studies
    unaussagekräftig. seed=42 ist in der gesamten Masterarbeit konstant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministisch machen (minimal langsamer, aber reproduzierbar)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# BAUSTEIN 2 — Config laden
# =============================================================================

def load_config(config_path: str, phase_override: Optional[str] = None) -> dict:
    """
    Liest config.yaml und gibt ein dict zurück.

    Der optionale phase_override kommt vom --phase Argument.
    Er überschreibt model.phase aus der YAML-Datei — so kann man
    denselben config.yaml für beide Phasen verwenden:

        python train.py --config config.yaml --phase frame
        python train.py --config config.yaml --phase cell

    ohne die Datei jedesmal zu editieren.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)     # YAML → Python dict

    if phase_override is not None:
        cfg["model"]["phase"] = phase_override
        print(f"[Config] Phase überschrieben: '{phase_override}'")

    return cfg


# =============================================================================
# BAUSTEIN 3 — Loss Funktion
# =============================================================================

def compute_loss(
    pred:         torch.Tensor,   # [B, 256, 128, 128] — Modell-Output
    target:       torch.Tensor,   # [B, 256, 128, 128] — echter nächster Latent
    use_cosine:   bool  = True,
    lambda_cos:   float = 0.1,
    lambda_dist:  float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kombinierter Loss aus drei Termen — jeder adressiert ein anderes Ziel:

    ┌─────────────────────────────────────────────────────────────────┐
    │ TERM 1: Multi-Scale MSE  (Perceptual Loss)                      │
    │   Warum: Normaler MSE optimiert nur auf Pixel-Ebene.            │
    │   Das Modell lernt "blur" als günstige Strategie — niedrigerer   │
    │   MSE als scharfe Prediction, weil Mittelwerte nah am Target.   │
    │   Multi-Scale zwingt das Modell gleichzeitig:                   │
    │     - Grobe Struktur richtig zu haben (32×32)                   │
    │     - Mittlere Struktur richtig zu haben (64×64)                │
    │     - Feine Details richtig zu haben (128×128)                  │
    │   Formula: (MSE_128 + MSE_64 + MSE_32) / 3                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 2: Cosine Similarity                                       │
    │   Warum: Bei sparsem Input (mean≈0.14) kann MSE kollabieren.    │
    │   "Alles auf 0" gibt niedrigen MSE aber ist inhaltlich falsch.  │
    │   Cosine misst Richtung unabhängig vom Betrag — verhindert      │
    │   den Null-Kollaps.                                             │
    │   Formula: 1 - CosSim(pred.flatten, target.flatten)            │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 3: Channel-Statistik (Distribution Loss)                   │
    │   Warum: Das ist euer PRIMÄRZIEL — predicted Latents sollen     │
    │   dieselbe statistische Verteilung haben wie echte              │
    │   BEVFusion-Latents. Nur dann ist der BEVFusion-Decoder         │
    │   kompatibel (BatchNorm im Decoder ist auf diese Statistiken    │
    │   kalibriert).                                                  │
    │   Was er misst: mean und std pro Channel [B, 256] müssen        │
    │   übereinstimmen. Wenn loss_dist → 0, sind eure Latents         │
    │   statistisch ununterscheidbar von echten BEVFusion-Latents.   │
    │   Das ist euer frühester Indikator während des Trainings ob     │
    │   der BEVFusion-Decoder später funktionieren wird.              │
    │   Formula: MSE(mean_pred, mean_target) + MSE(std_pred,std_target)│
    └─────────────────────────────────────────────────────────────────┘

    Returns:
        (loss_total, loss_mse, loss_cos, loss_dist)
        — alle vier separat für wandb-Logging
    """
    # ------------------------------------------------------------------
    # TERM 1: Multi-Scale MSE
    # ------------------------------------------------------------------
    # Auflösungsstufe 1: Original 128×128
    loss_128 = F.mse_loss(pred, target)

    # Auflösungsstufe 2: 64×64 — AvgPool(2) halbiert H und W
    loss_64  = F.mse_loss(
        F.avg_pool2d(pred,   kernel_size=2),
        F.avg_pool2d(target, kernel_size=2),
    )

    # Auflösungsstufe 3: 32×32 — entspricht dem Transformer-Arbeitsraum
    # Besonders wichtig: das ist die Auflösung auf der der Transformer
    # direkt operiert — Fehler hier sind strukturell, nicht nur Details
    loss_32  = F.mse_loss(
        F.avg_pool2d(pred,   kernel_size=4),
        F.avg_pool2d(target, kernel_size=4),
    )

    # Gleichgewichteter Durchschnitt der drei Skalen
    loss_mse = (loss_128 + loss_64 + loss_32) / 3.0

    # ------------------------------------------------------------------
    # TERM 2: Cosine Similarity
    # ------------------------------------------------------------------
    if use_cosine:
        # Flatten: [B, 256, 128, 128] → [B, 256*128*128]
        # Jeder Batch-Eintrag wird als ein langer Vektor verglichen
        loss_cos = 1.0 - F.cosine_similarity(
            pred.flatten(1),
            target.flatten(1),
            dim=1,
        ).mean()
    else:
        loss_cos = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # TERM 3: Channel-Statistik (Distribution Loss)
    # ------------------------------------------------------------------
    # mean und std über H und W, pro Channel und Batch-Element
    # dim=[-2,-1] bedeutet: über die letzten zwei Dims (H=128, W=128) mitteln
    # Ergebnis: [B, 256] — ein Wert pro Channel pro Sample
    pred_mean   = pred.mean(dim=[-2, -1])      # [B, 256]
    target_mean = target.mean(dim=[-2, -1])    # [B, 256]
    pred_std    = pred.std(dim=[-2, -1])       # [B, 256]
    target_std  = target.std(dim=[-2, -1])     # [B, 256]

    # MSE zwischen den Statistiken — zwei Teilterme:
    #   loss_mean: predicted Channel-Mittelwerte müssen stimmen
    #   loss_std:  predicted Channel-Streuung muss stimmen
    # Wenn beide → 0: predicted Latents sind statistisch kompatibel
    # mit echten BEVFusion-Latents → BEVFusion-Decoder wird funktionieren
    loss_dist = (
        F.mse_loss(pred_mean, target_mean) +
        F.mse_loss(pred_std,  target_std)
    )

    # ------------------------------------------------------------------
    # Kombination
    # ------------------------------------------------------------------
    loss_total = loss_mse + lambda_cos * loss_cos + lambda_dist * loss_dist
    return loss_total, loss_mse, loss_cos, loss_dist


# =============================================================================
# BAUSTEIN 4 — Eine Trainings-Epoch
# =============================================================================

def train_one_epoch(
    model:       torch.nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler:      torch.cuda.amp.GradScaler,
    device:      torch.device,
    use_cosine:  bool,
) -> Tuple[float, float, float, float]:
    """
    Führt genau eine Trainings-Epoch durch.

    Iteriert über alle Batches im train_loader, berechnet Loss,
    propagiert Gradienten zurück und updated die Gewichte.

    MIXED PRECISION (autocast + GradScaler):
        Normalerweise: alles in float32 (4 Byte/Zahl)
        Mit Mixed Precision: Forward-Pass in float16 (2 Byte/Zahl)
            → 2× schneller, ~halber VRAM-Bedarf
            → Besonders wichtig für Phase 2 (3072 Tokens)

        GradScaler-Problem:
            float16 hat weniger Präzision — kleine Gradienten werden 0.
            GradScaler multipliziert den Loss intern mit einem großen
            Skalierungsfaktor vor backward(), teilt danach wieder.
            So bleiben auch kleine Gradienten in float16 darstellbar.

    GRADIENT CLIPPING:
        clip_grad_norm_(max_norm=1.0):
            Wenn die Gesamtnorm aller Gradienten > 1.0, werden alle
            Gradienten proportional skaliert bis die Norm = 1.0 ist.
            Verhindert explodierende Gradienten, besonders in den ersten
            Epochs wenn die Weights noch zufällig sind.

    Returns:
        (avg_total_loss, avg_mse_loss, avg_cos_loss) — Durchschnitte über Epoch
    """
    model.train()   # Dropout AN, BatchNorm im Trainings-Modus

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_dist = 0.0
    n_batches  = len(loader)

    for batch in loader:
        # Daten auf GPU (oder CPU wenn kein GPU)
        # non_blocking=True: Transfer läuft asynchron, CPU kann weiterarbeiten
        inputs = batch["input"].to(device, non_blocking=True)   # [B, 3, 256, 128, 128]
        target = batch["target"].to(device, non_blocking=True)  # [B, 256, 128, 128]

        # Gradienten vom letzten Schritt löschen
        # set_to_none=True ist schneller als zero_grad() weil keine Zuweisung auf 0
        optimizer.zero_grad(set_to_none=True)

        # --- Forward Pass in float16 ---
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs)                                 # [B, 256, 128, 128]
            loss, loss_mse, loss_cos, loss_dist = compute_loss(pred, target, use_cosine)

        # --- Backward Pass (skaliert) ---
        scaler.scale(loss).backward()

        # Gradienten de-skalieren bevor Clipping (sonst falscher Vergleich)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Gewichte updaten (GradScaler prüft ob Gradienten inf/nan hatten)
        scaler.step(optimizer)

        # Scale-Faktor für nächsten Schritt anpassen
        # (wird vergrößert wenn alles OK, verkleinert bei inf/nan)
        scaler.update()

        # Losses akkumulieren (.item() löst den Tensor vom Berechnungsgraph)
        total_loss += loss.item()
        total_mse  += loss_mse.item()
        total_cos  += loss_cos.item()
        total_dist += loss_dist.item()

    return (total_loss / n_batches, total_mse / n_batches,
            total_cos  / n_batches, total_dist / n_batches)


# =============================================================================
# BAUSTEIN 5 — Validation
# =============================================================================

def validate(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    use_cosine: bool,
) -> Tuple[float, float, float, float]:
    """
    Wertet das Modell auf dem Validation-Set aus.

    KEIN Gradient-Update — nur Forward-Pass + Loss.

    model.eval() vs model.train():
        eval():   Dropout AUS (deterministische Outputs)
                  BatchNorm nutzt gespeicherte running_mean/var
        train():  Dropout AN (zufällige Neuronen deaktiviert)
                  BatchNorm berechnet Statistiken frisch

    torch.no_grad():
        Sagt PyTorch: "Kein Autograd-Tracking nötig."
        PyTorch speichert normalerweise alle Zwischenergebnisse für
        backward() — das kostet viel Speicher. Mit no_grad() entfällt das.
        Wichtig: nur für Evaluation/Inference, nie im Training!

    Returns:
        (avg_total_loss, avg_mse_loss, avg_cos_loss)
    """
    model.eval()

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_dist = 0.0
    n_batches  = len(loader)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(inputs)
                loss, loss_mse, loss_cos, loss_dist = compute_loss(pred, target, use_cosine)

            total_loss += loss.item()
            total_mse  += loss_mse.item()
            total_cos  += loss_cos.item()
            total_dist += loss_dist.item()

    return (total_loss / n_batches, total_mse / n_batches,
            total_cos  / n_batches, total_dist / n_batches)


# =============================================================================
# BAUSTEIN 6 — Visualisierung für wandb
# =============================================================================

def visualize_predictions(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    epoch:     int,
    n_samples: int = 4,
) -> None:
    """
    Erstellt channel-gemittelte Heatmaps (Predicted vs. Real) und
    lädt sie als wandb.Image hoch.

    WARUM CHANNEL-MITTELUNG?
        Der Latent-Tensor hat 256 Channels — nicht direkt visualisierbar.
        .mean(dim=1): [B, 256, 128, 128] → [B, 128, 128]
        Jeder Pixel zeigt die durchschnittliche Aktivierung über alle Channels.
        Das gibt ein intuitives Bild: helle Bereiche = viel BEV-Aktivität
        (z.B. detektierte Objekte, Straßen), dunkle = leer.

    Wird nur aufgerufen wenn matplotlib UND wandb verfügbar sind.
    """
    if not MPL_AVAILABLE or not WANDB_AVAILABLE:
        return

    model.eval()

    # Einen einzelnen Batch für Visualisierung holen
    # iter() + next() = ersten Batch, ohne den ganzen DataLoader zu durchlaufen
    batch = next(iter(loader))
    inputs = batch["input"][:n_samples].to(device)
    target = batch["target"][:n_samples].to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs)

    # Auf CPU, float32, numpy für matplotlib
    pred_vis   = pred.float().mean(dim=1).cpu().numpy()    # [n_samples, 128, 128]
    target_vis = target.float().mean(dim=1).cpu().numpy()  # [n_samples, 128, 128]

    # Grid: n_samples Zeilen × 3 Spalten (Predicted | Real | Differenz)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]  # Konsistente Indexierung bei n_samples=1

    for i in range(n_samples):
        diff = np.abs(pred_vis[i] - target_vis[i])

        im0 = axes[i, 0].imshow(pred_vis[i],   cmap="viridis", interpolation="nearest")
        axes[i, 0].set_title(f"Sample {i}: Predicted t+1", fontsize=9)
        axes[i, 0].axis("off")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(target_vis[i], cmap="viridis", interpolation="nearest")
        axes[i, 1].set_title(f"Sample {i}: Real t+1", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(diff,           cmap="hot",     interpolation="nearest")
        axes[i, 2].set_title(f"Sample {i}: |Pred - Real|", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Epoch {epoch:03d}", fontsize=11)
    plt.tight_layout()

    wandb.log({"predictions": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


# =============================================================================
# BAUSTEIN 7 — DataLoader aufbauen (aus config.yaml)
# =============================================================================

def build_dataloaders(cfg: dict) -> Tuple:
    """
    Liest data-Sektion aus config.yaml und baut die DataLoader.

    Zwei Modi (aus bev_dataloader.py):
        "split":    Eine PKL, intern auf Train/Val/Test aufgeteilt
                    → für nuScenes Mini / Entwicklung
        "explicit": Getrennte PKLs pro Split
                    → für Full nuScenes (700/150/150 Szenen)

    Die YAML-Konfiguration sieht so aus:

        data:
          mode: "split"
          sources:
            - pkl_path:   "/data/nuscenes/infos_val.pkl"
              latent_dir: "/data/nuscenes/latents/val"
          train_ratio: 0.70
          val_ratio:   0.15

    Diese Funktion übersetzt das in den passenden Funktionsaufruf.
    """
    # Lazy Import: erst hier importieren damit smoke_test.py train.py
    # importieren kann ohne dass bev_dataloader sofort ausgeführt wird
    from bev_dataloader import make_dataloaders_split, make_dataloaders_explicit

    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    mode       = data_cfg.get("mode", "split")

    if mode == "split":
        # sources aus YAML: Liste von {pkl_path, latent_dir} Dicts
        # → umwandeln in Liste von Tupeln (wie bev_dataloader erwartet)
        sources = [
            (s["pkl_path"], s["latent_dir"])
            for s in data_cfg["sources"]
        ]
        loaders = make_dataloaders_split(
            sources       = sources,
            train_ratio   = data_cfg.get("train_ratio", 0.70),
            val_ratio     = data_cfg.get("val_ratio",   0.15),
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
        )

    elif mode == "explicit":
        def _parse(key):
            return [(s["pkl_path"], s["latent_dir"]) for s in data_cfg.get(key, [])]

        loaders = make_dataloaders_explicit(
            train_sources = _parse("train_sources"),
            val_sources   = _parse("val_sources"),
            test_sources  = _parse("test_sources") or None,
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
        )

    else:
        raise ValueError(f"data.mode muss 'split' oder 'explicit' sein, nicht '{mode}'.")

    return loaders["train"], loaders["val"], loaders.get("test")


# =============================================================================
# BAUSTEIN 8 — Hauptschleife
# =============================================================================

def main() -> None:
    """
    Orchestriert den gesamten Trainingsprozess.

    Ablauf:
        1. Argumente parsen, Config laden
        2. Seed setzen (Reproduzierbarkeit)
        3. Device bestimmen (GPU/CPU)
        4. Modell, Optimizer, Scheduler, GradScaler aufbauen
        5. DataLoader aufbauen
        6. wandb initialisieren
        7. Optional: Checkpoint laden (--resume)
        8. Training Loop:
           for epoch in range(epochs):
               train_one_epoch()
               validate()
               scheduler.step()
               wandb.log()
               visualize() (alle log_every Epochs)
               save_checkpoint() (wenn Val-Loss verbessert)
               save_checkpoint() (periodischer Snapshot)
               Early Stopping prüfen
        9. wandb.finish()
    """

    # -------------------------------------------------------------------------
    # 1. Argumente & Config
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="BEV World Model Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Pfad zur config.yaml")
    parser.add_argument("--phase",  type=str, default=None,
                        choices=["frame", "cell"],
                        help="Überschreibt model.phase in config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Pfad zu einem Checkpoint (.pt) zum Fortsetzen")
    args = parser.parse_args()

    cfg = load_config(args.config, args.phase)
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    ckpt_cfg   = cfg.get("checkpoints", {})
    wandb_cfg  = cfg.get("wandb", {})

    # -------------------------------------------------------------------------
    # 2. Reproduzierbarkeit
    # -------------------------------------------------------------------------
    set_seed(train_cfg["seed"])

    # -------------------------------------------------------------------------
    # 3. Device
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BEV World Model Training")
    print(f"  Phase:  {model_cfg['phase']}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # 4. Modell
    # -------------------------------------------------------------------------
    # ModelConfig ist die Python dataclass aus code/config.py
    # **model_cfg entpackt das dict aus der YAML in Keyword-Argumente
    model_config = ModelConfig(**model_cfg)
    model        = BEVWorldModel(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {n_params:,} Parameter  (Phase: {model_config.phase})")
    print(f"Tokens: {model_config.n_tokens} pro Forward-Pass\n")

    # -------------------------------------------------------------------------
    # 4b. Optimizer
    # AdamW = Adam + Weight Decay Korrektur (Loshchilov & Hutter, 2019)
    # Weight Decay wirkt als L2-Regularisierung: drängt Weights Richtung 0
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg["lr"],
        weight_decay = train_cfg["weight_decay"],
    )

    # -------------------------------------------------------------------------
    # 4c. Learning Rate Scheduler
    # CosineAnnealingLR: LR folgt einer Kosinus-Kurve von lr bis eta_min
    #
    #   lr(t) = eta_min + 0.5*(lr - eta_min) * (1 + cos(π * t / T_max))
    #
    # t=0: LR = lr (Maximum)
    # t=T_max/2: LR ≈ eta_min + (lr-eta_min)/2 (Halbzeit)
    # t=T_max: LR = eta_min (Minimum)
    #
    # Vorteil gegenüber StepLR: kein abrupter Sprung, sanftes Abkühlen
    # -------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = train_cfg["epochs"],
        eta_min = 1e-6,
    )

    # -------------------------------------------------------------------------
    # 4d. GradScaler für Mixed Precision
    # enabled=False auf CPU (AMP nur auf CUDA sinnvoll)
    # -------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------------------------------------------------------------------------
    # 5. DataLoader
    # -------------------------------------------------------------------------
    print("Lade Daten...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"  Train: {len(train_loader)} Batches")
    print(f"  Val:   {len(val_loader)} Batches")
    if test_loader:
        print(f"  Test:  {len(test_loader)} Batches")
    print()

    # -------------------------------------------------------------------------
    # 6. wandb initialisieren
    # -------------------------------------------------------------------------
    run_name = f"phase_{model_config.phase}_{time.strftime('%m%d_%H%M')}"

    if WANDB_AVAILABLE:
        wandb.init(
            project = wandb_cfg.get("project", "bev_worldmodel"),
            entity  = wandb_cfg.get("entity"),   # None = persönlicher Account
            name    = run_name,
            config  = cfg,    # gesamte config als wandb-Hyperparameter
        )
        wandb.watch(model, log="gradients", log_freq=100)
    print(f"Run: {run_name}\n")

    # -------------------------------------------------------------------------
    # 7. Optional: Checkpoint laden (--resume)
    # -------------------------------------------------------------------------
    start_epoch   = 0
    best_val_loss = float("inf")
    patience_cnt  = 0

    if args.resume:
        print(f"Lade Checkpoint: {args.resume}")
        ckpt = load_checkpoint(
            checkpoint_path = args.resume,
            model           = model,
            optimizer       = optimizer,
            scheduler       = scheduler,
            device          = str(device),
        )
        start_epoch   = ckpt["epoch"] + 1      # nächste Epoch
        best_val_loss = ckpt["best_val_loss"]
        print(f"  → Starte ab Epoch {start_epoch}, "
              f"bester Val-Loss bisher: {best_val_loss:.6f}\n")

    # -------------------------------------------------------------------------
    # 8. TRAINING LOOP
    # -------------------------------------------------------------------------
    checkpoint_dir = ckpt_cfg.get("dir", "checkpoints")
    save_every     = ckpt_cfg.get("save_every", 10)
    log_every      = train_cfg.get("log_every", 5)
    use_cosine     = train_cfg.get("use_cosine", True)
    n_epochs       = train_cfg["epochs"]
    patience       = train_cfg["patience"]

    print(f"Starte Training: {n_epochs} Epochs, Patience={patience}")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'Zeit':>8}")
    print("-" * 52)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # --- Trainings-Epoch ---
        train_loss, train_mse, train_cos = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_cosine
        )

        # --- Validation ---
        val_loss, val_mse, val_cos = validate(
            model, val_loader, device, use_cosine
        )

        # --- Scheduler: LR für nächste Epoch anpassen ---
        # WICHTIG: scheduler.step() NACH validate(), nicht davor!
        # (Häufiger Fehler: step() vor erstem validate() → erste LR falsch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # --- Konsolen-Ausgabe ---
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
              f"dist={val_dist:>8.6f}  {current_lr:>10.2e}  {elapsed:>6.1f}s")

        # --- wandb Logging ---
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch":       epoch,
                "train/loss":  train_loss,
                "train/mse":   train_mse,
                "train/cos":   train_cos,
                "train/dist":  train_dist,   # Channel-Statistik — Hauptindikator
                "val/loss":    val_loss,
                "val/mse":     val_mse,
                "val/cos":     val_cos,
                "val/dist":    val_dist,     # → 0 = Decoder-kompatibel
                "lr":          current_lr,
            })

        # --- Visualisierungen alle log_every Epochs ---
        if epoch % log_every == 0:
            visualize_predictions(model, val_loader, device, epoch)

        # --- Checkpoint: bestes Modell ---
        # Nur speichern wenn Val-Loss besser als alles bisherige
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0

            save_checkpoint(
                model          = model,
                optimizer      = optimizer,
                scheduler      = scheduler,
                epoch          = epoch,
                val_loss       = val_loss,
                best_val_loss  = best_val_loss,
                config         = model_config,
                checkpoint_dir = checkpoint_dir,
                filename       = "best_val_loss.pt",
            )
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"]    = epoch
        else:
            patience_cnt += 1

        # --- Checkpoint: periodischer Snapshot ---
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model          = model,
                optimizer      = optimizer,
                scheduler      = scheduler,
                epoch          = epoch,
                val_loss       = val_loss,
                best_val_loss  = best_val_loss,
                config         = model_config,
                checkpoint_dir = checkpoint_dir,
                filename       = f"epoch_{epoch+1:03d}.pt",
            )

        # --- Early Stopping ---
        if patience_cnt >= patience:
            print(f"\nEarly Stopping: {patience_cnt} Epochs ohne Val-Verbesserung.")
            break

    # -------------------------------------------------------------------------
    # 9. Abschluss
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training abgeschlossen.")
    print(f"  Bester Val-Loss: {best_val_loss:.6f}")
    best_ckpt = find_best_checkpoint(checkpoint_dir, model_config.phase)
    if best_ckpt:
        print(f"  Bester Checkpoint: {best_ckpt}")
    print(f"{'='*60}\n")

    if WANDB_AVAILABLE:
        wandb.finish()


# =============================================================================
# Einstiegspunkt
# =============================================================================

if __name__ == "__main__":
    main()
