"""
train.py — BEV World Model: Hauptskript Training
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

# SSIM Loss — optional, fällt auf MSE zurück wenn nicht installiert
# Installation: pip install pytorch-msssim
try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("[Info] pytorch-msssim nicht installiert. SSIM Loss deaktiviert.")
    print("       Installation: pip install pytorch-msssim")

# --- code/ dem Python-Suchpfad hinzufügen ---
# __file__ = .../bev_worldmodel/train.py
# Path(__file__).parent = .../bev_worldmodel/
# / "code" = .../bev_worldmodel/code/
CODE_DIR = Path(__file__).parent / "Code"
sys.path.insert(0, str(CODE_DIR))

# Jetzt können wir alle Subskripte direkt importieren
from config          import ModelConfig
from bev_world_model import BEVWorldModel
# bev_dataloader wird lazy in build_dataloaders() importiert (smoke_test-kompatibel)
from checkpointing   import save_checkpoint, load_checkpoint, find_best_checkpoint

# Seg-Decoder (pure PyTorch, kein mmdet3d)
# seg_decoder_torch.py liegt in Code/ neben den anderen Modulen.
# Schlägt der Import fehl (z.B. bevfusion-seg.pth nicht erreichbar), läuft
# das Training trotzdem weiter — nur validate_seg() wird übersprungen.
try:
    from seg_decoder_torch import build_seg_decoder, compute_iou, MAP_CLASSES, MAP_SCORE
    SEG_DECODER_AVAILABLE = True
except ImportError as _seg_err:
    SEG_DECODER_AVAILABLE = False
    print(f"[Warnung] seg_decoder_torch nicht importierbar: {_seg_err}")
    print("          validate_seg() wird übersprungen (nur Val-Loss aktiv).")

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
    # benchmark=True laesst cuDNN pro Conv-Shape den schnellsten
    # Algorithmus autotunen. Unsere Shapes sind konstant -> einmalige Suche im
    # ersten Step, danach reiner Gewinn. Braucht deterministic=False; die
    # Bit-Reproduzierbarkeit DESSELBEN Seeds entfaellt damit. Die Rauschboden-Schwellen
    # basieren auf Seed-ZU-Seed-Varianz und bleiben davon unberuehrt.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


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
    # --- alle Gewichte explizit, vollständig YAML-steuerbar ---
    lambda_mse:   float = 1.0,   # Multi-Scale MSE (war implizit 1.0)
    lambda_cos:   float = 0.1,   # Cosine-Similarity (Null-Kollaps-Schutz)
    lambda_mean:  float = 0.1,   # Channel-Mittelwert-Term  (NEU: getrennt von std)
    lambda_std:   float = 0.1,   # Channel-Std-Term         (NEU: eigener Regler)
    lambda_grad:  float = 0.0,   # Gradient-Term / Anti-Blur (NEU, Default AUS)
    lambda_ssim:  float = 0.1,   # SSIM-Term (abschaltbar via 0.0)
    recon_loss:   str   = "mse", # TERM-1-Distanz — "mse"|"l1"|"smooth_l1"
    cell_weight:  torch.Tensor = None,  # [B,128,128] Zellgewichte fuer
                                        # TERM 1 (None = alter Pfad, bit-identisch)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kombinierter Loss — vollständig per YAML konfigurierbar.

    Jeder Term hat ein eigenes lambda; lambda=0 schaltet den Term
    rechenaufwandsfrei ab (kein Gradient, keine Allokation).

    ┌─────────────────────────────────────────────────────────────────┐
    │ TERM 1: Multi-Scale MSE  (lambda_mse, Default 1.0)              │
    │   (MSE_128 + MSE_64 + MSE_32) / 3 — grobe + mittlere + feine   │
    │   Struktur gleichzeitig optimiert.                              │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 2: Cosine Similarity  (lambda_cos, Default 0.1)            │
    │   1 - CosSim(pred.flatten, target.flatten)                      │
    │   Verhindert Null-Kollaps bei sparsem Input (mean≈0.14).        │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 3a: Channel-Mittelwert  (lambda_mean, Default 0.1)         │
    │   MSE(pred_mean_per_channel, target_mean_per_channel)           │
    │   Stellt statistische Decoder-Kompatibilität sicher.            │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 3b: Channel-Std  (lambda_std, Default 0.1)                 │
    │   MSE(pred_std_per_channel, target_std_per_channel)             │
    │   Primärer Regler gegen Regression-to-Mean / std-Gap.           │
    │   KERN der Sweep-Studie.                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 4: Räumlicher Gradient  (lambda_grad, Default 0.0 = AUS)   │
    │   MSE(∂pred/∂x, ∂target/∂x) + MSE(∂pred/∂y, ∂target/∂y)       │
    │   Anti-Blur: bestraft fehlende Hochfrequenz / räumliche Kanten. │
    │   Finite Differenzen in x und y — kein Mehraufwand wenn 0.      │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 5: SSIM  (lambda_ssim, Default 0.1, abschaltbar via 0.0)   │
    │   1 - SSIM(channel-gemittelt)                                   │
    │   Strukturähnlichkeit: Kanten, Kontraste, lokale Muster.        │
    └─────────────────────────────────────────────────────────────────┘

    Returns:
        (loss_total, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim)
        — alle 7 Terme separat für wandb-Logging und Korrektheits-Check.

    Baseline-Äquivalenz:
        lambda_mse=1.0, lambda_cos=0.1, lambda_mean=0.1, lambda_std=0.1,
        lambda_grad=0.0, lambda_ssim=0.1
        → loss_total ≈ alter Loss (< 1e-6 Abweichung durch
          Float32-Reassoziation; kein semantischer Unterschied).
    """
    # ------------------------------------------------------------------
    # TERM 1: Multi-Scale Rekonstruktion (Distanz waehlbar)
    #   recon_loss steuert NUR TERM 1: "mse" (Default, rueckwaerts-kompatibel),
    #   "l1" (Betrag) oder "smooth_l1" (Huber, DINO-Foresight-naeh). Der
    #   Return-Name bleibt loss_mse (Logging-/Tuple-Kompatibilitaet).
    # ------------------------------------------------------------------
    if recon_loss == "mse":
        _recon = F.mse_loss
    elif recon_loss == "l1":
        _recon = F.l1_loss
    elif recon_loss == "smooth_l1":
        _recon = F.smooth_l1_loss
    else:
        raise ValueError(
            f"recon_loss muss 'mse'|'l1'|'smooth_l1' sein, war '{recon_loss}'.")
    if cell_weight is None:
        # Alter Pfad — UNVERAENDERT (bit-identisch zu allen historischen Laeufen).
        loss_128 = _recon(pred, target)
        loss_64  = _recon(
            F.avg_pool2d(pred,   kernel_size=2),
            F.avg_pool2d(target, kernel_size=2),
        )
        loss_32  = _recon(
            F.avg_pool2d(pred,   kernel_size=4),
            F.avg_pool2d(target, kernel_size=4),
        )
    else:
        # zellgewichteter TERM 1. Pro Skala: Elementloss ueber Kanaele
        # mitteln -> [B,H,W], mit (mitskalierten) Gewichten mitteln und durch
        # das Gewichtsmittel normieren — bei w==1 exakt der ungewichtete Wert.
        def _wrecon(p, t, w):
            el = _recon(p, t, reduction="none").mean(dim=1)   # [B,H,W]
            return (el * w).mean() / w.mean()
        w128 = cell_weight
        w64  = F.avg_pool2d(w128.unsqueeze(1), kernel_size=2).squeeze(1)
        w32  = F.avg_pool2d(w128.unsqueeze(1), kernel_size=4).squeeze(1)
        loss_128 = _wrecon(pred, target, w128)
        loss_64  = _wrecon(F.avg_pool2d(pred, 2), F.avg_pool2d(target, 2), w64)
        loss_32  = _wrecon(F.avg_pool2d(pred, 4), F.avg_pool2d(target, 4), w32)
    loss_mse = (loss_128 + loss_64 + loss_32) / 3.0

    # ------------------------------------------------------------------
    # TERM 2: Cosine Similarity
    # ------------------------------------------------------------------
    if lambda_cos > 0:
        loss_cos = 1.0 - F.cosine_similarity(
            pred.flatten(1),
            target.flatten(1),
            dim=1,
        ).mean()
    else:
        loss_cos = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # TERM 3a+b: Channel-Statistik (aufgespalten für Sweep)
    # ------------------------------------------------------------------
    # mean/std über H und W pro Channel: [B, 256]
    pred_ch_mean   = pred.mean(dim=[-2, -1])
    target_ch_mean = target.mean(dim=[-2, -1])
    pred_ch_std    = pred.std(dim=[-2, -1])
    target_ch_std  = target.std(dim=[-2, -1])

    # Getrennte Terme → getrennte Regler (std, optional mean)
    loss_mean = F.mse_loss(pred_ch_mean, target_ch_mean)
    loss_std  = F.mse_loss(pred_ch_std,  target_ch_std)

    # ------------------------------------------------------------------
    # TERM 4: Räumlicher Gradient (Anti-Blur, Default AUS)
    # ------------------------------------------------------------------
    # Finite Differenzen: pred_dx [B,256,H,W-1], pred_dy [B,256,H-1,W]
    # Nur berechnen wenn lambda_grad > 0 → kein Overhead im Default-Fall
    if lambda_grad > 0:
        pred_dx   = pred[..., :, 1:]   - pred[..., :, :-1]    # ∂/∂x
        pred_dy   = pred[..., 1:, :]   - pred[..., :-1, :]    # ∂/∂y
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        loss_grad = (F.mse_loss(pred_dx, target_dx) +
                     F.mse_loss(pred_dy, target_dy))
    else:
        loss_grad = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # TERM 5: SSIM (abschaltbar via lambda_ssim=0)
    # ------------------------------------------------------------------
    if SSIM_AVAILABLE and lambda_ssim > 0:
        pred_2d    = pred.mean(dim=1, keepdim=True).float()
        target_2d  = target.mean(dim=1, keepdim=True).float()
        data_range = float(target_2d.max() - target_2d.min() + 1e-6)
        loss_ssim  = (1.0 - ssim_fn(pred_2d, target_2d,
                                     data_range=data_range,
                                     size_average=True))
    else:
        loss_ssim = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # Kombination — vollständig YAML-gesteuert
    # ------------------------------------------------------------------
    loss_total = (lambda_mse  * loss_mse
                + lambda_cos  * loss_cos
                + lambda_mean * loss_mean
                + lambda_std  * loss_std
                + lambda_grad * loss_grad
                + lambda_ssim * loss_ssim)

    return loss_total, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim


# =============================================================================
# BAUSTEIN 4 — Eine Trainings-Epoch
# =============================================================================

def train_one_epoch(
    model:       torch.nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler:      torch.cuda.amp.GradScaler,
    device:      torch.device,
    lambda_mse:   float = 1.0,
    lambda_cos:   float = 0.1,
    lambda_mean:  float = 0.1,
    lambda_std:   float = 0.1,
    lambda_grad:  float = 0.0,
    lambda_ssim:  float = 0.1,
    recon_loss:   str   = "mse",   # an compute_loss durchgereicht
    ema_state:    Optional[dict] = None,   # Shadow-Weights fuer EMA
    ema_decay:    float = 0.9999,          # None -> EMA-Zweig komplett inaktiv
    seg_decoder=None,                      # eingefrorener Decoder fuer Task-Loss
    lambda_task:  float = 0.0,             # 16f: Gewicht BCE(decode(pred), decode(real))
    task_every:   int   = 1,               # Task-Loss nur auf jedem n-ten Batch
    vae_active:   bool  = False,           # 18/B1: CVAE-Pfad (Posterior-Sampling) an
    vae_beta:     float = 0.0,             # 18/B1: annealtes KL-Gewicht dieser Epoche
    kl_free_bits: float = 0.05,            # 18/B1: Mindest-KL/Dim ohne Strafe
    flow_active:  bool  = False,           # 18/B2: FM-Residual-Training (Backbone frozen)
) -> Tuple[float, float, float, float, float, float, float, dict]:
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

    TIMING-INSTRUMENTIERUNG:
        Pro Batch wird data_time (Warten auf den naechsten Batch vom Loader,
        d.h. Zeit zwischen Ende des letzten Compute-Schritts und Erhalt des
        naechsten Batches) und compute_time (Forward+Backward+Optimizer-Step)
        per time.perf_counter() gemessen und ueber die Epoche aufsummiert.
        data_time_ratio = data_time / (data_time + compute_time):
            hoch  -> I/O ist weiterhin der Engpass (Loader kommt nicht hinterher)
            niedrig -> GPU ist der Engpass, Datenpipeline reicht aus
        Macht das Entscheidungs-Gate MESSBAR statt per nvidia-smi
        geschaetzt.

    Returns:
        (avg_total_loss, avg_mse_loss, avg_cos_loss, avg_mean_loss,
         avg_std_loss, avg_grad_loss, avg_ssim_loss, timing) — Durchschnitte
        ueber die Epoche; timing ist ein dict mit data_time/compute_time/
        data_time_ratio (Sekunden bzw. Anteil 0..1).
    """
    model.train()   # Dropout AN, BatchNorm im Trainings-Modus

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_mean = 0.0
    total_std  = 0.0
    total_grad = 0.0
    total_ssim = 0.0
    total_task = 0.0     # 16f: Decoder-Task-Loss (nur ueber Task-Batches gemittelt)
    n_task     = 0
    total_kl   = 0.0     # 18/B1: roher KL (vor free-bits/beta), Collapse-Diagnose
    n_kl       = 0
    n_batches  = len(loader)

    # tee-fester Fortschritt: schreibt direkt auf stdout mit \r + flush.
    # Anders als tqdm (das durch eine Pipe puffert) ist das auch in der
    # tee-Logdatei / im tmux-Scrollback live sichtbar.
    _t_epoch = time.time()

    # Timing-Instrumentierung: data_time = Wartezeit auf den Loader (Zeit
    # zwischen Ende des vorigen Compute-Schritts und Erhalt dieses Batches),
    # compute_time = alles danach bis inkl. optimizer-Step.
    total_data_time    = 0.0
    total_compute_time = 0.0
    _t_mark = time.perf_counter()   # Start: vor dem ersten Batch-Warten

    for _bi, batch in enumerate(loader):
        _t_got_batch = time.perf_counter()
        total_data_time += _t_got_batch - _t_mark

        # Daten auf GPU (oder CPU wenn kein GPU)
        # non_blocking=True: Transfer läuft asynchron, CPU kann weiterarbeiten
        # 1d: der Loader liefert fp16 (halbes h2d-Volumen); Upcast auf fp32
        # erst HIER auf der GPU (.float() nach .to), vor dem Modell/AvgPool.
        inputs = batch["input"].to(device, non_blocking=True).float()   # [B, 3, 256, 128, 128]
        target = batch["target"].to(device, non_blocking=True).float()  # [B, 256, 128, 128]

        # Gradienten vom letzten Schritt löschen
        # set_to_none=True ist schneller als zero_grad() weil keine Zuweisung auf 0
        optimizer.zero_grad(set_to_none=True)

        _ego = batch["ego_delta"].to(device).float() if "ego_delta" in batch else None
        # Zellgewichte (nur vorhanden, wenn data.gt_masks_train_path gesetzt)
        _cw = batch["cell_weight"].to(device, non_blocking=True).float() \
            if "cell_weight" in batch else None

        # --- 18/B2: Flow-Matching-Zweig (ersetzt den Regressions-Loss) -------
        # Backbone frozen liefert x_det; gelernt wird nur v_theta auf dem
        # Residuum r = target - x_det (Mathe: TASK18_METHODIK_VAE_DIFFUSION.md).
        if flow_active:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    x_det = model(inputs, _ego)
                r    = target - x_det
                eps  = torch.randn_like(r)
                tau  = torch.rand(r.shape[0], device=device)
                r_tau = (1.0 - tau[:, None, None, None]) * eps + tau[:, None, None, None] * r
                v    = model.flow(r_tau, tau, x_det, inputs[:, -1])
                loss = F.mse_loss(v.float(), (r - eps).float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _t_mark = time.perf_counter()
            total_compute_time += _t_mark - _t_got_batch
            if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
                elapsed = time.time() - _t_epoch
                it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
                sys.stdout.write(
                    f"\r  Train {100.0*(_bi+1)/n_batches:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                    f"fm_loss={total_loss/(_bi+1):.4f}  {it_s:4.1f} it/s  ETA {eta/60:4.1f}min   ")
                sys.stdout.flush()
            continue

        # --- Forward Pass in float16 ---
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs, _ego,
                         target=target if vae_active else None)  # [B, 256, 128, 128]
            loss, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim = compute_loss(
                pred, target,
                lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
                recon_loss=recon_loss,
                cell_weight=_cw,   # None = alter Pfad
            )
            # --- KL-Term (18/B1) -----------------------------------------
            # KL(q||p) zweier diagonaler Gauss, pro Dim; free-bits clampen
            # (unterhalb der Schwelle kein Gradient -> Anti-Collapse), dann
            # mit dem annealten beta gewichtet. Roh-KL separat geloggt.
            if vae_active:
                mu_q, lv_q, mu_p, lv_p = [t.float() for t in model.last_vae_stats]
                kl_dim = 0.5 * (lv_p - lv_q
                                + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p)
                                - 1.0)                                  # [B, Z]
                kl_raw = kl_dim.sum(dim=1).mean()
                kl_pen = kl_dim.clamp(min=kl_free_bits).sum(dim=1).mean()
                loss = loss + vae_beta * kl_pen
                total_kl += kl_raw.item()
                n_kl     += 1

        # --- Decoder-Task-Loss  ------------------------------------
        # Richtet das Training an der Metrik aus (LAW/DriveFuture-Muster):
        # BCE zwischen decode(pred) und decode(real) — der Decoder ist Multi-
        # Label-Sigmoid, daher BCE statt CE. Decoder bleibt frozen/eval; die
        # Gradienten fliessen NUR durch ihn hindurch zum pred-Latent. Bewusst
        # ausserhalb autocast (Decoder ist float32-kalibriert, wie validate_seg).
        _task_active = (seg_decoder is not None and lambda_task > 0.0
                        and (_bi % task_every == 0))
        if _task_active:
            with torch.no_grad():
                _real_logits = seg_decoder.forward_logits(target)
                _real_mask   = (torch.sigmoid(_real_logits) >= MAP_SCORE).float()
            _pred_logits = seg_decoder.forward_logits_grad(pred.float())
            loss_task = F.binary_cross_entropy_with_logits(_pred_logits, _real_mask)
            loss = loss + lambda_task * loss_task
            total_task += loss_task.item()
            n_task     += 1

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

        # --- EMA-Update -----------------------------------------
        # Nach dem Optimizer-Step: Shadow-Weights folgen dem gerade
        # aktualisierten Modell (ema = decay*ema + (1-decay)*param). Nur aktiv
        # wenn ema_state uebergeben wurde -> sonst bit-identisch zum Altpfad.
        if ema_state is not None:
            with torch.no_grad():
                for _n, _p in model.named_parameters():
                    ema_state[_n].mul_(ema_decay).add_(_p.detach(), alpha=1.0 - ema_decay)

        # Losses akkumulieren (.item() löst den Tensor vom Berechnungsgraph)
        total_loss += loss.item()
        total_mse  += loss_mse.item()
        total_cos  += loss_cos.item()
        total_mean += loss_mean.item()
        total_std  += loss_std.item()
        total_grad += loss_grad.item()
        total_ssim += loss_ssim.item()

        _t_mark = time.perf_counter()
        total_compute_time += _t_mark - _t_got_batch

        # --- Fortschritt alle 20 Batches (tee-fest) ---
        if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
            elapsed = time.time() - _t_epoch
            it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
            eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
            pct     = 100.0 * (_bi + 1) / n_batches
            sys.stdout.write(
                f"\r  Train {pct:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                f"loss={total_loss/(_bi+1):.4f}  "
                f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
            )
            sys.stdout.flush()

    # Fortschrittszeile loeschen -> nur die Epochen-Uebersicht bleibt am Ende
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    _denom = total_data_time + total_compute_time
    timing = {
        "data_time":       total_data_time,
        "compute_time":    total_compute_time,
        "data_time_ratio": (total_data_time / _denom) if _denom > 0 else 0.0,
        # mittlerer Task-Loss (im timing-dict, um die Rueckgabe-Aritaet
        # stabil zu halten); None wenn Task-Loss inaktiv.
        "task_loss":       (total_task / n_task) if n_task > 0 else None,
        # 18/B1: roher KL (Collapse-Diagnose); None wenn CVAE inaktiv.
        "kl_loss":         (total_kl / n_kl) if n_kl > 0 else None,
    }

    return (total_loss / n_batches, total_mse  / n_batches,
            total_cos  / n_batches, total_mean / n_batches,
            total_std  / n_batches, total_grad / n_batches,
            total_ssim / n_batches, timing)


# =============================================================================
# BAUSTEIN 5 — Validation
# =============================================================================

def validate(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    lambda_mse:   float = 1.0,
    lambda_cos:   float = 0.1,
    lambda_mean:  float = 0.1,
    lambda_std:   float = 0.1,
    lambda_grad:  float = 0.0,
    lambda_ssim:  float = 0.1,
    recon_loss:   str   = "mse",   # an compute_loss durchgereicht
    flow_active:  bool  = False,   # 18/B2: FM-Val-Loss statt Regressions-Loss
) -> Tuple[float, float, float, float, float, float, float]:
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
        (avg_total_loss, avg_mse_loss, avg_cos_loss, avg_dist_loss, avg_ssim_loss)
    """
    model.eval()

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_mean = 0.0
    total_std  = 0.0
    total_grad = 0.0
    total_ssim = 0.0
    n_batches  = len(loader)

    _t_val = time.time()

    with torch.no_grad():
        for _bi, batch in enumerate(loader):
            inputs = batch["input"].to(device, non_blocking=True).float()   # 1d: GPU-Upcast
            target = batch["target"].to(device, non_blocking=True).float()

            _ego = batch["ego_delta"].to(device).float() if "ego_delta" in batch else None
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if flow_active:
                    # 18/B2: deterministischer FM-Val-Loss — tau/eps aus einem
                    # FIX geseedeten Generator, damit die Val-Loss-Patience auf
                    # einer rauscharmen, vergleichbaren Groesse laeuft.
                    _g = torch.Generator(device=device.type); _g.manual_seed(1234 + _bi)
                    x_det = model(inputs, _ego)
                    r    = target - x_det
                    eps  = torch.empty_like(r).normal_(generator=_g)
                    tau  = torch.empty(r.shape[0], device=device).uniform_(generator=_g)
                    r_tau = (1.0 - tau[:, None, None, None]) * eps + tau[:, None, None, None] * r
                    v    = model.flow(r_tau, tau, x_det, inputs[:, -1])
                    loss = F.mse_loss(v.float(), (r - eps).float())
                    z = torch.zeros((), device=device)
                    loss_mse = loss_cos = loss_mean = loss_std = loss_grad = loss_ssim = z
                else:
                    pred = model(inputs, _ego)
                    loss, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim = compute_loss(
                        pred, target,
                        lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
                        recon_loss=recon_loss,
                    )

            total_loss += loss.item()
            total_mse  += loss_mse.item()
            total_cos  += loss_cos.item()
            total_mean += loss_mean.item()
            total_std  += loss_std.item()
            total_grad += loss_grad.item()
            total_ssim += loss_ssim.item()

            # --- Fortschritt alle 20 Batches (tee-fest, wird am Ende geloescht) ---
            if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
                elapsed = time.time() - _t_val
                it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
                pct     = 100.0 * (_bi + 1) / n_batches
                sys.stdout.write(
                    f"\r  Val   {pct:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                    f"loss={total_loss/(_bi+1):.4f}  "
                    f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
                )
                sys.stdout.flush()

    # Fortschrittszeile loeschen
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    return (total_loss / n_batches, total_mse / n_batches,
            total_cos  / n_batches, total_mean / n_batches,
            total_std  / n_batches, total_grad / n_batches,
            total_ssim / n_batches)


# =============================================================================
# BAUSTEIN 5b — Decode-Validierung
# =============================================================================

def validate_seg(
    model:        torch.nn.Module,
    seg_decoder,                        # SegDecoder-Instanz, eingefroren
    subset_inputs:  torch.Tensor,       # [N, 3, 256, 128, 128] val-Subset
    subset_targets: torch.Tensor,       # [N, 256, 128, 128]    zugehörige reals
    real_masks_cache: list,             # vorberechnete real-Masken (list of [6,200,200])
    device:       torch.device,
    flow_steps:   int = 0,              # 18/B2: >0 -> pred = flow_sample(steps),
                                        # fix geseedet (Selektion auf Sampling-Qualitaet)
) -> Tuple[float, dict]:
    """
    Decodiert val-Subset-Predictions durch den eingefrorenen BEVFusion-Seg-
    Decoder und gibt mean mIoU + per-Klasse zurück.

    ABLAUF pro Sample:
      1. model(input) -> pred-Latent [256,128,128]              (World-Model)
      2. seg_decoder.latent_to_mask(pred) -> pred_mask [6,200,200] (BEVFusion)
      3. compute_iou(pred_mask, real_mask_cached)
      -> mIoU über alle Subset-Samples gemittelt

    OPTIMIERUNG: Die real-Masken hängen nur vom festen Target-Latent und dem
    eingefrorenen Decoder ab — sie ändern sich über Epochen nie. Sie werden
    einmalig in build_seg_val_cache() vorberechnet und hier nur gelesen.
    Damit braucht validate_seg() pro Epoche nur N forward-Passes durch das
    World-Model + N Decoder-Passes für die preds (statt 2N).

    MODULARITAET:
    Für PNG-Ausgabe (pred/real/comparison) kann die Visualisierung hier nach
    compute_iou() einfach render_mask() + save_comparison() aufrufen —
    die Funktionen sind in seg_decoder_torch.py bereit, werden hier
    bewusst NICHT aufgerufen.

    Returns:
        (mean_miou, per_class_dict)
    """
    if not SEG_DECODER_AVAILABLE or seg_decoder is None:
        return 0.0, {}

    model.eval()
    all_ious = []
    n_total  = len(subset_inputs)
    _t_dec   = time.time()

    with torch.no_grad():
        # Samples einzeln verarbeiten: Decoder erwartet [1,256,128,128],
        # Batch-Decode möglich aber unnötig (N ist klein, ~300).
        for i in range(n_total):
            inp = subset_inputs[i:i+1].to(device)   # [1,3,256,128,128]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if flow_steps > 0:                   # 18/B2: ein FM-Sample
                    _g = torch.Generator(device=device.type); _g.manual_seed(9000 + i)
                    pred_latent = model.flow_sample(inp, steps=flow_steps, generator=_g)
                else:
                    pred_latent = model(inp)         # [1,256,128,128]

            # Decoder läuft bewusst in float32 (autocast=False in latent_to_mask)
            pred_mask, _ = seg_decoder.latent_to_mask(
                pred_latent[0], device=device)       # [6,200,200] bool

            real_mask = real_masks_cache[i]          # [6,200,200] bool, vorberechnet

            iou = compute_iou(pred_mask, real_mask, MAP_CLASSES)
            all_ious.append(iou)

            # --- Fortschritt alle 20 Samples (tee-fest, wird am Ende geloescht) ---
            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                elapsed = time.time() - _t_dec
                it_s    = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_total - i - 1) / it_s if it_s > 0 else 0.0
                pct     = 100.0 * (i + 1) / n_total
                run_miou = float(np.mean([d['mIoU'] for d in all_ious]))
                sys.stdout.write(
                    f"\r  Decode {pct:5.1f}%  [{i+1:>3}/{n_total}]  "
                    f"mIoU={run_miou:.4f}  "
                    f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
                )
                sys.stdout.flush()

    # Fortschrittszeile loeschen
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Mittelwert über Subset
    mean_miou = float(np.mean([d['mIoU'] for d in all_ious]))
    per_class = {}
    for cls in MAP_CLASSES:
        vals = [d[cls] for d in all_ious if d.get(cls) is not None]
        per_class[cls] = round(float(np.mean(vals)), 4) if vals else None

    return mean_miou, per_class


def build_seg_val_cache(
    seg_decoder,
    val_loader:     torch.utils.data.DataLoader,
    device:         torch.device,
    n_subset:       int   = 300,
    seed:           int   = 42,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Waehlt ein fixes, reproduzierbares Subset aus dem Val-Dataset und
    berechnet die real-Masken einmalig vor.

    WICHTIG: Greift direkt auf val_loader.dataset per Index zu (kein
    Loader-Iteration) -- vermeidet Worker-Deadlocks beim break aus einem
    laufenden DataLoader mit num_workers > 0.

    Returns:
        subset_inputs   [N, 3, 256, 128, 128]  CPU-Tensor
        subset_targets  [N, 256, 128, 128]      CPU-Tensor
        real_masks      list of N bool [6,200,200]
    """
    if not SEG_DECODER_AVAILABLE or seg_decoder is None:
        return None, None, []

    print(f"[validate_seg] Baue Val-Subset (n={n_subset}, seed={seed})...")
    rng     = np.random.default_rng(seed)
    dataset = val_loader.dataset
    n_avail = len(dataset)
    n_take  = min(n_subset, n_avail)

    # Reproduzierbare Index-Auswahl direkt auf dem Dataset
    idx = sorted(rng.choice(n_avail, size=n_take, replace=False).tolist())

    # Direkte __getitem__-Zugriffe -- kein DataLoader, kein Worker
    inputs_list  = []
    targets_list = []
    print(f"[validate_seg] Lade {n_take} Samples direkt aus Dataset...")
    for i in idx:
        sample = dataset[i]
        inputs_list.append(sample["input"])
        targets_list.append(sample["target"])

    # 1d: der Loader/Dataset liefert jetzt fp16. Der Seg-Decoder (real-Masken
    # unten, decode_validation) laeuft aber bewusst in float32 und erzeugt die
    # mIoU-Referenz -> hier EINMALIG auf fp32 upcasten, damit der Decode-/Metrik-
    # Pfad bit-identisch zu vorher bleibt (kein fp16 in die mIoU).
    subset_inputs  = torch.stack(inputs_list,  dim=0).cpu().float()
    subset_targets = torch.stack(targets_list, dim=0).cpu().float()

    # Real-Masken einmalig dekodieren (aendert sich nie)
    print(f"[validate_seg] Dekodiere {n_take} real-Masken (einmalig)...")
    real_masks = []
    with torch.no_grad():
        for i in range(n_take):
            mask, _ = seg_decoder.latent_to_mask(
                subset_targets[i], device=device)
            real_masks.append(mask)

    print(f"[validate_seg] Cache fertig: {n_take} Samples, "
          f"{len(real_masks)} real-Masken vorberechnet.")
    return subset_inputs, subset_targets, real_masks


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
    inputs = batch["input"][:n_samples].to(device).float()   # 1d: GPU-Upcast
    target = batch["target"][:n_samples].to(device).float()
    _ego = batch["ego_delta"][:n_samples].to(device).float() if "ego_delta" in batch else None

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs, _ego)

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

    Für "explicit" mit HDF5-Cache (voller Datensatz):

        data:
          mode: "explicit"
          train_sources:
            - pkl_path:   "/data/nuscenes/nuscenes_infos_train.pkl"
              latent_dir: "/data/nuscenes/latents/seg/train"
          val_sources:
            - pkl_path:   "/data/nuscenes/nuscenes_infos_val.pkl"
              latent_dir: "/data/nuscenes/latents/seg/val"
          train_h5_path: "/data/Cache/bev_latents_train.h5"
          val_h5_path:   "/data/Cache/bev_latents_val.h5"

    Diese Funktion übersetzt das in den passenden Funktionsaufruf.
    """
    # Lazy Import: erst hier importieren damit smoke_test.py train.py
    # importieren kann ohne dass bev_dataloader sofort ausgeführt wird
    from bev_dataloader import make_dataloaders_split, make_dataloaders_explicit

    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    mode       = data_cfg.get("mode", "split")

    # --- Memmap-Packing -- integrierter, abschaltbarer Teil ---
    # data.use_packed steuert ALLES ueber einen Schalter:
    #   true       -> fehlende Packs beim Start erzeugen (idempotent, Skip wenn
    #                 vorhanden) UND zur Laufzeit aus dem Memmap lesen
    #   false/weg  -> reines .npy wie bisher; packed_dir wird ignoriert
    # Kein separates Skript noetig -- Packen ist Teil des train-Zyklus.
    use_packed   = data_cfg.get("use_packed", False)
    packed_dtype = data_cfg.get("packed_dtype", "float32")
    if use_packed:
        # packed_dir-Pflicht pruefen -> kein stiller Teil-Packed-Zustand
        if mode == "explicit":
            missing = [k for k in ("train_packed_dir", "val_packed_dir")
                       if not data_cfg.get(k)]
        else:
            missing = [] if data_cfg.get("packed_dir") else ["packed_dir"]
        if missing:
            raise ValueError(
                f"data.use_packed=true, aber {missing} fehlt in der Config. "
                f"Entweder packed_dir setzen oder use_packed auf false."
            )
        import pack_latents
        print(f"[build_dataloaders] use_packed=true (dtype={packed_dtype}) -> "
              f"Packs sicherstellen (idempotent, Skip wenn vorhanden)...", flush=True)
        pack_latents.ensure_packed(cfg, dtype_str=packed_dtype)

    # --- Node-lokales /dev/shm-Staging -- optionaler Schalter ---
    # data.stage_to_shm: true -> die GEPACKTEN Dateien werden einmalig nach
    # /dev/shm kopiert und die *_packed_dir-Keys der Config zeigen danach
    # dorthin (RAM-Speed statt BeeGFS-Netzwerk). Muss NACH ensure_packed
    # laufen (staged das fertige packed.npy). Cleanup am Job-Ende via
    # cleanup_shm_staging() (siehe run_training) bzw. sbatch-trap.
    if data_cfg.get("stage_to_shm", False):
        import shm_staging
        shm_staging.stage_and_rewrite(cfg)   # biegt data_cfg[*_packed_dir] IN-PLACE um

    if mode == "split":
        # sources aus YAML: Liste von {pkl_path, latent_dir} Dicts
        # → umwandeln in Liste von Tupeln (wie bev_dataloader erwartet)
        sources = [
            (s["pkl_path"], s["latent_dir"])
            for s in data_cfg["sources"]
        ]

        # HDF5-Modus: optionaler h5_path in config.yaml
        # Wenn gesetzt → BEVDatasetConfig.h5_path wird übergeben
        # (latent_dir bleibt für Szenen-Erkennung, h5_path für Datenladen)
        h5_path = data_cfg.get("h5_path", None)

        # Memmap-Pack-Modus: nur nutzen wenn use_packed=true.
        packed_dir = data_cfg.get("packed_dir", None) if use_packed else None

        loaders = make_dataloaders_split(
            sources       = sources,
            train_ratio   = data_cfg.get("train_ratio", 0.70),
            val_ratio     = data_cfg.get("val_ratio",   0.15),
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
            h5_path       = h5_path,
            packed_dir    = packed_dir,
            prefetch_factor = train_cfg.get("prefetch_factor", 4),
            latent_scale  = data_cfg.get("latent_scale", 1.0),   # Det->Seg-Skala
        )

    elif mode == "explicit":
        def _parse(key):
            return [(s["pkl_path"], s["latent_dir"]) for s in data_cfg.get(key, [])]

        # HDF5-Modus: je Split optional ein eigener h5_path (anders als bei
        # "split" gibt es hier keine EINE gemeinsame Quelle, sondern bereits
        # getrennte Train-/Val-/Test-PKLs → entsprechend auch getrennte Caches.
        loaders = make_dataloaders_explicit(
            train_sources = _parse("train_sources"),
            val_sources   = _parse("val_sources"),
            test_sources  = _parse("test_sources") or None,
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
            train_h5_path = data_cfg.get("train_h5_path", None),
            val_h5_path   = data_cfg.get("val_h5_path",   None),
            test_h5_path  = data_cfg.get("test_h5_path",  None),
            train_packed_dir = data_cfg.get("train_packed_dir", None) if use_packed else None,
            val_packed_dir   = data_cfg.get("val_packed_dir",   None) if use_packed else None,
            latent_scale     = data_cfg.get("latent_scale", 1.0),   # Det->Seg-Skala
            test_packed_dir  = data_cfg.get("test_packed_dir",  None) if use_packed else None,
            prefetch_factor = train_cfg.get("prefetch_factor", 4),
            # Sampler-/Gewichtungs-Optionen (alle Defaults AUS = bit-identisch)
            gt_masks_train_path  = data_cfg.get("gt_masks_train_path", None),
            rare_classes         = tuple(data_cfg.get("rare_classes",
                                         ("stop_line", "ped_crossing", "divider"))),
            rare_weight          = float(data_cfg.get("rare_weight", 1.0)),
            sampler_mode         = data_cfg.get("sampler_mode", "off"),
            sampler_alpha        = float(data_cfg.get("sampler_alpha", 0.5)),
            dynamics_scores_path = data_cfg.get("dynamics_scores_path", None),
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
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Checkpoint, aus dem NUR die Modellgewichte "
                             "geladen werden (Finetune; Optimizer/Epoche frisch)")
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

    # --- 18/B2: Flow-Modus — Backbone EINFRIEREN, nur der Kopf lernt ---------
    # Muss VOR dem Optimizer-Bau passieren (der filtert auf requires_grad).
    flow_active = getattr(model_config, "flow_head", "off") == "flow"
    if flow_active:
        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith("flow."):
                p.requires_grad = False
                n_frozen += 1
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[18/B2] Flow-Modus: Backbone frozen ({n_frozen} Tensoren), "
              f"trainierbar nur flow.* ({n_train:,} Params)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {n_params:,} Parameter  (Phase: {model_config.phase})")
    print(f"Tokens: {model_config.n_tokens} pro Forward-Pass\n")

    # -------------------------------------------------------------------------
    # 4b. Optimizer
    # AdamW = Adam + Weight Decay Korrektur (Loshchilov & Hutter, 2019)
    # Weight Decay wirkt als L2-Regularisierung: drängt Weights Richtung 0
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        # 18/B2: nur trainierbare Params (im Flow-Modus ist das Backbone
        # frozen; sonst identisch zu model.parameters()).
        [p for p in model.parameters() if p.requires_grad],
        lr           = train_cfg["lr"],
        weight_decay = train_cfg["weight_decay"],
        # fused=True buendelt die per-Parameter-Updates in einen
        # CUDA-Kernel (Stufe-0-Profil: ~1000 aten::add_-Aufrufe/Step). Nur auf
        # GPU verfuegbar -> an device gekoppelt; kompatibel mit GradScaler.
        fused        = (device.type == "cuda"),
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
    # 5b. Seg-Decoder + Val-Subset-Cache
    # -------------------------------------------------------------------------
    # decode_cfg steuert die gesamte Decode-Validierung aus der YAML.
    # Fehlt der Block oder ist enabled: false, laeuft das Training wie bisher
    # (nur Val-Loss, kein mIoU). Rueckwaertskompatibel.
    decode_cfg     = cfg.get("decode_validation", {})
    decode_enabled = decode_cfg.get("enabled", False) and SEG_DECODER_AVAILABLE
    seg_decoder    = None
    seg_subset_inputs  = None
    seg_subset_targets = None
    seg_real_masks     = []
    decode_every = decode_cfg.get("decode_every", 5)
    n_subset     = decode_cfg.get("n_subset", 300)

    if decode_enabled:
        seg_ckpt = decode_cfg.get("seg_checkpoint")
        if seg_ckpt and os.path.exists(seg_ckpt):
            print(f"[seg_decoder] Lade Seg-Decoder: {seg_ckpt}")
            try:
                seg_decoder = build_seg_decoder(seg_ckpt, device, verbose=True)
                seg_subset_inputs, seg_subset_targets, seg_real_masks = \
                    build_seg_val_cache(seg_decoder, val_loader, device,
                                        n_subset=n_subset,
                                        seed=train_cfg.get("seed", 42))
                print(f"[seg_decoder] Decode-Validierung aktiv: "
                      f"alle {decode_every} Epochen, {len(seg_real_masks)} Samples\n")
            except Exception as e:
                print(f"[seg_decoder] Decoder-Laden fehlgeschlagen: {e}")
                print("          Falle zurueck auf Val-Loss-only.\n")
                decode_enabled = False
                seg_decoder    = None
        else:
            print(f"[seg_decoder] seg_checkpoint nicht gefunden: {seg_ckpt}")
            print("          Falle zurueck auf Val-Loss-only.\n")
            decode_enabled = False

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
    resume_ckpt   = None    # haelt den Checkpoint fuer den
                             # mIoU-Restore-Block weiter unten (die Ziel-
                             # Variablen existieren erst NACH diesem Block).

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
        resume_ckpt   = ckpt                   # mIoU-Plateau-Resume
        print(f"  → Starte ab Epoch {start_epoch}, "
              f"bester Val-Loss bisher: {best_val_loss:.6f}\n")
    elif args.init_weights:
        # init_weights: NUR Gewichte laden (Finetune) -- Optimizer/Scheduler/Epoche
        # bleiben frisch (kein Resume). Zaehler/Best-Werte starten bei 0/-1,
        # damit der Finetune-Lauf seine eigenen best_*.pt schreibt.
        print(f"Init-Weights (Finetune, kein Resume): {args.init_weights}")
        load_checkpoint(
            checkpoint_path = args.init_weights,
            model           = model,
            device          = str(device),
            # 18/B2: neue Zusatzmodule (flow.*) fehlen im alten Checkpoint
            # und behalten ihre frische Initialisierung.
            allow_missing   = flow_active,
        )

    # -------------------------------------------------------------------------
    # 8. TRAINING LOOP
    # -------------------------------------------------------------------------
    checkpoint_dir = ckpt_cfg.get("dir", "checkpoints")
    save_every     = ckpt_cfg.get("save_every", 10)
    log_every      = train_cfg.get("log_every", 5)
    use_cosine     = train_cfg.get("use_cosine", True)   # Legacy-Key, nur noch für Print/Logging
    lambda_mse     = train_cfg.get("lambda_mse",  1.0)
    lambda_cos     = train_cfg.get("lambda_cos",  0.1)
    lambda_mean    = train_cfg.get("lambda_mean", 0.1)
    lambda_std     = train_cfg.get("lambda_std",  0.1)
    lambda_grad    = train_cfg.get("lambda_grad", 0.0)
    lambda_ssim    = train_cfg.get("lambda_ssim", 0.1)
    recon_loss     = train_cfg.get("recon_loss", "mse")   # TERM-1-Distanz
    lambda_task    = float(train_cfg.get("lambda_task", 0.0))   # 16f: Decoder-Task-Loss
    task_every     = int(train_cfg.get("task_loss_every", 1))   # jeder n-te Batch
    if lambda_task > 0.0 and seg_decoder is None:
        raise SystemExit("[task_loss] lambda_task > 0 erfordert die Decode-Validierung "
                         "(decode_validation.enabled + ladbarer seg_checkpoint) -- "
                         "ohne Decoder kein Task-Loss.")
    if lambda_task > 0.0:
        print(f"[task_loss] Decoder-Task-Loss aktiv: lambda_task={lambda_task}, "
              f"task_loss_every={task_every}")

    # --- 18/B1: CVAE-Konfiguration + Konsistenz-Guards -----------------------
    lambda_kl       = float(train_cfg.get("lambda_kl", 0.0))
    kl_anneal_ep    = int(train_cfg.get("kl_anneal_epochs", 5))
    kl_free_bits    = float(train_cfg.get("kl_free_bits", 0.05))
    vae_active      = getattr(model_config, "vae_mode", "off") == "cvae"
    if vae_active and lambda_kl <= 0.0:
        raise SystemExit("[18/B1] vae_mode=cvae erfordert lambda_kl > 0 "
                         "(sonst unregularisierter Posterior).")
    if not vae_active and lambda_kl > 0.0:
        raise SystemExit("[18/B1] lambda_kl > 0 erfordert model.vae_mode=cvae.")
    if vae_active:
        print(f"[18/B1] CVAE aktiv: z_dim={getattr(model_config, 'vae_z_dim', 32)}, "
              f"lambda_kl={lambda_kl}, anneal={kl_anneal_ep} Ep, "
              f"free_bits={kl_free_bits}/Dim")

    # --- 18/B2: Flow-Guards ---------------------------------------------------
    flow_val_steps = int(train_cfg.get("flow_val_steps", 10))
    if flow_active and vae_active:
        raise SystemExit("[18/B2] flow_head=flow und vae_mode=cvae gleichzeitig "
                         "ist nicht vorgesehen (ein Generativ-Modus pro Lauf).")
    if flow_active and not (args.init_weights or args.resume):
        raise SystemExit("[18/B2] flow_head=flow erfordert --init_weights "
                         "(frozen Backbone vom Headline-Checkpoint) oder --resume.")
    n_epochs       = train_cfg["epochs"]
    patience       = train_cfg["patience"]

    # mIoU-Tracking parallel zum Val-Loss
    best_miou     = -1.0
    miou_patience_cnt = 0
    best_epoch_miou = -1

    # --- EMA: Weight-Averaging. Default AUS -> ema_state bleibt None
    # und Trainings-/Validierungspfad sind bit-identisch zum Altverhalten. Nicht
    # resume-sicher (Shadow-Weights werden nicht im Checkpoint persistiert) --
    # akzeptabel, weil die Headline-Laeufe single-shot laufen. -----------------
    ema_enabled   = bool(train_cfg.get("ema", False))
    ema_decay     = float(train_cfg.get("ema_decay", 0.9999))
    ema_state     = None
    best_miou_ema = -1.0
    if ema_enabled:
        ema_state = {name: p.detach().clone()
                     for name, p in model.named_parameters()}
        print(f"[EMA] aktiv: decay={ema_decay}, {len(ema_state)} Param-Tensoren")
    stop_reason     = "max_epochs"     # wird bei Break ueberschrieben
    # Early-Stopping laeuft weiterhin auf Val-Loss (unveraendert).
    # best_miou.pt wird zusaetzlich gespeichert (Checkpoint fuer die
    # mIoU-Selektion). Beide Checkpoints laufen parallel.

    # --- mIoU-Plateau-Early-Stopping ---. Idee: die letzten ~25 Epochen bis zum mIoU-Optimum bringen
    # oft nur noch marginalen Gewinn -> abschneiden, sobald kein SIGNIFIKANTER
    # mIoU-Fortschritt mehr kommt. "Signifikant" = groesser als der Seed-Rausch-
    # boden (miou_min_delta), damit Rausch-Dips nicht faelschlich killen.
    # Vergleich gegen eine SEPARATE Referenz (miou_stop_ref), NICHT gegen
    # best_miou -- sonst wuerde eine winzige 0.001-Verbesserung den Zaehler
    # zuruecksetzen und nie stoppen. Greift NUR wenn Decode aktiv ist (sonst
    # gibt es kein mIoU -> sauberer Fallback auf reines Val-Loss-Stopping).
    miou_early_stop = decode_enabled and train_cfg.get("miou_early_stop", False)
    miou_patience   = train_cfg.get("miou_patience", 3)      # "dreimal in Folge"
                                                             # (2 war im Test zu scharf
                                                             # bei strenger min_delta)
    miou_min_delta  = train_cfg.get("miou_min_delta", 0.014) # Rauschboden (Seed-Variabilitaet)
    miou_stop_ref   = -1.0   # letzter mIoU-Wert, der signifikant besser war

    # --- mIoU-Plateau-Zustand aus Resume-Checkpoint wiederherstellen ---
    # Ohne dies bliebe best_miou nach Resume bei -1.0 -> die naechste Decode-
    # Epoche wuerde IMMER als "neues Bestes" gelten und best_miou.pt womoeglich
    # mit einem SCHLECHTEREN Wert ueberschreiben. Alte Checkpoints (vor diesem
    # Patch gespeichert) liefern ueber .get() exakt die Defaults von oben ->
    # Resume von einem alten Checkpoint verhaelt sich unveraendert wie bisher.
    if resume_ckpt is not None:
        best_miou         = resume_ckpt.get("best_miou", best_miou)
        best_epoch_miou   = resume_ckpt.get("best_epoch_miou", best_epoch_miou)
        miou_patience_cnt = resume_ckpt.get("miou_patience_cnt", miou_patience_cnt)
        miou_stop_ref      = resume_ckpt.get("miou_stop_ref", miou_stop_ref)
        patience_cnt        = resume_ckpt.get("patience_cnt", patience_cnt)
        print(f"  → mIoU-Zustand wiederhergestellt: best_miou={best_miou:.4f} "
              f"(Epoch {best_epoch_miou}), Plateau={miou_patience_cnt}/{miou_patience}\n")

    print(f"Starte Training: {n_epochs} Epochs, Patience={patience}")
    print(f"  Loss: {lambda_mse}*{recon_loss.upper()} + {lambda_cos}*CosSim + {lambda_mean}*Mean "
          f"+ {lambda_std}*Std + {lambda_grad}*Grad + {lambda_ssim}*SSIM")
    if decode_enabled:
        print(f"  Decode-Validierung: alle {decode_every} Epochen, "
              f"{len(seg_real_masks)} Val-Subset-Samples")
    if miou_early_stop:
        print(f"  mIoU-Plateau-Stopping: aktiv (Patience={miou_patience} Messungen, "
              f"min_delta={miou_min_delta}, zusaetzlich zu Val-Loss-Patience={patience})")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'mIoU':>8}  {'LR':>10}  {'Zeit':>8}  {'data%':>7}")
    print("-" * 78)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # 18/B1: beta-Annealing — linear 0 -> lambda_kl ueber kl_anneal_epochs
        vae_beta = lambda_kl * min(1.0, (epoch + 1) / max(1, kl_anneal_ep)) \
                   if vae_active else 0.0

        # --- Trainings-Epoch ---
        train_loss, train_mse, train_cos, train_mean, train_std, train_grad, train_ssim, train_timing = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
            recon_loss=recon_loss,
            ema_state=ema_state, ema_decay=ema_decay,
            seg_decoder=seg_decoder if lambda_task > 0.0 else None,
            lambda_task=lambda_task, task_every=task_every,
            vae_active=vae_active, vae_beta=vae_beta, kl_free_bits=kl_free_bits,
            flow_active=flow_active,
        )

        # --- Validation (Latent-Raum) ---
        val_loss, val_mse, val_cos, val_mean, val_std, val_grad, val_ssim = validate(
            model, val_loader, device,
            lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
            recon_loss=recon_loss,
            flow_active=flow_active,
        )

        # --- Decode-Validierung (alle decode_every Epochen) ---
        # Erste Epoche immer decodieren fuer einen fruehen Referenzpunkt.
        miou_this_epoch  = None
        per_class_this   = {}
        do_decode = (decode_enabled and seg_decoder is not None and
                     (epoch == start_epoch or (epoch + 1) % decode_every == 0))
        if do_decode:
            miou_this_epoch, per_class_this = validate_seg(
                model, seg_decoder,
                seg_subset_inputs, seg_subset_targets, seg_real_masks,
                device,
                flow_steps=flow_val_steps if flow_active else 0,
            )

        # --- Scheduler ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # --- Konsolen-Ausgabe ---
        miou_str = f"{miou_this_epoch:>8.4f}" if miou_this_epoch is not None else "        "
        _kl_str = (f"  KL={train_timing['kl_loss']:.3f} (b={vae_beta:.4f})"
                   if train_timing.get("kl_loss") is not None else "")
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
              f"{miou_str}  {current_lr:>10.2e}  {elapsed:>6.1f}s  "
              f"data%={train_timing['data_time_ratio']*100:5.1f}{_kl_str}", flush=True)

        # --- wandb Logging ---
        log_dict = {
            "epoch":        epoch,
            "train/loss":   train_loss,
            "train/mse":    train_mse,
            "train/cos":    train_cos,
            "train/mean":   train_mean,
            "train/std":    train_std,
            "train/grad":   train_grad,
            "train/ssim":   train_ssim,
            "val/loss":     val_loss,
            "val/mse":      val_mse,
            "val/cos":      val_cos,
            "val/mean":     val_mean,
            "val/std":      val_std,
            "val/grad":     val_grad,
            "val/ssim":     val_ssim,
            "lr":           current_lr,
            # Timing-Instrumentierung -- macht das
            # I/O-vs-Compute-Entscheidungsgate messbar.
            "train/data_time_s":    train_timing["data_time"],
            "train/compute_time_s": train_timing["compute_time"],
            "train/data_time_ratio": train_timing["data_time_ratio"],
        }
        if train_timing.get("task_loss") is not None:              # Task-Loss
            log_dict["train/task"] = train_timing["task_loss"]
        if train_timing.get("kl_loss") is not None:                # 18/B1
            log_dict["train/kl"]      = train_timing["kl_loss"]
            log_dict["train/kl_beta"] = vae_beta
        if miou_this_epoch is not None:
            log_dict["val/miou"] = miou_this_epoch
            for cls, v in per_class_this.items():
                if v is not None:
                    log_dict[f"val/miou_{cls}"] = v
        if WANDB_AVAILABLE:
            wandb.log(log_dict)

        # --- Visualisierungen alle log_every Epochs ---
        if epoch % log_every == 0:
            visualize_predictions(model, val_loader, device, epoch)

        # --- Checkpoint: bestes Modell nach Val-Loss (wie bisher) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename="best_val_loss.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU-Plateau-Zustand
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch_loss"] = epoch
        else:
            patience_cnt += 1

        # --- Checkpoint: bestes Modell nach mIoU ---
        # Wird nur in Epochen gespeichert, in denen decode_seg lief.
        # Parallel zu best_val_loss.pt — keiner ersetzt den anderen.
        if miou_this_epoch is not None and miou_this_epoch > best_miou:
            best_miou = miou_this_epoch
            best_epoch_miou = epoch
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename="best_miou.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU-Plateau-Zustand
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )
            if WANDB_AVAILABLE:
                wandb.run.summary["best_miou"]       = best_miou
                wandb.run.summary["best_epoch_miou"] = epoch
            print(f"         -> neues best_miou.pt: {best_miou:.4f} (Epoch {epoch})", flush=True)

        # --- EMA-Decode-Validierung + best_miou_ema.pt ----------
        # Nur in Decode-Epochen und nur wenn EMA aktiv. Die Live-Weights werden
        # kurz gegen die Shadow-Weights getauscht, validiert und wieder
        # zurueckgetauscht -> das laufende Training bleibt voellig unberuehrt.
        # Der mIoU-Plateau-Early-Stop laeuft weiter auf dem RAW-mIoU (unten),
        # EMA greift nicht in die Abbruchlogik ein.
        if ema_enabled and miou_this_epoch is not None:
            _backup = {name: p.detach().clone()
                       for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                p.data.copy_(ema_state[name])
            miou_ema, _per_class_ema = validate_seg(
                model, seg_decoder,
                seg_subset_inputs, seg_subset_targets, seg_real_masks, device,
            )
            for name, p in model.named_parameters():
                p.data.copy_(_backup[name])
            print(f"         [EMA] mIoU={miou_ema:.4f} (raw={miou_this_epoch:.4f})", flush=True)
            if WANDB_AVAILABLE:
                wandb.log({"val/miou_ema": miou_ema, "epoch": epoch})
            if miou_ema > best_miou_ema:
                best_miou_ema = miou_ema
                for name, p in model.named_parameters():
                    p.data.copy_(ema_state[name])
                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                    config=model_config, checkpoint_dir=checkpoint_dir,
                    filename="best_miou_ema.pt",
                    best_miou=best_miou_ema, best_epoch_miou=epoch,
                    miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                    patience_cnt=patience_cnt,
                )
                for name, p in model.named_parameters():
                    p.data.copy_(_backup[name])
                if WANDB_AVAILABLE:
                    wandb.run.summary["best_miou_ema"] = best_miou_ema
                print(f"         -> neues best_miou_ema.pt: {best_miou_ema:.4f} (Epoch {epoch})", flush=True)

        # --- mIoU-Plateau-Zaehler aktualisieren (nur in Decode-Epochen) ---
        # Signifikanter Fortschritt (> Rauschboden ueber der Referenz) setzt den
        # Zaehler zurueck und hebt die Referenz. Sonst hochzaehlen. best_miou.pt
        # oben bleibt davon unberuehrt -> der gespeicherte Checkpoint ist IMMER
        # der beste, nie der vom Abbruchzeitpunkt.
        if miou_early_stop and miou_this_epoch is not None:
            if miou_this_epoch > miou_stop_ref + miou_min_delta:
                miou_stop_ref = miou_this_epoch
                miou_patience_cnt = 0
            else:
                miou_patience_cnt += 1
                print(f"         mIoU-Plateau: {miou_patience_cnt}/{miou_patience} "
                      f"Messungen ohne signifikanten Fortschritt "
                      f"(>{miou_min_delta:.3f} ueber {miou_stop_ref:.4f})", flush=True)

        # --- Checkpoint: periodischer Snapshot ---
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename=f"epoch_{epoch+1:03d}.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU-Plateau-Zustand
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )

        # --- Early Stopping (Val-Loss ODER mIoU-Plateau) ---
        # Val-Loss-Patience wie bisher (Kontinuitaet). ZUSAETZLICH mIoU-Plateau,
        # falls aktiv -- was zuerst greift, stoppt. Der beste Checkpoint
        # (best_miou.pt / best_val_loss.pt) ist zu diesem Zeitpunkt bereits
        # gespeichert, wir werfen also nur noch marginale End-Epochen weg.
        if patience_cnt >= patience:
            stop_reason = "val_loss_patience"
            print(f"\nEarly Stopping: {patience_cnt} Epochs ohne Val-Loss-Verbesserung.")
            break
        if miou_early_stop and miou_patience_cnt >= miou_patience:
            stop_reason = "miou_plateau"
            print(f"\nEarly Stopping: {miou_patience_cnt} mIoU-Messungen ohne "
                  f"signifikanten Fortschritt (>{miou_min_delta:.3f}). "
                  f"Bestes mIoU {best_miou:.4f} bereits als best_miou.pt gesichert.")
            break

    # -------------------------------------------------------------------------
    # 9. Abschluss
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training abgeschlossen.")
    print(f"  Bester Val-Loss: {best_val_loss:.6f}")
    if decode_enabled and best_miou >= 0:
        print(f"  Beste mIoU:      {best_miou:.4f}  -> best_miou.pt")
    best_ckpt = find_best_checkpoint(checkpoint_dir, model_config.phase)
    if best_ckpt:
        print(f"  Bester Checkpoint (Loss): {best_ckpt}")
    print(f"{'='*60}\n")

    # --- maschinenlesbares Run-Summary fuer sweep_runner.py ---
    import json as _json
    _phase_num = 2 if model_config.phase == "cell" else 1
    _summary_path = Path(checkpoint_dir) / f"phase{_phase_num}" / "run_summary.json"
    _summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_summary_path, "w") as _f:
        _json.dump({
            "best_miou":       best_miou,
            "best_epoch_miou": best_epoch_miou,
            "best_val_loss":   best_val_loss,
            "stop_reason":     stop_reason,
            "epochs_run":      epoch + 1,
            "lambdas": {"mse": lambda_mse, "cos": lambda_cos, "mean": lambda_mean,
                        "std": lambda_std, "grad": lambda_grad, "ssim": lambda_ssim},
            "recon_loss": recon_loss,   # TERM-1-Distanz (mse|l1|smooth_l1)
            "seed": train_cfg.get("seed", 42),
        }, _f, indent=2)
    print(f"[summary] run_summary.json -> {_summary_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

    # --- /dev/shm-Staging aufraeumen ---
    # Am NORMALEN Job-Ende. Bei hartem Kill (scancel/Timeout/OOM) laeuft diese
    # Zeile nicht mehr -> dafuer ist der trap im sbatch-Skript die Absicherung
    # (bewusst redundant, weil liegengebliebener /dev/shm-Muell den geteilten
    # Knoten fuer andere Nutzer blockiert).
    if cfg["data"].get("stage_to_shm", False):
        import shm_staging
        shm_staging.cleanup(cfg)


# =============================================================================
# Einstiegspunkt
# =============================================================================

if __name__ == "__main__":
    main()
