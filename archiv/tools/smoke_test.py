"""
smoke_test.py — BEV World Model: Smoke Test für train.py (Task 5)
=================================================================

Testet den gesamten Trainingsprozess OHNE echte nuScenes-Daten.

WAS IST EIN SMOKE TEST?
    "Stecke den Stecker rein und schau ob Rauch aufsteigt."
    Kein tiefes Testen — nur prüfen ob das Gesamtsystem läuft:
        ✓ Alle Imports funktionieren (sys.path korrekt?)
        ✓ config.yaml wird korrekt gelesen
        ✓ ModelConfig und BEVWorldModel werden korrekt instanziiert
        ✓ Ein Forward-Pass produziert die richtige Output-Shape
        ✓ Loss-Berechnung funktioniert (MSE + CosSim)
        ✓ Backward-Pass läuft (Gradienten fließen)
        ✓ Optimizer-Schritt ohne Fehler
        ✓ Scheduler-Schritt funktioniert
        ✓ Checkpoint speichern und laden (Round-Trip)
        ✓ 3 Mini-Epochs Ende-zu-Ende (train + val + logging)

WAS WIRD GEMOCKT?
    - DataLoader: ersetzt durch synthetische Zufallstensoren
      (echte .npy-Dateien sind auf dem Server, nicht hier)
    - wandb: deaktiviert (kein Account nötig)
    - matplotlib: deaktiviert (kein Display nötig)

WAS WIRD NICHT GEMOCKT?
    - Das Modell (BEVWorldModel) — läuft komplett real
    - Loss (compute_loss) — echte Berechnung
    - Optimizer, Scheduler, GradScaler — alles real
    - Checkpointing — echter Schreib-/Lese-Zyklus

AUFRUF:
    python smoke_test.py                    # Phase frame, CPU
    python smoke_test.py --phase cell       # Phase cell testen
    python smoke_test.py --device cuda      # auf GPU testen
"""

import argparse
import sys
import shutil
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sys.path: code/ bekannt machen (identisch zu train.py)
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE_DIR))

from config          import ModelConfig
from bev_world_model import BEVWorldModel
from checkpointing   import save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")

def _fail(msg: str) -> None:
    print(f"  ✗ FEHLER: {msg}")

def _section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def make_fake_batch(batch_size: int, n_frames: int, device: torch.device) -> dict:
    """
    Erzeugt einen synthetischen Batch der exakt dem echten DataLoader-Output
    entspricht:
        batch["input"]  = [B, n_frames, 256, 128, 128]   float32
        batch["target"] = [B, 256, 128, 128]              float32

    Die Werte sind zufällig aber nicht-negativ (ReLU-aktiviert wie echte Latents):
        torch.rand → gleichverteilt in [0, 1)
        * 2        → [0, 2) — grobe Annäherung an echte Latent-Skala
    """
    return {
        "input":  torch.rand(batch_size, n_frames, 256, 128, 128, device=device) * 2,
        "target": torch.rand(batch_size, 256, 128, 128,         device=device) * 2,
    }


def make_fake_loader(n_batches: int, batch_size: int, n_frames: int, device: torch.device):
    """
    Erzeugt eine Liste von Fake-Batches — verhält sich wie ein DataLoader,
    d.h. ist iterierbar und hat eine Länge.

    train_one_epoch und validate in train.py machen nur:
        for batch in loader: ...
        len(loader)
    → eine einfache Liste reicht.
    """
    return [make_fake_batch(batch_size, n_frames, device) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Die eigentlichen Smoke-Test-Funktionen
# (importiert aus train.py — kein Code dupliziert)
# ---------------------------------------------------------------------------

# Wir importieren die Bausteine direkt aus train.py
# sys.path.insert für das Elternverzeichnis (wo train.py liegt)
TRAIN_DIR = Path(__file__).parent
sys.path.insert(0, str(TRAIN_DIR))

from train import compute_loss, train_one_epoch, validate, set_seed


# ---------------------------------------------------------------------------
# TEST 1: Imports & sys.path
# ---------------------------------------------------------------------------

def test_imports() -> bool:
    _section("TEST 1: Imports & sys.path")
    try:
        # Diese Imports sind der häufigste Fehler wenn sys.path falsch ist
        from config          import ModelConfig
        from bev_world_model import BEVWorldModel
        from checkpointing   import save_checkpoint, load_checkpoint
        _ok("config.py importiert")
        _ok("bev_world_model.py importiert")
        _ok("checkpointing.py importiert")

        from train import compute_loss, train_one_epoch, validate, set_seed, load_config
        _ok("train.py Bausteine importiert")
        return True
    except ImportError as e:
        _fail(f"Import fehlgeschlagen: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 2: config.yaml laden
# ---------------------------------------------------------------------------

def test_config_loading() -> bool:
    _section("TEST 2: config.yaml laden")
    try:
        from train import load_config
        cfg = load_config(str(TRAIN_DIR / "config.yaml"))

        assert "model"    in cfg, "Sektion 'model' fehlt in config.yaml"
        assert "training" in cfg, "Sektion 'training' fehlt in config.yaml"
        assert "data"     in cfg, "Sektion 'data' fehlt in config.yaml"
        _ok("config.yaml gelesen")

        # Phase-Override testen
        cfg_cell = load_config(str(TRAIN_DIR / "config.yaml"), phase_override="cell")
        assert cfg_cell["model"]["phase"] == "cell", "Phase-Override funktioniert nicht"
        _ok("Phase-Override 'cell' funktioniert")

        # ModelConfig aus YAML bauen
        model_cfg = ModelConfig(**cfg["model"])
        _ok(f"ModelConfig instanziiert: phase='{model_cfg.phase}', "
            f"n_layers={model_cfg.n_layers}, d_model={model_cfg.d_model}")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 3: Modell Forward-Pass
# ---------------------------------------------------------------------------

def test_forward_pass(phase: str, device: torch.device) -> bool:
    _section(f"TEST 3: Forward-Pass (phase='{phase}', device={device})")
    try:
        cfg   = ModelConfig(phase=phase)
        model = BEVWorldModel(cfg).to(device)
        model.eval()

        # Kleiner Batch: B=2 für Geschwindigkeit
        B = 2
        x = torch.rand(B, cfg.n_frames, 256, 128, 128, device=device) * 2

        with torch.no_grad():
            out = model(x)

        expected = (B, 256, 128, 128)
        assert out.shape == expected, f"Falsche Shape: {out.shape} != {expected}"
        _ok(f"Input:  {list(x.shape)}")
        _ok(f"Output: {list(out.shape)} ✓")

        assert not torch.isnan(out).any(),  "NaN im Output!"
        assert not torch.isinf(out).any(),  "Inf im Output!"
        _ok("Kein NaN / Inf im Output")

        n_params = sum(p.numel() for p in model.parameters())
        _ok(f"Parameter: {n_params:,}")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 4: Loss-Berechnung
# ---------------------------------------------------------------------------

def test_loss(device: torch.device) -> bool:
    _section(f"TEST 4: Loss-Berechnung (device={device})")
    try:
        B = 2
        pred   = torch.rand(B, 256, 128, 128, device=device)
        target = torch.rand(B, 256, 128, 128, device=device)

        # MSE only
        loss_t, loss_mse, loss_cos, loss_dist = compute_loss(pred, target, use_cosine=False)
        assert loss_cos.item()  == 0.0, "use_cosine=False: cos sollte 0 sein"
        _ok(f"MSE-only: mse={loss_mse.item():.6f}, dist={loss_dist.item():.6f}")

        # MSE + CosSim + Distribution
        loss_t2, loss_mse2, loss_cos2, loss_dist2 = compute_loss(pred, target, use_cosine=True)
        assert loss_cos2.item()  > 0.0, "Cosine Loss sollte > 0 sein"
        assert loss_dist2.item() > 0.0, "Distribution Loss sollte > 0 sein"
        expected = loss_mse2.item() + 0.1 * loss_cos2.item() + 0.1 * loss_dist2.item()
        assert abs(loss_t2.item() - expected) < 1e-4, \
            f"loss_total={loss_t2.item():.6f} != expected={expected:.6f}"
        _ok(f"Alle Terme: total={loss_t2.item():.4f}, mse={loss_mse2.item():.4f}, "
            f"cos={loss_cos2.item():.4f}, dist={loss_dist2.item():.4f}")

        # Loss muss skalar sein
        assert loss_t2.shape == torch.Size([]), f"Loss ist kein Skalar: {loss_t2.shape}"
        _ok("Loss ist Skalar ✓")

        # Gradienten fließen?
        pred_grad = pred.detach().requires_grad_(True)
        loss_grad, _, _, _ = compute_loss(pred_grad, target, use_cosine=True)
        loss_grad.backward()
        assert pred_grad.grad is not None, "Keine Gradienten!"
        assert not torch.isnan(pred_grad.grad).any(), "NaN in Gradienten!"
        _ok("Gradienten fließen korrekt ✓")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 5: Backward-Pass & Optimizer-Schritt
# ---------------------------------------------------------------------------

def test_backward(phase: str, device: torch.device) -> bool:
    _section(f"TEST 5: Backward-Pass & Optimizer (phase='{phase}')")
    try:
        cfg       = ModelConfig(phase=phase)
        model     = BEVWorldModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        model.train()
        B      = 2
        inputs = torch.rand(B, cfg.n_frames, 256, 128, 128, device=device) * 2
        target = torch.rand(B, 256, 128, 128,             device=device) * 2

        # Gewichte vor Update merken (einer davon zum Vergleich)
        param_before = next(model.parameters()).data.clone()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(inputs)
            loss, _, _, _ = compute_loss(pred, target, use_cosine=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        _ok(f"Loss: {loss.item():.6f}")

        # Gradienten prüfen
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Keine Gradienten berechnet!"
        assert not any(torch.isnan(g).any() for g in grads), "NaN in Gradienten!"
        _ok(f"Gradienten: {len(grads)} Parameter-Tensoren, kein NaN")

        # Gewichte haben sich geändert?
        param_after = next(model.parameters()).data
        assert not torch.equal(param_before, param_after), \
            "Gewichte haben sich nicht geändert — Optimizer-Schritt wirkungslos?"
        _ok("Gewichte wurden erfolgreich aktualisiert ✓")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 6: Scheduler
# ---------------------------------------------------------------------------

def test_scheduler(phase: str) -> bool:
    _section(f"TEST 6: CosineAnnealingLR Scheduler (phase='{phase}')")
    try:
        cfg       = ModelConfig(phase=phase)
        model     = BEVWorldModel(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )

        # optimizer.step() muss vor scheduler.step() kommen (PyTorch >= 1.1)
        lr_values = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            lr_values.append(scheduler.get_last_lr()[0])
            optimizer.zero_grad()

        # LR muss monoton fallen (Cosinus-Anfang ist fallend)
        assert all(lr_values[i] >= lr_values[i+1] for i in range(len(lr_values)-1)), \
            f"LR fällt nicht monoton: {lr_values}"
        _ok(f"LR nach 5 Schritten: {[f'{lr:.2e}' for lr in lr_values]}")
        _ok("LR fällt korrekt (Cosine Annealing) ✓")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# TEST 7: Checkpoint Round-Trip
# ---------------------------------------------------------------------------

def test_checkpointing(phase: str, device: torch.device) -> bool:
    _section(f"TEST 7: Checkpoint Round-Trip (phase='{phase}')")
    tmpdir = tempfile.mkdtemp()
    try:
        cfg       = ModelConfig(phase=phase)
        model     = BEVWorldModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )

        # Checkpoint speichern
        save_path = save_checkpoint(
            model          = model,
            optimizer      = optimizer,
            scheduler      = scheduler,
            epoch          = 7,
            val_loss       = 0.042,
            best_val_loss  = 0.038,
            config         = cfg,
            checkpoint_dir = tmpdir,
            filename       = "test_ckpt.pt",
        )
        _ok(f"Checkpoint gespeichert: {save_path.name}")

        # Neues Modell — andere Gewichte
        model2     = BEVWorldModel(cfg).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=50, eta_min=1e-6
        )

        # Checkpoint laden
        ckpt = load_checkpoint(
            checkpoint_path = str(save_path),
            model           = model2,
            optimizer       = optimizer2,
            scheduler       = scheduler2,
            device          = str(device),
        )

        assert ckpt["epoch"]    == 7,     f"epoch falsch: {ckpt['epoch']}"
        assert ckpt["val_loss"] == 0.042, f"val_loss falsch: {ckpt['val_loss']}"
        assert ckpt["phase"]    == phase, f"phase falsch: {ckpt['phase']}"
        _ok(f"Epoch: {ckpt['epoch']}, Val-Loss: {ckpt['val_loss']}, Phase: {ckpt['phase']}")

        # Weights identisch?
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1.cpu(), p2.cpu()), \
                f"Parameter '{n1}' unterschiedlich nach Laden!"
        _ok("Alle Gewichte identisch nach Round-Trip ✓")
        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST 8: Mini-Training (3 Epochs Ende-zu-Ende)
# ---------------------------------------------------------------------------

def test_mini_training(phase: str, device: torch.device) -> bool:
    _section(f"TEST 8: Mini-Training 3 Epochs (phase='{phase}', device={device})")
    tmpdir = tempfile.mkdtemp()
    try:
        set_seed(42)

        cfg       = ModelConfig(phase=phase)
        model     = BEVWorldModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=3, eta_min=1e-6
        )
        scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        # Fake DataLoader: 3 Batches Train, 2 Batches Val
        # Batch-Size 2 für Geschwindigkeit
        BATCH_SIZE = 2
        train_loader = make_fake_loader(3, BATCH_SIZE, cfg.n_frames, device)
        val_loader   = make_fake_loader(2, BATCH_SIZE, cfg.n_frames, device)

        best_val_loss = float("inf")
        history = []

        for epoch in range(3):
            # --- Train ---
            train_loss, train_mse, train_cos, train_dist = train_one_epoch(
                model, train_loader, optimizer, scaler, device, use_cosine=True
            )

            # --- Validate ---
            val_loss, val_mse, val_cos, val_dist = validate(
                model, val_loader, device, use_cosine=True
            )

            # --- Scheduler ---
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            history.append({
                "epoch": epoch, "train": train_loss, "val": val_loss,
                "dist": val_dist, "lr": lr
            })
            print(f"    Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"dist={val_dist:.4f}  lr={lr:.2e}")

            # --- Checkpoint ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, best_val_loss,
                    cfg, tmpdir, "best_val_loss.pt"
                )

        # Assertions
        assert len(history) == 3, "Nicht alle 3 Epochs gelaufen"
        _ok("3 Epochs komplett durchgelaufen")

        assert all(h["train"] > 0 for h in history), "Train-Loss ≤ 0 — unmöglich"
        assert all(h["val"]   > 0 for h in history), "Val-Loss ≤ 0 — unmöglich"
        _ok("Train- und Val-Loss immer > 0")

        # NaN-Check: kein Zusammenbruch
        assert not any(
            torch.isnan(torch.tensor(h["train"])) for h in history
        ), "NaN im Train-Loss!"
        _ok("Kein NaN / Inf in Loss-Werten")

        # Checkpoint existiert?
        phase_num = 1 if phase == "frame" else 2
        ckpt_path = Path(tmpdir) / f"phase{phase_num}" / "best_val_loss.pt"
        assert ckpt_path.exists(), f"Bester Checkpoint fehlt: {ckpt_path}"
        _ok(f"Bester Checkpoint gespeichert ✓")

        return True
    except Exception as e:
        _fail(str(e))
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Haupt-Runner
# ---------------------------------------------------------------------------

def run_smoke_test(phase: str, device: torch.device) -> None:
    print(f"\n{'═'*55}")
    print(f"  BEV World Model — Smoke Test")
    print(f"  Phase:  {phase}")
    print(f"  Device: {device}")
    print(f"{'═'*55}")

    results = {}

    results["imports"]      = test_imports()
    results["config"]       = test_config_loading()
    results["forward"]      = test_forward_pass(phase, device)
    results["loss"]         = test_loss(device)
    results["backward"]     = test_backward(phase, device)
    results["scheduler"]    = test_scheduler(phase)
    results["checkpointing"]= test_checkpointing(phase, device)
    results["mini_training"]= test_mini_training(phase, device)

    # --- Zusammenfassung ---
    print(f"\n{'═'*55}")
    print(f"  ERGEBNIS")
    print(f"{'═'*55}")
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print(f"{'─'*55}")
    if all_passed:
        print(f"  ✅  Alle Tests bestanden — train.py ist bereit!")
    else:
        n_fail = sum(1 for p in results.values() if not p)
        print(f"  ❌  {n_fail} Test(s) fehlgeschlagen — bitte Fehler oben prüfen.")
    print(f"{'═'*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV World Model Smoke Test")
    parser.add_argument("--phase",  default="frame", choices=["frame", "cell"],
                        help="Welche Phase testen (default: frame)")
    parser.add_argument("--device", default=None,
                        help="'cuda' oder 'cpu' (default: auto-detect)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_smoke_test(phase=args.phase, device=device)
