#!/usr/bin/env python3
# =============================================================================
# bench_resources.py — Task 26: Ressourcenbedarf (Inferenzzeit/Speicher/Params)
# =============================================================================
# EIN Skript fuer alle World-Model-Varianten. KEIN Training, nur Inferenz.
# Aufruf aus dem Projektroot:
#   python3 archiv/tools/bench_resources.py
# Protokoll: >=20 Warmup, >=200 Messiterationen, torch.cuda.synchronize() vor
# jeder Zeitnahme, Batch 1, fp16-autocast wie im Betrieb. Det: latent_scale-
# Pfad (Werte werden real skaliert wie beim Laden). Flow: steps=1, Backbone-
# Forward UND +1 Euler-Schritt getrennt und gesamt.
# =============================================================================
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "Code")
sys.path.insert(0, ".")
import train_linux as T

DEV = torch.device("cuda")
OUT = Path("predictions/task26_ressourcen")
WARMUP, ITERS = 30, 300

# (name, config, checkpoint, is_flow, is_det)
VARIANTS = [
    ("seg_wm_minimal", "config_sweep_base_fp16.yaml",
     "checkpoints/task16b/headline/smoothl1_minimal/phase2/best_miou.pt", False, False),
    ("seg_wm_8layer", "archiv/configs/config_cap_l8.yaml",
     "checkpoints/task16f/cap_l8/phase2/best_miou.pt", False, False),
    ("det_wm_6term", "config_det_baseline_eval.yaml",
     "checkpoints/task20/det_baseline/phase2/best_val_loss.pt", False, True),
    ("flow_head", "archiv/configs/config_flow.yaml",
     "checkpoints/task18/flow/phase2/best_miou.pt", True, False),
    ("cvae_head", "archiv/configs/config_vae_0.01.yaml",
     "checkpoints/task18/vae_0.01/phase2/best_miou.pt", False, False),
]


def hw_stamp():
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True)
    name, drv = [s.strip() for s in q.stdout.strip().split(",")]
    return {"gpu": name, "driver": drv, "torch": torch.__version__,
            "cuda": torch.version.cuda}


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": ITERS}


@torch.no_grad()
def bench(name, cfg_path, ckpt, is_flow, is_det, hw):
    print(f"\n=== {name} ===", flush=True)
    cfg = T.load_config(cfg_path, "cell")
    mcfg = T.ModelConfig(**cfg["model"])
    model = T.BEVWorldModel(mcfg).to(DEV).eval()
    if Path(ckpt).exists():
        T.load_checkpoint(ckpt, model, device=str(DEV))
    scale = float(cfg.get("data", {}).get("latent_scale", 1.0))

    g = mcfg.grid_size * 4 if hasattr(mcfg, "grid_size") else 128
    # Latent-Aufloesung aus Config: seg 128, det 180
    H = 180 if is_det else 128
    C = 256
    x = torch.randn(1, mcfg.n_frames, C, H, H, device=DEV) * scale  # betriebsnah skaliert

    # --- statisch ---
    total = sum(p.numel() for p in model.parameters())
    params = {"total": total}
    if is_flow and hasattr(model, "flow"):
        fp = sum(p.numel() for p in model.flow.parameters())
        params["backbone"] = total - fp
        params["kopf"] = fp
    ckpt_mb = Path(ckpt).stat().st_size / 2**20 if Path(ckpt).exists() else None
    # Gewichte im Speicher (fp32 wie geladen; fp16 = Haelfte)
    w_fp32 = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

    # --- Speicher + Zeit ---
    torch.cuda.reset_peak_memory_stats()
    use_amp = True

    def fwd():
        with torch.cuda.amp.autocast(enabled=use_amp):
            return model(x)

    res = {"params": params, "ckpt_mb": ckpt_mb, "weights_fp32_mb": w_fp32,
           "weights_fp16_mb": w_fp32 / 2, "hardware": hw,
           "precision": "fp16-autocast"}

    if is_flow:
        # Backbone-Forward (det. Vorhersage) UND +1 Euler-Schritt (steps=1)
        def full():
            with torch.cuda.amp.autocast(enabled=use_amp):
                return model.flow_sample(x, steps=1)
        t_bb = timed(fwd)
        t_full = timed(full)
        res["t_forward_ms"] = t_bb
        res["t_flow_total_ms"] = t_full
        res["t_flow_step_ms"] = {"mean": t_full["mean"] - t_bb["mean"],
                                 "note": "steps=1 (18/B2: identisch zu steps=10)"}
    else:
        res["t_forward_ms"] = timed(fwd)

    torch.cuda.synchronize()
    res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["batch"] = 1
    # optional: Batch-8-Durchsatz
    x8 = torch.randn(8, mcfg.n_frames, C, H, H, device=DEV) * scale

    def fwd8():
        with torch.cuda.amp.autocast(enabled=use_amp):
            return model(x8)
    t8 = timed(fwd8)
    res["t_batch8_ms"] = t8
    res["throughput_b8_fps"] = 8000.0 / t8["mean"]

    del model, x, x8
    torch.cuda.empty_cache()
    m = res["t_forward_ms"]
    print(f"  params {total/1e6:.3f}M | forward {m['mean']:.2f}+-{m['std']:.2f} ms "
          f"| peak {res['peak_vram_mb']:.0f} MB", flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw = hw_stamp()
    print("Hardware:", hw)
    results = {}
    for name, cfg, ckpt, is_flow, is_det in VARIANTS:
        try:
            results[name] = bench(name, cfg, ckpt, is_flow, is_det, hw)
        except Exception as e:
            print(f"  FEHLER {name}: {e}", flush=True)
            results[name] = {"error": str(e)}
    # Persistenz-Zeile (nicht gebencht)
    results["persistenz"] = {"params": {"total": 0}, "ckpt_mb": 0,
                             "t_forward_ms": {"mean": 0.0, "std": 0.0, "n": 0},
                             "note": "0 Parameter, ~0 ms (Kopie des letzten Frames)"}
    (OUT / "ressourcen.json").write_text(json.dumps(results, indent=2))
    print("\ngeschrieben:", OUT / "ressourcen.json")


if __name__ == "__main__":
    main()
