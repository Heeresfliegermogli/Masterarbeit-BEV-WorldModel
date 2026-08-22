#!/usr/bin/env python3
# =============================================================================
# profile_train.py  --  TEMPORAER (Task 16a.1B, Stufe 0)
# =============================================================================
# ZWECK: einmalige Attribution "wohin geht die Epochenzeit" auf dem IST-Zustand,
#   BEVOR Stufe-1/2-Umbauten in train_linux.py landen. Beantwortet die zwei
#   Fragen aus dem Arbeitsplan:
#     (1) Dominiert SSIM auf [256,128,128] die Step-Zeit?  -> A/B (ssim on/off)
#     (2) ConvTranspose-Head vs. SSIM-Interna?             -> Top-Op-Tabelle
#
# NICHT TEIL DER PIPELINE. Kein Sweep/Anker ruft das je auf. Analog zu
#   diagnose_mean_gap.py (16a.1): einmal laufen, Zahlen ziehen, in den
#   Abschlussbericht. Aendert train_linux.py bewusst NICHT -> das Baseline-
#   Profil bleibt ehrlich (unveraenderte Config: contiguous, benchmark=False).
#
# METRIK-DESIGN:
#   - Per-Phase-Wanduhr via CUDA-Events (h2d/forward/loss/backward/optim),
#     gemittelt ueber die aktiven Steps. data_wait separat (CPU, perf_counter).
#   - SSIM-Kosten NICHT via record_function um den Vorwaerts-SSIM-Call gemessen
#     (das verschluckt SSIMs BACKWARD-Anteil), sondern als A/B-Delta: identischer
#     Messlauf mit lambda_ssim=<config> vs. lambda_ssim=0.0. Delta der mittleren
#     Step-Zeit = SSIMs voller Forward+Backward-Aufwand. Das ist die Zahl, die
#     ueber Stufe-2-(f) [SSIM@32 statt @128] entscheidet.
#   - torch.profiler-Op-Tabelle (CUDA-Zeit, Top-N) trennt aten::conv_transpose2d
#     (Upsampling-Head) sauber von den SSIM-Kerneln.
#
# VERWENDUNG (auf alre-server-u20, im Code_final/-Verzeichnis neben train_linux.py):
#   python profile_train.py --config config_sweep_base_fp16.yaml --phase cell
#   python profile_train.py --config config_sweep_base_fp16.yaml --no-ssim-ab \
#          --no-trace                 # nur Phasen-Profil, ohne A/B, ohne Trace
#
# Danach loeschen ODER liegen lassen (Header markiert es als temporaer).
# =============================================================================

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch

# --- echte Interfaces aus dem kanonischen Code wiederverwenden --------------
# (Import fuehrt train_linux.py-Top-Level aus -- nur Defs + SSIM-Info-Print,
#  main() liegt hinter __main__; harmlos.)
from train_linux import load_config, compute_loss, build_dataloaders, set_seed
from config import ModelConfig
from bev_world_model import BEVWorldModel

PHASES = ["h2d", "forward", "loss", "backward", "optim"]


# -----------------------------------------------------------------------------
def build(cfg, device):
    """Modell + Optimizer + Scaler exakt wie im echten Trainingsstart bauen."""
    model = BEVWorldModel(ModelConfig(**cfg["model"])).to(device)
    model.train()
    tr = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"],
        fused=(device.type == "cuda"),   # 1b: wie in train_linux.py
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    return model, optimizer, scaler


def lambdas_from_cfg(cfg, ssim_override=None):
    tr = cfg["training"]
    # .get mit den 15.1-Defaults statt hartem Key-Zugriff: macht das Skript
    # robust gegen alte 3-Term-Configs (z.B. config_nuscenes_full_cluster_fp16.yaml
    # traegt noch lambda_dist statt mean/std/grad). Defaults == Baseline-Loss
    # (15.3B §8). Fuer Datenpfad-Messungen ist der exakte lambda-Satz irrelevant.
    ssim = tr.get("lambda_ssim", 0.1) if ssim_override is None else ssim_override
    return (tr.get("lambda_mse", 1.0), tr.get("lambda_cos", 0.1),
            tr.get("lambda_mean", 0.1), tr.get("lambda_std", 0.1),
            tr.get("lambda_grad", 0.0), ssim)


def _events():
    return {p: (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for p in PHASES}


# -----------------------------------------------------------------------------
def measure(model, loader, optimizer, scaler, device, lambdas,
            n_wait, n_warmup, n_active, profiler=None):
    """
    Faehrt (n_wait + n_warmup + n_active) echte Trainings-Steps. Timing NUR
    ueber die letzten n_active. Bildet den heissen Loop aus train_linux.py
    train_one_epoch() 1:1 nach (autocast + GradScaler + clip max_norm=1.0).
    Gibt (phase_ms{}, data_wait_ms, step_ms) zurueck -- Mittel je aktivem Step.
    """
    cuda = device.type == "cuda"
    totals = {p: 0.0 for p in PHASES}
    data_wait_ms = 0.0
    n_total = n_wait + n_warmup + n_active
    it = iter(loader)

    step_t_start = None
    step_ms_total = 0.0

    ctx = profiler if profiler is not None else nullcontext()
    with ctx:
        for step in range(n_total):
            timed = step >= (n_wait + n_warmup)

            t0 = time.perf_counter()
            batch = next(it)
            dw = (time.perf_counter() - t0) * 1000.0

            ev = _events() if timed else None
            if timed and cuda:
                torch.cuda.synchronize()
                step_t_start = time.perf_counter()

            # --- h2d ---  (1d: Loader liefert fp16, Upcast auf fp32 auf der GPU,
            #               .float() im h2d-Block gemessen -- wie im echten Loop.
            #               Auf fp32-Packs ist .float() ein No-op.)
            if timed:
                ev["h2d"][0].record()
            inputs = batch["input"].to(device, non_blocking=True).float()   # [B,3,256,128,128]
            target = batch["target"].to(device, non_blocking=True).float()  # [B,256,128,128]
            if timed:
                ev["h2d"][1].record()

            optimizer.zero_grad(set_to_none=True)

            # --- forward + loss (unter autocast, wie im echten Loop) ---
            with torch.cuda.amp.autocast(enabled=cuda):
                if timed:
                    ev["forward"][0].record()
                pred = model(inputs)
                if timed:
                    ev["forward"][1].record()
                    ev["loss"][0].record()
                loss, *_ = compute_loss(pred, target, *lambdas)
                if timed:
                    ev["loss"][1].record()

            # --- backward ---
            if timed:
                ev["backward"][0].record()
            scaler.scale(loss).backward()
            if timed:
                ev["backward"][1].record()

            # --- optim (unscale + clip + step + update) ---
            if timed:
                ev["optim"][0].record()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if timed:
                ev["optim"][1].record()

            if timed:
                if cuda:
                    torch.cuda.synchronize()
                    step_ms_total += (time.perf_counter() - step_t_start) * 1000.0
                for p in PHASES:
                    totals[p] += ev[p][0].elapsed_time(ev[p][1])
                data_wait_ms += dw

            if profiler is not None:
                profiler.step()

    phase_ms = {p: totals[p] / n_active for p in PHASES}
    return phase_ms, data_wait_ms / n_active, step_ms_total / n_active


# -----------------------------------------------------------------------------
def fmt_report(phase_ms, data_wait_ms, step_ms, ssim_delta_ms, cfg, args, top_ops):
    tr = cfg["training"]
    lines = []
    lines.append("# TASK 16a.1B -- STUFE 0 PROFILING (profile_train.py, temporaer)\n")
    lines.append(f"- Config:      {args.config}")
    lines.append(f"- Phase:       {args.phase}")
    lines.append(f"- batch_size:  {tr['batch_size']}   "
                 f"(lambda_ssim={tr.get('lambda_ssim', 0.1)}, "
                 f"lambda_grad={tr.get('lambda_grad', 0.0)}, "
                 f"lambda_std={tr.get('lambda_std', 0.1)})")
    lines.append(f"- Steps:       wait={args.wait}, warmup={args.warmup}, active={args.active}")
    lines.append(f"- Ist-Zustand: channels_first, "
                 f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
                 f"fused_adamw={torch.cuda.is_available()}\n")

    lines.append("## Per-Phase-Wanduhr (Mittel je Step, CUDA-Events)\n")
    lines.append("| Phase | ms/Step | % Compute |")
    lines.append("|---|---:|---:|")
    compute = sum(phase_ms.values())
    for p in PHASES:
        pct = 100.0 * phase_ms[p] / compute if compute > 0 else 0.0
        lines.append(f"| {p} | {phase_ms[p]:.2f} | {pct:.1f}% |")
    lines.append(f"| **compute (Summe)** | **{compute:.2f}** | 100% |")
    lines.append("")
    lines.append(f"- data_wait (CPU, DataLoader-Warten): {data_wait_ms:.2f} ms/Step "
                 f"(mit prefetch idealerweise ~0)")
    lines.append(f"- Voller Step (sync-zu-sync, inkl. h2d/optim): {step_ms:.2f} ms/Step\n")

    lines.append("## SSIM-Attribution (A/B: lambda_ssim on vs. 0.0)\n")
    if ssim_delta_ms is None:
        lines.append("- (uebersprungen, --no-ssim-ab)\n")
    else:
        pct = 100.0 * ssim_delta_ms / step_ms if step_ms > 0 else 0.0
        lines.append(f"- SSIM-Vollkosten (Forward+Backward): **{ssim_delta_ms:.2f} ms/Step "
                     f"= {pct:.1f}% des Steps**")
        verdict = ("-> Stufe-2-(f) [SSIM@32] LOHNT (Schwelle >20%)" if pct > 20
                   else "-> Stufe-2-(f) [SSIM@32] LOHNT NICHT (< 20%), SSIM so lassen")
        lines.append(f"- {verdict}\n")

    lines.append("## Top-Operatoren nach CUDA-Zeit (Head vs. SSIM trennen)\n")
    if top_ops:
        lines.append("```")
        lines.append(top_ops)
        lines.append("```")
    else:
        lines.append("- (kein Profiler-Lauf; --no-trace impliziert weiterhin Op-Tabelle, "
                     "hier leer nur falls Profiler deaktiviert)")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stufe-0 Profiling (temporaer, 16a.1B)")
    ap.add_argument("--config", default="config_sweep_base_fp16.yaml")
    ap.add_argument("--phase", default="cell", choices=["cell", "frame"])
    ap.add_argument("--wait", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--active", type=int, default=8)
    ap.add_argument("--no-ssim-ab", action="store_true",
                    help="A/B-Vergleich (ssim on/off) ueberspringen")
    ap.add_argument("--no-trace", action="store_true",
                    help="Chrome-Trace nicht schreiben (Op-Tabelle bleibt)")
    ap.add_argument("--out", default="predictions/task16a_1b_profile")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[profile_train] Kein CUDA -- GPU-Profiling nicht moeglich.")

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, phase_override=args.phase)
    set_seed(cfg["training"].get("seed", 42))

    print(f"[profile_train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[profile_train] Dataloader bauen (num_workers="
          f"{cfg['training'].get('num_workers', 4)}, packed fp16)...", flush=True)
    train_loader, _val, _test = build_dataloaders(cfg)

    model, optimizer, scaler = build(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[profile_train] Modell: {n_params:,} Parameter, Phase={args.phase}\n", flush=True)

    # --- Hauptmessung MIT Profiler (Phasen + Op-Tabelle + optional Trace) ---
    lam_on = lambdas_from_cfg(cfg)
    sched = torch.profiler.schedule(
        wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)

    trace_handler = None
    if not args.no_trace:
        trace_path = out_dir / "trace.json"

        def trace_handler(prof):
            prof.export_chrome_trace(str(trace_path))

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=False,
    )

    print("[profile_train] Messlauf (SSIM ON) laeuft...", flush=True)
    phase_ms, data_wait_ms, step_ms = measure(
        model, train_loader, optimizer, scaler, device, lam_on,
        args.wait, args.warmup, args.active, profiler=prof)

    top_ops = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=args.topk)

    # --- A/B: identischer Lauf mit lambda_ssim=0.0, nur Step-Zeit-Delta ---
    ssim_delta_ms = None
    if not args.no_ssim_ab and cfg["training"].get("lambda_ssim", 0.1) > 0:
        print("[profile_train] A/B-Messlauf (SSIM OFF) laeuft...", flush=True)
        lam_off = lambdas_from_cfg(cfg, ssim_override=0.0)
        # frisches Modell/Optimizer -> gleiche Startbedingung, kein Zustand aus
        # dem ON-Lauf (Scaler/Momentum) verzerrt den Vergleich.
        model2, optimizer2, scaler2 = build(cfg, device)
        _p2, _d2, step_ms_off = measure(
            model2, train_loader, optimizer2, scaler2, device, lam_off,
            args.wait, args.warmup, args.active, profiler=None)
        ssim_delta_ms = step_ms - step_ms_off
        print(f"[profile_train] Step ON={step_ms:.2f}ms  OFF={step_ms_off:.2f}ms  "
              f"-> SSIM={ssim_delta_ms:.2f}ms", flush=True)

    report = fmt_report(phase_ms, data_wait_ms, step_ms, ssim_delta_ms, cfg, args, top_ops)
    report_path = out_dir / "profile_report.md"
    report_path.write_text(report)

    print("\n" + report)
    print(f"\n[profile_train] Report -> {report_path}")
    if not args.no_trace:
        print(f"[profile_train] Chrome-Trace -> {out_dir / 'trace.json'} "
              f"(chrome://tracing bzw. https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
