#!/usr/bin/env python3
# =============================================================================
# sweep_runner.py  —  Task 16a.0 (Weg 1: feste, begruendete Wertelisten)
# =============================================================================
#
# Automatisiert den LAMBDA-SWEEP-Zyklus fuer EINEN Regler ueber eine
# VORGEGEBENE Werteliste. Verallgemeinert mk_lambda_cfg.py (nur lambda_std)
# auf jeden Regler und schliesst die Schleife bis zur Ergebnis-CSV.
#
# Pro Wert L:
#   1. Basis-Config per Zeilen-Regex patchen: <regler> -> L, checkpoints.dir
#      -> <sweep-root>/<regler>_<L>.  (patch_line 1:1 aus mk_lambda_cfg.py,
#      whitespace-flexibel, Exact-1-Match-Guard, Kommentare bleiben.)
#   2. train_linux.py seriell (blockierend, tmux-safe) -> best_miou.pt +
#      run_summary.json  (mIoU-Proxy 300 Subset, Plateau-Stopping killt frueh).
#   3. inference.py post-hoc auf best_miou.pt, erste --n-infer Val-Samples
#      -> inference_log.json  (std-Ratio/pred_std/MSE/CosSim, wie 15.3).
#   4. (optional --full-val) eval_full_val.py -> Full-Val-mIoU (headline).
#   5. Ergebnisse einsammeln -> Zeile in sweep_<regler>.csv.
#
# IDEMPOTENZ (Schritt-Ebene): jeder Schritt wird uebersprungen, wenn sein
#   Output schon existiert. Runner-Neustart nach Abbruch nimmt genau da wieder
#   auf, wo er stehen geblieben ist. --force ignoriert vorhandene Outputs.
#
# BEWUSSTE SCOPE-GRENZE: Weg 1, NICHT adaptiv. Der Runner entscheidet NICHT,
#   wo als naechstes gemessen wird -- die Werteliste kommt vom Menschen (--values).
#   Das "Knie" liest man aus der CSV selbst ab und misst bei Bedarf manuell feiner.
#
# USAGE (in tmux!):
#   python sweep_runner.py --regler lambda_grad --values 0 0.05 0.1 0.3 1.0
#   python sweep_runner.py --regler lambda_ssim --values 0 0.1 0.3 --full-val
#   python sweep_runner.py --regler lambda_grad --values 0 0.05 0.1 0.3 1.0 --dry-run
# =============================================================================

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# --- Standard-Konventionen (an die lokale Umgebung angepasst) ----------------
DEFAULT_BASE       = "config_sweep_base_fp16.yaml"
DEFAULT_SWEEP_ROOT = "/home/vima/Desktop/Masterarbeit/Code_final/checkpoints/task16a"
PHASE              = "cell"          # phase2-Checkpoints
PATCHABLE_REGLER   = {"lambda_mse", "lambda_cos", "lambda_mean",
                      "lambda_std", "lambda_grad", "lambda_ssim"}


# =============================================================================
# Config-Patch (1:1 aus mk_lambda_cfg.py uebernommen, Task 15.3)
# =============================================================================
def patch_line(text: str, key: str, new_value: str, label: str) -> str:
    """
    Ersetzt GENAU EINE Zeile '<indent><key>: <alt><rest>' durch
    '<indent><key>: <neu><rest>'. Einrueckung + Inline-Kommentar bleiben.
    Bricht ab, wenn der Schluessel nicht exakt einmal vorkommt -> kein
    stiller Fehlpatch (verifizierbar via `diff base gepatcht`).
    """
    pattern = re.compile(
        rf'^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*'
        rf'(?P<val>\S+)(?P<rest>[ \t]*(#.*)?)$',
        re.MULTILINE,
    )
    n = len(pattern.findall(text))
    if n != 1:
        raise SystemExit(
            f"[sweep_runner] ABBRUCH: Schluessel '{key}' {n}x gefunden "
            f"(erwartet: genau 1). Patch abgelehnt -- Config-Struktur pruefen."
        )

    def _repl(m: "re.Match") -> str:
        return f"{m.group('indent')}{key}: {new_value}{m.group('rest')}"

    print(f"[sweep_runner] {label:22s} {key} -> {new_value}")
    return pattern.sub(_repl, text)


# =============================================================================
# Prozess-Ausfuehrung (tmux-safe: blockierend, Live-Echo + tee ins Logfile)
# =============================================================================
def run_and_tee(cmd: str, log_path: Path, env: dict, dry: bool) -> int:
    """Fuehrt cmd via bash aus, streamt stdout live UND in log_path (tee)."""
    print(f"\n[sweep_runner] $ {cmd}")
    print(f"[sweep_runner]   log -> {log_path}")
    if dry:
        print("[sweep_runner]   (dry-run, nicht ausgefuehrt)")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode


# =============================================================================
# Harvest: run_summary.json (Training) + inference_log.json (Latent-Metriken)
# =============================================================================
def harvest(summary_path: Path, infer_log: Path, fullval_path: Path):
    """Liest die JSONs eines Laufs und liefert das CSV-Zeilen-Dict."""
    row = {}

    # --- Training-Summary (Task-16a.0-Hook in train_linux.py) ---
    with open(summary_path) as f:
        s = json.load(f)
    row["best_miou"]     = round(s.get("best_miou", float("nan")), 4)
    row["best_epoch"]    = s.get("best_epoch_miou", -1)
    row["best_val_loss"] = round(s.get("best_val_loss", float("nan")), 6)
    row["epochs_run"]    = s.get("epochs_run", -1)
    row["stop_reason"]   = s.get("stop_reason", "?")

    # --- Latent-Metriken (inference.py, erste N Val-Samples) ---
    #     std-Ratio = mean(pred_std) / mean(real_std)  -- exakt die 15.3-Rechnung.
    with open(infer_log) as f:
        inf = json.load(f)
    recs = inf["records"]
    pred_std = sum(r["pred_std"] for r in recs) / len(recs)
    real_std = sum(r["real_std"] for r in recs) / len(recs)
    row["std_ratio"] = round(pred_std / real_std, 4)
    row["pred_std"]  = round(pred_std, 4)
    row["real_std"]  = round(real_std, 4)
    row["mse"]       = round(inf["summary"]["mean_mse"], 5)
    row["cossim"]    = round(inf["summary"]["mean_cosine_sim"], 4)

    # --- optional Full-Val-mIoU (eval_full_val.py) ---
    if fullval_path.exists():
        with open(fullval_path) as f:
            row["fullval_miou"] = json.load(f).get("mIoU")
    else:
        row["fullval_miou"] = ""

    return row


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 16a.0 Lambda-Sweep-Runner (Weg 1, feste Werteliste).")
    ap.add_argument("--regler", required=True, choices=sorted(PATCHABLE_REGLER),
                    help="Zu sweepender Loss-Term (genau EINER pro Sweep).")
    ap.add_argument("--values", required=True, nargs="+",
                    help="Werteliste, z.B. 0 0.05 0.1 0.3 1.0 (als Strings uebernommen).")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Sweep-Basis-Config.")
    ap.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT,
                    help="Wurzel fuer Configs/Checkpoints/Logs/Predictions/CSV.")
    ap.add_argument("--n-infer", type=int, default=300,
                    help="Val-Samples fuer die post-hoc Latent-Metriken (15.3: 300).")
    ap.add_argument("--full-val", action="store_true",
                    help="Zusaetzlich eval_full_val.py (Full-Val-mIoU, headline).")
    ap.add_argument("--force", action="store_true",
                    help="Vorhandene Outputs ignorieren und alles neu rechnen.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Nur Configs schreiben + Kommandos zeigen, nichts starten.")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"[sweep_runner] Basis-Config fehlt: {base.resolve()}")
    base_text = base.read_text()

    root      = Path(args.sweep_root)
    cfg_dir   = root / "configs"
    log_dir   = root / "logs"
    csv_path  = root / f"sweep_{args.regler}.csv"
    for d in (cfg_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")   # Logs laufen durch tee

    fields = ["regler", "value", "best_miou", "best_epoch", "std_ratio",
              "pred_std", "real_std", "mse", "cossim", "best_val_loss",
              "epochs_run", "stop_reason", "fullval_miou", "ckpt_dir"]
    rows = []

    print(f"[sweep_runner] Regler={args.regler}  Werte={args.values}")
    print(f"[sweep_runner] Basis={base}  Wurzel={root}")
    print(f"[sweep_runner] CSV  ={csv_path}")
    if not args.dry_run and "TMUX" not in os.environ:
        print("[sweep_runner] WARNUNG: nicht in tmux -- SSH-Abbruch killt den Sweep.")

    for L in args.values:
        tag       = f"{args.regler}_{L}"
        ckpt_dir  = root / tag
        pred_dir  = root / "predictions" / tag
        cfg_path  = cfg_dir / f"config_{tag}.yaml"
        log_path  = log_dir / f"{tag}.log"
        best_ckpt = ckpt_dir / f"phase{2 if PHASE == 'cell' else 1}" / "best_miou.pt"
        summary   = ckpt_dir / f"phase{2 if PHASE == 'cell' else 1}" / "run_summary.json"
        infer_log = pred_dir / "inference_log.json"
        fullval   = pred_dir / "fullval.json"

        print(f"\n{'='*72}\n[sweep_runner] === {tag} ===\n{'='*72}")

        # --- 1. Config patchen (nur <regler> + dir; Rest byte-identisch) ---
        text = patch_line(base_text, args.regler, L, "Sweep-Variable:")
        text = patch_line(text, "dir", f'"{ckpt_dir}"', "Checkpoint-Verz.:")
        cfg_path.write_text(text)
        print(f"[sweep_runner] Config -> {cfg_path}  "
              f"(Kontrolle: diff {base.name} {cfg_path.name} = nur 2 Zeilen)")

        cfg_q = shlex.quote(str(cfg_path))

        # --- 2. Training (Skip, wenn best_miou.pt + run_summary.json da) ---
        if args.force or not (best_ckpt.exists() and summary.exists()):
            env["WANDB_NAME"] = f"t16a_{tag}"
            rc = run_and_tee(
                f"python -u train_linux.py --config {cfg_q} --phase {PHASE}",
                log_path, env, args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"[sweep_runner] TRAINING fehlgeschlagen (rc={rc}) -> {tag} uebersprungen.")
                rows.append({"regler": args.regler, "value": L,
                             "stop_reason": f"TRAIN_FAIL_rc{rc}", "ckpt_dir": str(ckpt_dir)})
                continue
        else:
            print(f"[sweep_runner] Training SKIP (best_miou.pt + run_summary.json vorhanden).")

        # --- 3. Inference post-hoc auf best_miou.pt (Latent-Metriken) ---
        if args.force or not infer_log.exists():
            rc = run_and_tee(
                f"python -u inference.py --config {cfg_q} --phase {PHASE} "
                f"--checkpoint {shlex.quote(str(best_ckpt))} "
                f"--output {shlex.quote(str(pred_dir))} --max_samples {args.n_infer}",
                log_dir / f"{tag}_infer.log", env, args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"[sweep_runner] INFERENCE fehlgeschlagen (rc={rc}).")
        else:
            print(f"[sweep_runner] Inference SKIP (inference_log.json vorhanden).")

        # --- 4. optional Full-Val-mIoU ---
        if args.full_val and (args.force or not fullval.exists()):
            run_and_tee(
                f"python -u eval_full_val.py --config {cfg_q} --phase {PHASE} "
                f"--checkpoint {shlex.quote(str(best_ckpt))} "
                f"--output {shlex.quote(str(fullval))}",
                log_dir / f"{tag}_fullval.log", env, args.dry_run)
        elif args.full_val:
            print(f"[sweep_runner] Full-Val SKIP (fullval.json vorhanden).")

        # --- 5. Harvest -> CSV-Zeile ---
        if args.dry_run:
            continue
        try:
            row = harvest(summary, infer_log, fullval)
            row.update({"regler": args.regler, "value": L, "ckpt_dir": str(ckpt_dir)})
            rows.append(row)
            print(f"[sweep_runner] {tag}: mIoU={row['best_miou']} "
                  f"std-Ratio={row['std_ratio']} (E{row['best_epoch']}, {row['stop_reason']})")
        except (FileNotFoundError, KeyError, ZeroDivisionError) as e:
            print(f"[sweep_runner] HARVEST fehlgeschlagen fuer {tag}: {e}")
            rows.append({"regler": args.regler, "value": L,
                         "stop_reason": f"HARVEST_FAIL", "ckpt_dir": str(ckpt_dir)})

    # --- CSV schreiben (immer komplett, aus allen gesammelten Zeilen) ---
    if rows and not args.dry_run:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\n[sweep_runner] CSV geschrieben: {csv_path}  ({len(rows)} Zeilen)")
    print("[sweep_runner] fertig.")


if __name__ == "__main__":
    main()
