#!/usr/bin/env python3
# =============================================================================
# harvest_sweep.py  —  Task 16b.3c (Cluster-Sweep, CSV-Ernte)
# =============================================================================
#
# Sammelt die Ergebnisse EINES Sweeps (alle Werte eines Reglers) in EINE CSV.
# Liest je Wert:
#   <sweep-root>/<regler>_<L>/phase2/run_summary.json   (Training, 16a.0-Hook)
#   <pred-root>/<regler>_<L>/inference_log.json         (Latent-Metriken)
#   <pred-root>/<regler>_<L>/fullval.json  (optional)   (Full-Val-mIoU)
#
# Die harvest()-Logik ist 1:1 aus sweep_runner.py uebernommen -> die Cluster-
# CSV ist spaltenidentisch zu den lokalen 16a-Sweeps (direkt vergleichbar).
# std-Ratio = mean(pred_std)/mean(real_std), exakt die 15.3-Rechnung.
#
# KEIN COMPUTE: reine JSON-Ernte. Kann als Jobende-Schritt im sbatch laufen
# ODER lokal nach `rsync` der Cluster-Ergebnisse.
#
# USAGE (Konvention aus sbatch_sweep_parallel.sh):
#   python harvest_sweep.py --regler lambda_grad --values 0 0.05 0.1 0.3 1.0 \
#     --sweep-root $HOME/Code_final/checkpoints/task16b/lambda_grad_sweep \
#     --pred-root  $HOME/Code_final/predictions/task16b/lambda_grad_sweep \
#     --out        $HOME/Code_final/predictions/task16b/sweep_lambda_grad.csv
#
#   # Werte automatisch aus den vorhandenen Ordnern ableiten (--values weglassen):
#   python harvest_sweep.py --regler lambda_grad --sweep-root ... --pred-root ... --out ...
# =============================================================================

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = ["regler", "value", "best_miou", "best_epoch", "std_ratio",
          "pred_std", "real_std", "mse", "cossim", "best_val_loss",
          "epochs_run", "stop_reason", "fullval_miou", "ckpt_dir"]


def harvest(summary_path: Path, infer_log: Path, fullval_path: Path) -> dict:
    """Liest die JSONs EINES Laufs -> CSV-Zeilen-Dict. 1:1 aus sweep_runner.py."""
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


def discover_values(sweep_root: Path, regler: str) -> list:
    """Leitet die Werteliste aus vorhandenen <regler>_<L>-Ordnern ab (sortiert)."""
    pat = re.compile(rf"^{re.escape(regler)}_(.+)$")
    vals = []
    for d in sweep_root.iterdir() if sweep_root.exists() else []:
        m = pat.match(d.name)
        if d.is_dir() and m:
            vals.append(m.group(1))
    # numerisch sortieren, wo moeglich (0 < 0.05 < 0.1 < 0.3 < 1.0)
    def _key(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)
    return sorted(vals, key=_key)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 16b.3c -- CSV-Ernte fuer EINEN Cluster-Sweep.")
    ap.add_argument("--regler", required=True,
                    help="Gesweepter Loss-Term (fuer Ordner-Muster + CSV-Spalte).")
    ap.add_argument("--values", nargs="+", default=None,
                    help="Werteliste. Weggelassen -> aus <sweep-root> abgeleitet.")
    ap.add_argument("--sweep-root", required=True,
                    help="Checkpoint-Wurzel des Sweeps (<root>/<regler>_<L>/phase2/...).")
    ap.add_argument("--pred-root", required=True,
                    help="Prediction-Wurzel des Sweeps (<root>/<regler>_<L>/inference_log.json).")
    ap.add_argument("--out", required=True, help="Ziel-CSV.")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    pred_root  = Path(args.pred_root)

    values = args.values or discover_values(sweep_root, args.regler)
    if not values:
        raise SystemExit(
            f"[harvest] Keine Werte gefunden unter {sweep_root} fuer Regler "
            f"'{args.regler}'. --values explizit angeben?")

    print(f"[harvest] Regler={args.regler}  Werte={values}")
    print(f"[harvest] sweep-root={sweep_root}")
    print(f"[harvest] pred-root ={pred_root}\n")

    rows = []
    for L in values:
        ckpt_dir = sweep_root / f"{args.regler}_{L}"
        summary  = ckpt_dir / "phase2" / "run_summary.json"
        pred_dir = pred_root / f"{args.regler}_{L}"
        infer    = pred_dir / "inference_log.json"
        fullval  = pred_dir / "fullval.json"

        base = {"regler": args.regler, "value": L, "ckpt_dir": str(ckpt_dir)}
        try:
            row = harvest(summary, infer, fullval)
            row.update(base)
            rows.append(row)
            print(f"[harvest] {args.regler}={L}: mIoU={row['best_miou']} "
                  f"std-Ratio={row['std_ratio']} "
                  f"(E{row['best_epoch']}, {row['stop_reason']})")
        except (FileNotFoundError, KeyError, ZeroDivisionError) as e:
            base["stop_reason"] = f"HARVEST_FAIL: {type(e).__name__}"
            rows.append(base)
            print(f"[harvest] {args.regler}={L}: FEHLT/UNVOLLSTAENDIG ({e})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n[harvest] CSV geschrieben: {out}  ({len(rows)} Zeilen)")


if __name__ == "__main__":
    main()
