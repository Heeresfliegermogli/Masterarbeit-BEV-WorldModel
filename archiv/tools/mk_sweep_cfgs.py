#!/usr/bin/env python3
# =============================================================================
# mk_sweep_cfgs.py  —  Task 16b.2 (Cluster-Sweep, Config-Vorab-Generierung)
# =============================================================================
#
# Erzeugt VORAB je Sweep-Wert eine Per-Lambda-Config aus der EINEN Cluster-
# Basis-Config. Aendert pro Config AUSSCHLIESSLICH ZWEI Zeilen:
#
#     training.<regler>   ->  der Sweep-Wert L (unabhaengige Variable)
#     checkpoints.dir     ->  isoliertes Verzeichnis <sweep-root>/<regler>_<L>
#
# Alles andere bleibt BYTE-IDENTISCH (Seed, die uebrigen 5 Lambdas, Daten,
# Split, Architektur, Staging-Schalter). Das ist die Sweep-Disziplin aus dem
# Arbeitsplan: eine Variable pro Lauf, Rest eingefroren.
#
# VERALLGEMEINERT mk_lambda_cfg.py (nur lambda_std, ein Wert) auf JEDEN Regler
# und eine ganze Werteliste -- fuehrt aber NICHTS aus (nur Configs schreiben).
# Das Starten macht sbatch_sweep_parallel.sh. Diese Trennung erlaubt die vom
# Arbeitsplan geforderte diff-Verifikation VOR dem Compute.
#
# WARUM ZEILEN-REGEX STATT YAML-LOAD/DUMP (1:1 wie mk_lambda_cfg.py):
#   PyYAML wuerde reformatieren und Kommentare verlieren. Hier zeilenbasiert,
#   whitespace-flexibel, mit Exact-1-Match-Guard -> kein stiller Fehlpatch,
#   verifizierbar via `diff <basis> <config>` (darf NUR 2 Zeilen zeigen).
#
# NAMENS-KONVENTION (MUSS mit sbatch_sweep_parallel.sh uebereinstimmen):
#   Config-Datei:     <out-dir>/config_<regler>_<L>.yaml
#   checkpoints.dir:  <sweep-root>/<regler>_<L>
#   -> best_miou.pt:  <sweep-root>/<regler>_<L>/phase2/best_miou.pt
#
# USAGE (i.d.R. aus dem sbatch heraus aufgerufen, nicht von Hand):
#   python mk_sweep_cfgs.py --regler lambda_grad --values 0 0.05 0.1 0.3 1.0 \
#          --base config_sweep_base_cluster_fp16.yaml \
#          --sweep-root $HOME/Code_final/checkpoints/task16b/lambda_grad_sweep \
#          --out-dir    $HOME/Code_final/configs/task16b/lambda_grad_sweep
#
#   # Trockenlauf: nur zeigen, was gepatcht wuerde, nichts schreiben
#   python mk_sweep_cfgs.py --regler lambda_grad --values 0 0.05 --dry-run ...
# =============================================================================

import argparse
import re
import sys
from pathlib import Path

# Die 6 patchbaren Loss-Terme (15.1-Schema) + recon_loss (16b.7, kategorial:
# mse|l1|smooth_l1, kein Lambda -- patch_line setzt einfach den String-Wert).
PATCHABLE_REGLER = {"lambda_mse", "lambda_cos", "lambda_mean",
                    "lambda_std", "lambda_grad", "lambda_ssim",
                    "recon_loss"}


def patch_line(text: str, key: str, new_value: str, label: str) -> str:
    """
    Ersetzt GENAU EINE Zeile '<indent><key>: <alt><rest>' durch
    '<indent><key>: <neu><rest>'. Einrueckung + Inline-Kommentar bleiben.
    Bricht ab, wenn der Schluessel nicht exakt einmal vorkommt (kein stiller
    Fehlpatch). 1:1 aus mk_lambda_cfg.py / sweep_runner.py uebernommen.
    """
    pattern = re.compile(
        rf'^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*'
        rf'(?P<val>\S+)(?P<rest>[ \t]*(#.*)?)$',
        re.MULTILINE,
    )
    n = len(pattern.findall(text))
    if n != 1:
        raise SystemExit(
            f"[mk_sweep_cfgs] ABBRUCH: Schluessel '{key}' {n}x gefunden "
            f"(erwartet: genau 1). Patch abgelehnt -- Config-Struktur pruefen."
        )

    def _repl(m: "re.Match") -> str:
        return f"{m.group('indent')}{key}: {new_value}{m.group('rest')}"

    print(f"[mk_sweep_cfgs] {label:20s} {key} -> {new_value}")
    return pattern.sub(_repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 16b.2 -- Per-Lambda-Configs fuer einen Cluster-Sweep erzeugen.")
    ap.add_argument("--regler", required=True, choices=sorted(PATCHABLE_REGLER),
                    help="Zu sweepender Loss-Term (genau EINER pro Sweep).")
    ap.add_argument("--values", required=True, nargs="+",
                    help="Werteliste, z.B. 0 0.05 0.1 0.3 1.0 (als Strings uebernommen).")
    ap.add_argument("--base", required=True,
                    help="Cluster-Sweep-Basis-Config (config_sweep_base_cluster_fp16.yaml).")
    ap.add_argument("--sweep-root", required=True,
                    help="Wurzel fuer die checkpoints.dir je Wert (<root>/<regler>_<L>).")
    ap.add_argument("--out-dir", required=True,
                    help="Zielverzeichnis fuer die erzeugten config_<regler>_<L>.yaml.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Nur zeigen, nichts schreiben.")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"[mk_sweep_cfgs] Basis-Config fehlt: {base.resolve()}")
    base_text = base.read_text()

    out_dir    = Path(args.out_dir)
    sweep_root = args.sweep_root.rstrip("/")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mk_sweep_cfgs] Regler={args.regler}  Werte={args.values}")
    print(f"[mk_sweep_cfgs] Basis={base}")
    print(f"[mk_sweep_cfgs] Sweep-Root={sweep_root}")
    print(f"[mk_sweep_cfgs] Out-Dir={out_dir}\n")

    written = []
    for L in args.values:
        ckpt_dir = f"{sweep_root}/{args.regler}_{L}"
        cfg_path = out_dir / f"config_{args.regler}_{L}.yaml"

        # --- die einzigen zwei Aenderungen (Rest byte-identisch) ---
        text = patch_line(base_text, args.regler, L,          "Sweep-Variable:")
        text = patch_line(text,      "dir", f'"{ckpt_dir}"',  "Checkpoint-Verz.:")

        if args.dry_run:
            print(f"[mk_sweep_cfgs]   (dry-run) wuerde schreiben: {cfg_path}\n")
            continue

        cfg_path.write_text(text)
        written.append(cfg_path)
        print(f"[mk_sweep_cfgs]   geschrieben: {cfg_path}")
        print(f"[mk_sweep_cfgs]   Kontrolle:   diff {base.name} {cfg_path.name} "
              f"(darf NUR die 2 gepatchten Zeilen zeigen)\n")

    if not args.dry_run:
        print(f"[mk_sweep_cfgs] fertig: {len(written)} Configs -> {out_dir}")


if __name__ == "__main__":
    main()
