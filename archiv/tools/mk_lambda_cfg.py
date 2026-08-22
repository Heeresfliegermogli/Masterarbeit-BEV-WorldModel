#!/usr/bin/env python3
# =============================================================================
# mk_lambda_cfg.py  —  Task 15.3 (Sweep lambda_std, Kern-Regler)
# =============================================================================
#
# Leitet aus der EINEN Quell-Config `config_nuscenes_task15.yaml` eine
# Per-Level-Config ab. Aendert AUSSCHLIESSLICH zwei Zeilen:
#
#     training.lambda_std   ->  das Sweep-Level L (unabhaengige Variable)
#     checkpoints.dir       ->  isoliertes Verzeichnis pro Level
#
# Alles andere bleibt BYTE-IDENTISCH: Seed 42, die restlichen 5 Lambdas
# (mse/cos/mean/grad/ssim), Daten, Split, Architektur, Optimizer. Genau das
# ist die Sweep-Disziplin aus dem Arbeitsplan (eine Variable pro Lauf).
#
# WARUM ZEILEN-REGEX STATT YAML-LOAD/DUMP:
#   PyYAML wuerde die Datei reformatieren und alle Kommentare verlieren.
#   Hier wird zeilenbasiert und WHITESPACE-FLEXIBEL gepatcht, mit
#   Match-Verifikation: trifft ein Schluessel nicht GENAU einmal, bricht das
#   Skript ab (kein stiller Fehlpatch). Damit ist garantiert, dass sich der
#   Rest der Config nicht bewegt -- verifizierbar via `diff`.
#
# USAGE:
#     python mk_lambda_cfg.py 0.0
#     python mk_lambda_cfg.py 0.3
#     python mk_lambda_cfg.py 1.0
#
# ERZEUGT (im selben Verzeichnis wie die Quell-Config):
#     config_task15_3_lstd_<L>.yaml
#
# Danach z.B.:
#     export WANDB_NAME=t15.3_lstd_0.3
#     python -u train_linux.py --config config_task15_3_lstd_0.3.yaml \
#            --phase cell 2>&1 | tee logs/t15.3_lstd_0.3.log
# =============================================================================

import re
import sys
from pathlib import Path

# --- Feste Konventionen fuer Task 15.3 -------------------------------------
SRC       = Path("config_nuscenes_task15.yaml")
CKPT_ROOT = "/home/vima/Desktop/Masterarbeit/Code_final/checkpoints/task15/std_sweep"


def patch_line(text: str, key: str, new_value: str, label: str) -> str:
    """
    Ersetzt GENAU EINE Zeile '<indent><key>: <alt><rest>' durch
    '<indent><key>: <neu><rest>'. Einrueckung und ein etwaiger
    Inline-Kommentar bleiben erhalten.

    Bricht ab, wenn der Schluessel nicht exakt einmal vorkommt -> kein
    stiller Fehlpatch (z.B. falls sich die Config-Struktur geaendert hat).
    """
    pattern = re.compile(
        rf'^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*'
        rf'(?P<val>\S+)(?P<rest>[ \t]*(#.*)?)$',
        re.MULTILINE,
    )
    n = len(pattern.findall(text))
    if n != 1:
        raise SystemExit(
            f"[mk_lambda_cfg] ABBRUCH: Schluessel '{key}' {n}x gefunden "
            f"(erwartet: genau 1). Patch abgelehnt -- Config-Struktur pruefen."
        )

    def _repl(m: "re.Match") -> str:
        return f"{m.group('indent')}{key}: {new_value}{m.group('rest')}"

    new_text = pattern.sub(_repl, text)
    print(f"[mk_lambda_cfg] {label:22s} {key} -> {new_value}")
    return new_text


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python mk_lambda_cfg.py <lambda_std>   (z.B. 0.0 / 0.3 / 1.0)"
        )

    L   = sys.argv[1].strip()
    tag = f"lstd_{L}"                       # -> lstd_0.3

    if not SRC.exists():
        raise SystemExit(
            f"[mk_lambda_cfg] Quell-Config nicht gefunden: {SRC.resolve()}\n"
            f"                (Skript im Verzeichnis der Config ausfuehren.)"
        )

    text     = SRC.read_text()
    ckpt_dir = f"{CKPT_ROOT}/{tag}"

    # --- die einzigen zwei Aenderungen -------------------------------------
    text = patch_line(text, "lambda_std", L,                "Sweep-Variable:")
    text = patch_line(text, "dir",        f'"{ckpt_dir}"',  "Checkpoint-Verz.:")

    out = SRC.parent / f"config_task15_3_{tag}.yaml"
    out.write_text(text)

    print(f"[mk_lambda_cfg] geschrieben: {out}")
    print(f"[mk_lambda_cfg]   -> Checkpoints landen unter: {ckpt_dir}/phase2/")
    print(f"[mk_lambda_cfg]   -> seed + restliche 5 lambda UNVERAENDERT")
    print(f"[mk_lambda_cfg]   Kontrolle:  diff {SRC.name} {out.name}   "
          f"(darf NUR die 2 gepatchten Zeilen zeigen)")


if __name__ == "__main__":
    main()
