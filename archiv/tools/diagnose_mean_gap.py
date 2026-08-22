#!/usr/bin/env python3
"""
diagnose_mean_gap.py — Task 16a.1 VORAB-DIAGNOSE (mean-Gap)
============================================================

Misst den Rest-mean-Gap am Loss-Arbeitspunkt (lambda_std=1.0, lambda_mean=0.1)
aus einem VORHANDENEN Vorwaertslauf — kein Training, kein GPU. Analog zur
std-Gap-Diagnose aus 15.2 (vgl. RAUSCH_REPORT.md). Ergebnis: Entscheidung
voll / mini / skip fuer den lambda_mean-Sweep (16a.4).

EINGABE:
  inference_log.json aus inference.py, erzeugt am Anker
  checkpoints/task15/std_sweep/lstd_1.0/phase2/best_miou.pt
  (>= ~300 Val-Samples empfohlen; 20 reichen fuer eine grobe Tendenz).

WARUM VORZEICHENSICHER (nicht der naive mean-Ratio):
  BEV-Seg-Latents sind ~channel-zentriert -> mean(real_mean) ~ 0, und
  positive/negative per-Channel-Mittel heben sich beim Skalar-Mittel weg.
  Ein Ratio mean(pred_mean)/mean(real_mean) kippt dort (Division ~0) und
  verschleiert Channel-weise Fehler. Daher wird der Gap ueber den
  per-Channel-MSE gemessen, wo sich nichts wegkuerzt:

    dist_mean (pro Record) = mean_c (pm_c - rm_c)^2      (= r['dist_mean'])
    dist_std  (pro Record) = mean_c (ps_c - rs_c)^2      (= r['dist_std'])

    rms_mean_err = sqrt( mean_records dist_mean )   [Latent-Einheiten]
    rms_std_err  = sqrt( mean_records dist_std  )   [Latent-Einheiten]
    r            = rms_mean_err / rms_std_err        (mean-Rest vs. std-Rest,
                   der std-Rest ist die Groesse, gegen die lambda_std aktiv
                   arbeitet -> r << 1 heisst "mean vergleichsweise geloest")
    rel_mean_gap = rms_mean_err / mean(real_std)     (mean-Rest rel. zum
                   typischen Channel-Spread ~0.26 -> die "3-5%"-Groesse
                   aus dem Arbeitsplan)

Der naive Skalar-mean-Gap wird NUR zur Info mitgefuehrt (mit Caveat).
"""
import argparse
import json
import math
from pathlib import Path


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load(path):
    with open(path) as f:
        d = json.load(f)
    recs = d.get("records", [])
    if not recs:
        raise SystemExit(f"[FEHLER] keine 'records' in {path}")
    return recs, d.get("summary", {})


def diagnose(recs):
    need = ("dist_mean", "dist_std", "pred_mean", "real_mean", "pred_std", "real_std")
    for k in need:
        if not all(k in r for r in recs):
            raise SystemExit(f"[FEHLER] Feld '{k}' fehlt in mindestens einem Record "
                             f"(alte inference.py-Version?).")

    n = len(recs)
    md_mean = _mean(r["dist_mean"] for r in recs)   # = summary.mean_dist_mean
    md_std  = _mean(r["dist_std"]  for r in recs)   # = summary.mean_dist_std
    rms_mean_err = math.sqrt(md_mean)
    rms_std_err  = math.sqrt(md_std)

    pred_mean = _mean(r["pred_mean"] for r in recs)
    real_mean = _mean(r["real_mean"] for r in recs)
    pred_std  = _mean(r["pred_std"]  for r in recs)
    real_std  = _mean(r["real_std"]  for r in recs)

    r_ratio      = rms_mean_err / rms_std_err if rms_std_err else float("inf")
    rel_mean_gap = rms_mean_err / real_std     if real_std     else float("inf")
    std_ratio    = pred_std / real_std         if real_std     else float("inf")
    std_gap_pct  = (1.0 - std_ratio) * 100.0

    # naiv, nur zur Info (Vorzeichen-Ausloeschung!)
    naive_abs = abs(pred_mean - real_mean)
    naive_rel = naive_abs / abs(real_mean) * 100.0 if abs(real_mean) > 1e-9 else float("inf")

    return dict(n=n, md_mean=md_mean, md_std=md_std,
                rms_mean_err=rms_mean_err, rms_std_err=rms_std_err,
                r_ratio=r_ratio, rel_mean_gap=rel_mean_gap,
                pred_mean=pred_mean, real_mean=real_mean,
                pred_std=pred_std, real_std=real_std,
                std_ratio=std_ratio, std_gap_pct=std_gap_pct,
                naive_abs=naive_abs, naive_rel=naive_rel)


def decide(m, rel_skip, rel_mini, r_small):
    rel = m["rel_mean_gap"] * 100.0
    r = m["r_ratio"]
    if rel < rel_skip * 100.0 and r < r_small:
        return ("SKIP -> MINI",
                f"rel_mean_gap {rel:.2f}% < {rel_skip*100:.0f}% UND r {r:.3f} < {r_small}: "
                f"mean am Arbeitspunkt praktisch geloest, minimaler Rest gegenueber dem "
                f"std-Rest. 16a.4 lambda_mean nur als Notwendigkeits-Nachweis: {{0, 0.1}} "
                f"(der 0-Punkt belegt, dass lambda_mean ueberhaupt gebraucht wird), KEIN "
                f"feiner Sweep. Zeitersparnis ~10-15h.")
    if rel < rel_mini * 100.0:
        return ("MINI",
                f"rel_mean_gap {rel:.2f}% im mittleren Band: kleiner, aber nicht "
                f"vernachlaessigbarer Rest. 16a.4 verkuerzt fahren: {{0, 0.1, 0.3}}.")
    return ("VOLL",
            f"rel_mean_gap {rel:.2f}% >= {rel_mini*100:.0f}% (oder r {r:.3f} gross): "
            f"echter mean-Rest vorhanden. 16a.4 mit voller Werteliste fahren "
            f"(z.B. {{0, 0.1, 0.3, 1.0}}), Sweep-Motivation belegt.")


def fmt(m, decision, reason, log_path):
    d, ratio_line = decision, ""
    lines = []
    P = lines.append
    P("=" * 68)
    P("TASK 16a.1 — VORAB-DIAGNOSE mean-Gap")
    P("=" * 68)
    P(f"Quelle : {log_path}")
    P(f"Samples: {m['n']}")
    P("")
    P("  per-Channel-MSE (Loss-Term-Skala, Latent^2):")
    P(f"    dist_mean (mean-Rest) : {m['md_mean']:.6e}")
    P(f"    dist_std  (std-Rest)  : {m['md_std']:.6e}")
    P("")
    P("  RMS per-Channel-Fehler (Latent-Einheiten):")
    P(f"    rms_mean_err          : {m['rms_mean_err']:.5f}")
    P(f"    rms_std_err           : {m['rms_std_err']:.5f}")
    P("")
    P("  >> DIAGNOSE-KENNZAHLEN:")
    P(f"    r = mean-Rest/std-Rest: {m['r_ratio']:.3f}")
    P(f"    rel_mean_gap          : {m['rel_mean_gap']*100:.2f} %   (rms_mean_err / mean(real_std))")
    P("")
    P("  Kontroll-Groessen (Cross-Check vs. 15.3):")
    P(f"    pred_std / real_std   : {m['pred_std']:.4f} / {m['real_std']:.4f}")
    P(f"    std-Ratio             : {m['std_ratio']:.4f}   (15.3-Anker: ~0.915)")
    P(f"    std-Gap               : {m['std_gap_pct']:.2f} %")
    P("")
    P("  Naiver Skalar-mean-Gap (NUR Info — Vorzeichen-Ausloeschung!):")
    P(f"    pred_mean / real_mean : {m['pred_mean']:+.5f} / {m['real_mean']:+.5f}")
    P(f"    |delta| abs / rel     : {m['naive_abs']:.5f} / {m['naive_rel']:.1f} %")
    P("")
    P("-" * 68)
    P(f"  ENTSCHEIDUNG 16a.4 (lambda_mean): {decision}")
    P(f"  Begruendung: {reason}")
    P("=" * 68)
    return "\n".join(lines)


def md_block(m, decision, reason, log_path):
    return f"""### 16a.1 mean-Gap-Diagnose (Kurznotiz)

Vorwaertslauf am Arbeitspunkt (`lambda_std=1.0`, `lambda_mean=0.1`, Rest Baseline),
Anker `checkpoints/task15/std_sweep/lstd_1.0/phase2/best_miou.pt`, {m['n']} Val-Samples.
Vorzeichensicher via per-Channel-MSE gemessen (mean-Ratio kippt bei ~zentrierten Latents).

| Kennzahl | Wert |
|---|---|
| dist_mean (mean-Rest, Latent^2) | {m['md_mean']:.3e} |
| dist_std (std-Rest, Latent^2) | {m['md_std']:.3e} |
| rms_mean_err | {m['rms_mean_err']:.5f} |
| rms_std_err | {m['rms_std_err']:.5f} |
| **r = mean-Rest / std-Rest** | **{m['r_ratio']:.3f}** |
| **rel_mean_gap** | **{m['rel_mean_gap']*100:.2f} %** |
| std-Ratio (Cross-Check, 15.3 ~0.915) | {m['std_ratio']:.4f} |

**Entscheidung 16a.4:** {decision} — {reason}
"""


def main():
    ap = argparse.ArgumentParser(description="Task 16a.1 mean-Gap Vorab-Diagnose")
    ap.add_argument("--log", required=True, help="Pfad zum inference_log.json des Ankers")
    ap.add_argument("--rel-skip", type=float, default=0.03,
                    help="rel_mean_gap-Schwelle fuer SKIP->MINI (Default 0.03 = 3%%)")
    ap.add_argument("--rel-mini", type=float, default=0.08,
                    help="rel_mean_gap-Schwelle fuer MINI vs VOLL (Default 0.08 = 8%%)")
    ap.add_argument("--r-small", type=float, default=0.15,
                    help="Sekundaerkriterium r fuer SKIP (Default 0.15)")
    ap.add_argument("--md", action="store_true",
                    help="zusaetzlich einen fertigen Markdown-Block ausgeben")
    args = ap.parse_args()

    recs, _ = load(args.log)
    m = diagnose(recs)
    decision, reason = decide(m, args.rel_skip, args.rel_mini, args.r_small)
    print(fmt(m, decision, reason, args.log))
    if args.md:
        print()
        print(md_block(m, decision, reason, args.log))


if __name__ == "__main__":
    main()
