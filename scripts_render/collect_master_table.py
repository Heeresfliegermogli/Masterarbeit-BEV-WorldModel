#!/usr/bin/env python3
# =============================================================================
# collect_master_table.py — Master-Ergebnistabelle des Seg-Strangs
# =============================================================================
# Sammelt ALLE Full-Val-Ergebnisse (16b.9 Headline -> 16d/e/f -> 18) plus
# Persistenz-Baselines aus den vorhandenen JSONs (kein Recompute) und schreibt
# predictions/task19/master_table.csv + eine Markdown-Tabelle (stdout).
# Referenz: off = SmoothL1-Minimal 0.6946; Signifikanzschwelle 0.014 (15.2).
# Flow-Sample-Metriken (300er-Subset) als separater Block gekennzeichnet.
# =============================================================================
import csv
import json
from pathlib import Path

P = Path("predictions")
THR = 0.014


def j(path):
    return json.load(open(path))


def verdict(delta):
    if delta is None:
        return "--"
    if delta > THR:
        return "BESSER (sig.)"
    if delta < -THR:
        return "schlechter (sig.)"
    return "n.s."


rows = []  # (gruppe, name, miou, n, params_str, kommentar)

# --- Baselines (modellfrei) ----------------------------------------------
ep = j(P / "task16c/ego_pers.json")["curve"]["1"]
rows.append(("Baseline", "Persistenz naiv (F3 kopieren)", ep["pers_naiv"], 5467, "0", "16c/16d-Vorstufe"))
rows.append(("Baseline", "Persistenz ego-gewarpt",        ep["pers_ego"],  5467, "0", "16d-Vorstufe"))

# --- Headline (16b.9) -----------------------------------------------------
hb = j(P / "task19/headline_baseline_fullval.json")
hm = j(P / "task19/headline_minimal_fullval.json")
rows.append(("Headline", "6-Term-Baseline (mse+5 Regler)", hb["mIoU"], hb["n_samples"], "6.05M", "16b.9"))
rows.append(("Headline", "SmoothL1-Minimal (smooth_l1+std) = OFF", hm["mIoU"], hm["n_samples"], "6.05M", "16b.9 SIEGER"))
OFF = hm["mIoU"]

# --- 16d Ego-Conditioning -------------------------------------------------
for lab, name in [("gate", "Ego gate (Bias auf Gate-Logits)"),
                  ("action", "Ego FiLM action"), ("token", "Ego-Token"),
                  ("state", "Ego FiLM state")]:
    d = j(P / f"task16d/ego_{lab}_fullval.json")
    rows.append(("16d Ego", name, d["mIoU"], d["n_samples"], "6.06-6.19M", ""))

# --- 16e Kontext/EMA ------------------------------------------------------
for lab, name, pstr in [("nf4", "4 Input-Frames", "6.05M"),
                        ("ema_shadow", "EMA (Shadow-Weights)", "6.05M"),
                        ("ema_raw", "EMA-Lauf raw (Konsistenz-Check)", "6.05M")]:
    d = j(P / f"task16e/{lab}_fullval.json")
    rows.append(("16e Kontext/EMA", name, d["mIoU"], d["n_samples"], pstr, ""))

# --- 16f Task-Loss/Kapazitaet/Residual -----------------------------------
for lab, name, pstr in [("tl_01", "Decoder-Task-Loss 0.1 (Finetune)", "6.05M"),
                        ("tl_05", "Decoder-Task-Loss 0.5", "6.05M"),
                        ("tl_10", "Decoder-Task-Loss 1.0", "6.05M"),
                        ("cap_l8", "Kapazität: 8 Layer", "9.21M"),
                        ("residual", "Hartes Residual statt Gate", "6.05M")]:
    d = j(P / f"task16f/{lab}_fullval.json")
    rows.append(("16f Hebel", name, d["mIoU"], d["n_samples"], pstr, ""))

# --- 18 Generativ (Full-Val, z=mean) -------------------------------------
for v in ["0.001", "0.01", "0.1"]:
    d = j(P / f"task18/vae_{v}_fullval.json")
    rows.append(("18 Generativ", f"CVAE beta={v} (z=mean)", d["mIoU"], d["n_samples"], "6.42M", "Guard"))
rows.append(("18 Generativ", "Flow Matching (mean = frozen Backbone)", OFF, hm["n_samples"], "6.05M+6.52M", "per Konstruktion"))

# --- Tabelle bauen --------------------------------------------------------
csv_path = P / "task19/master_table.csv"
csv_path.parent.mkdir(exist_ok=True)
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gruppe", "variante", "miou_fullval", "n", "delta_vs_off", "verdikt", "params", "kommentar"])
    for g, n, m, ns, p, k in rows:
        delta = None if g == "Baseline" else round(m - OFF, 4)
        w.writerow([g, n, m, ns, delta if delta is not None else "", verdict(delta), p, k])

print(f"geschrieben: {csv_path}\n")
print(f"| Gruppe | Variante | mIoU | d vs off | Verdikt |")
print(f"|---|---|---:|---:|---|")
for g, n, m, ns, p, k in rows:
    delta = None if g == "Baseline" else m - OFF
    ds = "--" if delta is None else f"{delta:+.4f}"
    print(f"| {g} | {n} | {m:.4f} | {ds} | {verdict(delta)} |")

# --- Anhang: Flow-Sample-Block (300er-Subset, KEINE Full-Val-Zeilen) ------
print("\nANHANG Flow-Samples (300er-Subset, eigener Massstab):")
for tag, lbl in [("eval_flow_s1_v2", "Flow Sample steps=1")]:
    d = j(P / f"task18/{tag}.json")
    print(f"  {lbl}: miou_sample={d['miou_sample']} best_of_8={d['miou_best_of_k']} "
          f"div_mask={d['diversity_mask']} std_s={d['std_ratio_sample']} (mean={d['miou_mean']})")
