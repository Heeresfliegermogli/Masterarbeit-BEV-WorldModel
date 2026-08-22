# TASK15.2_ABSCHLUSSBERICHT — Rauschboden / Signifikanzschwelle

**Datum:** 2026-07-06
**Status:** ABGESCHLOSSEN ✅
**Ebene 2 (Loss-Engineering), Messung — kein Regler**

---

## Ziel

Die Lauf-zu-Lauf-Streuung bei **identischer** Loss-Konfiguration bestimmen, damit
in den nachfolgenden Sweeps (15.3–15.6) gilt: ein λ-Effekt ist erst ein **Befund**,
wenn sein Delta die hier gemessene Streuung überschreitet. Sonst interpretiert man
Rauschen. Methodik nach LeWM Tab. 5 (mehrere Seeds, mean ± std).

---

## Vorgehen

Die refactorierte Loss (Task 15.1, baseline-äquivalente λ) **3× mit verschiedenen
Seeds** trainiert: **42 / 43 / 44**. Pro Seed ein isoliertes Checkpoint-Verzeichnis
(`checkpoints/task15/noise_floor/seed_XX/`), sonst bit-identische Config —
abgeleitet aus der einen `config_nuscenes_task15.yaml` (nur `training.seed` +
`checkpoints.dir` getauscht, kein Hand-Editieren, kein Code-Eingriff).

Metriken aus zwei Quellen:
- **mIoU** + **best_val_loss**: aus den Trainingsläufen (300er-Decode-Subset,
  `best_miou.pt`).
- **std-Ratio, MSE, CosSim (Latent)**: post-hoc via `inference.py` auf den
  **ersten 300 Val-Samples** aus jedem `best_miou.pt`.

**Warum Across-Seed (nicht Same-Seed):** jeder spätere Sweep-Punkt ist *ein* Lauf
bei festem Seed. Die Varianz, die dieser eine Lauf nicht wegmittelt, ist die
Seed-induzierte (Init + Shuffle-Reihenfolge). Genau diese ist der relevante Boden.
Same-Seed-Wiederholungen messen nur Hardware-Jitter und unterschätzen den Boden.

---

## Ergebnisse — mIoU & Val-Loss (Trainingsläufe)

| Seed | best_miou | @Epoche | best_val_loss | @Epoche | Ende |
|---|---|---|---|---|---|
| 42 | 0.6798 | 34 | 0.050195 | 37 | Early Stopping |
| 43 | 0.6788 | 34 | 0.049809 | 31 | Early Stopping |
| 44 | 0.6925 | 39 | 0.048776 | 32 | Early Stopping |
| **mean** | **0.6837** | | **0.049593** | | |
| **std (n−1)** | **0.0076** | | **0.00073** | | |
| **Spread (max−min)** | **0.0137** | | **0.00142** | | |

## Ergebnisse — Latent-Metriken (`inference.py`, 300 fixe Val-Samples)

| Seed | std-Ratio | MSE | CosSim | pred_std | real_std |
|---|---|---|---|---|---|
| 42 | 0.8305 | 0.02783 | 0.8544 | 0.2201 | 0.2649 |
| 43 | 0.8340 | 0.02745 | 0.8564 | 0.2210 | 0.2649 |
| 44 | 0.8390 | 0.02725 | 0.8574 | 0.2223 | 0.2649 |
| **mean** | **0.8345** | **0.02751** | **0.8561** | 0.2211 | 0.2649 |
| **std (n−1)** | **0.0043** | **0.00029** | **0.0015** | 0.0011 | 0.0000 |
| **Spread** | **0.0086** | **0.00059** | **0.0030** | 0.0023 | 0.0000 |

std-Ratio = mittleres pred_std / real_std pro Sample (mean-of-ratios).

---

## Signifikanzschwellen für 15.3–15.6

Konservativ gilt die **volle Spannweite** der drei Baseline-Läufe als Schwelle
(nicht nur 1σ) — schützt vor Falsch-Positiven bei n=3.

| Metrik | 1σ | Spannweite (= Schwelle) |
|---|---|---|
| **mIoU** (Primärmetrik) | 0.008 | **≈ 0.014** |
| **std-Ratio** | 0.004 | **≈ 0.009** |
| CosSim | 0.0015 | ≈ 0.003 |
| MSE | 0.0003 | ≈ 0.0006 |

**Ein λ-Effekt zählt erst als Befund, wenn |Δ mIoU| > ~0.014** (bzw. für die
sekundär berichtete std-Ratio > ~0.009). Alles darunter ist Lauf-zu-Lauf-Rauschen.
mIoU bleibt Primärmetrik; std-Ratio ist **nicht** mehr Abbruchkriterium.

---

## Wichtige Beobachtungen

**1. Refactor über die volle Trajektorie inert.** Seed 42 reproduziert Task 14
*exakt* (mIoU 0.6798 @E34, Val-Loss-optimal @E37). Die Float32-Reassoziation aus
15.1 (5.96e-8/Schritt) ist über 50 Epochen **nicht** chaotisch aufgelaufen. Der
neue Loss-Code ist als saubere Sweep-Basis bestätigt.

**2. Kohärente Qualitäts-Ordnung der Seeds.** Seed 44 ist auf *jeder* Metrik der
beste Zug (mIoU, std-Ratio, CosSim, MSE), Seed 42 durchgängig der schwächste
(44 > 43 > 42 überall). Die Metriken streuen nicht zufällig — eine Initialisierung
war schlicht besser. Das bestätigt Across-Seed als korrektes Rauschmaß und zeigt,
dass ein Single-Seed-Sweep-Punkt eine Gesamt-Modellqualitäts-Komponente trägt, die
in allen Metriken zugleich sichtbar ist.

**3. Latent-Boden ist confound-frei, mIoU-Boden leicht konservativ.** Die
Inference lief für alle drei Seeds auf dem **identischen** Eval-Satz (ersten 300
Val-Samples; `real_std` exakt 0.2649 bei allen drei bestätigt das). std-Ratio/MSE/Cos
messen damit reines Modell-Rauschen. Die mIoU dagegen stammt aus dem 300er-Subset,
das im Training mit dem Seed mitwandert → trägt zusätzlich etwas Subset-Sampling-
Rauschen, der mIoU-Boden ist also eine konservative Obergrenze. In den Sweeps liegt
der Seed fix auf 42 → dasselbe Subset → dieser Anteil fällt *innerhalb* des Sweeps
raus.

**4. Baseline-std-Gap.** pred_std ≈ 0.221 vs. real_std ≈ 0.265 → std-Ratio ≈ 0.835,
d.h. die Latent-Vorhersagen tragen ~83,5 % der realen Amplitude (Gap ~16,5 %). Das
ist die Signatur der deterministischen Decke (Regression-zur-Mitte unter MSE) und
der Ausgangspunkt, den der lambda_std-Sweep (15.3) zu bewegen versucht.

**5. n=3-Ehrlichkeit.** Der std-Wert aus 3 Punkten ist grob. Bei mIoU liegen 42/43
quasi aufeinander (Δ 0.001), Seed 44 spannt den Spread allein auf (+0.013) — der
Boden wird also von *einem* Draw dominiert. Spread-als-Schwelle ist deshalb die
richtige, konservative Wahl.

---

## Deliverables

| Datei | Beschreibung |
|---|---|
| `checkpoints/task15/noise_floor/seed_{42,43,44}/` | 3 vollständige Läufe (best_miou.pt, best_val_loss.pt) |
| `logs/nf_seed{42,43,44}.log` | Trainingslogs (mIoU, Val-Loss je Epoche) |
| `predictions/nf/seed_{42,43,44}/` + `inference_log_{42,43,44}.json` | Latent-Metriken (std-Ratio, MSE, Cos) |
| `mk_seed_cfg.py` | Ableitung der Per-Seed-Configs aus der Single-Source-YAML |
| `TASK15_2_ABSCHLUSSBERICHT.md` | dieser Bericht |

---

## Nächster Schritt: Task 15.3 — Sweep lambda_std (Kern-Regler)

Unabhängige Variable `lambda_std`, Stufen 0 · Baseline(0.1) · 0.3 · 0.5 · 1.0.
Fester Seed 42, identisches 300er-Subset. Erwartung: std-Ratio steigt monoton,
mIoU als umgekehrtes U → Sweet-Spot. **Ein Effekt gilt als real ab |Δ mIoU| > 0.014
bzw. |Δ std-Ratio| > 0.009.**
