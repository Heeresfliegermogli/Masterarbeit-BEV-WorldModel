# TASK15.3_ABSCHLUSSBERICHT — Sweep `lambda_std` (Kern-Regler)

**Datum:** 2026-07-12
**Status:** ABGESCHLOSSEN ✅
**Ebene 2 (Loss-Engineering), Regler-Sweep — kontrollierte Dosis-Wirkung**

---

## Ziel

Den Effekt des Channel-Std-Terms (`lambda_std`) als **eine** unabhängige Variable
messen: schließt er den in 15.2 dokumentierten **std-Gap** (pred_std < real_std,
Regression-zur-Mitte unter MSE-Loss), und was kostet das auf der Zielmetrik
**mIoU**? Aufgesetzt als Dosis-Wirkungs-Studie mit einem λ=0-Anker (Term aus) und
einem bewusst hohen Deckenpunkt, um einen etwaigen Über-Regularisierungs-Onset
sichtbar zu machen.

---

## Vorgehen

Adaptiver **zweistufiger** Scan statt fixem Gitter (Beschluss vor Beginn; eine
allgemeine Joint-Optimierung ist ohnehin nach 15.7 als Task 15.8 angesetzt):

1. **Coarse-Scan** grob log-verteilt: `0.0 · 0.1 · 0.3 · 1.0` — Kurvenform grob
   lokalisieren, inkl. λ=0 (Anker) und eines hohen Punktes.
2. **Verfeinerung nach oben:** Da der Coarse-Scan in [0, 1.0] weder einen
   mIoU-Knick noch eine std-Ratio-Sättigung zeigte, wurde die Decke sukzessive
   hochgezogen (`3.0`, dann `10.0`), bis Sättigung **und** Trade-off einsetzten.

Alle Läufe aus der **einen** `config_nuscenes_task15.yaml` abgeleitet (Helper
`mk_lambda_cfg.py`: patcht ausschließlich `training.lambda_std` +
`checkpoints.dir`, Seed fix auf 42, restliche fünf λ unberührt; per `diff`
verifiziert). λ=0.1 ist der Seed-42-Baseline-Lauf aus 15.2 (wiederverwendet, nicht
neu gefahren). Alle Läufe seriell (TITAN RTX), tmux.

**Metriken aus drei Quellen:**
- **std-Ratio, pred_std, MSE, CosSim (Latent):** post-hoc `inference.py`, erste 300
  Val-Samples je `best_miou.pt` — vergleichbar zum 15.2-Anker.
- **mIoU (Auswahl-Proxy):** In-Training-Decode, 300er-Subset (seed 42) — steuert
  `best_miou.pt`, dient **nicht** als Berichtszahl.
- **mIoU (Hauptmetrik, Full-Val):** neues `eval_full_val.py` über **alle 5743**
  Val-Sequenzen je `best_miou.pt`. Prinzip „auf 300 selektieren, auf Full-Val
  berichten". Sanity-Check: `eval_full_val.py --n_subset 300` reproduziert den
  Trainings-Proxy → Decode-Kette bit-korrekt herausgelöst.

---

## Ergebnisse — Latent-Metriken (`inference.py`, 300 fixe Val-Samples)

| λ_std | std-Ratio | pred_std | MSE | CosSim |
|-------|:---:|:---:|:---:|:---:|
| 0.0 (Term aus) | 0.8120 | 0.2152 | 0.02790 | 0.8537 |
| 0.1 (Anker 15.2) | ~0.8305 | ~0.221 | — | — |
| 0.3 | 0.8615 | 0.2283 | 0.02792 | 0.8550 |
| 1.0 | 0.9150 | 0.2424 | 0.02815 | 0.8570 |
| 3.0 | 0.9625 | 0.2549 | 0.02966 | 0.8537 |
| 10.0 | 0.9909 | 0.2624 | 0.03203 | 0.8457 |

`real_std` konstant 0.2649 (Dateneigenschaft). std-Ratio steigt **monoton** von
0.812 auf 0.991; der std-Gap schließt sich von 18.8 % auf 0.9 %. Jeder Schritt liegt
weit über der 15.2-Schwelle für std-Ratio (0.009) → **signifikant über den ganzen
Bereich**. Der gesamte Effekt ist pred_std, das Richtung real_std klettert — exakt
der konstruierte Mechanismus.

---

## Ergebnisse — mIoU Full-Val (`eval_full_val.py`, alle 5743 Val-Sequenzen)

| λ_std | mIoU **Full-Val** | mIoU (300-Proxy) | best@Epoche | best Val-Loss |
|-------|:---:|:---:|:---:|:---:|
| 0.0 (Term aus) | 0.6842 | 0.6801 | 34 | 0.04973 |
| 0.1 (Anker) | 0.6836 | 0.6798 | 34 | — |
| 0.3 | 0.6864 | 0.6827 | 34 | 0.05046 |
| 1.0 | 0.6900 | 0.6840 | 39 | 0.05105 |
| 3.0 | **0.6916** | **0.6896** | 44 | 0.05209 |
| 10.0 | 0.6833 | 0.6792 | 44 | 0.05483 |

Form: **monoton steigend bis λ=3.0, Abfall bei λ=10.0.** Peak bei 3.0, Knick bei
10.0. Die Full-Val-Messung reproduziert die Proxy-Reihenfolge exakt und **ohne**
eigene Kalibrierung — bestätigt zugleich die Korrektheit von `eval_full_val.py`.
Der Val-Loss steigt erwartungsgemäß monoton mit λ (der Nicht-MSE-Term handelt
gegen reinen MSE); die mIoU folgt dem gerade **nicht**, sondern hat ihr Maximum
bei moderater Dosis.

**Per-Klasse (Full-Val, Eckpunkte):**

| Klasse | 0.0 | 3.0 | 10.0 |
|--------|:---:|:---:|:---:|
| drivable_area | 0.8921 | **0.8975** | 0.8941 |
| ped_crossing | 0.6174 | **0.6207** | 0.6131 |
| walkway | 0.7249 | **0.7339** | 0.7269 |
| stop_line | 0.5361 | **0.5456** | 0.5360 |
| carpark_area | 0.6772 | **0.6829** | 0.6742 |
| divider | 0.6309 | **0.6400** | 0.6251 |

Bei λ=3.0 liegt jede Klasse über „Term aus", am stärksten die dünnen/schwierigen
(stop_line, divider, walkway). Bei λ=10.0 fällt jede Klasse unter den 3.0-Peak,
vier von sechs unter die Baseline. Richtungskonsistent — Signifikanz siehe unten.

---

## Full-Val-Rauschboden (Rework, seed 42/43/44)

Die 300er-Schwelle aus 15.2 (0.014) gilt für Full-Val **nicht**. Daher die drei
Seed-Checkpoints (baseline-äquivalent, λ=0.1) per `eval_full_val.py` neu auf
Full-Val gemessen:

| Seed | mIoU (Full-Val) |
|------|:---:|
| 42 (= λ=0.1-Anker) | 0.6836 |
| 43 | 0.6876 |
| 44 | 0.6952 |

Mittel **0.6888**, **volle Spannweite 0.0116** (σ ≈ 0.0059). Konservativ (volle
Spanne, nicht 1σ) bei n=3, wie in 15.2.

**Wichtige Erkenntnis:** Full-Val zog die Schwelle **nicht** enger (erwartet war
~0.004). Der Seed-Spread wird nicht vom Subset-Sampling dominiert — das eliminiert
Full-Val —, sondern von **Seed-zu-Seed-Trainingsvarianz** (Init, Datenreihenfolge →
anderes konvergiertes Modell). Diese Komponente ist sample-count-invariant.
Full-Val macht die Zahl belastbarer, nicht rauschärmer im Seed-Sinn.

**Signifikanzschwelle 15.3 (Full-Val-mIoU): |Δ mIoU| > 0.0116.**

---

## Signifikanz-Verdikt

Referenz „Term aus" (λ=0.0 = 0.6842):

| λ_std | mIoU FV | Δ vs 0.0 | signifikant (>0.0116)? |
|-------|:---:|:---:|:---:|
| 0.1 | 0.6836 | −0.0006 | nein |
| 0.3 | 0.6864 | +0.0022 | nein |
| 1.0 | 0.6900 | +0.0058 | nein |
| 3.0 | 0.6916 | +0.0074 | nein |
| 10.0 | 0.6833 | −0.0009 | nein |

Die **gesamte** mIoU-Spanne des Sweeps (0.6916 − 0.6833 = **0.0083**) ist kleiner als
der Seed-Boden (**0.0116**). Nach dem 15.2-Standard: **kein signifikanter
mIoU-Sweet-Spot.** Der Peak bei 3.0 liegt im Seed-Rauschen. Auch per Klasse liegen
die 3.0-Anstiege innerhalb der je eigenen Seed-Böden (Beispiel divider: Boden 0.0163
vs. Sweep-Range 0.0149).

Auf der std-Ratio dagegen ist jeder Schritt hochsignifikant (Boden 0.009,
Schrittweiten 0.03–0.05).

---

## Synthese — die belastbare Aussage von 15.3

Das Ergebnis ist **nicht** „λ=3.0 maximiert die mIoU" (hält den Signifikanztest
nicht), sondern:

> **`lambda_std` schließt den std-Gap massiv und signifikant (std-Ratio 0.81 → 0.99),
> ohne messbare Kosten auf der Segmentierungs-mIoU bis λ ≈ 3.0. Jenseits davon
> (λ = 10.0) setzt Über-Regularisierung ein: die std-Ratio sättigt an 1.0, während
> mIoU, MSE und CosSim gemeinsam zu bröckeln beginnen.**

Die mIoU sagt zum Std-Term „**kein Preis**", nicht „Gewinn". Zwei Signale stützen
das als **konvergente** (nicht signifikante) Evidenz und werden bewusst nicht
überverkauft:
- Die mIoU steigt **monoton** 0.0 → 0.3 → 1.0 → 3.0 und fällt bei 10.0 — vier Punkte
  in exakter Reihenfolge (unter reinem Rauschen ~4 %).
- Diese Ordnung reproduziert sich **unabhängig** über 300er-Proxy und Full-Val und
  fällt zeitlich mit der Latent-Degradation (MSE +8 %, CosSim ↓ bei 10.0) und der
  std-Ratio-Sättigung zusammen. Alle Signale zeigen auf: **3.0 gutartig, 10.0
  über-regularisiert.**

**Empfohlener Arbeitspunkt: λ_std ≈ 1.0–3.0** — „maximale Gap-Schließung ohne
mIoU-Risiko". λ=1.0 ist die konservative Wahl (std-Ratio 0.915, nachweislich
kostenfrei, kleinste Latent-Störung); λ=3.0 holt die Gap-Schließung fast
vollständig (0.963) bei gerade noch unauffälligen Latent-Kosten. λ=10.0 wird
**nicht** empfohlen (über-regularisiert).

Wissenschaftlich sauberer als ein herausgepresstes Optimum: ein signifikanter
Effekt auf der Zielgröße (std-Ratio), ein Null-Kosten-Nachweis auf der Hauptmetrik
(mIoU), ein sichtbarer Über-Regularisierungs-Onset.

---

## Wichtige Beobachtungen

- **std-Gap ist regelbar, aber gutartig.** Der Frozen-Encoder-Aufbau schließt Collapse
  konstruktiv aus; `lambda_std` adressiert die MSE-bedingte Varianz-Unterschätzung
  gezielt und dosisabhängig.
- **Magnituden-Frage geklärt.** Die anfängliche Sorge, roher `loss_std` sei gegen
  MSE=1.0 zu klein, war unbegründet: schon λ=1.0 bewegt die std-Ratio deutlich; das
  Knie liegt erst bei λ=10.0.
- **Trainingsdauer wächst mit λ.** best-mIoU-Epoche wandert 34 → 44; der Std-Term
  verlangsamt die Konvergenz leicht (Early-Stopping fängt das ab).
- **Full-Val ≠ rauschärmer.** Der dominante Rauschterm ist Seed-Trainingsvarianz, nicht
  Subset-Sampling — relevant für alle folgenden Sweeps (15.4–15.6): deren Effekte
  müssen ebenfalls > ~0.012 Full-Val sein, um signifikant zu heißen.
- **`eval_full_val.py` verifiziert.** `--n_subset 300` reproduziert den Trainings-Proxy;
  Full-Val-Ordnung deckt sich mit Proxy-Ordnung.

---

## Deliverables

- `mk_lambda_cfg.py` — Per-Level-Config-Ableitung (nur `lambda_std` + `dir`).
- `eval_full_val.py` — Streaming-Full-Val-mIoU-Wrapper (speichersicher, nutzt die
  Trainings-Decode-Kette; `--checkpoint`/`--output`/`--n_subset`).
- Sweep-Configs `config_task15_3_lstd_{0.0,0.3,1.0,3.0,10.0}.yaml`.
- Checkpoints `checkpoints/task15/std_sweep/lstd_<L>/phase2/`.
- Latent-Metriken `predictions/task15_3/lstd_<L>/inference_log.json`.
- Full-Val-mIoU `predictions/task15_3/fullval/lstd_<L>_fullval.json`.
- Full-Val-Rauschboden `predictions/task15_3/fullval_floor/seed_{42,43,44}_fullval.json`.

---

## Offene Punkte / Ausblick

- **λ=0.1-Anker auf Full-Val** ist mit seed_42 (0.6836) erledigt (offener Hinweis aus
  früheren Sessions geschlossen).
- **Task 15.4 — Sweep `lambda_grad`** (Gradient-/Kanten-Term) als nächster Regler,
  gleiche Apparatur; Signifikanz jetzt gegen den Full-Val-Boden (~0.012).
- **Visualisierung / Task 16:** Feature-Map-/Masken-Rendering (`render_mask` /
  `save_comparison` in `seg_decoder_torch.py`, bereitgestellt, in `validate_seg`
  noch nicht aufgerufen) — bei Aktivierung mit dem Pipeline-Cleanup verzahnen
  (`inference_seg.py` als toter Docker-Pfad entfernen; Eval-Einstiege konsolidieren:
  `inference.py` = Latent, `eval_full_val.py` = Seg-mIoU).
- **Task 15.8 (nach 15.7):** optionale Joint-Optimierung — der hier gefundene
  Arbeitspunkt ist konditional auf den festen Loss-Schnitt (One-at-a-time-Grenze).

---

## Fazit

`lambda_std` ist ein **wirksamer, gutartiger Regulator**: er schließt den
deterministischen std-Gap nahezu vollständig, ohne die Segmentierungsqualität zu
kosten, mit klar lokalisiertem Über-Regularisierungs-Onset bei λ=10.0. Empfohlener
Arbeitspunkt **λ ≈ 1.0–3.0**. Die λ_std-Achse ist vollständig ausgemessen; Dosis-
Wirkung, Signifikanzverdikt und per-Klasse-Bild liegen für die 15.7-Synthese bereit.
