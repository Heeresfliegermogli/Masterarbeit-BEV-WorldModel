# TASK 16a.1 — ABSCHLUSSBERICHT

**Titel:** Vorab-Diagnose mean-Gap (billige Inferenz-Messung vor dem lambda_mean-Sweep)
**Datum:** 2026-07-12
**Server:** alre-server-u20 (NVIDIA TITAN RTX, 24 GB), Python 3.8.10
**Status:** ABGESCHLOSSEN — Anker unabhaengig validiert, Entscheidung fuer 16a.4 getroffen


## 1. Aufgabe

Vor dem teuren `lambda_mean`-Sweep (16a.4, ~3-5h/Wert) mit einer BILLIGEN
Inferenz-Messung (kein Training, nur Vorwaertslauf auf dem Val-Subset mit einem
VORHANDENEN Checkpoint) pruefen, ob der Regler ueberhaupt etwas zu tun hat.
Dasselbe Diagnose-Prinzip wie die std-Gap-Messung in 15.2 (vgl. RAUSCH_REPORT.md).
Deliverable laut Arbeitsplan: kurze Diagnose-Notiz (mean-Gap-Wert + Entscheidung
voll/mini/skip). Asymmetrischer Payoff: kleiner Gap spart 10-15h, grosser Gap
kostet 15 Min (dann laeuft der Sweep wie geplant, wissenschaftlich sogar besser
belegt). Man kann nur gewinnen oder neutral rauskommen.


## 2. Vorgehen / Design-Entscheidungen

### 2.1 Anker: vorhandener Checkpoint statt Neu-Training
Der 16a-Anker (`.../task16a/lambda_grad_0/...`) existierte NICHT — der 16a.0-Smoke
war aufgeraeumt, der versehentliche Full-Run hat nichts Auffindbares hinterlassen.
Neu-Training war aber unnoetig: `checkpoints/task15/std_sweep/lstd_1.0/phase2/best_miou.pt`
(2026-07-06) ist bereits der exakte Arbeitspunkt. Der 15.3-std-Sweep patchte nur
`lambda_std` + `checkpoints.dir`, "restliche fuenf λ unberuehrt" (15.3-Bericht Z.33)
-> `lstd_1.0` lief mit `lambda_mean=0.1, lambda_cos=0.1, lambda_ssim=0.1,
lambda_grad=0, lambda_std=1.0` = byte-fuer-byte der Sweep-Basis-Arbeitspunkt
(nur fp32 statt fp16; fuer eine mean-Gap-DIAGNOSE irrelevant, der Gap ist noch
fp-unempfindlicher als mIoU).

### 2.2 Metrik: vorzeichensicher, NICHT der naive mean-Ratio
Ein `mean(pred_mean)/mean(real_mean)`-Ratio (analog zur std-Ratio) ist hier
untauglich: BEV-Latents koennen Channel-weise positive UND negative Mittel haben,
die sich beim Skalar-Mittel wegheben. Der Gap wird daher ueber den per-Channel-MSE
gemessen, wo sich nichts auskuerzt (identisch zu `inference.py`s `dist_mean`):

    dist_mean (pro Record) = mean_c (pm_c - rm_c)^2       [Latent^2]
    dist_std  (pro Record) = mean_c (ps_c - rs_c)^2       [Latent^2]
    rms_mean_err = sqrt(mean_records dist_mean)           [Latent-Einheiten]
    rms_std_err  = sqrt(mean_records dist_std)            [Latent-Einheiten]
    r            = rms_mean_err / rms_std_err     (mean-Rest vs. std-Rest)
    rel_mean_gap = rms_mean_err / mean(real_std)  (mean-Rest rel. Channel-Spread;
                   die "3-5%"-Groesse aus dem Arbeitsplan)

`r` misst den mean-Rest gegen den std-Rest — also gegen genau die Groesse, gegen
die `lambda_std=1.0` bereits aktiv arbeitet. `r << 1` heisst "mean vergleichsweise
geloest". Der naive Skalar-Gap wird nur zur Info mitgefuehrt (mit Caveat).

### 2.3 Scope-Grenze (ehrlich vermerkt)
Die Inferenz-only-Diagnose misst den Rest-mean-Gap AM Arbeitspunkt (`lambda_mean=0.1`
war beim Anker aktiv) — NICHT den sauberen "MSE-zieht-den-Mittelwert-allein"-Gap.
Der waere `lambda_mean=0` und braeuchte Training. Dieser definitive Test faellt
ohnehin als Nullpunkt aus 16a.4 heraus (siehe Abschnitt 5). Ein reiner *Skip* ist
damit strukturell ausgeschlossen; die Diagnose entscheidet zwischen *voll* und *mini*.


## 3. Durchfuehrung

Ein Vorwaertslauf (kein Training) auf dem Anker, 300 Val-Samples, in ein Scratch-
Verzeichnis; `.npy`-Dumps danach geloescht (JSON bleibt):

```bash
python inference.py --config config_sweep_base_fp16.yaml \
                    --phase cell --max_samples 300 \
                    --checkpoint checkpoints/task15/std_sweep/lstd_1.0/phase2/best_miou.pt \
                    --output predictions/task16a_1_meandiag
python diagnose_mean_gap.py --log predictions/task16a_1_meandiag/inference_log.json --md
find predictions/task16a_1_meandiag -name "*.npy" -delete
```

`--checkpoint` MUSSTE explizit gesetzt werden: die Ur-Config
`config_task15_3_lstd_1.0.yaml` traegt in `checkpoint_phase2` noch
`task15_baseline/...` (der std-Sweep patchte `checkpoint_phase2` nie, die Inferenz
lief damals per `--checkpoint`-Override). Ueber die aktuelle fp16-Basis-Config +
Override werden zugleich veraltete Latent-Pfade vermieden (Loader upcastet auf float32).


## 4. Ergebnisse (300 Val-Samples)

| Kennzahl | Wert |
|---|---|
| dist_mean (mean-Rest, Latent^2) | 9.846e-05 |
| dist_std (std-Rest, Latent^2) | 8.332e-04 |
| rms_mean_err | 0.00992 |
| rms_std_err | 0.02886 |
| **r = mean-Rest / std-Rest** | **0.344** |
| **rel_mean_gap** | **3.75 %** |
| pred_std / real_std | 0.2424 / 0.2649 |
| std-Ratio (Cross-Check) | 0.9151 |
| std-Gap | 8.49 % |
| pred_mean / real_mean (naiv, Info) | +0.14951 / +0.14611 |
| naiver Skalar-mean-Gap abs / rel | 0.00340 / 2.3 % |

**Anker-Validierung:** std-Ratio 0.9151 deckt sich praktisch exakt mit dem
15.3-Wert (0.915). Das bestaetigt (a) den korrekten Arbeitspunkt-Checkpoint und
(b) dass die fp16-Basis-Config auf den fp32-Gewichten die 15.3-Zahlen reproduziert.

**Zur Metrik:** `real_mean = +0.146` — die Latents sind NICHT zentriert, sondern
haben einen positiven DC-Offset. Deshalb ist der naive Skalar-Gap hier nicht
explodiert (2.3%). Aber `rms_mean_err` (0.00992) ist ~3x die naive Skalar-Differenz
(0.00340) -> es gibt sehr wohl per-Channel-Auskuerzung, die der Skalar verschluckt.
Die vorzeichensichere Zahl ist die ehrlichere; dass beide hier nah beieinander
liegen (2-4%), plausibilisiert sie gegenseitig.


## 5. Interpretation & Entscheidung

**rel_mean_gap = 3.75%** liegt im "<~3-5%"-Skip/Minimier-Band des Arbeitsplans.
**r = 0.344**: der mean-Rest ist ~ein Drittel des std-Rests — nicht null, aber der
std-Rest ist die Groesse, die trotz aktiver Regularisierung (lambda_std=1.0) noch
bei 8.49% steht. Der Mittelwert ist vergleichsweise gut in Form.

**ENTSCHEIDUNG 16a.4: MINI mit Pflicht-Nullpunkt -> Werteliste `{0, 0.1, 0.3}`.**

Begruendung:
- 3.75% ist klein (Skip/Minimier-Band), ein feiner Voll-Sweep ist nicht
  gerechtfertigt.
- Ein reiner *Skip* ist trotzdem ausgeschlossen: der `lambda_mean=0`-Punkt ist
  (a) der eigentliche "MSE-allein"-Test, den die Inferenz-Diagnose strukturell
  nicht leisten konnte (Anker hatte 0.1 an), und (b) der fuer die Arbeit noetige
  Notwendigkeits-Nachweis (belegt, dass `lambda_mean` ueberhaupt gebraucht wird).
- `0.3` bestaetigt, dass die Kurve oberhalb des Arbeitspunkts flach/gesaettigt ist
  (Schutz gegen "mehr mean-Reg haette geholfen"). Bei Deadline-Druck ist `0.3` der
  streichbare Punkt; irreduzibler Kern ist `{0, 0.1}`.

**Reihenfolge-Optimierung (zweiter Nutzen der Diagnose):** `lambda_mean` ist als
flacher Low-Prio-Regler empirisch belegt -> wandert in der Sweep-Reihenfolge ans
Ende. Ablauf bleibt: 16a.2 (`lambda_grad`, vielversprechendster) -> 16a.3
(`lambda_ssim`) -> 16a.4 (`lambda_mean`/`lambda_cos`, kurz). Falls die Zeit ausgeht,
sind die wichtigen Ergebnisse zuerst da, `lambda_mean` guten Gewissens kuerzbar.


## 6. Learnings / Merkposten

- **Stale `checkpoint_phase2` in den 15.3-Sweep-Configs:** der std-Sweep patchte
  `checkpoint_phase2` nie -> alle `config_task15_3_lstd_*.yaml` zeigen dort auf
  `task15_baseline/...`. Fuer Nachrechnungen IMMER `--checkpoint` explizit setzen
  (oder die aktuelle Basis-Config + Override nehmen).
- **`inference.py` dumpt pro Sample `pred_*.npy` + `real_*.npy`** (je [256,128,128]
  fp32 ~16.7 MB -> ~10 GB/300 Samples). Fuer Diagnosen/Sweeps unnoetig (nur die
  JSON-Skalare zaehlen). -> `--no-save`-Patch fuer 16a.2 vormerken, wo sich das
  ueber die Werteliste multipliziert. Disk stand bei 88% (368 GB frei); Scratch+rm
  war fuer den Einzellauf ausreichend.
- **fp16-Loader auf fp32-Gewichten reproduziert 15.3** (std-Ratio 0.9151 vs 0.915)
  — nuetzliche Nebenbestaetigung, dass Basis-Config + `--checkpoint`-Override ein
  gueltiger Nachrechen-Pfad ist.


## 7. Offen / uebergeben an 16a.4

- **16a.4 lambda_mean:** Werteliste `{0, 0.1, 0.3}`, `lambda_std=1.0` fix, Rest
  Baseline. Der `0`-Punkt schliesst den offenen "MSE-allein"-Test und den
  Notwendigkeits-Nachweis. ZULETZT fahren (nach 16a.2/16a.3).
- **Signifikanzschwellen (aus 15.2, gelten weiter):** |Δ mIoU| > ~0.014,
  |Δ std-Ratio| > ~0.009. Ein mean-Sweep-Effekt unterhalb dieser Schwellen gilt
  als "im Rauschen, grob bestaetigt".


## 8. Artefakte

- `diagnose_mean_gap.py` (Post-hoc-Reader, server-unabhaengig, nur JSON-abhaengig)
- `predictions/task16a_1_meandiag/inference_log.json` (300-Sample-Diagnose-Log)
- Diese Notiz: `TASK16A_1_ABSCHLUSSBERICHT.md`
