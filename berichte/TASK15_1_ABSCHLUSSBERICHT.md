# TASK15.1_ABSCHLUSSBERICHT — Loss-Umbau + Korrektheits-Check

**Datum:** 2026-07-01
**Status:** ABGESCHLOSSEN ✅
**Kein Training — reine Apparatur**

---

## Ziel

`compute_loss()` vollständig YAML-konfigurierbar machen: jeden Term einzeln
steuerbar, ohne Code-Edit zwischen Sweep-Läufen. Danach beweisen, dass der
Refactor die Task-13-Baseline exakt reproduziert — denn ohne bestandenen
Check misst jeder spätere Sweep den Refactor-Bug statt das Lambda.

---

## Durchgeführte Änderungen

### `train_linux.py` — `compute_loss()`

**Alte Signatur (Task 13):**
```python
compute_loss(pred, target, use_cosine, lambda_cos, lambda_dist, lambda_ssim)
# → 5-Tupel: (total, mse, cos, dist, ssim)
```

**Neue Signatur (Task 15):**
```python
compute_loss(pred, target,
             lambda_mse=1.0, lambda_cos=0.1,
             lambda_mean=0.1, lambda_std=0.1,
             lambda_grad=0.0, lambda_ssim=0.1)
# → 7-Tupel: (total, mse, cos, mean, std, grad, ssim)
```

**Konkrete Änderungen:**

| Änderung | Detail |
|---|---|
| `lambda_dist` → `lambda_mean` + `lambda_std` | Aufspaltung schafft getrennte Regler; `loss_mean` und `loss_std` werden separat berechnet und geloggt |
| `lambda_mse` explizit (Default 1.0) | War implizit 1.0 — jetzt YAML-steuerbar |
| `lambda_grad` neu (Default 0.0 = AUS) | Anti-Blur-Term: MSE(∂pred/∂x, ∂target/∂x) + MSE(∂pred/∂y, ∂target/∂y); bei 0 kein Rechenaufwand |
| `use_cosine: bool` entfernt | Ersetzt durch `lambda_cos=0.0` — konsistenter, kein Sonderpfad |
| Return: 5-Tupel → 7-Tupel | (total, mse, cos, mean, std, grad, ssim) |

**Neue Loss-Gleichung:**
```
loss_total = lambda_mse  * loss_mse
           + lambda_cos  * loss_cos
           + lambda_mean * loss_mean
           + lambda_std  * loss_std
           + lambda_grad * loss_grad
           + lambda_ssim * loss_ssim
```

### `train_one_epoch()` und `validate()`

- Signaturen auf 6 Lambda-Parameter umgestellt (statt `use_cosine` + 3 λ)
- Akkumulatoren: `total_dist` → `total_mean` + `total_std` + `total_grad`
- Return-Tupel: 5 → 7 Werte

### wandb-Logging

| Alt | Neu |
|---|---|
| `train/dist` | `train/mean` + `train/std` + `train/grad` |
| `val/dist`   | `val/mean` + `val/std` + `val/grad` |

### `config_nuscenes_task15.yaml` (neu erstellt)

Einzige Steuerdatei für alle Task-15-Sweep-Läufe. Baseline-äquivalente Startwerte:

```yaml
lambda_mse:   1.0    # Multi-Scale MSE
lambda_cos:   0.1    # Cosine Similarity
lambda_mean:  0.1    # Channel-Mean-Term  (entspricht altem lambda_dist/2)
lambda_std:   0.1    # Channel-Std-Term   (Kernregler Sweep 15.3)
lambda_grad:  0.0    # Gradient/Anti-Blur (AUS, Sweep 15.4)
lambda_ssim:  0.1    # SSIM-Term          (Ablation 15.5)
```

Zwischen Sweep-Läufen wird **ausschließlich** das jeweilige Lambda in dieser
YAML geändert — kein Code-Edit.

---

## Korrektheits-Check — Ergebnisse auf dem Server (CUDA)

`python test_loss_task15.py` auf dem Trainingsserver (TITAN RTX, CUDA):

| # | Test | Ergebnis | Detail |
|---|---|---|---|
| 1a | `loss_mean + loss_std ≈ alter loss_dist` | ✅ | Δ = 0.00e+00 |
| 1b | `loss_total NEU ≈ loss_total ALT` | ✅ | Δ = 5.96e-08 << 1e-6 |
| 1c | `loss_mse` identisch | ✅ | Δ = 0.00e+00 |
| 1d | `loss_cos` identisch | ✅ | Δ = 0.00e+00 |
| 2 | `lambda_grad=0 → loss_grad == 0.0` (exakt) | ✅ | 0.0 |
| 3 | `lambda_cos=0 → loss_cos == 0.0` (exakt) | ✅ | 0.0 |
| 4 | `lambda_grad=1.0 → loss_grad > 0` | ✅ | 7.998893 |
| 5 | Gradientenfluss — kein NaN/Inf | ✅ | — |
| 6 | NaN/Inf in allen 6 Termen | ✅ | alle sauber |
| 7 | Return-Tupel hat 7 Elemente | ✅ | — |

**Zum Delta 5.96e-08 in Test 1b:** Das ist exakt die erwartete Float32-Reassoziation.
Der alte Code rechnete `0.1 * (mse_mean + mse_std)` (erst addieren, dann skalieren),
der neue `0.1 * mse_mean + 0.1 * mse_std` (erst skalieren, dann addieren).
Mathematisch identisch, in IEEE-754 Float32 durch unterschiedliche Zwischenwerte
~1 ULP verschieden. Kein Bug — normales Floating-Point-Verhalten, dokumentiert
und innerhalb der definierten Toleranz.

---

## Entscheidungen

| Frage | Entscheidung | Begründung |
|---|---|---|
| Rückwärts-Kompat-Shim für alte Configs? | Nein | Neue Config von Anfang an explizit; kein versteckter Code-Pfad, kein Confounder |
| Korrektheits-Schwelle | < 1e-6 | Ehrlich gegenüber Float32-Arithmetik; kein Vortäuschen bit-exakter Gleichheit |
| `use_cosine: bool` | Entfernt | `lambda_cos=0.0` ist äquivalent, konsistenter, vermeidet Sonderpfad |

---

## Deliverables

| Datei | Beschreibung |
|---|---|
| `train_linux.py` | Refactorierte `compute_loss()`, angepasste `train_one_epoch()`, `validate()`, wandb-Logging |
| `config_nuscenes_task15.yaml` | Neue Task-15-Config, alle 6 λ explizit, Baseline-äquivalente Startwerte |
| `test_loss_task15.py` | Korrektheits-Check, 7 Tests — auf CUDA bestanden |

---

## Nächster Schritt: Task 15.2 — Rauschboden

Die refactorierte Loss mit baseline-äquivalenten Lambda **2–3× mit verschiedenen
Seeds** trainieren (z.B. Seed 42, 43, 44). Ziel: `mean ± std` je Metrik
(mIoU, std-Ratio, MSE, CosSim) → Signifikanzschwelle für alle Sweep-Läufe in
15.3–15.6.

Ein Lambda-Effekt ist erst ein Befund, wenn sein Delta die Lauf-zu-Lauf-Streuung
dieser Baseline überschreitet.
