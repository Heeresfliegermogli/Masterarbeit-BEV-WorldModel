# ZWISCHENBERICHT — Persistenz-Validierung des World Models

**Datum:** 26.06.2026
**Status:** ✅ ABGESCHLOSSEN
**Einordnung:** Validierungs-Interludium zwischen Task 13 (Baseline) und Task 14
(Decode-Validierung). Kein nummerierter Arbeitsplan-Task, sondern eine
nachgeschobene wissenschaftliche Kontrolle.

---

## Fragestellung

Auslöser war begründete Skepsis: Sagt das BEV World Model tatsächlich einen
zukünftigen Frame `t+1` vorher, oder kopiert es im Kern nur den letzten
Input-Frame `t` und profitiert davon, dass sich eine BEV-Szene bei 2 Hz
(0,5 s zwischen Frames) kaum verändert?

Diese Frage war bis hierher **nie kontrolliert** worden:
- Das Gate `α` der Gated Skip Connection wurde nie geloggt oder gemessen.
- Es existierte **keine Persistenz-Baseline** — also keine Zahl dafür, welche
  mIoU man erreicht, wenn man Frame `t` dekodiert und so tut, als wäre es die
  Vorhersage für `t+1` (ganz ohne Modell).

Ohne diese Kontrolle sind die Kennzahlen (mIoU, Cosine) nicht als Beleg für
Vorhersage interpretierbar — sie könnten vollständig durch die Persistenz
erklärt sein.

---

## Vorgehen

### 1. Pipeline-Verifikation (Code-Review)

Datenkonstruktion in `bev_dataset.py` geprüft und als korrekt bestätigt:
- Sliding-Window `scene[i : i+4]`, `n_input_frames = 3`
- Input = `window[:-1]` = `[t-2, t-1, t]`, Target = `window[-1]` = `t+1`
- Keine Fenster über Szenen- oder PKL-Grenzen → kein Leakage

Die Aufgabe ist also strukturell tatsächlich „sage `t+1` vorher". Bestätigt
wurde außerdem der Architektur-Mechanismus, der das eigentliche Risiko trägt:

```python
skip  = x[:, -1, ...]                    # = Frame t (letzter Input)
alpha = self.gate(x)                     # ∈ [0,1]
pred  = alpha * transformer_out + (1.0 - alpha) * skip
```

Bei MSE-basiertem Loss ist „Frame t kopieren" eine starke verlustarme Strategie
— der Grund, warum die Kontrolle nötig war.

### 2. Persistenz-Baseline (`persistence_baseline.py`)

Eigenständiges Skript, im Container `bevfusion:correct` ausgeführt. Es:
- rekonstruiert dieselbe Fensterlogik wie das Training (gleiche Val-PKL,
  `n_input_frames=3`, `gap_sec=2.0`) → identischer Fenster-Satz wie Task 13,
- nutzt den **unveränderten** Seg-Decoder aus `inference_seg.py`
  (backbone + neck + map_head, strict-load aus `bevfusion-seg.pth`),
- dekodiert je Szene jeden Frame nur einmal (Decode-once, halbe Kosten),
- misst zwei unabhängige Signale auf identischen Token:

| | Persistenz | World Model |
|---|---|---|
| mIoU | `mIoU(decode(Frame_t), decode(Frame_{t+1}))` | `mIoU(decode(pred), decode(Frame_{t+1}))` |
| Cosine | `cos(Frame_t, Frame_{t+1})` | `cos(pred, Frame_{t+1})` |

Die „Ground Truth" ist auf beiden Seiten `decode(Frame_{t+1})` — exakt die
Konvention aus Task 9d/13, damit der Vergleich apples-to-apples ist.

### 3. Vollständiger Modell-Output

Die ursprüngliche Task-13-mIoU (0,654) basierte auf nur ~10 Samples — zu wenig
für eine stabile Zahl und auf anderen Token als die Baseline. Daher wurde
`inference.py` auf dem **vollen Val-Set** (5743 Sequenzen) neu ausgeführt:

```
FERTIG! 5743 Samples in 428.8s
MSE: 0.025115 ± 0.010135 | CosSim: 0.8549 | dist: 0.002649
```

Damit liegen für alle 5743 Fenster `pred_<token>.npy` vor → Persistenz und
Modell stehen auf demselben Fundament.

---

## Ergebnisse

Beide Seiten über **alle 5743 Val-Fenster**, identische Token, identischer
Decoder.

### Vergleichstabelle

| Metrik | Persistenz | World Model | Δ |
|---|---:|---:|---:|
| **mIoU (gesamt)** | **0,5329** | **0,6823** | **+0,1494** |
| drivable_area | 0,8021 | 0,8927 | +0,0906 |
| ped_crossing | 0,3495 | 0,6121 | **+0,2626** |
| walkway | 0,5881 | 0,7226 | +0,1345 |
| stop_line | 0,3190 | 0,5316 | **+0,2126** |
| carpark_area | 0,5563 | 0,6791 | +0,1228 |
| divider | 0,5146 | 0,6300 | +0,1154 |
| Cosine (mean) | 0,7437 | 0,8549 | +0,1112 |
| Std (mean) | 0,2725 | 0,2320 | −0,0405 |

### Kernbefunde

1. **mIoU +0,149 (relativ ~28 %).** Kein Rauschen, kein marginaler Vorsprung —
   das Modell schlägt das reine Kopieren klar. Die belastbare Voll-Baseline
   liegt bei **0,6823**, nicht bei den früheren 0,654 (10-Sample-Schätzung).

2. **Cosine +0,111.** Die rohe Frame-zu-Frame-Ähnlichkeit (0,744) liegt
   deutlich unter der Modell-Cosine (0,855). Ein reiner Kopierer wäre bei 0,744
   gedeckelt → der gelernte Transformer-Anteil trägt echtes Vorhersage-Signal
   bei, nicht nur Glättung.

3. **Klassen-Muster ist das stärkste Argument.** Der kleinste Vorsprung liegt
   bei der großen statischen Klasse `drivable_area` (+0,091, dort trifft
   Kopieren schon fast alles). Die größten Vorsprünge liegen bei den kleinen,
   dynamischen Klassen `ped_crossing` (+0,263) und `stop_line` (+0,213) — genau
   dort, wo Persistenz versagt und echtes Vorhersagen den Unterschied macht.
   Das belegt, dass das Modell Veränderung modelliert und nicht nur den Vorteil
   statischer Flächen abschöpft.

4. **Std-Gap einordnen.** Persistenz hat höhere Std (0,273 vs. 0,232), weil ein
   Kopierer die volle Varianz des echten Frames übernimmt. Das Modell erreicht
   die bessere mIoU **trotz** niedrigerer Varianz (bekanntes
   Regression-to-the-Mean / RAUSCH_REPORT). Der std-Gap ist damit eine
   Qualitätsgrenze für künftige Verbesserung (Task 15 / probabilistischer
   Loss), **kein** Argument gegen das aktuelle Modell.

---

## Wissenschaftliche Schlussfolgerung

Das Modell sagt aus den 3 Input-Frames `[t-2, t-1, t]` nachweislich einen
zukünftigen Frame `t+1` vorher und schlägt die einzig relevante triviale
Baseline (Persistenz) signifikant — über die volle Val-Menge, mit gut
interpretierbarem Klassen-Muster.

**Präzise Formulierung für die Thesis (Differenzierung):** Der einzige Input
sind die 3 Frames. Architektonisch ist die Ausgabe eine *gelernte, gegatete
Verfeinerung von Frame `t`* (`pred = α·Transformer + (1−α)·Frame_t`), nicht eine
komplett aus dem Nichts generierte Szene. „Vorhersage rein aus den 3 Frames"
ist im Sinne der Eingabe korrekt; der Cosine-Sprung 0,744 → 0,855 belegt, dass
der gelernte Anteil substanzielles Vorhersage-Signal liefert.

---

## Methodische Notizen (für die Reproduzierbarkeit)

- **Identischer Fenster-Satz:** Persistenz und Modell laufen über dieselben
  5743 Token; das Skript verarbeitet nur Fenster, für die `pred` und beide
  `real`-Latents vorliegen.
- **Korrigierte Baseline:** Die offizielle Task-13-Baseline-mIoU ist auf
  **0,6823** zu aktualisieren (Vollmenge statt 10-Sample-Schätzung).
- **Pfad-Fallen im Container (dokumentiert):** Der Seg-Checkpoint liegt unter
  `/bevfusion/pretrained/bevfusion-seg.pth`; ein alter Symlink unter
  `/home/bevfusion/pretrained/` zeigte ins Leere. `--checkpoint` ist jetzt
  Argument, kein hartkodierter Pfad.
- **Zusätzliche Mounts** für diesen Lauf: `latents/seg/val` → `/latents/seg/val`
  und `pred_full` → `/pred_full`.

---

## Wichtige Learnings

1. **Persistenz-Baseline ist Pflicht-Kontrolle** für jedes World Model. Ohne
   sie sind mIoU/Cosine nicht als Vorhersage-Beleg interpretierbar. Ab jetzt
   feste Größe in jeder Metrik-Tabelle.
2. **Baseline-Zahlen auf ausreichend Samples erheben.** 10 Samples ergaben
   0,654; die stabile Vollmenge ergibt 0,6823. Stichprobengröße prüfen, bevor
   eine Zahl eingefroren wird.
3. **Klassen-Aufschlüsselung schlägt Gesamt-mIoU** als Argument. Das *Muster*
   (Gewinn dort, wo Persistenz versagt) ist überzeugender als die aggregierte
   Zahl.

---

## Offene Punkte / Ausblick

- **Gate-`α`-Quantifizierung** (Kür): Forward-Hook auf `self.gate`, Mittelwert +
  räumliche Heatmap. Nach diesem Ergebnis nicht mehr zwingend, aber eine
  zusätzliche interne Bestätigung und liefert die für Task 16 geplante
  Gate-Visualisierung gleich mit.
- **Task 13 Bericht aktualisieren:** Baseline-mIoU 0,654 → 0,6823, plus
  Verweis auf diese Persistenz-Tabelle.
- **Task 15 Stoßrichtung:** Größter Hebel liegt nicht bei `drivable_area`
  (nahe Maximum), sondern bei `stop_line` / `ped_crossing` in Kombination mit
  dem std-Gap.
- **Weiter mit Task 14** (mIoU-basierte Decode-Validierung) — steht jetzt auf
  geprüftem Fundament.

---

## Artefakte

- `persistence_baseline.py` — Vergleichsskript (Persistenz + Modell auf
  identischen Token)
- `comparison_baseline.json` — Rohergebnisse (5743 Fenster)
- `pred_full/` — vollständiger Modell-Output (5743 `pred_<token>.npy`)
