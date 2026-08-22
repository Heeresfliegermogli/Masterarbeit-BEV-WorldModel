# Task 16e — mIoU-Hebel: 4-Frames + EMA (Retrain-Ablation)

Datum: 2026-07-22. Job 123807 (mlever16e, 2 GPU parallel, COMPLETED 11:27).
Frage: Bewegt einer der beiden billigen Hebel (#4 mehr temporaler Kontext,
#6 Weight-Averaging) die Kernmetrik mIoU ueber die Baseline (off = SmoothL1-
Minimal-Headline, 16b.9, Full-Val 0.6946)? Fokus bewusst mIoU, nicht std-Gap.

## Ergebnis (Full-Val vs. off=0.6946)

| Variante            | Injektion            | mIoU   | n    | d vs off | Signifikanz |
|---------------------|----------------------|-------:|-----:|---------:|-------------|
| off (Referenz)      | keine                | 0.6946 | 5743 |  —       | —           |
| nf4 (4 Frames)      | n_frames 3->4        | 0.6915 | 5651 | -0.0031  | **n.s.**    |
| ema (EMA shadow)    | Weight-Averaging     | 0.6945 | 5743 | -0.0001  | **n.s.**    |
| ema (raw best_miou) | -- (Konsistenz-Check)| 0.6952 | 5743 | +0.0006  | n.s.        |

Signifikanzschwelle |dmIoU| > 0.014 (15.2). Figur:
visualizations/16e_miou_lever.{png,pdf}.

## KERNBEFUND

**Beide Hebel sind flach — kein mIoU-Gewinn.**

- **4 Frames (nf4): -0.0031, n.s.** Mehr temporaler Kontext addiert keinen mIoU.
  Der 3-Frame-Kontext deckt den Ein-Schritt-Horizont bereits ab; ein viertes
  Input-Frame bringt keine zusaetzliche pradiktive Information. (Nebenpunkt zur
  Fairness: nf4 evaluiert auf 5651 statt 5743 Fenstern -- window_size 5 statt 4,
  also ein Fenster pro Szene weniger. Overlap ~98%, aendert die Aussage nicht.)
- **EMA (shadow): -0.0001, n.s.** Weight-Averaging ist praktisch wirkungslos.
  Konsistent mit der Vorab-Erwartung: EMA hilft bei verrauschtem Training, das
  hier aber same-seed nahezu deterministisch laeuft -- es gibt kaum Rausch-
  Trajektorie, ueber die sich mitteln liesse.
- **Konsistenz bestaetigt:** Der raw best_miou.pt des EMA-Laufs (n_frames=3,
  smooth_l1+std, seed 42, EMA nur passiver Schatten) reproduziert off:
  0.6952 vs. 0.6946 (+0.0006, im Rauschboden). Damit ist der same-seed-
  Determinismus (16b.5-Befund) erneut belegt.

## STRATEGISCHE EINORDNUNG (mIoU-Decke)

Zusammen mit Task 16d (Ego-Conditioning flach: bestenfalls neutral) sind damit
ALLE billigen bis mittleren mIoU-Hebel des Seg-Strangs ausgeschoepft:

| Hebel                         | Task   | mIoU-Effekt          |
|-------------------------------|--------|----------------------|
| Loss-Diaet (mse+std -> minimal)| 16b.9 | erreicht Baseline    |
| Ego-Conditioning (gate/FiLM/token)| 16d| bestenfalls neutral  |
| 4 Frames                      | 16e    | flach (n.s.)         |
| EMA                           | 16e    | flach (n.s.)         |

Der deterministische mIoU-Plateau bei **~0.69 ist die Decke dieses ~6M-Setups**
-- genau die Arbeitshypothese ("mIoU-Plateau bei persistierendem Schaerfe-Limit").
Realistischer Rest-Headroom liegt nur noch in:
  (a) KAPAZITAET -- groesseres Modell / staerkerer Output-Head (Architektur wurde
      als Confounder-Regel bewusst eingefroren; ein sauberer Kapazitaets-Sweep
      waere ein eigener, groesserer Schritt), oder
  (b) GENERATIV (Task 18) -- aber dann fuer SCHAERFE der Masken, nicht fuer mIoU;
      der std-Gap ist bereits ~geschlossen (16b.7), der Nutzen waere qualitativ.

Empfehlung: den mIoU-Optimierungsstrang hier ABSCHLIESSEN. Beste Konfig bleibt
die 16b.9-Minimal (smooth_l1+std, 0.6946). Naechster sinnvoller Schritt:
Task 19 (Seg-Strang-Abschluss/Zusammenfassung) -- oder bewusste Entscheidung
fuer Task 18 als generatives Kapitel (Schaerfe, nicht mIoU).

## Artefakte

- JSONs: predictions/task16e/{nf4,ema_raw,ema_shadow}_fullval.json
  (Cluster: checkpoints/task16e/{nf4,ema}/phase2/miou_fullval*.json).
- Figur: visualizations/16e_miou_lever.{png,pdf} (render_16e.py, lokal).
- Konfigs: config_nf4.yaml (n_frames=4), config_ema.yaml (ema+ema_decay),
  beide aus der 16b.9-Minimal-Basis, ego off.
- Code: EMA in train_linux.py (train_one_epoch: ema_state/ema_decay; main:
  Shadow-Setup + best_miou_ema.pt). Guarded per train_cfg.get("ema", False)
  -> Default AUS = bit-identisch zum Altpfad (lokal verifiziert: off-Modell
  6.054.144 Params, keine Ego-Module; nf4 +256 Params = eine pe_t-Zeile).
- sbatch: sbatch_16e.sh (2 GPU parallel, plain, ohne --export).
