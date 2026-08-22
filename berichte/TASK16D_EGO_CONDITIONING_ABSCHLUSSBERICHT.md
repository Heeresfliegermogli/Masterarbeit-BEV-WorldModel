# Task 16d — Ego-Motion-Conditioning (Retrain-Ablation)

Datum: 2026-07-21. Jobs 123599 (FiLM state/action, abgebrochen), 123621
(ego16d2: gate + token), Eval 123628 (FiLM-Nachhol-Full-Val).
Frage (#1 aus VERBESSERUNGSVORSCHLAEGE_MODELL.md): Bringt echtes
Ego-Motion-Conditioning einen mIoU-Gewinn ueber die konditionslose Baseline
("off" = SmoothL1-Minimal-Headline, 16b.9, Full-Val mIoU 0.6946)?

## Ergebnis (Full-Val, 5743 Fenster, vs. off=0.6946)

| Variante | Injektion             | Full-Val mIoU | d vs off | Signifikanz |
|----------|-----------------------|--------------:|---------:|-------------|
| off      | keine (Referenz)      | 0.6946        |  —       | —           |
| gate     | Bias auf Gate-Logits  | 0.6910        | -0.0036  | **n.s.**    |
| action   | FiLM (Feature-Mod.)   | 0.6175        | -0.0771  | signifikant |
| token    | Ego-Token (prepend)   | 0.6164        | -0.0782  | signifikant |
| state    | FiLM (Feature-Mod.)   | 0.6094        | -0.0852  | signifikant |

Signifikanzschwelle |dmIoU| > 0.014 (15.2). Figur:
visualizations/16d_ego_conditioning.{png,pdf}.

## KERNBEFUND

**Kein Ego-Conditioning schlaegt off.** Die beste Variante (gate, 0.6910) ist
statistisch NICHT von off zu unterscheiden (-0.0036, n.s.) — sie ERHAELT die
Baseline, uebertrifft sie aber nicht. Alle uebrigen Injektionen (FiLM state/action,
Ego-Token) SCHADEN klar signifikant (-0.077 bis -0.085).

**Entscheidend ist der Injektionsmechanismus, nicht das Ego-Signal.**
- FiLM (state 0.6094, action 0.6175) und Ego-Token (0.6164) liegen alle im
  selben Schaden-Regime (~-0.08). Sie speisen eine gelernte Modulation in JEDEN
  Token-Feature-Vektor ein und stoeren damit die gut eingestellte Latent-
  Repraesentation des 16b.9-Modells.
- gate ist die einzige minimale, quasi-identische Stoerung: ein zero-init-Bias
  auf die bestehenden Gate-Logits (alpha = sigmoid(Conv(x) + b(ego))). Er
  veraendert nur die Copy-vs-Predict-Balance leicht, nicht den Feature-Strom —
  und landet exakt auf der Baseline zurueck.
- Dass ausgerechnet der schwaechste Eingriff am besten abschneidet, zeigt: das
  Ego-Delta traegt bei k=1 KEINE zusaetzliche pradiktive Information, die der
  3-Frame-Kontext nicht schon implizit enthaelt. Jede staerkere Einspeisung ist
  reines Rauschen fuer das Netz.

**Signal-Vergleich (nachrangig):** action (F3->F4, 0.6175) schlaegt state
(F1->F2,F2->F3, 0.6094) unter FiLM um +0.008 — das Aktionssignal traegt marginal
mehr als der Zustand, aber beide bleiben deutlich negativ. Der Unterschied ist
zu klein, um die Gesamtaussage zu aendern.

## Einordnung in den Strang

Konsistent mit der ego-gewarpten Persistenz-Vorstufe (16d-Vorstufe): dort war der
Dynamik-Beitrag des Modells zwar real, aber moderat und mit Horizont schrumpfend.
Hier zeigt sich die andere Seite derselben Muenze — explizites Ego-Conditioning
addiert bei k=1 keinen mIoU. Der 3-Frame-Kontext deckt die Ego-Bewegung fuer den
Ein-Schritt-Horizont bereits ab.

Damit ist Verbesserungsansatz #1 (Ego-Conditioning) als mIoU-Hebel ENTKRAEFTET:
bestenfalls neutral (gate), sonst schaedlich. Fuer die Thesis ein sauberes
Negativ-Ergebnis mit klarer Mechanismus-Erklaerung.

## Offen / nicht getriggert

- Rollout-Eval (k>1) fuer die konditionierten Modelle: NICHT durchgefuehrt.
  Motivation entfaellt — da schon bei k=1 kein Gewinn, und Ego-Delta ueber den
  Horizont nur bei mehrschrittiger Aktions-Kette (per-Schritt-action) ueberhaupt
  wirken koennte; der Aufwand (rollout_eval-Ego-Delta-Patch) lohnt bei k=1-Nulleffekt
  nicht. Falls doch gewuenscht: per-Schritt-action = ego(idx(1+k)->idx(2+k)).
- Verbleibende billige Ablationen (unabhaengig von Ego): #4 Kontext 3->4 Frames,
  #6 EMA. Task 18 (generativ) bleibt motiviert durch std-Gap-Rest + Rollout-Drift,
  nicht durch Ego-Conditioning.

## Artefakte

- JSONs: predictions/task16d/ego_{gate,token,action,state}_fullval.json
  (Cluster: checkpoints/task16d/ego_*/phase2/miou_fullval.json).
- Figur: visualizations/16d_ego_conditioning.{png,pdf} (render_16d.py, lokal).
- Konfigs: config_ego_{state,action,gate,token}.yaml (Training),
  config_ego_{state,action}_eval.yaml (stage_to_shm=false fuer Nachhol-Full-Val).
- Code: FiLM/gate/token-Zweige in Code/bev_world_model.py (zero-init =
  Identitaet -> off bit-identisch), ego_delta in Code/bev_dataset.py,
  ego_cond_mode in Code/config.py.
- sbatch: sbatch_ego.sh (FiLM), sbatch_ego2.sh (gate/token),
  sbatch_eval_film.sh (Nachhol-Eval, plain, ohne shm-Staging).

## Lektionen (SLURM/Eval)

- Full-Val-Eval NIE mit stage_to_shm=true auf begrenztem --mem: das 286G-tmpfs-
  Staging zaehlt gegen das Step-Memory-cgroup -> sofort OOM. Loesung: eval-Configs
  mit stage_to_shm=false (Dataset mmap't packed.npy direkt von BeeGFS, ~50GB val,
  RAM-Footprint winzig, Decode dominiert ohnehin).
- srun-Helper-Skripte gehoeren ins geteilte Home (~/Code_final/), NICHT ins
  node-lokale /tmp des Login-Nodes (Compute-Node sieht es nicht -> exit 127).
