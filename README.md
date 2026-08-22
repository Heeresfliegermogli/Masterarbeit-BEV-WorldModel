# BEV-Weltmodell — Vorhersage latenter BEV-Repräsentationen für das autonome Fahren

Transformer-basiertes Weltmodell, das die nächste latente BEV-Repräsentation
(kurz: Latent) aus den drei vorherigen Frames vorhersagt. Die Latents stammen
aus dem **eingefrorenen BEVFusion-Encoder** (nuScenes); dekodiert wird
ebenfalls mit den eingefrorenen BEVFusion-Köpfen (Segmentierung und
Detektion). Masterarbeit an der Universität der Bundeswehr München.

## Kernergebnisse

| Metrik (Vollvalidierung, gegen die nuScenes-Annotation) | Persistenz | Weltmodell | Referenz (reales Latent) |
|---|---|---|---|
| Segmentierung (mIoU, 6 Klassen) | 0.4647 | **0.5898** | 0.6295 |
| Detektion (mAP) | 0.1696 | **0.3438** | 0.6858 |

- **Loss-Minimalismus („from six to two"):** In der Segmentierung erreicht
  eine Zwei-Term-Zielfunktion (Smooth-L1 + Streuungsterm) die volle
  Sechs-Term-Baseline (0.6946 vs. 0.6920 auf der Proxy-Skala); in der
  Detektion tragen die Zusatzterme dagegen ~5 % rel. mAP — die minimale
  Loss-Menge ist eine Eigenschaft der Zielmetrik, keine allgemeine Regel.
- **Rollout k=1..4:** Das Weltmodell schlägt naive und ego-kompensierte
  Persistenz auf jedem Horizont; der Streuungsterm hält die Streuung über
  den Rollout kalibriert (Streuungsverhältnis 0.99–1.01).
- **Kopfadaptation:** Nachtrainieren der eingefrorenen Wahrnehmungsköpfe
  auf Weltmodell-Vorhersagen holt ~26–29 % des Abstands zur Referenz mit
  realem Latent zurück (Seg +0.011 mIoU, Det +0.089 mAP) —
  über mehrere Seeds abgesichert.
- **Generative Köpfe (CVAE, Flow Matching):** liefern Variation zwischen
  den Stichproben, aber keinen Genauigkeitsgewinn gegenüber dem
  deterministischen Modell.
- **Ressourcen:** 4.4 ms Inferenz (fp16, Batch 1, TITAN RTX), <270 MB VRAM,
  ~6M Parameter — <1–2 % eines 500-ms-Wahrnehmungszyklus.

## Repository-Struktur

```
├── train_linux.py            # Training (eine Codebasis lokal + Cluster)
├── inference.py              # Proxy-Evaluation (Teilstichprobe mit 300
│                             #   Fenstern; mIoU, Streuungsverhältnis)
├── eval_full_val.py          # Vollvalidierung (5743 Fenster)
├── rollout_eval.py           # autoregressiver Rollout k=1..4
├── config*.yaml              # aktive Basis-/Eval-Configs
├── Code/                     # Modell-Module (Dataset, Embedding, Transformer,
│                             #   Output-/Upsampling-Head, Flow-Head, Loss)
├── scripts_render/           # ALLE Thesis-Figuren (thesis_style.py = Stylesheet;
│                             #   Aufruf aus dem Repo-Root: python3 scripts_render/<x>.py)
├── predictions/              # kleine Ergebnis-JSONs/CSVs (Datengrundlage der Figuren;
│                             #   grosse Dumps sind nicht im Repo)
├── visualizations/           # gerenderte Figuren (PDF/PNG)
├── berichte/                 # Task-Abschlussberichte = Labor-Journal (deutsch)
├── archiv/                   # historische Configs, SLURM-Skripte, Einmal-Tools
└── checkpoints/              # (nicht im Repo — Gewichte, siehe unten)
```

## Setup

```bash
# Python 3.10, CUDA 12.1
pip install -r requirements.txt
```

## Daten & Gewichte (nicht im Repo)

Aus Lizenzgründen enthält das Repo **weder nuScenes-Daten/-Latents noch
BEVFusion-Gewichte** und keine trainierten Checkpoints. Zum Reproduzieren:

1. **nuScenes** (v1.0-trainval): [nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes)
   (Download nach Registrierung, nuScenes-Lizenzbedingungen beachten).
2. **BEVFusion**: offizielles Repo [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion)
   mit den offiziellen Seg-/Det-Gewichten (`bevfusion-seg.pth` /
   `bevfusion-det.pth`).
3. **Latents extrahieren:** über den `latent_saver`-Hook im
   BEVFusion-Modell (Env-Schalter `SAVE_BEV_LATENTS`; Gegenstück
   `LOAD_BEV_LATENTS` injiziert Latents für die annotationsbasierte
   Evaluation). Der Hook ist ein **externes Werkzeug aus einer
   vorangegangenen Projektarbeit** und lebt als kleiner Patch im
   BEVFusion-Repo/Docker-Container — er ist nicht Teil dieses Repos.
   Ergebnis: je Split ein Verzeichnis einzelner `.npy`-Dateien
   (Seg: 256×128×128, Det: 256×180×180, fp16).
4. **Packen:** `python -u Code/pack_latents.py --config <config> --split val
   --dtype float16` (idempotent; memmap-fähige Packs für den Loader).
5. **Trainieren:** `python -u train_linux.py --config config.yaml`
   (Cluster-Sweeps: `archiv/sbatch/`, Werte in der Datei setzen, plain
   `sbatch` ohne CLI-`--export`).
6. **Evaluieren:** `inference.py` (Proxy-Skala, 300 Fenster),
   `eval_full_val.py` (Vollvalidierung), `rollout_eval.py` (k=1..4).
   Die annotationsverankerten Metriken (mIoU/mAP gegen die
   nuScenes-Annotation) laufen per Latent-Injektion im
   BEVFusion-Container (`LOAD_BEV_LATENTS`-Hook, siehe `berichte/TASK20/21`).

## Figuren reproduzieren

Alle Thesis-Abbildungen entstehen aus den eingecheckten Ergebnis-JSONs/CSVs:

```bash
python3 scripts_render/render_seg_levers.py    # Beispiel: Übersicht der Modellvarianten
bash scripts_render/optimize_pdfs.sh           # PDF-Font-Subsetting
```

## Dokumentation

`berichte/TASK*_ABSCHLUSSBERICHT.md` dokumentieren jeden Arbeitsschritt
(Methodik, Jobs, Befunde, Lektionen) chronologisch — inklusive der
Negativergebnisse (Ego-Konditionierung, Selten-Klassen-Gewichtung,
Task-Loss-Kopplung), die die 0.69-Plateau-Analyse tragen.

## Lizenz

MIT-Lizenz (siehe [LICENSE](LICENSE)). nuScenes-Daten und
BEVFusion-Gewichte unterliegen den Lizenzen der jeweiligen Anbieter und
sind nicht Teil dieses Repositorys.
