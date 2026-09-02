# BEV-Weltmodell — Vorhersage latenter BEV-Repräsentationen für das autonome Fahren

Transformer-basiertes Weltmodell, das die nächste latente BEV-Repräsentation
(kurz: Latent) aus den drei vorherigen Frames vorhersagt. Die Latents stammen
aus den **eingefrorenen BEVFusion-Konfigurationen** für Segmentierung und
Detektion (nuScenes), abgegriffen nach der Sensorfusion und vor den
Wahrnehmungsköpfen. Ausgewertet werden die Vorhersagen mit den jeweils
zugehörigen eingefrorenen Wahrnehmungsköpfen. Masterarbeit an der
Universität der Bundeswehr München.

## Kernergebnisse

| Metrik (Vollvalidierung, gegen die nuScenes-Annotation) | Persistenz | Weltmodell | Referenz (reales Latent) |
|---|---|---|---|
| Segmentierung (mIoU, 6 Klassen) | 0.4647 | **0.5898** | 0.6295 |
| Detektion (mAP) | 0.1696 | **0.3438** | 0.6858 |

- **Zielfunktion:** Für die Segmentierung reichen Smooth-L1 und der
  Streuungsterm aus (0.6946 gegenüber 0.6920 mit sechs Termen).
  In der Detektion verbessern die zusätzlichen Terme dagegen die mAP.
- **Rollout:** Das Weltmodell liegt für k=1..4 über der naiven und
  ego-kompensierten Persistenz.
- **Kopfadaptation:** Auf Vorhersagen nachtrainierte Wahrnehmungsköpfe
  verbessern die Segmentierung um 0.011 mIoU und die Detektion um
  0.089 mAP. Die Werte wurden über mehrere Trainingsläufe bestimmt.
- **Generative Varianten:** CVAE und Flow Matching erzeugen unterschiedliche
  Stichproben. Die Punktmetrik liegt dabei nicht über dem deterministischen
  Modell.
- **Laufzeit:** Ein Vorhersageschritt benötigt auf einer TITAN RTX etwa
  4,4 ms für die Segmentierung und 11,5 ms für die Detektion.
  Der höchste gemessene PyTorch-Speicherbedarf liegt bei 266 MB.

## Repository-Struktur

```
├── Masterarbeit_VincentMann.pdf   # die vollständige Masterarbeit
├── train_linux.py            # Training (eine Codebasis lokal + Cluster)
├── inference.py              # Proxy-Evaluation (Teilstichprobe mit 300
│                             #   Fenstern; mIoU, Streuungsverhältnis)
├── eval_full_val.py          # Vollvalidierung (5743 Fenster)
├── rollout_eval.py           # autoregressiver Rollout k=1..4
├── Code/                     # Modell-Module (Dataset, Embedding, Transformer,
│                             #   Output-/Upsampling-Head, Flow-Head, Loss)
└── examples/                 # Beispiel-Config (Training Segmentierung) und
                              #   Beispiel-Render-Skript (Abb. 5.8 der Thesis)
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
   Für die Evaluation gegen die nuScenes-Annotation wird dieser Hook
   benötigt. Ergebnis: je Split ein Verzeichnis einzelner
   `.npy`-Dateien (Seg: 256×128×128, Det: 256×180×180, fp16).
4. **Packen:** `python -u Code/pack_latents.py --config <config> --split val
   --dtype float16` (idempotent; memmap-fähige Packs für den Loader).
5. **Trainieren:** `python -u train_linux.py --config examples/config_seg_beispiel.yaml`
   (Pfade in der Config an die eigene Umgebung anpassen).
6. **Evaluieren:** `inference.py` (Proxy-Skala, 300 Fenster),
   `eval_full_val.py` (Vollvalidierung), `rollout_eval.py` (k=1..4).
   Die annotationsverankerten Metriken (mIoU/mAP gegen die
   nuScenes-Annotation) laufen per Latent-Injektion im
   BEVFusion-Container (`LOAD_BEV_LATENTS`-Hook).

## Beispiel: Thesis-Figur rendern

`examples/` zeigt den Aufbau der Thesis-Figuren (gemeinsames Stylesheet
`thesis_style.py`, kleine Ergebnis-JSONs als Datenquelle) am Beispiel
von Abbildung 5.8 (mIoU je Klasse):

```bash
python3 examples/render_beispiel_klassen.py
```

## Dokumentation

Weitere Details zu Methodik und Experimenten stehen in der Masterarbeit:
[`Masterarbeit_VincentMann.pdf`](Masterarbeit_VincentMann.pdf).

## Lizenz

MIT-Lizenz (siehe [LICENSE](LICENSE)). nuScenes-Daten und
BEVFusion-Gewichte unterliegen den Lizenzen der jeweiligen Anbieter und
sind nicht Teil dieses Repositorys.
