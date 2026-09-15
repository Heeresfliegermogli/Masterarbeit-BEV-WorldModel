# tools/ — Kopf-Adaptation

Nachtrainieren der eingefrorenen Wahrnehmungsköpfe auf
Weltmodell-Vorhersagen (Weltmodell und Encoder bleiben unverändert;
Thesis, Kapitel Kopfadaptation). Alle drei Skripte laufen **im
BEVFusion-Docker-Container** (siehe bevfusion_patch/ und README-Root).

| Skript | Zweck |
|---|---|
| `mk_gt_masks_seg.py` | rastert die nuScenes-Map-GT ins Zielgitter (Voraussetzung für die Seg-Adaptation) |
| `adapt_seg_head.py` | trainiert den Seg-Decoder-Stack (SECOND + FPN + SegHead) auf (Vorhersage-Latent → Map-GT) |
| `adapt_det_head.py` | trainiert den TransFusion-Kopf auf Vorhersage-Latents; mit Real-Mix und Val-Loss-Selektion |

Reihenfolge Seg: erst `mk_gt_masks_seg.py`, dann `adapt_seg_head.py`.
Ergebnisse der Thesis: Seg +0,011 mIoU, Det +0,089 mAP (Vollvalidierung,
mehrfach geseedet).
