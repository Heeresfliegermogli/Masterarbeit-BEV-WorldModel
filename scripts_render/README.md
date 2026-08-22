# scripts_render — Figuren-Reproduktion

Alle Skripte AUS DEM PROJEKTROOT ausfuehren:

    python3 scripts_render/<skript>.py

Die relativen Pfade zeigen dann unveraendert auf visualizations/ und
predictions/ im Projektroot. `thesis_style.py` ist das gemeinsame
Stylesheet (Druckbreiten-Logik, Okabe-Ito, Umlaut-Normalisierung);
`optimize_pdfs.sh` subsettet danach die Fonts aller PDFs.

## Skript -> Figuren

| Skript | Figuren (visualizations/) |
|---|---|
| synth_loss_sweeps.py | 16b9_tradeoff, 16b9_dose_response |
| render_legacy_figs.py | 16c_rollout, 16d_ego_persistence, 17_miou_over_epochs, 17_stdgap_beforeafter |
| render_ego_conditioning.py | 16d_ego_conditioning |
| render_context_ema.py | 16e_miou_lever |
| render_capacity_levers.py | 16f_miou_lever2 |
| render_thesis.py | 17_gate_alpha, 17_masks_overview |
| filmstrip_render.py | 17_filmstrip_statisch, 17_filmstrip_dynamisch |
| render_vae_panel.py | 18_vae_panel_* (CVAE + Flow) |
| render_flow_calibration.py | 18_flow_calibration |
| render_flow_rollout.py | 18_flow_rollout |
| collect_master_table.py | (kein Plot — baut predictions/task19/master_table.csv) |
| render_seg_levers.py | 19_all_levers, 19_per_class |
| render_det_korridor.py | 20_det_korridor |
| render_det_boxes.py | 20_det_boxes_panel |
| render_seg_gt_korridor.py | 21_seg_gt_korridor |
| render_error_decomposition.py | 21_error_decomposition |
| render_decoder_panel.py | 23_decoder_panel |
| render_det_boxes_adapted.py | 24_det_boxes_panel |
| render_det_boxes_peds.py | 24_det_boxes_peds |
| render_det_boxes_versions.py | 25_det_boxes_versions |
| render_head_comparison.py | 25_head_comparison |
| render_ressourcen.py | ressourcen_inferenzzeit |
| render_pareto.py | pareto |
| render_thesis_synthesis.py | korridor_overview, rueckholquote |

Hinweis: Die Figuren-DATEINAMEN tragen weiterhin die historischen
Tasknummern-Praefixe, weil die Thesis-LaTeX sie unter diesen Namen
einbindet. Die Zuordnung Task -> Inhalt steht in berichte/.
