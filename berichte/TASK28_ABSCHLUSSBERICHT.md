# TASK 28 — GitHub-Upload des Projekts

## 28.1 Repo-Abgrenzung + Aufraeumen (16.08., ERLEDIGT)

AUFRAEUMEN (verschoben, NICHTS geloescht; lokal + Cluster identisch):
- 13 ausgemusterte Root-Configs -> archiv/configs/ (config_base,
  config_det_smoke_local, config_nuscenes_full* [7 inkl. SANITYCHECK/
  cluster-Varianten], config_nuscenes_val, config_nuscenes_task15,
  config_27_flow_local). Alle nur noch aus archiv/-Skripten bzw.
  Docstrings referenziert (per grep verifiziert, nichts bricht).
- Alt-Skripte Task-5/6-Aera -> archiv/tools/: train_base.py,
  visualize_predictions.py, smoke_test.py (durch train_linux.py bzw.
  scripts_render/ abgeloest).
- Streu-Logs (anchor/fp16_run/pack_*) -> logs/; Handout-PDFs ->
  berichte/. Cluster-Streuner einsortiert: adapt_seg_head.py
  (bit-identisches Duplikat, dedupe) + render_ressourcen.py-ALTVERSION
  im Cluster-Root -> archiv/tools/render_ressourcen_alt_clusterroot.py
  (aktuelle Version in scripts_render/ blieb unberuehrt, md5-geprueft).
- Root enthaelt jetzt NUR: 5 Orchestrierungs-Skripte (train_linux,
  inference, eval_full_val, rollout_eval, Arbeitsplan), 10 aktive
  Basis-/Eval-Configs, CLAUDE.md. scripts_render/ blieb komplett
  (= Reproduktionspfad aller Thesis-Figuren).

.GITIGNORE (Repo-Abgrenzung):
- Raus: checkpoints/, logs/, wandb/, alle *.pt/*.pth/*.h5/*.npy/*.npz,
  latents*/dumps*-Verzeichnisse (nuScenes/BEVFusion-Redistribution
  verboten), __pycache__, *.log, .claude/.
- predictions/: Dumps raus, aber kleine Ergebnis-JSONs/CSVs REIN
  (Whitelist-Muster) — Datengrundlage ALLER scripts_render/-Figuren,
  damit sind die Abbildungen aus dem Repo reproduzierbar (~20 MB;
  einzige Ausnahme task16a-trace.json ~13 MB Profiler-Rohdaten).
- Trocken-Test mit temporaerem Git-Index: 404 Dateien / 20 MB,
  Stichprobe sauber (keine Gewichte/Latents/Logs). Verteilung:
  archiv 120, predictions(JSON/CSV) 90, visualizations 76,
  berichte 53, scripts_render 27, Code 16, Root 21.

HINWEIS fuer 28.3 (macht Vincent manuell): CLAUDE.md + berichte/
enthalten Klarnamen und Cluster-Pfade — fuer ein PRIVATES Repo ok,
vor einem oeffentlichen Repo putzen oder weglassen.

OFFEN: 28.2 README.md + requirements.txt (+ Lizenz nach Entscheidung),
28.3 git init/push (Vincent manuell).

## 28.2 README + requirements (16.08., ERLEDIGT)
README.md (deutsch): Projektueberblick, Kernergebnis-Tabelle (GT-verankerte
Korridore Seg/Det), Befund-Highlights (from-six-to-two metrikabhaengig,
Rollout, Decoder-Adaptation, generative Koepfe, Ressourcen), Repo-Struktur,
Setup, Daten-/Gewichte-Kapitel (nuScenes/BEVFusion NICHT im Repo,
5-Schritt-Reproduktion), Figuren-Reproduktion, Berichte-Hinweis.
requirements.txt aus bevwm-Referenz-Env (Cluster: py3.10.20, torch 2.1.2
+cu121, numpy 1.26.4, h5py 3.16, pyyaml 6, wandb 0.28; matplotlib/pandas
als optionale Render-Gruppe markiert — im Cluster-Env bewusst absent).
Beide Dateien nach head:~/Code_final gesynct. LIZENZ weiter OFFEN
(Entscheidung Vincent), 28.3 manuell.

## 28.2b Renderskripte task-frei benannt (16.08., ERLEDIGT)
18 Skripte in scripts_render/ von Task-Nummern auf INHALTLICHE Namen
umbenannt (lokal + Cluster per mv, nichts geloescht); Task-Prosa aus
allen Headern/Kommentaren entfernt. Mapping (alt -> neu):
collect_task19->collect_master_table, 16d->render_ego_conditioning,
16e->render_context_ema, 16f->render_capacity_levers,
18_calibration->render_flow_calibration, 18_panel->render_vae_panel,
18_rollout->render_flow_rollout, 19->render_seg_levers,
20_boxes->render_det_boxes, 20_korridor->render_det_korridor,
21_decomposition->render_error_decomposition,
21_korridor->render_seg_gt_korridor, 23_panel->render_decoder_panel,
24_boxes->render_det_boxes_adapted, 24_boxes_peds->render_det_boxes_peds,
25_boxes_versions->render_det_boxes_versions,
25_heads->render_head_comparison, synth_16b9->synth_loss_sweeps.
scripts_render/README.md: neuer Skript->Figuren-Index.
FIGUR-FIX nebenbei (16f_miou_lever2): Balkenlabels "task 0.1/0.5/1.0"
-> "lambda_task 0.1/0.5/1.0", "cap_l8" -> "8 Layer (Kapazitaet)",
"residual" -> "hartes Residual" (letzte Task-/Codename-Reste IM Bild);
neu gerendert + optimiert.
BEWUSST NICHT umbenannt: (a) Figuren-DATEINAMEN (16b9_*, 21_* ...) —
in der Thesis-LaTeX eingebunden, Umbenennung nur koordiniert mit der
Schreibsession (Mapping-Tabelle auf Zuruf); (b) predictions/task*/-
Datenpfade und berichte/TASK*-Dateien (Labor-Journal, historisch).
Alle 26 Skripte py_compile-gruen nach Umbenennung.

## 28.2c Task-Erwaehnungen aus der allgemeinen Codebasis entfernt (16.08.)
Sweep ueber train_linux.py, inference.py, eval_full_val.py,
rollout_eval.py, Code/*.py und alle 10 Root-Configs: saemtliche
"Task N"-Prosa UND nackte Task-IDs (16b.9, 15.3B, 22.1 ...) aus
Kommentaren/Docstrings/Headern entfernt bzw. inhaltlich umformuliert
(z.B. "15.3-Arbeitspunkt" -> "etablierter Arbeitspunkt", "16b.9
MINIMAL" -> "MINIMAL", "15.2-Rauschboden" -> "Rauschboden
(Seed-Variabilitaet)"). Log-Praefixe umbenannt: "[Task 14]" ->
"[seg_decoder]", "[16f]" -> "[task_loss]", "[16a.0]" -> "[summary]"
(kein aktives Skript parst diese Praefixe — geprueft).
UNBERUEHRT: Fachbegriff "Task-Loss"/lambda_task/task_loss_every;
funktionale Pfade (checkpoints/task*, predictions/task*, dumps_task*);
archiv/ (historisch, bleibt wie es ist); Arbeitsplan.py (IST der
Task-Plan) und berichte/TASK* (Labor-Journal).
VERIFIKATION: AST-Vergleich alt/neu mit maskierten String-Literalen
-> alle 19 Python-Dateien STRUKTURIDENTISCH (nur Kommentare/Strings
geaendert); alle 10 YAMLs geparst WERTGLEICH -> bit-identisches
Verhalten garantiert. py_compile gruen. Cluster gesynct.

## 28.2d README-Ergaenzung Links + latent_saver-Herkunft (16.08.)
Daten-Kapitel ueberarbeitet: expliziter nuScenes-Link
(https://www.nuscenes.org/nuscenes, Registrierung/Lizenz-Hinweis) +
offizielles BEVFusion-Repo (https://github.com/mit-han-lab/bevfusion,
bevfusion-seg.pth/-det.pth). KORREKTUR: latent_saver-Hook korrekt als
EXTERNES Werkzeug aus der vorangegangenen Projektarbeit ausgewiesen
(Patch in bevfusion-main/ + latent_saver.py im Container, Env-Schalter
SAVE_/LOAD_BEV_LATENTS) — NICHT Teil dieses Repos; die fruehere
archiv/tools/-Angabe im README war falsch (dort liegt nur der
WM-seitige eval_dump_latents.py). Beleg: TASK12-Bericht behandelt den
Mechanismus als bestehend; Hook liegt in bevfusion-main/mmdet3d/.
