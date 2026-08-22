# Task 21 — Seg-GT-Verankerung (Abschlussbericht, 31.07.)

Datum: 2026-07-30 bis 31. Ausloeser: Betreuer-Input (Leon Pohl) — unsere
Seg-Kernmetrik war selbstreferenziell (mIoU gegen decode(reales Latent t+1)
= Pseudo-GT des frozen Decoders), waehrend der Det-Strang gegen echte
GT-Boxen misst. Task 21 verankert den Seg-Strang an der ECHTEN
nuScenes-Map-GT — mit derselben Injection-Messmaschine wie Task 20.

## Protokoll

- Injection-Hook (bevfusion.py, shape-agnostisch — asserted zur Laufzeit
  gegen die Fuser-Shape) + tools/test.py --eval map,
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml + bevfusion-seg.pth.
- Metrik: IoU@max (Maximum ueber 7 Schwellen 0.35-0.65, BEVFusion-Konvention),
  6 Map-Klassen, Full-Val 6019 Samples, 276 kontextlose Frames real gefuellt
  (Luecken-Konvention identisch 20.1).
- SHAPE-KORREKTUR: Seg-Latents sind (1,256,128,128) fp32 — 256 Kanaele,
  nicht 512 (alte CLAUDE.md-Angabe war falsch; konsistent mit der
  d_model=256-Kopplung aus 16f).
- 21.0 ORAKEL-GATE bestanden: injizierte reale Latents -> mIoU 0.62946 vs
  Task-12-Referenz 0.62947 (identisch bis 4. Nachkommastelle, alle 6
  Klassen) -> Messkette bewiesen.
- 21.1 Dumps beider Headline-Modelle (16b.9: smoothl1_minimal + baseline,
  best_miou.pt) via eval_dump_latents.py (generisch; latent_scale=1.0)
  nach BeeGFS dumps_task21/ (Job 125598; Platz-Vorpruefung aktiv).
  Persistenz-Baseline als Symlink-Konstruktion (mk_pers_symlinks_seg.py,
  Szenen-Split identisch bev_dataset: 92 Szenen, 5743+276).
- 21.2 vier sequenzielle Injection-Evals lokal (Docker seg_gate, zweiter
  Container mit ro-Mount der realen Latents; Logs eval_seg_*.log).

## Ergebnisse (IoU@max vs. echte nuScenes-GT, Full-Val 6019)

| Variante   | mean  | drivable | ped_cross | walkway | stop_line | carpark | divider |
|------------|------:|---------:|----------:|--------:|----------:|--------:|--------:|
| Persistenz | 0.4647| 0.7627   | 0.3421    | 0.5105  | 0.3102    | 0.4757  | 0.3868  |
| WM minimal | 0.5898| 0.8341   | 0.5618    | 0.6281  | 0.4746    | 0.5637  | 0.4761  |
| WM 6-Term  | 0.5882| 0.8338   | 0.5599    | 0.6284  | 0.4711    | 0.5594  | 0.4763  |
| Orakel     | 0.6295| 0.8535   | 0.6074    | 0.6734  | 0.5114    | 0.5951  | 0.5360  |

## Befunde

BEFUND 1 — PSEUDO-GT-METRIK VALIDIERT: Die Rangfolge bleibt unter echter GT
exakt erhalten: Minimal (0.5898) ~ 6-Term (0.5882, d=+0.0016 zugunsten
Minimal, gleiche Groessenordnung wie die +0.0026 n.s. der Pseudo-GT-Metrik)
>> Persistenz (0.4647). "from six to two" gilt auf Seg auch gegen echte GT
— die metrik-spezifische Grenze aus 20.3 (Det braucht 6 Terme) bleibt
davon unberuehrt und wird sogar geschaerft: gleiche Messkette, gleicher
Injection-Weg, nur die Zielmetrik unterscheidet die Empfehlung.

BEFUND 2 — SELTEN-KLASSEN-EINBRUCH = UEBERWIEGEND DECODER-DECKE: Retention
(Minimal/Orakel) je Klasse: drivable 97.7% | carpark 94.7% | walkway 93.3%
| stop_line 92.8% | ped_crossing 92.5% | divider 88.8% | mean 93.7%.
Das World-Model verliert also je Klasse nur 2-11% relativ zum Orakel; dass
stop_line absolut bei 0.47 liegt, ist zu ~75% die Schwaeche des frozen
BEVFusion-Decoders selbst (Orakel stop_line nur 0.5114 vs drivable 0.8535).
Der groesste WM-eigene Verlust sitzt bei DUENNEN Strukturen (divider 11%,
ped_crossing 7.5%) — konsistent mit der IoU-Randempfindlichkeits-Analyse
aus Task 19.

BEFUND 3 — EHRLICHE EINORDNUNG DER 0.69: Pseudo-GT-mIoU 0.6946 und
GT-mIoU 0.5898 messen verschiedene Dinge (WM-Fehler allein vs WM-Fehler
x Decoder-Fehler). Fuer die Thesis: beide berichten, Verkettung erklaeren;
das Modell erreicht 93.7% der Wahrnehmungs-Decke.

## Konsequenz fuer den Betreuer-Vorschlag (CE/Weighted CE/Focal/Balancing)

Headroom (Orakel minus Minimal) je Klasse: drivable 0.019, carpark 0.031,
stop_line 0.037, walkway 0.045, ped_crossing 0.046, divider 0.060; im
Mittel 0.040. Klassengewichtung im World-Model kann NUR diesen Anteil
adressieren (der Rest ist Decoder-Decke, frozen). Dazu kommt 16f-H1:
Task-Loss durch den Decoder schadete monoton (ungewichtete BCE). Ein
Task 22 (gewichtete Latent-Loss auf Selten-Klassen-Regionen / Balanced
Sampler / Focal-Task-Loss) haette als realistisches Ziel divider+
ped_crossing-Retention Richtung 95% (~+0.01-0.02 mean-GT-mIoU) — moeglich,
aber begrenzt; Entscheidung beim Nutzer.

## Artefakte

- predictions/task21/gt_eval.json (alle Zahlen + Retention/Headroom)
- visualizations/21_seg_gt_korridor.{png,pdf} (Korridor + per-Klasse)
- config_seg_dump_eval.yaml, sbatch_21_dump_seg.sh, mk_pers_symlinks_seg.py,
  render_21_korridor.py
- BeeGFS: dumps_task21/dump_seg_{minimal,baseline}; lokal (temporaer):
  det_latents_out/dump_seg_*, latents_seg_pers, eval_seg_*.log,
  gate_seg_oracle.log; Docker-Container seg_gate (ro-Mount der Latents).

## Lektionen

- pgrep-Selbstmatch: ein Ketten-Wächter, dessen eigenes Kommando das
  gesuchte Muster ALS TEXT enthaelt, blockiert sich selbst — Muster so
  waehlen, dass es die eigene Kommandozeile nicht matcht.
- Hardlinks auf root-eigene Dateien scheitern (fs.protected_hardlinks);
  Loesung: zweiter Container mit zusaetzlichem ro-Bind-Mount.
- Die Injection-Messmaschine ist jetzt fuer BEIDE Straenge kanonisch:
  ein Hook, zwei Configs, echte GT auf beiden Seiten.
