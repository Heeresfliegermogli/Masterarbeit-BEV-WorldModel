# BEVFusion-Patch: Latent-Extraktion und Latent-Injektion

Dieses Verzeichnis enthält das Werkzeug, mit dem die BEV-Latents aus
BEVFusion extrahiert bzw. für die Evaluation wieder injiziert werden.
Es besteht aus zwei Teilen:

1. **`latent_saver.py`** — fertiges Modul, wird ins Wurzelverzeichnis
   des BEVFusion-Checkouts kopiert (neben `tools/`).
2. **Ein ~30-Zeilen-Hook** in
   `mmdet3d/models/fusion_models/bevfusion.py`, direkt **nach dem
   ConvFuser und vor dem Decoder** (siehe Code unten).

Beides stammt ursprünglich aus einer vorangegangenen Projektarbeit und
wird hier zur Reproduzierbarkeit mitgeliefert. BEVFusion selbst
(mit-han-lab/bevfusion, Apache 2.0) ist nicht Teil dieses Repos.

## Der Hook (in `BEVFusion.forward_single`, nach dem Fuser)

```python
        # =========================================================
        # Latent-Saver-Hook (Steuerung ueber Umgebungsvariablen)
        # =========================================================
        import os
        if not self.training and os.environ.get("SAVE_BEV_LATENTS", "0") == "1":
            from latent_saver import save_latent
            save_latent(x, metas)

        # =========================================================
        # Latent-Injektions-Hook: ersetzt den Fuser-Output durch ein
        # geladenes Latent (Roh-Skala!) — macht die nachgelagerte Kette
        # (Decoder -> Head -> Eval) zur Messmaschine fuer
        # Weltmodell-Vorhersagen.
        # Env: LOAD_BEV_LATENTS=1, LATENT_LOAD_DIR=<dir mit bev_latent_<tok>.npy>
        # Default aus -> Verhalten unveraendert. Nicht zusammen mit SAVE nutzen.
        # =========================================================
        if not self.training and os.environ.get("LOAD_BEV_LATENTS", "0") == "1":
            import numpy as np
            _dir = os.environ.get("LATENT_LOAD_DIR", "/output/latents_in")
            _tok = None
            if isinstance(metas, (list, tuple)) and len(metas) > 0:
                _tok = metas[0].get("sample_idx", None) or metas[0].get("token", None)
            _p = os.path.join(_dir, "bev_latent_{}.npy".format(_tok))
            _arr = np.load(_p).astype(np.float32)
            _t = torch.from_numpy(_arr).to(x.device)
            if _t.ndim == 3:
                _t = _t.unsqueeze(0)
            assert _t.shape == x.shape, \
                "latent_injection: shape {} != fuser {}".format(tuple(_t.shape), tuple(x.shape))
            x = _t
```

Einfügestelle: in `forward_single`, unmittelbar nach

```python
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
```

und vor `x = self.decoder["backbone"](x)`.

## Verwendung

**Extraktion** (je Split ein Verzeichnis einzelner `.npy`-Dateien;
Seg: 256×128×128, Det: 256×180×180, fp16):

```bash
SAVE_BEV_LATENTS=1 FLATTEN_BEV_LATENTS=0 LATENT_DTYPE=float32 \
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map
# Train-Split: zusaetzlich
#   --cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_train.pkl
```

**Injektion** (annotationsverankerte Auswertung von Vorhersagen —
Latents vorher auf Roh-Skala zurückrechnen, Det: Faktor 1/0,3641):

```bash
LOAD_BEV_LATENTS=1 LATENT_LOAD_DIR=/output/latents_in \
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map
```

Weitere Schalter von `latent_saver.py` (alle per Env): `LATENT_DTYPE`
(`float16`/`float32`), `FLATTEN_BEV_LATENTS` (räumliches Mittel statt
voller Karte), `LATENT_POOL`/`LATENT_POOL_FACTOR`/`LATENT_POOL_MODE`
(optionales räumliches Pooling). Das Zielverzeichnis ist fest
`/output/latents` (Docker-Mount des Containers); bei Bedarf in
`latent_saver.py` Zeile 46 anpassen.
