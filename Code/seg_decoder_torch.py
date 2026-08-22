"""
seg_decoder_torch.py — BEVFusion-Seg-Decoder als reines PyTorch
================================================================================

ZWECK
-----
Decodiert einen BEV-Seg-Latent [256,128,128] zu einer Segmentierungs-Maske
[6,200,200] — OHNE mmdet3d / mmcv. Damit kann die echte mIoU direkt in der
Trainingsumgebung (PyTorch 2.1.2) gemessen werden, statt im BEVFusion-Docker.

Der Decoder ist ein 1:1-Nachbau dessen, was inference_seg.py im
Docker über die mmdet3d-Builder zusammensetzt:

    Latent [256,128,128]
       -> SECOND backbone     -> ([128,128,128], [256,64,64])
       -> SECONDFPN neck       -> [512,128,128]
       -> BEVGridTransform     -> [512,200,200]
       -> Classifier           -> [6,200,200] logits
       -> sigmoid + threshold  -> bool-Maske [6,200,200]

Die mmdet3d-abhängigen Teile (SECOND, SECONDFPN) sind hier als schlichte
nn.Module nachgebaut, sodass die Gewichts-Keys EXAKT denen im Checkpoint
entsprechen und mit strict=True geladen werden koennen. BEVGridTransform und
Classifier sind bereits in inference_seg.py reines PyTorch und werden
uebernommen.

MODULARITAET
------------------------------
render_mask() und save_comparison() sind hier mit enthalten, werden aber von
der Trainings-Validierung NICHT aufgerufen. So kann die Visualisierung die PNG-Ausgabe
(pred/real/comparison) ohne Code-Duplikat einfach importieren und einhaengen.

GEWICHTS-KEYS (Mapping Checkpoint -> Submodul)
----------------------------------------------
    decoder.backbone.*          -> SECONDBackbone   (self.blocks.*)
    decoder.neck.*              -> SECONDFPNNeck     (self.deblocks.*)
    heads.map.classifier.*      -> classifier        (nn.Sequential, 0..6)
    (BEVGridTransform hat keine lernbaren Parameter — reine Geometrie)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reihenfolge & Schwelle EXAKT wie inference_seg.py. Nicht aendern,
# sonst stimmen die mIoU-Zahlen nicht mehr mit der Baseline ueberein.
MAP_CLASSES = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]
MAP_SCORE = 0.5

MAP_PALETTE = {
    'drivable_area': (166, 206, 227),
    'ped_crossing':  (251, 154, 153),
    'walkway':       (227,  26,  28),
    'stop_line':     (253, 191, 111),
    'carpark_area':  (255, 127,   0),
    'divider':       (106,  61, 154),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECOND backbone (Decoder-Backbone) — reiner PyTorch-Nachbau
# ─────────────────────────────────────────────────────────────────────────────
# Aequivalent zu mmdet3d SECOND mit:
#   in_channels=256, out_channels=[128,256], layer_nums=[5,5],
#   layer_strides=[1,2], norm=BN(eps=1e-3, momentum=0.01), conv bias=False
#
# Aufbau pro Block i:
#   Conv(in->out, 3, stride=layer_strides[i], pad=1, bias=False), BN, ReLU
#   dann layer_nums[i]x:  Conv(out->out,3,pad=1,bias=False), BN, ReLU
# Keys: blocks.{i}.{0..(3+3*layer_num-1)}  -> matcht decoder.backbone.*
class SECONDBackbone(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=(128, 256),
                 layer_nums=(5, 5),
                 layer_strides=(1, 2),
                 bn_eps=1e-3,
                 bn_momentum=0.01):
        super().__init__()
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            layers = [
                nn.Conv2d(in_filters[i], out_channels[i], 3,
                          stride=layer_strides[i], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[i], eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                layers.append(nn.Conv2d(out_channels[i], out_channels[i], 3,
                                        padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels[i], eps=bn_eps,
                                             momentum=bn_momentum))
                layers.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)


# ─────────────────────────────────────────────────────────────────────────────
# SECONDFPN neck — reiner PyTorch-Nachbau
# ─────────────────────────────────────────────────────────────────────────────
# Aequivalent zu mmdet3d SECONDFPN mit:
#   in_channels=[128,256], out_channels=[256,256], upsample_strides=[1,2],
#   upsample_cfg=deconv(bias=False), use_conv_for_no_stride=True,
#   conv_cfg=Conv2d(bias=False), norm=BN(eps=1e-3, momentum=0.01)
#
# stride==1 + use_conv_for_no_stride=True -> Conv2d(k=1,s=1) statt Deconv
# stride==2                               -> ConvTranspose2d(k=2,s=2)
# Keys: deblocks.{i}.{0,1}  -> matcht decoder.neck.*
class SECONDFPNNeck(nn.Module):
    def __init__(self,
                 in_channels=(128, 256),
                 out_channels=(256, 256),
                 upsample_strides=(1, 2),
                 bn_eps=1e-3,
                 bn_momentum=0.01):
        super().__init__()
        self.in_channels = in_channels
        deblocks = []
        for i, out_c in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                up = nn.ConvTranspose2d(in_channels[i], out_c,
                                        kernel_size=stride, stride=stride,
                                        bias=False)
            else:
                # use_conv_for_no_stride=True -> 1x1-Conv statt Upsampling
                k = int(round(1.0 / stride))
                up = nn.Conv2d(in_channels[i], out_c, kernel_size=k, stride=k,
                               bias=False)
            deblocks.append(nn.Sequential(
                up,
                nn.BatchNorm2d(out_c, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        assert len(x) == len(self.in_channels), \
            f"Neck erwartet {len(self.in_channels)} Feature-Maps, bekam {len(x)}"
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        out = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return [out]


# ─────────────────────────────────────────────────────────────────────────────
# BEVGridTransform — 1:1 aus inference_seg.py (reine Geometrie, keine Gewichte)
# ─────────────────────────────────────────────────────────────────────────────
class BEVGridTransform(nn.Module):
    def __init__(self, *, input_scope, output_scope, prescale_factor=1):
        super().__init__()
        self.input_scope     = input_scope
        self.output_scope    = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x):
        if self.prescale_factor != 1:
            x = F.interpolate(x, scale_factor=self.prescale_factor,
                              mode='bilinear', align_corners=False)
        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
                self.input_scope, self.output_scope):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))
        u, v = torch.meshgrid(coords, indexing='ij')
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Gekapselter Decoder: Latent -> (logits/probs/mask)
# ─────────────────────────────────────────────────────────────────────────────
class SegDecoder(nn.Module):
    """
    Vollstaendiger eingefrorener Seg-Decoder. Nach build_seg_decoder() bereit
    fuer reine Forward-Inferenz. Traegt keine Gradienten und wird im Training
    nie aktualisiert.
    """
    def __init__(self, map_classes=MAP_CLASSES):
        super().__init__()
        self.map_classes = list(map_classes)
        n_cls = len(self.map_classes)

        self.backbone = SECONDBackbone()
        self.neck     = SECONDFPNNeck()
        self.transform = BEVGridTransform(
            input_scope =[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope=[[-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]],
        )
        # Classifier EXAKT wie inference_seg.py / BEVSegmentationHead
        # (BN-Default-eps hier, NICHT 1e-3 — bewusst wie im Original).
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, n_cls, 1),
        )

    def _forward_logits_impl(self, latent_bchw):
        """Gemeinsamer Kern (OHNE no_grad-Dekorator, damit der Task-Loss
        Gradienten DURCH den eingefrorenen Decoder zum Latent fliessen lassen
        kann; die Decoder-Parameter selbst bleiben frozen/eval)."""
        x = self.backbone(latent_bchw)     # tuple
        x = self.neck(x)                    # [ [B,512,128,128] ]
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.transform(x)               # [B,512,200,200]
        x = self.classifier(x)              # [B,n_cls,200,200]
        return x

    @torch.no_grad()
    def forward_logits(self, latent_bchw):
        """latent_bchw: [B,256,128,128] float32 -> logits [B,n_cls,200,200]"""
        return self._forward_logits_impl(latent_bchw)

    def forward_logits_grad(self, latent_bchw):
        """Wie forward_logits, aber MIT Gradientenfluss (Decoder-Task-Loss).
        Aufrufer muss float32 sichern (autocast aus) — wie latent_to_mask."""
        return self._forward_logits_impl(latent_bchw)

    @torch.no_grad()
    def latent_to_mask(self, latent, device=None, threshold=MAP_SCORE):
        """
        latent: np.ndarray ODER torch.Tensor, [256,128,128] oder [B,256,128,128]
        returns: (mask_bool, probs)  je [B,n_cls,200,200] bzw. ohne B wenn Einzel
        """
        single = False
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent.astype(np.float32))
        latent = latent.float()
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
            single = True
        if device is not None:
            latent = latent.to(device)

        # WICHTIG: autocast hier explizit AUS -> float32, wie Docker-Referenz.
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.forward_logits(latent)
            probs  = torch.sigmoid(logits)

        mask = (probs >= threshold)
        probs = probs.detach().cpu().numpy()
        mask  = mask.detach().cpu().numpy()
        if single:
            return mask[0], probs[0]
        return mask, probs


# ─────────────────────────────────────────────────────────────────────────────
# Gewichte laden (strict=True ist der eigentliche Korrektheits-Beweis)
# ─────────────────────────────────────────────────────────────────────────────
def build_seg_decoder(checkpoint_path, device, map_classes=MAP_CLASSES,
                      verbose=True):
    """
    Baut den SegDecoder und laedt die Gewichte aus bevfusion-seg.pth.
    strict=True bei jedem Submodul: wenn auch nur ein Key-Name oder eine Shape
    nicht stimmt, fliegt hier ein Fehler -> staerkster Architektur-Check.
    """
    dec = SegDecoder(map_classes=map_classes)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    def _sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    bb_sd  = _sub('decoder.backbone.')
    neck_sd = _sub('decoder.neck.')
    cls_sd = _sub('heads.map.classifier.')

    dec.backbone.load_state_dict(bb_sd, strict=True)
    dec.neck.load_state_dict(neck_sd, strict=True)
    dec.classifier.load_state_dict(cls_sd, strict=True)

    if verbose:
        print(f"[seg_decoder] backbone   keys: {len(bb_sd)}")
        print(f"[seg_decoder] neck       keys: {len(neck_sd)}")
        print(f"[seg_decoder] classifier keys: {len(cls_sd)}")
        # Sanity: letzte Conv muss n_cls Ausgaben haben
        last_w = cls_sd.get('6.weight')
        if last_w is not None:
            assert last_w.shape[0] == len(map_classes), \
                f"Klassen-Mismatch: ckpt {last_w.shape[0]} vs {len(map_classes)}"
            print(f"[seg_decoder] Klassen-Check OK: {len(map_classes)} -> {map_classes}")
        print(f"[seg_decoder] strict-load erfolgreich -> Architektur korrekt.")

    dec = dec.to(device).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


# ─────────────────────────────────────────────────────────────────────────────
# IoU — 1:1 aus inference_seg.py
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou(pred_mask, real_mask, classes=MAP_CLASSES):
    """pred_mask/real_mask: bool [n_cls,200,200] -> dict{klasse: iou|None, mIoU}"""
    results = {}
    valid = []
    for c, name in enumerate(classes):
        inter = np.logical_and(pred_mask[c], real_mask[c]).sum()
        union = np.logical_or(pred_mask[c], real_mask[c]).sum()
        if union > 0:
            iou = float(inter) / float(union)
            results[name] = round(iou, 4)
            valid.append(iou)
        else:
            results[name] = None
    results['mIoU'] = round(float(np.mean(valid)), 4) if valid else 0.0
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Render-Helfer — nur Visualisierung (von der Validierung NICHT aufgerufen)
# ─────────────────────────────────────────────────────────────────────────────
def render_mask(mask, classes=MAP_CLASSES, background=(240, 240, 240)):
    """mask: bool [C,H,W] -> RGB uint8 [H,W,3]"""
    canvas = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    canvas[:] = background
    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[mask[k]] = MAP_PALETTE[name]
    return canvas


def save_comparison(pred_mask, real_mask, fpath, classes=MAP_CLASSES):
    """pred | real | diff nebeneinander als PNG (benoetigt cv2)."""
    import cv2
    pred_img = render_mask(pred_mask, classes)
    real_img = render_mask(real_mask, classes)
    H, W = pred_mask.shape[1], pred_mask.shape[2]
    diff = np.full((H, W, 3), 240, dtype=np.uint8)
    both_active = np.any(real_mask, axis=0)
    agree       = np.all(pred_mask == real_mask, axis=0)
    diff[agree & both_active] = (100, 200, 100)
    diff[~agree]              = (220,  50,  50)
    sep = np.full((H, 4, 3), 100, dtype=np.uint8)
    combined = np.concatenate([pred_img, sep, real_img, sep, diff], axis=1)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    cv2.imwrite(fpath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Selbsttest (ohne echte Gewichte): Architektur-Verdrahtung + Key-Namen
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Selbsttest seg_decoder_torch (Random-Gewichte) ===")
    dec = SegDecoder()
    dec.eval()

    # 1) Shapes durch die ganze Kette
    x = torch.randn(2, 256, 128, 128)
    bb = dec.backbone(x)
    print("backbone out shapes:", [tuple(t.shape) for t in bb])
    assert tuple(bb[0].shape) == (2, 128, 128, 128)
    assert tuple(bb[1].shape) == (2, 256, 64, 64)
    nk = dec.neck(bb)
    print("neck out shape:     ", tuple(nk[0].shape))
    assert tuple(nk[0].shape) == (2, 512, 128, 128)
    tr = dec.transform(nk[0])
    print("transform out shape:", tuple(tr.shape))
    assert tuple(tr.shape) == (2, 512, 200, 200)
    lo = dec.classifier(tr)
    print("classifier out:     ", tuple(lo.shape))
    assert tuple(lo.shape) == (2, 6, 200, 200)

    # 2) latent_to_mask End-to-End
    m, p = dec.latent_to_mask(np.random.randn(256, 128, 128).astype(np.float32),
                              device='cpu')
    print("mask shape:", m.shape, "dtype:", m.dtype, "| probs:", p.shape)
    assert m.shape == (6, 200, 200) and m.dtype == bool

    # 3) compute_iou plausibel (gleiche Maske -> mIoU 1.0)
    iou = compute_iou(m, m)
    print("self-IoU (muss 1.0 sein wo union>0):", iou['mIoU'])

    # 4) State-Dict-Keys ausgeben (muessen zum Checkpoint-Mapping passen)
    print("\n--- backbone keys (erste 6) ---")
    for k in list(dec.backbone.state_dict().keys())[:6]:
        print("  ", k)
    print("--- neck keys ---")
    for k in dec.neck.state_dict().keys():
        print("  ", k)
    print("--- classifier keys ---")
    for k in dec.classifier.state_dict().keys():
        print("  ", k)

    print("\n[OK] Architektur-Verdrahtung & Key-Namen plausibel.")
