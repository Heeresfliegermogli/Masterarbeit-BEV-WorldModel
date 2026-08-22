"""
upsampling_head.py — BEV World Model: Upsampling Head
==========================================================================

Skaliert das 32×32 Feature-Grid zurück auf die originale BEV-Auflösung
128×128 — identisch für Phase 1 und Phase 2.

KONTEXT IN DER PIPELINE:
    ...
    Output Head      [B, 256, 32, 32]   ← output_head.py
         ↓
    Upsampling Head  [B, 256, 128, 128] ← DIESE DATEI
         ↓
    Predicted Latent [B, 256, 128, 128] ← MSE gegen echten t+1

WARUM DIESER SCHRITT?
    Das Downsampling am Anfang der Pipeline hat aus [128,128] ein [32,32]
    Grid gemacht (AvgPool, Faktor 4). Der Output Head bringt uns wieder
    auf [32,32]. Jetzt müssen wir die originale Auflösung wiederherstellen:
    [32,32] → [128,128], also wieder Faktor 4.

    Warum nicht einfach bilinear interpolieren?
    Bilinear Upsample kopiert/interpoliert nur — er kann keine neuen
    Feature-Strukturen lernen. ConvTranspose2d hingegen hat lernbare
    Gewichte: er kann echte räumliche Details rekonstruieren, nicht nur
    glätten. Das ist entscheidend weil der MSE-Loss direkt auf den
    128×128 Latents gemessen wird.

IMPORTS FÜR ANDERE MODULE:
    from upsampling_head import UpsamplingHead
"""

import torch
import torch.nn as nn

from config import ModelConfig


# ===========================================================================
# UpsamplingHead — Phase 1 und Phase 2 identisch
# ===========================================================================

class UpsamplingHead(nn.Module):
    """
    Skaliert [B, 256, 32, 32] → [B, 256, 128, 128] mit lernbaren Schichten.

    ARCHITEKTUR — 3 Schichten:

        ConvTranspose2d(256→256, kernel=4, stride=2, padding=1)
        + BatchNorm2d(256) + GELU
            → [B, 256, 64, 64]    (×2 in jeder Spatial-Dim)

        ConvTranspose2d(256→256, kernel=4, stride=2, padding=1)
        + BatchNorm2d(256) + GELU
            → [B, 256, 128, 128]  (nochmal ×2)

        Conv2d(256→256, kernel=3, padding=1)
            → [B, 256, 128, 128]  (Shape unverändert — Feintuning)

    WARUM ZWEI ConvTranspose STATT EINEM?
        Wir brauchen Faktor 4 insgesamt (32 → 128).
        Zwei Schritte à Faktor 2 sind stabiler als ein Schritt à Faktor 4:
        - Faktor 4 in einem ConvTranspose braucht kernel=8 — sehr groß,
          schwierig zu optimieren, neigt zu Checkerboard-Artefakten.
        - Zwei Schritte à Faktor 2 (kernel=4, stride=2) sind der
          Standard in modernen Decoder-Architekturen (z.B. U-Net, DCGAN).

    WARUM ConvTranspose2d MIT kernel=4, stride=2, padding=1?
        Diese Kombination verdoppelt die Spatial-Dimension exakt:
            Output_size = (Input_size - 1) * stride - 2*padding + kernel
                        = (32 - 1) * 2 - 2*1 + 4
                        = 62 - 2 + 4 = 64  ✓  (erste Schicht)
                        = (64 - 1) * 2 - 2*1 + 4 = 128  ✓  (zweite Schicht)
        kernel=4, stride=2, padding=1 ist die "magische Kombination"
        für exakte 2× Upsampling ohne Randeffekte.

    WARUM BATCHNORM NACH CONVTRANSPOSE?
        ConvTranspose kann intern sehr unterschiedliche Aktivierungs-
        skalen erzeugen. BatchNorm normalisiert die Channels auf
        Mittelwert≈0 und Std≈1 — das stabilisiert das Training
        und erlaubt höhere Lernraten.

    WARUM ABSCHLIESSENDE Conv2d?
        Die ConvTranspose-Schichten kümmern sich ums Upsampling.
        Die finale Conv2d(3×3) ist ein "Refinement-Step": sie kann
        lokale Strukturen glätten oder schärfen ohne die Auflösung
        zu ändern. Sie gibt dem Modell die Möglichkeit, die
        upgesampelten Features noch einmal zu verfeinern bevor
        der MSE-Loss berechnet wird.

    WARUM KEIN Sigmoid/Tanh AM ENDE?
        BEVFusion-Latents sind rohe Feature-Werte ohne festen
        Wertebereich — sie können beliebig positiv oder negativ sein.
        Eine Sigmoid würde den Output auf [0,1] begrenzen,
        Tanh auf [-1,1] — beides wäre falsch und würde den MSE-Loss
        verzerren. Wir lassen die Aktivierungen offen (linear).

    DATENFLUSS:
        [B, 256,  32,  32]  ← Output Head
            ↓  ConvTranspose(stride=2) + BN + GELU
        [B, 256,  64,  64]
            ↓  ConvTranspose(stride=2) + BN + GELU
        [B, 256, 128, 128]
            ↓  Conv2d(3×3) — kein stride, keine Größenänderung
        [B, 256, 128, 128]  ← Predicted BEV Latent

    Input:  [B, d_model, grid_size, grid_size]      — z.B. [B, 256,  32,  32]
    Output: [B, d_model, grid_size*4, grid_size*4]  — z.B. [B, 256, 128, 128]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        C = config.d_model  # 256 — Channel-Zahl bleibt konstant durch alle Schichten

        # ------------------------------------------------------------------
        # Schicht 1: 32×32 → 64×64
        # ------------------------------------------------------------------
        # ConvTranspose2d: "transponierte Faltung" / "deconvolution"
        #   in_channels=C, out_channels=C: Channel-Zahl bleibt 256
        #   kernel_size=4: 4×4 Filterkernel
        #   stride=2: Output-Pixel decken 2× mehr Fläche als Input-Pixel
        #   padding=1: schneidet 1 Pixel am Rand weg → exakte 2× Vergrößerung
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=C),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Schicht 2: 64×64 → 128×128
        # ------------------------------------------------------------------
        # Identische Konfiguration wie Schicht 1 — nochmal ×2 Upsampling.
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=C),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Schicht 3: 128×128 → 128×128 (Refinement, keine Größenänderung)
        # ------------------------------------------------------------------
        # Conv2d mit kernel=3, padding=1: Output hat dieselbe Größe wie Input.
        # Kein BatchNorm, kein GELU — direkte lineare Ausgabe.
        # Das ist die letzte Schicht vor dem MSE-Loss, lineare Ausgabe
        # gibt dem Loss den vollen Wertebereich.
        self.refine = nn.Conv2d(C, C, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, grid_size, grid_size]  — z.B. [B, 256, 32, 32]

        Returns:
            [B, d_model, grid_size*4, grid_size*4]  — z.B. [B, 256, 128, 128]
        """
        x = self.up1(x)      # [B, 256,  32,  32] → [B, 256,  64,  64]
        x = self.up2(x)      # [B, 256,  64,  64] → [B, 256, 128, 128]
        x = self.refine(x)   # [B, 256, 128, 128] → [B, 256, 128, 128]
        return x
