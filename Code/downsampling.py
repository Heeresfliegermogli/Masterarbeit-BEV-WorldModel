"""
downsampling.py — BEV World Model: Downsampling-Modul
================================================================

Reduziert die räumliche Auflösung der BEV-Latents von 128×128 auf 32×32
bevor sie in das Embedding-Modul fließen.

WARUM DOWNSAMPLING?
    BEVFusion liefert Latents der Form [256, 128, 128].
    In Phase 2 (Cell-Level) wird jede räumliche Zelle ein Token.
    Ohne Downsampling: 3 × 128 × 128 = 49.152 Tokens
    Mit Downsampling:  3 ×  32 ×  32 =  3.072 Tokens

    Die Attention-Matrix skaliert quadratisch mit der Token-Zahl:
        49.152² = 2,4 Milliarden Einträge  → nicht trainierbar
         3.072² =     9,4 Millionen Einträge → handhabbar auf A100

WARUM AvgPool STATT MaxPool ODER CONV?
    - AvgPool: mittelt alle 4×4 Pixel → erhält globale Aktivierungsstärke.
               Gut für kontinuierliche BEV-Feature-Maps.
    - MaxPool: nimmt nur den stärksten Pixel → verliert schwache Aktivierungen.
    - Strided Conv: hätte lernbare Parameter → würde BEVFusion-Latents
                    transformieren statt nur zu komprimieren.
    AvgPool ist der "ehrlichste" Downsampler: keine Gewichte, keine
    Transformation, nur Komprimierung durch Mittelung.

WARUM kernel=4, stride=4?
    128 / 4 = 32 — passt exakt ohne Padding.
    stride = kernel → kein Overlap, jede 4×4 Region fließt genau
    einmal in genau einen Output-Pixel.

SHAPE-ÜBERBLICK:
    Input:  [B, T, C, 128, 128]
    Output: [B, T, C,  32,  32]
    Parameter: 0
"""

import torch
import torch.nn as nn


class DownsamplingModule(nn.Module):
    """
    Räumliches Downsampling von BEV-Latent-Sequenzen via Average Pooling.

    Zustandslos (0 Parameter) — identisch für Phase 1 und Phase 2.

    Args:
        kernel_size: Größe des Pooling-Fensters (Standard: 4)
        stride:      Schrittweite (Standard: 4 — kein Overlap)

    Beispiel:
        >>> ds = DownsamplingModule()
        >>> x = torch.randn(4, 3, 256, 128, 128)
        >>> ds(x).shape
        torch.Size([4, 3, 256, 32, 32])
    """

    def __init__(self, kernel_size: int = 4, stride: int = 4):
        super().__init__()
        # nn.AvgPool2d erwartet [N, C, H, W].
        # kernel=stride=4: 128 → (128-4)/4 + 1 = 32. Exakt, kein Padding nötig.
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
               B=Batch, T=Frames(3), C=Channels(256), H=W=128

        Returns:
            [B, T, C, H//stride, W//stride]  →  [B, 3, 256, 32, 32]

        Strategie — B und T temporär zusammenfalten:
            AvgPool2d versteht keine 5D-Tensoren, nur [N, C, H, W].
            Wir behandeln B*T als eine große Batch-Dimension,
            führen das Pooling durch, und falten danach zurück.
            view() teilt den Speicher mit x — keine Kopie.
        """
        B, T, C, H, W = x.shape

        # Schritt 1: [B, T, C, H, W] → [B*T, C, H, W]
        # Alle T Frames aller B Samples werden zu einer Batch behandelt.
        x = x.view(B * T, C, H, W)

        # Schritt 2: Pooling auf H und W — N und C bleiben unberührt.
        x = self.pool(x)                      # [B*T, C, 32, 32]

        # Schritt 3: [B*T, C, H_out, W_out] → [B, T, C, H_out, W_out]
        # H_out und W_out aus dem Ergebnis lesen (nicht hardcoded),
        # damit das Modul auch mit anderen kernel/stride-Configs funktioniert.
        _, _, H_out, W_out = x.shape
        x = x.view(B, T, C, H_out, W_out)

        return x

    def extra_repr(self) -> str:
        """Wird von print(model) angezeigt."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, params=0"
