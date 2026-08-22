"""
output_head.py — BEV World Model: Output Heads
===================================================================

Wandelt den Transformer-Output zurück in ein räumliches Feature-Grid.

KONTEXT IN DER PIPELINE:
    ...
    Transformer      [B,    3, 256]  (Phase 1)
                  or [B, 3072, 256]  (Phase 2)
         ↓
    Output Head      [B, 256, 32, 32]   ← DIESE DATEI
         ↓
    Upsampling Head  [B, 256, 128, 128] ← upsampling_head.py

ZWEI KLASSEN — EINE PRO PHASE:
    FrameLevelOutputHead   Phase 1 — 3 Frame-Tokens rein, letzter Token raus
    CellLevelOutputHead    Phase 2 — 3072 Cell-Tokens rein, Frame-t-Grid raus

    Beide liefern denselben Output: [B, 256, 32, 32]
    → der UpsamplingHead danach ist für beide Phasen identisch.

IMPORTS FÜR ANDERE MODULE:
    from output_head import FrameLevelOutputHead, CellLevelOutputHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ===========================================================================
# FrameLevelOutputHead — Phase 1
# ===========================================================================

class FrameLevelOutputHead(nn.Module):
    """
    Konvertiert [B, 3, 256] → [B, 256, 32, 32].

    STRATEGIE:
        Phase 1 hat nur 3 Frame-Tokens — einen pro Frame (t-2, t-1, t).
        Wir nehmen nur den letzten Token (Frame t), weil er durch
        Self-Attention bereits die Information aller 3 Frames absorbiert hat.
        Dann projizieren wir ihn mit Linear+GELU und blasen ihn
        mit bilinearem Upsample von [1,1] auf [32,32] auf.

    WARUM NUR DEN LETZTEN TOKEN?
        Der Transformer ist kein Causal-Modell (kein Masking) —
        jeder Token hat alle anderen gesehen. Token t hat also bereits
        t-2 und t-1 "verarbeitet". Er ist die kompakteste Zusammenfassung
        des aktuellen Zustands → ideale Basis für die t+1 Prediction.

    WARUM BILINEAR UPSAMPLE STATT ConvTranspose?
        Wir gehen von [1,1] auf [32,32] — Faktor 32 in einem Schritt.
        Bilinear ist parameterfrei und stabil. Die eigentliche lernbare
        Rekonstruktion der räumlichen Struktur macht der UpsamplingHead
        danach mit ConvTranspose-Schichten.

    DATENFLUSS:
        [B,  3, 256]  Transformer Output
            ↓  [:, -1, :]       letzter Token = Frame t
        [B, 256]
            ↓  Linear(256→256) + GELU
        [B, 256]
            ↓  unsqueeze(-1).unsqueeze(-1)
        [B, 256, 1, 1]
            ↓  Bilinear Upsample → (32, 32)
        [B, 256, 32, 32]

    Input:  [B, n_frames, d_model]
    Output: [B, d_model, grid_size, grid_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.proj   = nn.Linear(config.d_model, config.d_model)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_frames, d_model]  — z.B. [B, 3, 256]

        Returns:
            [B, d_model, grid_size, grid_size]  — z.B. [B, 256, 32, 32]
        """
        G = self.config.grid_size   # 32

        # Schritt 1: Letzten Frame-Token nehmen
        # x[:, -1, :] → nimmt Index n_frames-1 = Frame t
        x = x[:, -1, :]                             # [B, d_model]

        # Schritt 2: Linear-Projektion + Aktivierung
        x = self.act(self.proj(x))                  # [B, d_model]

        # Schritt 3: Zwei Spatial-Dimensionen anhängen (H=1, W=1)
        # F.interpolate braucht 4D: [B, C, H, W]
        x = x.unsqueeze(-1).unsqueeze(-1)           # [B, d_model, 1, 1]

        # Schritt 4: Bilinear Upsample auf grid_size × grid_size
        x = F.interpolate(
            x,
            size=(G, G),
            mode="bilinear",
            align_corners=False,
        )                                            # [B, d_model, 32, 32]

        return x


# ===========================================================================
# CellLevelOutputHead — Phase 2
# ===========================================================================

class CellLevelOutputHead(nn.Module):
    """
    Konvertiert [B, 3072, 256] → [B, 256, 32, 32].

    STRATEGIE:
        Phase 2 hat 3072 Tokens = 3 Frames × 32×32 Zellen.
        Die Token-Reihenfolge aus dem Embedding ist:
            Tokens    0..1023  → Frame t-2
            Tokens 1024..2047  → Frame t-1
            Tokens 2048..3071  → Frame t   ← die wollen wir

        Wir nehmen die letzten 1024 Tokens (= Frame t), projizieren
        jeden einzeln mit Linear+GELU, und falten dann direkt zurück
        in ein [B, 256, 32, 32] Grid — kein Upsample nötig, weil die
        räumliche Struktur bereits im Embedding kodiert war.

    WARUM KEIN BILINEAR UPSAMPLE WIE IN PHASE 1?
        In Phase 1 hatten wir nur 1 Vektor ohne räumliche Info —
        wir mussten die Struktur "erfinden".
        In Phase 2 hat jeder der 1024 Tokens bereits eine genaue
        räumliche Position (durch PE_xy im Embedding gelernt).
        Das Reshape von [B,1024,256] → [B,256,32,32] ist verlustfrei:
        wir ordnen nur um, erfinden nichts.
        Das ist der wissenschaftliche Kern von Phase 2.

    WARUM [:, -n_spatial:, :] STATT [:, 2048:3072, :]?
        Robustheit für Ablation Studies: bei n_frames=2 gibt es nur
        2048 Tokens, bei n_frames=4 sind es 4096.
        "Letzte 1024 Tokens" = immer Frame t, egal wie viele Frames.

    DATENFLUSS:
        [B, 3072, 256]  Transformer Output
            ↓  [:, -1024:, :]   letzte 1024 Tokens = Frame t
        [B, 1024, 256]
            ↓  Linear(256→256) + GELU
        [B, 1024, 256]
            ↓  permute(0, 2, 1)
        [B, 256, 1024]
            ↓  reshape(B, 256, 32, 32)
        [B, 256, 32, 32]

    Input:  [B, n_frames * grid_size², d_model]  — z.B. [B, 3072, 256]
    Output: [B, d_model, grid_size, grid_size]   — z.B. [B,  256,  32,  32]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.n_spatial = config.n_spatial_tokens   # 32×32 = 1024
        self.proj      = nn.Linear(config.d_model, config.d_model)
        self.act       = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_frames * grid_size², d_model]  — z.B. [B, 3072, 256]

        Returns:
            [B, d_model, grid_size, grid_size]  — z.B. [B, 256, 32, 32]
        """
        B = x.shape[0]
        G = self.config.grid_size    # 32
        n = self.n_spatial           # 1024

        # Schritt 1: Letzte n_spatial Tokens = Frame t herausschneiden
        # Funktioniert für beliebiges n_frames (Ablation-sicher)
        x = x[:, -n:, :]            # [B, 1024, d_model]

        # Schritt 2: Linear-Projektion + Aktivierung
        # nn.Linear arbeitet auf der letzten Dim → jeder Token unabhängig
        x = self.act(self.proj(x))  # [B, 1024, d_model]

        # Schritt 3: Channel-Dim nach vorne bringen, dann räumlich falten
        # permute: [B, 1024, 256] → [B, 256, 1024]
        # reshape: [B, 256, 1024] → [B, 256, 32, 32]
        # Die Reihenfolge stimmt mit CellLevelEmbedding überein:
        # dort wurde zeilenweise geflattened → hier zeilenweise zurückgefaltet
        x = x.permute(0, 2, 1)                      # [B, d_model, n_spatial]
        x = x.reshape(B, self.config.d_model, G, G) # [B, d_model, 32, 32]

        return x
