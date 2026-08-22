"""
embedding.py — BEV World Model: Token Embedding Module
========================================================================

Wandelt die downgesampelten BEV-Latents [B, T, C, 32, 32] in eine
Sequenz von d_model-dimensionalen Tokens um, die der Transformer
verarbeiten kann.

ZWEI VARIANTEN — NUR EINE ZEILE UNTERSCHIED IM HAUPTMODELL:

  Phase 1 (Frame-Level):   3 Tokens    — ein Token pro Frame
  Phase 2 (Cell-Level):  3072 Tokens  — ein Token pro räumlicher Zelle

  # In BEVWorldModel:
  self.embedding = FrameLevelEmbedding(config)  # Phase 1
  self.embedding = CellLevelEmbedding(config)   # Phase 2 — nur diese Zeile tauschen

WARUM BRAUCHEN WIR ÜBERHAUPT EIN EMBEDDING?
    Ein Transformer verarbeitet Sequenzen von Vektoren — konkret:
    [Sequenzlänge, d_model]. Unsere BEV-Latents sind aber 3D-Feature-Maps
    [C, H, W]. Das Embedding löst diesen Mismatch: es projiziert die
    räumlichen Feature-Maps in eine flache Sequenz von d_model-Vektoren.

    Zusätzlich fügt das Positional Encoding dem Transformer Informationen
    darüber hinzu WO (räumlich) und WANN (zeitlich) ein Token herkommt —
    ohne PE wäre die Reihenfolge der Tokens bedeutungslos.

WARUM POSITIONAL ENCODING?
    Transformer-Attention ist permutationsinvariant: vertauscht man alle
    Tokens, ändert sich der Output nicht. Für unser World Model ist aber
    entscheidend dass t-2 vor t-1 vor t kommt (zeitlich), und in Phase 2
    auch wo im BEV-Grid jede Zelle liegt (räumlich). PE macht diese
    Struktur explizit durch addierte gelernte Vektoren.
"""

import torch
import torch.nn as nn

from config import ModelConfig


# ---------------------------------------------------------------------------
# Phase 1: Frame-Level Embedding (3 Tokens)
# ---------------------------------------------------------------------------

class FrameLevelEmbedding(nn.Module):
    """
    Komprimiert jeden BEV-Frame zu einem einzigen d_model-Vektor (Token).

    Strategie:
        1. GlobalAvgPool: [C, 32, 32] → [C]
           Fasst die räumliche Information jedes Channels zu einem
           Skalar zusammen: "wie aktiv ist Channel k in diesem Frame?"
        2. Linear(C, d_model): projiziert in den Transformer-Raum
        3. + PE_t: addiert einen gelernten Vektor pro Frame-Position
           damit der Transformer weiß ob er Frame t-2, t-1 oder t sieht

    Resultat: [B, T, d_model] — T Tokens, einer pro Frame.

    WARUM GlobalAvgPool STATT Flatten?
        Flatten würde [C, 32, 32] → [C×32×32] = [262.144]-dim Vektor.
        Linear(262144, 256) hätte 67 Mio Parameter — nur für das Embedding.
        GlobalAvgPool + Linear(256, 256) hat nur ~66k Parameter.
        Dafür verlieren wir die räumliche Information innerhalb eines Frames —
        das ist der Trade-off von Phase 1. Phase 2 (CellLevelEmbedding)
        behält die räumliche Struktur durch Cell-Tokens.

    Args:
        config: ModelConfig mit d_model und n_frames

    Input:  [B, T, C, H, W]  (nach DownsamplingModule: H=W=32, C=256)
    Output: [B, T, d_model]  — T=3 Tokens
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # GlobalAvgPool: mittelt über H und W → [B*T, C]
        # AdaptiveAvgPool2d(1) bedeutet: Output-Size = 1×1, egal wie groß
        # der Input ist. Damit funktioniert das Modul auch wenn H/W
        # sich durch andere Downsampling-Configs ändert.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Lineare Projektion: C → d_model
        # Auch wenn C == d_model (beide 256), ist diese Projektion wichtig:
        # sie gibt dem Modell die Möglichkeit, eine andere Basis zu lernen
        # als BEVFusion sie intern nutzt. Ohne sie wäre es nur eine
        # Identitätsfunktion — mit ihr ist es eine gelernte Umprojektion.
        self.proj = nn.Linear(config.d_model, config.d_model)

        # Temporal Positional Encoding: eine gelernte Lookup-Tabelle
        # mit n_frames Einträgen, je d_model-dimensional.
        # PE_t[0] = "ich bin Frame t-2"
        # PE_t[1] = "ich bin Frame t-1"
        # PE_t[2] = "ich bin Frame t"
        # Diese Vektoren werden während des Trainings gelernt.
        self.pe_t = nn.Embedding(config.n_frames, config.d_model)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  — downgesampelte BEV-Latents

        Returns:
            tokens: [B, T, d_model]  — T Frame-Tokens
        """
        B, T, C, H, W = x.shape

        # --- Schritt 1: B und T zusammenfalten ---
        # AdaptiveAvgPool2d erwartet [N, C, H, W].
        x = x.view(B * T, C, H, W)                      # [B*T, C, 32, 32]

        # --- Schritt 2: GlobalAvgPool über H und W ---
        # squeeze(-1).squeeze(-1): entfernt die beiden Size-1 Dimensionen
        # die AdaptiveAvgPool2d(1) anhängt → [B*T, C, 1, 1] → [B*T, C]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B*T, C]

        # --- Schritt 3: Lineare Projektion ---
        x = self.proj(x)                                  # [B*T, d_model]

        # --- Schritt 4: Zurückfalten zu [B, T, d_model] ---
        x = x.view(B, T, self.config.d_model)             # [B, T, 256]

        # --- Schritt 5: Temporal Positional Encoding addieren ---
        # torch.arange(T) = [0, 1, 2] für T=3
        # pe_t(t_idx) = Lookup → [T, d_model] = [3, 256]
        # Broadcasting: [B, T, 256] + [T, 256] → [B, T, 256]
        # Jeder Batch-Eintrag bekommt dieselben PE-Vektoren addiert.
        t_idx = torch.arange(T, device=x.device)         # [T]
        x = x + self.pe_t(t_idx)                          # [B, T, 256]

        return x  # [B, 3, 256] — 3 Tokens, bereit für den Transformer


# ---------------------------------------------------------------------------
# Phase 2: Cell-Level Embedding (3072 Tokens)
# ---------------------------------------------------------------------------

class CellLevelEmbedding(nn.Module):
    """
    Behandelt jede räumliche Zelle jedes Frames als eigenen Token.

    Strategie:
        1. Flatten: [C, 32, 32] → [32×32, C] = [1024, C] pro Frame
           Jede der 1024 Zellen wird ein Kandidat-Token.
        2. Linear(C, d_model): projiziert in den Transformer-Raum
        3. + PE_xy: addiert räumliches PE (concat aus PE_x und PE_y)
           damit der Transformer weiß wo im BEV-Grid eine Zelle liegt
        4. + PE_t: addiert zeitliches PE

    Resultat: [B, T×H×W, d_model] = [B, 3072, 256] — 3072 Tokens.

    WARUM SEPARATE PE_x UND PE_y STATT 2D-EMBEDDING?
        Eine 2D-Embedding-Tabelle hätte 32×32 = 1024 Einträge à 256 dim
        = 262k Parameter. Durch Faktorizierung in PE_x (32 × 128) und
        PE_y (32 × 128) mit Concatenation haben wir nur 2 × 32 × 128
        = 8192 Parameter — Faktor 32 sparsamer, trotzdem vollständig.
        Außerdem kann das Modell x- und y-Struktur getrennt lernen.

    Args:
        config: ModelConfig mit d_model, n_frames, grid_size

    Input:  [B, T, C, H, W]  (nach DownsamplingModule: H=W=32, C=256)
    Output: [B, T×H×W, d_model]  — 3072 Tokens
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        G = config.grid_size  # 32

        # Projektion: C → d_model (identisch zu Phase 1)
        self.proj = nn.Linear(config.d_model, config.d_model)

        # Räumliches PE — faktorisiert in x und y:
        # PE_x: Embedding für Spaltenposition (0..31) → d_model//2 = 128 dim
        # PE_y: Embedding für Zeilenposition   (0..31) → d_model//2 = 128 dim
        # Concatenation → 256 dim = d_model
        # d_model muss gerade sein (durch __post_init__ in ModelConfig geprüft)
        self.pe_x = nn.Embedding(G, config.d_model // 2)
        self.pe_y = nn.Embedding(G, config.d_model // 2)

        # Zeitliches PE — identisch zu Phase 1
        self.pe_t = nn.Embedding(config.n_frames, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  — downgesampelte BEV-Latents

        Returns:
            tokens: [B, T*H*W, d_model]  — 3072 Tokens
        """
        B, T, C, H, W = x.shape
        G = self.config.grid_size

        # --- Schritt 1: Spatial Flatten ---
        # Ziel: jede (frame, zeile, spalte) Kombination wird ein Token.
        # permute: [B, T, C, H, W] → [B, T, H, W, C]
        #   (Channel an letzte Stelle für einfaches Reshape)
        # reshape: → [B, T*H*W, C]
        #   alle Zellen aller Frames in eine lange Sequenz
        x = x.permute(0, 1, 3, 4, 2)               # [B, T, H, W, C]
        x = x.reshape(B, T * H * W, C)              # [B, 3072, 256]

        # --- Schritt 2: Lineare Projektion ---
        x = self.proj(x)                             # [B, 3072, d_model]

        # --- Schritt 3: Räumliches PE aufbauen ---
        # col: [0, 1, ..., 31] — Spaltenindizes
        # row: [0, 1, ..., 31] — Zeilenindizes
        col = torch.arange(W, device=x.device)       # [32]
        row = torch.arange(H, device=x.device)       # [32]

        # PE_x lookup: [32, 128]
        # PE_y lookup: [32, 128]
        pe_x = self.pe_x(col)                        # [W, d_model//2]
        pe_y = self.pe_y(row)                        # [H, d_model//2]

        # Gitter aufspannen: jede Zelle bekommt ihren (x,y) PE-Vektor.
        # pe_x.unsqueeze(0).expand(H, -1, -1): [H, W, d_model//2]
        #   → dieselbe Spalten-PE für alle Zeilen
        # pe_y.unsqueeze(1).expand(-1, W, -1): [H, W, d_model//2]
        #   → dieselbe Zeilen-PE für alle Spalten
        # torch.cat(..., dim=-1): [H, W, d_model]
        # reshape: [H*W, d_model] = [1024, 256]
        pe_xy = torch.cat([
            pe_x.unsqueeze(0).expand(H, -1, -1),    # [H, W, d_model//2]
            pe_y.unsqueeze(1).expand(-1, W, -1),    # [H, W, d_model//2]
        ], dim=-1).reshape(H * W, self.config.d_model)  # [1024, 256]

        # PE_xy für alle T Frames wiederholen:
        # [1024, 256] → repeat(T, 1) → [3072, 256]
        pe_xy = pe_xy.unsqueeze(0).expand(T, -1, -1).reshape(
            T * H * W, self.config.d_model
        )                                            # [3072, 256]

        # --- Schritt 4: Zeitliches PE ---
        # Jede Zelle gehört zu genau einem Frame.
        # t_idx: für die ersten H*W Tokens t=0, nächste H*W t=1, etc.
        # repeat_interleave: [0,0,...0, 1,1,...1, 2,2,...2] (je H*W mal)
        t_idx = torch.arange(T, device=x.device).repeat_interleave(H * W)
        pe_t  = self.pe_t(t_idx)                    # [3072, 256]

        # --- Schritt 5: PE addieren ---
        # Broadcasting: [B, 3072, 256] + [3072, 256] → [B, 3072, 256]
        x = x + pe_xy + pe_t                        # [B, 3072, 256]

        return x  # [B, 3072, 256] — bereit für den Transformer

    def extra_repr(self) -> str:
        G = self.config.grid_size
        T = self.config.n_frames
        return (f"grid={G}×{G}, n_frames={T}, "
                f"n_tokens={T*G*G}, d_model={self.config.d_model}")
