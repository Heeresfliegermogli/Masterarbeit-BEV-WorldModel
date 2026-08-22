"""
config.py — BEV World Model: Zentrale Modell-Konfiguration
=====================================================================

Definiert ModelConfig als Python dataclass.

Was ist ein dataclass?
    Ein dataclass ist eine Python-Klasse die hauptsächlich dazu dient,
    Daten zu halten — ähnlich einem struct in C oder einer NamedTuple,
    aber mit Typ-Annotationen, Default-Werten und automatisch
    generiertem __repr__. Man schreibt weniger Boilerplate als bei
    einer normalen Klasse.

Warum eine zentrale Config-Klasse?
    Ohne Config würden alle Hyperparameter (d_model=256, n_heads=8, ...)
    als einzelne Argumente durch alle Module durchgereicht werden.
    Das bedeutet: eine Änderung muss an 5 Stellen gemacht werden.
    Mit ModelConfig gibt es genau einen Ort — alle Module lesen
    ihre Parameter von dort. Das ist besonders wichtig wenn wir
    zwischen Phase 1 (Frame-Level) und Phase 2 (Cell-Level) wechseln,
    weil dann nur `phase` geändert werden muss.

Verwendung:
    from model.config import ModelConfig

    # Standard — Phase 1 (Frame-Level, 3 Tokens)
    cfg = ModelConfig()

    # Phase 2 (Cell-Level, 3072 Tokens)
    cfg = ModelConfig(phase="cell", n_layers=4)

    # Ablation: weniger Layer
    cfg = ModelConfig(phase="cell", n_layers=2)

    # Zugriff auf Parameter
    print(cfg.d_model)    # 256
    print(cfg.n_heads)    # 8
    print(cfg)            # hübscher __repr__ vom dataclass-Decorator
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Zentrale Konfiguration für das BEV World Model.

    EMBEDDING-PARAMETER:
        d_model:     Dimension aller Token-Vektoren im Transformer.
                     256 ist der natürliche Wert weil BEVFusion-Latents
                     256 Channels haben → kein dimensionaler Bruch.

        n_frames:    Anzahl aufeinanderfolgender BEV-Frames als Input.
                     Standard 3 = (t-2, t-1, t) → predict (t+1).
                     In Ablation Study testen wir auch 2 und 4.

        grid_size:   Räumliche Auflösung nach Downsampling.
                     128×128 → AvgPool(stride=4) → 32×32.
                     32 ist der Standard. Wird nur in Phase 2
                     (CellLevelEmbedding) gebraucht um PE_x und PE_y
                     zu dimensionieren.

    TRANSFORMER-PARAMETER:
        n_heads:     Anzahl Attention-Heads im Multi-Head Self-Attention.
                     8 Heads bei d_model=256 → d_head = 256/8 = 32
                     pro Head. Mehr Heads = mehr "Blickwinkel" auf
                     die Sequenz, aber auch mehr Parameter.

        d_ff:        Dimension der inneren Schicht im Feed-Forward Network
                     (FFN) jedes Transformer-Blocks.
                     1024 = 4 × d_model — das ist der Standard-Faktor
                     aus dem ursprünglichen Transformer-Paper (Vaswani 2017).

        n_layers:    Anzahl gestapelter Transformer-Blöcke (Tiefe).
                     Standard 4. In Ablation Study testen wir 2 und 6.
                     Mehr Layer = mehr sequentielle Verarbeitungsschritte,
                     höhere Kapazität, aber auch mehr VRAM.

        dropout:     Dropout-Rate für Regularisierung.
                     0.1 = 10% der Aktivierungen werden zufällig auf 0
                     gesetzt während des Trainings. Verhindert Overfitting.
                     Wird auf Attention-Output und FFN-Output angewendet.

    PHASEN-STEUERUNG:
        phase:       "frame" = Phase 1 (3 Tokens, schnell, für Prototyping)
                     "cell"  = Phase 2 (3072 Tokens, vollständige spatiale
                                        Attention — der wissenschaftliche Kern)

                     Dieser eine String-Wert steuert welches Embedding-Modul
                     und welcher Output-Head im BEVWorldModel instanziiert wird.

    ABGELEITETE WERTE (Properties):
        n_tokens:    Gesamtzahl der Tokens nach Embedding.
                     Phase 1: 3 (ein Token pro Frame)
                     Phase 2: n_frames × grid_size² = 3 × 32² = 3072
                     Wichtig für VRAM-Abschätzung (Attention ist O(n²)).

        d_head:      Dimension pro Attention-Head = d_model / n_heads.
                     Muss ganzzahlig sein — wird in __post_init__ geprüft.
    """

    # --- Embedding ---
    d_model:    int   = 256     # Token-Dimension (= BEVFusion Channel-Zahl)
    n_frames:   int   = 3       # Anzahl Input-Frames
    grid_size:  int   = 32      # Räumliche Auflösung nach Downsampling (32×32)

    # --- Transformer ---
    n_heads:    int   = 8       # Attention-Heads
    d_ff:       int   = 1024    # FFN innere Dimension (= 4 × d_model)
    n_layers:   int   = 4       # Anzahl Transformer-Blöcke
    dropout:    float = 0.1     # Dropout-Rate

    # --- Phasen-Steuerung ---
    phase:      str   = "frame" # "frame" (Phase 1) oder "cell" (Phase 2)

    # --- Ego-Motion-Conditioning (FiLM) ---
    ego_cond_mode: str = "off"  # off | state | action | both (Default off = kompatibel)
    output_mode: str = "gated"  # gated (Default, wie bisher) | residual
                                # (pred = letzter Frame + Head-Output, kein Gate)
    vae_mode:   str = "off"     # 18/B1: off (Default, keine Module) | cvae
                                # (Prior/Posterior/z-FiLM, zero-init = off-Start)
    vae_z_dim:  int = 32        # 18/B1: Dimension der latenten Variable z
    flow_head:  str = "off"     # 18/B2: off (Default, kein Modul) | flow
                                # (FM-Residual-Kopf auf frozen Backbone; forward
                                # bleibt unveraendert, Sampling via flow.sample)
    flow_dim:   int = 224       # 18/B2: Kanalbreite v_theta (~6.5M = Backbone-Paritaet)
    flow_layers: int = 8        # 18/B2: Anzahl Residual-Conv-Bloecke

    # ------------------------------------------------------------------
    # Validierung nach __init__
    # ------------------------------------------------------------------

    def __post_init__(self):
        """
        Wird automatisch nach dem generierten __init__ aufgerufen.

        Hier validieren wir Constraints die der Typ-Annotator nicht
        ausdrücken kann — z.B. dass d_model durch n_heads teilbar ist.
        Frühe Fehlermeldungen sind besser als mysteriöse CUDA-Errors
        im Training.
        """
        # d_model muss durch n_heads teilbar sein (sonst ist d_head nicht int)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) muss durch n_heads ({self.n_heads}) "
                f"teilbar sein. d_head = {self.d_model}/{self.n_heads} "
                f"= {self.d_model / self.n_heads} ist nicht ganzzahlig."
            )

        # Phase muss "frame" oder "cell" sein
        if self.phase not in ("frame", "cell"):
            raise ValueError(
                f"phase muss 'frame' oder 'cell' sein, nicht '{self.phase}'."
            )

        # n_frames mindestens 2 (1 wäre kein "temporal reasoning")
        if self.n_frames < 2:
            raise ValueError(
                f"n_frames muss mindestens 2 sein, ist {self.n_frames}."
            )

        # Dropout zwischen 0 und 1
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout muss in [0, 1) liegen, ist {self.dropout}."
            )

    # ------------------------------------------------------------------
    # Abgeleitete Properties (read-only, nicht im dataclass-State)
    # ------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        """
        Dimension pro Attention-Head.

        Im Multi-Head Attention wird d_model auf n_heads aufgeteilt:
        jeder Head arbeitet in einem d_head-dimensionalen Unterraum.
        Durch __post_init__ garantiert dass dies ganzzahlig ist.
        """
        return self.d_model // self.n_heads

    @property
    def n_tokens(self) -> int:
        """
        Gesamtzahl der Tokens nach Embedding — phase-abhängig.

        Phase 1 (frame): 3 Tokens — ein pro Frame.
            → Attention-Matrix: 3×3 (trivial, kaum VRAM)

        Phase 2 (cell): n_frames × grid_size² Tokens.
            → Bei Standard: 3 × 32 × 32 = 3072 Tokens
            → Attention-Matrix: 3072×3072 (~9M Werte pro Batch-Element!)
            → Daher Batch-Size 4 statt 16 in Phase 2.

        Diese Property wird in Unit Tests und VRAM-Schätzungen genutzt.
        """
        if self.phase == "frame":
            return self.n_frames
        else:  # "cell"
            return self.n_frames * self.grid_size * self.grid_size

    @property
    def n_spatial_tokens(self) -> int:
        """
        Anzahl räumlicher Tokens pro Frame (nur relevant für Phase 2).
        grid_size² = 32² = 1024 — wird im CellLevelOutputHead gebraucht
        um die Tokens des letzten Frames herauszuschneiden.
        """
        return self.grid_size * self.grid_size


# ---------------------------------------------------------------------------
# Kurzer Selbst-Test wenn Datei direkt ausgeführt wird
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ModelConfig Selbst-Test ===\n")

    # Standard-Config (Phase 1)
    cfg1 = ModelConfig()
    print(f"Phase 1 Config:\n{cfg1}")
    print(f"  d_head   = {cfg1.d_head}")
    print(f"  n_tokens = {cfg1.n_tokens}")
    print()

    # Phase 2 Config
    cfg2 = ModelConfig(phase="cell")
    print(f"Phase 2 Config:\n{cfg2}")
    print(f"  d_head   = {cfg2.d_head}")
    print(f"  n_tokens = {cfg2.n_tokens}  (= {cfg2.n_frames} × {cfg2.grid_size}² )")
    print()

    # Ablation: weniger Layer, mehr Frames
    cfg3 = ModelConfig(phase="cell", n_layers=2, n_frames=4)
    print(f"Ablation Config (n_layers=2, n_frames=4):\n{cfg3}")
    print(f"  n_tokens = {cfg3.n_tokens}  (= {cfg3.n_frames} × {cfg3.grid_size}² )")
    print()

    # Fehler-Test: d_model nicht durch n_heads teilbar
    print("Fehler-Test (d_model=100, n_heads=8 → sollte ValueError):")
    try:
        bad = ModelConfig(d_model=100, n_heads=8)
    except ValueError as e:
        print(f"  ✓ ValueError korrekt abgefangen: {e}")
    print()

    # Fehler-Test: ungültige Phase
    print("Fehler-Test (phase='invalid' → sollte ValueError):")
    try:
        bad = ModelConfig(phase="invalid")
    except ValueError as e:
        print(f"  ✓ ValueError korrekt abgefangen: {e}")
