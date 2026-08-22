"""
transformer.py — BEV World Model: Transformer-Kern
============================================================

Implementiert den Transformer-Stack der das Herzstück des World Models ist.

ARCHITEKTUR-ÜBERBLICK:
    Input:  [B, seq_len, d_model]
    Output: [B, seq_len, d_model]

    seq_len ist PHASE-AGNOSTISCH:
        Phase 1 (Frame-Level):  seq_len =    3  (ein Token pro Frame)
        Phase 2 (Cell-Level):   seq_len = 3072  (ein Token pro BEV-Zelle)

    Der Transformer merkt den Unterschied nicht — er verarbeitet einfach
    eine Sequenz von Vektoren, egal wie lang sie ist.

WARUM PRE-LAYERNORM?
    Es gibt zwei Varianten:
        Post-LN (originales Transformer-Paper):
            x = LayerNorm(x + Attention(x))   ← Instabiler in der Praxis
        Pre-LN (moderner Standard, z.B. GPT-2):
            x = x + Attention(LayerNorm(x))   ← Stabiler, konvergiert besser

    Pre-LN normalisiert BEVOR man rechnet. Das bedeutet die Attention
    arbeitet immer mit normalisierten Werten — keine explodierten Gradienten,
    keine abgestorbenen Neuronen. Für unsere Trainingssetup (6 Wochen,
    limitiertes VRAM) ist Stabilität wichtiger als letztes Prozent Performance.

WARUM RESIDUAL-VERBINDUNGEN (x + ...)?
    Das Netz lernt nur was es ÄNDERN muss, nicht alles von Grund auf.
    Mathematisch: Gradienten fließen direkt durch die Addition zurück
    ohne durch Attention oder FFN gehen zu müssen → kein Vanishing Gradient.
    Praktisch: selbst wenn Attention "falsch" liegt, ist das Ergebnis
    mindestens so gut wie die direkte Weiterleitung von x.

WARUM FFN NACH DER ATTENTION?
    Attention mischt Information zwischen Tokens ("was sagen die anderen?").
    FFN transformiert jeden Token einzeln nichtlinear ("was bedeutet das für mich?").
    Beide Schritte sind notwendig — Attention allein kann keine komplexen
    Funktionen pro Token lernen, FFN allein kann keine Interaktion zwischen
    Tokens modellieren.

DATEISTRUKTUR:
    model/
    ├── config.py — ModelConfig dataclass
    ├── downsampling.py — DownsamplingModule
    ├── embedding.py — FrameLevelEmbedding, CellLevelEmbedding
    └── transformer.py — TransformerBlock, TransformerEncoder ← diese Datei
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# TransformerBlock — ein einzelner Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Ein einzelner Transformer-Block in Pre-LayerNorm Variante.

    VOLLSTÄNDIGE FORWARD-GLEICHUNG:
        # Attention-Zweig:
        x = x + dropout(Attention(LayerNorm(x)))
        # FFN-Zweig:
        x = x + dropout(FFN(LayerNorm(x)))

    AUFBAU IM DETAIL:

    1. SELF-ATTENTION (nn.MultiheadAttention):
       - "Self" bedeutet: Query, Key und Value kommen alle aus x selbst.
         Jeder Token fragt sich: "was ist an anderen Tokens relevant für mich?"
       - Query Q = x @ W_Q   [B, seq_len, d_model]
         Key   K = x @ W_K   [B, seq_len, d_model]
         Value V = x @ W_V   [B, seq_len, d_model]
       - Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_head)) @ V
         Das @ K.T gibt die "Ähnlichkeit" zwischen jedem Token-Paar.
         Division durch sqrt(d_head) verhindert dass die Werte zu groß werden.
       - KEIN MASKING: Alle Tokens dürfen alle anderen sehen.
         Das ist anders als in Sprach-Modellen (dort darf t nicht t+1 sehen).
         Für uns ist das ok — wir predicten den ZUKÜNFTIGEN Frame t+1,
         aber alle Input-Frames (t-2, t-1, t) sind gleichberechtigt.
       - n_heads=8, d_head=32: Statt einer großen Attention gibt es 8 parallele
         "Köpfe", jeder arbeitet in d_head=32 Dimensionen. Verschiedene Köpfe
         können verschiedene Beziehungstypen lernen (z.B. räumlich nahe Zellen
         vs. zeitlich ähnliche Frames).

    2. FEED-FORWARD NETWORK (FFN):
       - Linear(256 → 1024): Expansion — mehr Kapazität für nichtlineare Transformation
       - GELU Aktivierung: "Gaussian Error Linear Unit", weicher als ReLU,
         moderner Standard seit GPT-2/BERT. Kein harter Schnitt bei 0.
       - Dropout(0.1): Regularisierung — 10% der Neuronen werden zufällig
         abgeschaltet während des Trainings. Verhindert Overfitting.
       - Linear(1024 → 256): Zurück auf d_model-Dimension

    3. LAYERNORM (Pre-LN):
       - Normalisiert die Aktivierungen auf Mittelwert≈0, Std≈1
       - elementwise_affine=True (Standard): lernbare γ und β Parameter
         die das Netz benutzt um die Normalisiung zu "entrollen" wenn nötig
       - Läuft über die letzte Dimension (d_model=256), nicht über den Batch

    Args:
        config: ModelConfig mit d_model, n_heads, d_ff, dropout

    Input:  [B, seq_len, d_model]  (seq_len = 3 oder 3072)
    Output: [B, seq_len, d_model]  (gleiche Shape — Transformer ist shape-preserving)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # --- Pre-LayerNorm vor Attention ---
        # normalized_shape=d_model: normalisiert nur über die Feature-Dimension,
        # nicht über Batch oder Sequenz. Jeder Token wird unabhängig normalisiert.
        self.norm1 = nn.LayerNorm(config.d_model)

        # --- Multi-Head Self-Attention ---
        # embed_dim=d_model: Eingabe- und Ausgabedimension
        # num_heads=n_heads: Anzahl paralleler Attention-Köpfe
        # dropout=dropout: Dropout auf die Attention-Gewichte (nicht Output!)
        # batch_first=True: erwartet [B, seq_len, d_model] statt [seq_len, B, d_model]
        #                   Das ist wichtig — PyTorch-Standard war früher batch_last!
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --- Dropout auf Attention-Output ---
        # Separater Dropout nach dem Attention-Output (zusätzlich zum Attention-internen).
        # Das folgt der Architektur aus dem originalen "Attention is All You Need" Paper.
        self.dropout1 = nn.Dropout(config.dropout)

        # --- Pre-LayerNorm vor FFN ---
        self.norm2 = nn.LayerNorm(config.d_model)

        # --- Feed-Forward Network ---
        # Sequential kombiniert mehrere Layer zu einem Modul.
        # Die Reihenfolge: Linear → GELU → Dropout → Linear
        # Warum GELU statt ReLU?
        #   ReLU:  max(0, x)  — harter Schnitt, Gradient=0 für x<0 (tote Neuronen)
        #   GELU:  x * Φ(x)  — weicher Übergang, kein hartes 0-Clipping
        #   In der Praxis: GELU konvergiert zuverlässiger bei Transformer-Architekturen
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),    # 256 → 1024 (Expansion)
            nn.GELU(),                                  # Aktivierung
            nn.Dropout(config.dropout),                 # Regularisierung
            nn.Linear(config.d_ff, config.d_model),    # 1024 → 256 (Rücktransformation)
        )

        # --- Dropout auf FFN-Output ---
        self.dropout2 = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-LayerNorm Transformer Block Forward Pass.

        Args:
            x: [B, seq_len, d_model]  — Token-Sequenz

        Returns:
            x: [B, seq_len, d_model]  — transformierte Token-Sequenz (gleiche Shape!)

        SCHRITT-FÜR-SCHRITT:

        Schritt 1 — Attention-Zweig:
            normed = LayerNorm(x)           # Normalisieren
            attn_out, _ = Attention(normed, normed, normed)  # Self-Attention
            x = x + Dropout(attn_out)       # Residual + Dropout

        Schritt 2 — FFN-Zweig:
            normed = LayerNorm(x)           # Normalisieren (jetzt mit Attention-Info)
            ffn_out = FFN(normed)           # Feed-Forward
            x = x + Dropout(ffn_out)        # Residual + Dropout

        Das gleiche x durchläuft beide Zweige und wird jeweils mit dem
        Ergebnis des Zweiges addiert — das sind die Residual-Verbindungen.
        """

        # ---------------------------------------------------------------
        # ATTENTION-ZWEIG
        # ---------------------------------------------------------------

        # Schritt 1a: LayerNorm vor Attention (Pre-LN)
        normed = self.norm1(x)              # [B, seq_len, d_model]

        # Schritt 1b: Multi-Head Self-Attention
        # query=normed, key=normed, value=normed — "Self" weil alle drei gleich sind
        # need_weights=False: wir brauchen die Attention-Gewichte NICHT für den
        #   Training-Forward-Pass — das spart Speicher und ist etwas schneller.
        #   Für Visualisierung in den Unit-Tests holen wir sie separat.
        attn_out, _ = self.attn(
            query=normed,
            key=normed,
            value=normed,
            need_weights=False,
        )                                   # [B, seq_len, d_model]

        # Schritt 1c: Dropout + Residual-Addition
        x = x + self.dropout1(attn_out)    # [B, seq_len, d_model]

        # ---------------------------------------------------------------
        # FFN-ZWEIG
        # ---------------------------------------------------------------

        # Schritt 2a: LayerNorm vor FFN (Pre-LN)
        normed = self.norm2(x)              # [B, seq_len, d_model]

        # Schritt 2b: Feed-Forward Network
        ffn_out = self.ffn(normed)          # [B, seq_len, d_model]

        # Schritt 2c: Dropout + Residual-Addition
        x = x + self.dropout2(ffn_out)     # [B, seq_len, d_model]

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hilfsfunktion für Visualisierung und Unit-Tests.
        Gibt die Attention-Gewichte zurück ohne den normalen Forward-Pass
        zu beeinflussen.

        Wird NUR für Debugging/Visualisierung aufgerufen — nicht im Training.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            weights: [B, n_heads, seq_len, seq_len]
                     weights[b, h, i, j] = wie viel Token i auf Token j achtet
                     (Head h, Batch-Element b)
        """
        self.eval()  # Dropout deaktivieren für konsistente Attention-Gewichte
        with torch.no_grad():
            normed = self.norm1(x)
            _, weights = self.attn(
                query=normed,
                key=normed,
                value=normed,
                need_weights=True,
                average_attn_weights=False,  # Gewichte pro Head, nicht gemittelt
            )
        return weights  # [B, n_heads, seq_len, seq_len]


# ---------------------------------------------------------------------------
# TransformerEncoder — Stack aus N Blöcken
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack aus N TransformerBlock-Instanzen.

    Das ist die eigentliche "Tiefe" des Modells. N=4 Blöcke bedeutet:
    die Sequenz wird 4 Mal durch den Attention+FFN-Zyklus geführt.

    WARUM MEHRERE BLÖCKE?
        Jeder Block kann nur "eine Schicht" an Beziehungen modellieren.
        Block 1 lernt vielleicht: "Zelle (x,y) bei t-1 ähnelt (x,y) bei t"
        Block 2 lernt: "Wenn (x,y) sich bewegt hat, bewegt sich (x+1,y) auch"
        Block 3 lernt: "Diese Bewegungsmuster deuten auf ein fahrendes Auto hin"
        Block 4 lernt: "Autos fahren weiter geradeaus wenn keine Kurve"
        → Hierarchisches, abstraktes Reasoning über mehrere Ebenen

    WARUM nn.ModuleList STATT einer Python-Liste?
        Python-Liste: PyTorch weiß nichts von den Modulen darin.
            → Parameter werden nicht registriert → nicht trainiert!
        nn.ModuleList: PyTorch registriert alle enthaltenen Module.
            → Parameter erscheinen in model.parameters() → werden trainiert.

    Args:
        config: ModelConfig mit n_layers, und allen Block-Parametern

    Input:  [B, seq_len, d_model]
    Output: [B, seq_len, d_model]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Erstelle N identisch konfigurierte Blöcke.
        # Jeder Block hat seine EIGENEN Parameter — sie teilen sich die
        # Architektur, aber nicht die Gewichte. Das ist wichtig!
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Führt x sequentiell durch alle N Transformer-Blöcke.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            x: [B, seq_len, d_model]  — nach N Attention+FFN-Zyklen

        Die Shape ändert sich NICHT durch den Stack — das ist das Schöne
        an der Transformer-Architektur. Ob 3 oder 3072 Tokens: der Code
        ist identisch, nur die Laufzeit und der VRAM-Bedarf ändern sich
        (quadratisch mit seq_len durch die Attention-Matrix).
        """
        for block in self.blocks:
            x = block(x)       # [B, seq_len, d_model] → [B, seq_len, d_model]
        return x

    def get_all_attention_weights(self, x: torch.Tensor):
        """
        Gibt Attention-Gewichte aller Blöcke zurück.
        Für Ablation Studies und Thesis-Visualisierungen.

        Returns:
            List of [B, n_heads, seq_len, seq_len] — eine pro Block
        """
        all_weights = []
        for block in self.blocks:
            weights = block.get_attention_weights(x)
            all_weights.append(weights)
            # x durch den Block schicken (ohne Gewichte zu speichern)
            with torch.no_grad():
                x = block(x)
        return all_weights
