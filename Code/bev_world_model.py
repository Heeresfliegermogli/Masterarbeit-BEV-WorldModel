"""
bev_world_model.py — BEV World Model: Komplettes Modell
============================================================================

Steckt alle Einzel-Module zu einem einzigen nn.Module zusammen.

PIPELINE KOMPLETT:

    Input:  [B, 3, 256, 128, 128]   ← 3 aufeinanderfolgende BEV-Latents
         |
         v
    DownsamplingModule               128×128 → 32×32 (AvgPool, 0 Parameter)
         |
         v
    FrameLevelEmbedding              [B, 3, 256, 32, 32] → [B,    3, 256]
    oder CellLevelEmbedding          [B, 3, 256, 32, 32] → [B, 3072, 256]
         |
         v
    TransformerEncoder               [B, seq, 256] → [B, seq, 256]
    (N=4 Blöcke, phase-agnostisch)
         |
         v
    FrameLevelOutputHead             [B,    3, 256] → [B, 256, 32, 32]
    oder CellLevelOutputHead         [B, 3072, 256] → [B, 256, 32, 32]
         |
         v
    UpsamplingHead                   [B, 256, 32, 32] → [B, 256, 128, 128]
         |
         v
    Output: [B, 256, 128, 128]      ← predicted nächster BEV-Latent t+1

PHASE WECHSELN:
    Nur config.phase ändern — "frame" oder "cell".
    Alle anderen Module bleiben identisch.

    cfg = ModelConfig(phase="frame")   # Phase 1: schnell, 3 Tokens
    cfg = ModelConfig(phase="cell")    # Phase 2: vollständig, 3072 Tokens

IMPORTS FÜR TRAIN.PY:
    from bev_world_model import BEVWorldModel
    from config          import ModelConfig

    cfg   = ModelConfig(phase="frame")
    model = BEVWorldModel(cfg).to(device)
"""

import torch
import torch.nn as nn

from config          import ModelConfig
from downsampling    import DownsamplingModule
from embedding       import FrameLevelEmbedding, CellLevelEmbedding
from transformer     import TransformerEncoder
from output_head     import FrameLevelOutputHead, CellLevelOutputHead
from upsampling_head import UpsamplingHead


class BEVWorldModel(nn.Module):
    """
    Komplettes BEV World Model — orchestriert alle Teilmodule.

    Nimmt 3 aufeinanderfolgende BEV-Latents und sagt den nächsten vorher.

    WELCHE MODULE WERDEN GELADEN?
        Abhängig von config.phase wählt __init__ die richtigen Klassen:

        Phase 1 ("frame"):
            embedding   = FrameLevelEmbedding  → 3 Tokens
            output_head = FrameLevelOutputHead  → letzter Token → Grid

        Phase 2 ("cell"):
            embedding   = CellLevelEmbedding   → 3072 Tokens
            output_head = CellLevelOutputHead  → letzter Frame → Grid

        Alle anderen Module (downsample, transformer, upsample) sind
        phase-agnostisch — identischer Code für beide Phasen.

    PARAMETER-ÜBERSICHT (Phase 1, N=4):
        DownsamplingModule       0 Parameter   (AvgPool — keine Gewichte)
        FrameLevelEmbedding     ~132k Parameter (Linear + PE_t)
        TransformerEncoder      ~3.2M Parameter (4 Blöcke)
        FrameLevelOutputHead     ~66k Parameter (Linear)
        UpsamplingHead          ~2.7M Parameter (ConvTranspose × 2 + Conv)
        ─────────────────────────────────────────
        Gesamt Phase 1:         ~6.1M Parameter

    PARAMETER-ÜBERSICHT (Phase 2, N=4):
        DownsamplingModule       0 Parameter
        CellLevelEmbedding      ~74k Parameter  (Linear + PE_x + PE_y + PE_t)
        TransformerEncoder      ~3.2M Parameter
        CellLevelOutputHead     ~66k Parameter
        UpsamplingHead          ~2.7M Parameter
        ─────────────────────────────────────────
        Gesamt Phase 2:         ~6.0M Parameter

    Args:
        config: ModelConfig mit allen Hyperparametern und phase-Steuerung

    Input:  [B, n_frames, C, H, W]      — z.B. [B, 3, 256, 128, 128]
    Output: [B, C, H, W]                — z.B. [B, 256, 128, 128]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Downsampling — identisch für beide Phasen
        # ------------------------------------------------------------------
        # 128×128 → 32×32, 0 Parameter, kein Training nötig
        self.downsample = DownsamplingModule()

        # ------------------------------------------------------------------
        # 2. Embedding — phase-abhängig
        # ------------------------------------------------------------------
        # config.phase steuert welche Klasse instanziiert wird.
        # Um auf Phase 2 zu wechseln: nur config.phase = "cell" setzen.
        if config.phase == "frame":
            self.embedding = FrameLevelEmbedding(config)   # → [B, 3, 256]
        else:
            self.embedding = CellLevelEmbedding(config)    # → [B, 3072, 256]

        # ------------------------------------------------------------------
        # 3. Transformer — identisch für beide Phasen
        # ------------------------------------------------------------------
        # Verarbeitet [B, seq_len, 256] — seq_len ist 3 oder 3072,
        # der Transformer merkt den Unterschied nicht.
        self.transformer = TransformerEncoder(config)

        # ------------------------------------------------------------------
        # 4. Output Head — phase-abhängig
        # ------------------------------------------------------------------
        # Beide Heads liefern [B, 256, 32, 32] — UpsamplingHead ist identisch.
        if config.phase == "frame":
            self.output_head = FrameLevelOutputHead(config)  # letzter Token
        else:
            self.output_head = CellLevelOutputHead(config)   # letzter Frame

        # ------------------------------------------------------------------
        # 5. Upsampling — identisch für beide Phasen
        # ------------------------------------------------------------------
        # [B, 256, 32, 32] → [B, 256, 128, 128]
        self.upsample = UpsamplingHead(config)

        # ------------------------------------------------------------------
        # 6. Gated Skip Connection
        # ------------------------------------------------------------------
        # Lernbares Gate α ∈ [0,1] pro Channel und räumlicher Position.
        # Steuert wie stark der Transformer-Output vs. Frame t genutzt wird.
        #
        # WARUM GATE statt simpler Addition?
        #   Einfache Addition:  pred = transformer_out + skip
        #   → Modell muss exakt Δ(t→t+1) lernen
        #   → Bei großer Bewegung (Fahrzeuge) kämpft skip gegen transformer_out
        #
        #   Gated:  pred = α * transformer_out + (1-α) * skip
        #   → α=0: nur skip (statische Bereiche — Straße, Gebäude)
        #   → α=1: nur transformer (dynamische Bereiche — Fahrzeuge)
        #   → Das Modell lernt selbst wo es predicten und wo es kopieren soll
        #
        # IMPLEMENTIERUNG: 1×1 Conv → Sigmoid
        #   Input:  transformer_out [B, 256, 128, 128]
        #   Output: gate α          [B, 256, 128, 128] ∈ [0,1]
        #   Parameter: 256×256+256 = 65.792 (klein aber lernbar)
        C = config.d_model  # 256
        self.gate = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=True),  # 1x1 Conv, channel-wise
            nn.Sigmoid(),                                # → [0, 1]
        )
        # Gate-Bias auf 0 initialisieren → Sigmoid(0) = 0.5
        # Am Anfang: 50/50 zwischen transformer und skip
        nn.init.zeros_(self.gate[0].bias)

        # --- Ego-Motion-Conditioning (FiLM auf den eingebetteten Tokens) ---
        # ego_delta pro Sample: [n_frames*4] = [dx,dy,cos,sin] je Schritt (inkl.
        # F_last->target). Modus waehlt den Slice: action=letzte 4, state=erste
        # (n_frames-1)*4, both=alle. FiLM startet als IDENTITAET (gamma=1,beta=0)
        # -> ego_cond_mode="off" ist bit-identisch zum Modell ohne Conditioning.
        # Modi: off | action/state/both (FiLM auf Tokens) | gate (Bias auf Gate-
        # Logits) | token (Ego-Token vor die Sequenz). action/gate/token nutzen die
        # F3->F4-Aktion (4-dim). Alle Zusatzkoepfe ZERO-INIT -> Start = altes Modell.
        self.ego_cond_mode = getattr(config, "ego_cond_mode", "off")
        self._n_frames = config.n_frames
        if self.ego_cond_mode != "off":
            _ed = {"action": 4, "gate": 4, "token": 4,
                   "state": (config.n_frames - 1) * 4,
                   "both": config.n_frames * 4}[self.ego_cond_mode]
            if self.ego_cond_mode in ("action", "state", "both"):
                self.film = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, 2 * C))
                nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            elif self.ego_cond_mode == "gate":
                # per-Channel additiver Bias auf die Gate-Logits (zero-init = No-op)
                self.film_gate = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, C))
                nn.init.zeros_(self.film_gate[-1].weight); nn.init.zeros_(self.film_gate[-1].bias)
            elif self.ego_cond_mode == "token":
                # Ego-Token, der der Token-Sequenz vorangestellt wird
                self.ego_token_mlp = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, C))
                nn.init.zeros_(self.ego_token_mlp[-1].weight); nn.init.zeros_(self.ego_token_mlp[-1].bias)

        # --- CVAE-Kopf -------------------------------------------
        # Latente Variable z kodiert "welche der moeglichen Zukuenfte". Prior
        # p(z|h) sieht nur den Kontext (Token-Mean h), Posterior q(z|h,target)
        # zusaetzlich das echte Zielframe (via vorhandenem Downsampling, kein
        # neuer Encoder). z wird per FiLM auf die Tokens vor dem Output-Head
        # injiziert; FiLM ZERO-INIT -> am Init hat z null Effekt = off-identischer
        # Hauptpfad. vae_mode="off" (Default) baut KEINES dieser Module.
        self.vae_mode = getattr(config, "vae_mode", "off")
        if self.vae_mode == "cvae":
            Z = getattr(config, "vae_z_dim", 32)
            self.vae_prior = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 2 * Z))
            self.vae_post  = nn.Sequential(nn.Linear(2 * C, C), nn.GELU(), nn.Linear(C, 2 * Z))
            self.vae_film  = nn.Sequential(nn.Linear(Z, C), nn.GELU(), nn.Linear(C, 2 * C))
            nn.init.zeros_(self.vae_film[-1].weight); nn.init.zeros_(self.vae_film[-1].bias)
        # KL-Statistik des letzten Forwards: (mu_q, logvar_q, mu_p, logvar_p)
        # bzw. (None, None, mu_p, logvar_p) bei Inferenz. Immer vorhanden.
        self.last_vae_stats = None

        # --- Flow-Matching-Residual-Kopf -------------------------
        # WICHTIG: veraendert forward() NICHT — der Kopf wird ausschliesslich
        # explizit genutzt (Training: model.flow(...); Sampling:
        # model.flow_sample(...)). flow_head="off" baut nichts.
        if getattr(config, "flow_head", "off") == "flow":
            from flow_head import FlowHead
            self.flow = FlowHead(config)

    def _ego_sig(self, ego_delta):
        """Signal-Slice je Modus: state=erste (n-1)*4, both=alle, sonst F3->F4 (letzte 4)."""
        if self.ego_cond_mode == "state":
            return ego_delta[:, :(self._n_frames - 1) * 4]
        if self.ego_cond_mode == "both":
            return ego_delta
        return ego_delta[:, -4:]

    def forward(self, x: torch.Tensor, ego_delta: torch.Tensor = None,
                target: torch.Tensor = None, z_mode: str = "mean") -> torch.Tensor:
        """
        Kompletter Forward-Pass durch alle 5 Stufen + Skip Connection.

        Args:
            x: [B, n_frames, C, H, W]  — z.B. [B, 3, 256, 128, 128]
               Batch aufeinanderfolgender BEV-Latents (t-2, t-1, t)

        Returns:
            [B, C, H, W]  — z.B. [B, 256, 128, 128]
            Predicted BEV-Latent für Zeitschritt t+1

        SKIP CONNECTION — WARUM UND WO:
            skip = x[:, -1, ...]  speichert Frame t (den aktuellsten Input)
            BEVOR das Downsampling die räumlichen Details zerstört.

            Nach dem Upsampling wird skip addiert:
                output = upsample(transformer(embedding(downsample(x)))) + skip

            Semantisch bedeutet das:
                pred(t+1) = Δ + t
                          = "was ändert sich" + "was jetzt gerade ist"

            Der Transformer lernt also nur die DIFFERENZ (Δ) zwischen
            t und t+1 — nicht t+1 von Grund auf. Das ist viel einfacher
            weil bei 2Hz (0.5s zwischen Frames) der Großteil der BEV-Szene
            identisch bleibt (Straße, statische Objekte, Gebäude).

            Zusätzlicher Effekt: Die feinen Channel-Strukturen aus BEVFusion
            (hochfrequente Details bei 128×128) bleiben erhalten — direkt
            wichtig für die spätere Decoder-Kompatibilität.
        """

        # --- Skip Connection: Frame t merken BEVOR Downsampling ---
        # x[:, -1, :, :, :] = letzter Frame in der Zeitdimension = Frame t
        # Shape: [B, 256, 128, 128] — volle Auflösung, alle Details
        skip = x[:, -1, :, :, :]                    # [B, C, H, W]

        # Stufe 1: Räumliches Downsampling
        # [B, 3, 256, 128, 128] → [B, 3, 256, 32, 32]
        x = self.downsample(x)

        # Stufe 2: Token Embedding + Positional Encoding
        # Phase 1: [B, 3, 256, 32, 32] → [B,    3, 256]
        # Phase 2: [B, 3, 256, 32, 32] → [B, 3072, 256]
        x = self.embedding(x)

        # Ego-Conditioning. FiLM (action/state/both) moduliert die Tokens;
        # token stellt ein Ego-Token voran; gate wirkt spaeter auf die Gate-Logits.
        _ego_tok = False
        if self.ego_cond_mode != "off" and ego_delta is not None:
            e = self._ego_sig(ego_delta)
            if self.ego_cond_mode in ("action", "state", "both"):
                gamma, beta = self.film(e).chunk(2, dim=-1)      # je [B, C]
                x = (1.0 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)
            elif self.ego_cond_mode == "token":
                tok = self.ego_token_mlp(e).unsqueeze(1)         # [B,1,C]
                x = torch.cat([tok, x], dim=1)                   # [B, seq+1, C]
                _ego_tok = True

        # Stufe 3: Transformer (Self-Attention + FFN × N)
        # [B, seq, 256] → [B, seq, 256]  — Shape unveränderlich
        x = self.transformer(x)
        if _ego_tok:
            x = x[:, 1:]                                         # Ego-Token wieder abschneiden

        # --- CVAE — z ziehen und per FiLM injizieren -------------
        # Training (target gegeben): z ~ q(z|h,target) via Reparametrisierung.
        # Inferenz (target=None): z aus p(z|h) — z_mode "mean" (deterministisch,
        # Default: alle bestehenden Eval-Pfade bleiben reproduzierbar) oder
        # "sample" (stochastisch, fuer Diversitaets-/Schaerfe-Eval).
        # logvar-Clamp [-8, 8] gegen exp-Overflow in fruehen Epochen.
        if self.vae_mode == "cvae":
            h = x.mean(dim=1)                                    # [B, C] Kontext
            mu_p, lv_p = self.vae_prior(h).chunk(2, dim=-1)
            lv_p = lv_p.clamp(-8.0, 8.0)
            if target is not None:
                t = self.downsample(target.unsqueeze(1)).squeeze(1)   # [B,C,32,32]
                t = t.flatten(2).mean(dim=2)                          # [B, C]
                mu_q, lv_q = self.vae_post(torch.cat([h, t], dim=-1)).chunk(2, dim=-1)
                lv_q = lv_q.clamp(-8.0, 8.0)
                z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * lv_q)
                self.last_vae_stats = (mu_q, lv_q, mu_p, lv_p)
            else:
                if z_mode == "sample":
                    z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * lv_p)
                else:
                    z = mu_p
                self.last_vae_stats = (None, None, mu_p, lv_p)
            gz, bz = self.vae_film(z).chunk(2, dim=-1)           # je [B, C]
            x = (1.0 + gz).unsqueeze(1) * x + bz.unsqueeze(1)

        # Stufe 4: Output Head — Tokens zurück in ein Grid
        # Phase 1: [B,    3, 256] → [B, 256, 32, 32]
        # Phase 2: [B, 3072, 256] → [B, 256, 32, 32]
        x = self.output_head(x)

        # Stufe 5: Upsampling auf Original-Auflösung
        # [B, 256, 32, 32] → [B, 256, 128, 128]
        x = self.upsample(x)

        # --- Gated Skip Connection ---
        # Gate α = Sigmoid(Conv1x1(transformer_out)) ∈ [0,1]
        # pred = α * transformer_out + (1-α) * skip
        #
        # α nahe 0: skip dominiert → statische Bereiche werden kopiert
        # α nahe 1: transformer dominiert → Bewegung wird predictet
        #
        # Vorteil gegenüber einfacher Addition (x + skip):
        #   - Kein Kampf zwischen Δ und skip bei großer Bewegung
        #   - Modell kann räumlich differenzieren (pro Pixel, pro Channel)
        #   - Visualisierbar: gate.mean(dim=1) zeigt wo Modell predictet vs kopiert
        # gate-Modus: per-Channel Ego-Bias auf die Gate-Logits (vor Sigmoid),
        # sonst der normale Gate. Zero-init -> Start identisch zum Gate ohne Ego.
        # residual-Modus (ResWorld/DeltaWorld-Muster): hartes Residual
        # statt Gate-Blend -- der Head sagt das DELTA zum letzten Frame vorher.
        if getattr(self.config, "output_mode", "gated") == "residual":
            return skip + x                         # [B, 256, 128, 128]

        if self.ego_cond_mode == "gate" and ego_delta is not None:
            b = self.film_gate(self._ego_sig(ego_delta))          # [B, C]
            alpha = torch.sigmoid(self.gate[0](x) + b[:, :, None, None])
        else:
            alpha = self.gate(x)                    # [B, 256, 128, 128] ∈ [0,1]
        x     = alpha * x + (1.0 - alpha) * skip    # [B, 256, 128, 128]

        return x

    @torch.no_grad()
    def flow_sample(self, x_input, ego_delta=None, steps: int = 10, generator=None,
                    scale: float = 1.0):
        """
        18/B2: EIN stochastisches Sample x = x_det + scale * r_hat.
        x_input: [B, n_frames, C, H, W] (wie forward). Erfordert flow_head=flow.
        scale (18/B2.1): post-hoc Residual-Skalierung (Temperature-Analogon)
        zur Varianz-Kalibrierung; 1.0 = unveraendert.
        """
        x_det = self.forward(x_input, ego_delta)
        skip = x_input[:, -1, :, :, :]
        r_hat = self.flow.sample(x_det, skip, steps=steps, generator=generator)
        return x_det + scale * r_hat

    def count_parameters(self) -> dict:
        """
        Gibt Parameter-Anzahl pro Modul zurück.
        Nützlich für VRAM-Abschätzung und Thesis-Tabelle.

        Returns:
            dict mit Modulnamen → Anzahl Parameter
        """
        modules = {
            "downsample":   self.downsample,
            "embedding":    self.embedding,
            "transformer":  self.transformer,
            "output_head":  self.output_head,
            "upsample":     self.upsample,
            "gate":         self.gate,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in modules.items()
        }
        counts["total"] = sum(counts.values())
        return counts


# ===========================================================================
# Selbst-Test wenn Datei direkt ausgeführt wird
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BEVWorldModel — Selbst-Test")
    print("=" * 60)

    B = 2   # kleiner Batch für schnellen Test

    for phase in ["frame", "cell"]:
        print(f"\n--- Phase: '{phase}' ---")

        cfg   = ModelConfig(phase=phase)
        model = BEVWorldModel(cfg)
        model.eval()

        # Simulierter Input: 3 BEV-Latents [B, 3, 256, 128, 128]
        x = torch.randn(B, cfg.n_frames, 256, 128, 128)
        print(f"  Input:  {list(x.shape)}")

        with torch.no_grad():
            out = model(x)
        print(f"  Output: {list(out.shape)}")

        assert out.shape == (B, 256, 128, 128), \
            f"Shape falsch: {out.shape}"
        assert not torch.isnan(out).any(), "NaN im Output!"
        print(f"  ✓ Shape korrekt | ✓ Kein NaN")

        # Parameter-Übersicht
        counts = model.count_parameters()
        print(f"\n  Parameter pro Modul:")
        for name, n in counts.items():
            if name != "total":
                print(f"    {name:<15} {n:>10,}")
        print(f"    {'─'*26}")
        print(f"    {'total':<15} {counts['total']:>10,}")

    print(f"\n{'='*60}")
    print("  ✅  Selbst-Test bestanden.")
    print(f"{'='*60}\n")
