"""
flow_head.py — Flow-Matching-Residual-Kopf.

Lernt die VERTEILUNG des Residuums r = x_real - x_det per Flow Matching
(Lipman 2023, Rectified-Flow-Pfade; Anker VFMF/FlowWM 2025/26):

    r_tau = (1 - tau) * eps + tau * r,   eps ~ N(0,I), tau ~ U[0,1]
    L_FM  = || v_theta(r_tau, tau, cond) - (r - eps) ||^2

Sampling: Euler-ODE von tau=0 (Rauschen) nach tau=1 (Residuum), N Schritte.
Konditionierung cond = [x_det, letzter Input-Frame] (concat als Kanaele);
tau als Sinus-Embedding, per FiLM auf jeden Block. Groesse ueber flow_dim/
flow_layers config-skalierbar; Default ~6.5M Params = Paritaet zum 6M-Backbone
("gleiche Kapazitaet, anderes Objective"). Output-Conv zero-init (v startet
bei 0 -> stabiler Trainingsbeginn).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(tau: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """tau [B] in [0,1] -> Sinus/Kosinus-Embedding [B, dim] (DDPM-Konvention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
    )
    ang = tau.float()[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowHead(nn.Module):
    """Geschwindigkeitsfeld v_theta(r_tau, tau, cond) -> [B, C, H, W]."""

    def __init__(self, config):
        super().__init__()
        C = config.d_model                                  # Latent-Kanaele (256)
        W = getattr(config, "flow_dim", 224)
        L = getattr(config, "flow_layers", 8)
        self.n_layers = L
        self.width = W

        # Input: r_tau + x_det + skip (letzter Input-Frame) als Kanaele
        self.inp = nn.Conv2d(3 * C, W, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(W, W, kernel_size=3, padding=1) for _ in range(L)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(8, W) for _ in range(L))
        # tau-Embedding -> per-Block FiLM (Scale, Shift)
        self.tau_mlp = nn.Sequential(
            nn.Linear(128, W), nn.SiLU(), nn.Linear(W, 2 * W * L),
        )
        self.out = nn.Conv2d(W, C, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, r_tau, tau, x_det, skip):
        """
        r_tau: [B,C,H,W] Punkt auf dem Rausch->Residuum-Pfad
        tau:   [B]       Pfadposition in [0,1]
        x_det: [B,C,H,W] deterministische Backbone-Vorhersage (Konditionierung)
        skip:  [B,C,H,W] letzter Input-Frame (Konditionierung)
        """
        B = r_tau.shape[0]
        film = self.tau_mlp(sinusoidal_embedding(tau)).view(B, self.n_layers, 2, self.width)
        x = self.inp(torch.cat([r_tau, x_det, skip], dim=1))
        for i, (blk, nrm) in enumerate(zip(self.blocks, self.norms)):
            h = nrm(x)
            s, b = film[:, i, 0], film[:, i, 1]
            h = h * (1.0 + s[:, :, None, None]) + b[:, :, None, None]
            x = x + blk(F.silu(h))
        return self.out(F.silu(x))

    @torch.no_grad()
    def sample(self, x_det, skip, steps: int = 10, generator=None):
        """
        Euler-Integration dx/dtau = v_theta von tau=0 (eps~N(0,I)) nach tau=1.
        Rueckgabe: Residuum-Sample r_hat [B,C,H,W]. generator (torch.Generator
        auf demselben Device) macht das Sample reproduzierbar.
        """
        if generator is not None:
            r = torch.empty_like(x_det).normal_(generator=generator)
        else:
            r = torch.randn_like(x_det)
        dt = 1.0 / steps
        for k in range(steps):
            tau = torch.full((x_det.shape[0],), k * dt, device=x_det.device)
            r = r + dt * self.forward(r, tau, x_det, skip)
        return r
