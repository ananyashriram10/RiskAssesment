"""Lawsuit-vector encoder and a wrapper that feeds it into the sector-gated GNN."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.nn import functional as F

from risk_gnn import GateConfig, GraphSnapshot, SectorRiskGNN


class LawsuitEncoder(nn.Module):
    """One vector per company from its lawsuit records.

    Each lawsuit (cause, court, result, age in years, recent flag) is embedded; an attention score
    plus a learned recency decay weights the lawsuits, and the weighted sum is the company vector.
    Output: [pooled vector (d), log(1 + #lawsuits), has-lawsuit flag].

    ``use_timing=False`` removes the lawsuit age and recent flag from the inputs and the recency decay
    from the attention, leaving cause, court, result and the lawsuit count (SMEsD measures lawsuit age
    from an outcome-aligned observation date, so age does not transfer to later periods)."""

    def __init__(self, dim: int = 16, use_timing: bool = True):
        super().__init__()
        self.dim = dim
        self.use_timing = use_timing
        self.cause, self.court, self.result = nn.Embedding(11, 6), nn.Embedding(4, 3), nn.Embedding(4, 3)
        self.mlp = nn.Sequential(nn.Linear(6 + 3 + 3 + (2 if use_timing else 0), 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.score = nn.Linear(dim, 1)
        self.raw_decay = nn.Parameter(torch.tensor(-1.0))   # decay per year = softplus(raw_decay)

    def forward(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        years, mask = seq["years"], seq["mask"]
        parts = [self.cause(seq["cause"]), self.court(seq["court"]), self.result(seq["result"])]
        if self.use_timing:
            parts += [years.unsqueeze(-1), seq["recent"].unsqueeze(-1)]
        v = self.mlp(torch.cat(parts, dim=-1))
        logit = self.score(v).squeeze(-1)
        if self.use_timing:
            logit = logit - F.softplus(self.raw_decay) * years
        logit = logit.masked_fill(~mask, float("-inf"))
        has = mask.any(dim=1)
        safe = torch.where(has.unsqueeze(-1), logit, torch.zeros_like(logit))
        attn = torch.softmax(safe, dim=1) * mask
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-9)
        pooled = (attn.unsqueeze(-1) * v).sum(dim=1) * has.unsqueeze(-1)
        return torch.cat([pooled, torch.log1p(mask.sum(dim=1, keepdim=True).float()), has.unsqueeze(-1).float()], dim=-1)

    @property
    def out_dim(self) -> int:
        return self.dim + 2


class SMEModel(nn.Module):
    """Sector-gated GNN, optionally fed by the lawsuit encoder (persons get zero lawsuit features)."""

    def __init__(self, base_dim: int, sectors: int, relations: int, config: GateConfig, lawsuit_dim: int | None = None,
                 bin_edges: torch.Tensor | None = None, lawsuit_timing: bool = True):
        super().__init__()
        self.encoder = LawsuitEncoder(lawsuit_dim, lawsuit_timing) if lawsuit_dim else None
        extra = self.encoder.out_dim if self.encoder is not None else 0
        self.gnn = SectorRiskGNN(base_dim + extra, sectors, relations, config, bin_edges)

    def forward(self, snapshot: GraphSnapshot, sector: int, seq: dict | None = None):
        if self.encoder is not None:
            v = self.encoder(seq)
            n_nodes = snapshot.features.shape[0]
            pad = v.new_zeros((n_nodes - v.shape[0], v.shape[1]))
            snapshot = dataclasses.replace(snapshot, features=torch.cat([snapshot.features, torch.cat([v, pad])], dim=1))
        return self.gnn(snapshot, sector)
