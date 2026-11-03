"""Point-in-time company and documented-relationship inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch


@dataclass(frozen=True)
class GraphSnapshot:
    """One prediction-date snapshot. Documented edges are directed records.

    ``features`` and ``documented_edges`` must contain only information public
    by ``as_of``. The CSV loader below enforces this using availability dates.
    """

    as_of: date
    company_ids: tuple[str, ...]
    features: torch.Tensor  # [companies, financial features]
    sectors: torch.Tensor  # [companies], stable sector vocabulary IDs
    documented_edges: torch.Tensor  # [2, records], source -> target
    documented_types: torch.Tensor  # [records], stable relation vocabulary IDs
    sector_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    extras: object = None  # optional side tensors (e.g. person mask, hypergraph incidence)

    def __post_init__(self) -> None:
        n = len(self.company_ids)
        if len(set(self.company_ids)) != n:
            raise ValueError("company_ids must be unique")
        if self.features.ndim != 2 or self.features.shape[0] != n:
            raise ValueError("features must have one row per company")
        if not torch.is_floating_point(self.features) or not torch.isfinite(self.features).all():
            raise ValueError("features must be finite floating point values")
        if self.sectors.shape != (n,) or self.sectors.dtype != torch.long:
            raise ValueError("sectors must be a long tensor with one ID per company")
        if n and (self.sectors.min() < 0 or self.sectors.max() >= len(self.sector_names)):
            raise ValueError("sector ID outside sector_names")
        if self.documented_edges.ndim != 2 or self.documented_edges.shape[0] != 2:
            raise ValueError("documented_edges must have shape [2, records]")
        e = self.documented_edges.shape[1]
        if self.documented_edges.dtype != torch.long or self.documented_types.shape != (e,):
            raise ValueError("documented edges and types must be long tensors")
        if self.documented_types.dtype != torch.long:
            raise ValueError("documented_types must be a long tensor")
        if e and (self.documented_edges.min() < 0 or self.documented_edges.max() >= n):
            raise ValueError("documented edge endpoint outside company_ids")
        if e and (self.documented_types.min() < 0 or self.documented_types.max() >= len(self.relation_names)):
            raise ValueError("documented type outside relation_names")


def load_snapshot_csv(
    companies_csv: str | Path,
    relationships_csv: str | Path,
    *,
    as_of: date,
    feature_columns: tuple[str, ...],
    sector_vocab: Mapping[str, int],
    relation_vocab: Mapping[str, int],
) -> GraphSnapshot:
    """Load the latest available company row and eligible links at ``as_of``.

    Companies CSV: company_id, sector, available_on, and specified feature columns.
    Relationships CSV: source, target, relation, available_on. ``available_on``
    means first public availability, not the underlying fiscal period end.
    Vocabulary IDs must be consecutive from zero and reused across all dates.
    """

    if not feature_columns:
        raise ValueError("at least one feature column is required")
    for name, vocab in (("sector_vocab", sector_vocab), ("relation_vocab", relation_vocab)):
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError(f"{name} IDs must be consecutive from zero")

    companies = pd.read_csv(companies_csv, dtype={"company_id": str, "sector": str})
    needed = {"company_id", "sector", "available_on", *feature_columns}
    if not needed.issubset(companies.columns):
        raise ValueError(f"companies CSV missing {sorted(needed - set(companies.columns))}")
    companies["available_on"] = pd.to_datetime(companies["available_on"], errors="raise")
    companies = companies.loc[companies["available_on"] <= pd.Timestamp(as_of)].copy()
    companies = companies.sort_values(["company_id", "available_on"])
    companies = companies.drop_duplicates("company_id", keep="last").reset_index(drop=True)
    if companies.empty:
        raise ValueError("no company features available by as_of")
    unknown_sectors = set(companies["sector"]) - set(sector_vocab)
    if unknown_sectors:
        raise ValueError(f"unknown sectors: {sorted(unknown_sectors)}")

    features = companies[list(feature_columns)].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError("missing feature values; impute using training data only before loading")
    ids = tuple(companies["company_id"].tolist())
    index = {company_id: i for i, company_id in enumerate(ids)}

    links = pd.read_csv(relationships_csv, dtype={"source": str, "target": str, "relation": str})
    needed_links = {"source", "target", "relation", "available_on"}
    if not needed_links.issubset(links.columns):
        raise ValueError(f"relationships CSV missing {sorted(needed_links - set(links.columns))}")
    links["available_on"] = pd.to_datetime(links["available_on"], errors="raise")
    links = links.loc[links["available_on"] <= pd.Timestamp(as_of)]
    links = links.loc[links["source"].isin(index) & links["target"].isin(index)]
    unknown_relations = set(links["relation"]) - set(relation_vocab)
    if unknown_relations:
        raise ValueError(f"unknown relations: {sorted(unknown_relations)}")
    links = links.drop_duplicates(["source", "target", "relation"])
    endpoints = torch.tensor(
        [[index[x] for x in links["source"]], [index[x] for x in links["target"]]],
        dtype=torch.long,
    ).reshape(2, -1)
    types = torch.tensor([relation_vocab[x] for x in links["relation"]], dtype=torch.long)

    return GraphSnapshot(
        as_of=as_of,
        company_ids=ids,
        features=torch.tensor(features.to_numpy(dtype="float32")),
        sectors=torch.tensor([sector_vocab[x] for x in companies["sector"]], dtype=torch.long),
        documented_edges=endpoints,
        documented_types=types,
        sector_names=tuple(sorted(sector_vocab, key=sector_vocab.get)),
        relation_names=tuple(sorted(relation_vocab, key=relation_vocab.get)),
    )
