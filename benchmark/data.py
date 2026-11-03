"""SMEsD loader: official split, node features, graph-degree features, industries,
and company-company edges projected through shared persons.

Pickles are loaded with a restricted unpickler (plain data + NumPy arrays only).
Company nodes are 0..3975; person nodes are >= 3976 (see ComRisk/SMEsD.md).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
import pickle

import numpy as np
import torch

from risk_gnn import GraphSnapshot

DATA = Path(__file__).resolve().parents[1] / "ComRisk" / "data"
N_COMPANIES = 3976
CAUSES, COURTS, RESULTS = 11, 4, 4
RELATIONS = ("shared_supervisor", "shared_executive", "shared_stakeholder", "shared_investor",
             "cc_invest", "cc_branch", "cc_finance")
PERSON_FAMILY = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 8: 3, 9: 3}   # edge type -> shared_* relation
CC_FAMILY = {6: 4, 7: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6, 19: 6}
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype")}


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _load(name: str):
    with open(DATA / name, "rb") as handle:
        return _Safe(handle).load()


def raw_splits() -> dict[str, dict]:
    files = {"train": "train_data.pkl", "valid": "validate_data.pkl", "test": "test_data.pkl"}
    ids = _load("split_data_idx.pkl")
    out = {}
    for position, (name, file) in enumerate(files.items()):
        lawsuits, attrs, graph, hyper, labels = _load(file)
        out[name] = {"ids": [int(i) for i in ids[position]], "lawsuits": lawsuits, "attrs": attrs,
                     "edge_index": np.asarray(graph[0]), "edge_type": np.asarray(graph[1]),
                     "hyper": hyper, "labels": np.asarray(labels, dtype=float)}
        if not (len(out[name]["ids"]) == len(attrs) == len(labels)):
            raise ValueError(f"{name}: id/attribute/label length mismatch")
    return out


def lawsuit_features(records) -> np.ndarray:
    """Counts by cause/court/result, totals, recent counts, and month statistics."""
    cause, court, result = np.zeros(CAUSES), np.zeros(COURTS), np.zeros(RESULTS)
    months, recent = [], 0
    for c, k, r, m, flag in records:
        cause[int(c)] += 1; court[int(k)] += 1; result[int(r)] += 1
        months.append(float(m)); recent += int(flag == 1)
    n = len(records)
    stats = [n, recent, np.min(months) if months else 0.0, np.mean(months) if months else 0.0]
    return np.concatenate([np.log1p(cause), np.log1p(court), np.log1p(result),
                           np.log1p([stats[0], stats[1]]), [stats[2] / 12.0, stats[3] / 12.0, float(n > 0)]])


def node_features(split: dict) -> np.ndarray:
    rows = []
    for cid, attr in zip(split["ids"], split["attrs"]):
        reg, paid, setup = (float(attr[0]), float(attr[1]), float(attr[2]))
        base = [np.log1p(max(reg, 0)), np.log1p(max(paid, 0)), setup / 12.0]
        rows.append(np.concatenate([base, lawsuit_features(split["lawsuits"].get(cid, []))]))
    return np.asarray(rows, dtype=np.float64)


def degree_features(split: dict) -> np.ndarray:
    """Per-relation-type edge counts incident to each company (20 types) plus total."""
    pos = {c: i for i, c in enumerate(split["ids"])}
    deg = np.zeros((len(pos), 20))
    for (a, b), t in zip(split["edge_index"], split["edge_type"]):
        for node in (int(a), int(b)):
            if node in pos:
                deg[pos[node], int(t)] += 1
    return np.column_stack([np.log1p(deg), np.log1p(deg.sum(1))])


def projected_edges(split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Company-company edges (positions in split['ids']) and relation ids.

    Two companies sharing a person under one relation family are linked; direct
    company-company relations are kept. Pairs are undirected and stored once."""
    pos = {c: i for i, c in enumerate(split["ids"])}
    by_person: dict[tuple[int, int], set[int]] = defaultdict(set)
    pairs: dict[tuple[int, int, int], None] = {}
    for (a, b), t in zip(split["edge_index"], split["edge_type"]):
        a, b, t = int(a), int(b), int(t)
        if t in PERSON_FAMILY:
            person, company = (b, a) if a < N_COMPANIES else (a, b)
            if person >= N_COMPANIES and company in pos:
                by_person[(PERSON_FAMILY[t], person)].add(pos[company])
        elif t in CC_FAMILY and a in pos and b in pos and a != b:
            i, j = sorted((pos[a], pos[b]))
            pairs[(i, j, CC_FAMILY[t])] = None
    for (relation, _), members in by_person.items():
        ordered = sorted(members)
        for x in range(len(ordered)):
            for y in range(x + 1, len(ordered)):
                pairs[(ordered[x], ordered[y], relation)] = None
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    arr = np.array(list(pairs.keys()), dtype=np.int64)
    return arr[:, :2].T.copy(), arr[:, 2].copy()


def industries(split: dict) -> list[str]:
    owner: dict[int, str] = {}
    for group, nodes in split["hyper"]["industry"].items():
        for node in nodes:
            owner.setdefault(int(node), str(group))
    return [owner.get(c, "unknown") for c in split["ids"]]


def make_snapshot(split: dict, features: np.ndarray, sector_vocab: dict[str, int],
                  edge_index: np.ndarray, edge_type: np.ndarray) -> GraphSnapshot:
    return GraphSnapshot(
        as_of=date(2021, 12, 31),
        company_ids=tuple(str(c) for c in split["ids"]),
        features=torch.tensor(features, dtype=torch.float32),
        sectors=torch.tensor([sector_vocab[s] for s in industries(split)], dtype=torch.long),
        documented_edges=torch.tensor(edge_index, dtype=torch.long).reshape(2, -1),
        documented_types=torch.tensor(edge_type, dtype=torch.long),
        sector_names=tuple(sorted(sector_vocab, key=sector_vocab.get)),
        relation_names=RELATIONS,
    )


def shuffle_edges(edge_index: np.ndarray, edge_type: np.ndarray, seed: int) -> np.ndarray:
    """Within each relation, permute the destination column: both degree marginals are kept
    but who is connected to whom is random. Self-loops are dropped by re-pairing."""
    rng = np.random.default_rng(seed)
    out = edge_index.copy()
    for relation in np.unique(edge_type):
        cols = np.where(edge_type == relation)[0]
        out[1, cols] = out[1, rng.permutation(cols)]
    loops = out[0] == out[1]
    if loops.any():  # swap destinations of self-loops with other edges of the same relation
        for col in np.where(loops)[0]:
            same = np.where((edge_type == edge_type[col]) & ~loops)[0]
            if len(same):
                other = rng.choice(same)
                out[1, col], out[1, other] = out[1, other], out[1, col]
    return out


def labels(split: dict) -> np.ndarray:
    return split["labels"]


def meta_features(split: dict) -> np.ndarray:
    """Pretrained metapath2vec company embeddings shipped with ComRisk (unsupervised, graph only)."""
    company_emb, person_emb = _load("meta_emb.pkl")   # (3976, 32) companies, (2405, 32) persons
    if np.asarray(company_emb).shape[0] != N_COMPANIES:
        raise ValueError("unexpected company embedding shape")
    return np.asarray(company_emb, dtype=np.float64)[np.asarray(split["ids"])]


def person_embeddings(person_ids: list[int]) -> np.ndarray:
    """Pretrained metapath2vec embeddings of person nodes (row = person index - 3976)."""
    _, person_emb = _load("meta_emb.pkl")
    person_emb = np.asarray(person_emb, dtype=np.float64)
    rows = np.asarray(person_ids) - N_COMPANIES
    if rows.size and (rows.min() < 0 or rows.max() >= person_emb.shape[0]):
        raise ValueError("person index outside the embedding table")
    return person_emb[rows]


def group_features(split: dict, base: np.ndarray, degree_total: np.ndarray | None) -> np.ndarray:
    """Hypergraph context: for industry, area and qualify groups, log group size and the mean of
    three company attributes and total degree over the *other* members of the same split."""
    pos = {c: i for i, c in enumerate(split["ids"])}
    signal = base[:, :3] if degree_total is None else np.column_stack([base[:, :3], degree_total])
    out = []
    for kind in ("industry", "area", "qualify"):
        size = np.zeros(len(pos)); mean = np.zeros((len(pos), signal.shape[1]))
        member_of: dict[int, list[int]] = defaultdict(list)
        for group, nodes in split["hyper"][kind].items():
            idx = [pos[int(n)] for n in nodes if int(n) in pos]
            for i in idx:
                member_of[i].append(len(member_of[i]))  # placeholder to count memberships
            if len(idx) < 2:
                continue
            total = signal[idx].sum(0)
            for i in idx:
                size[i] += len(idx) - 1
                mean[i] += (total - signal[i])
        denom = np.maximum(size, 1)[:, None]
        out.append(np.column_stack([np.log1p(size), mean / denom]))
    return np.concatenate(out, axis=1)


def hetero_graph(split: dict) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Person nodes as real graph nodes. Returns (person ids, edges, relation ids) where edge
    endpoints index into [companies of the split] + [persons], company-person edges stored
    once as (company, person) with relations 0-3, company-company edges (undirected, once)
    with relations 4-6."""
    pos = {c: i for i, c in enumerate(split["ids"])}
    persons: dict[int, int] = {}
    edges: dict[tuple[int, int, int], None] = {}
    for (a, b), t in zip(split["edge_index"], split["edge_type"]):
        a, b, t = int(a), int(b), int(t)
        if t in PERSON_FAMILY:
            person, company = (b, a) if a < N_COMPANIES else (a, b)
            if person >= N_COMPANIES and company in pos:
                persons.setdefault(person, len(split["ids"]) + len(persons))
                edges[(pos[company], persons[person], PERSON_FAMILY[t])] = None
        elif t in CC_FAMILY and a in pos and b in pos and a != b:
            i, j = sorted((pos[a], pos[b]))
            edges[(i, j, CC_FAMILY[t])] = None
    arr = np.array(list(edges.keys()), dtype=np.int64) if edges else np.zeros((0, 3), dtype=np.int64)
    return list(persons), arr[:, :2].T.copy(), arr[:, 2].copy()


def make_hetero_snapshot(split: dict, features: np.ndarray, sector_vocab: dict[str, int],
                         person_block=None) -> GraphSnapshot:
    persons, ei, et = hetero_graph(split)
    n_c, n_p = len(split["ids"]), len(persons)
    feats = np.zeros((n_c + n_p, features.shape[1] + 1), dtype=np.float64)
    feats[:n_c, :-1] = features
    if person_block is not None:                       # callable: person ids -> (n_p, width) scaled block
        feats[n_c:, :-1] = person_block(persons)
    feats[n_c:, -1] = 1.0   # is_person indicator; person nodes carry no other attributes
    person_sector = sector_vocab["__person__"]
    sectors = [sector_vocab[g] for g in industries(split)] + [person_sector] * n_p
    is_person = torch.zeros(n_c + n_p, dtype=torch.bool)
    is_person[n_c:] = True
    return GraphSnapshot(
        as_of=date(2021, 12, 31),
        extras={"is_person": is_person, "hyper": hyper_incidence(split)},
        company_ids=tuple(str(c) for c in split["ids"]) + tuple(f"p{p}" for p in persons),
        features=torch.tensor(feats, dtype=torch.float32),
        sectors=torch.tensor(sectors, dtype=torch.long),
        documented_edges=torch.tensor(ei, dtype=torch.long).reshape(2, -1),
        documented_types=torch.tensor(et, dtype=torch.long),
        sector_names=tuple(sorted(sector_vocab, key=sector_vocab.get)),
        relation_names=RELATIONS,
    )


def hyper_incidence(split: dict):
    """(node positions, hyperedge ids, number of hyperedges) for industry, area and qualify groups."""
    pos = {c: i for i, c in enumerate(split["ids"])}
    out = []
    for kind in ("industry", "area", "qualify"):
        nodes, edges = [], []
        for edge_id, (_, members) in enumerate(sorted(split["hyper"][kind].items(), key=lambda kv: str(kv[0]))):
            for m in members:
                if int(m) in pos:
                    nodes.append(pos[int(m)]); edges.append(edge_id)
        out.append((torch.tensor(nodes, dtype=torch.long), torch.tensor(edges, dtype=torch.long),
                    len(split["hyper"][kind])))
    return out


def lawsuit_tensors(split: dict, max_len: int = 48) -> dict[str, torch.Tensor]:
    """Per-lawsuit records for every company, padded to max_len (most recent kept)."""
    n = len(split["ids"])
    cause = torch.zeros(n, max_len, dtype=torch.long); court = torch.zeros_like(cause); result = torch.zeros_like(cause)
    months = torch.zeros(n, max_len); recent = torch.zeros(n, max_len); mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, cid in enumerate(split["ids"]):
        records = sorted(split["lawsuits"].get(cid, []), key=lambda r: float(r[3]))[:max_len]   # smallest months = most recent
        for j, (c, k, r, m, flag) in enumerate(records):
            cause[i, j], court[i, j], result[i, j] = int(c), int(k), int(r)
            months[i, j], recent[i, j], mask[i, j] = float(m) / 12.0, float(flag == 1), True
    return {"cause": cause, "court": court, "result": result, "years": months, "recent": recent, "mask": mask}
