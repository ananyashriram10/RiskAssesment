"""Sector-conditioned node/edge gate followed by a typed message-passing GNN."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import GraphSnapshot


@dataclass(frozen=True)
class GateConfig:
    hidden_dim: int = 32
    max_context_nodes: int = 64
    max_learned_neighbors: int = 5
    max_incoming_edges: int = 12
    candidate_chunk_size: int = 256
    message_passing_layers: int = 2
    aggregation: str = "mean"
    documented_context_quota: int = 0
    edge_mode: str = "combined"
    gate_mode: str = "learned"
    candidate_mode: str = "encoder"
    gate_seed: int = 42
    context_mode: str = "node_first"
    # Documented relations (shared director/owner) are undirected: reuse one
    # type ID for both message directions instead of encoding CIK order.
    symmetric_documented: bool = False
    # Divide received messages by the true summed edge weight (a real
    # weighted mean) rather than max(1, summed weight).
    true_mean: bool = False
    # Add an explicit per-relation edge-count channel to each message-passing update, so the
    # model can use degree information that a (weighted) mean discards.
    degree_aware: bool = False
    # Context budget grows with the sector: max(max_context_nodes, ceil(context_per_target * targets)).
    context_per_target: float = 0.0
    # Always keep every outside node with a documented edge to the target sector (node-first
    # mode); the learned gate then picks up to ``max_context_nodes`` additional context nodes.
    adjacent_context: bool = False
    # Also keep outside companies that share a kept person with the target sector (two hops).
    two_hop_context: bool = False
    # Dropout on the encoder, after each message-passing layer, and in the head.
    dropout: float = 0.0
    # Group-level (hyperedge) message passing over snapshot.extras["hyper"].
    hyper_layer: bool = False
    # First layer: person nodes summarize their companies with sector-conditioned in-sector and
    # out-of-sector weights (learned from the sector query); needs >= 2 layers and extras["is_person"].
    sector_blend: bool = False
    # L1 weight on the mean out-of-sector gate, so cross-industry contagion turns on only if it helps.
    blend_l1: float = 0.0
    # Round 4. Piecewise-linear (quantile-bin) encoding of every numeric input before the encoder;
    # needs ``bin_edges`` at construction (see ``quantile_edges``).
    numeric_embedding: str = "none"
    embedding_bins: int = 8
    embedding_dim: int = 4
    # Message passing also sees a max aggregate and a degree-amplified mean (PNA-style), not only the mean.
    multi_aggregate: bool = False
    # Edge-count channel goes through a small MLP on [log1p(counts), counts > 0] instead of one linear map.
    degree_mlp: bool = False
    # The risk head sees every message-passing layer's output, not only the last.
    jumping_knowledge: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.numeric_embedding not in {"none", "ple"}:
            raise ValueError("numeric_embedding must be none or ple")
        if self.embedding_bins < 2 or self.embedding_dim < 1:
            raise ValueError("embedding_bins must be >= 2 and embedding_dim >= 1")
        if self.degree_mlp and not self.degree_aware:
            raise ValueError("degree_mlp needs degree_aware")
        if self.aggregation != "mean" and (self.multi_aggregate or self.degree_mlp):
            raise ValueError("multi_aggregate and degree_mlp apply to the mean aggregation only")
        if self.sector_blend and self.message_passing_layers < 2:
            raise ValueError("sector_blend needs at least two message-passing layers")
        if self.context_per_target < 0:
            raise ValueError("context_per_target must be nonnegative")
        if self.hidden_dim < 2 or self.max_context_nodes < 0:
            raise ValueError("invalid hidden_dim or max_context_nodes")
        if self.max_learned_neighbors < 0 or self.max_incoming_edges < 0:
            raise ValueError("edge budgets must be nonnegative")
        if self.candidate_chunk_size < 1 or self.message_passing_layers < 1:
            raise ValueError("chunk size and layer count must be positive")
        if self.aggregation not in {"mean", "relation_attention"}:
            raise ValueError("aggregation must be mean or relation_attention")
        if not 0 <= self.documented_context_quota <= self.max_context_nodes:
            raise ValueError("documented context quota must fit the context budget")
        if self.edge_mode not in {"combined", "documented", "inferred", "random", "none"}:
            raise ValueError("unknown edge mode")
        if self.gate_mode not in {"learned", "fixed", "random"}:
            raise ValueError("unknown gate mode")
        if self.candidate_mode not in {"encoder", "raw"}:
            raise ValueError("unknown candidate mode")
        if self.context_mode not in {"node_first", "edge_first"}:
            raise ValueError("unknown context mode")
        if self.context_mode == "edge_first" and (
            self.edge_mode != "combined" or self.candidate_mode != "raw" or
            self.max_incoming_edges < 1 or self.max_learned_neighbors < 1 or
            self.documented_context_quota != 0
        ):
            raise ValueError("edge-first context requires combined raw candidates, positive edge budgets, and no node quota")


@dataclass
class ModelOutput:
    logits: torch.Tensor  # one bankruptcy logit per target-sector company
    target_indices: torch.Tensor
    selected_indices: torch.Tensor
    edge_index: torch.Tensor  # selected, directed message edges in global indices
    edge_type: torch.Tensor  # 0..R-1 forward, R..2R-1 reverse, 2R inferred
    edge_weight: torch.Tensor
    node_score: torch.Tensor  # selection scores; learned gate logits in learned mode
    edge_score: torch.Tensor  # selected-edge scores; learned gate logits in learned mode


def quantile_edges(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Per-column quantile bin edges, shape (columns, bins + 1), nondecreasing. Fit on training rows only."""
    levels = torch.linspace(0.0, 1.0, bins + 1, dtype=x.dtype)
    return torch.quantile(x, levels, dim=0).T.contiguous()


class PiecewiseLinearEmbedding(nn.Module):
    """Piecewise-linear encoding of each numeric column, then a per-column linear map and ReLU.

    A value is described by how far it has climbed through each training-quantile bin (0 below the
    bin, 1 above it), so the encoding saturates outside the training range and a monotone shift in
    the inputs cannot push the network into an extrapolation regime. Output: (rows, columns * dim)."""

    def __init__(self, edges: torch.Tensor, dim: int):
        super().__init__()
        self.register_buffer("lower", edges[:, :-1].clone())
        self.register_buffer("width", (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-6))
        columns, bins = self.lower.shape
        self.weight = nn.Parameter(torch.randn(columns, bins, dim) / math.sqrt(bins))
        self.bias = nn.Parameter(torch.zeros(columns, dim))
        self.out_dim = columns * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = ((x.unsqueeze(-1) - self.lower) / self.width).clamp(0.0, 1.0)
        return F.relu(torch.einsum("nfb,fbd->nfd", steps, self.weight) + self.bias).flatten(1)


class RelationalLayer(nn.Module):
    def __init__(self, hidden_dim: int, number_edge_types: int, true_mean: bool = False,
                 degree_aware: bool = False, multi_aggregate: bool = False, degree_mlp: bool = False):
        super().__init__()
        self.true_mean = true_mean
        self.number_edge_types = number_edge_types
        self.degree_net = (nn.Sequential(nn.Linear(2 * number_edge_types, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim))
                           if degree_aware and degree_mlp else None)
        self.degree_linear = (nn.Linear(number_edge_types, hidden_dim, bias=False)
                              if degree_aware and not degree_mlp else None)
        self.extra_linear = nn.Linear(2 * hidden_dim, hidden_dim, bias=False) if multi_aggregate else None
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_scale = nn.Embedding(number_edge_types, hidden_dim)
        nn.init.ones_(self.relation_scale.weight)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, weight: torch.Tensor,
        blend=None,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return F.relu(self.self_linear(h))
        src, dst = edge_index
        messages = self.message_linear(h[src]) * self.relation_scale(edge_type)
        extra = None
        if self.extra_linear is not None:
            # Max of the weighted messages, and the plain weighted mean scaled by log(1 + summed weight):
            # the amplified mean carries the neighbour count that a mean alone discards.
            scaled = messages * weight.unsqueeze(-1)
            summed_weight = h.new_zeros(h.shape[0]).index_add(0, dst, weight)
            plain_mean = torch.zeros_like(h).index_add(0, dst, scaled) / summed_weight.clamp_min(1e-6).unsqueeze(-1)
            peak = h.new_full(h.shape, -float("inf")).scatter_reduce(
                0, dst.unsqueeze(-1).expand(-1, h.shape[1]), scaled, reduce="amax", include_self=True)
            peak = peak.masked_fill(summed_weight.unsqueeze(-1) <= 0, 0.0)
            extra = self.extra_linear(torch.cat((plain_mean * torch.log1p(summed_weight).unsqueeze(-1), peak), dim=1))
        if blend is None:
            weighted = messages * weight.unsqueeze(-1)
            received = torch.zeros_like(h).index_add(0, dst, weighted)
            total_weight = h.new_zeros(h.shape[0]).index_add(0, dst, weight).clamp_min(
                1e-6 if self.true_mean else 1.0)
            update = self.self_linear(h) + received / total_weight.unsqueeze(-1)
        else:
            # Person nodes receive a learned blend of their in-sector and out-of-sector companies;
            # every other node keeps the ordinary weighted mean.
            in_sector, is_person, gate_in, gate_out = blend
            relation = edge_type.clamp_max(gate_in.shape[0] - 1)
            to_person = is_person[dst]
            from_in = in_sector[src]
            eps = 1e-6
            w_std = weight * (~to_person)
            w_in = weight * (to_person & from_in)
            w_out = weight * (to_person & ~from_in)

            def mean_of(w, scale=None):
                scaled = w if scale is None else w * scale
                total = torch.zeros_like(h).index_add(0, dst, messages * scaled.unsqueeze(-1))
                norm = h.new_zeros(h.shape[0]).index_add(0, dst, w).clamp_min(eps)
                return total / norm.unsqueeze(-1)

            received = (mean_of(w_std) + mean_of(w_in, gate_in[relation]) + mean_of(w_out, gate_out[relation]))
            update = self.self_linear(h) + received
        if extra is not None:
            update = update + extra
        if self.degree_linear is not None or self.degree_net is not None:
            counts = h.new_zeros((h.shape[0], self.number_edge_types))
            counts = counts.index_put((dst, edge_type), torch.ones_like(weight), accumulate=True)
            if self.degree_net is not None:
                update = update + self.degree_net(torch.cat((torch.log1p(counts), (counts > 0).to(h.dtype)), dim=1))
            else:
                update = update + self.degree_linear(torch.log1p(counts))
        return F.relu(update)


class RelationalAttentionLayer(nn.Module):
    """Typed two-head attention over the gate's selected incoming messages."""

    def __init__(self, hidden_dim: int, number_edge_types: int, heads: int = 2):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(number_edge_types, hidden_dim)
        self.relation_value = nn.Embedding(number_edge_types, hidden_dim)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return F.relu(self.self_linear(h))
        src, dst = edge_index
        n, e = h.shape[0], src.numel()
        shape = (e, self.heads, self.head_dim)
        query = self.query(h[dst]).reshape(shape)
        key = self.key(h[src]).reshape(shape) + self.relation_key(edge_type).reshape(shape)
        score = F.leaky_relu((query * key).sum(-1) / self.head_dim**0.5, negative_slope=0.2)
        dst_heads = dst[:, None].expand(-1, self.heads)
        maximum = score.new_full((n, self.heads), -float("inf"))
        maximum.scatter_reduce_(0, dst_heads, score.detach(), reduce="amax", include_self=True)
        exp_score = (score - maximum[dst]).exp()
        denominator = score.new_zeros((n, self.heads)).index_add_(0, dst, exp_score)
        attention = exp_score / denominator[dst].clamp_min(1e-12)
        value = self.value(h[src]).reshape(shape) + self.relation_value(edge_type).reshape(shape)
        messages = value * attention.unsqueeze(-1) * weight[:, None, None]
        received = h.new_zeros((n, self.heads, self.head_dim)).index_add_(0, dst, messages)
        return F.relu(self.self_linear(h) + self.output_linear(received.reshape(n, -1)))


class HyperLayer(nn.Module):
    """Group message passing: node -> hyperedge (mean) -> node (mean), one transform per group type.

    Only nodes kept by the gate contribute, so the groups are summarized over the selected graph."""

    def __init__(self, hidden_dim: int, group_types: int = 3):
        super().__init__()
        self.edge_linear = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(group_types))
        self.node_linear = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(group_types))

    def forward(self, h: torch.Tensor, incidences, selected_mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h)
        for t, (node, edge, n_edges) in enumerate(incidences):
            node, edge = node.to(h.device), edge.to(h.device)
            keep = selected_mask[node]
            node, edge = node[keep], edge[keep]
            if node.numel() == 0:
                continue
            ones = h.new_ones(node.numel())
            sums = h.new_zeros((n_edges, h.shape[1])).index_add(0, edge, self.edge_linear[t](h[node]))
            counts = h.new_zeros(n_edges).index_add(0, edge, ones).clamp_min(1.0)
            hyper = F.relu(sums / counts.unsqueeze(-1))
            back = h.new_zeros(h.shape).index_add(0, node, hyper[edge])
            degree = h.new_zeros(h.shape[0]).index_add(0, node, ones).clamp_min(1.0)
            out = out + self.node_linear[t](back / degree.unsqueeze(-1))
        return h + out


class SectorRiskGNN(nn.Module):
    """Select context companies and links for a sector, then score its firms.

    Hard top-k decisions bound graph size. The selected node and edge scores
    remain in the message weights, so supervised loss can train both scorers.
    This is a practical gate, not a fully differentiable top-k operator.
    """

    def __init__(
        self,
        feature_dim: int,
        number_sectors: int,
        number_documented_relations: int,
        config: GateConfig = GateConfig(),
        bin_edges: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or number_sectors < 1 or number_documented_relations < 0:
            raise ValueError("invalid feature or vocabulary size")
        self.config = config
        self.feature_dim = feature_dim
        self.number_sectors = number_sectors
        self.number_documented_relations = number_documented_relations
        h = config.hidden_dim
        self.embedding = None
        encoder_in = feature_dim
        if config.numeric_embedding == "ple":
            if bin_edges is None or bin_edges.shape[0] > feature_dim:
                raise ValueError("numeric_embedding='ple' needs bin_edges with one row per leading feature column")
            self.embedding = PiecewiseLinearEmbedding(bin_edges, config.embedding_dim)
            encoder_in = feature_dim + self.embedding.out_dim   # raw columns stay available next to the encoding
        self.company_encoder = nn.Sequential(nn.Linear(encoder_in, h), nn.ReLU(),
                                             nn.Dropout(config.dropout), nn.Linear(h, h))
        self.dropout = nn.Dropout(config.dropout)
        self.sector_embedding = nn.Embedding(number_sectors, h)
        self.node_gate = nn.Sequential(nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))
        self.type_embedding = nn.Embedding(2 * number_documented_relations + 1, h)
        self.edge_gate = nn.Sequential(nn.Linear(h * 5, h), nn.ReLU(), nn.Linear(h, 1))
        types = 2 * number_documented_relations + 1
        if config.aggregation == "mean":
            self.layers = nn.ModuleList(
                RelationalLayer(h, types, config.true_mean, config.degree_aware,
                                config.multi_aggregate, config.degree_mlp)
                for _ in range(config.message_passing_layers))
        else:
            self.layers = nn.ModuleList(
                RelationalAttentionLayer(h, types)
                for _ in range(config.message_passing_layers))
        self._reverse_offset = 0 if config.symmetric_documented else number_documented_relations
        head_parts = (config.message_passing_layers if config.jumping_knowledge else 1) + 2
        self.head = nn.Sequential(nn.Linear(h * head_parts, h), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(h, 1))
        self.hyper = HyperLayer(h) if config.hyper_layer else None
        self.blend_gate = None
        if config.sector_blend:
            r = number_documented_relations
            self.blend_gate = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * r))
            with torch.no_grad():  # start with in-sector open (~0.88) and out-of-sector mostly closed (~0.12)
                self.blend_gate[-1].bias[:r] = 2.0
                self.blend_gate[-1].bias[r:] = -2.0
        self.aux_loss: torch.Tensor | None = None
        self.last_gates: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cross_candidate_cache: dict[tuple[int, int, str], tuple] = {}

    def _cosine_candidates(self, h: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Exact cosine shortlist, computed in chunks to bound peak memory."""
        k = min(self.config.max_learned_neighbors, selected.numel() - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=h.device)
        with torch.no_grad():
            z = F.normalize(h[selected].detach(), dim=1)
            sources: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for start in range(0, len(selected), self.config.candidate_chunk_size):
                end = min(start + self.config.candidate_chunk_size, len(selected))
                similarity = z[start:end] @ z.T
                rows = torch.arange(end - start, device=h.device)
                similarity[rows, rows + start] = -float("inf")
                neighbors = similarity.topk(k, dim=1).indices
                sources.append(selected[neighbors.reshape(-1)])
                targets.append(selected[torch.arange(start, end, device=h.device).repeat_interleave(k)])
            return torch.stack((torch.cat(sources), torch.cat(targets)))

    def _gate_random(self, count: int, snapshot: GraphSnapshot, sector: int,
                     offset: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.gate_seed + snapshot.as_of.toordinal() * 1009
                              + sector * 1000003 + offset)
        return torch.rand(count, generator=generator, device=device)

    def _edge_first_candidates(
        self, snapshot: GraphSnapshot, features: torch.Tensor, target: torch.Tensor,
        external: torch.Tensor, sector: int, documented: torch.Tensor,
        doc_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Point-in-time external -> target candidate messages, before node selection."""
        key = (id(snapshot), sector, str(features.device))
        cached = self._cross_candidate_cache.get(key)
        if cached is not None:
            return cached[0], cached[1]
        device = features.device
        inferred_type = 2 * self.number_documented_relations
        k = min(self.config.max_context_nodes, external.numel())
        with torch.no_grad():
            if k:
                source_vectors = F.normalize(features[external], dim=1)
                target_vectors = F.normalize(features[target], dim=1)
                raw_sources, raw_targets = [], []
                for start in range(0, len(target), self.config.candidate_chunk_size):
                    stop = min(start + self.config.candidate_chunk_size, len(target))
                    nearest = (target_vectors[start:stop] @ source_vectors.T).topk(k, dim=1).indices
                    raw_sources.append(external[nearest.reshape(-1)])
                    raw_targets.append(target[torch.arange(start, stop, device=device).repeat_interleave(k)])
                inferred = torch.stack((torch.cat(raw_sources), torch.cat(raw_targets)))
            else:
                inferred = documented[:, :0]
            sectors = snapshot.sectors.to(device)
            src, dst = documented
            forward = (sectors[src] != sector) & (sectors[dst] == sector)
            reverse = (sectors[dst] != sector) & (sectors[src] == sector)
            doc_forward = documented[:, forward]
            doc_reverse = documented[:, reverse].flip(0)
            candidate_edges = torch.cat((doc_forward, doc_reverse, inferred), dim=1)
            candidate_types = torch.cat((
                doc_types[forward],
                doc_types[reverse] + self._reverse_offset,
                torch.full((inferred.shape[1],), inferred_type, dtype=torch.long, device=device),
            ))
        # Keep the snapshot alive so its id() cannot be reused by another object.
        self._cross_candidate_cache[key] = candidate_edges, candidate_types, snapshot
        return candidate_edges, candidate_types

    def _edge_first_context(
        self, snapshot: GraphSnapshot, features: torch.Tensor, h: torch.Tensor,
        q: torch.Tensor, node_score: torch.Tensor, target: torch.Tensor,
        external: torch.Tensor, sector: int, documented: torch.Tensor,
        doc_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates, types = self._edge_first_candidates(
            snapshot, features, target, external, sector, documented, doc_types)
        count = min(self.config.max_context_nodes, external.numel(),
                    self.config.max_incoming_edges * target.numel())
        if count == 0:
            return external[:0], candidates[:, :0], types[:0]
        src, dst = candidates
        inferred_type = 2 * self.number_documented_relations
        with torch.no_grad():
            if self.config.gate_mode == "learned":
                pair_score = self.edge_gate(torch.cat((
                    h[src], h[dst], q[src], h[src] * h[dst], self.type_embedding(types)
                ), dim=1)).squeeze(-1)
                rank = node_score[src] + pair_score
            elif self.config.gate_mode == "fixed":
                similarity = (F.normalize(features[src], dim=1) *
                              F.normalize(features[dst], dim=1)).sum(1)
                rank = node_score[src] + similarity + (types != inferred_type).to(similarity.dtype) * 0.25
            else:
                rank = self._gate_random(len(src), snapshot, sector, 2, features.device)
            order = torch.argsort(rank, descending=True, stable=True).tolist()
            chosen_sources: set[int] = set()
            target_capacity: dict[int, int] = {}
            assignments: list[int] = []
            for index in order:
                source, destination = int(src[index]), int(dst[index])
                if source in chosen_sources or target_capacity.get(destination, 0) >= self.config.max_incoming_edges:
                    continue
                chosen_sources.add(source)
                target_capacity[destination] = target_capacity.get(destination, 0) + 1
                assignments.append(index)
                if len(assignments) == count:
                    break
            if len(assignments) != count:
                raise AssertionError("edge-first candidate pool could not fill the context budget")
            indices = torch.tensor(assignments, dtype=torch.long, device=features.device)
            return src[indices], candidates[:, indices], types[indices]

    def forward(self, snapshot: GraphSnapshot, target_sector: str | int) -> ModelOutput:
        if snapshot.features.shape[1] != self.feature_dim:
            raise ValueError("snapshot feature count differs from model")
        if len(snapshot.sector_names) != self.number_sectors:
            raise ValueError("snapshot sector vocabulary differs from model")
        if len(snapshot.relation_names) != self.number_documented_relations:
            raise ValueError("snapshot relationship vocabulary differs from model")
        if isinstance(target_sector, str):
            try:
                target_sector = snapshot.sector_names.index(target_sector)
            except ValueError as exc:
                raise ValueError(f"unknown target sector: {target_sector}") from exc
        if target_sector < 0 or target_sector >= self.number_sectors:
            raise ValueError("target sector ID outside vocabulary")

        device = self.sector_embedding.weight.device
        features = snapshot.features.to(device)
        sectors = snapshot.sectors.to(device)
        target = torch.where(sectors == target_sector)[0]
        if target.numel() == 0:
            raise ValueError("snapshot has no companies in target sector")
        extras = snapshot.extras if isinstance(snapshot.extras, dict) else {}
        person_mask = (extras["is_person"].to(device) if "is_person" in extras
                       else torch.zeros(features.shape[0], dtype=torch.bool, device=device))
        encoded = features
        if self.embedding is not None:
            leading = self.embedding.lower.shape[0]
            encoded = torch.cat((features, self.embedding(features[:, :leading])), dim=1)
        h = self.company_encoder(encoded) + self.sector_embedding(sectors)
        query = torch.tanh(self.sector_embedding.weight[target_sector] + h[target].mean(0))
        q = query.expand(h.shape[0], -1)
        documented = snapshot.documented_edges.to(device)
        doc_types = snapshot.documented_types.to(device)
        target_mask = sectors == target_sector
        # Encode disclosed direct links to the target sector before ranking
        # outside firms; otherwise a hard top-k gate can discard these links.
        evidence = torch.zeros_like(h)
        linked_external = torch.zeros(h.shape[0], dtype=torch.bool, device=device)
        if documented.shape[1] and self.config.edge_mode not in {"inferred", "none"}:
            src, dst = documented
            cross_forward = target_mask[src] & ~target_mask[dst]
            cross_reverse = target_mask[dst] & ~target_mask[src]
            if cross_forward.any():
                linked_external[dst[cross_forward]] = True
                evidence = evidence.index_add(
                    0, dst[cross_forward], self.type_embedding(doc_types[cross_forward])
                )
            if cross_reverse.any():
                linked_external[src[cross_reverse]] = True
                evidence = evidence.index_add(
                    0, src[cross_reverse], self.type_embedding(doc_types[cross_reverse])
                )
        if self.config.gate_mode == "learned":
            node_score = self.node_gate(torch.cat((h, q, h * q, evidence), dim=1)).squeeze(-1)
        elif self.config.gate_mode == "fixed":
            # Raw, train-split-scaled ratios and missingness; no fitted gate or outcome labels.
            reference = F.normalize(features[target].mean(0, keepdim=True), dim=1)
            node_score = (F.normalize(features, dim=1) @ reference.T).squeeze(-1)
        else:
            node_score = self._gate_random(h.shape[0], snapshot, target_sector, 0, device)
        external = torch.where(sectors != target_sector)[0]
        assigned_edges = documented[:, :0]
        assigned_types = doc_types[:0]
        if self.config.context_mode == "edge_first":
            context, assigned_edges, assigned_types = self._edge_first_context(
                snapshot, features, h, q, node_score, target, external,
                target_sector, documented, doc_types)
            count = context.numel()
        else:
            budget = max(self.config.max_context_nodes,
                         int(math.ceil(self.config.context_per_target * target.numel())))
            count = min(budget, external.numel())
        if self.config.context_mode == "node_first" and self.config.adjacent_context:
            is_linked = linked_external[external]
            linked, rest = external[is_linked], external[~is_linked]
            two_hop = external[:0]
            if self.config.two_hop_context and linked.numel():
                person_flag = person_mask
                persons = linked[person_flag[linked]]
                if persons.numel():
                    d_src, d_dst = documented
                    other = torch.cat((d_src[torch.isin(d_dst, persons)], d_dst[torch.isin(d_src, persons)]))
                    other = other[(~person_flag[other]) & (sectors[other] != target_sector)]
                    two_hop = torch.unique(other)
                    two_hop = two_hop[~torch.isin(two_hop, linked)]
                    rest = rest[~torch.isin(rest, two_hop)]
            take = min(self.config.max_context_nodes, rest.numel())
            extra = rest[node_score[rest].topk(take).indices] if take else rest[:0]
            context = torch.cat((linked, two_hop, extra))
            count = context.numel()
        elif self.config.context_mode == "node_first" and count and self.config.documented_context_quota:
            linked = external[linked_external[external]]
            reserve = min(self.config.documented_context_quota, count, linked.numel())
            reserved = linked[node_score[linked].topk(reserve).indices] if reserve else linked[:0]
            remaining = external[~torch.isin(external, reserved)]
            fill = remaining[node_score[remaining].topk(count - reserve).indices]
            context = torch.cat((reserved, fill))
        elif self.config.context_mode == "node_first":
            context = external[node_score[external].topk(count).indices] if count else external[:0]
        selected = torch.cat((target, context))
        selected_mask = torch.zeros(h.shape[0], dtype=torch.bool, device=device)
        selected_mask[selected] = True
        node_weight = torch.zeros_like(node_score).scatter(0, target, 1.0)
        if count:
            context_weight = (torch.sigmoid(node_score[context])
                              if self.config.gate_mode == "learned"
                              else torch.full((count,), 0.5, device=device))
            node_weight = node_weight.scatter(0, context, context_weight)

        keep = selected_mask[documented[0]] & selected_mask[documented[1]]
        documented, doc_types = documented[:, keep], doc_types[keep]
        if self.config.edge_mode in {"inferred", "none"}:
            documented, doc_types = documented[:, :0], doc_types[:0]
        reverse = documented.flip(0)
        candidate_features = features if self.config.candidate_mode == "raw" else h
        inferred = (self._cosine_candidates(candidate_features, selected)
                    if self.config.edge_mode in {"combined", "inferred", "random"}
                    else documented[:, :0])
        edge_index = torch.cat((documented, reverse, inferred), dim=1)
        inferred_type = 2 * self.number_documented_relations
        edge_type = torch.cat(
            (
                doc_types,
                doc_types + self._reverse_offset,
                torch.full((inferred.shape[1],), inferred_type, dtype=torch.long, device=device),
            )
        )
        reserved_indices = torch.empty(0, dtype=torch.long, device=device)
        if self.config.context_mode == "edge_first" and assigned_edges.shape[1]:
            edge_index = torch.cat((edge_index, assigned_edges), dim=1)
            edge_type = torch.cat((edge_type, assigned_types))
            triples = torch.cat((edge_index.T, edge_type[:, None]), dim=1)
            unique_triples, inverse = torch.unique(triples, dim=0, return_inverse=True)
            reserved_indices = inverse[-assigned_edges.shape[1]:]
            edge_index = unique_triples[:, :2].T.contiguous()
            edge_type = unique_triples[:, 2]

        if edge_index.shape[1]:
            src, dst = edge_index
            if self.config.gate_mode == "learned":
                pair_score = self.edge_gate(
                    torch.cat((h[src], h[dst], q[src], h[src] * h[dst], self.type_embedding(edge_type)), dim=1)
                ).squeeze(-1)
            elif self.config.gate_mode == "fixed":
                pair_score = (F.normalize(features[src], dim=1) *
                              F.normalize(features[dst], dim=1)).sum(1)
                pair_score = pair_score + (edge_type != inferred_type).to(pair_score.dtype) * 0.25
            else:
                pair_score = self._gate_random(src.numel(), snapshot, target_sector, 1, device)
            chosen = []
            for destination in selected.tolist():
                incoming = torch.where(dst == destination)[0]
                limit = min(self.config.max_incoming_edges, incoming.numel())
                if limit:
                    if self.config.context_mode == "edge_first":
                        reserved = reserved_indices[dst[reserved_indices] == destination]
                        available = incoming[~torch.isin(incoming, reserved)]
                        fill = min(limit - reserved.numel(), available.numel())
                        ranked = available[pair_score[available].topk(fill).indices] if fill else available[:0]
                        chosen.append(torch.cat((reserved, ranked)))
                    else:
                        chosen.append(incoming[pair_score[incoming].topk(limit).indices])
            chosen_indices = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=device)
            edge_index = edge_index[:, chosen_indices]
            edge_type = edge_type[chosen_indices]
            if self.config.edge_mode == "random" and selected.numel() > 1:
                # Fixed graph null: preserve each destination and edge type, but
                # replace the source. The original gate still chooses the slots.
                generator = torch.Generator(device=device)
                generator.manual_seed(17171 + int(target_sector))
                dst = edge_index[1]
                choices = torch.randint(selected.numel() - 1, (dst.numel(),),
                                        generator=generator, device=device)
                destination_position = torch.searchsorted(selected.sort().values, dst)
                # Map sampled indices through a sorted selected list, skipping self.
                sorted_selected = selected.sort().values
                choices = choices + (choices >= destination_position).long()
                edge_index = torch.stack((sorted_selected[choices], dst))
                src, dst = edge_index
                edge_score = self.edge_gate(torch.cat(
                    (h[src], h[dst], q[src], h[src] * h[dst], self.type_embedding(edge_type)),
                    dim=1)).squeeze(-1)
            else:
                edge_score = pair_score[chosen_indices]
            src, dst = edge_index
            gate_weight = (torch.sigmoid(edge_score) if self.config.gate_mode == "learned"
                           else torch.full_like(edge_score, 0.5))
            edge_weight = gate_weight * node_weight[src] * node_weight[dst]
        else:
            edge_score = h.new_empty(0)
            edge_weight = h.new_empty(0)

        updated = h
        blend = None
        self.aux_loss, self.last_gates = None, None
        if self.blend_gate is not None:
            r = self.number_documented_relations
            gates = torch.sigmoid(self.blend_gate(query))
            gate_in, gate_out = gates[:r], gates[r:]
            blend = (sectors == target_sector, person_mask, gate_in, gate_out)
            self.aux_loss = self.config.blend_l1 * gate_out.mean()
            self.last_gates = (gate_in.detach(), gate_out.detach())
        layer_outputs = []
        for index, layer in enumerate(self.layers):
            if isinstance(layer, RelationalLayer):
                updated = layer(updated, edge_index, edge_type, edge_weight, blend if index == 0 else None)
            else:
                updated = layer(updated, edge_index, edge_type, edge_weight)
            updated = self.dropout(updated)
            layer_outputs.append(updated)
        if self.hyper is not None and "hyper" in extras:
            updated = self.hyper(updated, extras["hyper"], selected_mask)
            layer_outputs[-1] = updated
        target_query = query.expand(target.numel(), -1)
        summaries = layer_outputs if self.config.jumping_knowledge else [updated]
        logits = self.head(torch.cat([s[target] for s in summaries] + [h[target], target_query], dim=1)).squeeze(-1)
        return ModelOutput(
            logits=logits,
            target_indices=target,
            selected_indices=selected,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            node_score=node_score,
            edge_score=edge_score,
        )
