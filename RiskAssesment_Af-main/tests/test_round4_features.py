"""Tests for the round-4 architecture options (no dataset needed)."""

from datetime import date

import pytest
import torch

from benchmark.model_ext import LawsuitEncoder, SMEModel
from benchmark.run4 import VARIANTS, pairwise_rank_loss
from risk_gnn import GateConfig, GraphSnapshot, SectorRiskGNN, quantile_edges
from risk_gnn.model import PiecewiseLinearEmbedding


def toy_snapshot() -> GraphSnapshot:
    """Companies a,b (sector 0), c,d (sector 1); persons p,q (sector 2). p links a,b,c; q links d."""
    return GraphSnapshot(
        as_of=date(2021, 12, 31), company_ids=tuple("abcdpq"), features=torch.randn(6, 3),
        sectors=torch.tensor([0, 0, 1, 1, 2, 2]),
        documented_edges=torch.tensor([[0, 1, 2, 3], [4, 4, 4, 5]]), documented_types=torch.tensor([0, 0, 1, 1]),
        sector_names=("x", "y", "person"), relation_names=("r0", "r1"),
        extras={"is_person": torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)})


def model(edges=None, **cfg) -> SectorRiskGNN:
    base = dict(max_context_nodes=1, max_learned_neighbors=0, max_incoming_edges=8, edge_mode="documented",
                symmetric_documented=True, true_mean=True, adjacent_context=True, degree_aware=True)
    base.update(cfg)
    torch.manual_seed(0)
    return SectorRiskGNN(3, 3, 2, GateConfig(**base), edges)


def test_quantile_edges_are_nondecreasing_per_column():
    x = torch.randn(200, 4)
    x[:, 3] = 0.0                                       # constant column
    edges = quantile_edges(x, 8)
    assert edges.shape == (4, 9) and (edges[:, 1:] >= edges[:, :-1]).all()


def test_piecewise_embedding_saturates_outside_training_range():
    x = torch.linspace(0, 1, 50).unsqueeze(1)
    layer = PiecewiseLinearEmbedding(quantile_edges(x, 4), 3)
    inside, high, far = layer(torch.tensor([[1.0]])), layer(torch.tensor([[2.0]])), layer(torch.tensor([[1000.0]]))
    assert torch.allclose(inside, high) and torch.allclose(inside, far)       # clamped beyond the top edge
    assert layer(torch.tensor([[-5.0]])).shape == (1, 3)


@pytest.mark.parametrize("cfg", [
    dict(multi_aggregate=True), dict(degree_mlp=True), dict(jumping_knowledge=True, message_passing_layers=2),
    dict(multi_aggregate=True, degree_mlp=True, jumping_knowledge=True, message_passing_layers=2, hyper_layer=False),
])
def test_new_layer_options_run_and_train(cfg):
    m = model(**cfg)
    out = m(toy_snapshot(), 0)
    assert out.logits.shape == (2,) and torch.isfinite(out.logits).all()
    out.logits.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_numeric_embedding_runs_and_is_trained():
    snap = toy_snapshot()
    m = model(edges=quantile_edges(snap.features, 4), numeric_embedding="ple", embedding_bins=4, embedding_dim=2)
    m(snap, 0).logits.sum().backward()
    assert m.embedding.weight.grad.abs().sum() > 0


def test_numeric_embedding_requires_edges_and_valid_options():
    with pytest.raises(ValueError, match="bin_edges"):
        model(numeric_embedding="ple")
    with pytest.raises(ValueError, match="degree_aware"):
        GateConfig(degree_mlp=True)
    with pytest.raises(ValueError, match="mean aggregation"):
        GateConfig(aggregation="relation_attention", multi_aggregate=True)


def test_jumping_knowledge_head_sees_every_layer():
    assert model(message_passing_layers=3, jumping_knowledge=True).head[0].in_features == 32 * 5
    assert model(message_passing_layers=3).head[0].in_features == 32 * 3


def test_max_aggregate_ignores_nodes_without_incoming_edges():
    m = model(multi_aggregate=True)
    out = m(toy_snapshot(), 0)                          # context picks few nodes; isolated nodes must stay finite
    assert torch.isfinite(out.logits).all()


def test_timing_free_lawsuit_encoder_ignores_age_and_recent_flags():
    seq = {"cause": torch.tensor([[0, 3]]), "court": torch.zeros(1, 2, dtype=torch.long),
           "result": torch.zeros(1, 2, dtype=torch.long), "years": torch.tensor([[0.5, 2.0]]),
           "recent": torch.tensor([[1.0, 0.0]]), "mask": torch.tensor([[True, True]])}
    other = dict(seq, years=torch.tensor([[7.0, 0.1]]), recent=torch.tensor([[0.0, 1.0]]))
    torch.manual_seed(0)
    free = LawsuitEncoder(8, use_timing=False)
    assert torch.equal(free(seq), free(other))
    torch.manual_seed(0)
    timed = LawsuitEncoder(8, use_timing=True)
    assert not torch.equal(timed(seq), timed(other))


def test_sme_model_passes_edges_and_timing_flag():
    snap = toy_snapshot()
    config = GateConfig(max_context_nodes=1, max_learned_neighbors=0, max_incoming_edges=8, edge_mode="documented",
                        numeric_embedding="ple", embedding_bins=4, embedding_dim=2)
    wrapped = SMEModel(3, 3, 2, config, lawsuit_dim=4, bin_edges=quantile_edges(snap.features, 4), lawsuit_timing=False)
    assert wrapped.encoder.use_timing is False and wrapped.gnn.embedding is not None


def test_pairwise_rank_loss_prefers_correct_order_and_handles_one_class():
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = pairwise_rank_loss(torch.tensor([2.0, 3.0, -1.0, -2.0]), target)
    bad = pairwise_rank_loss(torch.tensor([-2.0, -3.0, 1.0, 2.0]), target)
    assert good < bad
    assert float(pairwise_rank_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))) == 0.0


def test_every_round4_variant_is_timing_free():
    for name, spec in VARIANTS.items():
        assert {23, 24, 25} <= set(spec["drop"]), name
        assert spec.get("lawsuit_timing", False) is False or not spec.get("lawsuit"), name
