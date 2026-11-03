from datetime import date

import pytest
import torch

from risk_gnn import GateConfig, GraphSnapshot, SectorRiskGNN, load_snapshot_csv, selected_edge_rows
from risk_gnn.training import bankruptcy_loss


def test_csv_loader_uses_publication_dates_and_latest_known_features(tmp_path):
    companies = tmp_path / "companies.csv"
    companies.write_text(
        "company_id,sector,available_on,leverage\n"
        "A,Tech,2020-01-01,0.2\n"
        "A,Tech,2021-01-01,0.8\n"
        "B,Energy,2020-01-01,0.3\n"
        "C,Energy,2021-01-01,0.4\n"
    )
    links = tmp_path / "links.csv"
    links.write_text(
        "source,target,relation,available_on\n"
        "A,B,supplier,2020-06-01\n"
        "B,A,owner,2021-01-01\n"
        "A,C,supplier,2020-06-01\n"
    )
    snapshot = load_snapshot_csv(
        companies,
        links,
        as_of=date(2020, 12, 31),
        feature_columns=("leverage",),
        sector_vocab={"Tech": 0, "Energy": 1},
        relation_vocab={"supplier": 0, "owner": 1},
    )
    assert snapshot.company_ids == ("A", "B")
    assert snapshot.features[:, 0].tolist() == pytest.approx([0.2, 0.3])
    assert snapshot.documented_edges.tolist() == [[0], [1]]
    assert snapshot.documented_types.tolist() == [0]


def test_sector_gate_retains_targets_and_trains_both_selectors():
    torch.manual_seed(1)
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D"),
        features=torch.tensor([[0.3, 1.4], [0.8, 0.6], [0.4, 1.2], [0.5, 0.9]]),
        sectors=torch.tensor([0, 0, 1, 1]),
        documented_edges=torch.tensor([[0, 1], [2, 3]]),
        documented_types=torch.tensor([0, 1]),
        sector_names=("Tech", "Energy"),
        relation_names=("supplier", "shared_director"),
    )
    model = SectorRiskGNN(2, 2, 2, GateConfig(max_context_nodes=2, max_learned_neighbors=0))
    output = model(snapshot, "Tech")
    assert output.target_indices.tolist() == [0, 1]
    assert set(output.selected_indices.tolist()) == {0, 1, 2, 3}
    assert output.edge_index.shape[1] == 4  # two records and their reverse messages
    assert set(output.edge_type.tolist()) == {0, 1, 2, 3}
    trace = selected_edge_rows(snapshot, output)
    assert {row["kind"] for row in trace} == {"documented"}
    assert {row["direction"] for row in trace} == {"forward", "reverse_message"}
    loss = bankruptcy_loss(output, torch.tensor([0.0, 1.0]), positive_weight=2.0)
    loss.backward()
    assert model.node_gate[0].weight.grad.abs().sum() > 0
    assert model.edge_gate[0].weight.grad.abs().sum() > 0


def test_empty_relationship_graph_still_scores_every_target():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C"),
        features=torch.tensor([[0.1], [0.2], [0.3]]),
        sectors=torch.tensor([0, 0, 1]),
        documented_edges=torch.empty((2, 0), dtype=torch.long),
        documented_types=torch.empty(0, dtype=torch.long),
        sector_names=("Tech", "Energy"),
        relation_names=(),
    )
    model = SectorRiskGNN(
        1, 2, 0, GateConfig(max_context_nodes=0, max_learned_neighbors=0, max_incoming_edges=0)
    )
    output = model(snapshot, "Tech")
    assert output.logits.shape == (2,)
    assert output.selected_indices.tolist() == [0, 1]
    assert output.edge_index.shape == (2, 0)


def test_edge_ablation_modes_keep_targets_and_control_messages():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D"),
        features=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        sectors=torch.tensor([0, 0, 1, 1]),
        documented_edges=torch.tensor([[0, 1], [2, 3]]),
        documented_types=torch.tensor([0, 0]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director",),
    )
    outputs = {}
    for mode in ("combined", "documented", "inferred", "random", "none"):
        torch.manual_seed(8)
        config = GateConfig(max_context_nodes=2, max_learned_neighbors=1,
                            max_incoming_edges=4, edge_mode=mode)
        model = SectorRiskGNN(1, 2, 1, config)
        outputs[mode] = model(snapshot, "Tech")
        assert outputs[mode].target_indices.tolist() == [0, 1]
    assert outputs["documented"].edge_type.lt(2).all()
    assert outputs["inferred"].edge_type.eq(2).all()
    assert outputs["none"].edge_index.shape == (2, 0)
    combined = outputs["combined"]
    randomized = outputs["random"]
    assert randomized.edge_index.shape == combined.edge_index.shape
    assert torch.equal(randomized.edge_index[1], combined.edge_index[1])
    assert torch.equal(randomized.edge_type, combined.edge_type)
    assert not torch.equal(randomized.edge_index[0], combined.edge_index[0])
    assert (randomized.edge_index[0] != randomized.edge_index[1]).all()
    with pytest.raises(ValueError, match="edge mode"):
        GateConfig(edge_mode="unknown")


def test_gate_selection_controls_use_same_caps_and_repeatable_random_graph():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D", "E", "F"),
        features=torch.tensor([[1., 0.], [0.9, 0.1], [1., 0.], [0., 1.],
                               [0.2, 0.8], [0.8, 0.2]]),
        sectors=torch.tensor([0, 0, 1, 1, 1, 1]),
        documented_edges=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        documented_types=torch.tensor([0, 0, 0]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director",),
    )
    results = {}
    for mode in ("learned", "fixed", "random"):
        torch.manual_seed(8)
        config = GateConfig(max_context_nodes=2, max_learned_neighbors=1,
                            max_incoming_edges=2, gate_mode=mode,
                            candidate_mode="raw", gate_seed=17)
        model = SectorRiskGNN(2, 2, 1, config)
        output = model(snapshot, "Tech")
        results[mode] = output
        assert output.target_indices.tolist() == [0, 1]
        assert len(output.selected_indices) == 4
        assert all(int((output.edge_index[1] == node).sum()) <= 2
                   for node in output.selected_indices)
        assert (output.edge_index[0] != output.edge_index[1]).all()
    assert set(results["fixed"].selected_indices[2:].tolist()) == {2, 5}
    torch.manual_seed(99)
    repeat = SectorRiskGNN(2, 2, 1, GateConfig(
        max_context_nodes=2, max_learned_neighbors=1, max_incoming_edges=2,
        gate_mode="random", candidate_mode="raw", gate_seed=17))(snapshot, "Tech")
    assert torch.equal(results["random"].selected_indices, repeat.selected_indices)
    assert torch.equal(results["random"].edge_index, repeat.edge_index)
    with pytest.raises(ValueError, match="gate mode"):
        GateConfig(gate_mode="unknown")
    with pytest.raises(ValueError, match="candidate mode"):
        GateConfig(candidate_mode="unknown")


def test_edge_first_context_reaches_target_under_equal_message_budget():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D", "E", "F"),
        features=torch.tensor([[1., 0.], [0.9, 0.1], [1., 0.], [0., 1.],
                               [0.2, 0.8], [0.8, 0.2]]),
        sectors=torch.tensor([0, 0, 1, 1, 1, 1]),
        documented_edges=torch.tensor([[0, 1], [2, 3]]),
        documented_types=torch.tensor([0, 0]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director",),
    )
    for mode in ("learned", "fixed", "random"):
        torch.manual_seed(8)
        model = SectorRiskGNN(2, 2, 1, GateConfig(
            max_context_nodes=4, max_learned_neighbors=2, max_incoming_edges=2,
            gate_mode=mode, candidate_mode="raw", context_mode="edge_first",
        ))
        output = model(snapshot, "Tech")
        targets = set(output.target_indices.tolist())
        contexts = set(output.selected_indices.tolist()) - targets
        src, dst = output.edge_index.tolist()
        assert targets == {0, 1}
        assert len(contexts) == 4
        assert len(src) == 2 * len(output.selected_indices)
        assert all(dst.count(node) == 2 for node in output.selected_indices.tolist())
        assert all(any(s == node and d in targets for s, d in zip(src, dst)) for node in contexts)
        if mode == "learned":
            bankruptcy_loss(output, torch.tensor([0., 1.]), positive_weight=2.).backward()
            assert model.node_gate[0].weight.grad.abs().sum() > 0
            assert model.edge_gate[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="context mode"):
        GateConfig(context_mode="unknown")
    with pytest.raises(ValueError, match="no node quota"):
        GateConfig(context_mode="edge_first", candidate_mode="raw",
                   documented_context_quota=1)


def test_edge_first_reserves_reverse_documented_message():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("target", "other", "documented"),
        features=torch.tensor([[1., 0.], [1., 0.], [1., 0.]]),
        sectors=torch.tensor([0, 1, 1]),
        documented_edges=torch.tensor([[0], [2]]),
        documented_types=torch.tensor([0]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director",),
    )
    model = SectorRiskGNN(2, 2, 1, GateConfig(
        max_context_nodes=1, max_learned_neighbors=1,
        max_incoming_edges=1, gate_mode="fixed", candidate_mode="raw",
        context_mode="edge_first",
    ))
    output = model(snapshot, "Tech")
    assert output.selected_indices.tolist() == [0, 2]
    assert (2, 0, 1) in set(zip(output.edge_index[0].tolist(),
                                output.edge_index[1].tolist(),
                                output.edge_type.tolist()))


def test_documented_context_quota_retains_direct_partner():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("target", "unlinked", "linked"),
        features=torch.tensor([[0.1], [0.2], [0.3]]),
        sectors=torch.tensor([0, 1, 1]),
        documented_edges=torch.tensor([[0], [2]]),
        documented_types=torch.tensor([0]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director",),
    )
    model = SectorRiskGNN(
        1, 2, 1,
        GateConfig(max_context_nodes=1, documented_context_quota=1,
                   max_learned_neighbors=0),
    )
    output = model(snapshot, "Tech")
    assert output.selected_indices.tolist() == [0, 2]
    assert output.edge_index.shape[1] == 2
    with pytest.raises(ValueError, match="quota"):
        GateConfig(max_context_nodes=1, documented_context_quota=2)


def test_relation_attention_trains_with_both_documented_and_inferred_edges():
    torch.manual_seed(7)
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D"),
        features=torch.tensor([[0.3, 1.4], [0.8, 0.6], [0.4, 1.2], [0.5, 0.9]]),
        sectors=torch.tensor([0, 0, 1, 1]),
        documented_edges=torch.tensor([[0, 2], [2, 1]]),
        documented_types=torch.tensor([0, 1]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director", "shared_major_owner"),
    )
    model = SectorRiskGNN(
        2, 2, 2,
        GateConfig(hidden_dim=16, max_context_nodes=2, max_learned_neighbors=1,
                   aggregation="relation_attention"),
    )
    output = model(snapshot, "Tech")
    assert output.logits.shape == (2,)
    assert set(output.edge_type.tolist()) >= {0, 1, 2, 3, 4}
    loss = bankruptcy_loss(output, torch.tensor([0.0, 1.0]), positive_weight=2.0)
    loss.backward()
    assert model.layers[0].query.weight.grad.abs().sum() > 0
    assert model.layers[0].relation_value.weight.grad.abs().sum() > 0
    assert model.node_gate[0].weight.grad.abs().sum() > 0
    assert model.edge_gate[0].weight.grad.abs().sum() > 0


def test_unknown_bankruptcy_outcome_is_excluded_from_loss():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B"),
        features=torch.tensor([[0.1], [0.2]]),
        sectors=torch.tensor([0, 0]),
        documented_edges=torch.empty((2, 0), dtype=torch.long),
        documented_types=torch.empty(0, dtype=torch.long),
        sector_names=("Tech",),
        relation_names=(),
    )
    model = SectorRiskGNN(1, 1, 0, GateConfig(max_learned_neighbors=0))
    output = model(snapshot, "Tech")
    loss = bankruptcy_loss(output, torch.tensor([1.0, float("nan")]))
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="no known"):
        bankruptcy_loss(output, torch.tensor([float("nan"), float("nan")]))


def test_symmetric_documented_uses_one_type_for_both_directions():
    snapshot = GraphSnapshot(
        as_of=date(2020, 12, 31),
        company_ids=("A", "B", "C", "D"),
        features=torch.tensor([[0.3, 1.4], [0.8, 0.6], [0.4, 1.2], [0.5, 0.9]]),
        sectors=torch.tensor([0, 0, 1, 1]),
        documented_edges=torch.tensor([[0, 1], [2, 3]]),
        documented_types=torch.tensor([0, 1]),
        sector_names=("Tech", "Energy"),
        relation_names=("shared_director", "shared_major_owner"),
    )
    for context_mode, extra in (("node_first", {}), ("edge_first", {"candidate_mode": "raw"})):
        torch.manual_seed(3)
        config = GateConfig(max_context_nodes=2, max_learned_neighbors=1, max_incoming_edges=4,
                            symmetric_documented=True, context_mode=context_mode, **extra)
        output = SectorRiskGNN(2, 2, 2, config)(snapshot, "Tech")
        assert set(output.edge_type.tolist()) <= {0, 1, 4}  # no reverse-offset IDs 2, 3
        legacy = GateConfig(max_context_nodes=2, max_learned_neighbors=1, max_incoming_edges=4,
                            context_mode=context_mode, **extra)
        torch.manual_seed(3)
        legacy_output = SectorRiskGNN(2, 2, 2, legacy)(snapshot, "Tech")
        assert set(legacy_output.edge_type.tolist()) & {2, 3}


def test_true_mean_aggregation_is_a_weighted_mean():
    from risk_gnn.model import RelationalLayer

    torch.manual_seed(0)
    h = torch.randn(3, 4)
    edges = torch.tensor([[1, 2], [0, 0]])
    types = torch.zeros(2, dtype=torch.long)
    weight = torch.tensor([0.1, 0.1])
    legacy, mean = RelationalLayer(4, 1), RelationalLayer(4, 1, true_mean=True)
    mean.load_state_dict(legacy.state_dict())
    messages = mean.message_linear(h[[1, 2]]) * mean.relation_scale(types)
    expected = torch.relu(mean.self_linear(h)[0] + (messages * weight[:, None]).sum(0) / weight.sum())
    assert torch.allclose(mean(h, edges, types, weight)[0], expected, atol=1e-6)
    assert not torch.allclose(legacy(h, edges, types, weight)[0], expected, atol=1e-4)
