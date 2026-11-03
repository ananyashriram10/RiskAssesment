"""Human-readable trace of the selected graph."""

from .data import GraphSnapshot
from .model import ModelOutput


def selected_edge_rows(snapshot: GraphSnapshot, output: ModelOutput) -> list[dict[str, object]]:
    """Return selected directed links, clearly separating inferred associations."""

    relation_count = len(snapshot.relation_names)
    rows = []
    for edge_number in range(output.edge_index.shape[1]):
        source, target = output.edge_index[:, edge_number].tolist()
        type_id = output.edge_type[edge_number].item()
        if type_id == 2 * relation_count:
            kind, relation, direction = "inferred", "learned_association", "model"
        elif type_id < relation_count:
            kind, relation, direction = "documented", snapshot.relation_names[type_id], "forward"
        else:
            kind = "documented"
            relation = snapshot.relation_names[type_id - relation_count]
            direction = "reverse_message"
        rows.append(
            {
                "source": snapshot.company_ids[source],
                "target": snapshot.company_ids[target],
                "kind": kind,
                "relation": relation,
                "direction": direction,
                "gate_weight": float(output.edge_weight[edge_number].detach()),
            }
        )
    return rows
