"""Production-shape regression tests for the native relation-aware GNN."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.gnn_training.model import RelationMessagePassingLayer  # noqa: E402


def _per_edge_reference(
    layer: RelationMessagePassingLayer,
    node_states: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the former allocation-heavy equation for small test graphs."""

    source_index, destination_index = edge_index
    update = torch.zeros_like(node_states)
    transforms = layer.relation_weight[edge_type]
    messages = torch.einsum("ei,eij->ej", node_states[source_index], transforms)
    relation_gates = torch.sigmoid(layer.relation_gate_logits[edge_type])
    messages = messages * relation_gates.to(node_states.dtype).unsqueeze(-1)
    messages = messages * edge_weight.to(node_states.dtype).unsqueeze(-1)
    update.index_add_(0, destination_index, messages)
    return layer.norm(node_states + layer.activation(update))


def test_relation_grouped_aggregation_matches_per_edge_forward_and_gradients() -> None:
    torch.manual_seed(7)
    hidden_dim = 8
    relation_count = 4
    node_count = 13
    edge_count = 47
    edge_index = torch.stack(
        (
            torch.randint(node_count, (edge_count,)),
            torch.randint(node_count, (edge_count,)),
        )
    )
    edge_type = torch.randint(relation_count, (edge_count,))
    edge_weight = torch.rand(edge_count)

    actual_layer = RelationMessagePassingLayer(
        hidden_dim,
        relation_count,
        dropout=0.0,
    )
    reference_layer = RelationMessagePassingLayer(
        hidden_dim,
        relation_count,
        dropout=0.0,
    )
    reference_layer.load_state_dict(actual_layer.state_dict())
    actual_states = torch.randn(node_count, hidden_dim, requires_grad=True)
    reference_states = actual_states.detach().clone().requires_grad_(True)

    actual = actual_layer(actual_states, edge_index, edge_type, edge_weight)
    reference = _per_edge_reference(
        reference_layer,
        reference_states,
        edge_index,
        edge_type,
        edge_weight,
    )

    assert torch.allclose(actual, reference, rtol=2e-5, atol=2e-6)

    actual.square().sum().backward()
    reference.square().sum().backward()
    assert torch.allclose(
        actual_states.grad,
        reference_states.grad,
        rtol=3e-5,
        atol=3e-6,
    )
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        actual_layer.named_parameters(),
        reference_layer.named_parameters(),
        strict=True,
    ):
        assert actual_name == reference_name
        assert actual_parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.allclose(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=3e-5,
            atol=3e-6,
        )


def test_relation_grouped_aggregation_handles_protected_scale_edge_count() -> None:
    """Exercise the edge count that previously implied a 5.5 GiB temporary."""

    torch.manual_seed(11)
    hidden_dim = 128
    relation_count = 11
    node_count = 4_096
    edge_count = 91_000
    layer = RelationMessagePassingLayer(
        hidden_dim,
        relation_count,
        dropout=0.0,
    )
    node_states = torch.randn(node_count, hidden_dim)
    edge_index = torch.stack(
        (
            torch.randint(node_count, (edge_count,)),
            torch.randint(node_count, (edge_count,)),
        )
    )
    edge_type = torch.randint(relation_count, (edge_count,))
    edge_weight = torch.rand(edge_count)

    output = layer(node_states, edge_index, edge_type, edge_weight)

    assert output.shape == (node_count, hidden_dim)
    assert torch.isfinite(output).all()
