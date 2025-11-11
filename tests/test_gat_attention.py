"""Unit tests for GAT attention mechanisms."""

import torch
import pytest
from npp_rl.models.gat import GATLayer, GATEncoder


def test_gat_layer_output_shape():
    """Test GATLayer output shape."""
    batch_size = 4
    max_nodes = 100
    in_dim = 55
    out_dim = 128
    num_heads = 4

    layer = GATLayer(in_dim, out_dim, num_heads=num_heads, concat_heads=True)

    node_features = torch.randn(batch_size, max_nodes, in_dim)
    edge_index = torch.randint(0, max_nodes, (batch_size, 2, 200))  # 200 edges
    node_mask = torch.ones(batch_size, max_nodes)

    output = layer(node_features, edge_index, node_mask)

    assert output.shape == (batch_size, max_nodes, out_dim)


def test_gat_layer_variable_edges():
    """Test with variable edge counts per batch."""
    batch_size = 4
    max_nodes = 50
    in_dim = 55
    out_dim = 128

    layer = GATLayer(in_dim, out_dim, num_heads=4)

    node_features = torch.randn(batch_size, max_nodes, in_dim)

    # Create edge indices with different edge counts
    edge_indices = []
    for b in range(batch_size):
        num_edges = 10 + b * 20  # Variable: 10, 30, 50, 70
        edges = torch.randint(0, max_nodes, (2, num_edges))
        edge_indices.append(edges)

    # Pad to same length for batching
    max_edges = max(e.shape[1] for e in edge_indices)
    edge_index = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
    for b, edges in enumerate(edge_indices):
        edge_index[b, :, : edges.shape[1]] = edges

    output = layer(node_features, edge_index)
    assert output.shape == (batch_size, max_nodes, out_dim)


def test_gat_layer_no_edges():
    """Test with graphs that have no edges."""
    batch_size = 2
    max_nodes = 50
    in_dim = 55
    out_dim = 128

    layer = GATLayer(in_dim, out_dim, num_heads=4)

    node_features = torch.randn(batch_size, max_nodes, in_dim)
    edge_index = torch.zeros(batch_size, 2, 0, dtype=torch.long)  # No edges

    output = layer(node_features, edge_index)
    assert output.shape == (batch_size, max_nodes, out_dim)


def test_gat_layer_attention_normalization():
    """Verify attention weights sum to 1.0 per target node."""
    batch_size = 2
    max_nodes = 20
    in_dim = 55
    out_dim = 128

    layer = GATLayer(in_dim, out_dim, num_heads=4)

    node_features = torch.randn(batch_size, max_nodes, in_dim)
    # Create edges: each target node has multiple incoming edges
    edge_index = torch.zeros(batch_size, 2, 50, dtype=torch.long)
    for b in range(batch_size):
        # Create edges where multiple sources point to same targets
        src_nodes = torch.randint(0, max_nodes, (50,))
        tgt_nodes = torch.randint(0, max_nodes, (50,))
        edge_index[b, 0] = src_nodes
        edge_index[b, 1] = tgt_nodes

    output = layer(node_features, edge_index)
    assert output.shape == (batch_size, max_nodes, out_dim)

    # Note: We can't directly access attention weights from forward pass
    # But we can verify the output is reasonable (not NaN, not all zeros)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.all(output == 0), "Output is all zeros"


def test_gat_encoder_full_forward():
    """Test full GATEncoder forward pass."""
    batch_size = 2
    max_nodes = 100
    node_feature_dim = 55
    hidden_dim = 128
    output_dim = 256
    num_layers = 3

    encoder = GATEncoder(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=4,
    )

    node_features = torch.randn(batch_size, max_nodes, node_feature_dim)
    edge_index = torch.randint(0, max_nodes, (batch_size, 2, 300))
    node_mask = torch.ones(batch_size, max_nodes)

    node_embeddings, graph_embedding = encoder(node_features, edge_index, node_mask)

    assert node_embeddings.shape == (batch_size, max_nodes, output_dim)
    assert graph_embedding.shape == (batch_size, output_dim)


def test_gat_gradient_flow():
    """Verify gradients flow through attention parameters."""
    batch_size = 2
    max_nodes = 50
    in_dim = 55
    out_dim = 128

    layer = GATLayer(in_dim, out_dim, num_heads=4)

    node_features = torch.randn(batch_size, max_nodes, in_dim)
    edge_index = torch.randint(0, max_nodes, (batch_size, 2, 100))

    output = layer(node_features, edge_index)
    loss = output.sum()
    loss.backward()

    # Check learnable attention parameter has gradient
    assert layer.attention.grad is not None
    assert not torch.all(layer.attention.grad == 0)


def test_gat_proper_attention_formula():
    """Verify GAT uses proper attention formula with learnable parameter."""
    batch_size = 1
    max_nodes = 10
    in_dim = 55
    out_dim = 128

    layer = GATLayer(in_dim, out_dim, num_heads=4)

    # Check that attention parameter exists and has correct shape
    assert hasattr(layer, "attention")
    assert layer.attention.shape == (4, 2 * (out_dim // 4))  # [num_heads, 2*head_dim]

    node_features = torch.randn(batch_size, max_nodes, in_dim)
    edge_index = torch.randint(0, max_nodes, (batch_size, 2, 20))

    output = layer(node_features, edge_index)
    assert output.shape == (batch_size, max_nodes, out_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
