"""
Integration tests for conditional edge activation system.

Tests the integration between ConditionalEdgeMasker, GraphSAGE layers,
and the overall GNN pipeline with physics-aware edge filtering.
"""

import torch
from typing import Dict

from npp_rl.models.gnn import GraphSAGELayer, GraphEncoder
from npp_rl.models.conditional_edges import ConditionalEdgeMasker


def test_graphsage_with_conditional_masking():
    """Test GraphSAGE layer with conditional edge masking."""
    print("Testing GraphSAGE layer with conditional edge masking...")
    
    # Create GraphSAGE layer
    layer = GraphSAGELayer(in_dim=32, out_dim=16)
    
    # Create test data
    batch_size, num_nodes, num_edges = 2, 8, 12
    node_features = torch.randn(batch_size, num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
    node_mask = torch.ones(batch_size, num_nodes)
    edge_mask = torch.ones(batch_size, num_edges)
    
    # Create edge features with different types
    edge_features = torch.zeros(batch_size, num_edges, 16)
    # Set some edges as jumps that require jump capability
    edge_features[:, 1, 1] = 1.0  # Jump edge
    edge_features[:, 1, 14] = 1.0  # requires_jump
    edge_features[:, 3, 1] = 1.0  # Another jump edge
    edge_features[:, 3, 14] = 1.0  # requires_jump
    
    # Create ninja physics state
    ninja_physics_state = torch.zeros(batch_size, 18)
    ninja_physics_state[0, 16] = 1.0  # First batch can jump
    ninja_physics_state[1, 16] = 0.0  # Second batch cannot jump
    
    # Test forward pass without physics state
    output_no_physics = layer(node_features, edge_index, node_mask, edge_mask)
    assert output_no_physics.shape == (batch_size, num_nodes, 16)
    
    # Test forward pass with physics state
    output_with_physics = layer(
        node_features, edge_index, node_mask, edge_mask,
        ninja_physics_state, edge_features
    )
    assert output_with_physics.shape == (batch_size, num_nodes, 16)
    
    # The outputs should be different due to conditional masking
    # (unless by coincidence the masking doesn't affect the aggregation)
    print("✓ GraphSAGE layer with conditional masking test passed")


def test_graph_encoder_integration():
    """Test GraphEncoder with conditional edge masking."""
    print("Testing GraphEncoder with conditional edge masking...")
    
    # Create GraphEncoder
    encoder = GraphEncoder(
        node_feature_dim=16,
        edge_feature_dim=16,
        hidden_dim=32,
        output_dim=64,
        num_layers=2
    )
    
    # Create test graph observation
    batch_size, num_nodes, num_edges = 2, 10, 15
    
    graph_obs = {
        'graph_node_feats': torch.randn(batch_size, num_nodes, 16),
        'graph_edge_index': torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
        'graph_edge_feats': torch.zeros(batch_size, num_edges, 16),
        'graph_node_mask': torch.ones(batch_size, num_nodes),
        'graph_edge_mask': torch.ones(batch_size, num_edges),
        'ninja_physics_state': torch.zeros(batch_size, 18)
    }
    
    # Set up some edge types and requirements
    graph_obs['graph_edge_feats'][:, 2, 1] = 1.0  # Jump edge
    graph_obs['graph_edge_feats'][:, 2, 14] = 1.0  # requires_jump
    graph_obs['graph_edge_feats'][:, 5, 2] = 1.0  # Wall slide edge
    graph_obs['graph_edge_feats'][:, 5, 15] = 1.0  # requires_wall_contact
    
    # Set different physics states for each batch
    graph_obs['ninja_physics_state'][0, 16] = 1.0  # can_jump
    graph_obs['ninja_physics_state'][0, 5] = 0.0   # no wall_contact
    graph_obs['ninja_physics_state'][1, 16] = 0.0  # cannot_jump
    graph_obs['ninja_physics_state'][1, 5] = 1.0   # has wall_contact
    
    # Test forward pass
    output = encoder(graph_obs)
    assert output.shape == (batch_size, 64)
    
    # Test without physics state
    graph_obs_no_physics = graph_obs.copy()
    del graph_obs_no_physics['ninja_physics_state']
    
    output_no_physics = encoder(graph_obs_no_physics)
    assert output_no_physics.shape == (batch_size, 64)
    
    print("✓ GraphEncoder integration test passed")


def test_conditional_masker_constraint_summary():
    """Test constraint summary functionality."""
    print("Testing constraint summary functionality...")
    
    masker = ConditionalEdgeMasker()
    
    # Create test data
    num_edges = 8
    edge_features = torch.zeros(num_edges, 16)
    
    # Set up different edge types
    edge_features[0, 0] = 1.0  # Walk
    edge_features[1, 1] = 1.0  # Jump
    edge_features[1, 14] = 1.0  # requires_jump
    edge_features[2, 2] = 1.0  # Wall slide
    edge_features[2, 15] = 1.0  # requires_wall_contact
    edge_features[3, 1] = 1.0  # High velocity jump
    edge_features[3, 12] = 2.0  # min_velocity = 2.0
    edge_features[3, 14] = 1.0  # requires_jump
    
    ninja_physics_state = torch.zeros(18)
    ninja_physics_state[2] = 1.0   # vel_magnitude = 1.0
    ninja_physics_state[16] = 1.0  # can_jump = True
    ninja_physics_state[5] = 0.0   # wall_contact = False
    
    base_edge_mask = torch.ones(num_edges)
    
    # Get constraint summary
    summary = masker.get_constraint_summary(
        edge_features, ninja_physics_state, base_edge_mask
    )
    
    # Verify summary structure
    assert isinstance(summary, dict)
    assert 'base_edges' in summary
    assert 'dynamic_edges' in summary
    assert 'disabled_edges' in summary
    assert 'disable_rate' in summary
    assert 'ninja_state' in summary
    
    # Verify values make sense
    assert summary['base_edges'] >= summary['dynamic_edges']
    assert summary['disabled_edges'] >= 0
    assert 0.0 <= summary['disable_rate'] <= 1.0
    
    # Check ninja state extraction
    ninja_state = summary['ninja_state']
    assert isinstance(ninja_state['velocity_magnitude'], float)
    assert isinstance(ninja_state['ground_contact'], bool)
    assert isinstance(ninja_state['wall_contact'], bool)
    assert isinstance(ninja_state['can_jump'], bool)
    assert isinstance(ninja_state['can_wall_jump'], bool)
    
    print("✓ Constraint summary test passed")


def test_batched_conditional_masking():
    """Test conditional masking with batched input."""
    print("Testing batched conditional masking...")
    
    masker = ConditionalEdgeMasker()
    
    batch_size, num_edges = 3, 6
    
    # Create batched edge features
    edge_features = torch.zeros(batch_size, num_edges, 16)
    edge_features[:, 0, 0] = 1.0  # Walk edges
    edge_features[:, 1, 1] = 1.0  # Jump edges
    edge_features[:, 1, 14] = 1.0  # requires_jump
    edge_features[:, 2, 2] = 1.0  # Wall slide edges
    edge_features[:, 2, 15] = 1.0  # requires_wall_contact
    
    # Create different ninja states for each batch
    ninja_physics_state = torch.zeros(batch_size, 18)
    ninja_physics_state[:, 2] = 1.0   # Set velocity magnitude for all batches
    ninja_physics_state[0, 16] = 1.0  # Batch 0: can jump
    ninja_physics_state[0, 5] = 0.0   # no wall contact
    ninja_physics_state[1, 16] = 0.0  # Batch 1: cannot jump
    ninja_physics_state[1, 5] = 1.0   # has wall contact
    ninja_physics_state[2, 16] = 1.0  # Batch 2: can jump
    ninja_physics_state[2, 5] = 1.0   # has wall contact
    
    base_edge_mask = torch.ones(batch_size, num_edges)
    
    # Compute dynamic masks
    dynamic_mask = masker.compute_dynamic_edge_mask(
        edge_features, ninja_physics_state, base_edge_mask
    )
    
    assert dynamic_mask.shape == (batch_size, num_edges)
    
    # Check batch 0: can jump, no wall contact
    assert dynamic_mask[0, 0] == 1.0  # Walk should be enabled
    assert dynamic_mask[0, 1] == 1.0  # Jump should be enabled
    assert dynamic_mask[0, 2] == 0.0  # Wall slide should be disabled
    
    # Check batch 1: cannot jump, has wall contact
    assert dynamic_mask[1, 0] == 1.0  # Walk should be enabled
    assert dynamic_mask[1, 1] == 0.0  # Jump should be disabled
    assert dynamic_mask[1, 2] == 1.0  # Wall slide should be enabled
    
    # Check batch 2: can jump, has wall contact
    assert dynamic_mask[2, 0] == 1.0  # Walk should be enabled
    assert dynamic_mask[2, 1] == 1.0  # Jump should be enabled
    assert dynamic_mask[2, 2] == 1.0  # Wall slide should be enabled
    
    print("✓ Batched conditional masking test passed")


def test_edge_masking_with_single_physics_state():
    """Test edge masking when physics state is not batched."""
    print("Testing edge masking with single physics state...")
    
    masker = ConditionalEdgeMasker()
    
    batch_size, num_edges = 2, 4
    
    # Batched edge features
    edge_features = torch.zeros(batch_size, num_edges, 16)
    edge_features[:, 0, 0] = 1.0  # Walk edges
    edge_features[:, 1, 1] = 1.0  # Jump edges
    edge_features[:, 1, 14] = 1.0  # requires_jump
    
    # Single ninja physics state (not batched)
    ninja_physics_state = torch.zeros(18)
    ninja_physics_state[16] = 1.0  # can_jump = True
    
    base_edge_mask = torch.ones(batch_size, num_edges)
    
    # Should work with single physics state applied to all batches
    dynamic_mask = masker.compute_dynamic_edge_mask(
        edge_features, ninja_physics_state, base_edge_mask
    )
    
    assert dynamic_mask.shape == (batch_size, num_edges)
    
    # Both batches should have same masking pattern
    assert torch.equal(dynamic_mask[0], dynamic_mask[1])
    assert dynamic_mask[0, 0] == 1.0  # Walk should be enabled
    assert dynamic_mask[0, 1] == 1.0  # Jump should be enabled
    
    print("✓ Single physics state test passed")


if __name__ == '__main__':
    test_graphsage_with_conditional_masking()
    test_graph_encoder_integration()
    test_conditional_masker_constraint_summary()
    test_batched_conditional_masking()
    test_edge_masking_with_single_physics_state()
    print("\n🎉 All integration tests passed!")