"""
Basic test script for Phase 2 implementations.

This script tests the core components of Phase 2 to ensure they work correctly
before integrating them into the full training pipeline.
"""

import torch
import numpy as np
from gymnasium.spaces import Box, Dict as SpacesDict

from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.models.gnn import GraphEncoder, create_graph_encoder
from npp_rl.feature_extractors import NppMultimodalGraphExtractor
from nclone.graph.graph_builder import GraphBuilder, N_MAX_NODES, E_MAX_EDGES


def test_icm():
    """Test ICM implementation."""
    print("Testing ICM...")
    
    # Create ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        hidden_dim=256
    )
    
    # Create trainer
    trainer = ICMTrainer(icm, learning_rate=1e-3)
    
    # Test forward pass
    batch_size = 32
    features_current = torch.randn(batch_size, 512)
    features_next = torch.randn(batch_size, 512)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Test ICM forward
    output = icm(features_current, features_next, actions)
    
    assert 'intrinsic_reward' in output
    assert output['intrinsic_reward'].shape == (batch_size,)
    assert 'inverse_loss' in output
    assert 'forward_loss' in output
    
    # Test trainer update
    stats = trainer.update(features_current, features_next, actions)
    
    assert 'inverse_loss' in stats
    assert 'forward_loss' in stats
    assert 'total_loss' in stats
    assert 'mean_intrinsic_reward' in stats
    
    print("✅ ICM test passed")


def test_graph_builder():
    """Test graph builder."""
    print("Testing Graph Builder...")
    
    builder = GraphBuilder()
    
    # Mock level data
    level_data = {
        'tiles': np.zeros((23, 42), dtype=np.int32),
        'width': 42,
        'height': 23
    }
    
    # Mock entities
    entities = [
        {'type': 'exit_switch', 'x': 100, 'y': 200, 'active': False, 'state': 0.0},
        {'type': 'exit_door', 'x': 300, 'y': 400, 'active': True, 'state': 0.0}
    ]
    
    ninja_position = (150, 250)
    
    # Build graph
    graph_data = builder.build_graph(level_data, ninja_position, entities)
    
    assert graph_data.node_features.shape == (N_MAX_NODES, builder.node_feature_dim)
    assert graph_data.edge_index.shape == (2, E_MAX_EDGES)
    assert graph_data.edge_features.shape == (E_MAX_EDGES, builder.edge_feature_dim)
    assert graph_data.node_mask.shape == (N_MAX_NODES,)
    assert graph_data.edge_mask.shape == (E_MAX_EDGES,)
    
    # Check that we have some nodes and edges
    assert graph_data.num_nodes > 0
    assert graph_data.num_edges >= 0
    
    print(f"✅ Graph Builder test passed (nodes: {graph_data.num_nodes}, edges: {graph_data.num_edges})")


def test_gnn():
    """Test GNN encoder."""
    print("Testing GNN Encoder...")
    
    # Create mock graph observation
    batch_size = 4
    node_feat_dim = 67  # From GraphBuilder
    edge_feat_dim = 9   # From GraphBuilder
    
    graph_obs = {
        'graph_node_feats': torch.randn(batch_size, N_MAX_NODES, node_feat_dim),
        'graph_edge_index': torch.randint(0, N_MAX_NODES, (batch_size, 2, E_MAX_EDGES)),
        'graph_edge_feats': torch.randn(batch_size, E_MAX_EDGES, edge_feat_dim),
        'graph_node_mask': torch.ones(batch_size, N_MAX_NODES),
        'graph_edge_mask': torch.ones(batch_size, E_MAX_EDGES)
    }
    
    # Create GNN encoder
    gnn = create_graph_encoder(
        node_feature_dim=node_feat_dim,
        edge_feature_dim=edge_feat_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=128
    )
    
    # Test forward pass
    output = gnn(graph_obs)
    
    assert output.shape == (batch_size, 128)
    
    print("✅ GNN Encoder test passed")


def test_multimodal_extractor():
    """Test multimodal feature extractor."""
    print("Testing Multimodal Feature Extractor...")
    
    # Create observation space
    observation_space = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        'game_state': Box(low=-1, high=1, shape=(32,), dtype=np.float32),
        'graph_node_feats': Box(low=-1, high=1, shape=(N_MAX_NODES, 67), dtype=np.float32),
        'graph_edge_index': Box(low=0, high=N_MAX_NODES-1, shape=(2, E_MAX_EDGES), dtype=np.int32),
        'graph_edge_feats': Box(low=-1, high=1, shape=(E_MAX_EDGES, 9), dtype=np.float32),
        'graph_node_mask': Box(low=0, high=1, shape=(N_MAX_NODES,), dtype=np.float32),
        'graph_edge_mask': Box(low=0, high=1, shape=(E_MAX_EDGES,), dtype=np.float32)
    })
    
    # Test with graph observations
    extractor = NppMultimodalGraphExtractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=True,
    )
    
    # Create mock observations
    batch_size = 2
    observations = {
        'player_frame': torch.randint(0, 256, (batch_size, 64, 64, 4), dtype=torch.uint8),
        'global_view': torch.randint(0, 256, (batch_size, 128, 128, 1), dtype=torch.uint8),
        'game_state': torch.randn(batch_size, 32),
        'graph_node_feats': torch.randn(batch_size, N_MAX_NODES, 67),
        'graph_edge_index': torch.randint(0, N_MAX_NODES, (batch_size, 2, E_MAX_EDGES)),
        'graph_edge_feats': torch.randn(batch_size, E_MAX_EDGES, 9),
        'graph_node_mask': torch.ones(batch_size, N_MAX_NODES),
        'graph_edge_mask': torch.ones(batch_size, E_MAX_EDGES)
    }
    
    # Test forward pass
    output = extractor(observations)
    
    assert output.shape == (batch_size, 512)
    
    print("✅ Multimodal Feature Extractor test passed")


def test_without_graph():
    """Test components without graph observations."""
    print("Testing without graph observations...")
    
    # Create observation space without graph
    observation_space = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        'game_state': Box(low=-1, high=1, shape=(32,), dtype=np.float32)
    })
    
    # Test extractor without graph
    extractor = NppMultimodalGraphExtractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=False,
    )
    
    # Create mock observations
    batch_size = 2
    observations = {
        'player_frame': torch.randint(0, 256, (batch_size, 64, 64, 4), dtype=torch.uint8),
        'global_view': torch.randint(0, 256, (batch_size, 128, 128, 1), dtype=torch.uint8),
        'game_state': torch.randn(batch_size, 32)
    }
    
    # Test forward pass
    output = extractor(observations)
    
    assert output.shape == (batch_size, 512)
    
    print("✅ Test without graph observations passed")


def main():
    """Run all tests."""
    print("Running Phase 2 basic tests...\n")
    
    try:
        test_icm()
        test_graph_builder()
        test_gnn()
        test_multimodal_extractor()
        test_without_graph()
        
        print("\n🎉 All Phase 2 basic tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()