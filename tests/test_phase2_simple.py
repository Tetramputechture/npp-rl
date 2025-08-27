"""
Simplified Phase 2 test that doesn't require full environment setup.

This test focuses on the core Phase 2 components without the complex
environment integration that might have setup issues.
"""

import torch
import numpy as np
from gymnasium.spaces import Box, Dict as SpacesDict

from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.models.gnn import GraphEncoder, create_graph_encoder
from npp_rl.feature_extractors import NppMultimodalGraphExtractor
from npp_rl.eval.exploration_metrics import ExplorationMetrics
from npp_rl.config.phase2_config import Phase2Config, create_full_phase2_config
from nclone.graph.graph_builder import GraphBuilder, N_MAX_NODES, E_MAX_EDGES


def test_icm_standalone():
    """Test ICM as standalone component."""
    print("Testing ICM standalone...")
    
    # Create ICM
    icm = ICMNetwork(feature_dim=512, action_dim=6)
    trainer = ICMTrainer(icm)
    
    # Test with batch of features
    batch_size = 16
    current_features = torch.randn(batch_size, 512)
    next_features = torch.randn(batch_size, 512)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Test ICM forward pass
    output = icm(current_features, next_features, actions)
    
    assert output['intrinsic_reward'].shape == (batch_size,)
    assert 'inverse_loss' in output
    assert 'forward_loss' in output
    
    # Test trainer update
    stats = trainer.update(current_features, next_features, actions)
    assert 'total_loss' in stats
    
    # Test intrinsic reward computation
    rewards = trainer.get_intrinsic_reward(current_features, next_features, actions)
    assert rewards.shape == (batch_size,)
    
    print("✅ ICM standalone test passed")


def test_gnn_standalone():
    """Test GNN as standalone component."""
    print("Testing GNN standalone...")
    
    # Create mock graph data
    batch_size = 4
    node_feat_dim = 67
    edge_feat_dim = 9
    
    graph_obs = {
        'graph_node_feats': torch.randn(batch_size, N_MAX_NODES, node_feat_dim),
        'graph_edge_index': torch.randint(0, N_MAX_NODES, (batch_size, 2, E_MAX_EDGES)),
        'graph_edge_feats': torch.randn(batch_size, E_MAX_EDGES, edge_feat_dim),
        'graph_node_mask': torch.ones(batch_size, N_MAX_NODES),
        'graph_edge_mask': torch.ones(batch_size, E_MAX_EDGES)
    }
    
    # Create GNN
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
    
    print("✅ GNN standalone test passed")


def test_graph_builder_standalone():
    """Test graph builder as standalone component."""
    print("Testing graph builder standalone...")
    
    builder = GraphBuilder()
    
    # Create mock level data
    level_data = {
        'tiles': np.zeros((23, 42), dtype=np.int32),
        'width': 42,
        'height': 23
    }
    
    entities = [
        {'type': 'exit_switch', 'x': 100, 'y': 200, 'active': False, 'state': 0.0},
        {'type': 'exit_door', 'x': 300, 'y': 400, 'active': True, 'state': 0.0}
    ]
    
    ninja_position = (150, 250)
    
    # Build graph
    graph_data = builder.build_graph(level_data, ninja_position, entities)
    
    assert graph_data.node_features.shape == (N_MAX_NODES, builder.node_feature_dim)
    assert graph_data.edge_index.shape == (2, E_MAX_EDGES)
    assert graph_data.num_nodes > 0
    
    print(f"✅ Graph builder test passed (nodes: {graph_data.num_nodes}, edges: {graph_data.num_edges})")


def test_feature_extractor_standalone():
    """Test feature extractor with mock observation space."""
    print("Testing feature extractor standalone...")
    
    # Create mock observation space
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
        use_graph_obs=True
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
    
    # Test without graph observations
    obs_space_no_graph = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        'game_state': Box(low=-1, high=1, shape=(32,), dtype=np.float32)
    })
    
    extractor_no_graph = NppMultimodalGraphExtractor(
        observation_space=obs_space_no_graph,
        features_dim=512,
        use_graph_obs=False
    )
    
    obs_no_graph = {
        'player_frame': torch.randint(0, 256, (batch_size, 64, 64, 4), dtype=torch.uint8),
        'global_view': torch.randint(0, 256, (batch_size, 128, 128, 1), dtype=torch.uint8),
        'game_state': torch.randn(batch_size, 32)
    }
    
    output_no_graph = extractor_no_graph(obs_no_graph)
    assert output_no_graph.shape == (batch_size, 512)
    
    print("✅ Feature extractor standalone test passed")


def test_exploration_metrics_standalone():
    """Test exploration metrics as standalone component."""
    print("Testing exploration metrics standalone...")
    
    metrics = ExplorationMetrics()
    
    # Simulate episode
    metrics.reset_episode()
    
    # Simulate movement pattern
    positions = [
        (100, 100), (124, 100), (148, 100),  # Right
        (148, 124), (148, 148),              # Down
        (124, 148), (100, 148),              # Left
        (100, 124), (100, 100)               # Up (back to start)
    ]
    
    for i, pos in enumerate(positions):
        intrinsic_reward = 0.1 * (1.0 - i / len(positions))  # Decreasing reward
        metrics.update_step(pos, intrinsic_reward)
    
    # End episode
    episode_metrics = metrics.end_episode(success=True)
    
    assert 'coverage' in episode_metrics
    assert 'visitation_entropy' in episode_metrics
    assert 'mean_intrinsic_reward' in episode_metrics
    assert episode_metrics['success'] == 1.0
    assert episode_metrics['unique_cells_visited'] > 0
    
    # Test rolling metrics
    rolling = metrics.get_rolling_metrics()
    assert 'rolling_coverage' in rolling
    
    # Test statistics
    stats = metrics.get_episode_statistics()
    assert 'total_episodes' in stats
    assert stats['total_episodes'] == 1
    
    print("✅ Exploration metrics standalone test passed")


def test_config_system_standalone():
    """Test configuration system as standalone component."""
    print("Testing configuration system standalone...")
    
    # Test default config creation
    config = Phase2Config()
    assert hasattr(config, 'icm')
    assert hasattr(config, 'graph')
    assert hasattr(config, 'bc')
    
    # Test preset configs
    full_config = create_full_phase2_config()
    assert full_config.icm.enabled == True
    assert full_config.graph.enabled == True
    assert full_config.bc.enabled == True
    
    # Test serialization
    config_dict = config.to_dict()
    assert 'icm' in config_dict
    assert 'graph' in config_dict
    
    # Test deserialization
    loaded_config = Phase2Config.from_dict(config_dict)
    assert loaded_config.icm.enabled == config.icm.enabled
    
    # Test validation
    from npp_rl.config.phase2_config import validate_config
    messages = validate_config(config)
    # Should return list (may be empty or have warnings)
    assert isinstance(messages, list)
    
    print("✅ Configuration system standalone test passed")


def test_integration_without_env():
    """Test integration of components without environment."""
    print("Testing component integration...")
    
    # Create config
    config = Phase2Config()
    config.icm.enabled = True
    config.graph.enabled = True
    
    # Create mock observation space
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
    
    # Create feature extractor
    extractor = NppMultimodalGraphExtractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=True
    )
    
    # Create ICM
    icm = ICMNetwork(feature_dim=512, action_dim=6)
    icm_trainer = ICMTrainer(icm)
    
    # Create exploration metrics
    metrics = ExplorationMetrics()
    
    # Simulate training step
    batch_size = 4
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
    
    # Extract features
    features = extractor(observations)
    assert features.shape == (batch_size, 512)
    
    # Simulate ICM update
    current_features = features[:batch_size//2]
    next_features = features[batch_size//2:]
    actions = torch.randint(0, 6, (batch_size//2,))
    
    stats = icm_trainer.update(current_features, next_features, actions)
    assert 'total_loss' in stats
    
    # Simulate exploration tracking
    metrics.reset_episode()
    for i in range(10):
        pos = (100 + i * 10, 100 + i * 5)
        metrics.update_step(pos, stats['mean_intrinsic_reward'])
    
    episode_metrics = metrics.end_episode(success=True)
    assert 'coverage' in episode_metrics
    
    print("✅ Component integration test passed")


def main():
    """Run all standalone tests."""
    print("Running Phase 2 standalone tests...\n")
    
    try:
        test_icm_standalone()
        test_gnn_standalone()
        test_graph_builder_standalone()
        test_feature_extractor_standalone()
        test_exploration_metrics_standalone()
        test_config_system_standalone()
        test_integration_without_env()
        
        print("\n🎉 All Phase 2 standalone tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()