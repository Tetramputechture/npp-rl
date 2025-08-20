"""
Integration test for Phase 2 components.

This script tests the integration of all Phase 2 components to ensure
they work together correctly before running full training.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from npp_rl.config.phase2_config import Phase2Config, create_full_phase2_config
from npp_rl.data.bc_dataset import create_mock_replay_data, create_bc_dataloader
from npp_rl.models.feature_extractors import create_feature_extractor
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from npp_rl.eval.exploration_metrics import ExplorationMetrics
from nclone.nclone_environments.basic_level_no_gold.graph_observation import create_graph_enhanced_env


def test_environment_creation():
    """Test environment creation with graph observations."""
    print("Testing environment creation...")
    
    # Test without graph observations
    env = create_graph_enhanced_env(use_graph_obs=False)
    obs, info = env.reset()
    
    assert isinstance(obs, dict)
    assert 'player_frame' in obs
    assert 'global_view' in obs
    assert 'game_state' in obs
    
    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.close()
    
    # Test with graph observations
    env = create_graph_enhanced_env(use_graph_obs=True)
    obs, info = env.reset()
    
    assert isinstance(obs, dict)
    assert 'player_frame' in obs
    assert 'global_view' in obs
    assert 'game_state' in obs
    assert 'graph_node_feats' in obs
    assert 'graph_edge_index' in obs
    assert 'graph_edge_feats' in obs
    assert 'graph_node_mask' in obs
    assert 'graph_edge_mask' in obs
    
    env.close()
    
    print("✅ Environment creation test passed")


def test_feature_extractor_integration():
    """Test feature extractor with different configurations."""
    print("Testing feature extractor integration...")
    
    # Create environment to get observation space
    env = create_graph_enhanced_env(use_graph_obs=True)
    observation_space = env.observation_space
    env.close()
    
    # Test multimodal extractor with graph observations
    extractor = create_feature_extractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=True
    )
    
    # Create mock observation
    batch_size = 2
    obs = {}
    for key, space in observation_space.spaces.items():
        if key in ['player_frame', 'global_view']:
            obs[key] = torch.randint(0, 256, (batch_size,) + space.shape, dtype=torch.uint8)
        elif key == 'game_state':
            obs[key] = torch.randn(batch_size, *space.shape)
        elif key.startswith('graph_'):
            if key == 'graph_edge_index':
                obs[key] = torch.randint(0, space.high[0], (batch_size,) + space.shape, dtype=torch.int32)
            else:
                obs[key] = torch.randn(batch_size, *space.shape)
    
    # Test forward pass
    features = extractor(obs)
    assert features.shape == (batch_size, 512)
    
    print("✅ Feature extractor integration test passed")


def test_icm_integration():
    """Test ICM integration with feature extractor."""
    print("Testing ICM integration...")
    
    # Create environment and feature extractor
    env = create_graph_enhanced_env(use_graph_obs=False)
    observation_space = env.observation_space
    
    extractor = create_feature_extractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=False
    )
    
    # Create ICM
    icm = ICMNetwork(feature_dim=512, action_dim=6)
    trainer = ICMTrainer(icm)
    
    # Create intrinsic reward wrapper
    wrapped_env = IntrinsicRewardWrapper(
        env=env,
        icm_trainer=trainer,
        alpha=0.1
    )
    
    # Mock policy for feature extraction
    class MockPolicy:
        def __init__(self, extractor):
            self.features_extractor = extractor
    
    policy = MockPolicy(extractor)
    wrapped_env.set_policy(policy)
    
    # Test episode
    obs, info = wrapped_env.reset()
    
    for _ in range(10):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Check that intrinsic reward info is added
        assert 'r_int' in info
        assert 'r_ext' in info
        assert 'r_total' in info
        
        if terminated or truncated:
            break
    
    wrapped_env.close()
    
    print("✅ ICM integration test passed")


def test_bc_dataset():
    """Test BC dataset creation and loading."""
    print("Testing BC dataset...")
    
    # Create temporary directory for mock data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock replay data
        create_mock_replay_data(
            output_dir=temp_dir,
            num_episodes=10,
            episode_length=50
        )
        
        # Create environment to get observation space
        env = create_graph_enhanced_env(use_graph_obs=False)
        observation_space = env.observation_space.spaces
        action_space = env.action_space
        env.close()
        
        # Create dataloader
        dataloader = create_bc_dataloader(
            data_dir=temp_dir,
            observation_space=observation_space,
            action_space=action_space,
            batch_size=8
        )
        
        # Test loading batches
        for batch_idx, (observations, actions) in enumerate(dataloader):
            assert isinstance(observations, dict)
            assert 'player_frame' in observations
            assert actions.shape[0] <= 8  # Batch size
            
            if batch_idx >= 2:  # Test a few batches
                break
        
        # Test dataset statistics
        stats = dataloader.dataset.get_statistics()
        assert 'num_episodes' in stats
        assert 'num_transitions' in stats
        assert stats['num_episodes'] == 10
    
    print("✅ BC dataset test passed")


def test_exploration_metrics():
    """Test exploration metrics tracking."""
    print("Testing exploration metrics...")
    
    metrics = ExplorationMetrics()
    
    # Simulate episode
    metrics.reset_episode()
    
    # Simulate agent movement
    positions = [
        (100, 100), (124, 100), (148, 100),  # Moving right
        (148, 124), (148, 148),              # Moving down
        (124, 148), (100, 148)               # Moving left
    ]
    
    for pos in positions:
        metrics.update_step(pos, intrinsic_reward=0.1)
    
    # End episode
    episode_metrics = metrics.end_episode(success=True)
    
    assert 'coverage' in episode_metrics
    assert 'visitation_entropy' in episode_metrics
    assert 'mean_intrinsic_reward' in episode_metrics
    assert episode_metrics['success'] == 1.0
    
    # Test rolling metrics
    rolling = metrics.get_rolling_metrics()
    assert 'rolling_coverage' in rolling
    
    print("✅ Exploration metrics test passed")


def test_config_system():
    """Test configuration system."""
    print("Testing configuration system...")
    
    # Create config
    config = create_full_phase2_config()
    
    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save(f.name)
        
        # Test loading
        loaded_config = Phase2Config.load(f.name)
        
        assert loaded_config.icm.enabled == config.icm.enabled
        assert loaded_config.graph.enabled == config.graph.enabled
        assert loaded_config.bc.enabled == config.bc.enabled
        
        # Cleanup
        Path(f.name).unlink()
    
    # Test validation
    from npp_rl.config.phase2_config import validate_config
    messages = validate_config(config)
    # Should have some warnings about missing dataset
    
    print("✅ Configuration system test passed")


def test_full_integration():
    """Test full integration of all components."""
    print("Testing full integration...")
    
    # Create config
    config = Phase2Config()
    config.icm.enabled = True
    config.graph.enabled = False  # Disable for simpler test
    config.bc.enabled = False
    config.total_timesteps = 100  # Short test
    
    # Create environment
    env = create_graph_enhanced_env(use_graph_obs=config.graph.enabled)
    
    # Create feature extractor
    extractor = create_feature_extractor(
        observation_space=env.observation_space,
        features_dim=512,
        use_graph_obs=config.graph.enabled
    )
    
    # Create ICM if enabled
    icm_trainer = None
    if config.icm.enabled:
        icm = ICMNetwork(feature_dim=512, action_dim=6)
        icm_trainer = ICMTrainer(icm)
    
    # Wrap environment
    if icm_trainer:
        env = IntrinsicRewardWrapper(env, icm_trainer, alpha=0.1)
        
        # Mock policy
        class MockPolicy:
            def __init__(self, extractor):
                self.features_extractor = extractor
        
        env.set_policy(MockPolicy(extractor))
    
    # Create exploration metrics
    metrics = ExplorationMetrics()
    
    # Run short episode
    obs, info = env.reset()
    metrics.reset_episode()
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update metrics (would need position from info in real implementation)
        metrics.update_step((100 + step, 100), info.get('r_int', 0.0))
        
        if terminated or truncated:
            break
    
    # End episode
    episode_metrics = metrics.end_episode(success=info.get('success', False))
    
    env.close()
    
    print("✅ Full integration test passed")


def main():
    """Run all integration tests."""
    print("Running Phase 2 integration tests...\n")
    
    try:
        test_environment_creation()
        test_feature_extractor_integration()
        test_icm_integration()
        test_bc_dataset()
        test_exploration_metrics()
        test_config_system()
        test_full_integration()
        
        print("\n🎉 All Phase 2 integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()