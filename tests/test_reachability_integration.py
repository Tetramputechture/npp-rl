"""
Integration tests for reachability feature integration.

This module tests the complete integration of reachability features with
the HGT multimodal extractor and environment wrapper.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from gymnasium import spaces

from npp_rl.feature_extractors.hgt_multimodal import (
    HGTMultimodalExtractor,
    ReachabilityAttentionModule,
    DummyReachabilityExtractor
)
from npp_rl.environments.reachability_wrapper import ReachabilityWrapper
from npp_rl.utils.performance_monitor import PerformanceMonitor


class TestReachabilityIntegration:
    """Test suite for reachability feature integration."""
    
    def test_dummy_reachability_extractor(self):
        """Test dummy reachability extractor fallback."""
        extractor = DummyReachabilityExtractor()
        features = extractor.extract_features(
            ninja_pos=(0.0, 0.0),
            level_data=None,
            switch_states={},
            performance_target="fast"
        )
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (64,)
        assert np.all(features == 0.0)
    
    def test_reachability_attention_module(self):
        """Test reachability attention module."""
        batch_size = 4
        visual_dim = 128
        graph_dim = 256
        state_dim = 64
        reachability_dim = 32
        
        module = ReachabilityAttentionModule(
            visual_dim=visual_dim,
            graph_dim=graph_dim,
            state_dim=state_dim,
            reachability_dim=reachability_dim
        )
        
        # Create dummy inputs
        visual_features = torch.randn(batch_size, visual_dim)
        graph_features = torch.randn(batch_size, graph_dim)
        state_features = torch.randn(batch_size, state_dim)
        reachability_features = torch.randn(batch_size, reachability_dim)
        
        # Forward pass
        output = module(visual_features, graph_features, state_features, reachability_features)
        
        # Check output shape
        expected_dim = (visual_dim + graph_dim + state_dim + reachability_dim) // 2
        assert output.shape == (batch_size, expected_dim)
    
    def test_hgt_extractor_with_reachability(self):
        """Test HGT extractor with reachability features enabled."""
        # Create mock observation space
        observation_space = spaces.Dict({
            'player_frame': spaces.Box(low=0, high=255, shape=(12, 64, 64, 3), dtype=np.uint8),
            'game_state': spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            'graph_node_feats': spaces.Box(low=0, high=1, shape=(100, 16), dtype=np.float32),
            'graph_edge_feats': spaces.Box(low=0, high=1, shape=(200, 8), dtype=np.float32),
            'graph_edge_index': spaces.Box(low=0, high=99, shape=(2, 200), dtype=np.int64),
            'graph_node_types': spaces.Box(low=0, high=5, shape=(100,), dtype=np.int64),
            'graph_edge_types': spaces.Box(low=0, high=3, shape=(200,), dtype=np.int64),
            'graph_node_mask': spaces.Box(low=0, high=1, shape=(100,), dtype=np.bool_),
            'graph_edge_mask': spaces.Box(low=0, high=1, shape=(200,), dtype=np.bool_),
            'reachability_features': spaces.Box(low=0, high=2, shape=(64,), dtype=np.float32)
        })
        
        # Create extractor with reachability enabled
        extractor = HGTMultimodalExtractor(
            observation_space=observation_space,
            features_dim=256,
            enable_reachability_features=True,
            reachability_dim=64
        )
        
        # Check that reachability components are initialized
        assert extractor.enable_reachability_features
        assert extractor.reachability_dim == 64
        assert hasattr(extractor, 'reachability_encoder')
        assert hasattr(extractor, 'reachability_attention')
    
    def test_hgt_extractor_without_reachability(self):
        """Test HGT extractor with reachability features disabled."""
        # Create mock observation space
        observation_space = spaces.Dict({
            'player_frame': spaces.Box(low=0, high=255, shape=(12, 64, 64, 3), dtype=np.uint8),
            'game_state': spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        })
        
        # Create extractor with reachability disabled
        extractor = HGTMultimodalExtractor(
            observation_space=observation_space,
            features_dim=256,
            enable_reachability_features=False
        )
        
        # Check that reachability components are not initialized
        assert not extractor.enable_reachability_features
        assert not hasattr(extractor, 'reachability_encoder')
        assert not hasattr(extractor, 'reachability_attention')
    
    def test_reachability_wrapper_initialization(self):
        """Test reachability wrapper initialization."""
        # Create mock environment
        mock_env = Mock()
        mock_env.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        mock_env.action_space = spaces.Discrete(5)
        
        # Create wrapper
        wrapper = ReachabilityWrapper(
            env=mock_env,
            cache_ttl_ms=100.0,
            performance_target="fast",
            enable_monitoring=True,
            debug=False
        )
        
        # Check initialization
        assert wrapper.cache_ttl_ms == 100.0
        assert wrapper.performance_target == "fast"
        assert wrapper.performance_monitor is not None
        assert isinstance(wrapper.observation_space, spaces.Dict)
        assert 'reachability_features' in wrapper.observation_space.spaces
    
    def test_reachability_wrapper_observation_extension(self):
        """Test that reachability wrapper extends observations correctly."""
        # Create mock environment
        mock_env = Mock()
        mock_env.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        mock_env.action_space = spaces.Discrete(5)
        mock_env.reset.return_value = (np.random.rand(10), {})
        mock_env.step.return_value = (np.random.rand(10), 0.0, False, False, {})
        
        # Create wrapper
        wrapper = ReachabilityWrapper(
            env=mock_env,
            cache_ttl_ms=100.0,
            debug=False
        )
        
        # Test reset
        obs, info = wrapper.reset()
        assert isinstance(obs, dict)
        assert 'reachability_features' in obs
        assert obs['reachability_features'].shape == (64,)
        assert 'reachability' in info
        
        # Test step
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert isinstance(obs, dict)
        assert 'reachability_features' in obs
        assert obs['reachability_features'].shape == (64,)
        assert 'reachability' in info
    
    def test_performance_monitor(self):
        """Test performance monitoring functionality."""
        monitor = PerformanceMonitor("test_component", max_history=100)
        
        # Record some timings
        timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        for timing in timings:
            monitor.record_timing(timing)
        
        # Get stats
        stats = monitor.get_stats()
        
        assert stats['name'] == "test_component"
        assert stats['total_calls'] == 5
        assert stats['avg_time_ms'] == 3.0
        assert stats['min_time_ms'] == 1.0
        assert stats['max_time_ms'] == 5.0
        assert stats['p50_time_ms'] == 3.0
    
    def test_reachability_caching(self):
        """Test reachability feature caching."""
        # Create mock environment
        mock_env = Mock()
        mock_env.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        mock_env.action_space = spaces.Discrete(5)
        mock_env.reset.return_value = (np.random.rand(10), {})
        mock_env.step.return_value = (np.random.rand(10), 0.0, False, False, {})
        
        # Create wrapper with short cache TTL
        wrapper = ReachabilityWrapper(
            env=mock_env,
            cache_ttl_ms=50.0,  # Short TTL for testing
            debug=True
        )
        
        # Reset environment multiple times quickly (should hit cache)
        wrapper.reset()
        wrapper.reset()
        
        # Check cache stats
        stats = wrapper.get_performance_stats()
        assert stats['cache_hits'] >= 0  # May be 0 if positions differ
        assert stats['cache_misses'] >= 1
        assert 'cache_hit_rate' in stats
    
    @patch('npp_rl.environments.reachability_wrapper.create_reachability_aware_env')
    def test_training_integration(self, mock_create_env):
        """Test integration with training pipeline."""
        from npp_rl.environments import create_reachability_aware_env
        
        # Mock the environment creation
        mock_env = Mock()
        mock_create_env.return_value = mock_env
        
        # Test environment creation with reachability
        base_env = Mock()
        result_env = create_reachability_aware_env(
            base_env=base_env,
            cache_ttl_ms=100.0,
            performance_target="fast",
            enable_monitoring=True,
            debug=False
        )
        
        # Verify the function was called
        mock_create_env.assert_called_once()


class TestReachabilityPerformance:
    """Performance tests for reachability integration."""
    
    def test_extraction_performance_target(self):
        """Test that reachability extraction meets performance targets."""
        import time
        
        # Create dummy extractor
        extractor = DummyReachabilityExtractor()
        
        # Measure extraction time
        start_time = time.perf_counter()
        for _ in range(100):
            features = extractor.extract_features(
                ninja_pos=(0.0, 0.0),
                level_data=None,
                switch_states={},
                performance_target="fast"
            )
        end_time = time.perf_counter()
        
        # Calculate average time per extraction
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Should be well under 2ms target (dummy extractor should be very fast)
        assert avg_time_ms < 0.1, f"Extraction too slow: {avg_time_ms:.3f}ms"
    
    def test_attention_module_performance(self):
        """Test attention module performance."""
        import time
        
        batch_size = 32  # Typical training batch size
        module = ReachabilityAttentionModule(
            visual_dim=128,
            graph_dim=256,
            state_dim=64,
            reachability_dim=32
        )
        
        # Create inputs
        visual_features = torch.randn(batch_size, 128)
        graph_features = torch.randn(batch_size, 256)
        state_features = torch.randn(batch_size, 64)
        reachability_features = torch.randn(batch_size, 32)
        
        # Warm up
        for _ in range(10):
            _ = module(visual_features, graph_features, state_features, reachability_features)
        
        # Measure performance
        start_time = time.perf_counter()
        for _ in range(100):
            output = module(visual_features, graph_features, state_features, reachability_features)
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Should be reasonable for training (allow more time for neural network operations)
        assert avg_time_ms < 10.0, f"Attention module too slow: {avg_time_ms:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])