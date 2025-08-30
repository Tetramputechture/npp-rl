"""
Test suite for Task 3.1: Hybrid CNN-GNN Architecture Enhancement

This module tests the multimodal fusion architecture with
cross-modal attention, spatial attention, and transformer-based fusion.

Tests cover:
- Multimodal feature extractor functionality
- Cross-modal attention mechanisms
- Graph-informed spatial attention
- Transformer-based fusion layers
- Integration with existing PPO pipeline
- Backward compatibility
"""

import pytest
import torch
import numpy as np
from gymnasium.spaces import Dict as SpacesDict, Box

from npp_rl.feature_extractors.multimodal import (
    MultimodalGraphExtractor,
    create_hgt_multimodal_extractor
)
from npp_rl.models.spatial_attention import (
    SpatialAttentionModule,
    GraphSpatialGuidance,
    MultiScaleSpatialAttention
)


class TestSpatialAttentionModule:
    """Test spatial attention components."""
    
    def test_graph_spatial_guidance_initialization(self):
        """Test GraphSpatialGuidance initialization."""
        guidance = GraphSpatialGuidance(
            graph_dim=256,
            spatial_height=16,
            spatial_width=16,
            guidance_dim=64,
            num_attention_heads=4
        )
        
        assert guidance.graph_dim == 256
        assert guidance.spatial_height == 16
        assert guidance.spatial_width == 16
        assert guidance.guidance_dim == 64
        assert guidance.num_heads == 4
        
        # Check spatial positions buffer
        assert guidance.spatial_positions.shape == (256, 2)  # 16*16, 2
    
    def test_graph_spatial_guidance_forward(self):
        """Test GraphSpatialGuidance forward pass."""
        batch_size = 2
        num_nodes = 10
        graph_dim = 128
        
        guidance = GraphSpatialGuidance(
            graph_dim=graph_dim,
            spatial_height=8,
            spatial_width=8,
            guidance_dim=64
        )
        
        # Create mock graph features
        graph_features = torch.randn(batch_size, num_nodes, graph_dim)
        
        # Forward pass
        attention_map = guidance(graph_features)
        
        # Check output shape
        assert attention_map.shape == (batch_size, 8, 8)
        
        # Check attention values are in valid range
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)
    
    def test_spatial_attention_module_initialization(self):
        """Test SpatialAttentionModule initialization."""
        spatial_attention = SpatialAttentionModule(
            graph_dim=256,
            visual_dim=512,
            spatial_height=16,
            spatial_width=16,
            num_attention_heads=8
        )
        
        assert spatial_attention.graph_dim == 256
        assert spatial_attention.visual_dim == 512
        assert spatial_attention.spatial_height == 16
        assert spatial_attention.spatial_width == 16
        assert spatial_attention.num_heads == 8
    
    def test_spatial_attention_module_forward(self):
        """Test SpatialAttentionModule forward pass."""
        batch_size = 2
        num_nodes = 15
        graph_dim = 256
        visual_dim = 512
        
        spatial_attention = SpatialAttentionModule(
            graph_dim=graph_dim,
            visual_dim=visual_dim,
            spatial_height=8,
            spatial_width=8
        )
        
        # Create mock inputs
        visual_features = torch.randn(batch_size, visual_dim)
        graph_features = torch.randn(batch_size, num_nodes, graph_dim)
        
        # Forward pass
        enhanced_visual, attention_map = spatial_attention(
            visual_features, graph_features
        )
        
        # Check output shapes
        assert enhanced_visual.shape == (batch_size, visual_dim)
        assert attention_map.shape == (batch_size, 8, 8)
        
        # Check attention values
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)
    
    def test_multi_scale_spatial_attention(self):
        """Test MultiScaleSpatialAttention."""
        batch_size = 2
        num_nodes = 12
        graph_dim = 256
        visual_dim = 512
        scales = [4, 8, 16]
        
        multi_scale_attention = MultiScaleSpatialAttention(
            graph_dim=graph_dim,
            visual_dim=visual_dim,
            scales=scales
        )
        
        # Create mock inputs
        visual_features = torch.randn(batch_size, visual_dim)
        graph_features = torch.randn(batch_size, num_nodes, graph_dim)
        
        # Forward pass
        enhanced_features, attention_maps = multi_scale_attention(
            visual_features, graph_features
        )
        
        # Check output shape
        assert enhanced_features.shape == (batch_size, visual_dim)
        
        # Check attention maps for each scale
        assert len(attention_maps) == len(scales)
        for i, scale in enumerate(scales):
            key = f'scale_{scale}'
            assert key in attention_maps
            assert attention_maps[key].shape == (batch_size, scale, scale)


class TestEnhancedMultimodalExtractor:
    """Test enhanced multimodal feature extractor."""
    
    def create_mock_observation_space(self, include_graph=True):
        """Create mock observation space for testing."""
        spaces = {
            'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
            'global_view': Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
            'game_state': Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        }
        
        if include_graph:
            spaces.update({
                'graph_node_feats': Box(low=-1, high=1, shape=(100, 85), dtype=np.float32),
                'graph_edge_index': Box(low=0, high=99, shape=(2, 500), dtype=np.int64),
                'graph_edge_feats': Box(low=-1, high=1, shape=(500, 16), dtype=np.float32),
                'graph_node_mask': Box(low=0, high=1, shape=(100,), dtype=np.bool_),
                'graph_edge_mask': Box(low=0, high=1, shape=(500,), dtype=np.bool_)
            })
        
        return SpacesDict(spaces)
    
    def create_mock_observations(self, batch_size=2, include_graph=True):
        """Create mock observations for testing."""
        obs = {
            'player_frame': torch.randint(0, 256, (batch_size, 64, 64, 4), dtype=torch.uint8),
            'global_view': torch.randint(0, 256, (batch_size, 32, 32, 3), dtype=torch.uint8),
            'game_state': torch.randn(batch_size, 20)
        }
        
        if include_graph:
            obs.update({
                'graph_node_feats': torch.randn(batch_size, 100, 85),
                'graph_edge_index': torch.randint(0, 100, (batch_size, 2, 500)),
                'graph_edge_feats': torch.randn(batch_size, 500, 16),
                'graph_node_mask': torch.ones(batch_size, 100, dtype=torch.bool),
                'graph_edge_mask': torch.ones(batch_size, 500, dtype=torch.bool)
            })
        
        return obs
    
    def test_enhanced_extractor_initialization(self):
        """Test enhanced multimodal extractor initialization."""
        observation_space = self.create_mock_observation_space()
        
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=512,
            use_graph_obs=True,
            use_cross_modal_attention=True,
            use_spatial_attention=True,
            num_attention_heads=8
        )
        
        # Check attributes
        assert extractor.use_cross_modal_attention
        assert extractor.use_spatial_attention
        assert extractor.num_attention_heads == 8
        assert extractor.embed_dim == 512
        
        # Check components exist
        assert hasattr(extractor, 'cross_modal_attention')
        assert hasattr(extractor, 'graph_visual_fusion')
        assert hasattr(extractor, 'spatial_attention')
        assert hasattr(extractor, 'visual_projection')
        assert hasattr(extractor, 'graph_projection')
        assert hasattr(extractor, 'symbolic_projection')
    
    def test_enhanced_extractor_forward_with_attention(self):
        """Test enhanced extractor forward pass with attention mechanisms."""
        observation_space = self.create_mock_observation_space()
        batch_size = 2
        
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=256,
            use_graph_obs=True,
            use_cross_modal_attention=True,
            use_spatial_attention=True,
            gnn_output_dim=128
        )
        
        # Create mock observations
        observations = self.create_mock_observations(batch_size)
        
        # Create a mock graph encoder function that returns the right shape
        def mock_graph_encoder_fn(graph_obs):
            return torch.randn(batch_size, 128)
        
        # Use monkey patching to replace the forward method
        original_forward = extractor.graph_encoder.forward
        extractor.graph_encoder.forward = lambda x: mock_graph_encoder_fn(x)
        
        try:
            # Forward pass
            output = extractor(observations)
            
            # Check output shape
            assert output.shape == (batch_size, 256)
            
        finally:
            # Restore original forward method
            extractor.graph_encoder.forward = original_forward
    
    def test_enhanced_extractor_forward_without_attention(self):
        """Test enhanced extractor forward pass without attention mechanisms."""
        observation_space = self.create_mock_observation_space()
        batch_size = 2
        
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=256,
            use_graph_obs=True,
            use_cross_modal_attention=False,
            use_spatial_attention=False,
            gnn_output_dim=128
        )
        
        # Create mock observations
        observations = self.create_mock_observations(batch_size)
        
        # Create a mock graph encoder function
        def mock_graph_encoder_fn(graph_obs):
            return torch.randn(batch_size, 128)
        
        # Use monkey patching to replace the forward method
        original_forward = extractor.graph_encoder.forward
        extractor.graph_encoder.forward = lambda x: mock_graph_encoder_fn(x)
        
        try:
            # Forward pass
            output = extractor(observations)
            
            # Check output shape
            assert output.shape == (batch_size, 256)
            
        finally:
            # Restore original forward method
            extractor.graph_encoder.forward = original_forward
    
    def test_enhanced_extractor_visual_only(self):
        """Test enhanced extractor with visual observations only."""
        observation_space = self.create_mock_observation_space(include_graph=False)
        batch_size = 2
        
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=256,
            use_graph_obs=False,
            use_cross_modal_attention=False,  # Disable cross-modal attention for visual-only
            use_spatial_attention=False
        )
        
        # Create mock observations without graph
        observations = self.create_mock_observations(batch_size, include_graph=False)
        
        # Forward pass
        output = extractor(observations)
        
        # Check output shape
        assert output.shape == (batch_size, 256)
    
    def test_factory_function_hgt_enhanced(self):
        """Test HGT factory function with enhanced features."""
        observation_space = self.create_mock_observation_space()
        
        extractor = create_hgt_multimodal_extractor(
            observation_space=observation_space,
            features_dim=512,
            use_cross_modal_attention=True,
            use_spatial_attention=True,
            num_attention_heads=8
        )
        
        # Check it's the right type
        assert isinstance(extractor, MultimodalGraphExtractor)
        assert extractor.use_hgt
        assert extractor.use_cross_modal_attention
        assert extractor.use_spatial_attention
        assert extractor.num_attention_heads == 8
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing code."""
        observation_space = self.create_mock_observation_space()
        
        # Test that old interface still works
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=256,
            use_graph_obs=True,
            gnn_output_dim=128
        )
        
        # Should work with default enhanced features
        batch_size = 2
        observations = self.create_mock_observations(batch_size)
        
        # Create a mock graph encoder function
        def mock_graph_encoder_fn(graph_obs):
            return torch.randn(batch_size, 128)
        
        # Use monkey patching to replace the forward method
        original_forward = extractor.graph_encoder.forward
        extractor.graph_encoder.forward = lambda x: mock_graph_encoder_fn(x)
        
        try:
            output = extractor(observations)
            assert output.shape == (batch_size, 256)
            
        finally:
            # Restore original forward method
            extractor.graph_encoder.forward = original_forward


class TestIntegrationWithPPO:
    """Test integration with PPO training pipeline."""
    
    def test_extractor_with_ppo_interface(self):
        """Test that enhanced extractor works with PPO interface."""
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        
        observation_space = SpacesDict({
            'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
            'graph_node_feats': Box(low=-1, high=1, shape=(50, 85), dtype=np.float32),
            'graph_edge_index': Box(low=0, high=49, shape=(2, 200), dtype=np.int64),
            'graph_edge_feats': Box(low=-1, high=1, shape=(200, 16), dtype=np.float32),
            'graph_node_mask': Box(low=0, high=1, shape=(50,), dtype=np.bool_),
            'graph_edge_mask': Box(low=0, high=1, shape=(200,), dtype=np.bool_)
        })
        
        extractor = create_hgt_multimodal_extractor(
            observation_space=observation_space,
            features_dim=256,
            use_cross_modal_attention=True,
            use_spatial_attention=True
        )
        
        # Check it inherits from BaseFeaturesExtractor
        assert isinstance(extractor, BaseFeaturesExtractor)
        assert extractor.features_dim == 256
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through enhanced architecture."""
        observation_space = SpacesDict({
            'player_frame': Box(low=0, high=255, shape=(32, 32, 4), dtype=np.uint8),
            'graph_node_feats': Box(low=-1, high=1, shape=(20, 85), dtype=np.float32),
            'graph_edge_index': Box(low=0, high=19, shape=(2, 50), dtype=np.int64),
            'graph_edge_feats': Box(low=-1, high=1, shape=(50, 16), dtype=np.float32),
            'graph_node_mask': Box(low=0, high=1, shape=(20,), dtype=np.bool_),
            'graph_edge_mask': Box(low=0, high=1, shape=(50,), dtype=np.bool_)
        })
        
        extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=128,
            use_graph_obs=True,
            use_cross_modal_attention=True,
            use_spatial_attention=True,
            gnn_output_dim=64
        )
        
        # Create mock observations
        observations = {
            'player_frame': torch.randint(0, 256, (1, 32, 32, 4), dtype=torch.uint8),
            'graph_node_feats': torch.randn(1, 20, 85, requires_grad=True),
            'graph_edge_index': torch.randint(0, 20, (1, 2, 50)),
            'graph_edge_feats': torch.randn(1, 50, 16, requires_grad=True),
            'graph_node_mask': torch.ones(1, 20, dtype=torch.bool),
            'graph_edge_mask': torch.ones(1, 50, dtype=torch.bool)
        }
        
        # Create a mock graph encoder function that returns gradients
        def mock_graph_encoder_fn(graph_obs):
            return torch.randn(1, 64, requires_grad=True)
        
        # Use monkey patching to replace the forward method
        original_forward = extractor.graph_encoder.forward
        extractor.graph_encoder.forward = lambda x: mock_graph_encoder_fn(x)
        
        try:
            # Forward pass
            output = extractor(observations)
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist
            assert output.grad_fn is not None
            
            # Check that some parameters have gradients (not all may be used)
            has_gradients = False
            for param in extractor.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break
            assert has_gradients, "No parameters received gradients"
            
        finally:
            # Restore original forward method
            extractor.graph_encoder.forward = original_forward


if __name__ == '__main__':
    pytest.main([__file__, '-v'])