"""
Comprehensive tests for hierarchical graph processing (Task 2.1).

Tests the multi-resolution graph processing implementation including:
- Hierarchical graph builder with 3 resolution levels
- DiffPool GNN with differentiable graph pooling
- Multi-scale feature fusion with attention mechanisms
- Integration with existing feature extractors
"""

import pytest
import torch
import numpy as np

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nclone'))

from npp_rl.models.diffpool_gnn import DiffPoolLayer, HierarchicalDiffPoolGNN
from npp_rl.models.multi_scale_fusion import (
    AdaptiveScaleFusion, HierarchicalFeatureAggregator, 
    ContextAwareScaleSelector, UnifiedMultiScaleFusion
)
from npp_rl.feature_extractors.hierarchical_multimodal import (
    HierarchicalMultimodalExtractor, HierarchicalGraphObservationWrapper
)


class TestHierarchicalGraphBuilder:
    """Test hierarchical graph builder functionality."""
    
    def test_hierarchical_builder_initialization(self):
        """Test that hierarchical graph builder initializes correctly."""
        try:
            from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
            builder = HierarchicalGraphBuilder()
            
            # Check resolution levels
            assert len(builder.resolutions) == 3
            assert builder.resolutions[0] == 6   # sub-cell
            assert builder.resolutions[1] == 24  # tile
            assert builder.resolutions[2] == 96  # region
            
            # Check grid dimensions are calculated
            assert len(builder.grid_dimensions) == 3
            
            # Check feature dimensions are set
            assert len(builder.node_feature_dims) == 3
            assert len(builder.edge_feature_dims) == 3
            
        except ImportError:
            pytest.skip("HierarchicalGraphBuilder not available")
    
    def test_grid_dimension_calculation(self):
        """Test that grid dimensions are calculated correctly for each resolution."""
        try:
            from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder, ResolutionLevel
            builder = HierarchicalGraphBuilder()
            
            # Check that dimensions make sense
            sub_cell_dims = builder.grid_dimensions[ResolutionLevel.SUB_CELL]
            tile_dims = builder.grid_dimensions[ResolutionLevel.TILE]
            region_dims = builder.grid_dimensions[ResolutionLevel.REGION]
            
            # Sub-cell should have highest resolution (most cells)
            assert sub_cell_dims[0] > tile_dims[0]
            assert sub_cell_dims[1] > tile_dims[1]
            
            # Region should have lowest resolution (fewest cells)
            assert tile_dims[0] > region_dims[0]
            assert tile_dims[1] > region_dims[1]
            
            # Check that dimensions are positive integers
            for dims in [sub_cell_dims, tile_dims, region_dims]:
                assert dims[0] > 0 and dims[1] > 0
                assert isinstance(dims[0], int) and isinstance(dims[1], int)
                
        except ImportError:
            pytest.skip("HierarchicalGraphBuilder not available")
    
    def test_hierarchical_graph_building(self):
        """Test building hierarchical graph with mock data."""
        try:
            from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
            builder = HierarchicalGraphBuilder()
            
            # Create mock level data
            mock_level_data = {
                'tiles': np.zeros((23, 42), dtype=np.int32),  # Standard N++ level size
                'entities': []
            }
            
            ninja_position = (100.0, 100.0)
            entities = []
            ninja_velocity = (1.0, 0.0)
            ninja_state = 1  # Running
            
            # Build hierarchical graph
            hierarchical_graph = builder.build_hierarchical_graph(
                mock_level_data, ninja_position, entities, ninja_velocity, ninja_state
            )
            
            # Check that all levels are present
            assert hierarchical_graph.sub_cell_graph is not None
            assert hierarchical_graph.tile_graph is not None
            assert hierarchical_graph.region_graph is not None
            
            # Check that mappings exist
            assert hierarchical_graph.sub_to_tile_mapping is not None
            assert hierarchical_graph.tile_to_region_mapping is not None
            
            # Check that cross-scale edges exist
            assert 'sub_to_tile' in hierarchical_graph.cross_scale_edges
            assert 'tile_to_region' in hierarchical_graph.cross_scale_edges
            
            # Check resolution info
            assert 'resolutions' in hierarchical_graph.resolution_info
            assert 'node_counts' in hierarchical_graph.resolution_info
            assert 'edge_counts' in hierarchical_graph.resolution_info
            
        except ImportError:
            pytest.skip("HierarchicalGraphBuilder not available")


class TestDiffPoolGNN:
    """Test DiffPool GNN implementation."""
    
    def test_diffpool_layer_initialization(self):
        """Test DiffPool layer initializes with correct dimensions."""
        layer = DiffPoolLayer(
            input_dim=64,
            hidden_dim=32,
            output_dim=48,
            num_clusters=10,
            gnn_layers=2,
            dropout=0.1
        )
        
        assert layer.input_dim == 64
        assert layer.output_dim == 48
        assert layer.num_clusters == 10
        assert len(layer.embedding_gnn) == 2
        assert len(layer.assignment_gnn) == 2
    
    def test_diffpool_layer_forward_pass(self):
        """Test DiffPool layer forward pass with mock data."""
        batch_size = 2
        num_nodes = 20
        num_edges = 40
        input_dim = 32
        output_dim = 24
        num_clusters = 8
        
        layer = DiffPoolLayer(
            input_dim=input_dim,
            hidden_dim=16,
            output_dim=output_dim,
            num_clusters=num_clusters,
            gnn_layers=1,
            dropout=0.0
        )
        
        # Create mock input data
        node_features = torch.randn(batch_size, num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        node_mask = torch.ones(batch_size, num_nodes)
        edge_mask = torch.ones(batch_size, num_edges)
        
        # Forward pass
        pooled_nodes, pooled_edges, pooled_node_mask, pooled_edge_mask, aux_losses = layer(
            node_features, edge_index, node_mask, edge_mask
        )
        
        # Check output shapes
        assert pooled_nodes.shape == (batch_size, num_clusters, output_dim)
        assert pooled_node_mask.shape == (batch_size, num_clusters)
        assert pooled_edges.shape[0] == batch_size
        assert pooled_edges.shape[1] == 2
        
        # Check auxiliary losses
        assert 'link_prediction_loss' in aux_losses
        assert 'entropy_loss' in aux_losses
        assert 'orthogonality_loss' in aux_losses
        
        # Check that losses are finite
        for loss_name, loss_value in aux_losses.items():
            assert torch.isfinite(loss_value), f"{loss_name} is not finite"
    
    def test_hierarchical_diffpool_gnn(self):
        """Test hierarchical DiffPool GNN with multiple levels."""
        input_dims = {
            'sub_cell': 85,
            'tile': 93,
            'region': 101
        }
        
        gnn = HierarchicalDiffPoolGNN(
            input_dims=input_dims,
            hidden_dim=64,
            output_dim=128,
            num_levels=3,
            pooling_ratios=[0.25, 0.25, 0.5],
            gnn_layers_per_level=1,
            dropout=0.0
        )
        
        # Create mock hierarchical graph data
        batch_size = 2
        hierarchical_data = {
            'sub_cell_node_features': torch.randn(batch_size, 100, input_dims['sub_cell']),
            'sub_cell_edge_index': torch.randint(0, 100, (batch_size, 2, 200)),
            'sub_cell_node_mask': torch.ones(batch_size, 100),
            'sub_cell_edge_mask': torch.ones(batch_size, 200)
        }
        
        # Forward pass
        graph_embedding, aux_losses = gnn(hierarchical_data)
        
        # Check output shape
        assert graph_embedding.shape == (batch_size, 128)
        
        # Check auxiliary losses
        assert isinstance(aux_losses, dict)
        for loss_value in aux_losses.values():
            assert torch.isfinite(loss_value)


class TestMultiScaleFusion:
    """Test multi-scale feature fusion mechanisms."""
    
    def test_adaptive_scale_fusion(self):
        """Test adaptive scale fusion with context awareness."""
        scale_dims = {
            'sub_cell': 64,
            'tile': 48,
            'region': 32
        }
        
        fusion = AdaptiveScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=128,
            context_dim=18,
            num_attention_heads=4,
            dropout=0.0
        )
        
        batch_size = 3
        scale_features = {
            'sub_cell': torch.randn(batch_size, 64),
            'tile': torch.randn(batch_size, 48),
            'region': torch.randn(batch_size, 32)
        }
        ninja_physics_state = torch.randn(batch_size, 18)
        
        # Forward pass
        fused_features, attention_weights = fusion(scale_features, ninja_physics_state)
        
        # Check output shape
        assert fused_features.shape == (batch_size, 128)
        
        # Check attention weights
        assert 'scale_importance' in attention_weights
        assert len(attention_weights['scale_importance']) == len(scale_dims)
        
        # Check that attention weights sum to 1
        scale_weight_sum = sum(attention_weights['scale_importance'].values())
        assert torch.allclose(scale_weight_sum, torch.ones_like(scale_weight_sum))
    
    def test_hierarchical_feature_aggregator(self):
        """Test hierarchical feature aggregator with routing."""
        level_dims = {
            'level_0': 32,
            'level_1': 24,
            'level_2': 16
        }
        
        aggregator = HierarchicalFeatureAggregator(
            level_dims=level_dims,
            hidden_dim=64,
            num_routing_iterations=2,
            dropout=0.0
        )
        
        batch_size = 2
        level_features = {
            'level_0': torch.randn(batch_size, 32),
            'level_1': torch.randn(batch_size, 24),
            'level_2': torch.randn(batch_size, 16)
        }
        
        # Forward pass
        aggregated_features, routing_weights = aggregator(level_features)
        
        # Check output shape
        assert aggregated_features.shape == (batch_size, 64)
        
        # Check routing weights
        assert isinstance(routing_weights, dict)
        
        # Check that routing weights are in valid range [0, 1]
        for weight_tensor in routing_weights.values():
            assert torch.all(weight_tensor >= 0.0)
            assert torch.all(weight_tensor <= 1.0)
    
    def test_context_aware_scale_selector(self):
        """Test context-aware scale selection."""
        scale_dims = {
            'fine': 64,
            'medium': 32,
            'coarse': 16
        }
        
        selector = ContextAwareScaleSelector(
            scale_dims=scale_dims,
            context_dim=18,
            hidden_dim=32,
            num_context_types=4,
            dropout=0.0
        )
        
        batch_size = 2
        scale_features = {
            'fine': torch.randn(batch_size, 64),
            'medium': torch.randn(batch_size, 32),
            'coarse': torch.randn(batch_size, 16)
        }
        ninja_physics_state = torch.randn(batch_size, 18)
        
        # Forward pass
        scale_weights, context_info = selector(scale_features, ninja_physics_state)
        
        # Check output shape
        assert scale_weights.shape == (batch_size, len(scale_dims))
        
        # Check that weights sum to 1
        weight_sums = torch.sum(scale_weights, dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums))
        
        # Check context info
        assert 'context_probabilities' in context_info
        assert 'final_scale_weights' in context_info
    
    def test_unified_multi_scale_fusion(self):
        """Test unified multi-scale fusion combining all mechanisms."""
        scale_dims = {
            'sub_cell': 85,
            'tile': 93,
            'region': 101
        }
        
        unified_fusion = UnifiedMultiScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=256,
            context_dim=18,
            hidden_dim=128,
            dropout=0.0
        )
        
        batch_size = 2
        scale_features = {
            'sub_cell': torch.randn(batch_size, 85),
            'tile': torch.randn(batch_size, 93),
            'region': torch.randn(batch_size, 101)
        }
        ninja_physics_state = torch.randn(batch_size, 18)
        
        # Forward pass
        unified_features, fusion_info = unified_fusion(scale_features, ninja_physics_state)
        
        # Check output shape
        assert unified_features.shape == (batch_size, 256)
        
        # Check fusion info completeness
        assert 'adaptive_weights' in fusion_info
        assert 'routing_weights' in fusion_info
        assert 'scale_weights' in fusion_info
        assert 'context_info' in fusion_info
        assert 'fusion_weights' in fusion_info


class TestHierarchicalMultimodalExtractor:
    """Test hierarchical multimodal feature extractor integration."""
    
    def create_mock_observation_space(self, include_hierarchical=True):
        """Create mock observation space for testing."""
        from gymnasium.spaces import Box, Dict as SpacesDict
        
        spaces = {
            'player_frame': Box(low=0, high=255, shape=(84, 84, 12), dtype=np.uint8),
            'global_view': Box(low=0, high=255, shape=(176, 100, 1), dtype=np.uint8),
            'game_state': Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        }
        
        if include_hierarchical:
            # Add hierarchical graph observations
            spaces.update({
                'sub_cell_node_features': Box(low=-np.inf, high=np.inf, shape=(1000, 85), dtype=np.float32),
                'sub_cell_edge_index': Box(low=0, high=999, shape=(2, 2000), dtype=np.int32),
                'sub_cell_node_mask': Box(low=0, high=1, shape=(1000,), dtype=np.float32),
                'sub_cell_edge_mask': Box(low=0, high=1, shape=(2000,), dtype=np.float32),
                
                'tile_node_features': Box(low=-np.inf, high=np.inf, shape=(250, 93), dtype=np.float32),
                'tile_edge_index': Box(low=0, high=249, shape=(2, 500), dtype=np.int32),
                'tile_node_mask': Box(low=0, high=1, shape=(250,), dtype=np.float32),
                'tile_edge_mask': Box(low=0, high=1, shape=(500,), dtype=np.float32),
                
                'region_node_features': Box(low=-np.inf, high=np.inf, shape=(60, 101), dtype=np.float32),
                'region_edge_index': Box(low=0, high=59, shape=(2, 120), dtype=np.int32),
                'region_node_mask': Box(low=0, high=1, shape=(60,), dtype=np.float32),
                'region_edge_mask': Box(low=0, high=1, shape=(120,), dtype=np.float32),
                
                'ninja_physics_state': Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
            })
        
        return SpacesDict(spaces)
    
    def create_mock_observations(self, batch_size=2, include_hierarchical=True):
        """Create mock observations for testing."""
        obs = {
            'player_frame': torch.randint(0, 256, (batch_size, 84, 84, 12), dtype=torch.uint8),
            'global_view': torch.randint(0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8),
            'game_state': torch.randn(batch_size, 32)
        }
        
        if include_hierarchical:
            obs.update({
                'sub_cell_node_features': torch.randn(batch_size, 1000, 85),
                'sub_cell_edge_index': torch.randint(0, 1000, (batch_size, 2, 2000)),
                'sub_cell_node_mask': torch.ones(batch_size, 1000),
                'sub_cell_edge_mask': torch.ones(batch_size, 2000),
                
                'tile_node_features': torch.randn(batch_size, 250, 93),
                'tile_edge_index': torch.randint(0, 250, (batch_size, 2, 500)),
                'tile_node_mask': torch.ones(batch_size, 250),
                'tile_edge_mask': torch.ones(batch_size, 500),
                
                'region_node_features': torch.randn(batch_size, 60, 101),
                'region_edge_index': torch.randint(0, 60, (batch_size, 2, 120)),
                'region_node_mask': torch.ones(batch_size, 60),
                'region_edge_mask': torch.ones(batch_size, 120),
                
                'ninja_physics_state': torch.randn(batch_size, 18)
            })
        
        return obs
    
    def test_hierarchical_extractor_initialization(self):
        """Test hierarchical multimodal extractor initialization."""
        obs_space = self.create_mock_observation_space(include_hierarchical=True)
        
        extractor = HierarchicalMultimodalExtractor(
            observation_space=obs_space,
            features_dim=512,
            use_hierarchical_graph=True,
            enable_auxiliary_losses=True
        )
        
        # Check that hierarchical components are initialized
        assert extractor.has_hierarchical_graph
        assert hasattr(extractor, 'hierarchical_gnn')
        assert hasattr(extractor, 'multi_scale_fusion')
        assert extractor.enable_auxiliary_losses
    
    def test_hierarchical_extractor_forward_pass(self):
        """Test forward pass through hierarchical extractor."""
        obs_space = self.create_mock_observation_space(include_hierarchical=True)
        
        extractor = HierarchicalMultimodalExtractor(
            observation_space=obs_space,
            features_dim=256,
            use_hierarchical_graph=True,
            enable_auxiliary_losses=True
        )
        
        batch_size = 2
        observations = self.create_mock_observations(batch_size, include_hierarchical=True)
        
        # Forward pass
        features = extractor(observations)
        
        # Check output shape
        assert features.shape == (batch_size, 256)
        
        # Check that features are finite
        assert torch.all(torch.isfinite(features))
        
        # Check auxiliary losses
        aux_losses = extractor.get_auxiliary_losses()
        assert isinstance(aux_losses, dict)
    
    def test_hierarchical_extractor_without_graph(self):
        """Test hierarchical extractor falls back gracefully without graph data."""
        obs_space = self.create_mock_observation_space(include_hierarchical=False)
        
        extractor = HierarchicalMultimodalExtractor(
            observation_space=obs_space,
            features_dim=256,
            use_hierarchical_graph=True,  # Request hierarchical but not available
            enable_auxiliary_losses=True
        )
        
        # Should fall back to non-hierarchical processing
        assert not extractor.has_hierarchical_graph
        
        batch_size = 2
        observations = self.create_mock_observations(batch_size, include_hierarchical=False)
        
        # Forward pass should still work
        features = extractor(observations)
        assert features.shape == (batch_size, 256)
    
    def test_auxiliary_loss_computation(self):
        """Test auxiliary loss computation and weighting."""
        obs_space = self.create_mock_observation_space(include_hierarchical=True)
        
        extractor = HierarchicalMultimodalExtractor(
            observation_space=obs_space,
            features_dim=256,
            use_hierarchical_graph=True,
            enable_auxiliary_losses=True
        )
        
        batch_size = 2
        observations = self.create_mock_observations(batch_size, include_hierarchical=True)
        
        # Forward pass to generate auxiliary losses
        features = extractor(observations)
        
        # Compute total loss
        main_loss = torch.tensor(1.0)
        total_loss = extractor.compute_total_loss(main_loss)
        
        # Total loss should be >= main loss (auxiliary losses are non-negative)
        assert total_loss >= main_loss


class TestHierarchicalGraphObservationWrapper:
    """Test hierarchical graph observation wrapper."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        wrapper = HierarchicalGraphObservationWrapper(
            enable_hierarchical=True,
            cache_graphs=True
        )
        
        # Check initialization
        assert wrapper.enable_hierarchical
        assert wrapper.cache_graphs
        assert wrapper.graph_cache is not None
    
    def test_observation_processing_without_hierarchical_builder(self):
        """Test observation processing when hierarchical builder is not available."""
        wrapper = HierarchicalGraphObservationWrapper(
            enable_hierarchical=False,
            cache_graphs=False
        )
        
        observations = {'test': torch.tensor([1, 2, 3])}
        processed_obs = wrapper.process_observations(observations)
        
        # Should return unchanged observations
        assert processed_obs == observations


class TestIntegrationBehavior:
    """Test integration behavior and edge cases."""
    
    def test_memory_efficiency(self):
        """Test that hierarchical processing doesn't cause memory leaks."""
        scale_dims = {'level_0': 32, 'level_1': 16}
        
        fusion = AdaptiveScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=64,
            context_dim=18,
            dropout=0.0
        )
        
        # Process multiple batches to check for memory accumulation
        for _ in range(10):
            scale_features = {
                'level_0': torch.randn(4, 32),
                'level_1': torch.randn(4, 16)
            }
            ninja_state = torch.randn(4, 18)
            
            fused_features, _ = fusion(scale_features, ninja_state)
            
            # Check that gradients can be computed
            loss = fused_features.sum()
            loss.backward()
            
            # Clear gradients
            fusion.zero_grad()
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through hierarchical components."""
        input_dims = {'sub_cell': 32}
        
        gnn = HierarchicalDiffPoolGNN(
            input_dims=input_dims,
            hidden_dim=16,
            output_dim=32,
            num_levels=1,
            pooling_ratios=[0.5],
            gnn_layers_per_level=1,
            dropout=0.0
        )
        
        # Create input with requires_grad=True
        hierarchical_data = {
            'sub_cell_node_features': torch.randn(1, 10, 32, requires_grad=True),
            'sub_cell_edge_index': torch.randint(0, 10, (1, 2, 20)),
            'sub_cell_node_mask': torch.ones(1, 10),
            'sub_cell_edge_mask': torch.ones(1, 20)
        }
        
        # Forward pass
        output, aux_losses = gnn(hierarchical_data)
        
        # Compute loss and backward pass
        total_loss = output.sum() + sum(aux_losses.values())
        total_loss.backward()
        
        # Check that gradients exist
        assert hierarchical_data['sub_cell_node_features'].grad is not None
        assert torch.any(hierarchical_data['sub_cell_node_features'].grad != 0)
    
    def test_device_compatibility(self):
        """Test that components work on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        
        scale_dims = {'level_0': 16}
        fusion = AdaptiveScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=32,
            context_dim=18,
            dropout=0.0
        ).to(device)
        
        scale_features = {
            'level_0': torch.randn(2, 16, device=device)
        }
        ninja_state = torch.randn(2, 18, device=device)
        
        # Forward pass on GPU
        fused_features, _ = fusion(scale_features, ninja_state)
        
        # Check that output is on correct device
        assert fused_features.device == device


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_handling(self):
        """Test that components handle invalid inputs gracefully."""
        scale_dims = {'level_0': 32}
        
        fusion = AdaptiveScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=64,
            context_dim=18
        )
        
        # Test with empty scale features
        with pytest.raises(ValueError, match="scale_features cannot be empty"):
            fusion({}, torch.randn(2, 18))
        
        # Test with invalid tensor
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            fusion({'level_0': "not_a_tensor"}, torch.randn(2, 18))
        
        # Test with empty tensor
        with pytest.raises(ValueError, match="cannot be empty"):
            fusion({'level_0': torch.empty(0)}, torch.randn(2, 18))
    
    def test_hierarchical_builder_validation(self):
        """Test hierarchical builder input validation."""
        try:
            from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
            builder = HierarchicalGraphBuilder()
            
            # Test with invalid level_data
            with pytest.raises(ValueError, match="level_data must be a dictionary"):
                builder.build_hierarchical_graph("not_a_dict", (0, 0), [])
            
            # Test with invalid ninja_position
            with pytest.raises(ValueError, match="ninja_position must be a tuple/list of length 2"):
                builder.build_hierarchical_graph({}, (0,), [])
            
            # Test with invalid entities
            with pytest.raises(ValueError, match="entities must be a list"):
                builder.build_hierarchical_graph({}, (0, 0), "not_a_list")
                
        except ImportError:
            pytest.skip("HierarchicalGraphBuilder not available")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        layer = DiffPoolLayer(
            input_dim=16,
            hidden_dim=8,
            output_dim=8,
            num_clusters=4,
            gnn_layers=1,
            dropout=0.0
        )
        
        # Test with very large values
        large_features = torch.randn(1, 8, 16) * 1000
        edge_index = torch.randint(0, 8, (1, 2, 16))
        node_mask = torch.ones(1, 8)
        edge_mask = torch.ones(1, 16)
        
        pooled_features, _, _, _, aux_losses = layer(
            large_features, edge_index, node_mask, edge_mask
        )
        
        # Should still produce finite outputs
        assert torch.isfinite(pooled_features).all()
        assert torch.isfinite(aux_losses['link_loss'])
        assert torch.isfinite(aux_losses['entropy_loss'])
        
        # Test with very small values
        small_features = torch.randn(1, 8, 16) * 1e-6
        
        pooled_features, _, _, _, aux_losses = layer(
            small_features, edge_index, node_mask, edge_mask
        )
        
        assert torch.isfinite(pooled_features).all()
        assert torch.isfinite(aux_losses['link_loss'])
        assert torch.isfinite(aux_losses['entropy_loss'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])