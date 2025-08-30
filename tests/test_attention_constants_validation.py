"""
Tests for attention constants validation and edge case handling.

This module tests the improved validation and error handling in the
spatial attention modules after the code review improvements.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Dict as SpacesDict, Box

from npp_rl.models.spatial_attention import (
    GraphSpatialGuidance,
    SpatialAttentionModule,
    MultiScaleSpatialAttention
)
from npp_rl.models.attention_constants import (
    MIN_ATTENTION_HEADS,
    MIN_SPATIAL_DIM,
    MIN_FEATURE_DIM
)


class TestValidationAndEdgeCases:
    """Test validation and edge case handling in attention modules."""
    
    def test_graph_spatial_guidance_validation(self):
        """Test input validation in GraphSpatialGuidance."""
        # Test minimum feature dimension validation
        with pytest.raises(ValueError, match="graph_dim must be at least"):
            GraphSpatialGuidance(graph_dim=0)
        
        # Test minimum spatial dimension validation
        with pytest.raises(ValueError, match="Spatial dimensions must be at least"):
            GraphSpatialGuidance(graph_dim=64, spatial_height=2, spatial_width=16)
        
        with pytest.raises(ValueError, match="Spatial dimensions must be at least"):
            GraphSpatialGuidance(graph_dim=64, spatial_height=16, spatial_width=2)
        
        # Test minimum attention heads validation
        with pytest.raises(ValueError, match="num_attention_heads must be at least"):
            GraphSpatialGuidance(graph_dim=64, num_attention_heads=0)
        
        # Test guidance_dim divisibility by num_attention_heads
        with pytest.raises(ValueError, match="guidance_dim .* must be divisible by num_attention_heads"):
            GraphSpatialGuidance(graph_dim=64, guidance_dim=65, num_attention_heads=4)
    
    def test_spatial_attention_module_validation(self):
        """Test input validation in SpatialAttentionModule."""
        # Test minimum feature dimensions
        with pytest.raises(ValueError, match="Feature dimensions must be at least"):
            SpatialAttentionModule(graph_dim=0, visual_dim=128)
        
        with pytest.raises(ValueError, match="Feature dimensions must be at least"):
            SpatialAttentionModule(graph_dim=64, visual_dim=0)
        
        # Test minimum spatial dimensions
        with pytest.raises(ValueError, match="Spatial dimensions must be at least"):
            SpatialAttentionModule(graph_dim=64, visual_dim=128, spatial_height=2)
        
        # Test minimum attention heads
        with pytest.raises(ValueError, match="num_attention_heads must be at least"):
            SpatialAttentionModule(graph_dim=64, visual_dim=128, num_attention_heads=0)
    
    def test_multi_scale_spatial_attention_validation(self):
        """Test input validation in MultiScaleSpatialAttention."""
        # Test minimum feature dimensions
        with pytest.raises(ValueError, match="Feature dimensions must be at least"):
            MultiScaleSpatialAttention(graph_dim=0, visual_dim=128)
        
        # Test invalid scales
        with pytest.raises(ValueError, match="All scales must be at least"):
            MultiScaleSpatialAttention(graph_dim=64, visual_dim=128, scales=[8, 2, 16])
        
        # Test empty scales
        with pytest.raises(ValueError, match="All scales must be at least"):
            MultiScaleSpatialAttention(graph_dim=64, visual_dim=128, scales=[])
        
        # Test minimum attention heads
        with pytest.raises(ValueError, match="num_attention_heads must be at least"):
            MultiScaleSpatialAttention(graph_dim=64, visual_dim=128, num_attention_heads=0)
    
    def test_forward_pass_validation(self):
        """Test forward pass input validation."""
        guidance = GraphSpatialGuidance(graph_dim=64)
        
        # Test wrong feature dimension
        with pytest.raises(ValueError, match="Expected graph features with dim"):
            wrong_features = torch.randn(2, 10, 32)  # Wrong last dimension
            guidance(wrong_features)
        
        # Test empty graph features
        with pytest.raises(ValueError, match="Graph features cannot be empty"):
            empty_features = torch.randn(2, 0, 64)  # No nodes
            guidance(empty_features)
    
    def test_spatial_attention_forward_validation(self):
        """Test SpatialAttentionModule forward pass validation."""
        attention = SpatialAttentionModule(graph_dim=64, visual_dim=128)
        
        # Test dimension mismatch
        visual_features = torch.randn(2, 64)  # Wrong visual dimension
        graph_features = torch.randn(2, 10, 64)
        
        with pytest.raises(ValueError, match="Expected visual features with dim"):
            attention(visual_features, graph_features)
        
        # Test wrong graph feature shape
        visual_features = torch.randn(2, 128)
        graph_features = torch.randn(2, 64)  # Wrong shape (should be 3D)
        
        with pytest.raises(ValueError, match="Expected 3D graph features"):
            attention(visual_features, graph_features)
        
        # Test batch size mismatch
        visual_features = torch.randn(2, 128)
        graph_features = torch.randn(3, 10, 64)  # Different batch size
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            attention(visual_features, graph_features)
        
        # Test invalid feature map size
        visual_features = torch.randn(2, 128)
        graph_features = torch.randn(2, 10, 64)
        
        with pytest.raises(ValueError, match="Feature map size must be at least"):
            attention(visual_features, graph_features, feature_map_size=(2, 16))
    
    def test_edge_case_handling(self):
        """Test edge case handling in attention mechanisms."""
        attention = SpatialAttentionModule(graph_dim=64, visual_dim=128)
        
        # Test with minimal valid inputs
        visual_features = torch.randn(1, 128)
        graph_features = torch.randn(1, 1, 64)  # Single node
        
        enhanced_visual, attention_map = attention(visual_features, graph_features)
        
        assert enhanced_visual.shape == (1, 128)
        assert attention_map.shape == (1, 16, 16)  # Default spatial size
        
        # Test with very small attention weights (should be clamped)
        # This tests the attention weight clamping logic
        visual_features = torch.zeros(1, 128)
        graph_features = torch.zeros(1, 1, 64)
        
        enhanced_visual, attention_map = attention(visual_features, graph_features)
        
        # Should not crash due to zero weights
        assert enhanced_visual.shape == (1, 128)
        assert not torch.isnan(enhanced_visual).any()
    
    def test_residual_weight_clamping(self):
        """Test that residual weights are properly clamped."""
        attention = SpatialAttentionModule(graph_dim=64, visual_dim=128)
        
        # Manually set residual weight outside valid range
        attention.residual_weight.data = torch.tensor(1.5)
        
        visual_features = torch.randn(2, 128)
        graph_features = torch.randn(2, 10, 64)
        
        enhanced_visual, _ = attention(visual_features, graph_features)
        
        # Should not crash and should produce valid output
        assert enhanced_visual.shape == (2, 128)
        assert not torch.isnan(enhanced_visual).any()
    
    def test_constants_usage(self):
        """Test that constants are properly used instead of magic numbers."""
        # Test default initialization uses constants
        guidance = GraphSpatialGuidance(graph_dim=64)
        
        # Check that default values match constants
        from npp_rl.models.attention_constants import (
            DEFAULT_SPATIAL_HEIGHT,
            DEFAULT_SPATIAL_WIDTH,
            DEFAULT_GUIDANCE_DIM,
            DEFAULT_DROPOUT_RATE
        )
        
        assert guidance.spatial_height == DEFAULT_SPATIAL_HEIGHT
        assert guidance.spatial_width == DEFAULT_SPATIAL_WIDTH
        assert guidance.guidance_dim == DEFAULT_GUIDANCE_DIM
        assert guidance.dropout == DEFAULT_DROPOUT_RATE
    
    def test_multi_scale_default_scales(self):
        """Test that MultiScaleSpatialAttention uses default scales when none provided."""
        attention = MultiScaleSpatialAttention(graph_dim=64, visual_dim=128)
        
        from npp_rl.models.attention_constants import DEFAULT_SCALES
        
        assert attention.scales == DEFAULT_SCALES
        assert attention.num_scales == len(DEFAULT_SCALES)