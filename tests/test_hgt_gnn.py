"""
Tests for Heterogeneous Graph Transformer (HGT) implementation.

This module tests the HGT architecture, entity type system, and integration
with the multimodal feature extractors.
"""

import pytest
import torch
import numpy as np
from typing import Dict

from npp_rl.models.hgt_gnn import (
    HGTLayer, HGTEncoder, create_hgt_encoder,
    NodeType, EntityType, EdgeType
)
from npp_rl.models.entity_type_system import (
    EntityTypeSystem, EntitySpecializedEmbedding, HazardAwareAttention,
    EntityCategory, create_entity_type_system
)
from npp_rl.feature_extractors.multimodal import (
    MultimodalGraphExtractor, create_hgt_multimodal_extractor
)
from gymnasium.spaces import Dict as SpacesDict, Box


class TestEntityTypeSystem:
    """Test entity type system functionality."""
    
    def test_entity_type_system_creation(self):
        """Test entity type system creation and basic functionality."""
        entity_system = create_entity_type_system()
        
        # Test category mapping
        assert entity_system.get_entity_category(EntityType.GOLD) == EntityCategory.COLLECTIBLE
        assert entity_system.get_entity_category(EntityType.THWUMP) == EntityCategory.HAZARD
        assert entity_system.get_entity_category(EntityType.LAUNCH_PAD) == EntityCategory.MOVEMENT
        assert entity_system.get_entity_category(EntityType.EXIT) == EntityCategory.INTERACTIVE
    
    def test_hazard_detection(self):
        """Test hazard detection functionality."""
        entity_system = create_entity_type_system()
        
        # Test hazardous entities
        assert entity_system.is_hazardous(EntityType.THWUMP)
        assert entity_system.is_hazardous(EntityType.DEATH_BALL)
        assert entity_system.is_hazardous(EntityType.ACTIVE_MINE)
        
        # Test non-hazardous entities
        assert not entity_system.is_hazardous(EntityType.GOLD)
        assert not entity_system.is_hazardous(EntityType.EXIT)
        assert not entity_system.is_hazardous(EntityType.LAUNCH_PAD)
    
    def test_movement_impact(self):
        """Test movement impact detection."""
        entity_system = create_entity_type_system()
        
        # Test entities that affect movement
        assert entity_system.affects_movement(EntityType.LAUNCH_PAD)
        assert entity_system.affects_movement(EntityType.BOUNCE_BLOCK)
        assert entity_system.affects_movement(EntityType.THWUMP)
        
        # Test entities that don't affect movement
        assert not entity_system.affects_movement(EntityType.GOLD)
        assert not entity_system.affects_movement(EntityType.EXIT)
    
    def test_attention_weights(self):
        """Test attention weight assignment."""
        entity_system = create_entity_type_system()
        
        # Hazards should have highest attention
        hazard_weight = entity_system.get_attention_weight(EntityType.THWUMP)
        collectible_weight = entity_system.get_attention_weight(EntityType.GOLD)
        
        assert hazard_weight > collectible_weight


class TestEntitySpecializedEmbedding:
    """Test entity specialized embedding functionality."""
    
    def test_embedding_creation(self):
        """Test creation of specialized embedding layer."""
        entity_system = create_entity_type_system()
        embedding = EntitySpecializedEmbedding(
            input_dim=64,
            output_dim=128,
            entity_type_system=entity_system
        )
        
        assert embedding.input_dim == 64
        assert embedding.output_dim == 128
        assert len(embedding.category_embeddings) == len(EntityCategory)
    
    def test_embedding_forward(self):
        """Test forward pass through specialized embedding."""
        entity_system = create_entity_type_system()
        embedding = EntitySpecializedEmbedding(
            input_dim=64,
            output_dim=128,
            entity_type_system=entity_system
        )
        
        batch_size, num_nodes = 2, 10
        node_features = torch.randn(batch_size, num_nodes, 64)
        node_types = torch.randint(0, 3, (batch_size, num_nodes))
        entity_types = torch.randint(1, 29, (batch_size, num_nodes))
        
        output = embedding(node_features, node_types, entity_types)
        
        assert output.shape == (batch_size, num_nodes, 128)
        assert not torch.isnan(output).any()


class TestHazardAwareAttention:
    """Test hazard-aware attention mechanism."""
    
    def test_attention_creation(self):
        """Test creation of hazard-aware attention."""
        entity_system = create_entity_type_system()
        attention = HazardAwareAttention(
            embed_dim=128,
            num_heads=8,
            entity_type_system=entity_system
        )
        
        assert attention.embed_dim == 128
        assert attention.num_heads == 8
    
    def test_attention_forward(self):
        """Test forward pass through hazard-aware attention."""
        entity_system = create_entity_type_system()
        attention = HazardAwareAttention(
            embed_dim=128,
            num_heads=8,
            entity_type_system=entity_system
        )
        
        batch_size, seq_len = 2, 10
        query = torch.randn(batch_size, seq_len, 128)
        key = torch.randn(batch_size, seq_len, 128)
        value = torch.randn(batch_size, seq_len, 128)
        entity_types = torch.randint(1, 29, (batch_size, seq_len))
        
        output, weights = attention(query, key, value, entity_types)
        
        assert output.shape == (batch_size, seq_len, 128)
        assert weights.shape == (batch_size, seq_len, seq_len)
        assert not torch.isnan(output).any()


class TestHGTLayer:
    """Test HGT layer functionality."""
    
    def test_hgt_layer_creation(self):
        """Test HGT layer creation."""
        layer = HGTLayer(
            in_dim=64,
            out_dim=128,
            num_heads=8,
            num_node_types=3,
            num_edge_types=6
        )
        
        assert layer.in_dim == 64
        assert layer.out_dim == 128
        assert layer.num_heads == 8
        assert layer.d_k == 16  # 128 / 8
    
    def test_hgt_layer_forward(self):
        """Test forward pass through HGT layer."""
        layer = HGTLayer(
            in_dim=64,
            out_dim=64,
            num_heads=8,
            num_node_types=3,
            num_edge_types=6
        )
        
        batch_size, num_nodes, num_edges = 2, 20, 40
        
        node_features = torch.randn(batch_size, num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (batch_size, 2, num_edges))
        node_types = torch.randint(0, 3, (batch_size, num_nodes))
        edge_types = torch.randint(0, 6, (batch_size, num_edges))
        node_mask = torch.ones(batch_size, num_nodes)
        edge_mask = torch.ones(batch_size, num_edges)
        
        output = layer(node_features, edge_index, node_types, edge_types, node_mask, edge_mask)
        
        assert output.shape == (batch_size, num_nodes, 64)
        assert not torch.isnan(output).any()


class TestHGTEncoder:
    """Test HGT encoder functionality."""
    
    def test_hgt_encoder_creation(self):
        """Test HGT encoder creation."""
        encoder = HGTEncoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=128,
            num_layers=2,
            output_dim=256
        )
        
        assert encoder.node_feature_dim == 85
        assert encoder.edge_feature_dim == 16
        assert encoder.hidden_dim == 128
        assert encoder.output_dim == 256
        assert len(encoder.hgt_layers) == 2
    
    def test_hgt_encoder_forward(self):
        """Test forward pass through HGT encoder."""
        encoder = HGTEncoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=128,
            num_layers=2,
            output_dim=256
        )
        
        batch_size, num_nodes, num_edges = 2, 50, 100
        
        graph_obs = {
            'graph_node_feats': torch.randn(batch_size, num_nodes, 85),
            'graph_edge_index': torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
            'graph_edge_feats': torch.randn(batch_size, num_edges, 16),
            'graph_node_mask': torch.ones(batch_size, num_nodes),
            'graph_edge_mask': torch.ones(batch_size, num_edges),
            'graph_node_types': torch.randint(0, 3, (batch_size, num_nodes)),
            'graph_edge_types': torch.randint(0, 6, (batch_size, num_edges)),
            'graph_entity_types': torch.randint(1, 29, (batch_size, num_nodes))
        }
        
        output = encoder(graph_obs)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_create_hgt_encoder(self):
        """Test HGT encoder factory function."""
        encoder = create_hgt_encoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=256,
            num_layers=3,
            output_dim=512
        )
        
        assert isinstance(encoder, HGTEncoder)
        assert encoder.hidden_dim == 256
        assert encoder.output_dim == 512


class TestHGTIntegration:
    """Test HGT integration with multimodal feature extractors."""
    
    def create_test_observation_space(self):
        """Create test observation space for multimodal extractor."""
        return SpacesDict({
            'player_frame': Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8),
            'game_state': Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32),
            'graph_node_feats': Box(low=-np.inf, high=np.inf, shape=(1000, 85), dtype=np.float32),
            'graph_edge_index': Box(low=0, high=999, shape=(2, 2000), dtype=np.int32),
            'graph_edge_feats': Box(low=-np.inf, high=np.inf, shape=(2000, 16), dtype=np.float32),
            'graph_node_mask': Box(low=0, high=1, shape=(1000,), dtype=np.float32),
            'graph_edge_mask': Box(low=0, high=1, shape=(2000,), dtype=np.float32)
        })
    
    def test_hgt_multimodal_extractor_creation(self):
        """Test creation of HGT-enabled multimodal extractor."""
        obs_space = self.create_test_observation_space()
        
        extractor = create_hgt_multimodal_extractor(
            observation_space=obs_space,
            features_dim=512,
            gnn_hidden_dim=256,
            gnn_num_layers=3,
            gnn_output_dim=512
        )
        
        assert isinstance(extractor, MultimodalGraphExtractor)
        assert extractor.use_hgt == True
        assert extractor.features_dim == 512
    
    def test_hgt_multimodal_extractor_forward(self):
        """Test forward pass through HGT multimodal extractor."""
        obs_space = self.create_test_observation_space()
        
        extractor = create_hgt_multimodal_extractor(
            observation_space=obs_space,
            features_dim=512,
            gnn_hidden_dim=128,
            gnn_num_layers=2,
            gnn_output_dim=256
        )
        
        # Create test observations
        batch_size = 2
        observations = {
            'player_frame': torch.randint(0, 256, (batch_size, 84, 84, 4), dtype=torch.uint8),
            'game_state': torch.randn(batch_size, 32),
            'graph_node_feats': torch.randn(batch_size, 1000, 85),
            'graph_edge_index': torch.randint(0, 1000, (batch_size, 2, 2000)),
            'graph_edge_feats': torch.randn(batch_size, 2000, 16),
            'graph_node_mask': torch.ones(batch_size, 1000),
            'graph_edge_mask': torch.ones(batch_size, 2000)
        }
        
        output = extractor(observations)
        
        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()


class TestHGTPerformance:
    """Test HGT performance and computational efficiency."""
    
    def test_hgt_vs_graphsage_comparison(self):
        """Compare HGT and GraphSAGE performance."""
        from npp_rl.models.gnn import create_graph_encoder
        
        # Create both encoders
        hgt_encoder = create_hgt_encoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=128,
            num_layers=2,
            output_dim=256
        )
        
        graphsage_encoder = create_graph_encoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=128,
            num_layers=2,
            output_dim=256
        )
        
        # Create test data
        batch_size, num_nodes, num_edges = 1, 100, 200
        graph_obs = {
            'graph_node_feats': torch.randn(batch_size, num_nodes, 85),
            'graph_edge_index': torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
            'graph_edge_feats': torch.randn(batch_size, num_edges, 16),
            'graph_node_mask': torch.ones(batch_size, num_nodes),
            'graph_edge_mask': torch.ones(batch_size, num_edges)
        }
        
        # Test both encoders
        hgt_output = hgt_encoder(graph_obs)
        graphsage_output = graphsage_encoder(graph_obs)
        
        assert hgt_output.shape == graphsage_output.shape
        assert not torch.isnan(hgt_output).any()
        assert not torch.isnan(graphsage_output).any()
    
    def test_hgt_memory_usage(self):
        """Test HGT memory usage with large graphs."""
        encoder = create_hgt_encoder(
            node_feature_dim=85,
            edge_feature_dim=16,
            hidden_dim=64,  # Smaller to reduce memory
            num_layers=2,
            output_dim=128
        )
        
        # Test with larger graph
        batch_size, num_nodes, num_edges = 1, 500, 1000
        graph_obs = {
            'graph_node_feats': torch.randn(batch_size, num_nodes, 85),
            'graph_edge_index': torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
            'graph_edge_feats': torch.randn(batch_size, num_edges, 16),
            'graph_node_mask': torch.ones(batch_size, num_nodes),
            'graph_edge_mask': torch.ones(batch_size, num_edges)
        }
        
        output = encoder(graph_obs)
        
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__])