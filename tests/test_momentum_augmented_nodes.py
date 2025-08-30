"""
Tests for Momentum-Augmented Node Representations.

This module tests the integration of physics state features into graph node
representations for Task 1.2 of the graph plan.
"""

import numpy as np
import sys
import os

# Add nclone to path for testing
nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nclone')
if os.path.exists(nclone_path) and nclone_path not in sys.path:
    sys.path.insert(0, nclone_path)

from nclone.graph.graph_builder import GraphBuilder
from npp_rl.models.physics_state_extractor import PhysicsStateExtractor


class TestMomentumAugmentedNodes:
    """Test cases for momentum-augmented node representations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph_builder = GraphBuilder()
        self.physics_extractor = PhysicsStateExtractor()
        
        # Create minimal level data for testing
        self.level_data = {
            'tiles': np.zeros((23, 42), dtype=np.int32),  # Empty level
            'width': 42,
            'height': 23
        }
        
        # Basic ninja state
        self.ninja_position = (100.0, 200.0)
        self.ninja_velocity = (1.5, -0.5)
        self.ninja_state = {
            'movement_state': 1,  # Running
            'jump_buffer': 0,
            'floor_buffer': 2,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        self.entities = []  # No entities for basic tests
    
    def test_node_feature_dimensions(self):
        """Test that node feature dimensions are correctly updated."""
        # Original dimensions: 38 + 4 + 20 + 4 + 1 = 67
        # New dimensions: 67 + 18 = 85
        expected_dim = 38 + 4 + 20 + 4 + 1 + 18  # 85
        assert self.graph_builder.node_feature_dim == expected_dim
    
    def test_physics_extractor_initialization(self):
        """Test that physics extractor is properly initialized."""
        # Build a graph to trigger initialization
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Physics extractor should be initialized
        assert self.graph_builder.physics_extractor is not None
    
    def test_basic_graph_building_with_physics(self):
        """Test basic graph building with physics state."""
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Check that graph was built successfully
        assert graph_data.num_nodes > 0
        assert graph_data.node_features.shape[1] == 85  # Updated feature dimension
        
        # Check that at least one node has physics features
        physics_features_found = False
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                # Check ninja position flag (index 66)
                if node_features[66] > 0.5:  # Ninja is in this cell
                    # Physics features should be non-zero
                    physics_start = 67
                    physics_features = node_features[physics_start:physics_start + 18]
                    if np.any(physics_features != 0):
                        physics_features_found = True
                        break
        
        assert physics_features_found, "No physics features found in ninja's cell"
    
    def test_physics_features_in_ninja_cell(self):
        """Test that physics features are correctly added to ninja's cell."""
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Find ninja's cell
        ninja_cell_idx = None
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    ninja_cell_idx = i
                    break
        
        assert ninja_cell_idx is not None, "Could not find ninja's cell"
        
        # Extract physics features from ninja's cell
        node_features = graph_data.node_features[ninja_cell_idx]
        physics_start = 67
        physics_features = node_features[physics_start:physics_start + 18]
        
        # Verify physics features are reasonable
        # Velocity components should be normalized
        vx_norm = physics_features[0]
        vy_norm = physics_features[1]
        expected_vx = self.ninja_velocity[0] / self.physics_extractor.max_hor_speed
        expected_vy = self.ninja_velocity[1] / self.physics_extractor.max_hor_speed
        
        assert abs(vx_norm - expected_vx) < 1e-6
        assert abs(vy_norm - expected_vy) < 1e-6
        
        # Movement state should be normalized
        movement_state_norm = physics_features[3]
        expected_state = self.ninja_state['movement_state'] / 9.0
        assert abs(movement_state_norm - expected_state) < 1e-6
        
        # Contact flags should be set correctly for running state
        ground_contact = physics_features[4]
        wall_contact = physics_features[5]
        airborne = physics_features[6]
        
        assert ground_contact == 1.0  # Running state should have ground contact
        assert wall_contact == 0.0   # Not wall sliding
        assert airborne == 0.0       # Not airborne
    
    def test_physics_features_only_in_ninja_cell(self):
        """Test that physics features are only added to ninja's cell."""
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        ninja_cells = 0
        non_ninja_cells_with_physics = 0
        
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                ninja_flag = node_features[66]
                physics_start = 67
                physics_features = node_features[physics_start:physics_start + 18]
                
                if ninja_flag > 0.5:  # Ninja's cell
                    ninja_cells += 1
                    # Should have physics features
                    assert np.any(physics_features != 0), "Ninja cell missing physics features"
                else:  # Non-ninja cell
                    # Should not have physics features (all zeros)
                    if np.any(physics_features != 0):
                        non_ninja_cells_with_physics += 1
        
        assert ninja_cells == 1, f"Expected 1 ninja cell, found {ninja_cells}"
        assert non_ninja_cells_with_physics == 0, f"Found {non_ninja_cells_with_physics} non-ninja cells with physics features"
    
    def test_different_movement_states(self):
        """Test physics features for different movement states."""
        # Test airborne state
        airborne_state = self.ninja_state.copy()
        airborne_state['movement_state'] = 4  # Falling
        
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, airborne_state
        )
        
        # Find ninja's cell and check contact flags
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    physics_start = 67
                    physics_features = node_features[physics_start:physics_start + 18]
                    
                    # Check contact flags for falling state
                    ground_contact = physics_features[4]
                    wall_contact = physics_features[5]
                    airborne = physics_features[6]
                    
                    assert ground_contact == 0.0  # Not on ground
                    assert wall_contact == 0.0   # Not on wall
                    assert airborne == 1.0       # Airborne
                    break
    
    def test_without_physics_data(self):
        """Test graph building without physics data (backward compatibility)."""
        # Build graph without ninja_velocity and ninja_state
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities
        )
        
        # Should still work, but physics features should be zero
        assert graph_data.num_nodes > 0
        assert graph_data.node_features.shape[1] == 85  # Still correct dimensions
        
        # Find ninja's cell
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    physics_start = 67
                    physics_features = node_features[physics_start:physics_start + 18]
                    
                    # Physics features should be all zeros
                    assert np.all(physics_features == 0.0)
                    break
    
    def test_physics_feature_extraction_error_handling(self):
        """Test error handling in physics feature extraction."""
        # Create invalid ninja state that might cause errors
        invalid_state = {
            'movement_state': 'invalid',  # String instead of int
            'jump_buffer': None,
            'floor_buffer': [],
            'wall_buffer': 'test',
            'launch_pad_buffer': -1,
            'jump_input': 'maybe'
        }
        
        # Should not crash, should handle gracefully
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, invalid_state
        )
        
        assert graph_data.num_nodes > 0
        # Physics features should be zeros due to error handling
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    physics_start = 67
                    physics_features = node_features[physics_start:physics_start + 18]
                    # Should be all zeros due to error handling
                    assert np.all(physics_features == 0.0)
                    break
    
    def test_energy_calculations_in_graph(self):
        """Test that energy calculations are properly included in graph features."""
        # Set up ninja at different heights to test potential energy
        high_position = (100.0, 50.0)   # Near top
        low_position = (100.0, 500.0)   # Near bottom
        
        # Test high position
        graph_data_high = self.graph_builder.build_graph(
            self.level_data, high_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Test low position
        graph_data_low = self.graph_builder.build_graph(
            self.level_data, low_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Extract potential energy features
        pe_high = None
        pe_low = None
        
        for graph_data, pe_var in [(graph_data_high, 'pe_high'), (graph_data_low, 'pe_low')]:
            for i in range(graph_data.num_nodes):
                if graph_data.node_mask[i] > 0:
                    node_features = graph_data.node_features[i]
                    if node_features[66] > 0.5:  # Ninja position flag
                        physics_start = 67
                        physics_features = node_features[physics_start:physics_start + 18]
                        potential_energy = physics_features[10]  # PE index
                        
                        if pe_var == 'pe_high':
                            pe_high = potential_energy
                        else:
                            pe_low = potential_energy
                        break
        
        # High position should have higher potential energy
        assert pe_high is not None and pe_low is not None
        assert pe_high > pe_low, f"High PE ({pe_high}) should be > low PE ({pe_low})"
    
    def test_capability_flags(self):
        """Test jump capability flags in physics features."""
        # Test ground state (can jump)
        ground_state = self.ninja_state.copy()
        ground_state['movement_state'] = 0  # Immobile on ground
        
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            (0.0, 0.0), ground_state
        )
        
        # Find ninja's cell and check capabilities
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    physics_start = 67
                    physics_features = node_features[physics_start:physics_start + 18]
                    
                    can_jump = physics_features[16]
                    can_wall_jump = physics_features[17]
                    
                    assert can_jump == 1.0      # Can jump from ground
                    assert can_wall_jump == 0.0  # Cannot wall jump
                    break
    
    def test_feature_consistency(self):
        """Test that physics features are consistent with direct extraction."""
        # Extract physics features directly
        direct_features = self.physics_extractor.extract_ninja_physics_state(
            self.ninja_position, self.ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Extract from graph
        graph_data = self.graph_builder.build_graph(
            self.level_data, self.ninja_position, self.entities,
            self.ninja_velocity, self.ninja_state
        )
        
        # Find ninja's cell in graph
        graph_features = None
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0:
                node_features = graph_data.node_features[i]
                if node_features[66] > 0.5:  # Ninja position flag
                    physics_start = 67
                    graph_features = node_features[physics_start:physics_start + 18]
                    break
        
        assert graph_features is not None, "Could not find ninja's cell in graph"
        
        # Compare features (should be identical)
        np.testing.assert_array_almost_equal(
            direct_features, graph_features, decimal=6,
            err_msg="Graph physics features don't match direct extraction"
        )