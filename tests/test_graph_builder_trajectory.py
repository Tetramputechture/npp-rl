"""
Integration tests for GraphBuilder with trajectory features.

This module tests the integration of trajectory-based edge features
in the GraphBuilder for N++ RL.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add nclone to path for testing
nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nclone')
if os.path.exists(nclone_path):
    sys.path.insert(0, nclone_path)

try:
    from nclone.graph.graph_builder import GraphBuilder, EdgeType
except ImportError:
    # Skip tests if nclone is not available
    GraphBuilder = None
    EdgeType = None


@unittest.skipIf(GraphBuilder is None, "nclone not available")
class TestGraphBuilderTrajectory(unittest.TestCase):
    """Test GraphBuilder with trajectory features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = GraphBuilder()
        
        # Create mock level data
        self.level_data = {
            'tiles': [[0 for _ in range(10)] for _ in range(10)]  # 10x10 empty level
        }
        
        self.ninja_position = (120.0, 120.0)  # Center of level
        self.ninja_velocity = (50.0, -25.0)
        self.ninja_state = 1  # Jumping state
        self.entities = []
    
    def test_build_graph_with_trajectory_features(self):
        """Test building graph with trajectory features enabled."""
        # Mock trajectory calculator to be available
        with patch.object(self.builder, '_ensure_trajectory_calculator') as mock_ensure:
            mock_calc = Mock()
            mock_classifier = Mock()
            
            # Mock trajectory calculation
            mock_calc.calculate_jump_trajectory.return_value = [
                {'x': 120.0, 'y': 120.0, 'vx': 50.0, 'vy': -25.0, 't': 0.0},
                {'x': 140.0, 'y': 115.0, 'vx': 50.0, 'vy': -15.0, 't': 0.4},
                {'x': 160.0, 'y': 115.0, 'vx': 50.0, 'vy': -5.0, 't': 0.8}
            ]
            mock_calc.validate_trajectory_clearance.return_value = 0.85
            
            # Mock movement classification
            mock_classifier.classify_movement.return_value = 'JUMP'
            
            self.builder.trajectory_calc = mock_calc
            self.builder.movement_classifier = mock_classifier
            
            # Build graph
            graph_data = self.builder.build_graph(
                self.level_data,
                self.ninja_position,
                self.entities,
                self.ninja_velocity,
                self.ninja_state
            )
            
            # Verify graph structure
            self.assertIsNotNone(graph_data)
            self.assertGreater(graph_data.num_nodes, 0)
            self.assertGreater(graph_data.num_edges, 0)
            
            # Verify edge features have correct dimensions (16)
            self.assertEqual(graph_data.edge_features.shape[1], 16)
            
            # Check that some edges have trajectory information
            non_zero_edges = graph_data.edge_features[graph_data.edge_mask > 0]
            if len(non_zero_edges) > 0:
                # Check trajectory feature indices (9-15)
                trajectory_features = non_zero_edges[:, 9:16]
                
                # At least some edges should have non-zero trajectory features
                has_trajectory_info = np.any(trajectory_features != 0)
                self.assertTrue(has_trajectory_info, "No trajectory information found in edges")
    
    def test_build_graph_without_trajectory_calculator(self):
        """Test building graph when trajectory calculator is not available."""
        # Ensure trajectory calculator is None
        self.builder.trajectory_calc = None
        self.builder.movement_classifier = None
        
        # Build graph
        graph_data = self.builder.build_graph(
            self.level_data,
            self.ninja_position,
            self.entities,
            self.ninja_velocity,
            self.ninja_state
        )
        
        # Should still work without trajectory features
        self.assertIsNotNone(graph_data)
        self.assertGreater(graph_data.num_nodes, 0)
        
        # Edge features should still have correct dimensions but trajectory features should be zero
        self.assertEqual(graph_data.edge_features.shape[1], 16)
        
        # Check that trajectory features are zero when calculator is not available
        non_zero_edges = graph_data.edge_features[graph_data.edge_mask > 0]
        if len(non_zero_edges) > 0:
            trajectory_features = non_zero_edges[:, 9:16]
            # All trajectory features should be zero
            self.assertTrue(np.all(trajectory_features == 0))
    
    def test_build_graph_without_ninja_velocity(self):
        """Test building graph without ninja velocity information."""
        # Mock trajectory calculator but don't provide velocity
        with patch.object(self.builder, '_ensure_trajectory_calculator'):
            mock_calc = Mock()
            self.builder.trajectory_calc = mock_calc
            self.builder.movement_classifier = Mock()
            
            # Build graph without velocity
            graph_data = self.builder.build_graph(
                self.level_data,
                self.ninja_position,
                self.entities,
                ninja_velocity=None,
                ninja_state=self.ninja_state
            )
            
            # Should still work
            self.assertIsNotNone(graph_data)
            self.assertGreater(graph_data.num_nodes, 0)
            
            # Trajectory calculator should not be called without velocity
            mock_calc.calculate_jump_trajectory.assert_not_called()
    
    def test_edge_feature_dimensions(self):
        """Test that edge features have correct dimensions."""
        # Verify the calculated edge feature dimension
        expected_dim = (
            len(EdgeType) +  # Edge type one-hot (6)
            2 +  # Direction (dx, dy)
            1 +  # Traversability cost
            3 +  # Trajectory parameters (time_of_flight, energy_cost, success_probability)
            2 +  # Physics constraints (min_velocity, max_velocity)
            2    # Movement requirements (requires_jump, requires_wall_contact)
        )
        
        self.assertEqual(self.builder.edge_feature_dim, expected_dim)
        self.assertEqual(self.builder.edge_feature_dim, 16)
    
    def test_trajectory_feature_encoding(self):
        """Test that trajectory features are correctly encoded in edge features."""
        # Mock trajectory calculator with specific values
        with patch.object(self.builder, '_ensure_trajectory_calculator'):
            mock_calc = Mock()
            mock_classifier = Mock()
            
            # Mock specific trajectory info
            trajectory_info = {
                'time_of_flight': 1.5,
                'energy_cost': 0.8,
                'success_probability': 0.9,
                'min_velocity': 30.0,
                'max_velocity': 120.0,
                'requires_jump': True,
                'requires_wall_contact': False
            }
            
            self.builder.trajectory_calc = mock_calc
            self.builder.movement_classifier = mock_classifier
            
            # Mock the _determine_sub_cell_traversability method to return trajectory info
            original_method = self.builder._determine_sub_cell_traversability
            
            def mock_traversability(*args, **kwargs):
                return EdgeType.JUMP, 2.0, trajectory_info
            
            self.builder._determine_sub_cell_traversability = mock_traversability
            
            # Build graph
            graph_data = self.builder.build_graph(
                self.level_data,
                self.ninja_position,
                self.entities,
                self.ninja_velocity,
                self.ninja_state
            )
            
            # Find edges with trajectory information
            non_zero_edges = graph_data.edge_features[graph_data.edge_mask > 0]
            if len(non_zero_edges) > 0:
                # Check first edge with trajectory info
                edge_feat = non_zero_edges[0]
                
                # Verify trajectory features are encoded correctly
                base_idx = len(EdgeType) + 3  # After edge type, direction, cost
                
                self.assertAlmostEqual(edge_feat[base_idx], 1.5, places=2)      # time_of_flight
                self.assertAlmostEqual(edge_feat[base_idx + 1], 0.8, places=2)  # energy_cost
                self.assertAlmostEqual(edge_feat[base_idx + 2], 0.9, places=2)  # success_probability
                self.assertAlmostEqual(edge_feat[base_idx + 3], 30.0, places=1) # min_velocity
                self.assertAlmostEqual(edge_feat[base_idx + 4], 120.0, places=1) # max_velocity
                self.assertAlmostEqual(edge_feat[base_idx + 5], 1.0, places=2)  # requires_jump
                self.assertAlmostEqual(edge_feat[base_idx + 6], 0.0, places=2)  # requires_wall_contact
            
            # Restore original method
            self.builder._determine_sub_cell_traversability = original_method
    
    def test_ensure_trajectory_calculator_import_success(self):
        """Test successful import of trajectory calculator."""
        # Reset calculator
        self.builder.trajectory_calc = None
        self.builder.movement_classifier = None
        
        # Mock successful imports
        with patch('sys.path'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.__import__') as mock_import:
            
            # Mock the imported classes
            mock_trajectory_calc = Mock()
            mock_movement_classifier = Mock()
            
            def mock_import_side_effect(name, *args, **kwargs):
                if name == 'npp_rl.models.trajectory_calculator':
                    mock_module = Mock()
                    mock_module.TrajectoryCalculator = Mock(return_value=mock_trajectory_calc)
                    return mock_module
                elif name == 'npp_rl.models.movement_classifier':
                    mock_module = Mock()
                    mock_module.MovementClassifier = Mock(return_value=mock_movement_classifier)
                    return mock_module
                return Mock()
            
            mock_import.side_effect = mock_import_side_effect
            
            # Call ensure method
            self.builder._ensure_trajectory_calculator()
            
            # Verify calculators were set
            self.assertIsNotNone(self.builder.trajectory_calc)
            self.assertIsNotNone(self.builder.movement_classifier)
    
    def test_ensure_trajectory_calculator_import_failure(self):
        """Test handling of import failure for trajectory calculator."""
        # Reset calculator
        self.builder.trajectory_calc = None
        self.builder.movement_classifier = None
        
        # Mock import failure
        with patch('builtins.__import__', side_effect=ImportError("Module not found")), \
             patch('logging.warning') as mock_warning:
            
            # Call ensure method
            self.builder._ensure_trajectory_calculator()
            
            # Verify calculators remain None
            self.assertIsNone(self.builder.trajectory_calc)
            self.assertIsNone(self.builder.movement_classifier)
            
            # Verify warning was logged
            mock_warning.assert_called_once()


@unittest.skipIf(GraphBuilder is None, "nclone not available")
class TestGraphBuilderTrajectoryEdgeCases(unittest.TestCase):
    """Test edge cases for GraphBuilder trajectory integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = GraphBuilder()
    
    def test_trajectory_calculation_exception_handling(self):
        """Test handling of exceptions during trajectory calculation."""
        # Mock trajectory calculator that raises exception
        with patch.object(self.builder, '_ensure_trajectory_calculator'):
            mock_calc = Mock()
            mock_calc.calculate_jump_trajectory.side_effect = Exception("Calculation failed")
            
            self.builder.trajectory_calc = mock_calc
            self.builder.movement_classifier = Mock()
            
            # Should not crash when trajectory calculation fails
            level_data = {'tiles': [[0 for _ in range(5)] for _ in range(5)]}
            
            graph_data = self.builder.build_graph(
                level_data,
                (60.0, 60.0),
                [],
                (50.0, -25.0),
                1
            )
            
            # Should still produce valid graph
            self.assertIsNotNone(graph_data)
            self.assertGreater(graph_data.num_nodes, 0)
    
    def test_movement_classifier_exception_handling(self):
        """Test handling of exceptions during movement classification."""
        # Mock movement classifier that raises exception
        with patch.object(self.builder, '_ensure_trajectory_calculator'):
            mock_classifier = Mock()
            mock_classifier.classify_movement.side_effect = Exception("Classification failed")
            
            self.builder.trajectory_calc = Mock()
            self.builder.movement_classifier = mock_classifier
            
            # Should not crash when movement classification fails
            level_data = {'tiles': [[0 for _ in range(5)] for _ in range(5)]}
            
            graph_data = self.builder.build_graph(
                level_data,
                (60.0, 60.0),
                [],
                (50.0, -25.0),
                1
            )
            
            # Should still produce valid graph
            self.assertIsNotNone(graph_data)
            self.assertGreater(graph_data.num_nodes, 0)


if __name__ == '__main__':
    unittest.main()