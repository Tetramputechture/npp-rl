"""
Comprehensive integration tests for Task 1.1: Trajectory-Based Edge Features.

This module provides complete integration testing for the physics-informed
graph edge features across both npp-rl and nclone repositories.
"""

import unittest
import numpy as np
import sys
import os
import time

# Add nclone to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../nclone'))

from npp_rl.models.trajectory_calculator import TrajectoryCalculator, MovementState
from npp_rl.models.movement_classifier import MovementClassifier, MovementType, NinjaState
from nclone.graph.graph_builder import GraphBuilder
from npp_rl.config.phase2_config import GraphConfig

class TestTrajectoryCalculatorIntegration(unittest.TestCase):
    """Integration tests for TrajectoryCalculator with nclone GraphBuilder."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()

    def test_graph_builder_integration(self):
        """Test integration with GraphBuilder trajectory features."""
        # Create GraphBuilder instance
        builder = GraphBuilder()

        # Verify edge feature dimension is updated
        self.assertEqual(builder.edge_feature_dim, 16)

        # Test graph building with trajectory features
        level_data = {'tiles': np.zeros((10, 10), dtype=int)}
        ninja_position = (120.0, 120.0)
        entities = []

        # Test without trajectory features (backward compatibility)
        graph_data = builder.build_graph(level_data, ninja_position, entities)
        self.assertIsNotNone(graph_data)
        self.assertEqual(graph_data.edge_features.shape[1], 16)

        # Test with trajectory features
        graph_data_with_traj = builder.build_graph(
            level_data, ninja_position, entities,
            ninja_velocity=(50.0, -25.0), ninja_state=1
        )
        self.assertIsNotNone(graph_data_with_traj)
        self.assertEqual(graph_data_with_traj.edge_features.shape[1], 16)

    def test_collision_system_integration(self):
        """Test integration with nclone collision detection system."""
        # Test collision system integration

        # Create test trajectory
        trajectory = [
            {'x': 100.0, 'y': 100.0, 't': 0.0},
            {'x': 125.0, 'y': 90.0, 't': 10.0},
            {'x': 150.0, 'y': 80.0, 't': 20.0}
        ]

        # Create level with obstacles
        level_data = {
            'tiles': np.zeros((20, 20), dtype=int)
        }
        level_data['tiles'][5:7, 8:12] = 1  # Add obstacle

        # Test trajectory validation
        is_clear = self.calc.validate_trajectory_clearance(trajectory, level_data)
        self.assertIsInstance(is_clear, bool)

    def test_movement_state_integration(self):
        """Test integration with N++ movement states from sim_mechanics_doc.md."""
        # Test available movement states
        movement_states = [
            MovementState.IMMOBILE,
            MovementState.RUNNING,
            MovementState.JUMPING,
            MovementState.FALLING
        ]

        for state in movement_states:
            with self.subTest(ninja_state=state):
                result = self.calc.calculate_jump_trajectory(
                    start_pos=(100.0, 100.0),
                    end_pos=(150.0, 80.0),
                    ninja_state=state
                )

                self.assertIsNotNone(result)
                # Different states should potentially give different results
                self.assertIsInstance(result.feasible, bool)


class TestMovementClassifierIntegration(unittest.TestCase):
    """Integration tests for MovementClassifier with physics system."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()

    def test_ninja_state_integration(self):
        """Test integration with ninja state representation."""
        # Test comprehensive ninja state
        ninja_state = NinjaState(
            movement_state=3,  # Jumping
            velocity=(40.0, -30.0),
            position=(100.0, 100.0),
            ground_contact=False,
            wall_contact=False
        )

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 70.0),
            ninja_state=ninja_state
        )

        self.assertEqual(movement_type, MovementType.JUMP)
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

    def test_level_data_integration(self):
        """Test integration with level geometry data."""
        # Create realistic level data
        level_data = {
            'tiles': np.zeros((20, 20), dtype=int)
        }
        # Add walls and platforms
        level_data['tiles'][15:17, :] = 1  # Ground
        level_data['tiles'][10:15, 5] = 1  # Wall

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 300.0),  # On ground
            tgt_pos=(150.0, 300.0),  # Horizontal movement
            level_data=level_data
        )

        self.assertEqual(movement_type, MovementType.WALK)
        self.assertIsInstance(params, dict)


class TestFullSystemIntegration(unittest.TestCase):
    """Full system integration tests across npp-rl and nclone."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()
        self.classifier = MovementClassifier()

    def test_edge_feature_encoding_integration(self):
        """Test complete edge feature encoding pipeline."""
        builder = GraphBuilder()

        # Create test scenario
        level_data = {'tiles': np.zeros((15, 15), dtype=int)}
        ninja_position = (120.0, 120.0)
        ninja_velocity = (30.0, -20.0)
        ninja_state = 3  # Jumping
        entities = []

        # Build graph with trajectory features
        graph_data = builder.build_graph(
            level_data, ninja_position, entities,
            ninja_velocity=ninja_velocity, ninja_state=ninja_state
        )

        # Verify edge features have correct dimensions
        self.assertEqual(graph_data.edge_features.shape[1], 16)

        # Verify edge features contain trajectory information
        edge_features = graph_data.edge_features

        # Check that trajectory features (indices 9-15) are populated
        trajectory_features = edge_features[:, 9:16]

        # Should have trajectory features (may be zeros if no trajectory info available)
        self.assertEqual(trajectory_features.shape[1], 7)  # 7 trajectory features

        # Features should be finite (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(trajectory_features)))

        # Verify feature ranges are reasonable
        for i in range(9, 16):
            feature_col = edge_features[:, i]
            # No infinite or NaN values
            self.assertFalse(np.any(np.isinf(feature_col)))
            self.assertFalse(np.any(np.isnan(feature_col)))

    def test_trajectory_movement_classification_pipeline(self):
        """Test complete trajectory calculation and movement classification pipeline."""
        # Test positions
        start_pos = (100.0, 100.0)
        end_pos = (200.0, 50.0)  # Upward movement

        # Calculate trajectory
        trajectory_result = self.calc.calculate_jump_trajectory(start_pos, end_pos)

        # Classify movement
        ninja_state = NinjaState(
            movement_state=3,  # Jumping
            velocity=(50.0, -40.0),
            position=start_pos
        )

        movement_type, movement_params = self.classifier.classify_movement(
            start_pos, end_pos, ninja_state
        )

        # Verify results are consistent
        if trajectory_result.feasible:
            # Movement should be classified as jump for upward movement
            self.assertEqual(movement_type, MovementType.JUMP)

            # Parameters should be reasonable
            self.assertGreater(trajectory_result.time_of_flight, 0)
            self.assertGreater(trajectory_result.energy_cost, 0)
            self.assertGreaterEqual(trajectory_result.success_probability, 0.0)
            self.assertLessEqual(trajectory_result.success_probability, 1.0)

    def test_physics_validation_accuracy(self):
        """Test physics validation accuracy across the system."""
        # Test known physics scenarios
        test_cases = [
            {
                'name': 'horizontal_movement',
                'start': (0.0, 100.0),
                'end': (100.0, 100.0),
                'expected_type': MovementType.WALK
            },
            {
                'name': 'upward_jump',
                'start': (0.0, 100.0),
                'end': (50.0, 50.0),
                'expected_type': MovementType.JUMP
            },
            {
                'name': 'downward_fall',
                'start': (0.0, 50.0),
                'end': (50.0, 100.0),
                'expected_type': MovementType.FALL
            }
        ]

        for case in test_cases:
            with self.subTest(case=case['name']):
                # Calculate trajectory
                trajectory_result = self.calc.calculate_jump_trajectory(
                    case['start'], case['end']
                )

                # Classify movement
                movement_type, _ = self.classifier.classify_movement(
                    case['start'], case['end']
                )

                # Verify classification matches expectation
                self.assertEqual(movement_type, case['expected_type'])

                # Verify trajectory is reasonable
                if trajectory_result.feasible:
                    self.assertGreater(trajectory_result.time_of_flight, 0)
                    self.assertGreaterEqual(trajectory_result.energy_cost, 0)

    def test_backward_compatibility(self):
        """Test that new features maintain backward compatibility."""
        builder = GraphBuilder()

        # Test old API (without trajectory features)
        level_data = {'tiles': np.zeros((10, 10), dtype=int)}
        ninja_position = (120.0, 120.0)
        entities = []

        graph_data = builder.build_graph(level_data, ninja_position, entities)

        # Should still work and produce 16-dimensional features
        self.assertIsNotNone(graph_data)
        self.assertEqual(graph_data.edge_features.shape[1], 16)

        # Trajectory features should be zeros or defaults when not provided
        trajectory_features = graph_data.edge_features[:, 9:16]

        # Should handle missing trajectory data gracefully
        self.assertFalse(np.any(np.isnan(trajectory_features)))
        self.assertFalse(np.any(np.isinf(trajectory_features)))

    def test_performance_integration(self):
        """Test performance characteristics of integrated system."""
        builder = GraphBuilder()

        # Create larger test scenario
        level_data = {'tiles': np.zeros((30, 30), dtype=int)}
        ninja_position = (300.0, 300.0)
        entities = []

        # Time graph building without trajectory features
        start_time = time.time()
        graph_data_basic = builder.build_graph(level_data, ninja_position, entities)
        basic_time = time.time() - start_time

        # Time graph building with trajectory features
        start_time = time.time()
        graph_data_enhanced = builder.build_graph(
            level_data, ninja_position, entities,
            ninja_velocity=(40.0, -30.0), ninja_state=3
        )
        enhanced_time = time.time() - start_time

        # Verify both produce valid results
        self.assertIsNotNone(graph_data_basic)
        self.assertIsNotNone(graph_data_enhanced)

        # Enhanced version should not be dramatically slower
        # (Allow up to 3x slower for trajectory calculations)
        self.assertLess(enhanced_time, basic_time * 3.0)

if __name__ == '__main__':
    unittest.main()
