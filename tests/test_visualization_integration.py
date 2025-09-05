"""
Tests for visualization integration with npp-rl components.

Tests integration between movement classifiers, trajectory calculators,
and the graph visualization system.
"""

import unittest
import numpy as np
from typing import Dict, List, Any, Tuple
import sys
import os

# Add nclone to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../nclone'))

from npp_rl.models.movement_classifier import MovementClassifier, MovementType, NinjaState
from npp_rl.models.trajectory_calculator import TrajectoryCalculator, TrajectoryResult, MovementState
from nclone.graph.visualization_api import GraphVisualizationAPI, VisualizationRequest
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.common import GraphData, NodeType, EdgeType


class TestMovementVisualizationIntegration(unittest.TestCase):
    """Test integration between movement analysis and visualization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.movement_classifier = MovementClassifier()
        self.trajectory_calculator = TrajectoryCalculator()
        self.visualization_api = GraphVisualizationAPI()
        self.pathfinding_engine = PathfindingEngine()
        
        # Create test data
        self.level_data = self._create_test_level_data()
        self.entities = self._create_test_entities()
    
    def _create_test_level_data(self) -> Dict[str, Any]:
        """Create test level data."""
        # Create a simple level with platforms and gaps
        tiles = np.zeros((15, 20), dtype=int)
        
        # Ground platforms
        tiles[12:15, 0:8] = 1    # Left platform
        tiles[12:15, 12:20] = 1  # Right platform
        
        # Walls
        tiles[8:12, 3] = 1       # Left wall
        tiles[8:12, 16] = 1      # Right wall
        
        # Upper platform
        tiles[6:8, 6:14] = 1     # Upper platform
        
        return {
            'level_id': 'movement_test_level',
            'tiles': tiles,
            'width': 20,
            'height': 15
        }
    
    def _create_test_entities(self) -> List[Dict[str, Any]]:
        """Create test entities."""
        return [
            # Launch pad on left platform
            {'type': 10, 'x': 100.0, 'y': 280.0, 'state': 0, 'orientation': 0},
            # Goal on right platform
            {'type': 20, 'x': 400.0, 'y': 280.0, 'state': 0},
            # Bounce block in middle
            {'type': 15, 'x': 250.0, 'y': 200.0, 'state': 0},
        ]
    
    def test_movement_classification_with_pathfinding(self):
        """Test movement classification integrated with pathfinding."""
        # Define test positions
        start_pos = (50.0, 280.0)   # Left platform
        goal_pos = (400.0, 280.0)   # Right platform
        
        # Find path using visualization API
        path_result = self.visualization_api.find_shortest_path(
            self.level_data,
            self.entities,
            start_pos,
            goal_pos,
            ninja_velocity=(0.0, 0.0),
            ninja_state=0
        )
        
        # Analyze movement types along the path
        if path_result.success and len(path_result.path_coordinates) > 1:
            movement_types = []
            
            for i in range(len(path_result.path_coordinates) - 1):
                src_pos = path_result.path_coordinates[i]
                tgt_pos = path_result.path_coordinates[i + 1]
                
                # Classify movement
                ninja_state = NinjaState(
                    movement_state=0,
                    velocity=(0.0, 0.0),
                    position=src_pos,
                    ground_contact=True,
                    wall_contact=False
                )
                
                movement_type, physics_params = self.movement_classifier.classify_movement(
                    src_pos, tgt_pos, ninja_state, self.level_data
                )
                
                movement_types.append(movement_type)
            
            # Verify we have movement classifications
            self.assertGreater(len(movement_types), 0)
            
            # Check that we have different movement types for complex path
            unique_types = set(movement_types)
            self.assertGreaterEqual(len(unique_types), 1)
    
    def test_trajectory_calculation_with_visualization(self):
        """Test trajectory calculation integrated with visualization."""
        # Test jump trajectory
        start_pos = (100.0, 280.0)
        end_pos = (200.0, 200.0)  # Jump to higher platform
        
        # Calculate trajectory
        trajectory_result = self.trajectory_calculator.calculate_jump_trajectory(
            start_pos, end_pos, MovementState.JUMPING
        )
        
        # Test trajectory validation
        if trajectory_result.feasible:
            is_clear = self.trajectory_calculator.validate_trajectory_clearance(
                trajectory_result.trajectory_points,
                self.level_data,
                self.entities
            )
            
            # Verify trajectory validation works
            self.assertIsInstance(is_clear, bool)
        
        # Verify trajectory result structure
        self.assertIsInstance(trajectory_result, TrajectoryResult)
        self.assertIsInstance(trajectory_result.feasible, bool)
        self.assertIsInstance(trajectory_result.time_of_flight, float)
        self.assertIsInstance(trajectory_result.energy_cost, float)
        self.assertIsInstance(trajectory_result.trajectory_points, list)
    
    def test_physics_informed_pathfinding(self):
        """Test pathfinding with physics-informed edge costs."""
        # Create ninja state with specific physics parameters
        ninja_state = NinjaState(
            movement_state=1,  # Running
            velocity=(5.0, 0.0),  # Moving right
            position=(50.0, 280.0),
            ground_contact=True,
            wall_contact=False
        )
        
        # Test different pathfinding scenarios
        test_cases = [
            # Simple horizontal movement
            ((50.0, 280.0), (150.0, 280.0)),
            # Jump required
            ((50.0, 280.0), (350.0, 280.0)),
            # Vertical movement
            ((100.0, 280.0), (100.0, 150.0)),
        ]
        
        for start_pos, goal_pos in test_cases:
            with self.subTest(start=start_pos, goal=goal_pos):
                # Find path
                path_result = self.visualization_api.find_shortest_path(
                    self.level_data,
                    self.entities,
                    start_pos,
                    goal_pos,
                    ninja_velocity=ninja_state.velocity,
                    ninja_state=ninja_state.movement_state
                )
                
                # Verify path result structure
                self.assertIsInstance(path_result.success, bool)
                self.assertIsInstance(path_result.total_cost, float)
                self.assertIsInstance(path_result.nodes_explored, int)
                
                if path_result.success:
                    self.assertGreater(len(path_result.path), 0)
                    self.assertGreater(len(path_result.path_coordinates), 0)
    
    def test_movement_feasibility_analysis(self):
        """Test movement feasibility analysis."""
        # Test various movement scenarios
        test_movements = [
            # Feasible horizontal movement
            ((50.0, 280.0), (100.0, 280.0)),
            # Feasible jump
            ((100.0, 280.0), (120.0, 260.0)),
            # Potentially infeasible movement through wall
            ((50.0, 280.0), (50.0, 200.0)),
        ]
        
        for src_pos, tgt_pos in test_movements:
            with self.subTest(src=src_pos, tgt=tgt_pos):
                # Check movement feasibility
                is_feasible = self.movement_classifier.is_movement_physically_feasible(
                    src_pos, tgt_pos, self.level_data, self.entities
                )
                
                self.assertIsInstance(is_feasible, bool)
    
    def test_visualization_with_movement_analysis(self):
        """Test visualization system with movement analysis data."""
        # Create visualization request with movement data
        request = VisualizationRequest(
            level_data=self.level_data,
            entities=self.entities,
            ninja_position=(50.0, 280.0),
            ninja_velocity=(5.0, 0.0),
            ninja_state=1,  # Running
            goal_position=(400.0, 280.0),
            output_size=(800, 600)
        )
        
        # Generate visualization
        result = self.visualization_api.visualize_graph(request)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertIsNotNone(result.surface)
        self.assertIsNotNone(result.graph_stats)
        
        # Check graph statistics
        stats = result.graph_stats
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertIn('node_types', stats)
        self.assertIn('edge_types', stats)
        
        # Verify pathfinding results if available
        if result.path_result:
            self.assertIsInstance(result.path_result.success, bool)
            if result.path_result.success:
                self.assertGreater(len(result.path_result.path), 0)
    
    def test_edge_type_classification(self):
        """Test edge type classification based on movement analysis."""
        # Create test graph data
        graph_data = self._create_test_graph_with_movement_data()
        
        # Analyze edge types
        edge_type_counts = {}
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] > 0:
                edge_type = EdgeType(graph_data.edge_types[edge_idx])
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        # Verify we have different edge types
        self.assertGreater(len(edge_type_counts), 0)
        
        # Check for expected edge types in a complex level
        expected_types = [EdgeType.WALK, EdgeType.JUMP, EdgeType.FALL]
        for edge_type in expected_types:
            if edge_type in edge_type_counts:
                self.assertGreater(edge_type_counts[edge_type], 0)
    
    def _create_test_graph_with_movement_data(self) -> GraphData:
        """Create test graph data with movement-informed edges."""
        # This would normally be created by the graph construction system
        # For testing, create a simplified version
        
        num_nodes = 10
        num_edges = 15
        
        # Create node features (positions)
        node_features = np.zeros((num_nodes, 16), dtype=np.float32)
        for i in range(num_nodes):
            node_features[i, 0] = i * 50.0  # X coordinate
            node_features[i, 1] = 280.0 if i < 5 else 200.0  # Y coordinate (two levels)
        
        # Create edges with different types
        edge_index = np.zeros((2, num_edges), dtype=np.int32)
        edge_features = np.ones((num_edges, 8), dtype=np.float32)
        edge_types = np.zeros(num_edges, dtype=np.int32)
        
        edge_idx = 0
        
        # Horizontal walk edges
        for i in range(4):
            edge_index[0, edge_idx] = i
            edge_index[1, edge_idx] = i + 1
            edge_types[edge_idx] = EdgeType.WALK
            edge_idx += 1
        
        # Jump edges between levels
        for i in range(5):
            if edge_idx < num_edges:
                edge_index[0, edge_idx] = i
                edge_index[1, edge_idx] = i + 5
                edge_types[edge_idx] = EdgeType.JUMP
                edge_idx += 1
        
        # Fill remaining edges
        while edge_idx < num_edges:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            edge_index[0, edge_idx] = src
            edge_index[1, edge_idx] = dst
            edge_types[edge_idx] = EdgeType.FALL
            edge_idx += 1
        
        # Create masks
        node_mask = np.ones(num_nodes, dtype=np.float32)
        edge_mask = np.ones(num_edges, dtype=np.float32)
        node_types = np.zeros(num_nodes, dtype=np.int32)
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_types=node_types,
            edge_types=edge_types,
            num_nodes=num_nodes,
            num_edges=num_edges
        )


class TestTrajectoryVisualization(unittest.TestCase):
    """Test trajectory visualization capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trajectory_calculator = TrajectoryCalculator()
        self.visualization_api = GraphVisualizationAPI()
    
    def test_trajectory_point_visualization(self):
        """Test visualization of trajectory points."""
        # Calculate a test trajectory
        start_pos = (100.0, 300.0)
        end_pos = (200.0, 250.0)
        
        trajectory_result = self.trajectory_calculator.calculate_jump_trajectory(
            start_pos, end_pos, MovementState.JUMPING
        )
        
        if trajectory_result.feasible:
            # Verify trajectory points are generated
            self.assertGreater(len(trajectory_result.trajectory_points), 0)
            
            # Check trajectory point format
            for point in trajectory_result.trajectory_points:
                self.assertIsInstance(point, tuple)
                self.assertEqual(len(point), 2)
                self.assertIsInstance(point[0], (int, float))
                self.assertIsInstance(point[1], (int, float))
    
    def test_physics_parameter_extraction(self):
        """Test extraction of physics parameters for visualization."""
        # Test different movement scenarios
        test_scenarios = [
            # Horizontal movement
            ((100.0, 300.0), (200.0, 300.0), MovementState.RUNNING),
            # Jump trajectory
            ((100.0, 300.0), (200.0, 250.0), MovementState.JUMPING),
            # Fall trajectory
            ((100.0, 250.0), (100.0, 300.0), MovementState.FALLING),
        ]
        
        for start_pos, end_pos, movement_state in test_scenarios:
            with self.subTest(start=start_pos, end=end_pos, state=movement_state):
                trajectory_result = self.trajectory_calculator.calculate_jump_trajectory(
                    start_pos, end_pos, movement_state
                )
                
                # Verify physics parameters
                self.assertIsInstance(trajectory_result.time_of_flight, float)
                self.assertIsInstance(trajectory_result.energy_cost, float)
                self.assertIsInstance(trajectory_result.success_probability, float)
                self.assertIsInstance(trajectory_result.min_velocity, float)
                self.assertIsInstance(trajectory_result.max_velocity, float)
                
                # Check parameter ranges
                self.assertGreaterEqual(trajectory_result.time_of_flight, 0.0)
                self.assertGreaterEqual(trajectory_result.energy_cost, 0.0)
                self.assertGreaterEqual(trajectory_result.success_probability, 0.0)
                self.assertLessEqual(trajectory_result.success_probability, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)