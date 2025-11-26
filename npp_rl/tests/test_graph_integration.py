"""Integration tests for graph-based path prediction system.

Tests verify that the adjacency graph from GraphBuilder is properly integrated
with pattern extraction, path prediction, and route discovery components.
"""

import numpy as np

# Import the modules we're testing
from npp_rl.path_prediction import (
    GeneralizedPatternExtractor,
    ProbabilisticPathPredictor,
    OnlineRouteDiscovery,
    graph_utils,
)


class TestGraphUtils:
    """Test graph utility functions."""

    def test_snap_position_to_graph_node(self):
        """Test snapping pixel positions to graph nodes."""
        # Create simple test adjacency graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0)],
        }
        
        # Test exact match
        result = graph_utils.snap_position_to_graph_node((6, 6), adjacency, None, threshold=12)
        assert result == (6, 6)
        
        # Test nearby position (should snap to (6, 6))
        result = graph_utils.snap_position_to_graph_node((8, 8), adjacency, None, threshold=12)
        assert result == (6, 6)
        
        # Test position too far away
        result = graph_utils.snap_position_to_graph_node((100, 100), adjacency, None, threshold=12)
        assert result is None

    def test_validate_path_on_graph(self):
        """Test path validation against graph."""
        # Create simple test adjacency graph (horizontal line)
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0), ((42, 6), 1.0)],
            (42, 6): [((30, 6), 1.0)],
        }
        
        # Test valid path
        waypoints = [(6, 6), (18, 6), (30, 6)]
        is_valid, distance = graph_utils.validate_path_on_graph(waypoints, adjacency, None)
        assert is_valid is True
        assert distance == 2.0  # Two edges, each cost 1.0
        
        # Test invalid path (gap between nodes)
        waypoints = [(6, 6), (42, 6)]  # Not directly connected
        is_valid, distance = graph_utils.validate_path_on_graph(waypoints, adjacency, None)
        assert is_valid is False
        assert distance == 0.0

    def test_compute_graph_path(self):
        """Test shortest path computation."""
        # Create simple test adjacency graph (horizontal line)
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0), ((42, 6), 1.0)],
            (42, 6): [((30, 6), 1.0)],
        }
        
        # Test path finding
        path = graph_utils.compute_graph_path((6, 6), (42, 6), adjacency, None)
        assert path is not None
        assert path == [(6, 6), (18, 6), (30, 6), (42, 6)]
        
        # Test unreachable nodes
        adjacency[(42, 6)] = []  # No outgoing edges from (42, 6)
        adjacency[(100, 100)] = []  # Isolated node
        path = graph_utils.compute_graph_path((6, 6), (100, 100), adjacency, None)
        assert path is None

    def test_repair_path_with_graph(self):
        """Test path repair functionality."""
        # Create simple test adjacency graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0), ((42, 6), 1.0)],
            (42, 6): [((30, 6), 1.0)],
        }
        
        # Test repairing path with gaps (should insert intermediate nodes)
        waypoints = [(6, 6), (42, 6)]  # Directly use graph nodes
        repaired = graph_utils.repair_path_with_graph(waypoints, adjacency, None, snap_threshold=24)
        
        # Should insert intermediate nodes to connect the path
        assert len(repaired) >= 4  # Should include intermediate nodes
        assert repaired[0] == (6, 6)  # Start node
        assert repaired[-1] == (42, 6)  # End node
        # Path should be: (6,6) -> (18,6) -> (30,6) -> (42,6)


class TestPatternExtractorGraphIntegration:
    """Test pattern extractor with graph data."""

    def test_pattern_extractor_with_graph(self):
        """Test that pattern extractor can use graph data."""
        extractor = GeneralizedPatternExtractor()
        
        # Create test graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0)],
        }
        
        # Set graph data (should not raise error)
        extractor.set_graph_data(adjacency, None)
        
        # Verify graph is set
        assert extractor.adjacency_graph is not None
        assert extractor.spatial_hash is None


class TestPathPredictorGraphIntegration:
    """Test path predictor with graph data."""

    def test_path_predictor_with_graph(self):
        """Test that path predictor can use graph data."""
        
        predictor = ProbabilisticPathPredictor(
            graph_feature_dim=256,
            tile_pattern_dim=64,
            entity_feature_dim=32,
        )
        
        # Create test graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0)],
        }
        
        # Set graph data (should not raise error)
        predictor.set_graph_data(adjacency, None)
        
        # Verify graph is set
        assert predictor.adjacency_graph is not None
        assert predictor.spatial_hash is None


class TestRouteDiscoveryGraphIntegration:
    """Test route discovery with graph data."""

    def test_route_discovery_with_graph(self):
        """Test that route discovery can use graph data."""
        discovery = OnlineRouteDiscovery()
        
        # Create test graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0)],
        }
        
        # Set graph data (should not raise error)
        discovery.set_graph_data(adjacency, None)
        
        # Verify graph is set
        assert discovery.adjacency_graph is not None
        assert discovery.spatial_hash is None

    def test_path_feasibility_scoring(self):
        """Test path feasibility scoring with graph."""
        discovery = OnlineRouteDiscovery()
        
        # Create test graph
        adjacency = {
            (6, 6): [((18, 6), 1.0)],
            (18, 6): [((6, 6), 1.0), ((30, 6), 1.0)],
            (30, 6): [((18, 6), 1.0)],
        }
        
        discovery.set_graph_data(adjacency, None)
        
        # Test valid path (should have high feasibility)
        valid_path = {
            "waypoints": [(6, 6), (18, 6), (30, 6)],
            "path_type": "test",
        }
        score = discovery._compute_path_feasibility_score(valid_path)
        assert score > 0.0
        
        # Test invalid path (gap in graph connectivity)
        invalid_path = {
            "waypoints": [(6, 6), (100, 100)],
            "path_type": "test",
        }
        score = discovery._compute_path_feasibility_score(invalid_path)
        assert score == 0.0


class TestBackwardCompatibility:
    """Test that components work without graph data (backward compatibility)."""

    def test_pattern_extractor_without_graph(self):
        """Pattern extractor should work without graph."""
        extractor = GeneralizedPatternExtractor()
        
        # Should have None graph initially
        assert extractor.adjacency_graph is None
        assert extractor.spatial_hash is None
        
        # Should be able to extract patterns without graph
        trajectory = [
            {"position": (100, 100)},
            {"position": (120, 100)},
        ]
        level_data = {"tile_data": np.zeros((10, 10)), "entities": []}
        
        # Should not raise error
        patterns = extractor.extract_tile_entity_patterns(trajectory, level_data)
        # May be empty due to test data, but shouldn't crash

    def test_path_predictor_without_graph(self):
        """Path predictor should work without graph."""
        predictor = ProbabilisticPathPredictor()
        
        # Should have None graph initially
        assert predictor.adjacency_graph is None
        assert predictor.spatial_hash is None

    def test_route_discovery_without_graph(self):
        """Route discovery should work without graph."""
        discovery = OnlineRouteDiscovery()
        
        # Should have None graph initially
        assert discovery.adjacency_graph is None
        assert discovery.spatial_hash is None
        
        # Feasibility scoring should default to 1.0 without graph
        test_path = {"waypoints": [(100, 100), (200, 200)]}
        score = discovery._compute_path_feasibility_score(test_path)
        assert score == 1.0  # No graph, assume feasible

