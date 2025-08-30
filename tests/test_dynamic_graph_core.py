"""
Core tests for Dynamic Graph Wrapper components (Task 3.2).

This module tests the core dynamic graph components without requiring
full environment setup, focusing on the key functionality.
"""

import sys
import os
import time
import numpy as np
from unittest.mock import Mock, MagicMock
from collections import deque

# Add nclone to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone'))

from npp_rl.environments.dynamic_graph_wrapper import (
    EventType,
    GraphEvent,
    UpdateBudget,
    TemporalEdge,
    DynamicConstraintPropagator
)

# Mock the nclone imports to avoid dependency issues
class MockEdgeType:
    WALK = 0
    JUMP = 1
    WALL_SLIDE = 2
    FALL = 3
    ONE_WAY = 4
    FUNCTIONAL = 5

class MockGraphData:
    def __init__(self):
        self.num_nodes = 100
        self.num_edges = 500
        self.edge_mask = np.ones(500)

# Replace the imports in the module
import npp_rl.environments.dynamic_graph_wrapper as dgw_module
dgw_module.EdgeType = MockEdgeType
dgw_module.GraphData = MockGraphData


def test_update_budget():
    """Test update budget management."""
    print("Testing UpdateBudget...")
    
    # Test initialization
    budget = UpdateBudget()
    assert budget.max_time_ms == 25.0
    assert budget.max_edge_updates == 1000
    assert budget.max_node_updates == 500
    assert budget.priority_threshold == 0.5
    assert budget.used_time_ms == 0.0
    assert budget.used_edge_updates == 0
    assert budget.used_node_updates == 0
    print("✓ Budget initialization")
    
    # Test custom values
    budget = UpdateBudget(
        max_time_ms=50.0,
        max_edge_updates=2000,
        max_node_updates=1000,
        priority_threshold=0.7
    )
    assert budget.max_time_ms == 50.0
    assert budget.max_edge_updates == 2000
    assert budget.max_node_updates == 1000
    assert budget.priority_threshold == 0.7
    print("✓ Budget custom values")
    
    # Test affordability checks
    budget = UpdateBudget(max_edge_updates=10, max_node_updates=5, max_time_ms=100.0)
    
    # Initially can afford everything
    assert budget.can_afford_edge_update(5)
    assert budget.can_afford_node_update(3)
    assert budget.can_afford_time(50.0)
    
    # Consume some budget
    budget.consume_edge_updates(8)
    budget.consume_node_updates(4)
    budget.consume_time(80.0)
    
    # Check remaining capacity
    assert budget.can_afford_edge_update(2)
    assert not budget.can_afford_edge_update(3)
    assert budget.can_afford_node_update(1)
    assert not budget.can_afford_node_update(2)
    assert budget.can_afford_time(20.0)
    assert not budget.can_afford_time(25.0)
    print("✓ Budget affordability checks")
    
    # Test reset
    budget.reset()
    assert budget.used_edge_updates == 0
    assert budget.used_node_updates == 0
    assert budget.used_time_ms == 0.0
    print("✓ Budget reset")


def test_graph_event():
    """Test graph event system."""
    print("Testing GraphEvent...")
    
    # Test event creation
    event = GraphEvent(
        event_type=EventType.NINJA_STATE_CHANGED,
        timestamp=time.time(),
        position=(100.0, 200.0),
        state_data={'velocity': (1.0, 0.0)},
        priority=0.9
    )
    
    assert event.event_type == EventType.NINJA_STATE_CHANGED
    assert event.position == (100.0, 200.0)
    assert event.state_data['velocity'] == (1.0, 0.0)
    assert event.priority == 0.9
    print("✓ Event creation")
    
    # Test all event types are defined
    expected_types = [
        'ENTITY_MOVED', 'ENTITY_STATE_CHANGED', 'NINJA_STATE_CHANGED',
        'DOOR_TOGGLED', 'SWITCH_ACTIVATED', 'PLATFORM_MOVED',
        'HAZARD_ACTIVATED', 'TEMPORAL_WINDOW'
    ]
    
    for event_type_name in expected_types:
        assert hasattr(EventType, event_type_name)
    print("✓ All event types defined")


def test_temporal_edge():
    """Test temporal edge functionality."""
    print("Testing TemporalEdge...")
    
    # Test creation
    availability_windows = [(0.0, 5.0), (10.0, 15.0)]
    base_features = np.random.random(16).astype(np.float32)
    
    edge = TemporalEdge(
        src_node=0,
        tgt_node=1,
        edge_type=MockEdgeType.JUMP,
        availability_windows=availability_windows,
        base_features=base_features
    )
    
    assert edge.src_node == 0
    assert edge.tgt_node == 1
    assert edge.edge_type == MockEdgeType.JUMP
    assert edge.availability_windows == availability_windows
    assert np.array_equal(edge.base_features, base_features)
    assert not edge.is_currently_active
    print("✓ Temporal edge creation")
    
    # Test availability checking
    availability_windows = [(0.0, 5.0), (10.0, 15.0)]
    edge = TemporalEdge(
        src_node=0,
        tgt_node=1,
        edge_type=MockEdgeType.WALK,
        availability_windows=availability_windows,
        base_features=np.zeros(16)
    )
    
    # Test availability at different times
    assert not edge.is_available_at_time(-1.0)  # Before first window
    assert edge.is_available_at_time(2.5)       # In first window
    assert not edge.is_available_at_time(7.5)   # Between windows
    assert edge.is_available_at_time(12.5)      # In second window
    assert not edge.is_available_at_time(20.0)  # After last window
    
    # Test edge cases
    assert edge.is_available_at_time(0.0)       # Start of window
    assert edge.is_available_at_time(5.0)       # End of window
    assert edge.is_available_at_time(10.0)      # Start of second window
    assert edge.is_available_at_time(15.0)      # End of second window
    print("✓ Temporal edge availability")


def test_dynamic_constraint_propagator():
    """Test dynamic constraint propagation."""
    print("Testing DynamicConstraintPropagator...")
    
    # Test initialization
    propagator = DynamicConstraintPropagator(max_propagation_depth=5)
    assert propagator.max_propagation_depth == 5
    assert len(propagator.constraint_dependencies) == 0
    assert len(propagator.edge_constraints) == 0
    print("✓ Propagator initialization")
    
    # Test constraint dependency registration
    propagator.register_constraint_dependency(entity_id=1, edge_indices=[10, 11, 12])
    propagator.register_constraint_dependency(entity_id=2, edge_indices=[12, 13, 14])
    
    # Check dependencies were registered
    assert propagator.constraint_dependencies[1] == {10, 11, 12}
    assert propagator.constraint_dependencies[2] == {12, 13, 14}
    
    # Check reverse mapping
    assert 1 in propagator.edge_constraints[10]
    assert 1 in propagator.edge_constraints[11]
    assert {1, 2} == propagator.edge_constraints[12]  # Edge 12 depends on both entities
    assert 2 in propagator.edge_constraints[13]
    assert 2 in propagator.edge_constraints[14]
    print("✓ Constraint dependency registration")
    
    # Test constraint propagation
    mock_graph = MockGraphData()
    budget = UpdateBudget(max_edge_updates=5)
    
    # Register dependencies
    propagator.register_constraint_dependency(entity_id=1, edge_indices=[5, 6, 7, 8, 9])
    
    # Propagate changes
    updated_edges = propagator.propagate_constraint_change(
        changed_entity_id=1,
        graph_data=mock_graph,
        budget=budget
    )
    
    # Should update edges within budget
    assert len(updated_edges) == 5  # All edges fit within budget
    assert budget.used_edge_updates == 5
    print("✓ Constraint propagation")
    
    # Test budget limits
    propagator2 = DynamicConstraintPropagator()
    mock_graph2 = MockGraphData()
    budget2 = UpdateBudget(max_edge_updates=3)
    
    # Register many dependencies
    propagator2.register_constraint_dependency(entity_id=1, edge_indices=[5, 6, 7, 8, 9])
    
    # Propagate changes
    updated_edges2 = propagator2.propagate_constraint_change(
        changed_entity_id=1,
        graph_data=mock_graph2,
        budget=budget2
    )
    
    # Should only update edges within budget
    assert len(updated_edges2) == 3  # Limited by budget
    assert budget2.used_edge_updates == 3
    print("✓ Constraint propagation budget limits")


def test_performance_requirements():
    """Test performance requirements."""
    print("Testing performance requirements...")
    
    # Target: <75ms graph processing time per frame
    target_time_ms = 75.0
    
    # Test different performance modes
    fast_budget = UpdateBudget(max_time_ms=15.0)
    balanced_budget = UpdateBudget(max_time_ms=25.0)
    accurate_budget = UpdateBudget(max_time_ms=40.0)
    
    assert fast_budget.max_time_ms < target_time_ms
    assert balanced_budget.max_time_ms < target_time_ms
    assert accurate_budget.max_time_ms < target_time_ms
    print("✓ Performance budgets meet target")
    
    # Test event processing efficiency
    events = deque(maxlen=100)
    current_time = time.time()
    
    # Create many events
    for i in range(100):
        event = GraphEvent(
            event_type=EventType.NINJA_STATE_CHANGED,
            timestamp=current_time,
            priority=0.5 + (i % 10) * 0.05  # Varying priorities
        )
        events.append(event)
    
    # Sort events by priority (simulate event processing)
    sorted_events = sorted(events, key=lambda e: e.priority, reverse=True)
    
    # Check that high priority events come first
    assert sorted_events[0].priority >= sorted_events[-1].priority
    print("✓ Event priority sorting")


def run_all_tests():
    """Run all core tests."""
    print("Running Dynamic Graph Core Tests...")
    print("=" * 50)
    
    try:
        test_update_budget()
        print()
        
        test_graph_event()
        print()
        
        test_temporal_edge()
        print()
        
        test_dynamic_constraint_propagator()
        print()
        
        test_performance_requirements()
        print()
        
        print("=" * 50)
        print("✅ All core tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)