"""
Simple integration test for Dynamic Graph Wrapper (Task 3.2).

This test validates the integration without requiring full environment setup.
"""

import sys
import os
import time
import numpy as np
from unittest.mock import Mock, MagicMock
import gymnasium as gym
from gymnasium.spaces import Dict as SpacesDict, Box

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone'))

from npp_rl.environments.dynamic_graph_wrapper import (
    DynamicGraphWrapper,
    EventType,
    GraphEvent,
    UpdateBudget,
    TemporalEdge,
    DynamicConstraintPropagator
)


class MockEnvironment(gym.Env):
    """Mock environment for testing dynamic graph wrapper."""
    
    def __init__(self):
        self.observation_space = SpacesDict({
            'ninja_position': Box(low=0, high=1000, shape=(2,), dtype=np.float32),
            'ninja_velocity': Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'entities': Box(low=0, high=1, shape=(10, 5), dtype=np.float32),
            'level_data': Box(low=0, high=1, shape=(100,), dtype=np.float32)
        })
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Mock state
        self.ninja_pos = np.array([100.0, 200.0], dtype=np.float32)
        self.ninja_vel = np.array([1.0, 0.0], dtype=np.float32)
        self.entities = np.zeros((10, 5), dtype=np.float32)
        self.level_data = np.random.random(100).astype(np.float32)
        
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.step_count = 0
        self.ninja_pos = np.array([100.0, 200.0], dtype=np.float32)
        self.ninja_vel = np.array([1.0, 0.0], dtype=np.float32)
        
        obs = {
            'ninja_position': self.ninja_pos,
            'ninja_velocity': self.ninja_vel,
            'entities': self.entities,
            'level_data': self.level_data
        }
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Simulate ninja movement
        self.ninja_pos += self.ninja_vel
        
        # Occasionally change velocity to trigger events
        if self.step_count % 10 == 0:
            self.ninja_vel = np.random.uniform(-2, 2, 2).astype(np.float32)
        
        obs = {
            'ninja_position': self.ninja_pos,
            'ninja_velocity': self.ninja_vel,
            'entities': self.entities,
            'level_data': self.level_data
        }
        
        reward = 0.1
        terminated = self.step_count >= 100
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def get_current_state(self):
        """Mock method for graph building."""
        return {
            'level_data': {'tiles': self.level_data},
            'ninja_position': tuple(self.ninja_pos),
            'ninja_velocity': tuple(self.ninja_vel),
            'ninja_state': 1,  # Running state
            'entities': []
        }


def test_dynamic_graph_wrapper_basic():
    """Test basic dynamic graph wrapper functionality."""
    print("Testing DynamicGraphWrapper basic functionality...")
    
    mock_env = MockEnvironment()
    
    wrapper = DynamicGraphWrapper(
        env=mock_env,
        enable_dynamic_updates=True,
        event_buffer_size=50,
        temporal_window_size=5.0
    )
    
    assert wrapper.enable_dynamic_updates
    assert wrapper.temporal_window_size == 5.0
    assert wrapper.event_queue.maxlen == 50
    assert wrapper.graph_builder is not None
    assert wrapper.constraint_propagator is not None
    print("✓ Wrapper initialization")
    
    # Test reset
    obs, info = wrapper.reset()
    
    # Check that observation includes dynamic graph metadata
    if isinstance(obs, dict):
        assert 'dynamic_graph_metadata' in obs
        metadata = obs['dynamic_graph_metadata']
        assert len(metadata) == 10
        assert isinstance(metadata, np.ndarray)
    
    # Check that dynamic state was reset
    assert len(wrapper.event_queue) == 0
    assert len(wrapper.processed_events) == 0
    assert len(wrapper.temporal_edges) == 0
    print("✓ Reset functionality")
    
    # Test step
    action = mock_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    
    # Check that observation includes dynamic graph metadata
    if isinstance(obs, dict):
        assert 'dynamic_graph_metadata' in obs
        metadata = obs['dynamic_graph_metadata']
        assert len(metadata) == 10
    
    # Check that performance stats were updated
    stats = wrapper.get_performance_stats()
    assert stats['total_updates'] >= 1
    print("✓ Step functionality")


def test_temporal_edge_management():
    """Test temporal edge management."""
    print("Testing temporal edge management...")
    
    mock_env = MockEnvironment()
    wrapper = DynamicGraphWrapper(env=mock_env)
    
    # Add temporal edge
    availability_windows = [(0.0, 5.0), (10.0, 15.0)]
    base_features = np.random.random(16).astype(np.float32)
    
    edge_id = wrapper.add_temporal_edge(
        src_node=0,
        tgt_node=1,
        edge_type=wrapper.graph_builder.__class__.__dict__.get('EdgeType', type('EdgeType', (), {'JUMP': 1})).JUMP,
        availability_windows=availability_windows,
        base_features=base_features
    )
    
    assert edge_id in wrapper.temporal_edges
    assert len(wrapper.temporal_edges) == 1
    print("✓ Temporal edge addition")
    
    # Remove temporal edge
    wrapper.remove_temporal_edge(edge_id)
    assert edge_id not in wrapper.temporal_edges
    assert len(wrapper.temporal_edges) == 0
    print("✓ Temporal edge removal")


def test_performance_stats():
    """Test performance statistics tracking."""
    print("Testing performance statistics...")
    
    mock_env = MockEnvironment()
    wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
    
    # Reset and take some steps
    wrapper.reset()
    for _ in range(5):
        action = mock_env.action_space.sample()
        wrapper.step(action)
    
    # Check performance stats
    stats = wrapper.get_performance_stats()
    
    expected_keys = [
        'total_updates', 'avg_update_time_ms', 'budget_exceeded_count',
        'events_processed', 'events_skipped'
    ]
    
    for key in expected_keys:
        assert key in stats
    
    assert stats['total_updates'] >= 5
    assert stats['avg_update_time_ms'] >= 0.0
    print("✓ Performance statistics")


def test_event_queuing():
    """Test event queuing functionality."""
    print("Testing event queuing...")
    
    mock_env = MockEnvironment()
    wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
    
    # Reset environment
    wrapper.reset()
    
    # Create test event
    event = GraphEvent(
        event_type=EventType.NINJA_STATE_CHANGED,
        timestamp=time.time(),
        position=(100.0, 200.0),
        state_data={'velocity': (1.0, 0.0)},
        priority=0.9
    )
    
    # Queue event
    initial_queue_size = len(wrapper.event_queue)
    wrapper._queue_event(event)
    
    # Should have queued the event
    assert len(wrapper.event_queue) > initial_queue_size
    print("✓ Event queuing")


def test_performance_benchmarks():
    """Test performance benchmarks."""
    print("Testing performance benchmarks...")
    
    mock_env = MockEnvironment()
    wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
    
    # Run a small benchmark
    wrapper.reset()
    
    start_time = time.time()
    num_steps = 50
    
    for step in range(num_steps):
        action = mock_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        
        if terminated or truncated:
            wrapper.reset()
    
    total_time = time.time() - start_time
    avg_step_time_ms = (total_time / num_steps) * 1000
    
    # Performance should be reasonable (less than target 75ms per step)
    target_step_time_ms = 75.0
    print(f"Average step time: {avg_step_time_ms:.2f}ms (target: {target_step_time_ms}ms)")
    
    # This is a very lenient check since we're using mocks
    assert avg_step_time_ms < target_step_time_ms * 10  # Allow 10x overhead for mocking
    print("✓ Performance benchmark")


def run_integration_tests():
    """Run all integration tests."""
    print("Running Dynamic Graph Integration Tests...")
    print("=" * 60)
    
    try:
        test_dynamic_graph_wrapper_basic()
        print()
        
        test_temporal_edge_management()
        print()
        
        test_performance_stats()
        print()
        
        test_event_queuing()
        print()
        
        test_performance_benchmarks()
        print()
        
        print("=" * 60)
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)