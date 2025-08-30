"""
Test suite for Dynamic Graph Wrapper (Task 3.2).

This module tests the real-time graph adaptation system including:
- Event-driven graph updates
- Dynamic constraint propagation
- Computational budget management
- Temporal edge availability
- Performance benchmarks
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import gymnasium as gym
from gymnasium.spaces import Dict as SpacesDict, Box

from npp_rl.environments.dynamic_graph_wrapper import (
    DynamicGraphWrapper,
    EventType,
    GraphEvent,
    UpdateBudget,
    TemporalEdge,
    DynamicConstraintPropagator
)
from npp_rl.environments.dynamic_graph_integration import (
    create_dynamic_graph_env,
    validate_dynamic_graph_environment,
    benchmark_dynamic_graph_performance,
    DynamicGraphProfiler
)
from nclone.graph.graph_builder import EdgeType, GraphData


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


class TestUpdateBudget:
    """Test update budget management."""
    
    def test_budget_initialization(self):
        """Test budget initialization with default values."""
        budget = UpdateBudget()
        
        assert budget.max_time_ms == 25.0
        assert budget.max_edge_updates == 1000
        assert budget.max_node_updates == 500
        assert budget.priority_threshold == 0.5
        assert budget.used_time_ms == 0.0
        assert budget.used_edge_updates == 0
        assert budget.used_node_updates == 0
    
    def test_budget_custom_values(self):
        """Test budget initialization with custom values."""
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
    
    def test_budget_affordability_checks(self):
        """Test budget affordability checking."""
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
    
    def test_budget_reset(self):
        """Test budget reset functionality."""
        budget = UpdateBudget()
        
        # Consume some budget
        budget.consume_edge_updates(100)
        budget.consume_node_updates(50)
        budget.consume_time(20.0)
        
        assert budget.used_edge_updates == 100
        assert budget.used_node_updates == 50
        assert budget.used_time_ms == 20.0
        
        # Reset budget
        budget.reset()
        
        assert budget.used_edge_updates == 0
        assert budget.used_node_updates == 0
        assert budget.used_time_ms == 0.0


class TestGraphEvent:
    """Test graph event system."""
    
    def test_event_creation(self):
        """Test graph event creation."""
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
    
    def test_event_types(self):
        """Test all event types are defined."""
        expected_types = [
            'ENTITY_MOVED', 'ENTITY_STATE_CHANGED', 'NINJA_STATE_CHANGED',
            'DOOR_TOGGLED', 'SWITCH_ACTIVATED', 'PLATFORM_MOVED',
            'HAZARD_ACTIVATED', 'TEMPORAL_WINDOW'
        ]
        
        for event_type_name in expected_types:
            assert hasattr(EventType, event_type_name)


class TestTemporalEdge:
    """Test temporal edge functionality."""
    
    def test_temporal_edge_creation(self):
        """Test temporal edge creation."""
        availability_windows = [(0.0, 5.0), (10.0, 15.0)]
        base_features = np.random.random(16).astype(np.float32)
        
        edge = TemporalEdge(
            src_node=0,
            tgt_node=1,
            edge_type=EdgeType.JUMP,
            availability_windows=availability_windows,
            base_features=base_features
        )
        
        assert edge.src_node == 0
        assert edge.tgt_node == 1
        assert edge.edge_type == EdgeType.JUMP
        assert edge.availability_windows == availability_windows
        assert np.array_equal(edge.base_features, base_features)
        assert not edge.is_currently_active
    
    def test_temporal_availability(self):
        """Test temporal edge availability checking."""
        availability_windows = [(0.0, 5.0), (10.0, 15.0)]
        edge = TemporalEdge(
            src_node=0,
            tgt_node=1,
            edge_type=EdgeType.WALK,
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


class TestDynamicConstraintPropagator:
    """Test dynamic constraint propagation."""
    
    def test_propagator_initialization(self):
        """Test constraint propagator initialization."""
        propagator = DynamicConstraintPropagator(max_propagation_depth=5)
        
        assert propagator.max_propagation_depth == 5
        assert len(propagator.constraint_dependencies) == 0
        assert len(propagator.edge_constraints) == 0
    
    def test_constraint_dependency_registration(self):
        """Test constraint dependency registration."""
        propagator = DynamicConstraintPropagator()
        
        # Register dependencies
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
    
    def test_constraint_propagation(self):
        """Test constraint change propagation."""
        propagator = DynamicConstraintPropagator()
        
        # Set up mock graph data
        mock_graph = Mock(spec=GraphData)
        mock_graph.num_edges = 20
        
        # Set up budget
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
    
    def test_constraint_propagation_budget_limit(self):
        """Test constraint propagation respects budget limits."""
        propagator = DynamicConstraintPropagator()
        
        # Set up mock graph data
        mock_graph = Mock(spec=GraphData)
        mock_graph.num_edges = 20
        
        # Set up limited budget
        budget = UpdateBudget(max_edge_updates=3)
        
        # Register many dependencies
        propagator.register_constraint_dependency(entity_id=1, edge_indices=[5, 6, 7, 8, 9])
        
        # Propagate changes
        updated_edges = propagator.propagate_constraint_change(
            changed_entity_id=1,
            graph_data=mock_graph,
            budget=budget
        )
        
        # Should only update edges within budget
        assert len(updated_edges) == 3  # Limited by budget
        assert budget.used_edge_updates == 3


class TestDynamicGraphWrapper:
    """Test dynamic graph wrapper functionality."""
    
    def test_wrapper_initialization(self):
        """Test dynamic graph wrapper initialization."""
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
    
    def test_observation_space_extension(self):
        """Test observation space extension for dynamic graph metadata."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env)
        
        # Check that dynamic graph metadata was added
        assert hasattr(wrapper.env, 'observation_space')
        if hasattr(wrapper.env.observation_space, 'spaces'):
            assert 'dynamic_graph_metadata' in wrapper.env.observation_space.spaces
    
    def test_reset_functionality(self):
        """Test environment reset with dynamic graph state."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
        
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
    
    def test_step_functionality(self):
        """Test environment step with dynamic graph updates."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
        
        # Reset environment
        wrapper.reset()
        
        # Take a step
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
    
    def test_event_detection_and_queuing(self):
        """Test environmental change detection and event queuing."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
        
        # Reset environment
        wrapper.reset()
        
        # Mock ninja state change
        obs = {
            'ninja_position': np.array([150.0, 250.0]),
            'ninja_velocity': np.array([2.0, 1.0]),
            'entities': np.zeros((10, 5))
        }
        info = {}
        
        # Detect changes (this would normally be called internally)
        initial_queue_size = len(wrapper.event_queue)
        wrapper._detect_environmental_changes(obs, info)
        
        # Should have queued ninja state change event
        assert len(wrapper.event_queue) > initial_queue_size
    
    def test_temporal_edge_management(self):
        """Test temporal edge addition and management."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env)
        
        # Add temporal edge
        availability_windows = [(0.0, 5.0), (10.0, 15.0)]
        base_features = np.random.random(16).astype(np.float32)
        
        edge_id = wrapper.add_temporal_edge(
            src_node=0,
            tgt_node=1,
            edge_type=EdgeType.JUMP,
            availability_windows=availability_windows,
            base_features=base_features
        )
        
        assert edge_id in wrapper.temporal_edges
        assert len(wrapper.temporal_edges) == 1
        
        # Remove temporal edge
        wrapper.remove_temporal_edge(edge_id)
        assert edge_id not in wrapper.temporal_edges
        assert len(wrapper.temporal_edges) == 0
    
    def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
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


class TestDynamicGraphIntegration:
    """Test dynamic graph integration utilities."""
    
    @patch('npp_rl.environments.dynamic_graph_integration.VectorizationWrapper')
    def test_create_dynamic_graph_env(self, mock_vectorization_wrapper):
        """Test dynamic graph environment creation."""
        # Mock the base environment
        mock_base_env = MockEnvironment()
        mock_vectorization_wrapper.return_value = mock_base_env
        
        # Create dynamic graph environment
        env = create_dynamic_graph_env(
            env_kwargs={'test_param': 'test_value'},
            enable_dynamic_updates=True,
            performance_mode='balanced'
        )
        
        # Check that wrapper was applied
        assert isinstance(env, DynamicGraphWrapper)
        assert env.enable_dynamic_updates
        
        # Check that vectorization wrapper was called with correct args
        mock_vectorization_wrapper.assert_called_once()
        call_args = mock_vectorization_wrapper.call_args[0][0]
        assert call_args['test_param'] == 'test_value'
        assert call_args['use_graph_obs'] is True
    
    def test_performance_mode_configurations(self):
        """Test different performance mode configurations."""
        mock_env = MockEnvironment()
        
        # Test fast mode
        with patch('npp_rl.environments.dynamic_graph_integration.VectorizationWrapper', return_value=mock_env):
            env_fast = create_dynamic_graph_env(performance_mode='fast')
            assert env_fast.update_budget.max_time_ms == 15.0
            assert env_fast.update_budget.max_edge_updates == 500
        
        # Test balanced mode
        with patch('npp_rl.environments.dynamic_graph_integration.VectorizationWrapper', return_value=mock_env):
            env_balanced = create_dynamic_graph_env(performance_mode='balanced')
            assert env_balanced.update_budget.max_time_ms == 25.0
            assert env_balanced.update_budget.max_edge_updates == 1000
        
        # Test accurate mode
        with patch('npp_rl.environments.dynamic_graph_integration.VectorizationWrapper', return_value=mock_env):
            env_accurate = create_dynamic_graph_env(performance_mode='accurate')
            assert env_accurate.update_budget.max_time_ms == 40.0
            assert env_accurate.update_budget.max_edge_updates == 2000
    
    def test_environment_validation(self):
        """Test dynamic graph environment validation."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env)
        
        # Should pass validation
        assert validate_dynamic_graph_environment(wrapper)
        
        # Test with non-dynamic environment
        assert not validate_dynamic_graph_environment(mock_env)


class TestDynamicGraphProfiler:
    """Test dynamic graph profiler."""
    
    def test_profiler_basic_functionality(self):
        """Test basic profiler functionality."""
        profiler = DynamicGraphProfiler()
        
        # Start profiling
        profiler.start_profile('test_profile')
        assert profiler.current_profile is not None
        assert profiler.current_profile['name'] == 'test_profile'
        
        # Record operations
        profiler.record_operation('operation1', 10.5, {'param': 'value'})
        profiler.record_operation('operation2', 5.2)
        
        # End profiling
        profiler.end_profile()
        assert profiler.current_profile is None
        assert 'test_profile' in profiler.profiles
    
    def test_profile_summary(self):
        """Test profile summary generation."""
        profiler = DynamicGraphProfiler()
        
        # Create a profile with operations
        profiler.start_profile('summary_test')
        profiler.record_operation('op1', 10.0)
        profiler.record_operation('op1', 15.0)
        profiler.record_operation('op2', 5.0)
        profiler.end_profile()
        
        # Get summary
        summary = profiler.get_profile_summary('summary_test')
        
        assert summary is not None
        assert 'operation_summaries' in summary
        assert 'op1' in summary['operation_summaries']
        assert 'op2' in summary['operation_summaries']
        
        # Check op1 summary
        op1_summary = summary['operation_summaries']['op1']
        assert op1_summary['count'] == 2
        assert op1_summary['total_time_ms'] == 25.0
        assert op1_summary['avg_time_ms'] == 12.5
        assert op1_summary['max_time_ms'] == 15.0


class TestPerformanceBenchmarks:
    """Test performance benchmarks and requirements."""
    
    def test_update_budget_performance_target(self):
        """Test that update budgets meet performance targets."""
        # Target: <75ms graph processing time per frame
        target_time_ms = 75.0
        
        # Test different performance modes
        fast_budget = UpdateBudget(max_time_ms=15.0)
        balanced_budget = UpdateBudget(max_time_ms=25.0)
        accurate_budget = UpdateBudget(max_time_ms=40.0)
        
        assert fast_budget.max_time_ms < target_time_ms
        assert balanced_budget.max_time_ms < target_time_ms
        assert accurate_budget.max_time_ms < target_time_ms
    
    def test_event_processing_efficiency(self):
        """Test event processing efficiency."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
        
        # Create many events
        current_time = time.time()
        for i in range(100):
            event = GraphEvent(
                event_type=EventType.NINJA_STATE_CHANGED,
                timestamp=current_time,
                priority=0.5 + (i % 10) * 0.05  # Varying priorities
            )
            wrapper._queue_event(event)
        
        # Process events with limited budget
        wrapper.update_budget.reset()
        wrapper.update_budget.max_time_ms = 10.0  # Very limited budget
        
        start_time = time.time()
        wrapper._process_event_queue()
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Should respect time budget
        assert processing_time_ms <= wrapper.update_budget.max_time_ms * 2  # Allow some overhead
        
        # Should have processed some events
        stats = wrapper.get_performance_stats()
        assert stats['events_processed'] > 0 or stats['events_skipped'] > 0
    
    @pytest.mark.slow
    def test_real_time_performance_benchmark(self):
        """Test real-time performance benchmark (marked as slow test)."""
        mock_env = MockEnvironment()
        wrapper = DynamicGraphWrapper(env=mock_env, enable_dynamic_updates=True)
        
        # Run benchmark
        results = benchmark_dynamic_graph_performance(
            env=wrapper,
            num_steps=100,  # Reduced for testing
            target_fps=60.0
        )
        
        # Check results structure
        expected_keys = [
            'total_steps', 'total_time_s', 'avg_step_time_ms',
            'max_step_time_ms', 'min_step_time_ms', 'target_step_time_ms',
            'performance_ratio', 'meets_target_fps', 'graph_stats'
        ]
        
        for key in expected_keys:
            assert key in results
        
        # Performance should be reasonable
        assert results['avg_step_time_ms'] > 0
        assert results['total_steps'] == 100
        
        # Log results for analysis
        print(f"Benchmark results: {results}")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])