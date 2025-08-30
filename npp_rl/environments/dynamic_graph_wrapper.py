"""
Dynamic Graph Wrapper for Real-Time Graph Adaptation.

This module implements efficient real-time graph updates for dynamic environments,
providing event-driven graph modifications and incremental update mechanisms
while maintaining computational performance constraints.

Key Features:
- Event-driven graph updates triggered by environmental changes
- Incremental edge activation/deactivation based on dynamic constraints
- Priority-based update systems with computational budget management
- Temporal edge availability windows for time-dependent traversability
- Efficient graph modification algorithms optimized for real-time performance
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
from collections import deque, defaultdict
import numpy as np
import gymnasium as gym

from nclone.graph.graph_builder import GraphBuilder, GraphData, EdgeType, E_MAX_EDGES
class EventType(IntEnum):
    """Types of events that can trigger graph updates."""
    ENTITY_MOVED = 0          # Entity position changed
    ENTITY_STATE_CHANGED = 1  # Entity state/activation changed
    NINJA_STATE_CHANGED = 2   # Ninja physics state changed
    DOOR_TOGGLED = 3          # Door opened/closed
    SWITCH_ACTIVATED = 4      # Switch pressed/released
    PLATFORM_MOVED = 5        # Moving platform position changed
    HAZARD_ACTIVATED = 6      # Hazard became active/inactive
    TEMPORAL_WINDOW = 7       # Time-based edge availability changed


@dataclass
class GraphEvent:
    """Represents an event that may require graph updates."""
    event_type: EventType
    timestamp: float
    entity_id: Optional[int] = None
    position: Optional[Tuple[float, float]] = None
    state_data: Optional[Dict[str, Any]] = None
    priority: float = 1.0  # Higher values = higher priority
    affected_region: Optional[Tuple[int, int, int, int]] = None  # (min_row, min_col, max_row, max_col)


@dataclass
class UpdateBudget:
    """Manages computational budget for graph updates."""
    max_time_ms: float = 25.0  # Maximum time per frame for graph updates
    max_edge_updates: int = 1000  # Maximum edge updates per frame
    max_node_updates: int = 500   # Maximum node updates per frame
    priority_threshold: float = 0.5  # Minimum priority for processing
    
    # Runtime tracking
    used_time_ms: float = 0.0
    used_edge_updates: int = 0
    used_node_updates: int = 0
    
    def reset(self):
        """Reset budget counters for new frame."""
        self.used_time_ms = 0.0
        self.used_edge_updates = 0
        self.used_node_updates = 0
    
    def can_afford_edge_update(self, count: int = 1) -> bool:
        """Check if we can afford edge updates."""
        return self.used_edge_updates + count <= self.max_edge_updates
    
    def can_afford_node_update(self, count: int = 1) -> bool:
        """Check if we can afford node updates."""
        return self.used_node_updates + count <= self.max_node_updates
    
    def can_afford_time(self, time_ms: float) -> bool:
        """Check if we can afford time cost."""
        return self.used_time_ms + time_ms <= self.max_time_ms
    
    def consume_edge_updates(self, count: int):
        """Consume edge update budget."""
        self.used_edge_updates += count
    
    def consume_node_updates(self, count: int):
        """Consume node update budget."""
        self.used_node_updates += count
    
    def consume_time(self, time_ms: float):
        """Consume time budget."""
        self.used_time_ms += time_ms


@dataclass
class TemporalEdge:
    """Represents an edge with temporal availability constraints."""
    src_node: int
    tgt_node: int
    edge_type: EdgeType
    availability_windows: List[Tuple[float, float]]  # List of (start_time, end_time) windows
    base_features: np.ndarray
    is_currently_active: bool = False
    
    def is_available_at_time(self, timestamp: float) -> bool:
        """Check if edge is available at given timestamp."""
        for start_time, end_time in self.availability_windows:
            if start_time <= timestamp <= end_time:
                return True
        return False


class DynamicConstraintPropagator:
    """
    Handles dynamic constraint propagation with priority-based updates.
    
    This system efficiently propagates constraint changes through the graph
    while respecting computational budgets and maintaining real-time performance.
    """
    
    def __init__(self, max_propagation_depth: int = 3):
        """
        Initialize constraint propagator.
        
        Args:
            max_propagation_depth: Maximum depth for constraint propagation
        """
        self.max_propagation_depth = max_propagation_depth
        self.constraint_dependencies = defaultdict(set)  # entity_id -> set of dependent edges
        self.edge_constraints = {}  # edge_id -> set of constraint entity_ids
        
    def register_constraint_dependency(self, entity_id: int, edge_indices: List[int]):
        """Register which edges depend on an entity's state."""
        self.constraint_dependencies[entity_id].update(edge_indices)
        for edge_idx in edge_indices:
            if edge_idx not in self.edge_constraints:
                self.edge_constraints[edge_idx] = set()
            self.edge_constraints[edge_idx].add(entity_id)
    
    def propagate_constraint_change(
        self,
        changed_entity_id: int,
        graph_data: GraphData,
        budget: UpdateBudget
    ) -> List[int]:
        """
        Propagate constraint changes through dependent edges.
        
        Args:
            changed_entity_id: ID of entity whose constraints changed
            graph_data: Current graph data
            budget: Computational budget for updates
            
        Returns:
            List of edge indices that were updated
        """
        updated_edges = []
        
        if changed_entity_id not in self.constraint_dependencies:
            return updated_edges
        
        # Get directly affected edges
        affected_edges = list(self.constraint_dependencies[changed_entity_id])
        
        # Sort by priority (edges closer to ninja get higher priority)
        # This is a simplified priority - in practice, you'd use more sophisticated metrics
        affected_edges.sort(key=lambda x: self._calculate_edge_priority(x, graph_data))
        
        for edge_idx in affected_edges:
            if not budget.can_afford_edge_update():
                break
                
            # Update edge based on new constraints
            if self._update_edge_constraints(edge_idx, graph_data):
                updated_edges.append(edge_idx)
                budget.consume_edge_updates(1)
        
        return updated_edges
    
    def _calculate_edge_priority(self, edge_idx: int, graph_data: GraphData) -> float:
        """Calculate priority for edge updates (higher = more important)."""
        # Simple priority based on distance from ninja
        # In practice, this could consider path criticality, recent usage, etc.
        return 1.0  # Placeholder implementation
    
    def _update_edge_constraints(self, edge_idx: int, graph_data: GraphData) -> bool:
        """Update edge based on current constraint states."""
        # Placeholder for constraint evaluation logic
        # This would check entity states and update edge availability
        return True


class DynamicGraphWrapper(gym.Wrapper):
    """
    Environment wrapper that provides real-time graph adaptation capabilities.
    
    This wrapper maintains a dynamic graph representation that efficiently updates
    in response to environmental changes while respecting computational budgets.
    """
    
    def __init__(
        self,
        env: gym.Env,
        enable_dynamic_updates: bool = True,
        update_budget: Optional[UpdateBudget] = None,
        event_buffer_size: int = 100,
        temporal_window_size: float = 10.0  # seconds
    ):
        """
        Initialize dynamic graph wrapper.
        
        Args:
            env: Base environment to wrap
            enable_dynamic_updates: Whether to enable dynamic graph updates
            update_budget: Computational budget for updates
            event_buffer_size: Maximum size of event buffer
            temporal_window_size: Size of temporal window for edge availability
        """
        super().__init__(env)
        
        self.enable_dynamic_updates = enable_dynamic_updates
        self.update_budget = update_budget or UpdateBudget()
        self.temporal_window_size = temporal_window_size
        
        # Core components
        self.graph_builder = GraphBuilder()
        self.constraint_propagator = DynamicConstraintPropagator()
        
        # Event management
        self.event_queue = deque(maxlen=event_buffer_size)
        self.processed_events = set()  # Track processed event IDs to avoid duplicates
        
        # Graph state management
        self.current_graph = None
        self.last_full_rebuild_time = 0.0
        self.full_rebuild_interval = 1.0  # Rebuild full graph every N seconds
        
        # Temporal edge management
        self.temporal_edges = {}  # edge_id -> TemporalEdge
        self.active_temporal_edges = set()
        
        # Performance tracking
        self.update_stats = {
            'total_updates': 0,
            'avg_update_time_ms': 0.0,
            'budget_exceeded_count': 0,
            'events_processed': 0,
            'events_skipped': 0
        }
        
        # Initialize graph observation space extension
        self._extend_observation_space_for_dynamic_graph()
    
    def _extend_observation_space_for_dynamic_graph(self):
        """Extend observation space to include dynamic graph metadata."""
        if hasattr(self.env, 'observation_space') and hasattr(self.env.observation_space, 'spaces'):
            # Add dynamic graph metadata to observation space
            dynamic_graph_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(10,),  # Metadata: update_time, budget_usage, active_edges, etc.
                dtype=np.float32
            )
            self.env.observation_space.spaces['dynamic_graph_metadata'] = dynamic_graph_space
    
    def reset(self, **kwargs):
        """Reset environment and initialize dynamic graph state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset dynamic graph state
        self.event_queue.clear()
        self.processed_events.clear()
        self.temporal_edges.clear()
        self.active_temporal_edges.clear()
        self.last_full_rebuild_time = time.time()
        
        # Build initial graph
        if self.enable_dynamic_updates:
            self._rebuild_full_graph()
        
        # Add dynamic graph metadata to observation
        if isinstance(obs, dict):
            obs['dynamic_graph_metadata'] = self._get_dynamic_graph_metadata()
        
        return obs, info
    
    def step(self, action):
        """Step environment and update dynamic graph."""
        start_time = time.time()
        
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update dynamic graph if enabled
        if self.enable_dynamic_updates and self.current_graph is not None:
            self.update_budget.reset()
            
            # Detect and queue environmental changes
            self._detect_environmental_changes(obs, info)
            
            # Process queued events within budget
            self._process_event_queue()
            
            # Update temporal edges
            self._update_temporal_edges()
            
            # Check if full rebuild is needed
            current_time = time.time()
            if current_time - self.last_full_rebuild_time > self.full_rebuild_interval:
                self._rebuild_full_graph()
                self.last_full_rebuild_time = current_time
        
        # Add dynamic graph metadata to observation
        if isinstance(obs, dict):
            obs['dynamic_graph_metadata'] = self._get_dynamic_graph_metadata()
        
        # Update performance stats
        update_time_ms = (time.time() - start_time) * 1000
        self._update_performance_stats(update_time_ms)
        
        return obs, reward, terminated, truncated, info
    
    def _detect_environmental_changes(self, obs: Dict[str, Any], info: Dict[str, Any]):
        """Detect environmental changes and queue relevant events."""
        current_time = time.time()
        
        # Detect ninja state changes
        if 'ninja_position' in obs and 'ninja_velocity' in obs:
            ninja_pos = obs['ninja_position']
            ninja_vel = obs['ninja_velocity']
            
            # Check if ninja state changed significantly
            if self._ninja_state_changed(ninja_pos, ninja_vel):
                event = GraphEvent(
                    event_type=EventType.NINJA_STATE_CHANGED,
                    timestamp=current_time,
                    position=tuple(ninja_pos),
                    state_data={'velocity': tuple(ninja_vel)},
                    priority=0.9  # High priority for ninja state changes
                )
                self._queue_event(event)
        
        # Detect entity changes
        if 'entities' in obs:
            for entity_id, entity_data in enumerate(obs['entities']):
                if self._entity_state_changed(entity_id, entity_data):
                    event = GraphEvent(
                        event_type=EventType.ENTITY_STATE_CHANGED,
                        timestamp=current_time,
                        entity_id=entity_id,
                        position=tuple(entity_data.get('position', (0, 0))),
                        state_data=entity_data,
                        priority=0.7  # Medium priority for entity changes
                    )
                    self._queue_event(event)
    
    def _ninja_state_changed(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        """Check if ninja state changed significantly since last update."""
        # Simplified change detection - in practice, you'd track previous state
        return True  # Always assume change for now
    
    def _entity_state_changed(self, entity_id: int, entity_data: Dict[str, Any]) -> bool:
        """Check if entity state changed significantly since last update."""
        # Simplified change detection - in practice, you'd track previous state
        return False  # Assume no change for now
    
    def _queue_event(self, event: GraphEvent):
        """Queue an event for processing."""
        # Generate unique event ID to avoid duplicates
        event_id = hash((event.event_type, event.timestamp, event.entity_id, event.position))
        
        if event_id not in self.processed_events:
            self.event_queue.append(event)
    
    def _process_event_queue(self):
        """Process queued events within computational budget."""
        events_processed = 0
        events_skipped = 0
        
        # Sort events by priority (highest first)
        sorted_events = sorted(self.event_queue, key=lambda e: e.priority, reverse=True)
        
        for event in sorted_events:
            if not self.update_budget.can_afford_time(5.0):  # Assume 5ms per event
                events_skipped += 1
                continue
            
            if event.priority < self.update_budget.priority_threshold:
                events_skipped += 1
                continue
            
            start_time = time.time()
            
            # Process event based on type
            if event.event_type == EventType.NINJA_STATE_CHANGED:
                self._handle_ninja_state_change(event)
            elif event.event_type == EventType.ENTITY_STATE_CHANGED:
                self._handle_entity_state_change(event)
            elif event.event_type == EventType.DOOR_TOGGLED:
                self._handle_door_toggle(event)
            elif event.event_type == EventType.SWITCH_ACTIVATED:
                self._handle_switch_activation(event)
            
            # Track time consumption
            processing_time_ms = (time.time() - start_time) * 1000
            self.update_budget.consume_time(processing_time_ms)
            
            # Mark event as processed
            event_id = hash((event.event_type, event.timestamp, event.entity_id, event.position))
            self.processed_events.add(event_id)
            events_processed += 1
        
        # Clear processed events from queue
        self.event_queue.clear()
        
        # Update stats
        self.update_stats['events_processed'] += events_processed
        self.update_stats['events_skipped'] += events_skipped
    
    def _handle_ninja_state_change(self, event: GraphEvent):
        """Handle ninja state change events."""
        if self.current_graph is None:
            return
        
        # Update conditional edge masks based on new ninja state
        # This would integrate with the ConditionalEdgeMasker from Task 1.3
        ninja_pos = event.position
        ninja_vel = event.state_data.get('velocity', (0, 0))
        
        # Find edges that might be affected by ninja state change
        affected_edges = self._find_edges_near_position(ninja_pos, radius=100.0)
        
        for edge_idx in affected_edges:
            if not self.update_budget.can_afford_edge_update():
                break
            
            # Update edge availability based on new ninja state
            self._update_edge_for_ninja_state(edge_idx, ninja_pos, ninja_vel)
            self.update_budget.consume_edge_updates(1)
    
    def _handle_entity_state_change(self, event: GraphEvent):
        """Handle entity state change events."""
        if event.entity_id is None:
            return
        
        # Propagate constraint changes through dependent edges
        updated_edges = self.constraint_propagator.propagate_constraint_change(
            event.entity_id, self.current_graph, self.update_budget
        )
        
        logging.debug(f"Entity {event.entity_id} state change affected {len(updated_edges)} edges")
    
    def _handle_door_toggle(self, event: GraphEvent):
        """Handle door toggle events."""
        # Find edges blocked/unblocked by door state change
        door_pos = event.position
        affected_edges = self._find_edges_near_position(door_pos, radius=50.0)
        
        is_door_open = event.state_data.get('is_open', False)
        
        for edge_idx in affected_edges:
            if not self.update_budget.can_afford_edge_update():
                break
            
            # Enable/disable edges based on door state
            self._set_edge_availability(edge_idx, is_door_open)
            self.update_budget.consume_edge_updates(1)
    
    def _handle_switch_activation(self, event: GraphEvent):
        """Handle switch activation events."""
        # Find doors/mechanisms controlled by this switch
        switch_id = event.entity_id
        controlled_entities = self._find_entities_controlled_by_switch(switch_id)
        
        for controlled_entity_id in controlled_entities:
            # Create door toggle event
            door_event = GraphEvent(
                event_type=EventType.DOOR_TOGGLED,
                timestamp=event.timestamp,
                entity_id=controlled_entity_id,
                state_data={'is_open': event.state_data.get('is_activated', False)},
                priority=0.8
            )
            self._queue_event(door_event)
    
    def _update_temporal_edges(self):
        """Update temporal edge availability based on current time."""
        current_time = time.time()
        
        for edge_id, temporal_edge in self.temporal_edges.items():
            was_active = temporal_edge.is_currently_active
            is_now_active = temporal_edge.is_available_at_time(current_time)
            
            if was_active != is_now_active:
                temporal_edge.is_currently_active = is_now_active
                
                if is_now_active:
                    self.active_temporal_edges.add(edge_id)
                else:
                    self.active_temporal_edges.discard(edge_id)
                
                # Update graph edge mask
                if self.current_graph is not None:
                    self._set_edge_availability(edge_id, is_now_active)
    
    def _rebuild_full_graph(self):
        """Rebuild the complete graph from scratch."""
        if not hasattr(self.env, 'get_current_state'):
            return
        
        try:
            # Get current environment state
            state = self.env.get_current_state()
            
            # Build new graph
            self.current_graph = self.graph_builder.build_graph(
                level_data=state.get('level_data', {}),
                ninja_position=state.get('ninja_position', (0, 0)),
                entities=state.get('entities', []),
                ninja_velocity=state.get('ninja_velocity'),
                ninja_state=state.get('ninja_state')
            )
            
            logging.debug(f"Rebuilt full graph: {self.current_graph.num_nodes} nodes, {self.current_graph.num_edges} edges")
            
        except Exception as e:
            logging.warning(f"Failed to rebuild full graph: {e}")
    
    def _find_edges_near_position(self, position: Tuple[float, float], radius: float) -> List[int]:
        """Find edges near a given position."""
        # Simplified implementation - in practice, you'd use spatial indexing
        return []  # Placeholder
    
    def _update_edge_for_ninja_state(self, edge_idx: int, ninja_pos: Tuple[float, float], ninja_vel: Tuple[float, float]):
        """Update edge availability based on ninja state."""
        # Placeholder for ninja state-based edge updates
        pass
    
    def _set_edge_availability(self, edge_idx: int, is_available: bool):
        """Set edge availability in the current graph."""
        if self.current_graph is not None and edge_idx < len(self.current_graph.edge_mask):
            self.current_graph.edge_mask[edge_idx] = 1.0 if is_available else 0.0
    
    def _find_entities_controlled_by_switch(self, switch_id: int) -> List[int]:
        """Find entities controlled by a switch."""
        # Placeholder - would use level data to find switch->door mappings
        return []
    
    def _get_dynamic_graph_metadata(self) -> np.ndarray:
        """Get dynamic graph metadata for observation."""
        metadata = np.zeros(10, dtype=np.float32)
        
        if self.current_graph is not None:
            metadata[0] = self.update_budget.used_time_ms / self.update_budget.max_time_ms  # Time budget usage
            metadata[1] = self.update_budget.used_edge_updates / self.update_budget.max_edge_updates  # Edge budget usage
            metadata[2] = len(self.event_queue) / 100.0  # Event queue fullness
            metadata[3] = len(self.active_temporal_edges) / max(1, len(self.temporal_edges))  # Active temporal edges ratio
            metadata[4] = self.current_graph.num_edges / E_MAX_EDGES  # Graph edge density
            metadata[5] = self.update_stats['avg_update_time_ms'] / 100.0  # Normalized avg update time
        
        return metadata
    
    def _update_performance_stats(self, update_time_ms: float):
        """Update performance statistics."""
        self.update_stats['total_updates'] += 1
        
        # Update rolling average
        alpha = 0.1  # Smoothing factor
        self.update_stats['avg_update_time_ms'] = (
            alpha * update_time_ms + 
            (1 - alpha) * self.update_stats['avg_update_time_ms']
        )
        
        # Track budget exceeded
        if self.update_budget.used_time_ms > self.update_budget.max_time_ms:
            self.update_stats['budget_exceeded_count'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return self.update_stats.copy()
    
    def add_temporal_edge(
        self,
        src_node: int,
        tgt_node: int,
        edge_type: EdgeType,
        availability_windows: List[Tuple[float, float]],
        base_features: np.ndarray
    ) -> int:
        """
        Add a temporal edge with time-dependent availability.
        
        Args:
            src_node: Source node index
            tgt_node: Target node index
            edge_type: Type of edge
            availability_windows: List of (start_time, end_time) windows
            base_features: Base edge features
            
        Returns:
            Edge ID for the temporal edge
        """
        edge_id = len(self.temporal_edges)
        
        temporal_edge = TemporalEdge(
            src_node=src_node,
            tgt_node=tgt_node,
            edge_type=edge_type,
            availability_windows=availability_windows,
            base_features=base_features
        )
        
        self.temporal_edges[edge_id] = temporal_edge
        
        return edge_id
    
    def remove_temporal_edge(self, edge_id: int):
        """Remove a temporal edge."""
        if edge_id in self.temporal_edges:
            del self.temporal_edges[edge_id]
            self.active_temporal_edges.discard(edge_id)