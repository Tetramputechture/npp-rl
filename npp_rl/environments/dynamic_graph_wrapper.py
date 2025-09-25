"""
Simplified Dynamic Graph Wrapper for NPP-RL Production System.

This module provides a clean, production-ready graph wrapper that integrates
with nclone's graph system without over-engineering. It focuses on basic
connectivity updates and lets the HGT transformer learn movement patterns
emergently from multimodal context.

Key Design Principles:
- Use nclone for physics simulation and basic graph construction
- Provide simple connectivity updates when game state changes
- Let HGT learn complex patterns through attention mechanisms
- Maintain sub-millisecond performance for real-time RL training

This replaces the over-engineered 717-line dynamic_graph_wrapper.py with
a clean, focused implementation aligned with HGT design principles.
"""

import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

# nclone integration - proper abstraction level
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.common import GraphData
from nclone.graph.level_data import LevelData


@dataclass
class GraphUpdateInfo:
    """Simple container for graph update information."""

    nodes_updated: int = 0
    edges_updated: int = 0
    update_time_ms: float = 0.0
    switch_states: Dict[int, bool] = None

    def __post_init__(self):
        if self.switch_states is None:
            self.switch_states = {}


class DynamicGraphWrapper(gym.Wrapper):
    """
    Simplified dynamic graph wrapper for production NPP-RL system.

    This wrapper provides basic graph connectivity updates when game state
    changes, without complex event systems or computational budgeting.
    It integrates cleanly with nclone's graph system and lets HGT learn
    movement patterns emergently.

    Key Features:
    - Simple switch/door state tracking
    - Basic graph connectivity updates
    - Clean nclone integration
    - Sub-millisecond performance
    - HGT-friendly design
    """

    def __init__(
        self, env: gym.Env, enable_graph_updates: bool = True, debug: bool = False
    ):
        """
        Initialize simplified dynamic graph wrapper.

        Args:
            env: Base environment to wrap
            enable_graph_updates: Whether to enable graph updates
            debug: Enable debug logging
        """
        super().__init__(env)

        self.enable_graph_updates = enable_graph_updates
        self.debug = debug

        # Core nclone integration - proper abstraction
        self.graph_builder = HierarchicalGraphBuilder(debug=debug)

        # Simple state tracking
        self.current_graph: Optional[GraphData] = None
        self.current_hierarchical_graph = None
        self.last_switch_states: Dict[int, bool] = {}
        self.last_update_time = 0.0

        # Performance tracking (simple)
        self.update_stats = {
            "total_updates": 0,
            "avg_update_time_ms": 0.0,
            "last_update_info": GraphUpdateInfo(),
        }

        # Extend observation space for graph metadata
        self._extend_observation_space()

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

    def _extend_observation_space(self):
        """Add graph observations to observation space."""
        if hasattr(self.env, "observation_space") and hasattr(
            self.env.observation_space, "spaces"
        ):
            from nclone.graph.common import N_MAX_NODES, E_MAX_EDGES
            
            # Graph node features: [x, y, node_type] for each node
            self.env.observation_space.spaces["graph_node_feats"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(N_MAX_NODES, 3), dtype=np.float32
            )
            
            # Graph edge index: [2, max_edges] connectivity matrix
            self.env.observation_space.spaces["graph_edge_index"] = gym.spaces.Box(
                low=0, high=N_MAX_NODES-1, shape=(2, E_MAX_EDGES), dtype=np.int32
            )
            
            # Graph edge features: [weight] for each edge
            self.env.observation_space.spaces["graph_edge_feats"] = gym.spaces.Box(
                low=0.0, high=np.inf, shape=(E_MAX_EDGES, 1), dtype=np.float32
            )
            
            # Graph masks for variable-size graphs
            self.env.observation_space.spaces["graph_node_mask"] = gym.spaces.Box(
                low=0, high=1, shape=(N_MAX_NODES,), dtype=np.int32
            )
            
            self.env.observation_space.spaces["graph_edge_mask"] = gym.spaces.Box(
                low=0, high=1, shape=(E_MAX_EDGES,), dtype=np.int32
            )
            
            # Graph node and edge types
            self.env.observation_space.spaces["graph_node_types"] = gym.spaces.Box(
                low=0, high=10, shape=(N_MAX_NODES,), dtype=np.int32
            )
            
            self.env.observation_space.spaces["graph_edge_types"] = gym.spaces.Box(
                low=0, high=10, shape=(E_MAX_EDGES,), dtype=np.int32
            )
            
            # Simple graph metadata: [update_time, nodes, edges, switches_active]
            self.env.observation_space.spaces["graph_metadata"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            )

    def reset(self, **kwargs):
        """Reset environment and initialize graph state."""
        obs, info = self.env.reset(**kwargs)

        # Reset simple state
        self.current_graph = None
        self.current_hierarchical_graph = None
        self.last_switch_states.clear()
        self.last_update_time = time.time()

        # Build initial graph using nclone
        if self.enable_graph_updates:
            self._update_graph_from_env_state()

        # Add graph data to observation
        if isinstance(obs, dict):
            obs.update(self._get_graph_observations())

        return obs, info

    def step(self, action):
        """Step environment and update graph if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if graph update is needed (simple state-based)
        if self.enable_graph_updates and self._should_update_graph():
            start_time = time.time()
            self._update_graph_from_env_state()
            update_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update simple statistics
            self._update_performance_stats(update_time)

            if self.debug:
                self.logger.debug(f"Graph updated in {update_time:.2f}ms")

        # Add graph data to observation
        if isinstance(obs, dict):
            obs.update(self._get_graph_observations())

        return obs, reward, terminated, truncated, info

    def _should_update_graph(self) -> bool:
        """
        Simple check if graph update is needed.

        Only updates when switch states change - no complex event system.
        """
        # Get current switch states from environment
        current_switch_states = self._get_switch_states_from_env()

        # Check if any switch state changed
        if current_switch_states != self.last_switch_states:
            return True

        return False

    def _get_switch_states_from_env(self) -> Dict[int, bool]:
        """Extract switch states from environment for graph update detection."""
        switch_states = {}

        try:
            # Try direct switch state methods first
            if hasattr(self.env, "get_switch_states"):
                switch_states = self.env.get_switch_states()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_switch_states"):
                switch_states = self.env.unwrapped.get_switch_states()
            else:
                # Extract switch states from nclone environment
                switch_states = self._extract_switch_states_from_nclone()
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not extract switch states: {e}")

        return switch_states

    def _extract_switch_states_from_nclone(self) -> Dict[int, bool]:
        """Extract switch states from nclone environment entities."""
        switch_states = {}
        
        try:
            # Get nplay_headless instance
            nplay = None
            if hasattr(self.env, "nplay_headless"):
                nplay = self.env.nplay_headless
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "nplay_headless"):
                nplay = self.env.unwrapped.nplay_headless
            
            if nplay and hasattr(nplay, "sim") and hasattr(nplay.sim, "entity_dic"):
                entity_dic = nplay.sim.entity_dic
                
                # Extract switch states from different entity types
                # Exit switches (entity_dic key 3)
                if 3 in entity_dic:
                    exit_entities = entity_dic[3]
                    for i, entity in enumerate(exit_entities):
                        if hasattr(entity, "activated") and type(entity).__name__ == "EntityExitSwitch":
                            switch_states[f"exit_switch_{i}"] = bool(entity.activated)
                
                # Door switches (entity_dic key 4 - doors)
                if 4 in entity_dic:
                    door_entities = entity_dic[4]
                    for i, entity in enumerate(door_entities):
                        # Check for door state (open/closed)
                        if hasattr(entity, "open"):
                            switch_states[f"door_{i}"] = bool(entity.open)
                        elif hasattr(entity, "activated"):
                            switch_states[f"door_{i}"] = bool(entity.activated)
                
                # One-way platforms (entity_dic key 5)
                if 5 in entity_dic:
                    platform_entities = entity_dic[5]
                    for i, entity in enumerate(platform_entities):
                        if hasattr(entity, "activated"):
                            switch_states[f"platform_{i}"] = bool(entity.activated)
                
                # Other interactive entities
                for entity_type_id, entities in entity_dic.items():
                    if entity_type_id not in [3, 4, 5]:  # Skip already processed types
                        for i, entity in enumerate(entities):
                            if hasattr(entity, "activated"):
                                switch_states[f"entity_{entity_type_id}_{i}"] = bool(entity.activated)
                            elif hasattr(entity, "open"):
                                switch_states[f"entity_{entity_type_id}_{i}"] = bool(entity.open)
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not extract switch states from nclone: {e}")
        
        return switch_states

    def _update_graph_from_env_state(self):
        """
        Update graph using nclone's graph builder (simple approach).

        This uses nclone's hierarchical graph builder to create updated
        connectivity based on current game state. No complex event processing.
        """
        try:
            # Get level data from environment
            level_data = self._get_level_data_from_env()
            if level_data is None:
                return

            # Use nclone's graph builder - proper abstraction
            start_time = time.time()
            self.current_hierarchical_graph = self.graph_builder.build_graph(level_data)
            build_time = (time.time() - start_time) * 1000

            # Extract the fine-resolution graph as the primary graph for compatibility
            if self.current_hierarchical_graph:
                self.current_graph = self.current_hierarchical_graph.fine_graph
            else:
                self.current_graph = None

            # Update switch state tracking
            self.last_switch_states = self._get_switch_states_from_env()

            # Update simple statistics
            total_nodes = self.current_hierarchical_graph.total_nodes if self.current_hierarchical_graph else 0
            total_edges = self.current_hierarchical_graph.total_edges if self.current_hierarchical_graph else 0
            
            update_info = GraphUpdateInfo(
                nodes_updated=total_nodes,
                edges_updated=total_edges,
                update_time_ms=build_time,
                switch_states=self.last_switch_states.copy(),
            )
            self.update_stats["last_update_info"] = update_info

            if self.debug:
                self.logger.debug(
                    f"Hierarchical graph rebuilt: {total_nodes} total nodes, "
                    f"{total_edges} total edges in {build_time:.2f}ms"
                )
                if self.current_hierarchical_graph:
                    self.logger.debug(
                        f"  Fine: {self.current_hierarchical_graph.fine_graph.num_nodes} nodes, "
                        f"{self.current_hierarchical_graph.fine_graph.num_edges} edges"
                    )
                    self.logger.debug(
                        f"  Medium: {self.current_hierarchical_graph.medium_graph.num_nodes} nodes, "
                        f"{self.current_hierarchical_graph.medium_graph.num_edges} edges"
                    )
                    self.logger.debug(
                        f"  Coarse: {self.current_hierarchical_graph.coarse_graph.num_nodes} nodes, "
                        f"{self.current_hierarchical_graph.coarse_graph.num_edges} edges"
                    )

        except Exception as e:
            if self.debug:
                self.logger.error(f"Graph update failed: {e}")

    def _get_level_data_from_env(self) -> Optional[LevelData]:
        """Extract level data from environment for graph building."""
        try:
            # Try nclone environment interface first (preferred)
            if hasattr(self.env, "level_data"):
                level_data = self.env.level_data
                # Add player state if not already included
                if level_data.player is None:
                    level_data = self._add_player_state_to_level_data(level_data)
                return level_data
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "level_data"):
                level_data = self.env.unwrapped.level_data
                # Add player state if not already included
                if level_data.player is None:
                    level_data = self._add_player_state_to_level_data(level_data)
                return level_data
            
            # Try legacy get_level_data method
            if hasattr(self.env, "get_level_data"):
                return self.env.get_level_data()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_level_data"):
                return self.env.unwrapped.get_level_data()

            # Fallback: construct level data from available information
            return self._construct_level_data_fallback()
            
        except Exception as e:
            if self.debug:
                self.logger.error(f"Failed to extract level data: {e}")
            return None

    def _add_player_state_to_level_data(self, level_data: LevelData) -> LevelData:
        """Add player state to level data if missing."""
        from nclone.graph.level_data import PlayerState
        
        # Try to get player position from environment
        player_pos = self._get_player_position_from_env()
        if player_pos is None:
            return level_data
            
        # Try to get player velocity and state
        player_velocity = self._get_player_velocity_from_env()
        player_state_info = self._get_player_state_from_env()
        
        # Create player state
        player_state = PlayerState(
            position=player_pos,
            velocity=player_velocity or (0.0, 0.0),
            on_ground=player_state_info.get('on_ground', True),
            facing_right=player_state_info.get('facing_right', True),
            health=player_state_info.get('health', 1),
            frame=player_state_info.get('frame', 0)
        )
        
        # Create new level data with player state
        return LevelData(
            tiles=level_data.tiles,
            entities=level_data.entities,
            player=player_state,
            level_id=level_data.level_id,
            width=level_data.width,
            height=level_data.height
        )

    def _get_player_position_from_env(self) -> Optional[tuple]:
        """Extract player position from environment."""
        try:
            # Try nclone environment interface
            if hasattr(self.env, "nplay_headless") and hasattr(self.env.nplay_headless, "ninja_position"):
                return self.env.nplay_headless.ninja_position()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "nplay_headless"):
                if hasattr(self.env.unwrapped.nplay_headless, "ninja_position"):
                    return self.env.unwrapped.nplay_headless.ninja_position()
            
            # Try generic player position methods
            if hasattr(self.env, "get_player_position"):
                return self.env.get_player_position()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_player_position"):
                return self.env.unwrapped.get_player_position()
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not extract player position: {e}")
        
        return None

    def _get_player_velocity_from_env(self) -> Optional[tuple]:
        """Extract player velocity from environment."""
        try:
            # Try nclone environment interface
            if hasattr(self.env, "nplay_headless") and hasattr(self.env.nplay_headless, "ninja_velocity"):
                return self.env.nplay_headless.ninja_velocity()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "nplay_headless"):
                if hasattr(self.env.unwrapped.nplay_headless, "ninja_velocity"):
                    return self.env.unwrapped.nplay_headless.ninja_velocity()
            
            # Try generic velocity methods
            if hasattr(self.env, "get_player_velocity"):
                return self.env.get_player_velocity()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_player_velocity"):
                return self.env.unwrapped.get_player_velocity()
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not extract player velocity: {e}")
        
        return None

    def _get_player_state_from_env(self) -> Dict[str, Any]:
        """Extract additional player state information from environment."""
        state_info = {}
        
        try:
            # Try to extract various player state information
            if hasattr(self.env, "get_player_state"):
                state_info = self.env.get_player_state()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_player_state"):
                state_info = self.env.unwrapped.get_player_state()
            else:
                # Try individual state components
                if hasattr(self.env, "nplay_headless"):
                    nplay = self.env.nplay_headless
                elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "nplay_headless"):
                    nplay = self.env.unwrapped.nplay_headless
                else:
                    nplay = None
                
                if nplay:
                    # Extract what we can from nplay_headless
                    if hasattr(nplay, "ninja_on_ground"):
                        state_info['on_ground'] = nplay.ninja_on_ground()
                    if hasattr(nplay, "ninja_facing_right"):
                        state_info['facing_right'] = nplay.ninja_facing_right()
                    if hasattr(nplay, "ninja_health"):
                        state_info['health'] = nplay.ninja_health()
                    if hasattr(nplay, "frame") or hasattr(nplay, "get_frame"):
                        state_info['frame'] = getattr(nplay, 'frame', 0) or (nplay.get_frame() if hasattr(nplay, 'get_frame') else 0)
                        
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not extract player state: {e}")
        
        return state_info

    def _construct_level_data_fallback(self) -> Optional[LevelData]:
        """Construct level data from basic environment information as fallback."""
        try:
            # This is a basic fallback - in practice you'd need to adapt this
            # to your specific environment's interface
            
            # Try to get basic tile information
            tiles = None
            if hasattr(self.env, "get_tiles"):
                tiles = self.env.get_tiles()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_tiles"):
                tiles = self.env.unwrapped.get_tiles()
            
            # Try to get entities
            entities = []
            if hasattr(self.env, "get_entities"):
                entities = self.env.get_entities()
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_entities"):
                entities = self.env.unwrapped.get_entities()
            
            # Get player state
            player_pos = self._get_player_position_from_env()
            player_velocity = self._get_player_velocity_from_env()
            player_state_info = self._get_player_state_from_env()
            
            player_state = None
            if player_pos:
                from nclone.graph.level_data import PlayerState
                player_state = PlayerState(
                    position=player_pos,
                    velocity=player_velocity or (0.0, 0.0),
                    on_ground=player_state_info.get('on_ground', True),
                    facing_right=player_state_info.get('facing_right', True),
                    health=player_state_info.get('health', 1),
                    frame=player_state_info.get('frame', 0)
                )
            
            if tiles is not None or entities or player_state:
                return LevelData(
                    tiles=tiles,
                    entities=entities,
                    player=player_state,
                    level_id="fallback_level"
                )
                
        except Exception as e:
            if self.debug:
                self.logger.error(f"Fallback level data construction failed: {e}")
        
        return None

    def _get_graph_observations(self) -> Dict[str, np.ndarray]:
        """Get complete graph observations for HGT processing."""
        from nclone.graph.common import N_MAX_NODES, E_MAX_EDGES
        
        # Initialize empty graph observations
        graph_obs = {
            "graph_node_feats": np.zeros((N_MAX_NODES, 3), dtype=np.float32),
            "graph_edge_index": np.zeros((2, E_MAX_EDGES), dtype=np.int32),
            "graph_edge_feats": np.zeros((E_MAX_EDGES, 1), dtype=np.float32),
            "graph_node_mask": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_mask": np.zeros(E_MAX_EDGES, dtype=np.int32),
            "graph_node_types": np.zeros(N_MAX_NODES, dtype=np.int32),
            "graph_edge_types": np.zeros(E_MAX_EDGES, dtype=np.int32),
            "graph_metadata": self._get_graph_metadata(),
        }
        
        # Fill with actual graph data if available
        if self.current_graph is not None:
            # Use the fine-resolution graph for primary observations
            graph_data = self.current_graph
            
            # Copy node features (up to max nodes)
            num_nodes = min(graph_data.num_nodes, N_MAX_NODES)
            if hasattr(graph_data, 'node_features') and graph_data.node_features is not None:
                graph_obs["graph_node_feats"][:num_nodes] = graph_data.node_features[:num_nodes]
            
            # Copy edge index (up to max edges)
            num_edges = min(graph_data.num_edges, E_MAX_EDGES)
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                graph_obs["graph_edge_index"][:, :num_edges] = graph_data.edge_index[:, :num_edges]
            
            # Copy edge features (up to max edges)
            if hasattr(graph_data, 'edge_features') and graph_data.edge_features is not None:
                graph_obs["graph_edge_feats"][:num_edges] = graph_data.edge_features[:num_edges]
            
            # Set masks
            graph_obs["graph_node_mask"][:num_nodes] = 1
            graph_obs["graph_edge_mask"][:num_edges] = 1
            
            # Copy node and edge types
            if hasattr(graph_data, 'node_types') and graph_data.node_types is not None:
                graph_obs["graph_node_types"][:num_nodes] = graph_data.node_types[:num_nodes]
            
            if hasattr(graph_data, 'edge_types') and graph_data.edge_types is not None:
                graph_obs["graph_edge_types"][:num_edges] = graph_data.edge_types[:num_edges]
        
        return graph_obs

    def get_hierarchical_graph_data(self):
        """Get the full hierarchical graph data for advanced processing."""
        return self.current_hierarchical_graph

    def _get_graph_metadata(self) -> np.ndarray:
        """Get simple graph metadata for observation."""
        metadata = np.zeros(4, dtype=np.float32)

        if self.current_graph is not None:
            # Simple metadata: [update_time_norm, nodes_norm, edges_norm, switches_active_norm]
            metadata[0] = min(
                self.update_stats["avg_update_time_ms"] / 10.0, 1.0
            )  # Normalize to ~10ms max
            metadata[1] = min(
                self.current_graph.num_nodes / 1000.0, 1.0
            )  # Normalize to ~1000 nodes max
            metadata[2] = min(
                self.current_graph.num_edges / 5000.0, 1.0
            )  # Normalize to ~5000 edges max
            metadata[3] = len([s for s in self.last_switch_states.values() if s]) / max(
                1, len(self.last_switch_states)
            )

        return metadata

    def _update_performance_stats(self, update_time_ms: float):
        """Update simple performance statistics."""
        self.update_stats["total_updates"] += 1

        # Simple rolling average
        alpha = 0.1
        self.update_stats["avg_update_time_ms"] = (
            alpha * update_time_ms
            + (1 - alpha) * self.update_stats["avg_update_time_ms"]
        )

    def get_current_graph(self) -> Optional[GraphData]:
        """Get current graph data for external use."""
        return self.current_graph

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get simple performance statistics."""
        return self.update_stats.copy()

    def force_graph_update(self):
        """Force a graph update (for testing/debugging)."""
        if self.enable_graph_updates:
            self._update_graph_from_env_state()
