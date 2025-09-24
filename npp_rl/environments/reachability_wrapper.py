"""
Simplified Reachability Wrapper for NPP-RL.

This module replaces the complex tiered reachability system with a simple
flood fill approach optimized for real-time RL training.

Key simplifications:
- Uses only Tier 1 flood fill (<1ms performance target)
- No complex physics analysis or trajectory calculation
- Basic 4-connectivity reachability analysis
- Minimal feature extraction (8 dimensions vs 64)

This aligns with the principle that HGT should learn complex movement
patterns emergently rather than having them pre-computed.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Set, Optional, List
import gym
from gym import spaces

# Use nclone physics constants
TILE_PIXEL_SIZE = 24       # Standard N++ tile size
FULL_MAP_WIDTH_PX = 1056   # Standard N++ level width
FULL_MAP_HEIGHT_PX = 600   # Standard N++ level height


class SimpleReachabilityWrapper(gym.Wrapper):
    """
    Simplified reachability wrapper that provides basic connectivity analysis.
    
    Replaces complex 64-dimensional reachability features with 8-dimensional
    simplified features focused on basic connectivity and strategic information.
    """

    def __init__(self, env, debug: bool = False):
        """
        Initialize simplified reachability wrapper.

        Args:
            env: Base environment to wrap
            debug: Enable debug output and performance logging
        """
        super().__init__(env)
        self.debug = debug
        
        # Performance tracking
        self.reachability_times = []
        self.max_time_samples = 100
        
        # Simple cache for performance
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False
        
        # Update observation space to include simplified reachability features
        original_space = env.observation_space
        if isinstance(original_space, spaces.Dict):
            # Add simplified reachability features to existing dict space
            new_spaces = original_space.spaces.copy()
            new_spaces['reachability_features'] = spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            )
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # Fallback: create new dict space
            self.observation_space = spaces.Dict({
                'original': original_space,
                'reachability_features': spaces.Box(
                    low=0.0, high=1.0, shape=(8,), dtype=np.float32
                )
            })

    def reset(self, **kwargs):
        """Reset environment and clear reachability cache."""
        obs = self.env.reset(**kwargs)
        self._clear_cache()
        return self._add_reachability_features(obs)

    def step(self, action):
        """Step environment and update reachability features."""
        obs, reward, done, info = self.env.step(action)
        obs = self._add_reachability_features(obs)
        
        # Add performance info
        if self.reachability_times:
            avg_time = np.mean(self.reachability_times[-10:])  # Last 10 samples
            info['reachability_time_ms'] = avg_time * 1000
        
        return obs, reward, done, info

    def _add_reachability_features(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add simplified reachability features to observation."""
        start_time = time.time()
        
        # Extract ninja position
        ninja_pos = self._extract_ninja_position(obs)
        if ninja_pos is None:
            # Fallback: return zero features
            reachability_features = np.zeros(8, dtype=np.float32)
        else:
            reachability_features = self._compute_simple_reachability(obs, ninja_pos)
        
        # Track performance
        elapsed_time = time.time() - start_time
        self.reachability_times.append(elapsed_time)
        if len(self.reachability_times) > self.max_time_samples:
            self.reachability_times.pop(0)
        
        if self.debug and elapsed_time > 0.001:  # Warn if >1ms
            print(f"Warning: Reachability computation took {elapsed_time*1000:.2f}ms")
        
        # Add to observation
        if isinstance(obs, dict):
            obs['reachability_features'] = reachability_features
        else:
            obs = {
                'original': obs,
                'reachability_features': reachability_features
            }
        
        return obs

    def _extract_ninja_position(self, obs: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract ninja position from observation."""
        # Try different possible keys for ninja position
        position_keys = ['ninja_position', 'player_position', 'position']
        
        for key in position_keys:
            if key in obs:
                pos = obs[key]
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    return (int(pos[0]), int(pos[1]))
        
        # Try extracting from player_x, player_y
        if 'player_x' in obs and 'player_y' in obs:
            return (int(obs['player_x']), int(obs['player_y']))
        
        return None

    def _compute_simple_reachability(
        self, obs: Dict[str, Any], ninja_pos: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute simplified 8-dimensional reachability features.
        
        Features:
        1. Reachable area ratio (0-1)
        2. Distance to nearest switch (normalized)
        3. Distance to exit (normalized)
        4. Reachable switches count (normalized)
        5. Reachable hazards count (normalized)
        6. Connectivity score (0-1)
        7. Exit reachable flag (0-1)
        8. Switch-to-exit path exists (0-1)
        """
        # Check cache validity
        if (self._cache_valid and 
            self._last_ninja_pos == ninja_pos and 
            self._cached_reachability is not None):
            return self._cached_reachability
        
        # Get level data
        level_data = self._extract_level_data(obs)
        if level_data is None:
            return np.zeros(8, dtype=np.float32)
        
        # Perform simple flood fill reachability analysis
        reachable_positions = self._flood_fill_reachability(ninja_pos, level_data)
        
        # Compute 8 simplified features
        features = self._compute_reachability_features(
            ninja_pos, reachable_positions, level_data
        )
        
        # Update cache
        self._last_ninja_pos = ninja_pos
        self._cached_reachability = features
        self._cache_valid = True
        
        return features

    def _extract_level_data(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract level data from observation."""
        # Try to get level data from observation
        if 'level_data' in obs:
            return obs['level_data']
        
        # Try to construct from available information
        level_data = {}
        
        # Get tile information
        if 'tiles' in obs:
            level_data['tiles'] = obs['tiles']
        elif 'level_tiles' in obs:
            level_data['tiles'] = obs['level_tiles']
        
        # Get entity information
        if 'entities' in obs:
            level_data['entities'] = obs['entities']
        elif 'level_entities' in obs:
            level_data['entities'] = obs['level_entities']
        
        return level_data if level_data else None

    def _flood_fill_reachability(
        self, start_pos: Tuple[int, int], level_data: Dict[str, Any]
    ) -> Set[Tuple[int, int]]:
        """
        Perform simple flood fill reachability analysis.
        
        This is the core simplification - uses basic 4-connectivity
        instead of complex physics-based reachability.
        """
        # Get traversable positions from tiles
        traversable = self._get_traversable_positions(level_data)
        
        if not traversable:
            return {start_pos}
        
        # Snap start position to grid
        start_grid = self._snap_to_grid(start_pos)
        if start_grid not in traversable:
            # Find nearest traversable position
            start_grid = self._find_nearest_traversable(start_pos, traversable)
        
        # Simple flood fill with 4-connectivity
        visited = set()
        queue = [start_grid]
        visited.add(start_grid)
        
        directions = [
            (0, TILE_PIXEL_SIZE),    # Down
            (0, -TILE_PIXEL_SIZE),   # Up
            (TILE_PIXEL_SIZE, 0),    # Right
            (-TILE_PIXEL_SIZE, 0),   # Left
        ]
        
        while queue:
            current = queue.pop(0)
            x, y = current
            
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in traversable and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited

    def _get_traversable_positions(self, level_data: Dict[str, Any]) -> Set[Tuple[int, int]]:
        """Get traversable positions from level data."""
        traversable = set()
        
        tiles = level_data.get('tiles')
        if tiles is None:
            return traversable
        
        if isinstance(tiles, np.ndarray):
            height, width = tiles.shape
        else:
            # Assume standard N++ dimensions
            width, height = 42, 23
        
        for row in range(height):
            for col in range(width):
                if isinstance(tiles, np.ndarray):
                    tile_id = tiles[row, col]
                else:
                    tile_id = 0  # Default to traversable
                
                # Tile type 0 is empty space (traversable)
                if tile_id == 0:
                    pixel_x = col * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    pixel_y = row * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    traversable.add((pixel_x, pixel_y))
        
        return traversable

    def _snap_to_grid(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Snap position to tile grid."""
        x, y = position
        grid_x = int(x // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        grid_y = int(y // TILE_PIXEL_SIZE) * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
        return (grid_x, grid_y)

    def _find_nearest_traversable(
        self, pos: Tuple[int, int], traversable: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Find nearest traversable position."""
        if not traversable:
            return pos
        
        min_dist = float('inf')
        nearest = pos
        
        for t_pos in traversable:
            dist = (pos[0] - t_pos[0])**2 + (pos[1] - t_pos[1])**2
            if dist < min_dist:
                min_dist = dist
                nearest = t_pos
        
        return nearest

    def _compute_reachability_features(
        self,
        ninja_pos: Tuple[int, int],
        reachable_positions: Set[Tuple[int, int]],
        level_data: Dict[str, Any],
    ) -> np.ndarray:
        """Compute 8-dimensional reachability feature vector."""
        # Total level area (approximate)
        total_area = (FULL_MAP_WIDTH_PX // TILE_PIXEL_SIZE) * (FULL_MAP_HEIGHT_PX // TILE_PIXEL_SIZE)
        
        # 1. Reachable area ratio
        reachable_ratio = len(reachable_positions) / max(total_area, 1)
        reachable_ratio = np.clip(reachable_ratio, 0.0, 1.0)
        
        # Get entities for remaining features
        entities = level_data.get('entities', [])
        
        # 2. Distance to nearest switch (normalized)
        switch_distance = self._get_distance_to_entity_type(ninja_pos, entities, 'switch')
        switch_distance_norm = np.clip(switch_distance / 1000.0, 0.0, 1.0)  # Normalize by max level width
        
        # 3. Distance to exit (normalized)
        exit_distance = self._get_distance_to_entity_type(ninja_pos, entities, 'exit')
        exit_distance_norm = np.clip(exit_distance / 1000.0, 0.0, 1.0)
        
        # 4. Reachable switches count (normalized)
        reachable_switches = self._count_reachable_entities(reachable_positions, entities, 'switch')
        switches_norm = np.clip(reachable_switches / 5.0, 0.0, 1.0)  # Assume max 5 switches
        
        # 5. Reachable hazards count (normalized)
        reachable_hazards = self._count_reachable_entities(reachable_positions, entities, 'hazard')
        hazards_norm = np.clip(reachable_hazards / 10.0, 0.0, 1.0)  # Assume max 10 hazards
        
        # 6. Connectivity score (simple metric)
        connectivity = min(len(reachable_positions) / 100.0, 1.0)  # Simple connectivity metric
        
        # 7. Exit reachable flag
        exit_reachable = float(self._is_entity_reachable(reachable_positions, entities, 'exit'))
        
        # 8. Switch-to-exit path exists (simplified)
        switch_to_exit_path = float(reachable_switches > 0 and exit_reachable > 0)
        
        return np.array([
            reachable_ratio,
            switch_distance_norm,
            exit_distance_norm,
            switches_norm,
            hazards_norm,
            connectivity,
            exit_reachable,
            switch_to_exit_path,
        ], dtype=np.float32)

    def _get_distance_to_entity_type(
        self, pos: Tuple[int, int], entities: List, entity_type: str
    ) -> float:
        """Get distance to nearest entity of specified type."""
        min_distance = 1000.0  # Default large distance
        
        for entity in entities:
            if self._get_entity_type(entity) == entity_type:
                entity_pos = self._get_entity_position(entity)
                if entity_pos:
                    distance = np.sqrt((pos[0] - entity_pos[0])**2 + (pos[1] - entity_pos[1])**2)
                    min_distance = min(min_distance, distance)
        
        return min_distance

    def _count_reachable_entities(
        self, reachable_positions: Set[Tuple[int, int]], entities: List, entity_type: str
    ) -> int:
        """Count reachable entities of specified type."""
        count = 0
        
        for entity in entities:
            if self._get_entity_type(entity) == entity_type:
                entity_pos = self._get_entity_position(entity)
                if entity_pos:
                    entity_grid = self._snap_to_grid(entity_pos)
                    if entity_grid in reachable_positions:
                        count += 1
        
        return count

    def _is_entity_reachable(
        self, reachable_positions: Set[Tuple[int, int]], entities: List, entity_type: str
    ) -> bool:
        """Check if any entity of specified type is reachable."""
        return self._count_reachable_entities(reachable_positions, entities, entity_type) > 0

    def _get_entity_type(self, entity) -> str:
        """Get simplified entity type string."""
        if isinstance(entity, dict):
            entity_id = entity.get('type', 0)
        else:
            entity_id = getattr(entity, 'type', 0)
        
        # Map entity IDs to simplified types
        if entity_id == 3:
            return 'exit'
        elif entity_id == 4:
            return 'switch'
        elif entity_id in [1, 14, 20, 25, 26]:  # Various hazards
            return 'hazard'
        else:
            return 'other'

    def _get_entity_position(self, entity) -> Optional[Tuple[int, int]]:
        """Get entity position."""
        if isinstance(entity, dict):
            x = entity.get('x', entity.get('position', [0, 0])[0])
            y = entity.get('y', entity.get('position', [0, 0])[1])
        else:
            x = getattr(entity, 'x', getattr(entity, 'position', [0, 0])[0])
            y = getattr(entity, 'y', getattr(entity, 'position', [0, 0])[1])
        
        return (int(x), int(y))

    def _clear_cache(self):
        """Clear reachability cache."""
        self._last_ninja_pos = None
        self._cached_reachability = None
        self._cache_valid = False

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.reachability_times:
            return {}
        
        times_ms = [t * 1000 for t in self.reachability_times]
        return {
            'avg_time_ms': np.mean(times_ms),
            'max_time_ms': np.max(times_ms),
            'min_time_ms': np.min(times_ms),
            'std_time_ms': np.std(times_ms),
        }