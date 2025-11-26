"""Generalized pattern extractor for learning routing principles from expert demonstrations.

This module extracts generalizable tile/entity patterns -> waypoint preferences from
expert demonstrations without level-specific memorization, enabling zero-shot
performance on unseen level configurations.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TilePattern:
    """Represents a local tile configuration pattern."""

    pattern_id: str
    tile_config: np.ndarray  # 3x3 or 5x5 tile neighborhood
    center_tile_type: int
    pattern_size: int  # 3 or 5

    def __hash__(self):
        return hash(self.pattern_id)


@dataclass
class EntityContext:
    """Represents entity context around a waypoint."""

    mine_positions: List[Tuple[int, int]]  # Relative positions of mines
    door_positions: List[Tuple[int, int]]  # Relative positions of doors
    switch_positions: List[Tuple[int, int]]  # Relative positions of switches
    exit_positions: List[Tuple[int, int]]  # Relative positions of exits


@dataclass
class WaypointPreference:
    """Represents learned waypoint preferences for a pattern."""

    pattern_id: str
    preferred_directions: Dict[Tuple[int, int], float]  # direction -> confidence
    movement_type_prefs: Dict[str, float]  # 'walk', 'jump', 'wall_slide' -> confidence
    risk_assessment: float  # 0.0 (safe) to 1.0 (risky)
    success_count: int
    total_count: int

    @property
    def confidence(self) -> float:
        """Overall confidence based on observation count."""
        if self.total_count == 0:
            return 0.0
        return min(1.0, self.total_count / 50.0)  # Saturate at 50 observations


class GeneralizedPatternExtractor:
    """Extract generalizable routing patterns from expert demonstrations.

    This extractor learns tile/entity configuration -> waypoint preference mappings
    that generalize to unseen levels without memorizing level-specific information.
    """

    def __init__(
        self,
        neighborhood_sizes: List[int] = [3, 5],
        entity_context_radius: int = 24,  # pixels
        min_pattern_observations: int = 5,
    ):
        """Initialize pattern extractor.

        Args:
            neighborhood_sizes: Sizes of tile neighborhoods to extract (e.g., [3, 5])
            entity_context_radius: Radius in pixels to consider entities around waypoint
            min_pattern_observations: Minimum observations to consider pattern reliable
        """
        self.neighborhood_sizes = neighborhood_sizes
        self.entity_context_radius = entity_context_radius
        self.min_pattern_observations = min_pattern_observations

        # Pattern database: pattern_id -> WaypointPreference
        self.pattern_database: Dict[str, WaypointPreference] = {}

        # Tile type mappings for generalization
        self.tile_type_groups = self._initialize_tile_groups()

        # Graph data for validation (optional)
        self.adjacency_graph: Optional[Dict] = None
        self.spatial_hash: Optional[Any] = None

        # Statistics tracking
        self.patterns_extracted = 0
        self.demonstrations_processed = 0

    def _initialize_tile_groups(self) -> Dict[int, str]:
        """Initialize tile type groupings for pattern generalization.

        Groups similar tile types together to improve pattern generalization.
        Based on tile system from sim_mechanics_doc.md.
        """
        return {
            # Empty space
            0: "empty",
            # Full solid blocks
            1: "solid",
            # Half tiles (2-5)
            2: "half_tile",
            3: "half_tile",
            4: "half_tile",
            5: "half_tile",
            # 45-degree slopes (6-9)
            6: "slope_45",
            7: "slope_45",
            8: "slope_45",
            9: "slope_45",
            # Quarter moon - concave curves (10-13)
            10: "curve_concave",
            11: "curve_concave",
            12: "curve_concave",
            13: "curve_concave",
            # Quarter pipes - convex curves (14-17)
            14: "curve_convex",
            15: "curve_convex",
            16: "curve_convex",
            17: "curve_convex",
            # Mild slopes (18-25)
            18: "slope_mild",
            19: "slope_mild",
            20: "slope_mild",
            21: "slope_mild",
            22: "slope_mild",
            23: "slope_mild",
            24: "slope_mild",
            25: "slope_mild",
            # Steep slopes (26-33)
            26: "slope_steep",
            27: "slope_steep",
            28: "slope_steep",
            29: "slope_steep",
            30: "slope_steep",
            31: "slope_steep",
            32: "slope_steep",
            33: "slope_steep",
            # Glitched tiles (34-37) - treat as empty since no collision
            34: "empty",
            35: "empty",
            36: "empty",
            37: "empty",
        }

    def set_graph_data(self, adjacency: Dict, spatial_hash: Any) -> None:
        """Update graph data for validation.

        Args:
            adjacency: Graph adjacency dictionary from GraphBuilder
            spatial_hash: SpatialHash for fast node lookups
        """
        self.adjacency_graph = adjacency
        self.spatial_hash = spatial_hash
        if adjacency:
            logger.debug(
                f"Pattern extractor updated with graph: {len(adjacency)} nodes"
            )

    def extract_tile_entity_patterns(
        self, trajectory: List[Dict[str, Any]], level_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract generalizable tile/entity patterns from a demonstration trajectory.

        Args:
            trajectory: List of trajectory steps with positions and actions
            level_data: Level configuration including tile_data and entities

        Returns:
            List of extracted patterns with waypoint preferences
        """
        tile_data = level_data.get("tile_data", np.array([]))
        entities = level_data.get("entities", [])

        if tile_data.size == 0:
            logger.warning("No tile data available for pattern extraction")
            return []

        extracted_patterns = []

        # Process each trajectory step to extract local patterns
        for i, step in enumerate(trajectory):
            if i == 0:  # Skip first step (no previous position for direction)
                continue

            prev_step = trajectory[i - 1]

            # Extract position information
            current_pos = self._extract_position(step)
            prev_pos = self._extract_position(prev_step)

            if not current_pos or not prev_pos:
                continue

            # Snap to graph nodes if graph is available
            if self.adjacency_graph:
                from .graph_utils import snap_position_to_graph_node

                snapped_current = snap_position_to_graph_node(
                    current_pos, self.adjacency_graph, self.spatial_hash, threshold=24
                )
                snapped_prev = snap_position_to_graph_node(
                    prev_pos, self.adjacency_graph, self.spatial_hash, threshold=24
                )

                # Use snapped positions if available, otherwise use originals
                if snapped_current:
                    current_pos = snapped_current
                if snapped_prev:
                    prev_pos = snapped_prev

            # Calculate movement direction and type
            direction = self._calculate_direction(prev_pos, current_pos)
            movement_type = self._infer_movement_type(step, prev_step)

            # Extract tile patterns at current position
            tile_patterns = self._extract_tile_patterns(current_pos, tile_data)

            # Extract entity context
            entity_context = self._extract_entity_context(current_pos, entities)

            # Create pattern entries
            for pattern in tile_patterns:
                pattern_data = {
                    "pattern": pattern,
                    "entity_context": entity_context,
                    "direction": direction,
                    "movement_type": movement_type,
                    "position": current_pos,
                }
                extracted_patterns.append(pattern_data)

        self.patterns_extracted += len(extracted_patterns)
        return extracted_patterns

    def build_pattern_database(self, demonstrations: List[Dict[str, Any]]) -> None:
        """Build pattern database from multiple expert demonstrations.

        Args:
            demonstrations: List of demonstration data with trajectories and level_data
        """
        logger.info(
            f"Building pattern database from {len(demonstrations)} demonstrations"
        )

        # Reset database
        self.pattern_database.clear()
        pattern_observations = defaultdict(list)

        # Process all demonstrations
        for demo in demonstrations:
            trajectory = demo.get("trajectory", [])
            level_data = demo.get("level_data", {})

            if not trajectory:
                logger.warning("Empty trajectory in demonstration, skipping")
                continue

            # Extract patterns from this demonstration
            patterns = self.extract_tile_entity_patterns(trajectory, level_data)

            # Group patterns by pattern_id for aggregation
            for pattern_data in patterns:
                pattern_id = pattern_data["pattern"].pattern_id
                pattern_observations[pattern_id].append(pattern_data)

        # Build aggregated waypoint preferences for each pattern
        for pattern_id, observations in pattern_observations.items():
            if len(observations) < self.min_pattern_observations:
                continue  # Skip patterns with insufficient observations

            waypoint_pref = self._aggregate_waypoint_preferences(
                pattern_id, observations
            )
            self.pattern_database[pattern_id] = waypoint_pref

        self.demonstrations_processed = len(demonstrations)
        logger.info(
            f"Built pattern database with {len(self.pattern_database)} reliable patterns"
        )

    def get_waypoint_preferences(
        self, tile_pattern: np.ndarray, entity_context: EntityContext
    ) -> Optional[WaypointPreference]:
        """Get waypoint preferences for a given tile pattern and entity context.

        Args:
            tile_pattern: Local tile configuration (3x3 or 5x5)
            entity_context: Surrounding entity positions

        Returns:
            WaypointPreference if pattern exists in database, None otherwise
        """
        # Create pattern ID for lookup
        pattern_id = self._create_pattern_id(tile_pattern, entity_context)

        return self.pattern_database.get(pattern_id)

    def _extract_position(self, step: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract position from trajectory step."""
        # Try different possible position keys
        for pos_key in ["position", "player_pos", "ninja_pos"]:
            if pos_key in step:
                pos = step[pos_key]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    return (int(pos[0]), int(pos[1]))

        # Try extracting from observation
        obs = step.get("obs", {})
        if "player_x" in obs and "player_y" in obs:
            return (int(obs["player_x"]), int(obs["player_y"]))

        return None

    def _calculate_direction(
        self, prev_pos: Tuple[int, int], current_pos: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate normalized movement direction."""
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]

        # Normalize to unit direction
        if dx != 0:
            dx = 1 if dx > 0 else -1
        if dy != 0:
            dy = 1 if dy > 0 else -1

        return (dx, dy)

    def _infer_movement_type(
        self, step: Dict[str, Any], prev_step: Dict[str, Any]
    ) -> str:
        """Infer movement type from trajectory steps."""
        # Try to extract from game state if available
        obs = step.get("obs", {})

        # Check for movement state if available (from ninja physics)
        if "game_state" in obs and isinstance(obs["game_state"], np.ndarray):
            state = obs["game_state"]
            if len(state) > 7:  # Movement state is at indices 3-7
                movement_states = state[3:8]
                if movement_states[1] > 0.5:  # airborne
                    return "jump"
                elif movement_states[0] > 0.5:  # ground movement
                    return "walk"

        # Fallback: infer from position change
        current_pos = self._extract_position(step)
        prev_pos = self._extract_position(prev_step)

        if current_pos and prev_pos:
            dy = current_pos[1] - prev_pos[1]
            if dy < -2:  # Significant upward movement
                return "jump"
            elif abs(dy) <= 2:  # Mostly horizontal
                return "walk"

        return "unknown"

    def _extract_tile_patterns(
        self, position: Tuple[int, int], tile_data: np.ndarray
    ) -> List[TilePattern]:
        """Extract tile patterns around a position."""
        patterns = []

        # Convert pixel position to tile coordinates (24 pixels per tile)
        tile_x = position[0] // 24
        tile_y = position[1] // 24

        for size in self.neighborhood_sizes:
            pattern = self._extract_neighborhood(tile_x, tile_y, size, tile_data)
            if pattern is not None:
                patterns.append(pattern)

        return patterns

    def _extract_neighborhood(
        self, tile_x: int, tile_y: int, size: int, tile_data: np.ndarray
    ) -> Optional[TilePattern]:
        """Extract a tile neighborhood of given size."""
        half_size = size // 2

        # Check bounds
        if (
            tile_y - half_size < 0
            or tile_y + half_size >= tile_data.shape[0]
            or tile_x - half_size < 0
            or tile_x + half_size >= tile_data.shape[1]
        ):
            return None

        # Extract neighborhood
        neighborhood = tile_data[
            tile_y - half_size : tile_y + half_size + 1,
            tile_x - half_size : tile_x + half_size + 1,
        ]

        # Generalize tile types
        generalized = np.array(
            [
                [self.tile_type_groups.get(tile, "unknown") for tile in row]
                for row in neighborhood
            ]
        )

        # Create pattern ID
        center_tile = tile_data[tile_y, tile_x]
        pattern_id = self._create_neighborhood_id(generalized, size)

        return TilePattern(
            pattern_id=pattern_id,
            tile_config=generalized,
            center_tile_type=center_tile,
            pattern_size=size,
        )

    def _extract_entity_context(
        self, position: Tuple[int, int], entities: List[Dict[str, Any]]
    ) -> EntityContext:
        """Extract entity context around a position."""
        mine_positions = []
        door_positions = []
        switch_positions = []
        exit_positions = []

        px, py = position

        for entity in entities:
            ex, ey = entity.get("x", 0), entity.get("y", 0)
            distance = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)

            if distance > self.entity_context_radius:
                continue

            # Relative position
            rel_pos = (ex - px, ey - py)

            entity_type = entity.get("type", 0)

            # Categorize entities based on type (from sim_mechanics_doc.md)
            if entity_type in [1, 21]:  # Toggle mines
                mine_positions.append(rel_pos)
            elif entity_type in [5, 6, 8]:  # Doors (regular, locked, trap)
                door_positions.append(rel_pos)
            elif entity_type == 4:  # Exit switch
                switch_positions.append(rel_pos)
            elif entity_type == 3:  # Exit door
                exit_positions.append(rel_pos)

        return EntityContext(
            mine_positions=mine_positions,
            door_positions=door_positions,
            switch_positions=switch_positions,
            exit_positions=exit_positions,
        )

    def _create_pattern_id(
        self, tile_pattern: np.ndarray, entity_context: EntityContext
    ) -> str:
        """Create unique pattern ID combining tile and entity information."""
        # Convert tile pattern to string
        tile_str = "_".join("_".join(row) for row in tile_pattern)

        # Simplified entity context (just counts and general positions)
        entity_str = f"mines:{len(entity_context.mine_positions)}"
        entity_str += f"_doors:{len(entity_context.door_positions)}"
        entity_str += f"_switches:{len(entity_context.switch_positions)}"
        entity_str += f"_exits:{len(entity_context.exit_positions)}"

        # Create hash to keep ID manageable
        full_str = f"{tile_str}_{entity_str}"
        return hashlib.md5(full_str.encode()).hexdigest()[:12]

    def _create_neighborhood_id(self, neighborhood: np.ndarray, size: int) -> str:
        """Create ID for tile neighborhood pattern."""
        tile_str = "_".join("_".join(row) for row in neighborhood)
        full_str = f"size{size}_{tile_str}"
        return hashlib.md5(full_str.encode()).hexdigest()[:12]

    def _aggregate_waypoint_preferences(
        self, pattern_id: str, observations: List[Dict[str, Any]]
    ) -> WaypointPreference:
        """Aggregate multiple observations into waypoint preferences."""
        direction_counts = Counter()
        movement_type_counts = Counter()
        total_risk = 0.0

        for obs in observations:
            direction = obs["direction"]
            movement_type = obs["movement_type"]

            direction_counts[direction] += 1
            movement_type_counts[movement_type] += 1

            # Simple risk assessment based on entity context
            entity_context = obs["entity_context"]
            risk = len(entity_context.mine_positions) * 0.1  # More mines = more risk
            total_risk += min(1.0, risk)

        # Convert counts to preferences (probabilities)
        total_directions = sum(direction_counts.values())
        preferred_directions = {
            direction: count / total_directions
            for direction, count in direction_counts.items()
        }

        total_movements = sum(movement_type_counts.values())
        movement_type_prefs = {
            movement_type: count / total_movements
            for movement_type, count in movement_type_counts.items()
        }

        avg_risk = total_risk / len(observations)

        return WaypointPreference(
            pattern_id=pattern_id,
            preferred_directions=preferred_directions,
            movement_type_prefs=movement_type_prefs,
            risk_assessment=avg_risk,
            success_count=len(observations),  # All observations assumed successful
            total_count=len(observations),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "patterns_in_database": len(self.pattern_database),
            "demonstrations_processed": self.demonstrations_processed,
            "patterns_extracted": self.patterns_extracted,
            "reliable_patterns": sum(
                1
                for p in self.pattern_database.values()
                if p.total_count >= self.min_pattern_observations
            ),
        }

    def save_pattern_database(self, filepath: str) -> None:
        """Save pattern database to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.pattern_database, f)

        logger.info(
            f"Saved pattern database with {len(self.pattern_database)} patterns to {filepath}"
        )

    def load_pattern_database(self, filepath: str) -> None:
        """Load pattern database from file."""
        import pickle

        with open(filepath, "rb") as f:
            self.pattern_database = pickle.load(f)

        logger.info(
            f"Loaded pattern database with {len(self.pattern_database)} patterns from {filepath}"
        )
