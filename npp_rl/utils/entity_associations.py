"""
Simplified entity association utilities for N++ level analysis.

This module provides a clean, centralized approach to entity relationship analysis
that leverages the actual entity parent-child relationships from nclone instead of
distance-based heuristics.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from nclone.constants.entity_types import EntityType


@dataclass
class EntityPair:
    """Represents a switch-door pair with their 1:1 relationship."""

    switch_entity: Any
    door_entity: Any
    switch_pos: Tuple[float, float]
    door_pos: Tuple[float, float]
    switch_active: bool
    door_accessible: bool
    pair_type: str  # 'exit', 'locked', 'trap'
    entity_id: Optional[str] = None  # Unique identifier for the pair


@dataclass
class LevelCompletionInfo:
    """Contains level completion analysis."""

    exit_pairs: List[EntityPair]
    locked_door_pairs: List[EntityPair]
    trap_door_pairs: List[EntityPair]
    accessible_exits: List[EntityPair]
    required_switches: List[EntityPair]
    completion_strategy: str  # 'direct_exit', 'via_switches', 'blocked'


class EntityAssociationManager:
    """
    Manages entity associations using nclone entity relationships.

    Key Principles:
    - Each switch controls exactly one entity (1:1 relationship)
    - Exit switches have direct parent references to exit doors
    - Locked/trap door entities store both switch and door coordinates
    """

    def __init__(self):
        """Initialize the entity association manager."""
        pass

    def extract_entity_pairs_from_sim(self, sim) -> LevelCompletionInfo:
        """
        Extract entity pairs directly from simulation using actual relationships.

        Args:
            sim: The nclone simulation object

        Returns:
            LevelCompletionInfo with all entity relationships
        """
        exit_pairs = []
        locked_door_pairs = []
        trap_door_pairs = []

        # Extract exit door-switch pairs using direct parent-child relationships
        if hasattr(sim, "entity_dic"):
            # Get exit switches (type 4) - they have parent references to exit doors
            exit_switches = sim.entity_dic.get(EntityType.EXIT_SWITCH, [])

            for i, exit_switch in enumerate(exit_switches):
                if hasattr(exit_switch, "parent") and exit_switch.parent:
                    exit_door = exit_switch.parent

                    exit_pairs.append(
                        EntityPair(
                            switch_entity=exit_switch,
                            door_entity=exit_door,
                            switch_pos=(exit_switch.xpos, exit_switch.ypos),
                            door_pos=(exit_door.xpos, exit_door.ypos),
                            switch_active=exit_switch.active,
                            door_accessible=getattr(exit_door, "switch_hit", False),
                            pair_type="exit",
                            entity_id=f"exit_{i}",
                        )
                    )

            # Extract locked door pairs using coordinate references
            locked_doors = sim.entity_dic.get(EntityType.LOCKED_DOOR, [])

            for i, locked_door in enumerate(locked_doors):
                # Locked door entity is positioned at switch location
                # Door segment position is calculated from door entity properties
                if hasattr(locked_door, "segment") and locked_door.segment:
                    door_pos = (
                        (locked_door.segment.x1 + locked_door.segment.x2) * 0.5,
                        (locked_door.segment.y1 + locked_door.segment.y2) * 0.5,
                    )
                else:
                    door_pos = (locked_door.xpos, locked_door.ypos)

                locked_door_pairs.append(
                    EntityPair(
                        switch_entity=locked_door,  # Entity is at switch position
                        door_entity=locked_door,  # Same entity represents both switch and door
                        switch_pos=(
                            locked_door.xpos,
                            locked_door.ypos,
                        ),  # Switch position
                        door_pos=door_pos,  # Door segment position
                        switch_active=locked_door.active,
                        door_accessible=not getattr(locked_door, "closed", True),
                        pair_type="locked",
                        entity_id=f"locked_{i}",
                    )
                )

            # Extract trap door pairs using coordinate references
            trap_doors = sim.entity_dic.get(EntityType.TRAP_DOOR, [])

            for i, trap_door in enumerate(trap_doors):
                # Trap door entity is positioned at switch location
                # Door segment position is calculated from door entity properties
                if hasattr(trap_door, "segment") and trap_door.segment:
                    door_pos = (
                        (trap_door.segment.x1 + trap_door.segment.x2) * 0.5,
                        (trap_door.segment.y1 + trap_door.segment.y2) * 0.5,
                    )
                else:
                    door_pos = (trap_door.xpos, trap_door.ypos)

                trap_door_pairs.append(
                    EntityPair(
                        switch_entity=trap_door,  # Entity is at switch position
                        door_entity=trap_door,  # Same entity represents both switch and door
                        switch_pos=(trap_door.xpos, trap_door.ypos),  # Switch position
                        door_pos=door_pos,  # Door segment position
                        switch_active=trap_door.active,
                        door_accessible=not getattr(
                            trap_door, "closed", False
                        ),  # Trap doors start open
                        pair_type="trap",
                        entity_id=f"trap_{i}",
                    )
                )

        # Analyze completion strategy
        accessible_exits = [
            pair for pair in exit_pairs if pair.switch_active and pair.door_accessible
        ]
        required_switches = [
            pair
            for pair in exit_pairs
            if not (pair.switch_active and pair.door_accessible)
        ]

        if accessible_exits:
            completion_strategy = "direct_exit"
        elif required_switches:
            completion_strategy = "via_switches"
        else:
            completion_strategy = "blocked"

        return LevelCompletionInfo(
            exit_pairs=exit_pairs,
            locked_door_pairs=locked_door_pairs,
            trap_door_pairs=trap_door_pairs,
            accessible_exits=accessible_exits,
            required_switches=required_switches,
            completion_strategy=completion_strategy,
        )

    def find_best_exit_option(
        self,
        completion_info: LevelCompletionInfo,
        start_pos: Tuple[float, float],
        graph_data: Optional[Any] = None,
    ) -> Optional[EntityPair]:
        """
        Find the best exit option considering graph traversability.

        Args:
            completion_info: Level completion information
            start_pos: Starting position (x, y)
            graph_data: Optional hierarchical graph data for pathfinding

        Returns:
            Best exit pair to target, or None if no viable options
        """
        # If there are directly accessible exits, find the closest one
        if completion_info.accessible_exits:
            best_exit = min(
                completion_info.accessible_exits,
                key=lambda pair: self._calculate_path_cost(
                    start_pos, pair.door_pos, graph_data
                ),
            )
            return best_exit

        # If no direct exits, find the best switch-door sequence
        if completion_info.required_switches:
            best_sequence = min(
                completion_info.required_switches,
                key=lambda pair: (
                    self._calculate_path_cost(start_pos, pair.switch_pos, graph_data)
                    + self._calculate_path_cost(
                        pair.switch_pos, pair.door_pos, graph_data
                    )
                ),
            )
            return best_sequence

        return None

    def _calculate_path_cost(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        graph_data: Optional[Any] = None,
    ) -> float:
        """
        Calculate path cost using graph data if available, otherwise Euclidean distance.

        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            graph_data: Optional hierarchical graph data

        Returns:
            Path cost estimate
        """
        if graph_data is not None and hasattr(graph_data, "sub_cell_graph"):
            # Use hierarchical graph for pathfinding
            try:
                return self._calculate_graph_path_cost(start_pos, end_pos, graph_data)
            except Exception:
                # Fallback to Euclidean if graph pathfinding fails
                pass

        # Fallback to Euclidean distance
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        return (dx * dx + dy * dy) ** 0.5

    def _calculate_graph_path_cost(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        graph_data: Any,
    ) -> float:
        """
        Calculate path cost using hierarchical graph structure.

        Uses the graph's edge connectivity and features to estimate true path cost
        rather than straight-line distance.

        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            graph_data: Hierarchical graph data

        Returns:
            Graph-based path cost estimate
        """
        # Convert positions to sub-cell coordinates (6px resolution)
        start_sub_col = int(start_pos[0] // 6)
        start_sub_row = int(start_pos[1] // 6)
        end_sub_col = int(end_pos[0] // 6)
        end_sub_row = int(end_pos[1] // 6)

        # Get graph dimensions
        if hasattr(graph_data, "resolution_info"):
            sub_cell_dims = graph_data.resolution_info["grid_dimensions"].get(
                0, (176, 100)
            )  # 6px resolution
            sub_cell_width, sub_cell_height = sub_cell_dims
        else:
            sub_cell_width, sub_cell_height = 176, 100  # Default dimensions

        # Convert to node indices
        start_node = start_sub_row * sub_cell_width + start_sub_col
        end_node = end_sub_row * sub_cell_width + end_sub_col

        # Bounds checking
        if (
            start_node >= graph_data.sub_cell_graph.num_nodes
            or end_node >= graph_data.sub_cell_graph.num_nodes
            or start_node < 0
            or end_node < 0
        ):
            # Fall back to Euclidean distance
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            return (dx * dx + dy * dy) ** 0.5

        # Simple A* approximation using graph connectivity
        # For now, use a simplified heuristic based on graph structure

        # Check if direct edge exists
        edge_index = graph_data.sub_cell_graph.edge_index
        edge_features = graph_data.sub_cell_graph.edge_features
        edge_mask = graph_data.sub_cell_graph.edge_mask

        # Look for direct edge from start to end
        for i in range(graph_data.sub_cell_graph.num_edges):
            if edge_mask[i] > 0:
                src = edge_index[0, i]
                tgt = edge_index[1, i]

                if src == start_node and tgt == end_node:
                    # Direct edge exists - use its cost
                    if len(edge_features[i]) > 10:  # Energy cost feature
                        return float(edge_features[i][10])  # Use edge's energy cost

        # No direct edge - estimate using multi-hop pathfinding approximation
        # Use regional graph for faster long-distance estimates
        if hasattr(graph_data, "region_graph"):
            # Convert to region coordinates (96px resolution)
            start_region_col = start_sub_col // 16  # 96px / 6px = 16
            start_region_row = start_sub_row // 16
            end_region_col = end_sub_col // 16
            end_region_row = end_sub_row // 16

            region_distance = abs(end_region_col - start_region_col) + abs(
                end_region_row - start_region_row
            )

            # Estimate cost based on region-level pathfinding
            base_cost = region_distance * 96.0  # Manhattan distance in pixels

            # Apply complexity penalty based on graph connectivity
            complexity_factor = 1.2  # Assume 20% overhead for multi-hop paths

            return base_cost * complexity_factor

        # Ultimate fallback
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        return (dx * dx + dy * dy) ** 0.5

    def get_completion_requirements(
        self, completion_info: LevelCompletionInfo
    ) -> List[Dict[str, Any]]:
        """
        Get structured completion requirements based on entity analysis.

        Args:
            completion_info: Level completion information

        Returns:
            List of completion requirements in priority order
        """
        requirements = []

        if completion_info.completion_strategy == "direct_exit":
            requirements.append(
                {
                    "type": "reach_exit",
                    "priority": 1,
                    "positions": [
                        pair.door_pos for pair in completion_info.accessible_exits
                    ],
                    "description": "Touch any accessible exit door to complete level",
                    "critical": True,
                }
            )

        elif completion_info.completion_strategy == "via_switches":
            requirements.append(
                {
                    "type": "activate_exit_switch",
                    "priority": 2,
                    "positions": [
                        pair.switch_pos for pair in completion_info.required_switches
                    ],
                    "description": "Activate exit switch to unlock exit door",
                    "critical": True,
                }
            )

            requirements.append(
                {
                    "type": "reach_unlocked_exit",
                    "priority": 1,
                    "positions": [
                        pair.door_pos for pair in completion_info.required_switches
                    ],
                    "description": "Touch unlocked exit door to complete level",
                    "critical": True,
                }
            )

        # Add locked door considerations (lower priority)
        if completion_info.locked_door_pairs:
            blocking_locked_doors = [
                pair
                for pair in completion_info.locked_door_pairs
                if not pair.switch_active and not pair.door_accessible
            ]
            if blocking_locked_doors:
                requirements.append(
                    {
                        "type": "consider_locked_doors",
                        "priority": 3,
                        "positions": [
                            pair.switch_pos for pair in blocking_locked_doors
                        ],
                        "description": "May need to activate if blocking path to exit",
                        "critical": False,
                    }
                )

        # Add trap door warnings (lowest priority)
        if completion_info.trap_door_pairs:
            risky_traps = [
                pair
                for pair in completion_info.trap_door_pairs
                if not pair.switch_active and pair.door_accessible
            ]
            if risky_traps:
                requirements.append(
                    {
                        "type": "avoid_trap_switches",
                        "priority": 4,
                        "positions": [pair.switch_pos for pair in risky_traps],
                        "description": "Avoid unless necessary - will permanently close doors",
                        "critical": False,
                        "warning": True,
                    }
                )

        return sorted(requirements, key=lambda r: r["priority"])

    def get_controlled_entity(self, switch_pair: EntityPair) -> Any:
        """
        Get the entity controlled by a switch (always 1:1 relationship).

        Args:
            switch_pair: EntityPair representing the switch-door relationship

        Returns:
            The controlled entity (door entity for the switch)
        """
        return switch_pair.door_entity

    def get_switch_for_entity(
        self, entity_pairs: List[EntityPair], entity_id: str
    ) -> Optional[EntityPair]:
        """
        Find the switch that controls a specific entity.

        Args:
            entity_pairs: List of EntityPair objects
            entity_id: ID of the entity to find the controlling switch for

        Returns:
            EntityPair if found, None otherwise
        """
        for pair in entity_pairs:
            if pair.entity_id == entity_id:
                return pair
        return None


def demonstrate_simplified_usage():
    """
    Demonstrate how to use the simplified entity association system.

    Example usage:
    ```python
    # In your environment or trajectory calculator:
    from npp_rl.utils.entity_associations import EntityAssociationManager

    entity_manager = EntityAssociationManager()

    # Extract relationships directly from simulation
    completion_info = entity_manager.extract_entity_pairs_from_sim(sim)

    # Find best exit strategy
    best_exit = entity_manager.find_best_exit_option(completion_info, ninja_pos, graph_data)

    # Get completion requirements
    requirements = entity_manager.get_completion_requirements(completion_info)
    ```

    Benefits:
    - No proximity-based guessing
    - Uses actual parent-child relationships from nclone entities
    - 1:1 switch-entity relationship (each switch controls exactly one entity)
    - Integrates with hierarchical graph pathfinding
    - Simple, clean interfaces
    - Handles multiple exits correctly
    - Centralized entity association logic
    """
    pass
