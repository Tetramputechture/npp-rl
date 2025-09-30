"""
Integration layer for nclone reachability and exploration systems.

This module provides a clean interface to nclone's existing reachability analysis,
compact feature extraction, and frontier detection systems, avoiding duplication
and leveraging the optimized OpenCV-based implementations.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Import nclone reachability systems
from nclone.graph.reachability.reachability_system import ReachabilitySystem
from nclone.graph.reachability.feature_extractor import ReachabilityFeatureExtractor
from nclone.graph.reachability.frontier_detector import (
    FrontierDetector,
)
from nclone.graph.reachability.rl_integration import RLIntegrationAPI
from nclone.gym_environment.reward_calculation.exploration_reward_calculator import (
    ExplorationRewardCalculator,
)


class ReachabilityAwareExplorationCalculator:
    """
    Enhanced exploration reward calculator that integrates nclone's reachability analysis
    with the existing multi-scale exploration tracking.

    This extends the existing ExplorationRewardCalculator with reachability awareness,
    avoiding duplication while adding spatial accessibility context.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize reachability-aware exploration calculator.

        Args:
            debug: Enable debug output and performance logging
        """
        self.debug = debug

        # Core nclone systems
        self.base_calculator = ExplorationRewardCalculator()
        self.reachability_system = ReachabilitySystem(debug=debug)
        self.feature_extractor = ReachabilityFeatureExtractor(
            reachability_system=self.reachability_system, debug=debug
        )
        self.frontier_detector = FrontierDetector(debug=debug)
        self.rl_api = RLIntegrationAPI(self.reachability_system, debug=debug)

        # Reachability-aware exploration tracking
        self.reachable_visit_bonus = 1.0  # Full reward for reachable areas
        self.frontier_visit_bonus = 2.0  # Extra reward for frontier areas
        self.unreachable_visit_penalty = 0.1  # Reduced reward for unreachable areas

        # Cache for performance
        self._last_reachability_analysis = None
        self._last_position = None
        self._cache_valid = False
        self._cache_timeout = 100  # Cache valid for 100 calls
        self._cache_counter = 0

    def calculate_reachability_aware_reward(
        self,
        player_x: float,
        player_y: float,
        level_data: Optional[Any] = None,
        switch_states: Optional[Dict[int, bool]] = None,
    ) -> Dict[str, float]:
        """
        Calculate exploration reward with reachability awareness.

        Args:
            player_x: Player X position in pixels
            player_y: Player Y position in pixels
            level_data: Level data for reachability analysis
            switch_states: Current switch states

        Returns:
            Dictionary containing reward breakdown
        """
        # Get base exploration reward from existing system
        base_reward = self.base_calculator.calculate_exploration_reward(
            player_x, player_y
        )

        if level_data is None:
            raise ValueError(
                "Level data is required for reachability-aware exploration but not provided. "
                "Ensure the environment provides level_data in observations."
            )

        # Get reachability analysis
        reachability_info = self._get_reachability_analysis(
            player_x, player_y, level_data, switch_states
        )

        if reachability_info is None:
            raise ValueError(
                "Reachability analysis failed. This indicates a problem with the "
                "reachability system or level data format."
            )

        # Calculate reachability modulation
        modulation = self._calculate_reachability_modulation(
            player_x, player_y, reachability_info
        )

        # Apply modulation to base reward
        total_reward = base_reward * modulation

        return {
            "base_exploration": base_reward,
            "reachability_modulation": modulation,
            "total_reward": total_reward,
            "reachability_available": True,
            "frontiers_detected": len(reachability_info.get("frontiers", [])),
            "reachable_positions": len(
                reachability_info.get("reachable_positions", set())
            ),
        }

    def _get_reachability_analysis(
        self,
        player_x: float,
        player_y: float,
        level_data: Any,
        switch_states: Optional[Dict[int, bool]],
    ) -> Optional[Dict[str, Any]]:
        """Get reachability analysis with caching for performance."""
        current_position = (int(player_x), int(player_y))

        # Check cache validity (with timeout for performance)
        self._cache_counter += 1
        if (
            self._cache_valid
            and self._last_position == current_position
            and self._last_reachability_analysis is not None
            and self._cache_counter < self._cache_timeout
        ):
            return self._last_reachability_analysis

        # Get RL state from nclone integration API
        rl_state = self.rl_api.get_rl_state(
            level_data=level_data,
            ninja_position=(player_x, player_y),
            initial_switch_states=switch_states or {},
        )

        if rl_state is None:
            return None

        # Extract relevant information
        analysis = {
            "reachable_positions": rl_state.reachable_positions,
            "frontiers": rl_state.frontiers,
            "accessibility_map": rl_state.accessibility_map,
            "curiosity_map": rl_state.curiosity_map,
            "switch_states": rl_state.switch_states,
            "analysis_time": rl_state.analysis_time,
            "cache_hit": rl_state.cache_hit,
        }

        # Update cache
        self._last_reachability_analysis = analysis
        self._last_position = current_position
        self._cache_valid = True
        self._cache_counter = 0  # Reset counter

        return analysis

    def _calculate_reachability_modulation(
        self, player_x: float, player_y: float, reachability_info: Dict[str, Any]
    ) -> float:
        """Calculate reachability-based reward modulation."""
        # Convert to grid coordinates
        cell_x = int(player_x / self.base_calculator.CELL_SIZE)
        cell_y = int(player_y / self.base_calculator.CELL_SIZE)
        grid_pos = (cell_x, cell_y)

        # Check if position is reachable
        reachable_positions = reachability_info.get("reachable_positions", set())
        if grid_pos in reachable_positions:
            base_modulation = self.reachable_visit_bonus
        else:
            base_modulation = self.unreachable_visit_penalty

        # Check for frontier bonus
        frontiers = reachability_info.get("frontiers", [])
        frontier_bonus = 0.0

        for frontier in frontiers:
            frontier_pos = (
                frontier.position
                if hasattr(frontier, "position")
                else frontier.get("position")
            )
            if frontier_pos and self._is_near_frontier(grid_pos, frontier_pos):
                # Apply frontier bonus based on exploration value
                exploration_value = (
                    frontier.exploration_value
                    if hasattr(frontier, "exploration_value")
                    else frontier.get("exploration_value", 0.5)
                )
                frontier_bonus = max(
                    frontier_bonus, self.frontier_visit_bonus * exploration_value
                )

        return base_modulation + frontier_bonus

    def _is_near_frontier(
        self,
        position: Tuple[int, int],
        frontier_position: Tuple[int, int],
        threshold: int = 2,
    ) -> bool:
        """Check if position is near a frontier."""
        dx = abs(position[0] - frontier_position[0])
        dy = abs(position[1] - frontier_position[1])
        return dx <= threshold and dy <= threshold

    def extract_compact_features(
        self,
        level_data: Any,
        player_position: Tuple[float, float],
        switch_states: Optional[Dict[int, bool]] = None,
        entities: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Extract 8-dimensional compact reachability features.

        Args:
            level_data: Level data for analysis
            player_position: Current player position (x, y)
            switch_states: Current switch states
            entities: List of game entities

        Returns:
            8-dimensional feature vector
        """
        if entities is None:
            entities = []

        try:
            # Use nclone's ReachabilityFeatureExtractor for 8-dimensional features
            features = self.feature_extractor.extract_features(
                ninja_position=player_position,
                level_data=level_data,
                entities=entities,
                switch_states=switch_states,
            )
            return features
        except Exception as e:
            raise ValueError(
                f"Reachability feature extraction failed: {e}. "
                f"Check that level_data, entities, and switch_states are properly formatted."
            )

    def get_frontier_information(
        self,
        level_data: Any,
        player_position: Tuple[float, float],
        switch_states: Optional[Dict[int, bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get frontier information for curiosity-driven exploration.

        Args:
            level_data: Level data for analysis
            player_position: Current player position (x, y)
            switch_states: Current switch states

        Returns:
            List of frontier information dictionaries
        """
        reachability_info = self._get_reachability_analysis(
            player_position[0], player_position[1], level_data, switch_states
        )

        if reachability_info is None:
            raise ValueError(
                "Reachability analysis failed for frontier information. "
                "Check level_data format and reachability system configuration."
            )

        frontiers = reachability_info.get("frontiers", [])
        frontier_info = []

        for frontier in frontiers:
            info = {
                "position": frontier.position
                if hasattr(frontier, "position")
                else frontier.get("position"),
                "type": frontier.frontier_type.value
                if hasattr(frontier, "frontier_type")
                else frontier.get("type"),
                "exploration_value": frontier.exploration_value
                if hasattr(frontier, "exploration_value")
                else frontier.get("exploration_value", 0.5),
                "accessibility_score": frontier.accessibility_score
                if hasattr(frontier, "accessibility_score")
                else frontier.get("accessibility_score", 0.5),
                "potential_area": frontier.potential_area
                if hasattr(frontier, "potential_area")
                else frontier.get("potential_area", 0),
            }
            frontier_info.append(info)

        return frontier_info

    def reset(self):
        """Reset exploration tracking for new episode."""
        self.base_calculator.reset()
        self._cache_valid = False
        self._last_reachability_analysis = None
        self._last_position = None

    def is_nclone_available(self) -> bool:
        """Check if nclone integration is available."""
        return True


def extract_reachability_info_from_observations(
    observations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract reachability information from environment observations using real nclone systems.

    This uses nclone's ReachabilityFeatureExtractor for proper 8-dimensional
    compact feature extraction and reachability analysis.

    Args:
        observations: Environment observations dictionary

    Returns:
        Reachability information dictionary or None if not available
    """
    # Extract relevant data from observations
    player_x = observations.get("player_x")
    player_y = observations.get("player_y")
    level_data = observations.get("level_data")
    switch_states = observations.get("switch_states", {})
    entities = observations.get("entities", [])

    if player_x is None or player_y is None:
        raise ValueError(
            "Player position (player_x, player_y) is required in observations "
            "but not found. Ensure the environment provides player position."
        )

    # Handle batch dimensions
    if isinstance(player_x, (list, np.ndarray)) and len(player_x) > 0:
        player_x = player_x[0] if hasattr(player_x, "__getitem__") else float(player_x)
    if isinstance(player_y, (list, np.ndarray)) and len(player_y) > 0:
        player_y = player_y[0] if hasattr(player_y, "__getitem__") else float(player_y)

    # Create temporary calculator for feature extraction
    calculator = ReachabilityAwareExplorationCalculator()

    # Extract compact features (8-dimensional)
    compact_features = calculator.extract_compact_features(
        level_data=level_data,
        player_position=(float(player_x), float(player_y)),
        switch_states=switch_states,
        entities=entities,
    )

    # Get frontier information
    frontiers = calculator.get_frontier_information(
        level_data=level_data,
        player_position=(float(player_x), float(player_y)),
        switch_states=switch_states,
    )

    # Convert to expected format for ICM integration
    return {
        "compact_features": compact_features,
        "frontiers": frontiers,
        "player_position": (float(player_x), float(player_y)),
        "switch_states": switch_states,
        "nclone_available": True,
    }
