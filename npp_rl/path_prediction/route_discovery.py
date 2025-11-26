"""Online route discovery system for finding optimal paths during training.

This module implements adaptive route discovery that learns which predicted paths
work best for different level configurations during training, using multi-armed
bandit approaches to balance exploration and exploitation of route options.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class PathOutcome:
    """Represents the outcome of attempting a specific path."""

    path_hash: str
    success: bool
    completion_time: Optional[float]
    distance_traveled: float
    energy_consumed: float
    mines_triggered: int
    deaths_occurred: int
    final_reward: float
    graph_alignment_score: float = 0.0  # 0.0 = impossible, 1.0 = perfect alignment

    @property
    def quality_score(self) -> float:
        """Calculate overall path quality score (0.0 to 1.0)."""
        if not self.success:
            return 0.0

        # Base success score
        score = 0.6

        # Time bonus (faster is better)
        if self.completion_time:
            time_factor = max(
                0.0, min(1.0, (60.0 - self.completion_time) / 60.0)
            )  # Assume 60s is slow
            score += 0.2 * time_factor

        # Efficiency bonus (less distance/energy is better)
        if self.distance_traveled > 0:
            efficiency_factor = 1.0 / (
                1.0 + self.distance_traveled / 1000.0
            )  # Normalize
            score += 0.1 * efficiency_factor

        # Safety bonus (fewer deaths/mines)
        safety_factor = 1.0 / (1.0 + self.deaths_occurred + self.mines_triggered * 0.5)
        score += 0.1 * safety_factor

        return min(1.0, score)


@dataclass
class BanditArm:
    """Represents a path option in multi-armed bandit framework."""

    path_hash: str
    path_type: str
    successes: int = 0
    attempts: int = 0
    total_reward: float = 0.0
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=10))
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate empirical success rate."""
        return self.successes / max(1, self.attempts)

    @property
    def average_reward(self) -> float:
        """Calculate average reward per attempt."""
        return self.total_reward / max(1, self.attempts)

    @property
    def confidence_interval(self) -> float:
        """Calculate confidence interval width for UCB."""
        if self.attempts == 0:
            return float("inf")

        # Use Wilson score interval for better small-sample behavior
        z = 1.96  # 95% confidence
        p = self.success_rate
        n = self.attempts

        if n == 0:
            return 1.0

        denominator = 1 + (z**2 / n)
        adjustment = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))

        return adjustment / denominator

    def update(self, outcome: PathOutcome) -> None:
        """Update bandit arm statistics with new outcome."""
        self.attempts += 1
        self.total_reward += outcome.quality_score

        if outcome.success:
            self.successes += 1

        self.recent_outcomes.append(outcome.quality_score)
        self.last_used = time.time()


class LevelTopologyClassifier:
    """Classifies levels into topology types for path preference learning."""

    def __init__(self):
        self.topology_features = [
            "avg_density",  # Average tile density
            "corridor_ratio",  # Ratio of corridor-like areas
            "open_area_ratio",  # Ratio of open areas
            "mine_density",  # Mine count per area
            "vertical_extent",  # Height variation
            "connectivity",  # Graph connectivity measure
        ]

    def classify_level(self, level_data: Dict[str, Any]) -> str:
        """Classify level topology into one of several types.

        Args:
            level_data: Level configuration with tile_data and entities

        Returns:
            Topology type string (e.g., 'open', 'corridor', 'maze', 'vertical')
        """
        tile_data = level_data.get("tile_data", np.array([]))
        entities = level_data.get("entities", [])

        if tile_data.size == 0:
            return "unknown"

        # Calculate topology features
        features = self._extract_topology_features(tile_data, entities)

        # Simple rule-based classification
        if features["open_area_ratio"] > 0.6:
            return "open"
        elif features["corridor_ratio"] > 0.7:
            return "corridor"
        elif features["vertical_extent"] > 0.8:
            return "vertical"
        elif features["mine_density"] > 0.3:
            return "mine_heavy"
        else:
            return "mixed"

    def _extract_topology_features(
        self, tile_data: np.ndarray, entities: List[Dict]
    ) -> Dict[str, float]:
        """Extract numerical features for topology classification."""
        height, width = tile_data.shape
        total_tiles = height * width

        # Count solid tiles (non-zero, non-glitched)
        solid_tiles = np.sum((tile_data > 0) & (tile_data < 34))
        density = solid_tiles / total_tiles

        # Estimate corridor vs open area ratios
        # Corridor detection: look for horizontal/vertical passages
        corridor_tiles = 0
        open_tiles = 0

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tile_data[y, x] == 0:  # Empty space
                    # Check neighborhood for corridor patterns
                    neighbors = tile_data[y - 1 : y + 2, x - 1 : x + 2]
                    empty_neighbors = np.sum(neighbors == 0)

                    if empty_neighbors >= 5:  # More open area
                        open_tiles += 1
                    elif empty_neighbors == 3:  # Likely corridor
                        corridor_tiles += 1

        total_empty = np.sum(tile_data == 0)
        corridor_ratio = corridor_tiles / max(1, total_empty)
        open_area_ratio = open_tiles / max(1, total_empty)

        # Mine density
        mine_count = sum(1 for e in entities if e.get("type", 0) in [1, 21])
        mine_density = mine_count / total_tiles

        # Vertical extent (height variation)
        height_variation = 0
        if width > 0:
            for x in range(width):
                column = tile_data[:, x]
                top_solid = np.argmax(column > 0) if np.any(column > 0) else height
                bottom_solid = (
                    height - np.argmax(column[::-1] > 0) if np.any(column > 0) else 0
                )
                height_variation += (bottom_solid - top_solid) / height
            height_variation /= width

        # Simple connectivity measure
        empty_positions = np.where(tile_data == 0)
        connectivity = len(empty_positions[0]) / total_tiles if total_tiles > 0 else 0

        return {
            "avg_density": float(density),
            "corridor_ratio": float(corridor_ratio),
            "open_area_ratio": float(open_area_ratio),
            "mine_density": float(mine_density),
            "vertical_extent": float(height_variation),
            "connectivity": float(connectivity),
        }


class OnlineRouteDiscovery:
    """Discover optimal routes during training on unseen levels.

    Uses multi-armed bandit algorithms to balance exploration of new paths
    with exploitation of proven successful routes.
    """

    def __init__(
        self,
        exploration_factor: float = 2.0,
        min_exploration_prob: float = 0.1,
        topology_memory_size: int = 1000,
        outcome_window_size: int = 50,
    ):
        """Initialize route discovery system.

        Args:
            exploration_factor: UCB exploration parameter (higher = more exploration)
            min_exploration_prob: Minimum probability for random exploration
            topology_memory_size: Maximum outcomes to remember per topology
            outcome_window_size: Window size for recent performance tracking
        """
        self.exploration_factor = exploration_factor
        self.min_exploration_prob = min_exploration_prob
        self.topology_memory_size = topology_memory_size
        self.outcome_window_size = outcome_window_size

        # Multi-armed bandits for each level topology
        self.topology_bandits: Dict[str, Dict[str, BanditArm]] = defaultdict(dict)

        # Level topology classifier
        self.topology_classifier = LevelTopologyClassifier()

        # Graph data for validation (optional)
        self.adjacency_graph: Optional[Dict] = None
        self.spatial_hash: Optional[Any] = None

        # Global statistics
        self.total_paths_tried = 0
        self.successful_paths = 0
        self.topology_counts = defaultdict(int)

        # Recent performance tracking
        self.recent_outcomes = deque(maxlen=outcome_window_size)

        # Path similarity tracking for better generalization
        self.path_similarity_threshold = 0.8

        logger.info("Initialized OnlineRouteDiscovery system")

    def set_graph_data(self, adjacency: Dict, spatial_hash: Any) -> None:
        """Update graph data for validation.

        Args:
            adjacency: Graph adjacency dictionary from GraphBuilder
            spatial_hash: SpatialHash for fast node lookups
        """
        self.adjacency_graph = adjacency
        self.spatial_hash = spatial_hash
        if adjacency:
            logger.debug(f"Route discovery updated with graph: {len(adjacency)} nodes")

    def track_exploration_outcomes(
        self,
        level_id: str,
        level_data: Dict[str, Any],
        paths_tried: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> None:
        """Track outcomes of path exploration for a specific level.

        Args:
            level_id: Unique identifier for the level
            level_data: Level configuration data
            paths_tried: List of attempted path information
            results: List of outcome results for each path
        """
        if len(paths_tried) != len(results):
            logger.warning("Mismatch between paths tried and results")
            return

        # Classify level topology
        topology_type = self.topology_classifier.classify_level(level_data)
        self.topology_counts[topology_type] += 1

        # Process each path outcome
        for path_info, result in zip(paths_tried, results):
            outcome = self._create_path_outcome(path_info, result)

            # Update bandit for this topology
            self._update_topology_bandit(topology_type, outcome)

            # Update global statistics
            self.total_paths_tried += 1
            if outcome.success:
                self.successful_paths += 1

            # Track recent outcomes
            self.recent_outcomes.append(outcome.quality_score)

        logger.debug(
            f"Tracked {len(paths_tried)} path outcomes for {topology_type} level"
        )

    def adaptive_path_sampling(
        self,
        candidate_paths: List[Dict[str, Any]],
        level_data: Dict[str, Any],
        exploration_history: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Select which candidate paths to try using UCB algorithm.

        Args:
            candidate_paths: List of available path candidates
            level_data: Current level configuration
            exploration_history: Recent exploration history for this level

        Returns:
            List of indices indicating which paths to try (sorted by priority)
        """
        if not candidate_paths:
            return []

        # Classify current level topology
        topology_type = self.topology_classifier.classify_level(level_data)

        # Get or create bandits for this topology
        if topology_type not in self.topology_bandits:
            self.topology_bandits[topology_type] = {}

        topology_bandits = self.topology_bandits[topology_type]

        # Calculate UCB scores for each candidate path
        ucb_scores = []

        for i, path in enumerate(candidate_paths):
            path_hash = self._hash_path_candidate(path)

            # Get or create bandit arm for this path
            if path_hash not in topology_bandits:
                topology_bandits[path_hash] = BanditArm(
                    path_hash=path_hash, path_type=path.get("path_type", "unknown")
                )

            arm = topology_bandits[path_hash]

            # Calculate UCB score
            ucb_score = self._calculate_ucb_score(arm, topology_type)
            ucb_scores.append((i, ucb_score, path_hash))

        # Sort by UCB score (descending)
        ucb_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter out graph-infeasible paths if graph is available
        if self.adjacency_graph:
            feasible_scores = [
                (i, score, path_hash)
                for i, score, path_hash in ucb_scores
                if self._compute_path_feasibility_score(candidate_paths[i]) > 0.0
            ]

            if not feasible_scores:
                # All paths are infeasible - log warning and return empty list
                logger.warning(
                    f"All {len(ucb_scores)} candidate paths failed graph feasibility check. "
                    "No paths will be selected for exploration."
                )
                return []

            ucb_scores = feasible_scores

        # Apply exploration-exploitation balance
        selected_indices = self._apply_exploration_strategy(
            ucb_scores, exploration_history
        )

        return selected_indices

    def get_topology_performance(self, topology_type: str) -> Dict[str, float]:
        """Get performance statistics for a specific topology type.

        Args:
            topology_type: The topology type to analyze

        Returns:
            Dictionary with performance metrics
        """
        if topology_type not in self.topology_bandits:
            return {"success_rate": 0.0, "avg_reward": 0.0, "num_paths": 0}

        bandits = self.topology_bandits[topology_type]

        if not bandits:
            return {"success_rate": 0.0, "avg_reward": 0.0, "num_paths": 0}

        total_successes = sum(arm.successes for arm in bandits.values())
        total_attempts = sum(arm.attempts for arm in bandits.values())
        total_reward = sum(arm.total_reward for arm in bandits.values())

        success_rate = total_successes / max(1, total_attempts)
        avg_reward = total_reward / max(1, total_attempts)

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "num_paths": len(bandits),
            "total_attempts": total_attempts,
        }

    def _create_path_outcome(
        self, path_info: Dict[str, Any], result: Dict[str, Any]
    ) -> PathOutcome:
        """Create PathOutcome from path information and results."""
        return PathOutcome(
            path_hash=self._hash_path_candidate(path_info),
            success=result.get("success", False),
            completion_time=result.get("completion_time"),
            distance_traveled=result.get("distance_traveled", 0.0),
            energy_consumed=result.get("energy_consumed", 0.0),
            mines_triggered=result.get("mines_triggered", 0),
            deaths_occurred=result.get("deaths_occurred", 0),
            final_reward=result.get("final_reward", 0.0),
        )

    def _update_topology_bandit(self, topology_type: str, outcome: PathOutcome) -> None:
        """Update the bandit arm for a specific topology and path."""
        if topology_type not in self.topology_bandits:
            self.topology_bandits[topology_type] = {}

        bandits = self.topology_bandits[topology_type]

        # Create arm if it doesn't exist
        if outcome.path_hash not in bandits:
            bandits[outcome.path_hash] = BanditArm(
                path_hash=outcome.path_hash,
                path_type="discovered",  # Paths discovered during exploration
            )

        # Update the arm
        bandits[outcome.path_hash].update(outcome)

        # Limit memory usage by removing old/poor-performing arms
        if len(bandits) > self.topology_memory_size:
            self._prune_bandit_arms(bandits)

    def _calculate_ucb_score(self, arm: BanditArm, topology_type: str) -> float:
        """Calculate Upper Confidence Bound score for a bandit arm."""
        if arm.attempts == 0:
            return float("inf")  # Unexlored arms get highest priority

        # Get total attempts for this topology (for UCB calculation)
        total_attempts = sum(
            other_arm.attempts
            for other_arm in self.topology_bandits[topology_type].values()
        )

        if total_attempts <= 1:
            return arm.average_reward

        # UCB formula: average_reward + c * sqrt(ln(N) / n_i)
        exploration_bonus = self.exploration_factor * np.sqrt(
            np.log(total_attempts) / arm.attempts
        )

        ucb_score = arm.average_reward + exploration_bonus

        # Add recency bonus (prefer recently successful paths)
        time_since_use = time.time() - arm.last_used
        recency_bonus = max(
            0, 0.1 * (1.0 - time_since_use / 3600.0)
        )  # Decay over 1 hour

        return ucb_score + recency_bonus

    def _apply_exploration_strategy(
        self,
        ucb_scores: List[Tuple[int, float, str]],
        exploration_history: Optional[Dict[str, Any]],
    ) -> List[int]:
        """Apply exploration strategy to select final path indices."""
        if not ucb_scores:
            return []

        # Always try top UCB score
        selected_indices = [ucb_scores[0][0]]

        # Add exploration paths based on minimum exploration probability
        for i, (path_idx, score, path_hash) in enumerate(ucb_scores[1:], 1):
            if np.random.random() < self.min_exploration_prob:
                selected_indices.append(path_idx)

            # Limit total paths to try (avoid excessive exploration)
            if len(selected_indices) >= 3:
                break

        return selected_indices

    def _hash_path_candidate(self, path_info: Dict[str, Any]) -> str:
        """Create hash for path candidate."""
        # Use path type and key waypoints for hashing
        path_type = path_info.get("path_type", "unknown")
        waypoints = path_info.get("waypoints", [])

        if len(waypoints) <= 3:
            waypoint_str = "_".join(f"{x}_{y}" for x, y in waypoints)
        else:
            # Use start, middle, end for long paths
            key_points = [waypoints[0], waypoints[len(waypoints) // 2], waypoints[-1]]
            waypoint_str = "_".join(f"{x}_{y}" for x, y in key_points)

        hash_input = f"{path_type}_{waypoint_str}"
        return str(hash(hash_input) % 1000000)

    def _prune_bandit_arms(self, bandits: Dict[str, BanditArm]) -> None:
        """Remove poorly performing or old bandit arms to limit memory usage."""
        # Keep only top-performing arms and recently used ones
        arms_by_performance = sorted(
            bandits.items(),
            key=lambda x: (x[1].average_reward, x[1].attempts),
            reverse=True,
        )

        # Keep top 80% of memory limit
        keep_count = int(self.topology_memory_size * 0.8)
        arms_to_keep = dict(arms_by_performance[:keep_count])

        # Also keep recently used arms (last hour)
        current_time = time.time()
        for path_hash, arm in bandits.items():
            if path_hash not in arms_to_keep:
                if current_time - arm.last_used < 3600:  # 1 hour
                    arms_to_keep[path_hash] = arm

        # Update the bandit dictionary
        bandits.clear()
        bandits.update(arms_to_keep)

        logger.debug(f"Pruned bandit arms, keeping {len(arms_to_keep)} arms")

    def _compute_path_feasibility_score(self, path: Dict[str, Any]) -> float:
        """Compute path feasibility using graph validation.

        Args:
            path: Path information dictionary with waypoints

        Returns:
            Feasibility score: 0.0 if impossible, 1.0 if perfect, (0,1) for partial
        """
        if not self.adjacency_graph:
            return 1.0  # No graph, assume feasible

        waypoints = path.get("waypoints", [])
        if len(waypoints) < 2:
            return 1.0

        from .graph_utils import validate_path_on_graph

        is_valid, graph_dist = validate_path_on_graph(
            waypoints, self.adjacency_graph, self.spatial_hash
        )

        if not is_valid:
            return 0.0

        # Compute alignment score: how close is graph path to Euclidean
        euclidean_dist = sum(
            np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))
            for i in range(len(waypoints) - 1)
        )

        if euclidean_dist == 0:
            return 1.0

        # Lower ratio is better (graph distance close to Euclidean = direct path)
        # Normalize to [0, 1]: perfect alignment = 1.0
        alignment = euclidean_dist / graph_dist if graph_dist > 0 else 0.0
        return min(1.0, alignment)

    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery system statistics."""
        overall_success_rate = self.successful_paths / max(1, self.total_paths_tried)
        recent_performance = (
            np.mean(self.recent_outcomes) if self.recent_outcomes else 0.0
        )

        topology_stats = {}
        for topology_type in self.topology_bandits.keys():
            topology_stats[topology_type] = self.get_topology_performance(topology_type)

        return {
            "total_paths_tried": self.total_paths_tried,
            "successful_paths": self.successful_paths,
            "overall_success_rate": overall_success_rate,
            "recent_performance": recent_performance,
            "topology_counts": dict(self.topology_counts),
            "topology_performance": topology_stats,
        }

    def save_discovery_state(self, filepath: str) -> None:
        """Save discovery state to file."""
        state = {
            "topology_bandits": {
                topology_type: {
                    path_hash: {
                        "path_hash": arm.path_hash,
                        "path_type": arm.path_type,
                        "successes": arm.successes,
                        "attempts": arm.attempts,
                        "total_reward": arm.total_reward,
                        "recent_outcomes": list(arm.recent_outcomes),
                        "last_used": arm.last_used,
                    }
                    for path_hash, arm in bandits.items()
                }
                for topology_type, bandits in self.topology_bandits.items()
            },
            "topology_counts": dict(self.topology_counts),
            "total_paths_tried": self.total_paths_tried,
            "successful_paths": self.successful_paths,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved route discovery state to {filepath}")

    def load_discovery_state(self, filepath: str) -> None:
        """Load discovery state from file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Reconstruct bandit arms
            self.topology_bandits.clear()
            for topology_type, bandits_data in state["topology_bandits"].items():
                self.topology_bandits[topology_type] = {}

                for path_hash, arm_data in bandits_data.items():
                    arm = BanditArm(
                        path_hash=arm_data["path_hash"],
                        path_type=arm_data["path_type"],
                        successes=arm_data["successes"],
                        attempts=arm_data["attempts"],
                        total_reward=arm_data["total_reward"],
                        last_used=arm_data["last_used"],
                    )
                    arm.recent_outcomes = deque(arm_data["recent_outcomes"], maxlen=10)

                    self.topology_bandits[topology_type][path_hash] = arm

            # Restore other state
            self.topology_counts.update(state["topology_counts"])
            self.total_paths_tried = state["total_paths_tried"]
            self.successful_paths = state["successful_paths"]

            logger.info(f"Loaded route discovery state from {filepath}")

        except FileNotFoundError:
            logger.warning(f"Discovery state file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading discovery state: {e}")
