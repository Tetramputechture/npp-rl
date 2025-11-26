"""Generalized navigation system for N++ level completion.

This module integrates all pattern-based learning components to enable generalized
navigation and completion of N++ levels through learned tile/entity -> waypoint
principles, multi-path prediction, and online route discovery.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..path_prediction import (
    GeneralizedPatternExtractor,
    OnlineRouteDiscovery,
    ProbabilisticPathPredictor,
    create_multipath_predictor,
)
from ..models.pattern_aware_extractor import (
    PatternAwareExtractor,
    create_pattern_aware_extractor,
)
from nclone.gym_environment.reward_calculation.adaptive_pbrs import (
    AdaptiveMultiPathPBRS,
    create_adaptive_multipath_pbrs,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneralizedNavigationConfig:
    """Configuration for generalized N++ navigation system."""

    # Pattern extraction settings
    pattern_neighborhood_sizes: List[int] = None
    pattern_entity_radius: int = 24
    min_pattern_observations: int = 5

    # Multi-path prediction settings
    num_path_candidates: int = 4
    max_waypoints: int = 20
    graph_feature_dim: int = 256
    tile_pattern_dim: int = 64
    entity_feature_dim: int = 32

    # Route discovery settings
    exploration_factor: float = 2.0
    min_exploration_prob: float = 0.1
    topology_memory_size: int = 1000

    # Adaptive PBRS settings
    pbrs_learning_rate: float = 0.05
    uncertainty_bonus_weight: float = 0.1
    min_confidence_threshold: float = 0.1
    path_memory_size: int = 500

    # Training settings
    pattern_database_path: str = "pattern_database.pkl"
    route_discovery_state_path: str = "route_discovery_state.json"
    adaptive_pbrs_state_path: str = "adaptive_pbrs_state.pkl"

    # Evaluation settings
    evaluation_levels_per_topology: int = 10
    success_threshold: float = 0.7

    def __post_init__(self):
        if self.pattern_neighborhood_sizes is None:
            self.pattern_neighborhood_sizes = [3, 5]


class GeneralizedNavigationSystem:
    """Integrated system for generalized N++ navigation and level completion.

    This system combines:
    1. GeneralizedPatternExtractor - learns tile/entity -> waypoint patterns
    2. ProbabilisticPathPredictor - generates multiple candidate paths
    3. OnlineRouteDiscovery - discovers optimal routes during training
    4. AdaptiveMultiPathPBRS - adaptive reward shaping
    5. PatternAwareExtractor - enhanced feature extraction
    """

    def __init__(
        self,
        config: GeneralizedNavigationConfig,
        demonstration_data: Optional[List[Dict[str, Any]]] = None,
        save_dir: str = "navigation_models",
        env=None,
    ):
        """Initialize zero-shot training system.

        Args:
            config: Configuration for the system
            demonstration_data: Expert demonstrations for pattern learning
            save_dir: Directory to save trained models and states
            env: Environment instance for graph access (optional)
        """
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.env = env

        # Initialize core components
        self.pattern_extractor = self._initialize_pattern_extractor(demonstration_data)
        self.path_predictor = self._initialize_path_predictor()
        self.route_discovery = self._initialize_route_discovery()
        self.adaptive_pbrs = self._initialize_adaptive_pbrs()

        # Set graph data if environment available
        if self.env:
            self.update_graph_data(self.env)

        # Statistics tracking
        self.training_stats = {
            "episodes_trained": 0,
            "levels_seen": 0,
            "patterns_learned": 0,
            "routes_discovered": 0,
            "generalization_evaluations": 0,
        }

        logger.info("Initialized GeneralizedNavigationSystem")

    def _initialize_pattern_extractor(
        self, demonstration_data: Optional[List[Dict[str, Any]]]
    ) -> GeneralizedPatternExtractor:
        """Initialize and train pattern extractor from demonstrations."""

        extractor = GeneralizedPatternExtractor(
            neighborhood_sizes=self.config.pattern_neighborhood_sizes,
            entity_context_radius=self.config.pattern_entity_radius,
            min_pattern_observations=self.config.min_pattern_observations,
        )

        # Load existing pattern database if available
        pattern_db_path = self.save_dir / self.config.pattern_database_path
        if pattern_db_path.exists():
            logger.info(f"Loading existing pattern database from {pattern_db_path}")
            extractor.load_pattern_database(str(pattern_db_path))
        elif demonstration_data:
            logger.info(
                f"Building pattern database from {len(demonstration_data)} demonstrations"
            )
            extractor.build_pattern_database(demonstration_data)
            extractor.save_pattern_database(str(pattern_db_path))
        else:
            logger.warning(
                "No demonstration data provided, pattern extractor will learn online"
            )

        return extractor

    def _initialize_path_predictor(self) -> ProbabilisticPathPredictor:
        """Initialize multi-path predictor."""

        predictor_config = {
            "graph_feature_dim": self.config.graph_feature_dim,
            "tile_pattern_dim": self.config.tile_pattern_dim,
            "entity_feature_dim": self.config.entity_feature_dim,
            "num_path_candidates": self.config.num_path_candidates,
            "max_waypoints": self.config.max_waypoints,
        }

        predictor = create_multipath_predictor(predictor_config)
        
        # Load trained checkpoint if available
        if hasattr(self.config, 'path_predictor_checkpoint') and self.config.path_predictor_checkpoint:
            self.load_path_predictor_checkpoint(predictor, self.config.path_predictor_checkpoint)
        
        return predictor

    def _initialize_route_discovery(self) -> OnlineRouteDiscovery:
        """Initialize online route discovery system."""

        discovery = OnlineRouteDiscovery(
            exploration_factor=self.config.exploration_factor,
            min_exploration_prob=self.config.min_exploration_prob,
            topology_memory_size=self.config.topology_memory_size,
        )

        # Load existing state if available
        discovery_state_path = self.save_dir / self.config.route_discovery_state_path
        if discovery_state_path.exists():
            logger.info(f"Loading route discovery state from {discovery_state_path}")
            discovery.load_discovery_state(str(discovery_state_path))

        return discovery

    def _initialize_adaptive_pbrs(self) -> AdaptiveMultiPathPBRS:
        """Initialize adaptive multi-path PBRS calculator."""

        pbrs_config = {
            "learning_rate": self.config.pbrs_learning_rate,
            "uncertainty_bonus_weight": self.config.uncertainty_bonus_weight,
            "min_confidence_threshold": self.config.min_confidence_threshold,
            "path_memory_size": self.config.path_memory_size,
        }

        pbrs = create_adaptive_multipath_pbrs(pbrs_config)

        # Load existing state if available
        pbrs_state_path = self.save_dir / self.config.adaptive_pbrs_state_path
        if pbrs_state_path.exists():
            logger.info(f"Loading adaptive PBRS state from {pbrs_state_path}")
            pbrs.load_adaptive_state(str(pbrs_state_path))

        return pbrs

    def update_graph_data(self, env) -> None:
        """Extract and distribute graph data to all components.
        
        Args:
            env: Environment instance with get_graph_data method
        """
        from ..path_prediction.graph_utils import extract_graph_from_env
        
        adjacency, spatial_hash = extract_graph_from_env(env)
        
        if adjacency and spatial_hash:
            if self.pattern_extractor:
                self.pattern_extractor.set_graph_data(adjacency, spatial_hash)
            if self.path_predictor:
                self.path_predictor.set_graph_data(adjacency, spatial_hash)
            if self.route_discovery:
                self.route_discovery.set_graph_data(adjacency, spatial_hash)
            
            logger.info(f"Updated graph data: {len(adjacency)} nodes")
        else:
            logger.debug("Graph data not available yet")

    def create_enhanced_feature_extractor(
        self, observation_space, features_dim: int, architecture_config
    ) -> PatternAwareExtractor:
        """Create pattern-aware feature extractor for policy training.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            architecture_config: Architecture configuration

        Returns:
            PatternAwareExtractor with learned patterns integrated
        """
        return create_pattern_aware_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            architecture_config=architecture_config,
            pattern_extractor=self.pattern_extractor,
            path_predictor=self.path_predictor,
        )

    def process_training_episode(
        self,
        level_data: Dict[str, Any],
        trajectory_data: List[Dict[str, Any]],
        episode_outcome: Dict[str, Any],
    ) -> None:
        """Process a training episode to update all learning components.

        Args:
            level_data: Level configuration data
            trajectory_data: Trajectory steps from the episode
            episode_outcome: Episode results (success, time, etc.)
        """

        # Refresh graph data if level changed
        if self.env:
            self.update_graph_data(self.env)

        # Extract new patterns from this episode
        if self.pattern_extractor and trajectory_data:
            try:
                patterns = self.pattern_extractor.extract_tile_entity_patterns(
                    trajectory_data, level_data
                )

                # Update pattern database incrementally
                if patterns:
                    # For online learning, we could implement incremental updates here
                    # For now, patterns are learned offline from demonstrations
                    pass

            except Exception as e:
                logger.warning(f"Pattern extraction failed for episode: {e}")

        # Generate candidate paths for route discovery
        candidate_paths = self._generate_candidate_paths_for_episode(
            level_data, trajectory_data
        )

        # Track route discovery outcomes
        if candidate_paths:
            path_outcomes = self._extract_path_outcomes(
                trajectory_data, episode_outcome
            )

            self.route_discovery.track_exploration_outcomes(
                level_id=level_data.get("level_id", "unknown"),
                level_data=level_data,
                paths_tried=candidate_paths,
                results=path_outcomes,
            )

        # Update adaptive PBRS preferences
        if candidate_paths and "attempted_paths" in episode_outcome:
            pbrs_outcomes = {
                "attempted_paths": episode_outcome["attempted_paths"],
                "path_outcomes": episode_outcome.get("path_outcomes", []),
                "level_success": episode_outcome.get("success", False),
                "completion_time": episode_outcome.get("completion_time"),
            }

            self.adaptive_pbrs.update_path_rewards(pbrs_outcomes)

        # Update statistics
        self.training_stats["episodes_trained"] += 1
        if episode_outcome.get("success", False):
            self.training_stats["routes_discovered"] += 1

        # Periodic saves
        if self.training_stats["episodes_trained"] % 100 == 0:
            self.save_learned_state()

    def evaluate_generalization_performance(
        self, test_levels: List[Dict[str, Any]], policy_evaluator
    ) -> Dict[str, float]:
        """Evaluate generalization performance on completely unseen levels.

        Args:
            test_levels: List of unseen level configurations
            policy_evaluator: Function to evaluate policy on levels

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(
            f"Evaluating generalization performance on {len(test_levels)} unseen levels"
        )

        results = {
            "total_levels": len(test_levels),
            "successful_completions": 0,
            "average_completion_time": 0.0,
            "topology_performance": {},
            "pattern_recognition_accuracy": 0.0,
        }

        completion_times = []
        topology_results = {}

        for level_data in test_levels:
            # Classify level topology
            topology_type = self.route_discovery.topology_classifier.classify_level(
                level_data
            )
            if topology_type not in topology_results:
                topology_results[topology_type] = {
                    "attempts": 0,
                    "successes": 0,
                    "times": [],
                }

            # Evaluate policy on this level
            evaluation_result = policy_evaluator(level_data, self)

            # Track results
            topology_results[topology_type]["attempts"] += 1

            if evaluation_result.get("success", False):
                results["successful_completions"] += 1
                topology_results[topology_type]["successes"] += 1

                completion_time = evaluation_result.get("completion_time", 0.0)
                if completion_time > 0:
                    completion_times.append(completion_time)
                    topology_results[topology_type]["times"].append(completion_time)

        # Calculate aggregate metrics
        if completion_times:
            results["average_completion_time"] = np.mean(completion_times)

        results["success_rate"] = (
            results["successful_completions"] / results["total_levels"]
        )

        # Calculate per-topology performance
        for topology_type, topology_data in topology_results.items():
            if topology_data["attempts"] > 0:
                success_rate = topology_data["successes"] / topology_data["attempts"]
                avg_time = (
                    np.mean(topology_data["times"]) if topology_data["times"] else 0.0
                )

                results["topology_performance"][topology_type] = {
                    "success_rate": success_rate,
                    "average_time": avg_time,
                    "attempts": topology_data["attempts"],
                }

        # Update statistics
        self.training_stats["generalization_evaluations"] += 1

        logger.info(
            f"Generalization evaluation complete: {results['success_rate']:.2%} success rate"
        )

        return results

    def _generate_candidate_paths_for_episode(
        self, level_data: Dict[str, Any], trajectory_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate candidate paths that were explored during episode."""

        # This is a simplified version - in practice, this would extract
        # the actual paths the agent attempted during the episode
        candidate_paths = []

        if not trajectory_data:
            return candidate_paths

        # Extract waypoints from trajectory
        waypoints = []
        for step in trajectory_data[::5]:  # Sample every 5th step
            pos = step.get("position", step.get("obs", {}).get("player_pos"))
            if pos:
                waypoints.append((int(pos[0]), int(pos[1])))

        if len(waypoints) >= 2:
            candidate_paths.append(
                {
                    "waypoints": waypoints,
                    "path_type": "executed",
                    "estimated_cost": len(waypoints),
                    "risk_level": 0.5,
                }
            )

        return candidate_paths

    def _extract_path_outcomes(
        self, trajectory_data: List[Dict[str, Any]], episode_outcome: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract path outcome results from episode data."""

        path_outcome = {
            "success": episode_outcome.get("success", False),
            "completion_time": episode_outcome.get("completion_time"),
            "distance_traveled": len(trajectory_data) * 5.0,  # Approximate
            "energy_consumed": len(trajectory_data) * 0.1,
            "mines_triggered": episode_outcome.get("mines_triggered", 0),
            "deaths_occurred": episode_outcome.get("deaths", 0),
            "final_reward": episode_outcome.get("total_reward", 0.0),
        }

        return [path_outcome]

    def load_path_predictor_checkpoint(
        self, 
        predictor: ProbabilisticPathPredictor, 
        checkpoint_path: str
    ) -> None:
        """Load trained path predictor checkpoint.
        
        Args:
            predictor: Path predictor instance to load into
            checkpoint_path: Path to checkpoint file
        """
        import torch
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Path predictor checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading path predictor checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            predictor.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Successfully loaded path predictor checkpoint")
            
            # Log checkpoint info
            if "epoch" in checkpoint:
                logger.info(f"Checkpoint from epoch {checkpoint['epoch'] + 1}")
            if "best_val_loss" in checkpoint:
                logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        except Exception as e:
            logger.error(f"Failed to load path predictor checkpoint: {e}")
            logger.warning("Proceeding with random initialization")
    
    def save_path_predictor_checkpoint(self, checkpoint_path: str) -> None:
        """Save path predictor checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        import torch
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.path_predictor.state_dict(),
            "predictor_stats": self.path_predictor.get_statistics(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved path predictor checkpoint to {checkpoint_path}")

    def save_learned_state(self) -> None:
        """Save all learned states to disk."""

        # Save pattern database
        if self.pattern_extractor:
            pattern_path = self.save_dir / self.config.pattern_database_path
            self.pattern_extractor.save_pattern_database(str(pattern_path))

        # Save route discovery state
        discovery_path = self.save_dir / self.config.route_discovery_state_path
        self.route_discovery.save_discovery_state(str(discovery_path))

        # Save adaptive PBRS state
        pbrs_path = self.save_dir / self.config.adaptive_pbrs_state_path
        self.adaptive_pbrs.save_adaptive_state(str(pbrs_path))
        
        # Save path predictor checkpoint
        predictor_path = self.save_dir / "path_predictor_checkpoint.pt"
        self.save_path_predictor_checkpoint(str(predictor_path))

        # Save training statistics
        stats_path = self.save_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.training_stats, f, indent=2)

        logger.info(f"Saved learned state to {self.save_dir}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the learning system."""

        stats = {
            "training_stats": dict(self.training_stats),
            "pattern_extractor_stats": self.pattern_extractor.get_statistics()
            if self.pattern_extractor
            else {},
            "route_discovery_stats": self.route_discovery.get_statistics(),
            "adaptive_pbrs_stats": self.adaptive_pbrs.get_statistics(),
        }

        # Add pattern database size
        if self.pattern_extractor:
            stats["patterns_in_database"] = len(self.pattern_extractor.pattern_database)

        return stats

    def generate_training_report(self) -> str:
        """Generate a comprehensive training report."""

        stats = self.get_system_statistics()

        report = f"""
Generalized Navigation System Report
===================================

Training Progress:
- Episodes trained: {stats["training_stats"]["episodes_trained"]}
- Levels seen: {stats["training_stats"]["levels_seen"]}
- Routes discovered: {stats["training_stats"]["routes_discovered"]}
- Generalization evaluations: {stats["training_stats"]["generalization_evaluations"]}

Pattern Learning:
- Patterns in database: {stats.get("patterns_in_database", 0)}
- Reliable patterns: {stats["pattern_extractor_stats"].get("reliable_patterns", 0)}
- Demonstrations processed: {stats["pattern_extractor_stats"].get("demonstrations_processed", 0)}

Route Discovery:
- Total paths tried: {stats["route_discovery_stats"].get("total_paths_tried", 0)}
- Overall success rate: {stats["route_discovery_stats"].get("overall_success_rate", 0.0):.2%}
- Recent performance: {stats["route_discovery_stats"].get("recent_performance", 0.0):.3f}

Adaptive PBRS:
- Paths evaluated: {stats["adaptive_pbrs_stats"].get("paths_evaluated", 0)}
- Preference updates: {stats["adaptive_pbrs_stats"].get("preference_updates", 0)}
- Average path quality: {stats["adaptive_pbrs_stats"].get("avg_path_quality", 0.0):.3f}
- Average success rate: {stats["adaptive_pbrs_stats"].get("avg_success_rate", 0.0):.2%}

System Health:
- Pattern cache size: {stats.get("pattern_cache_size", 0)}
- Route memory usage: {len(stats.get("topology_performance", {}))} topologies
"""

        return report


def create_generalized_navigation_system(
    config_dict: Dict[str, Any],
    demonstrations: Optional[List[Dict[str, Any]]] = None,
    save_dir: str = "navigation_models",
    env=None,
) -> GeneralizedNavigationSystem:
    """Factory function to create GeneralizedNavigationSystem from config.

    Args:
        config_dict: Configuration dictionary
        demonstrations: Expert demonstration data
        save_dir: Directory for saving models
        env: Environment instance for graph access

    Returns:
        Configured GeneralizedNavigationSystem
    """
    config = GeneralizedNavigationConfig(**config_dict)
    return GeneralizedNavigationSystem(config, demonstrations, save_dir, env)
