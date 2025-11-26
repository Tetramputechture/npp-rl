"""Probabilistic multi-path predictor for generating candidate paths with uncertainty.

This module implements a neural network that predicts multiple candidate paths
with confidence scores for unseen level configurations, supporting online learning
of path preferences during training.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CandidatePath:
    """Represents a candidate path with confidence and metadata.

    Note: Waypoints are in normalized [0, 1] coordinate space.
    """

    waypoints: List[
        Tuple[float, float]
    ]  # Sequence of normalized waypoint coordinates [0,1]
    confidence: float  # 0.0 to 1.0 confidence in path quality
    path_type: str  # 'direct', 'mine_avoidance', 'wall_jump', etc.
    estimated_cost: float  # Predicted traversal cost
    risk_level: float  # 0.0 (safe) to 1.0 (risky)

    def __hash__(self):
        # Hash based on waypoints for deduplication
        return hash(tuple(self.waypoints))


class PathUncertaintyHead(nn.Module):
    """Neural network head for uncertainty quantification of paths."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.uncertainty_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [aleatoric, epistemic] uncertainty
        )

    def forward(self, path_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict both aleatoric and epistemic uncertainty.

        Args:
            path_features: Path representation features [batch, feature_dim]

        Returns:
            Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        """
        uncertainty_raw = self.uncertainty_layers(path_features)

        # Split into aleatoric and epistemic components
        aleatoric = torch.sigmoid(uncertainty_raw[:, 0])  # Data-inherent uncertainty
        epistemic = torch.sigmoid(uncertainty_raw[:, 1])  # Model uncertainty

        return aleatoric, epistemic


class MultiHeadPathDecoder(nn.Module):
    """Multi-head decoder for generating diverse path candidates."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        max_waypoints: int = 20,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.max_waypoints = max_waypoints

        # Separate decoder head for each path type
        # Output is constrained to [0,1] via Sigmoid for normalized coordinates
        self.path_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, max_waypoints * 2),  # x, y coordinates
                    nn.Sigmoid(),  # Constrain to [0, 1] normalized coordinate space
                )
                for _ in range(num_heads)
            ]
        )

        # Confidence prediction for each head
        self.confidence_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_heads)
            ]
        )

        # Path quality metrics
        self.cost_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Softplus(),  # Ensure positive costs
        )

        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Sigmoid(),  # 0-1 risk level
        )

        # Generic candidate labels (no prescribed path types)
        self.candidate_labels = [f"candidate_{i}" for i in range(num_heads)]

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate multiple candidate paths.

        Args:
            features: Input features [batch, input_dim]

        Returns:
            Dictionary containing path predictions and metadata
        """
        batch_size = features.size(0)

        # Generate waypoints for each head
        all_waypoints = []
        all_confidences = []

        for i in range(self.num_heads):
            waypoints = self.path_heads[i](features)  # [batch, max_waypoints * 2]
            waypoints = waypoints.view(batch_size, self.max_waypoints, 2)

            confidence = self.confidence_heads[i](features)  # [batch, 1]

            all_waypoints.append(waypoints)
            all_confidences.append(confidence)

        # Stack outputs
        waypoints = torch.stack(
            all_waypoints, dim=1
        )  # [batch, num_heads, max_waypoints, 2]
        confidences = torch.cat(all_confidences, dim=1)  # [batch, num_heads]

        # Predict costs and risks (path types are derived post-hoc from properties)
        costs = self.cost_head(features)  # [batch, num_heads]
        risks = self.risk_head(features)  # [batch, num_heads]

        return {
            "waypoints": waypoints,
            "confidences": confidences,
            "costs": costs,
            "risks": risks,
        }


class ProbabilisticPathPredictor(nn.Module):
    """Predict multiple candidate paths with uncertainty estimates.

    This network generates diverse path candidates for unseen levels and
    supports online learning of path preferences based on training outcomes.
    """

    def __init__(
        self,
        graph_feature_dim: int = 256,
        tile_pattern_dim: int = 64,
        entity_feature_dim: int = 32,
        num_path_candidates: int = 4,
        max_waypoints: int = 20,
        hidden_dim: int = 512,
    ):
        """Initialize the probabilistic path predictor.

        Args:
            graph_feature_dim: Dimension of graph-based features
            tile_pattern_dim: Dimension of tile pattern features
            entity_feature_dim: Dimension of entity context features
            num_path_candidates: Number of diverse paths to generate
            max_waypoints: Maximum waypoints per path
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.num_path_candidates = num_path_candidates
        self.max_waypoints = max_waypoints

        # Graph data for validation (optional)
        self.adjacency_graph: Optional[Dict] = None
        self.spatial_hash: Optional[Any] = None

        # Input feature processing
        total_input_dim = graph_feature_dim + tile_pattern_dim + entity_feature_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Multi-head path decoder
        self.path_decoder = MultiHeadPathDecoder(
            input_dim=hidden_dim,
            num_heads=num_path_candidates,
            max_waypoints=max_waypoints,
            hidden_dim=hidden_dim,
        )

        # Uncertainty quantification
        self.uncertainty_head = PathUncertaintyHead(
            input_dim=hidden_dim, hidden_dim=hidden_dim // 2
        )

        # Online learning components
        self.path_preference_memory = {}  # path_hash -> preference_score
        self.adaptation_rate = 0.1

        # Training statistics
        self.paths_generated = 0
        self.preference_updates = 0

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        graph_obs: torch.Tensor,
        tile_patterns: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass to generate candidate paths.

        Args:
            graph_obs: Graph observation features [batch, graph_feature_dim]
            tile_patterns: Local tile pattern features [batch, tile_pattern_dim]
            entity_features: Entity context features [batch, entity_feature_dim]

        Returns:
            Dictionary containing path predictions and uncertainty estimates
        """
        # Concatenate input features
        combined_features = torch.cat(
            [graph_obs, tile_patterns, entity_features], dim=1
        )

        # Feature fusion
        fused_features = self.feature_fusion(combined_features)

        # Generate paths
        path_outputs = self.path_decoder(fused_features)

        # Predict uncertainty
        aleatoric_uncertainty, epistemic_uncertainty = self.uncertainty_head(
            fused_features
        )

        # Combine outputs
        outputs = {
            **path_outputs,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "fused_features": fused_features,
        }

        return outputs

    def predict_candidate_paths(
        self,
        graph_obs: torch.Tensor,
        tile_patterns: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> List[CandidatePath]:
        """Generate candidate paths for a level configuration.

        Args:
            graph_obs: Graph observation features
            tile_patterns: Local tile pattern features
            entity_features: Entity context features

        Returns:
            List of CandidatePath objects with predictions
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(graph_obs, tile_patterns, entity_features)

            batch_size = graph_obs.size(0)
            all_candidate_paths = []

            for batch_idx in range(batch_size):
                candidate_paths = []

                for path_idx in range(self.num_path_candidates):
                    # Extract waypoints and filter out invalid ones
                    waypoints_raw = outputs["waypoints"][
                        batch_idx, path_idx
                    ]  # [max_waypoints, 2]
                    waypoints = self._filter_valid_waypoints(waypoints_raw)

                    # Validate and repair path using graph if available
                    if self.adjacency_graph and len(waypoints) > 0:
                        waypoints = self._validate_and_repair_path(waypoints)

                    if len(waypoints) == 0:
                        continue

                    # Get path metadata
                    confidence = outputs["confidences"][batch_idx, path_idx].item()
                    # Use generic label - actual path type will be derived from properties
                    path_type = self.path_decoder.candidate_labels[path_idx]
                    cost = outputs["costs"][batch_idx, path_idx].item()
                    risk = outputs["risks"][batch_idx, path_idx].item()

                    # Adjust confidence based on uncertainty
                    aleatoric = outputs["aleatoric_uncertainty"][batch_idx].item()
                    epistemic = outputs["epistemic_uncertainty"][batch_idx].item()
                    uncertainty_penalty = (aleatoric + epistemic) / 2.0
                    adjusted_confidence = confidence * (1.0 - uncertainty_penalty * 0.5)

                    # Apply online learning preferences if available
                    path_hash = self._hash_path(waypoints)
                    if path_hash in self.path_preference_memory:
                        preference_score = self.path_preference_memory[path_hash]
                        adjusted_confidence = (
                            0.7 * adjusted_confidence + 0.3 * preference_score
                        )

                    candidate_path = CandidatePath(
                        waypoints=waypoints,
                        confidence=float(adjusted_confidence),
                        path_type=path_type,
                        estimated_cost=float(cost),
                        risk_level=float(risk),
                    )

                    candidate_paths.append(candidate_path)

                # Sort by confidence and remove duplicates
                candidate_paths = self._deduplicate_paths(candidate_paths)
                candidate_paths.sort(key=lambda p: p.confidence, reverse=True)

                all_candidate_paths.extend(candidate_paths)

            self.paths_generated += len(all_candidate_paths)
            return all_candidate_paths

    def set_graph_data(self, adjacency: Dict, spatial_hash: Any) -> None:
        """Update graph data for validation.

        Args:
            adjacency: Graph adjacency dictionary from GraphBuilder
            spatial_hash: SpatialHash for fast node lookups
        """
        self.adjacency_graph = adjacency
        self.spatial_hash = spatial_hash
        if adjacency:
            logger.debug(f"Path predictor updated with graph: {len(adjacency)} nodes")

    def _validate_and_repair_path(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Snap waypoints to graph nodes and repair invalid connections.

        Args:
            waypoints: List of predicted waypoint positions (normalized [0,1] coordinates)

        Returns:
            Repaired path with valid graph connections (normalized [0,1] coordinates)
        """
        if not self.adjacency_graph:
            return waypoints

        from .graph_utils import repair_path_with_graph

        # Denormalize to pixel coordinates for graph operations
        waypoints_pixels = self.denormalize_waypoints(waypoints)
        # Convert to int tuples for graph operations
        waypoints_pixels_int = [(int(x), int(y)) for x, y in waypoints_pixels]

        # Repair using graph
        repaired_pixels = repair_path_with_graph(
            waypoints_pixels_int,
            self.adjacency_graph,
            self.spatial_hash,
            snap_threshold=24,
        )

        # Renormalize back to [0, 1]
        level_width, level_height = 1056.0, 600.0
        repaired_normalized = [
            (float(x) / level_width, float(y) / level_height)
            for x, y in repaired_pixels
        ]

        return repaired_normalized

    def update_path_preferences(
        self, attempted_paths: List[CandidatePath], outcomes: List[float]
    ) -> None:
        """Update path preferences based on observed outcomes.

        Args:
            attempted_paths: Paths that were attempted during training
            outcomes: Success scores for each path (0.0 = failure, 1.0 = success)
        """
        if len(attempted_paths) != len(outcomes):
            logger.warning("Mismatch between attempted paths and outcomes")
            return

        for path, outcome in zip(attempted_paths, outcomes):
            path_hash = self._hash_path(path.waypoints)

            # Update preference using exponential moving average
            current_preference = self.path_preference_memory.get(path_hash, 0.5)
            new_preference = (
                1 - self.adaptation_rate
            ) * current_preference + self.adaptation_rate * outcome

            self.path_preference_memory[path_hash] = new_preference
            self.preference_updates += 1

        logger.debug(f"Updated preferences for {len(attempted_paths)} paths")

    def _filter_valid_waypoints(
        self, waypoints_raw: torch.Tensor
    ) -> List[Tuple[float, float]]:
        """Filter out invalid waypoints and convert to coordinate list.

        Note: Waypoints are expected to be in normalized [0, 1] coordinate space.
        """
        import math

        waypoints = []

        for i in range(waypoints_raw.size(0)):
            x, y = waypoints_raw[i, 0].item(), waypoints_raw[i, 1].item()

            # Check for NaN or infinity first
            if not math.isfinite(x) or not math.isfinite(y):
                break  # Stop processing this path if we hit invalid coordinates

            # Basic validity checks (normalized coordinates should be ~[0,1])
            if abs(x) > 1.5 or abs(y) > 1.5:  # Allow slight extrapolation
                break

            # Avoid duplicate consecutive waypoints (~1 pixel at 1000px scale)
            if (
                waypoints
                and abs(x - waypoints[-1][0]) < 0.001
                and abs(y - waypoints[-1][1]) < 0.001
            ):
                continue

            waypoints.append((float(x), float(y)))

        return waypoints

    def _hash_path(self, waypoints: List[Tuple[float, float]]) -> str:
        """Create hash for path deduplication and memory lookup.

        Note: Works with normalized [0, 1] waypoint coordinates.
        """
        # Simplify waypoints to reduce hash complexity
        if len(waypoints) <= 5:
            path_str = "_".join(f"{x:.3f}_{y:.3f}" for x, y in waypoints)
        else:
            # For long paths, use start, middle, and end points
            key_points = [waypoints[0], waypoints[len(waypoints) // 2], waypoints[-1]]
            path_str = "_".join(f"{x:.3f}_{y:.3f}" for x, y in key_points)

        return str(hash(path_str) % 1000000)  # Limit hash size

    def _deduplicate_paths(self, paths: List[CandidatePath]) -> List[CandidatePath]:
        """Remove duplicate paths based on waypoint similarity."""
        seen_hashes = set()
        unique_paths = []

        for path in paths:
            path_hash = self._hash_path(path.waypoints)
            if path_hash not in seen_hashes:
                seen_hashes.add(path_hash)
                unique_paths.append(path)

        return unique_paths

    @staticmethod
    def denormalize_waypoints(
        normalized_waypoints: List[Tuple[float, float]],
        level_width: float = 1056.0,
        level_height: float = 600.0,
    ) -> List[Tuple[float, float]]:
        """Convert normalized [0,1] waypoints back to tile-data pixel coordinates.

        Note: Returns waypoints in TILE-DATA coordinates (no border).
        To convert to world coordinates for rendering, add NODE_WORLD_COORD_OFFSET (24px).

        Args:
            normalized_waypoints: Waypoints in [0, 1] normalized space
            level_width: Level width in pixels (default: 1056 = 44*24)
            level_height: Level height in pixels (default: 600 = 25*24)

        Returns:
            Waypoints in tile-data pixel coordinates (no border)
        """
        from .coordinate_utils import denormalize_waypoints as denorm_util

        return denorm_util(normalized_waypoints, level_width, level_height)

    def get_statistics(self) -> Dict[str, int]:
        """Get predictor statistics."""
        return {
            "paths_generated": self.paths_generated,
            "preference_updates": self.preference_updates,
            "paths_in_memory": len(self.path_preference_memory),
        }

    def clear_preference_memory(self) -> None:
        """Clear path preference memory (useful for new training phases)."""
        self.path_preference_memory.clear()
        logger.info("Cleared path preference memory")

    def save_preferences(self, filepath: str) -> None:
        """Save path preferences to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.path_preference_memory, f)

        logger.info(
            f"Saved {len(self.path_preference_memory)} path preferences to {filepath}"
        )

    def load_preferences(self, filepath: str) -> None:
        """Load path preferences from file."""
        import pickle

        try:
            with open(filepath, "rb") as f:
                self.path_preference_memory = pickle.load(f)

            logger.info(
                f"Loaded {len(self.path_preference_memory)} path preferences from {filepath}"
            )
        except FileNotFoundError:
            logger.warning(f"Preference file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")


def create_multipath_predictor(config: Dict[str, Any]) -> ProbabilisticPathPredictor:
    """Factory function to create ProbabilisticPathPredictor from config.

    Args:
        config: Configuration dictionary with network parameters

    Returns:
        Configured ProbabilisticPathPredictor instance
    """
    return ProbabilisticPathPredictor(
        graph_feature_dim=config.get("graph_feature_dim", 256),
        tile_pattern_dim=config.get("tile_pattern_dim", 64),
        entity_feature_dim=config.get("entity_feature_dim", 32),
        num_path_candidates=config.get("num_path_candidates", 4),
        max_waypoints=config.get("max_waypoints", 20),
        hidden_dim=config.get("hidden_dim", 512),
    )
