"""
Simplified Graph Processor for NPP-RL.

This module replaces the complex conditional edge system with a simplified
approach that performs basic logical masking only. Complex physics-based
edge filtering is removed to allow HGT emergent learning.

Key simplifications:
- Simple switch/door state masking only
- No complex physics-based edge filtering
- No trajectory analysis or movement state filtering
- Let HGT learn movement constraints emergently

This aligns with HGT design principles: provide basic logical constraints
and let the network learn complex movement patterns through attention.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

# Use nclone physics constants
TILE_PIXEL_SIZE = 24  # Standard N++ tile size


class EdgeType(IntEnum):
    """Simplified edge types for basic connectivity."""

    ADJACENT = 0  # Basic 4-connectivity between traversable tiles
    LOGICAL = 1  # Switch-door relationships
    REACHABLE = 2  # Simple flood-fill connectivity


@dataclass
class EdgeInfo:
    """Simple edge information."""

    source: Tuple[float, float]
    target: Tuple[float, float]
    edge_type: EdgeType
    features: np.ndarray


class ConditionalEdgeMasker(nn.Module):
    """
    Simplified graph processor for NPP-RL.

    Replaces complex physics-based edge filtering with simple logical masking.
    This allows the HGT multimodal network to learn movement constraints
    emergently rather than having them hard-coded.

    Only performs:
    - Switch/door state masking
    - Basic logical relationship filtering
    - No complex physics calculations
    """

    def __init__(self, debug: bool = False):
        """Initialize simplified graph processor."""
        super().__init__()
        self.debug = debug

        # Register buffers for device compatibility
        self.register_buffer("_device_check", torch.tensor(0.0))

    def compute_dynamic_edge_mask(
        self,
        edge_features: torch.Tensor,
        ninja_physics_state: torch.Tensor,
        base_edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute simplified edge mask with basic logical filtering only.

        Args:
            edge_features: Edge features [num_edges, 4] with basic connectivity info
            ninja_physics_state: Physics features [8] from simplified extractor
            base_edge_mask: Base edge mask [num_edges] indicating valid edges

        Returns:
            torch.Tensor: Updated edge mask [num_edges] with logical filtering applied
        """
        if edge_features.size(0) == 0:
            return base_edge_mask

        # Start with base mask
        dynamic_mask = base_edge_mask.clone()

        # Extract switch activation state from ninja physics state
        # Assuming switch_activated is the last feature (index 7)
        if ninja_physics_state.size(0) > 7:
            switch_activated = (ninja_physics_state[7] > 0.5).item()
        else:
            switch_activated = False

        # Simple logical masking: disable door edges if switch not activated
        # Assuming edge type is stored in edge_features[:, 0]
        if edge_features.size(1) > 0:
            edge_types = edge_features[:, 0]

            # Disable LOGICAL edges (switch-door relationships) if switch not activated
            logical_edges = edge_types == EdgeType.LOGICAL
            if not switch_activated:
                dynamic_mask = dynamic_mask & ~logical_edges

        if self.debug:
            print(
                f"Edge mask: {base_edge_mask.sum().item()} → {dynamic_mask.sum().item()} edges"
            )

        return dynamic_mask

    def process_edges(
        self, edges: List[EdgeInfo], game_state: Dict[str, Any]
    ) -> List[EdgeInfo]:
        """
        Process edges with simple logical filtering.

        Args:
            edges: List of edge information
            game_state: Current game state (switches, etc.)

        Returns:
            List of processed edges with logical filtering applied
        """
        processed_edges = []
        switch_activated = game_state.get("exit_switch_activated", False)

        for edge in edges:
            # Keep all edges except logical edges when switch not activated
            if edge.edge_type == EdgeType.LOGICAL and not switch_activated:
                continue  # Skip this edge
            else:
                processed_edges.append(edge)

        if self.debug:
            print(f"Processed edges: {len(edges)} → {len(processed_edges)}")

        return processed_edges

    def create_edge_features(self, edge: EdgeInfo) -> np.ndarray:
        """
        Create simplified edge features (4 dimensions).

        Args:
            edge: Edge information

        Returns:
            Array of 4 edge features:
            [0]: Edge type (normalized)
            [1]: Distance (normalized)
            [2]: Direction X (normalized)
            [3]: Direction Y (normalized)
        """
        sx, sy = edge.source
        tx, ty = edge.target

        # Calculate basic features
        distance = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
        max_distance = TILE_PIXEL_SIZE * 10  # Reasonable max distance
        distance_norm = min(distance / max_distance, 1.0)

        # Direction vector (normalized)
        if distance > 0:
            dir_x = (tx - sx) / distance
            dir_y = (ty - sy) / distance
        else:
            dir_x = dir_y = 0.0

        # Normalize direction to [0,1] range
        dir_x_norm = (dir_x + 1.0) / 2.0
        dir_y_norm = (dir_y + 1.0) / 2.0

        features = np.array(
            [
                edge.edge_type / 3.0,  # Normalize edge type
                distance_norm,
                dir_x_norm,
                dir_y_norm,
            ],
            dtype=np.float32,
        )

        return np.clip(features, 0.0, 1.0)

    # Legacy compatibility methods
    def forward(
        self, edge_features: torch.Tensor, ninja_state: torch.Tensor
    ) -> torch.Tensor:
        """Legacy forward method for compatibility."""
        base_mask = torch.ones(
            edge_features.size(0), dtype=torch.bool, device=edge_features.device
        )
        return self.compute_dynamic_edge_mask(edge_features, ninja_state, base_mask)
