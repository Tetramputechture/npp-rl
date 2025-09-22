"""
Heterogeneous Graph Transformer (HGT) implementation for N++ level understanding.

This module implements HGT for processing heterogeneous graphs with different
node types (grid cells, entities) and edge types (movement, functional relationships).
The HGT uses type-specific attention mechanisms and specialized processing for
different game elements.

Based on "Heterogeneous Graph Transformer" by Wang et al. (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
from enum import IntEnum

from .entity_type_system import (
    EntitySpecializedEmbedding,
    HazardAwareAttention,
    create_entity_type_system,
)


class EdgeType(IntEnum):
    """Types of edges in the heterogeneous graph."""

    # Movement edges
    WALK = 0
    JUMP = 1
    WALL_SLIDE = 2
    FALL = 3
    ONE_WAY = 4

    # Functional relationships
    FUNCTIONAL = 5  # switch->door, launchpad->target, etc.


class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer layer with type-specific attention.

    Implements multi-head attention with separate parameters for different
    node and edge types, enabling specialized processing of heterogeneous
    graph structures.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_node_types: int = 3,
        num_edge_types: int = 6,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        """
        Initialize HGT layer.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_heads: Number of attention heads
            num_node_types: Number of node types
            num_edge_types: Number of edge types
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.d_k = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Type-specific linear transformations
        # K, Q, V projections for each node type
        self.k_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_node_types)]
        )
        self.q_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_node_types)]
        )
        self.v_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for _ in range(num_node_types)]
        )

        # Edge-type specific attention parameters
        self.relation_pri = nn.Parameter(torch.ones(num_edge_types, num_heads))
        self.relation_att = nn.ModuleList(
            [nn.Linear(self.d_k, self.d_k, bias=False) for _ in range(num_edge_types)]
        )
        self.relation_msg = nn.ModuleList(
            [nn.Linear(self.d_k, self.d_k, bias=False) for _ in range(num_edge_types)]
        )

        # Output projection and normalization
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if use_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        for linear in self.k_linears + self.q_linears + self.v_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        for linear in self.relation_att + self.relation_msg:
            nn.init.xavier_uniform_(linear.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.ones_(self.relation_pri)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through HGT layer.

        Args:
            node_features: Node features [batch_size, num_nodes, in_dim]
            edge_index: Edge indices [batch_size, 2, num_edges]
            node_types: Node types [batch_size, num_nodes]
            edge_types: Edge types [batch_size, num_edges]
            node_mask: Node mask [batch_size, num_nodes]
            edge_mask: Edge mask [batch_size, num_edges]

        Returns:
            Updated node features [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        _, _, num_edges = edge_index.shape

        # Compute K, Q, V for each node based on its type
        K = torch.zeros(
            batch_size, num_nodes, self.out_dim, device=node_features.device
        )
        Q = torch.zeros(
            batch_size, num_nodes, self.out_dim, device=node_features.device
        )
        V = torch.zeros(
            batch_size, num_nodes, self.out_dim, device=node_features.device
        )

        for node_type in range(self.num_node_types):
            # Find nodes of this type
            type_mask = node_types == node_type  # [batch_size, num_nodes]

            if type_mask.any():
                # Apply type-specific transformations
                K[type_mask] = self.k_linears[node_type](node_features[type_mask])
                Q[type_mask] = self.q_linears[node_type](node_features[type_mask])
                V[type_mask] = self.v_linears[node_type](node_features[type_mask])

        # Reshape for multi-head attention
        K = K.view(batch_size, num_nodes, self.num_heads, self.d_k)
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.d_k)
        V = V.view(batch_size, num_nodes, self.num_heads, self.d_k)

        # Compute attention and messages
        output = torch.zeros_like(Q)

        for b in range(batch_size):
            # Get valid edges for this batch
            valid_edges = edge_mask[b].bool()
            if not valid_edges.any():
                continue

            src_nodes = edge_index[b, 0, valid_edges]  # [num_valid_edges]
            tgt_nodes = edge_index[b, 1, valid_edges]  # [num_valid_edges]
            edge_types_batch = edge_types[b, valid_edges]  # [num_valid_edges]

            # Process each edge type separately
            for edge_type in range(self.num_edge_types):
                edge_type_mask = edge_types_batch == edge_type
                if not edge_type_mask.any():
                    continue

                # Get edges of this type
                src_idx = src_nodes[edge_type_mask]
                tgt_idx = tgt_nodes[edge_type_mask]

                # Get source and target features
                src_k = K[b, src_idx]  # [num_edges_type, num_heads, d_k]
                src_v = V[b, src_idx]  # [num_edges_type, num_heads, d_k]
                tgt_q = Q[b, tgt_idx]  # [num_edges_type, num_heads, d_k]

                # Apply edge-type specific transformations
                src_k = self.relation_att[edge_type](src_k)
                src_v = self.relation_msg[edge_type](src_v)

                # Compute attention scores
                att_scores = torch.sum(
                    tgt_q * src_k, dim=-1
                )  # [num_edges_type, num_heads]
                att_scores = (
                    att_scores * self.relation_pri[edge_type] / math.sqrt(self.d_k)
                )
                att_scores = F.softmax(att_scores, dim=0)  # Normalize over source nodes

                # Apply attention to values
                messages = (
                    att_scores.unsqueeze(-1) * src_v
                )  # [num_edges_type, num_heads, d_k]

                # Aggregate messages to target nodes
                for i, tgt in enumerate(tgt_idx):
                    output[b, tgt] += messages[i]

        # Reshape and project output
        output = output.view(batch_size, num_nodes, self.out_dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Add residual connection and normalization
        if self.norm is not None:
            output = self.norm(
                output + node_features if self.in_dim == self.out_dim else output
            )

        # Apply node mask
        output = output * node_mask.unsqueeze(-1)

        return output


class HGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer encoder for N++ level understanding.

    Processes heterogeneous graphs with specialized attention mechanisms
    for different entity types and functional relationships.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        output_dim: int = 512,
        num_node_types: int = 3,
        num_edge_types: int = 6,
        dropout: float = 0.1,
        global_pool: str = "mean_max",
    ):
        """
        Initialize HGT encoder.

        Args:
            node_feature_dim: Input node feature dimension
            edge_feature_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            output_dim: Final output dimension
            num_node_types: Number of node types
            num_edge_types: Number of edge types
            dropout: Dropout probability
            global_pool: Global pooling method ('mean', 'max', 'mean_max')
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.global_pool = global_pool

        # Entity type system for specialized processing
        self.entity_type_system = create_entity_type_system()

        # Specialized input embedding
        self.input_embedding = EntitySpecializedEmbedding(
            input_dim=node_feature_dim,
            output_dim=hidden_dim,
            entity_type_system=self.entity_type_system,
            dropout=dropout,
        )

        # HGT layers
        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = HGTLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            self.hgt_layers.append(layer)

        # Hazard-aware attention for final processing
        self.hazard_attention = HazardAwareAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            entity_type_system=self.entity_type_system,
            dropout=dropout,
        )

        # Global pooling and output projection
        pool_dim = hidden_dim
        if global_pool == "mean_max":
            pool_dim = hidden_dim * 2

        self.output_projection = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through HGT encoder.

        Args:
            graph_obs: Dictionary containing:
                - graph_node_feats: [batch_size, num_nodes, node_feat_dim]
                - graph_edge_index: [batch_size, 2, num_edges]
                - graph_edge_feats: [batch_size, num_edges, edge_feat_dim]
                - graph_node_mask: [batch_size, num_nodes]
                - graph_edge_mask: [batch_size, num_edges]
                - graph_node_types: [batch_size, num_nodes] (optional)
                - graph_edge_types: [batch_size, num_edges] (optional)

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        node_features = graph_obs["graph_node_feats"]
        edge_index = graph_obs["graph_edge_index"]
        node_mask = graph_obs["graph_node_mask"]
        edge_mask = graph_obs["graph_edge_mask"]

        # Extract or infer node and edge types (prefer explicit types from graph builder)
        node_types = graph_obs.get("graph_node_types")
        edge_types = graph_obs.get("graph_edge_types")

        # Only use inference as fallback if explicit types are not provided
        if node_types is None:
            print(
                "Warning: Using node type inference fallback. Consider updating graph builder to provide explicit node_types."
            )
            node_types = self._infer_node_types(node_features)

        if edge_types is None:
            print(
                "Warning: Using edge type inference fallback. Consider updating graph builder to provide explicit edge_types."
            )
            edge_types = self._infer_edge_types(graph_obs.get("graph_edge_feats"))
        entity_types = graph_obs.get("graph_entity_types", None)

        # Specialized input embedding
        x = self.input_embedding(node_features, node_types, entity_types)

        # Apply HGT layers
        for layer in self.hgt_layers:
            x = layer(x, edge_index, node_types, edge_types, node_mask, edge_mask)

        # Apply hazard-aware attention
        x, _ = self.hazard_attention(
            x, x, x, entity_types, key_padding_mask=~node_mask.bool()
        )

        # Global pooling
        graph_embedding = self._global_pool(x, node_mask)

        # Output projection
        output = self.output_projection(graph_embedding)

        return output

    def _infer_node_types(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Infer node types from node features using robust feature analysis.

        This method uses multiple heuristics and validation to determine node types:
        1. Explicit type encoding (if present)
        2. Feature pattern analysis
        3. Statistical feature analysis
        4. Fallback heuristics with validation

        Args:
            node_features: Node feature tensor [batch_size, num_nodes, feature_dim]

        Returns:
            Node type tensor [batch_size, num_nodes] with values:
            0 = GRID_CELL, 1 = ENTITY, 2 = NINJA
        """
        batch_size, num_nodes, feature_dim = node_features.shape
        device = node_features.device

        # Initialize node types
        node_types = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            for node_idx in range(num_nodes):
                features = node_features[batch_idx, node_idx]

                # Method 1: Check for explicit type encoding
                # Many graph builders include one-hot type encoding in first few dimensions
                if feature_dim >= 3:
                    type_indicators = features[:3]
                    if torch.sum(type_indicators) > 0.8:  # Strong type signal
                        node_types[batch_idx, node_idx] = torch.argmax(type_indicators)
                        continue

                # Method 2: Physics-based feature analysis
                node_type = self._analyze_physics_features(features, feature_dim)
                if node_type is not None:
                    node_types[batch_idx, node_idx] = node_type
                    continue

                # Method 3: Statistical feature analysis
                node_type = self._analyze_statistical_features(features, feature_dim)
                if node_type is not None:
                    node_types[batch_idx, node_idx] = node_type
                    continue

                # Method 4: Fallback - assume grid cell
                node_types[batch_idx, node_idx] = 0  # GRID_CELL

        # Validation: Ensure at least some diversity in node types
        self._validate_node_type_distribution(node_types)

        return node_types

    def _analyze_physics_features(
        self, features: torch.Tensor, feature_dim: int
    ) -> Optional[int]:
        """Analyze physics-based features to determine node type."""
        # Check for velocity features (typically dimensions 2-3 or 7-8)
        if feature_dim > 8:
            # Check common velocity feature locations
            velocity_candidates = [
                features[2:4],  # Early velocity features
                features[7:9],  # Mid-range velocity features
            ]

            for vel_features in velocity_candidates:
                vel_magnitude = torch.norm(vel_features)
                if vel_magnitude > 0.1:  # Significant velocity
                    # High velocity suggests ninja node
                    if vel_magnitude > 5.0:
                        return 2  # NINJA
                    # Low velocity might be moving entity
                    elif vel_magnitude > 0.5:
                        return 1  # ENTITY

        # Check for entity-specific features (proximity, state indicators)
        if feature_dim > 20:
            # Entity proximity features (typically around index 20-22)
            entity_proximity = features[20:23] if feature_dim > 22 else features[-3:]
            if torch.any(entity_proximity > 0.1):
                return 0  # GRID_CELL (has entity proximity)

        return None

    def _analyze_statistical_features(
        self, features: torch.Tensor, feature_dim: int
    ) -> Optional[int]:
        """Analyze statistical properties of features to determine node type."""
        # Calculate feature statistics
        feature_mean = torch.mean(features)
        feature_std = torch.std(features)
        feature_max = torch.max(features)
        non_zero_count = torch.sum(features != 0).float()

        # Ninja nodes typically have diverse, high-magnitude features
        if (
            feature_std > 2.0
            and feature_max > 10.0
            and non_zero_count > feature_dim * 0.3
        ):
            return 2  # NINJA

        # Entity nodes have moderate diversity and magnitude
        elif (
            feature_std > 0.5
            and feature_max > 1.0
            and non_zero_count > feature_dim * 0.2
        ):
            return 1  # ENTITY

        # Grid cells typically have sparse, low-magnitude features
        elif non_zero_count < feature_dim * 0.1 or feature_max < 0.5:
            return 0  # GRID_CELL

        return None

    def _validate_node_type_distribution(self, node_types: torch.Tensor) -> None:
        """Validate that node type distribution is reasonable."""
        batch_size, num_nodes = node_types.shape

        for batch_idx in range(batch_size):
            types = node_types[batch_idx]

            # Count each type
            grid_count = torch.sum(types == 0).item()
            entity_count = torch.sum(types == 1).item()
            ninja_count = torch.sum(types == 2).item()

            # Validation rules
            # 1. Should have mostly grid cells
            if grid_count < num_nodes * 0.5:
                # Too few grid cells - convert some entities to grid cells
                entity_indices = torch.where(types == 1)[0]
                if len(entity_indices) > 0:
                    convert_count = min(
                        len(entity_indices), int(num_nodes * 0.5) - grid_count
                    )
                    node_types[batch_idx, entity_indices[:convert_count]] = 0

            # 2. Should have at most a few ninja nodes
            if ninja_count > max(1, num_nodes // 20):
                # Too many ninjas - convert excess to entities
                ninja_indices = torch.where(types == 2)[0]
                if len(ninja_indices) > 1:
                    keep_count = max(1, num_nodes // 20)
                    node_types[batch_idx, ninja_indices[keep_count:]] = 1

    def _infer_edge_types(self, edge_features: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Infer edge types from edge features using robust movement analysis.

        This method uses multiple approaches to determine edge types:
        1. Explicit type encoding (if present)
        2. Movement pattern analysis
        3. Physics-based classification
        4. Geometric analysis
        5. Validation and consistency checks

        Args:
            edge_features: Edge feature tensor [batch_size, num_edges, feature_dim]

        Returns:
            Edge type tensor [batch_size, num_edges] with values:
            0 = WALK, 1 = JUMP, 2 = WALL_JUMP, 3 = LAUNCH_PAD, 4 = FALL, 5 = SLIDE
        """
        if edge_features is None:
            # Default to WALK edges if no edge features
            batch_size = 1  # This should be extracted from other tensors
            num_edges = 1
            return torch.zeros(batch_size, num_edges, dtype=torch.long)

        batch_size, num_edges, feature_dim = edge_features.shape
        device = edge_features.device

        # Initialize edge types
        edge_types = torch.zeros(batch_size, num_edges, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            for edge_idx in range(num_edges):
                features = edge_features[batch_idx, edge_idx]

                # Method 1: Check for explicit type encoding
                if feature_dim >= 6:
                    type_indicators = features[:6]
                    if torch.sum(type_indicators) > 0.8:  # Strong type signal
                        edge_types[batch_idx, edge_idx] = torch.argmax(type_indicators)
                        continue

                # Method 2: Movement pattern analysis
                edge_type = self._analyze_movement_pattern(features, feature_dim)
                if edge_type is not None:
                    edge_types[batch_idx, edge_idx] = edge_type
                    continue

                # Method 3: Physics-based classification
                edge_type = self._analyze_physics_movement(features, feature_dim)
                if edge_type is not None:
                    edge_types[batch_idx, edge_idx] = edge_type
                    continue

                # Method 4: Geometric analysis
                edge_type = self._analyze_geometric_properties(features, feature_dim)
                if edge_type is not None:
                    edge_types[batch_idx, edge_idx] = edge_type
                    continue

                # Method 5: Fallback - assume walk
                edge_types[batch_idx, edge_idx] = 0  # WALK

        # Validation: Ensure edge type distribution is reasonable
        self._validate_edge_type_distribution(edge_types)

        return edge_types

    def _analyze_movement_pattern(
        self, features: torch.Tensor, feature_dim: int
    ) -> Optional[int]:
        """Analyze movement patterns to determine edge type."""
        if feature_dim < 8:
            return None

        # Look for velocity/displacement features
        # Common locations: [2:4], [4:6], [6:8]
        velocity_candidates = []
        if feature_dim > 4:
            velocity_candidates.append(features[2:4])
        if feature_dim > 6:
            velocity_candidates.append(features[4:6])
        if feature_dim > 8:
            velocity_candidates.append(features[6:8])

        for vel_features in velocity_candidates:
            vx, vy = vel_features[0], vel_features[1]

            # High upward velocity suggests jump
            if vy < -3.0:  # Negative Y is up in many coordinate systems
                if abs(vx) > 2.0:
                    return 2  # WALL_JUMP
                else:
                    return 1  # JUMP

            # High downward velocity suggests fall
            elif vy > 5.0:
                return 4  # FALL

            # High horizontal velocity with moderate vertical
            elif abs(vx) > 4.0 and abs(vy) < 2.0:
                return 0  # WALK (fast horizontal movement)

            # Very high velocity suggests launch pad
            elif torch.norm(vel_features) > 8.0:
                return 3  # LAUNCH_PAD

        return None

    def _analyze_physics_movement(
        self, features: torch.Tensor, feature_dim: int
    ) -> Optional[int]:
        """Analyze physics-based features to determine edge type."""
        if feature_dim < 12:
            return None

        # Look for physics indicators in later features
        physics_features = features[8:12] if feature_dim > 12 else features[-4:]

        # Energy-based classification
        if len(physics_features) >= 2:
            kinetic_energy = physics_features[0]
            potential_energy = physics_features[1] if len(physics_features) > 1 else 0

            # High kinetic energy suggests dynamic movement
            if kinetic_energy > 5.0:
                if potential_energy > 2.0:
                    return 1  # JUMP (high kinetic + potential)
                else:
                    return 0  # WALK (high kinetic, low potential)

            # High potential energy suggests vertical movement
            elif potential_energy > 3.0:
                return 4  # FALL (gaining potential energy)

        # Contact state analysis (if available)
        if feature_dim > 15:
            contact_features = features[12:15]
            ground_contact = contact_features[0] if len(contact_features) > 0 else 0
            wall_contact = contact_features[1] if len(contact_features) > 1 else 0

            if wall_contact > 0.5:
                return 5  # SLIDE (wall contact)
            elif ground_contact < 0.1:
                return 4  # FALL (no ground contact)

        return None

    def _analyze_geometric_properties(
        self, features: torch.Tensor, feature_dim: int
    ) -> Optional[int]:
        """Analyze geometric properties to determine edge type."""
        # Look for distance/direction features
        if feature_dim > 20:
            # Distance features often in later dimensions
            distance_features = features[-5:]

            # Very long distance suggests launch pad or special movement
            if torch.max(distance_features) > 10.0:
                return 3  # LAUNCH_PAD

            # Moderate distance with specific patterns
            elif torch.max(distance_features) > 3.0:
                # Check for vertical bias
                if len(distance_features) >= 2:
                    horizontal_dist = distance_features[0]
                    vertical_dist = distance_features[1]

                    if abs(vertical_dist) > abs(horizontal_dist) * 1.5:
                        return 1  # JUMP (vertical bias)

        return None

    def _validate_edge_type_distribution(self, edge_types: torch.Tensor) -> None:
        """Validate that edge type distribution is reasonable."""
        batch_size, num_edges = edge_types.shape

        for batch_idx in range(batch_size):
            types = edge_types[batch_idx]

            # Count each type
            walk_count = torch.sum(types == 0).item()
            jump_count = torch.sum(types == 1).item()
            wall_jump_count = torch.sum(types == 2).item()
            launch_pad_count = torch.sum(types == 3).item()
            fall_count = torch.sum(types == 4).item()
            slide_count = torch.sum(types == 5).item()

            # Validation rules
            # 1. Should have mostly walk edges
            if walk_count < num_edges * 0.3:
                # Too few walks - convert some other types to walks
                non_walk_indices = torch.where(types != 0)[0]
                if len(non_walk_indices) > 0:
                    convert_count = min(
                        len(non_walk_indices), int(num_edges * 0.3) - walk_count
                    )
                    edge_types[batch_idx, non_walk_indices[:convert_count]] = 0

            # 2. Launch pad edges should be rare
            if launch_pad_count > max(1, num_edges // 10):
                # Too many launch pads - convert excess to jumps
                launch_pad_indices = torch.where(types == 3)[0]
                if len(launch_pad_indices) > 1:
                    keep_count = max(1, num_edges // 10)
                    edge_types[batch_idx, launch_pad_indices[keep_count:]] = 1

    def _global_pool(
        self, node_features: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply global pooling to get graph-level representation."""
        # Mask out invalid nodes
        masked_features = node_features * node_mask.unsqueeze(-1)

        if self.global_pool == "mean":
            # Mean pooling
            num_valid_nodes = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            graph_emb = masked_features.sum(dim=1) / num_valid_nodes

        elif self.global_pool == "max":
            # Max pooling
            masked_features = masked_features.masked_fill(
                ~node_mask.unsqueeze(-1).bool(), float("-inf")
            )
            graph_emb = masked_features.max(dim=1)[0]

        elif self.global_pool == "mean_max":
            # Concatenate mean and max pooling
            num_valid_nodes = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pool = masked_features.sum(dim=1) / num_valid_nodes

            masked_for_max = masked_features.masked_fill(
                ~node_mask.unsqueeze(-1).bool(), float("-inf")
            )
            max_pool = masked_for_max.max(dim=1)[0]

            graph_emb = torch.cat([mean_pool, max_pool], dim=1)

        else:
            raise ValueError(f"Unknown global pooling method: {self.global_pool}")

        return graph_emb


def create_hgt_encoder(
    node_feature_dim: int, edge_feature_dim: int, **kwargs
) -> HGTEncoder:
    """
    Create an HGT encoder with default parameters.

    Args:
        node_feature_dim: Input node feature dimension
        edge_feature_dim: Input edge feature dimension
        **kwargs: Additional parameters for HGTEncoder

    Returns:
        Configured HGTEncoder instance
    """
    default_params = {
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 8,
        "output_dim": 512,
        "num_node_types": 3,
        "num_edge_types": 6,
        "dropout": 0.1,
        "global_pool": "mean_max",
    }

    # Override defaults with provided kwargs
    params = {**default_params, **kwargs}

    return HGTEncoder(
        node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, **params
    )
