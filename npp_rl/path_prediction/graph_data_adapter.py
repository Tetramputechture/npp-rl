"""Graph data adapter for converting adjacency dictionaries to GNN format.

This module converts adjacency graph data (from GraphBuilder) into the format
expected by GNN encoders (GATEncoder, GCNEncoder), providing:
- node_features: [batch, max_nodes, feat_dim]
- edge_index: [batch, 2, num_edges]
- node_mask: [batch, max_nodes]
- edge_mask: [batch, num_edges]

The adapter also handles node indexing, position-to-ID mapping, and feature extraction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphDataAdapter:
    """Converts adjacency dict to GNN-compatible tensor format."""

    def __init__(self, max_nodes: int = 5000, max_edges: int = 40000):
        """Initialize graph data adapter.

        Args:
            max_nodes: Maximum number of nodes for padding (typical: 500-3500)
            max_edges: Maximum number of edges for padding (typical: 5K-30K)
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def adjacency_to_tensors(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        node_feature_dim: int = 16,
        start_pos: Optional[Tuple[float, float]] = None,
        goal_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Convert adjacency dict to GNN tensor format with start/goal context.

        Args:
            adjacency: Graph adjacency dictionary {node_pos: [(neighbor_pos, cost), ...]}
            node_feature_dim: Dimension of node features (16 for enhanced features)
            start_pos: Starting position (x, y) for context features
            goal_positions: List of goal positions for context features

        Returns:
            Tuple of:
            - node_features: [max_nodes, node_feature_dim]
            - edge_index: [2, max_edges]
            - node_mask: [max_nodes] - 1 for valid nodes, 0 for padding
            - edge_mask: [max_edges] - 1 for valid edges, 0 for padding
            - metadata: Dict with position_to_id mapping and other info
        """
        if not adjacency:
            # Return empty tensors
            return (
                torch.zeros(self.max_nodes, node_feature_dim),
                torch.zeros(2, self.max_edges, dtype=torch.long),
                torch.zeros(self.max_nodes),
                torch.zeros(self.max_edges),
                {
                    "position_to_id": {},
                    "id_to_position": {},
                    "num_nodes": 0,
                    "num_edges": 0,
                },
            )

        # Create node position list and ID mapping
        node_positions = sorted(adjacency.keys())  # Consistent ordering
        num_nodes = len(node_positions)

        position_to_id = {pos: idx for idx, pos in enumerate(node_positions)}
        id_to_position = {idx: pos for idx, pos in enumerate(node_positions)}

        # Extract node features with start/goal context
        node_features = self._extract_node_features(
            node_positions,
            adjacency,
            node_feature_dim,
            start_pos=start_pos,
            goal_positions=goal_positions,
        )

        # Extract edges
        edge_list = []
        for src_pos, neighbors in adjacency.items():
            src_id = position_to_id[src_pos]
            for neighbor_pos, cost in neighbors:
                if neighbor_pos in position_to_id:
                    tgt_id = position_to_id[neighbor_pos]
                    edge_list.append((src_id, tgt_id))

        num_edges = len(edge_list)

        # Create padded tensors
        node_features_padded = torch.zeros(self.max_nodes, node_feature_dim)
        node_features_padded[:num_nodes] = torch.from_numpy(node_features).float()

        edge_index_padded = torch.zeros(2, self.max_edges, dtype=torch.long)
        if num_edges > 0:
            edge_array = np.array(edge_list, dtype=np.int64).T  # [2, num_edges]
            edge_index_padded[:, :num_edges] = torch.from_numpy(edge_array).long()

        node_mask = torch.zeros(self.max_nodes)
        node_mask[:num_nodes] = 1.0

        edge_mask = torch.zeros(self.max_edges)
        edge_mask[:num_edges] = 1.0

        metadata = {
            "position_to_id": position_to_id,
            "id_to_position": id_to_position,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }

        return node_features_padded, edge_index_padded, node_mask, edge_mask, metadata

    def _extract_node_features(
        self,
        node_positions: List[Tuple[int, int]],
        adjacency: Dict,
        feature_dim: int,
        start_pos: Optional[Tuple[float, float]] = None,
        goal_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Extract feature vector for each node with start/goal context.

        Feature layout (up to 16 dimensions, fills available slots):
        - Features 0-4: Base topology features (always filled)
        - Features 5-6: Distance to start/goals (normalized) (if feature_dim > 6)
        - Feature 7: Angle from start towards goal (if feature_dim > 7)
        - Features 8-11: Binary indicators (is_start, is_goal, is_near_start, is_near_goal) (if feature_dim > 11)
        - Features 12-15: Reserved for future features (if feature_dim > 15)

        Args:
            node_positions: List of node positions
            adjacency: Adjacency dictionary
            feature_dim: Target feature dimension (8 or 16, fills available slots)
            start_pos: Starting position for path (x, y)
            goal_positions: List of goal positions

        Returns:
            Node features array [num_nodes, feature_dim]
        """
        num_nodes = len(node_positions)
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        # Normalization constants (N++ level dimensions)
        LEVEL_WIDTH = 1056.0
        LEVEL_HEIGHT = 600.0
        LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
        NEAR_THRESHOLD = 100.0  # Threshold for "near" indicators

        for idx, pos in enumerate(node_positions):
            x, y = pos

            # Features 0-1: Normalized position
            features[idx, 0] = x / LEVEL_WIDTH
            features[idx, 1] = y / LEVEL_HEIGHT

            # Feature 2: Node degree (number of neighbors)
            degree = len(adjacency.get(pos, []))
            features[idx, 2] = min(degree / 8.0, 1.0)  # Normalize by max connectivity

            # Feature 3: Distance from center
            center_x, center_y = LEVEL_WIDTH / 2, LEVEL_HEIGHT / 2
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            features[idx, 3] = dist_from_center / max_dist

            # Feature 4: Distance from edges (boundary proximity)
            dist_to_edge = min(x, LEVEL_WIDTH - x, y, LEVEL_HEIGHT - y)
            features[idx, 4] = min(dist_to_edge / 200.0, 1.0)

            # Feature 5: Distance to start (normalized by level diagonal)
            if start_pos is not None:
                start_x, start_y = start_pos
                dist_to_start = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
                features[idx, 5] = min(dist_to_start / LEVEL_DIAGONAL, 1.0)
            else:
                features[idx, 5] = 0.0

            # Feature 6: Distance to nearest goal (normalized by level diagonal)
            if goal_positions is not None and len(goal_positions) > 0:
                min_goal_dist = min(
                    np.sqrt((x - gx) ** 2 + (y - gy) ** 2) for gx, gy in goal_positions
                )
                features[idx, 6] = min(min_goal_dist / LEVEL_DIAGONAL, 1.0)
            else:
                features[idx, 6] = 0.0

            # Feature 7: Angle from start towards goal
            if (
                start_pos is not None
                and goal_positions is not None
                and len(goal_positions) > 0
            ):
                start_x, start_y = start_pos
                # Use first goal for angle calculation
                goal_x, goal_y = goal_positions[0]

                # Vector from start to goal
                start_to_goal_x = goal_x - start_x
                start_to_goal_y = goal_y - start_y

                # Vector from start to current node
                start_to_node_x = x - start_x
                start_to_node_y = y - start_y

                # Compute cosine of angle
                dot_product = (
                    start_to_goal_x * start_to_node_x
                    + start_to_goal_y * start_to_node_y
                )
                mag_goal = np.sqrt(start_to_goal_x**2 + start_to_goal_y**2)
                mag_node = np.sqrt(start_to_node_x**2 + start_to_node_y**2)

                if mag_goal > 0 and mag_node > 0:
                    cos_angle = dot_product / (mag_goal * mag_node)
                    features[idx, 7] = np.clip(cos_angle, -1.0, 1.0)
                else:
                    features[idx, 7] = 0.0
            else:
                features[idx, 7] = 0.0

            # Feature 8: is_start indicator (only if feature_dim > 8)
            if feature_dim > 8:
                if start_pos is not None:
                    start_x, start_y = start_pos
                    dist_to_start = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
                    features[idx, 8] = 1.0 if dist_to_start < 10.0 else 0.0
                else:
                    features[idx, 8] = 0.0

            # Feature 9: is_goal indicator (only if feature_dim > 9)
            if feature_dim > 9:
                if goal_positions is not None and len(goal_positions) > 0:
                    is_goal = any(
                        np.sqrt((x - gx) ** 2 + (y - gy) ** 2) < 10.0
                        for gx, gy in goal_positions
                    )
                    features[idx, 9] = 1.0 if is_goal else 0.0
                else:
                    features[idx, 9] = 0.0

            # Feature 10: is_near_start indicator (only if feature_dim > 10)
            if feature_dim > 10:
                if start_pos is not None:
                    start_x, start_y = start_pos
                    dist_to_start = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
                    features[idx, 10] = 1.0 if dist_to_start < NEAR_THRESHOLD else 0.0
                else:
                    features[idx, 10] = 0.0

            # Feature 11: is_near_goal indicator (only if feature_dim > 11)
            if feature_dim > 11:
                if goal_positions is not None and len(goal_positions) > 0:
                    is_near_goal = any(
                        np.sqrt((x - gx) ** 2 + (y - gy) ** 2) < NEAR_THRESHOLD
                        for gx, gy in goal_positions
                    )
                    features[idx, 11] = 1.0 if is_near_goal else 0.0
                else:
                    features[idx, 11] = 0.0

            # Features 12-15: Reserved for future use
            # Could add: physics-aware features, temporal features, etc.

        return features

    def waypoints_to_node_ids(
        self,
        waypoints: List[Tuple[int, int]],
        position_to_id: Dict[Tuple[int, int], int],
        adjacency: Dict,
    ) -> List[int]:
        """Convert waypoint positions to node IDs.

        Args:
            waypoints: List of (x, y) positions
            position_to_id: Mapping from position to node ID
            adjacency: Adjacency dict for finding nearest nodes

        Returns:
            List of node IDs
        """
        node_ids = []
        node_positions = list(position_to_id.keys())

        for waypoint in waypoints:
            if waypoint in position_to_id:
                node_ids.append(position_to_id[waypoint])
            else:
                # Find nearest node
                nearest_pos = self._find_nearest_node(waypoint, node_positions)
                if nearest_pos in position_to_id:
                    node_ids.append(position_to_id[nearest_pos])
                    logger.debug(
                        f"Snapped waypoint {waypoint} to nearest node {nearest_pos}"
                    )

        return node_ids

    def _find_nearest_node(
        self,
        position: Tuple[int, int],
        node_positions: List[Tuple[int, int]],
        max_distance: float = 50.0,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest graph node to given position.

        Args:
            position: Query position (x, y)
            node_positions: List of graph node positions
            max_distance: Maximum distance to consider

        Returns:
            Nearest node position or None if none within max_distance
        """
        if not node_positions:
            return None

        x, y = position
        min_dist = float("inf")
        nearest = None

        for node_pos in node_positions:
            nx, ny = node_pos
            dist = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = node_pos

        return nearest

    def batch_adjacencies(
        self,
        adjacencies: List[Dict],
        node_feature_dim: int = 16,
        start_positions: Optional[List[Tuple[float, float]]] = None,
        goal_positions_list: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """Batch multiple adjacency dicts into single tensors with start/goal context.

        Args:
            adjacencies: List of adjacency dictionaries
            node_feature_dim: Feature dimension per node (16 for enhanced features)
            start_positions: List of start positions (one per graph)
            goal_positions_list: List of goal position lists (one list per graph)

        Returns:
            Tuple of:
            - node_features: [batch, max_nodes, node_feature_dim]
            - edge_index: [batch, 2, max_edges]
            - node_mask: [batch, max_nodes]
            - edge_mask: [batch, max_edges]
            - metadata_list: List of metadata dicts (one per graph)
        """
        batch_size = len(adjacencies)

        node_features_batch = torch.zeros(batch_size, self.max_nodes, node_feature_dim)
        edge_index_batch = torch.zeros(batch_size, 2, self.max_edges, dtype=torch.long)
        node_mask_batch = torch.zeros(batch_size, self.max_nodes)
        edge_mask_batch = torch.zeros(batch_size, self.max_edges)
        metadata_list = []

        for i, adjacency in enumerate(adjacencies):
            start_pos = start_positions[i] if start_positions is not None else None
            goal_positions = (
                goal_positions_list[i] if goal_positions_list is not None else None
            )

            node_feats, edge_idx, node_mask, edge_mask, metadata = (
                self.adjacency_to_tensors(
                    adjacency,
                    node_feature_dim,
                    start_pos=start_pos,
                    goal_positions=goal_positions,
                )
            )

            node_features_batch[i] = node_feats
            edge_index_batch[i] = edge_idx
            node_mask_batch[i] = node_mask
            edge_mask_batch[i] = edge_mask
            metadata_list.append(metadata)

        return (
            node_features_batch,
            edge_index_batch,
            node_mask_batch,
            edge_mask_batch,
            metadata_list,
        )

    def node_ids_to_positions(
        self,
        node_ids: torch.Tensor,
        id_to_position: Dict[int, Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Convert node IDs back to positions.

        Args:
            node_ids: Tensor of node IDs [num_waypoints] or [batch, num_waypoints]
            id_to_position: Mapping from node ID to position

        Returns:
            List of (x, y) positions
        """
        if node_ids.dim() == 1:
            # Single path
            positions = []
            for node_id in node_ids:
                node_id_int = int(node_id.item())
                if node_id_int in id_to_position:
                    positions.append(id_to_position[node_id_int])
            return positions
        else:
            # Batch of paths
            batch_positions = []
            for batch_item in node_ids:
                positions = []
                for node_id in batch_item:
                    node_id_int = int(node_id.item())
                    if node_id_int in id_to_position:
                        positions.append(id_to_position[node_id_int])
                batch_positions.append(positions)
            return batch_positions
