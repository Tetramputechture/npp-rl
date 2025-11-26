"""PyTorch Dataset for loading and processing N++ replay files.

This module provides a dataset loader that:
1. Loads binary CompactReplay files from expert demonstrations
2. Uses ReplayExecutor to regenerate observations deterministically
3. Extracts trajectory waypoints with graph-based validation
4. Provides features for training the path predictor network

NOTE: All replays are processed during initialization (not on-demand) for efficiency.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from tqdm import tqdm

from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import ReplayExecutor
from nclone.graph.reachability.pathfinding_utils import NODE_WORLD_COORD_OFFSET
from npp_rl.path_prediction.graph_utils import (
    snap_position_to_graph_node,
    validate_path_on_graph,
)
from npp_rl.path_prediction.graph_data_adapter import GraphDataAdapter

logger = logging.getLogger(__name__)


class PathReplayDataset(Dataset):
    """Dataset for loading N++ replay demonstrations with graph integration.

    This dataset:
    - Loads CompactReplay binary files (.replay format)
    - Executes replays using ReplayExecutor (which builds graphs internally)
    - Extracts waypoints from trajectories at regular intervals
    - Snaps waypoints to graph nodes (12px sub-tile resolution)
    - Normalizes waypoints to [0, 1] range for stable training
    - Validates paths using adjacency graph
    - Returns features for path predictor training

    All replays are processed during initialization for efficiency.
    """

    # Standard N++ level dimensions (tiles × pixels_per_tile)
    LEVEL_WIDTH = 44 * 24  # 1056 pixels
    LEVEL_HEIGHT = 25 * 24  # 600 pixels

    def __init__(
        self,
        replay_dir: str,
        waypoint_interval: int = 5,
        min_trajectory_length: int = 20,
        enable_rendering: bool = False,
        max_replays: Optional[int] = None,
        node_feature_dim: int = 8,
        max_nodes: int = 5000,
        max_edges: int = 40000,
        enable_augmentation: bool = True,
    ):
        """Initialize replay dataset.

        Args:
            replay_dir: Directory containing .replay files
            waypoint_interval: Extract waypoint every N frames
            min_trajectory_length: Minimum frames required for valid trajectory
            enable_rendering: Enable rendering in ReplayExecutor (slower)
            max_replays: Maximum number of replays to load (for debugging)
            node_feature_dim: Dimension of node features for graph adapter
            max_nodes: Maximum nodes for padding
            max_edges: Maximum edges for padding
            enable_augmentation: Enable data augmentation (horizontal flip, jitter, etc.)
        """
        self.replay_dir = Path(replay_dir)
        self.waypoint_interval = waypoint_interval
        self.min_trajectory_length = min_trajectory_length
        self.enable_rendering = enable_rendering
        self.node_feature_dim = node_feature_dim
        self.enable_augmentation = enable_augmentation

        # Graph data adapter for converting adjacency to GNN format
        self.graph_adapter = GraphDataAdapter(max_nodes=max_nodes, max_edges=max_edges)

        # Augmentation statistics
        self.aug_stats = {
            "total_samples": 0,
            "horizontal_flips": 0,
            "jitter_applied": 0,
            "goal_dropout": 0,
        }

        # Find all replay files
        self.replay_files = sorted(self.replay_dir.glob("*.replay"))

        if max_replays is not None:
            self.replay_files = self.replay_files[:max_replays]

        if len(self.replay_files) == 0:
            raise ValueError(f"No .replay files found in {replay_dir}")

        logger.info(f"Found {len(self.replay_files)} replay files in {replay_dir}")

        # Storage for processed samples
        self.samples: List[Dict[str, Any]] = []

        # Process all replays during initialization
        logger.info("Processing replays (this may take a while)...")
        self._process_all_replays()

        logger.info(
            f"Loaded {len(self.samples)} valid samples from {len(self.replay_files)} replays"
        )

    def _process_all_replays(self) -> None:
        """Process all replay files and extract samples."""
        executor = ReplayExecutor(enable_rendering=self.enable_rendering)

        for i, replay_path in enumerate(
            tqdm(self.replay_files, desc="Processing replays")
        ):
            try:
                logger.debug(
                    f"Processing replay {i + 1}/{len(self.replay_files)}: {replay_path.name}"
                )
                sample = self._process_single_replay(replay_path, executor)
                if sample and sample["trajectory_length"] > 0:
                    self.samples.append(sample)
                    logger.debug(
                        f"  ✓ Added sample with {len(sample['expert_waypoints'])} waypoints"
                    )
                else:
                    logger.debug("  ✗ Sample rejected (too short or invalid)")
            except KeyboardInterrupt:
                logger.warning("Processing interrupted by user")
                break
            except Exception as e:
                logger.warning(f"Failed to process {replay_path.name}: {e}")
                import traceback

                logger.debug(traceback.format_exc())
                continue

        try:
            executor.close()
        except Exception as e:
            logger.warning(f"Error closing executor: {e}")

    def _process_single_replay(
        self, replay_path: Path, executor: ReplayExecutor
    ) -> Optional[Dict[str, Any]]:
        """Process a single replay file.

        Args:
            replay_path: Path to replay file
            executor: ReplayExecutor instance

        Returns:
            Dictionary with processed sample or None if failed
        """
        # Load replay
        logger.debug("  Loading replay file...")
        with open(replay_path, "rb") as f:
            replay_data = f.read()

        logger.debug(f"  Parsing replay binary ({len(replay_data)} bytes)...")
        replay = CompactReplay.from_binary(replay_data, episode_id=replay_path.stem)
        logger.debug(f"  Replay has {len(replay.input_sequence)} input frames")

        # Execute replay to generate observations
        try:
            logger.debug("  Executing replay...")
            observations = executor.execute_replay(
                map_data=replay.map_data,
                input_sequence=replay.input_sequence,
            )
            logger.info(
                f"  Execution complete: {len(observations)} observations generated"
            )
        except Exception as e:
            logger.error(f"Replay execution failed for {replay.episode_id}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

        if not observations:
            logger.debug("  No observations generated")
            return None

        # Extract waypoints from trajectory
        logger.debug("  Extracting waypoints...")

        # DEFENSIVE: Log first observation player position
        if len(observations) > 0:
            first_obs = observations[0].get("observation", {})
            logger.info(
                f"  First observation player_x={first_obs.get('player_x', 'N/A')}, player_y={first_obs.get('player_y', 'N/A')}"
            )

        waypoints = self._extract_waypoints(observations)
        logger.info(
            f"  Extracted {len(waypoints)} raw waypoints, first 3: {waypoints[:3]}"
        )

        # Log first few waypoints for debugging
        if waypoints:
            logger.debug(f"  First 5 raw waypoints: {waypoints[:5]}")
            waypoint_xs = [w[0] for w in waypoints]
            waypoint_ys = [w[1] for w in waypoints]
            logger.debug(
                f"  Waypoint X range: [{min(waypoint_xs)}, {max(waypoint_xs)}]"
            )
            logger.debug(
                f"  Waypoint Y range: [{min(waypoint_ys)}, {max(waypoint_ys)}]"
            )

        if len(waypoints) < self.min_trajectory_length // self.waypoint_interval:
            logger.debug(f"  Trajectory too short: {len(waypoints)} waypoints")
            return None

        # Extract graph data from executor
        logger.debug("  Extracting graph data...")
        graph_data = executor._cached_graph_data
        if graph_data is None:
            logger.debug(f"  No graph data for replay {replay.episode_id}")
            return None

        adjacency = graph_data.get("adjacency", {})
        spatial_hash = graph_data.get("spatial_hash")

        if not adjacency or spatial_hash is None:
            logger.debug(f"  Invalid graph data for replay {replay.episode_id}")
            return None

        logger.debug(f"  Graph has {len(adjacency)} nodes")

        # Snap waypoints to graph nodes
        logger.debug("  Snapping waypoints to graph nodes...")
        logger.debug(f"  Graph has {len(adjacency)} nodes")

        # Sample a few graph node coordinates for debugging
        sample_nodes = list(adjacency.keys())[:5]
        logger.debug(f"  Sample graph nodes (first 5): {sample_nodes}")

        snapped_waypoints = []
        num_snapped = 0
        num_kept_original = 0

        for waypoint in waypoints:
            node = snap_position_to_graph_node(
                waypoint,
                adjacency,
                spatial_hash,
                threshold=24,  # One tile distance
            )
            if node is not None:
                snapped_waypoints.append(node)
                num_snapped += 1
            else:
                # Keep original if no graph node nearby
                snapped_waypoints.append(tuple(waypoint))
                num_kept_original += 1

        logger.info(f"  Snapped {len(snapped_waypoints)} waypoints total")
        logger.info(
            f"    - {num_snapped} waypoints successfully snapped to graph nodes"
        )
        logger.info(
            f"    - {num_kept_original} waypoints kept as original (no nearby node)"
        )

        # Log first few snapped waypoints
        if snapped_waypoints:
            logger.info(f"  First 5 snapped waypoints: {snapped_waypoints[:5]}")

            # DEFENSIVE: Check for duplicate waypoints after snapping
            unique_waypoints = set(snapped_waypoints)
            num_unique = len(unique_waypoints)
            num_total = len(snapped_waypoints)
            duplicate_ratio = 1.0 - (num_unique / num_total) if num_total > 0 else 0.0
            logger.info(
                f"  Unique waypoints: {num_unique}/{num_total} (duplicate ratio: {duplicate_ratio:.2%})"
            )

            if duplicate_ratio > 0.9:
                logger.warning(
                    f"  Very high duplicate ratio ({duplicate_ratio:.2%}) - most waypoints collapsed to same nodes!"
                )

        # Normalize waypoints to [0, 1] range for stable training
        logger.debug("  Normalizing waypoints to [0, 1] range...")
        normalized_waypoints = [
            (float(x) / self.LEVEL_WIDTH, float(y) / self.LEVEL_HEIGHT)
            for x, y in snapped_waypoints
        ]
        logger.info(
            f"  Normalized {len(normalized_waypoints)} waypoints to [0,1] range"
        )

        # Validate normalized coordinates
        if normalized_waypoints:
            from ..path_prediction.coordinate_utils import validate_normalized_coords

            norm_xs = [w[0] for w in normalized_waypoints]
            norm_ys = [w[1] for w in normalized_waypoints]
            logger.debug(
                f"  Normalized X range: [{min(norm_xs):.3f}, {max(norm_xs):.3f}]"
            )
            logger.debug(
                f"  Normalized Y range: [{min(norm_ys):.3f}, {max(norm_ys):.3f}]"
            )

            # Validate
            is_valid, msg = validate_normalized_coords(
                normalized_waypoints, tolerance=0.1
            )
            if not is_valid:
                logger.warning(f"  Coordinate validation failed: {msg}")
            else:
                logger.debug(f"  Coordinate validation: {msg}")

        # Validate path on graph
        logger.debug("  Validating path on graph...")
        if len(snapped_waypoints) >= 2:
            is_valid, graph_dist = validate_path_on_graph(
                snapped_waypoints, adjacency, spatial_hash
            )

            if not is_valid:
                logger.debug("  Path validation failed")

        # Extract features from final observation (has complete graph data)
        logger.debug("  Extracting features from observations...")
        if len(observations) > 0:
            final_obs = observations[-1]["observation"]

            graph_obs = self._extract_graph_features(final_obs)
            tile_patterns = self._extract_tile_patterns(final_obs)
            entity_features = self._extract_entity_features(final_obs)
        else:
            graph_obs = None
            tile_patterns = torch.zeros(64, dtype=torch.float32)
            entity_features = torch.zeros(32, dtype=torch.float32)

        logger.debug("  Features extracted successfully")

        # Convert normalized waypoints to tensor
        waypoints_tensor = torch.tensor(normalized_waypoints, dtype=torch.float32)

        # Convert adjacency to GNN tensor format
        logger.debug("  Converting adjacency to GNN format...")
        node_features, edge_index, node_mask, edge_mask, metadata = (
            self.graph_adapter.adjacency_to_tensors(adjacency, self.node_feature_dim)
        )
        logger.debug(
            f"  Graph tensors: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges"
        )

        # Convert expert waypoints to node IDs
        logger.debug("  Converting expert waypoints to node IDs...")
        expert_node_ids = self.graph_adapter.waypoints_to_node_ids(
            snapped_waypoints, metadata["position_to_id"], adjacency
        )
        logger.debug(f"  Converted {len(expert_node_ids)} waypoints to node IDs")

        # Create mask for valid expert waypoints
        max_waypoints = 20  # Match model's max_waypoints
        expert_node_ids_padded = expert_node_ids[:max_waypoints] + [0] * (
            max_waypoints - len(expert_node_ids)
        )
        expert_node_mask = [1.0] * len(expert_node_ids[:max_waypoints]) + [0.0] * (
            max_waypoints - len(expert_node_ids)
        )

        expert_node_ids_tensor = torch.tensor(expert_node_ids_padded, dtype=torch.long)
        expert_node_mask_tensor = torch.tensor(expert_node_mask, dtype=torch.float32)

        # Extract start position and convert to node ID for multimodal fusion
        logger.debug("  Extracting start position and ninja state...")
        start_pos = None
        ninja_state = None
        if len(observations) > 0:
            first_obs = observations[0]["observation"]
            player_x = first_obs.get("player_x", 0)
            player_y = first_obs.get("player_y", 0)
            # Convert from world coords to tile data coords
            start_pos = (
                int(player_x) - NODE_WORLD_COORD_OFFSET,
                int(player_y) - NODE_WORLD_COORD_OFFSET,
            )
            logger.debug(f"  Start position (tile coords): {start_pos}")

            # Extract full ninja state (40 dimensions) for physics context
            # Get game state channels from observation
            ninja_state = self._extract_ninja_state(first_obs)
            logger.debug(
                f"  Extracted ninja state: {ninja_state.shape if ninja_state is not None else 'None'}"
            )

        # Convert start position to node ID
        start_node_id = None
        if start_pos is not None:
            from nclone.graph.reachability.pathfinding_utils import (
                find_closest_node_to_position,
            )

            start_node = find_closest_node_to_position(
                start_pos, adjacency, threshold=50.0
            )
            if start_node is not None and start_node in metadata["position_to_id"]:
                start_node_id = metadata["position_to_id"][start_node]
                logger.debug(
                    f"  Start node ID: {start_node_id} (position: {start_node})"
                )

        # Extract goal positions (exit + switches) for multimodal fusion
        logger.debug("  Extracting goal positions...")
        goal_positions = []
        goal_node_ids = []

        # Get entities from executor's cached graph data
        if graph_data and "entities" in graph_data:
            entities = graph_data["entities"]
            from nclone.graph.reachability.pathfinding_utils import (
                find_closest_node_to_position,
            )

            # Find exit and switches (goal objects)
            for entity in entities:
                entity_type = entity.get("type", "")
                if entity_type in ["exit", "switch"]:
                    # Entity positions are in tile data coordinates
                    entity_pos = (int(entity["x"]), int(entity["y"]))
                    goal_positions.append(entity_pos)

                    # Convert to node ID
                    goal_node = find_closest_node_to_position(
                        entity_pos, adjacency, threshold=50.0
                    )
                    if (
                        goal_node is not None
                        and goal_node in metadata["position_to_id"]
                    ):
                        goal_node_ids.append(metadata["position_to_id"][goal_node])

            logger.debug(
                f"  Found {len(goal_positions)} goal positions: {goal_positions[:3]}"
            )
            logger.debug(f"  Converted to {len(goal_node_ids)} goal node IDs")

        # If no goals found, use last waypoint as goal (fallback)
        if len(goal_positions) == 0 and len(snapped_waypoints) > 0:
            goal_pos = snapped_waypoints[-1]
            goal_positions = [goal_pos]
            if goal_pos in metadata["position_to_id"]:
                goal_node_ids = [metadata["position_to_id"][goal_pos]]
            logger.debug(f"  No goals found, using last waypoint as goal: {goal_pos}")

        # Convert to tensors
        start_node_id_tensor = torch.tensor(
            [start_node_id if start_node_id is not None else 0], dtype=torch.long
        )
        goal_node_ids_tensor = torch.tensor(
            goal_node_ids if goal_node_ids else [0], dtype=torch.long
        )
        ninja_state_tensor = (
            ninja_state
            if ninja_state is not None
            else torch.zeros(40, dtype=torch.float32)
        )

        # Store adjacency for graph rebuilding during augmentation
        adjacency_for_sample = adjacency

        # Extract tiles and entities from cached level data (not graph_data)
        # graph_data only contains adjacency/spatial_hash, level_data has tiles/entities
        level_data = executor._cached_level_data
        if level_data is not None:
            # LevelData object has tiles and entities attributes
            tiles = level_data.tiles if hasattr(level_data, "tiles") else None
            entities = level_data.entities if hasattr(level_data, "entities") else []
        else:
            # Fallback: try to get from graph_data (shouldn't happen, but defensive)
            tiles = graph_data.get("tiles")
            entities = graph_data.get("entities", [])

        return {
            "graph_obs": graph_obs,
            "tile_patterns": tile_patterns,
            "entity_features": entity_features,
            "expert_waypoints": normalized_waypoints,  # Normalized [0,1] coordinates
            "expert_waypoints_tensor": waypoints_tensor,
            # NEW: Graph-based data
            "node_features": node_features,  # [max_nodes, node_feature_dim]
            "edge_index": edge_index,  # [2, max_edges]
            "node_mask": node_mask,  # [max_nodes]
            "edge_mask": edge_mask,  # [max_edges]
            "expert_node_ids": expert_node_ids_tensor,  # [max_waypoints]
            "expert_node_mask": expert_node_mask_tensor,  # [max_waypoints]
            # NEW: Multimodal fusion data
            "start_pos": start_pos,  # Tuple (x, y) in tile coords
            "goal_positions": goal_positions,  # List of tuples [(x, y), ...]
            "start_node_id": start_node_id_tensor,  # [1]
            "goal_node_ids": goal_node_ids_tensor,  # [num_goals]
            "ninja_state": ninja_state_tensor,  # [40] - full physics state
            # Metadata
            "position_to_id": metadata["position_to_id"],
            "id_to_position": metadata["id_to_position"],
            "adjacency": adjacency_for_sample,  # For graph rebuilding during augmentation
            "tiles": tiles,  # For graph rebuilding during augmentation
            "entities": entities,  # For graph rebuilding during augmentation
            "trajectory_length": len(observations),
            "success": replay.success,
            "replay_id": replay.episode_id,
        }

    def _extract_waypoints(
        self, observations: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """Extract waypoints from observation sequence.

        Args:
            observations: List of observation dicts with 'observation' key

        Returns:
            List of (x, y) waypoint positions in tile data coordinates
            (converted from world coordinates by subtracting NODE_WORLD_COORD_OFFSET)
        """
        waypoints = []

        for i in range(0, len(observations), self.waypoint_interval):
            obs_dict = observations[i]
            obs = obs_dict.get("observation", {})

            # Extract player position (in world coordinates)
            player_x = obs.get("player_x", 0)
            player_y = obs.get("player_y", 0)

            # CRITICAL FIX: Convert from world coordinates to tile data coordinates
            # World coordinates include 1-tile (24px) border, graph nodes don't
            # This matches the coordinate system used by graph nodes
            tile_data_x = int(player_x) - NODE_WORLD_COORD_OFFSET
            tile_data_y = int(player_y) - NODE_WORLD_COORD_OFFSET

            waypoints.append((tile_data_x, tile_data_y))

        return waypoints

    def _extract_graph_features(self, obs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract graph observation features from observation.

        Args:
            obs: Observation dictionary

        Returns:
            Graph features tensor or None
        """
        # Extract sparse graph observations if available
        if "graph_node_feats_sparse" in obs:
            node_feats = obs["graph_node_feats_sparse"]
            edge_index = obs["graph_edge_index_sparse"]
            edge_feats = obs["graph_edge_feats_sparse"]

            # Package as dictionary for path predictor
            return {
                "node_features": torch.from_numpy(node_feats).float(),
                "edge_index": torch.from_numpy(edge_index).long(),
                "edge_features": torch.from_numpy(edge_feats).float(),
                "num_nodes": obs.get("graph_num_nodes", np.array([0]))[0],
                "num_edges": obs.get("graph_num_edges", np.array([0]))[0],
            }

        return None

    def _extract_tile_patterns(self, obs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract tile pattern features from observation.

        Combines tile-based structural features with reachability data.

        Args:
            obs: Observation dictionary

        Returns:
            Tile pattern features tensor or None (always 64-dim)
        """
        from ..path_prediction.tile_feature_utils import (
            extract_tile_features_from_observation,
        )

        # Use centralized tile feature extraction for consistency
        features = extract_tile_features_from_observation(
            obs, target_dim=64, device="cpu"
        )

        return features

    def _extract_entity_features(self, obs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract entity context features from observation.

        Args:
            obs: Observation dictionary

        Returns:
            Entity features tensor (always 32-dim)
        """
        # Create fixed 32-dim feature vector
        features = torch.zeros(32, dtype=torch.float32)

        offset = 0

        # Extract locked door features if available
        if "locked_door_features" in obs:
            door_feats = obs["locked_door_features"]
            if isinstance(door_feats, np.ndarray):
                door_tensor = torch.from_numpy(door_feats.flatten()).float()
                n = min(len(door_tensor), 16)  # Use next 16 dims for doors
                features[offset : offset + n] = door_tensor[:n]

        return features

    def _extract_ninja_state(self, obs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract full ninja physics state for multimodal fusion.

        Extracts 40-dimensional game state vector that includes:
        - Player position, velocity, acceleration
        - Contact states (grounded, wall contact, ceiling)
        - Movement buffers (jump buffer, wall-slide buffer)
        - Physics properties

        Args:
            obs: Observation dictionary with 'game_state' key

        Returns:
            Ninja state tensor [40] or None if not available
        """
        from nclone.gym_environment.constants import GAME_STATE_CHANNELS

        if "game_state" in obs:
            game_state = obs["game_state"]

            # game_state is typically a numpy array of shape (GAME_STATE_CHANNELS,)
            if isinstance(game_state, np.ndarray):
                if game_state.shape[0] == GAME_STATE_CHANNELS:
                    return torch.from_numpy(game_state).float()
                else:
                    logger.warning(
                        f"Unexpected game_state shape: {game_state.shape}, "
                        f"expected ({GAME_STATE_CHANNELS},)"
                    )

        # Fallback: create zero state if not available
        logger.debug("game_state not found in observation, returning zero state")
        return torch.zeros(40, dtype=torch.float32)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample with optional augmentation.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data (potentially augmented)
        """
        sample = self.samples[idx].copy()

        if self.enable_augmentation:
            sample = self._augment_sample(sample)

        return sample

    def _augment_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to a sample.

        Augmentation techniques:
        1. Horizontal flip (50% prob) - mirror all coordinates
        2. Waypoint jitter (Gaussian noise)
        3. Goal dropout (10% prob) - randomly remove one goal

        Args:
            sample: Original sample dictionary

        Returns:
            Augmented sample dictionary
        """
        import random

        self.aug_stats["total_samples"] += 1

        # 1. Horizontal Flip (50% probability)
        if random.random() < 0.5:
            sample = self._apply_horizontal_flip(sample)
            self.aug_stats["horizontal_flips"] += 1

        # 2. Waypoint Jitter (add small Gaussian noise)
        if random.random() < 0.7:  # 70% probability
            sample = self._apply_waypoint_jitter(sample)
            self.aug_stats["jitter_applied"] += 1

        # 3. Goal Dropout (10% probability, only if multiple goals)
        if len(sample.get("goal_positions", [])) > 1 and random.random() < 0.1:
            sample = self._apply_goal_dropout(sample)
            self.aug_stats["goal_dropout"] += 1

        return sample

    def _apply_horizontal_flip(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply horizontal flip by rebuilding graph from flipped map data.

        GraphBuilder uses precomputed lookup tables (SUBNODE_VALIDITY_TABLE,
        WITHIN_TILE_CONNECTIVITY) making graph building very fast (<0.2ms).
        We rebuild the graph from flipped map to get correct node features.

        Args:
            sample: Original sample

        Returns:
            Horizontally flipped sample with rebuilt graph
        """
        from nclone.graph.reachability.graph_builder import GraphBuilder
        from nclone.graph.reachability.pathfinding_utils import (
            find_closest_node_to_position,
        )

        # Get tiles and entities for graph rebuilding
        tiles = sample.get("tiles")
        entities = sample.get("entities", [])

        if tiles is None:
            # Fallback: flip coordinates only without graph rebuild
            logger.warning(
                "No tiles data for graph rebuilding, using coordinate flip only"
            )
            return self._apply_horizontal_flip_coords_only(sample)

        # Flip tiles horizontally (numpy fliplr is very fast)
        flipped_tiles = np.fliplr(tiles)

        # Flip entity X coordinates
        flipped_entities = []
        for entity in entities:
            flipped_entity = entity.copy()
            if "x" in entity:
                flipped_entity["x"] = self.LEVEL_WIDTH - entity["x"]
            if "switch_x" in entity:  # Exit doors with switches
                flipped_entity["switch_x"] = self.LEVEL_WIDTH - entity["switch_x"]
            flipped_entities.append(flipped_entity)

        # Flip start position (tile coords)
        flipped_start_pos = None
        if sample.get("start_pos"):
            x, y = sample["start_pos"]
            flipped_start_pos = (self.LEVEL_WIDTH - x, y)

        # Flip goal positions (tile coords)
        flipped_goal_positions = []
        if sample.get("goal_positions"):
            flipped_goal_positions = [
                (self.LEVEL_WIDTH - gx, gy) for gx, gy in sample["goal_positions"]
            ]

        # Rebuild graph from flipped map (fast with lookup tables)
        graph_builder = GraphBuilder()
        level_data = {
            "tiles": flipped_tiles,
            "entities": flipped_entities,
        }

        graph_result = graph_builder.build_graph(
            level_data=level_data,
            ninja_pos=flipped_start_pos,
            filter_by_reachability=True,
        )

        flipped_adjacency = graph_result.get("adjacency", {})

        if not flipped_adjacency:
            logger.warning("Failed to build flipped graph, using coord flip only")
            return self._apply_horizontal_flip_coords_only(sample)

        # Convert adjacency to GNN tensors with graph adapter
        (
            node_features,
            edge_index,
            node_mask,
            edge_mask,
            metadata,
        ) = self.graph_adapter.adjacency_to_tensors(
            flipped_adjacency,
            self.node_feature_dim,
            start_pos=flipped_start_pos,
            goal_positions=flipped_goal_positions,
        )

        # Flip waypoints (tile coords -> normalized [0,1])
        flipped_waypoints_normalized = [
            (1.0 - x, y) for x, y in sample["expert_waypoints"]
        ]

        # Convert normalized waypoints back to tile coords for node ID conversion
        flipped_waypoints_tile_coords = [
            (int(x * self.LEVEL_WIDTH), int(y * self.LEVEL_HEIGHT))
            for x, y in flipped_waypoints_normalized
        ]

        # Convert waypoints to node IDs in flipped graph
        expert_node_ids = self.graph_adapter.waypoints_to_node_ids(
            flipped_waypoints_tile_coords,
            metadata["position_to_id"],
            flipped_adjacency,
        )

        # Pad/truncate to max_waypoints
        max_waypoints = 20
        expert_node_ids_padded = expert_node_ids[:max_waypoints] + [0] * (
            max_waypoints - len(expert_node_ids)
        )
        expert_node_mask = [1.0] * min(len(expert_node_ids), max_waypoints) + [0.0] * (
            max_waypoints - min(len(expert_node_ids), max_waypoints)
        )

        # Get start/goal node IDs in flipped graph
        start_node_id = torch.tensor([0], dtype=torch.long)
        if flipped_start_pos:
            start_node = find_closest_node_to_position(
                flipped_start_pos, flipped_adjacency, threshold=50.0
            )
            if start_node and start_node in metadata["position_to_id"]:
                start_node_id = torch.tensor(
                    [metadata["position_to_id"][start_node]], dtype=torch.long
                )

        goal_node_ids = torch.tensor([0], dtype=torch.long)
        if flipped_goal_positions:
            goal_ids = []
            for gp in flipped_goal_positions:
                goal_node = find_closest_node_to_position(
                    gp, flipped_adjacency, threshold=50.0
                )
                if goal_node and goal_node in metadata["position_to_id"]:
                    goal_ids.append(metadata["position_to_id"][goal_node])
            if goal_ids:
                goal_node_ids = torch.tensor(goal_ids, dtype=torch.long)

        # Update sample with flipped data
        flipped_sample = sample.copy()
        flipped_sample.update(
            {
                "tiles": flipped_tiles,
                "entities": flipped_entities,
                "start_pos": flipped_start_pos,
                "goal_positions": flipped_goal_positions,
                "expert_waypoints": flipped_waypoints_normalized,
                "expert_waypoints_tensor": torch.tensor(
                    flipped_waypoints_normalized, dtype=torch.float32
                ),
                "adjacency": flipped_adjacency,
                "position_to_id": metadata["position_to_id"],
                "id_to_position": metadata["id_to_position"],
                "node_features": node_features,
                "edge_index": edge_index,
                "node_mask": node_mask,
                "edge_mask": edge_mask,
                "expert_node_ids": torch.tensor(
                    expert_node_ids_padded, dtype=torch.long
                ),
                "expert_node_mask": torch.tensor(expert_node_mask, dtype=torch.float32),
                "start_node_id": start_node_id,
                "goal_node_ids": goal_node_ids,
            }
        )

        return flipped_sample

    def _apply_horizontal_flip_coords_only(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback: flip coordinates only without rebuilding graph.

        Used when tiles data is not available.
        """
        # Flip normalized waypoints (in [0, 1] range)
        if "expert_waypoints" in sample:
            flipped_waypoints = [(1.0 - x, y) for x, y in sample["expert_waypoints"]]
            sample["expert_waypoints"] = flipped_waypoints
            sample["expert_waypoints_tensor"] = torch.tensor(
                flipped_waypoints, dtype=torch.float32
            )

        # Flip start position (in pixel coordinates)
        if "start_pos" in sample and sample["start_pos"] is not None:
            x, y = sample["start_pos"]
            sample["start_pos"] = (self.LEVEL_WIDTH - x, y)

        # Flip goal positions (in pixel coordinates)
        if "goal_positions" in sample:
            flipped_goals = [
                (self.LEVEL_WIDTH - x, y) for x, y in sample["goal_positions"]
            ]
            sample["goal_positions"] = flipped_goals

        return sample

    def _apply_waypoint_jitter(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add small Gaussian noise to waypoint positions.

        Adds noise with std of 2-5 pixels (normalized to [0, 1] range).
        Helps model become robust to small position variations.

        Args:
            sample: Original sample

        Returns:
            Sample with jittered waypoints
        """
        import torch
        import numpy as np

        if "expert_waypoints" in sample:
            # Jitter in normalized coordinates
            jitter_std_x = 3.0 / self.LEVEL_WIDTH  # 3 pixels normalized
            jitter_std_y = 3.0 / self.LEVEL_HEIGHT  # 3 pixels normalized

            jittered_waypoints = []
            for x, y in sample["expert_waypoints"]:
                # Add Gaussian noise
                jittered_x = x + np.random.normal(0, jitter_std_x)
                jittered_y = y + np.random.normal(0, jitter_std_y)

                # Clamp to [0, 1] range
                jittered_x = np.clip(jittered_x, 0.0, 1.0)
                jittered_y = np.clip(jittered_y, 0.0, 1.0)

                jittered_waypoints.append((jittered_x, jittered_y))

            sample["expert_waypoints"] = jittered_waypoints
            sample["expert_waypoints_tensor"] = torch.tensor(
                jittered_waypoints, dtype=torch.float32
            )

        return sample

    def _apply_goal_dropout(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop one goal from goal_positions list.

        Helps model not over-rely on specific goal configurations.
        Only applies if multiple goals exist.

        Args:
            sample: Original sample

        Returns:
            Sample with one goal dropped
        """
        import random
        import torch

        if "goal_positions" in sample and len(sample["goal_positions"]) > 1:
            # Randomly select which goal to keep
            goals = sample["goal_positions"].copy()
            drop_idx = random.randint(0, len(goals) - 1)
            goals.pop(drop_idx)
            sample["goal_positions"] = goals

            # Also update goal_node_ids if present
            if "goal_node_ids" in sample and len(sample["goal_node_ids"]) > 1:
                goal_ids = sample["goal_node_ids"].tolist()
                goal_ids.pop(drop_idx)
                sample["goal_node_ids"] = torch.tensor(goal_ids, dtype=torch.long)

        return sample

    def get_augmentation_stats(self) -> Dict[str, float]:
        """Get augmentation statistics.

        Returns:
            Dictionary with augmentation rates
        """
        total = max(self.aug_stats["total_samples"], 1)
        return {
            "total_samples_served": total,
            "horizontal_flip_rate": self.aug_stats["horizontal_flips"] / total,
            "jitter_rate": self.aug_stats["jitter_applied"] / total,
            "goal_dropout_rate": self.aug_stats["goal_dropout"] / total,
        }


def collate_replay_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching replay samples.

    Handles variable-length waypoint sequences and graph sizes.

    Args:
        batch: List of samples from PathReplayDataset

    Returns:
        Batched dictionary with padded sequences
    """
    # Filter out empty samples
    valid_samples = [s for s in batch if s["trajectory_length"] > 0]

    if len(valid_samples) == 0:
        # Return empty batch
        return {
            "graph_obs": None,
            "tile_patterns": torch.zeros((0, 64), dtype=torch.float32),  # Fixed 64-dim
            "entity_features": torch.zeros(
                (0, 32), dtype=torch.float32
            ),  # Fixed 32-dim
            "expert_waypoints": [],
            "expert_waypoints_tensor": torch.zeros((0, 0, 2), dtype=torch.float32),
            "trajectory_lengths": torch.zeros(0, dtype=torch.long),
            "success": torch.zeros(0, dtype=torch.bool),
            "batch_size": 0,
        }

    batch_size = len(valid_samples)

    # Stack tile patterns and entity features
    tile_patterns = torch.stack([s["tile_patterns"] for s in valid_samples])
    entity_features = torch.stack([s["entity_features"] for s in valid_samples])

    # Pad waypoint sequences to max length in batch
    max_waypoints = max(len(s["expert_waypoints"]) for s in valid_samples)
    waypoints_padded = torch.zeros((batch_size, max_waypoints, 2), dtype=torch.float32)
    waypoint_masks = torch.zeros((batch_size, max_waypoints), dtype=torch.bool)

    for i, sample in enumerate(valid_samples):
        waypoints = sample["expert_waypoints_tensor"]
        num_waypoints = len(waypoints)
        if num_waypoints > 0:
            waypoints_padded[i, :num_waypoints] = waypoints
            waypoint_masks[i, :num_waypoints] = True

    # Collect other metadata
    trajectory_lengths = torch.tensor(
        [s["trajectory_length"] for s in valid_samples], dtype=torch.long
    )
    success = torch.tensor([s["success"] for s in valid_samples], dtype=torch.bool)

    # Graph observations are kept as list (variable size)
    graph_obs_list = [s["graph_obs"] for s in valid_samples]

    # Stack graph-based tensors
    node_features = torch.stack([s["node_features"] for s in valid_samples])
    edge_index = torch.stack([s["edge_index"] for s in valid_samples])
    node_mask = torch.stack([s["node_mask"] for s in valid_samples])
    edge_mask = torch.stack([s["edge_mask"] for s in valid_samples])
    expert_node_ids = torch.stack([s["expert_node_ids"] for s in valid_samples])
    expert_node_mask = torch.stack([s["expert_node_mask"] for s in valid_samples])

    # Stack multimodal fusion tensors
    start_node_ids = torch.stack(
        [s["start_node_id"] for s in valid_samples]
    )  # [batch, 1]
    start_node_ids = start_node_ids.squeeze(1)  # [batch]

    # Handle variable number of goals per sample
    max_goals = max(len(s["goal_node_ids"]) for s in valid_samples)
    goal_node_ids_padded = torch.zeros((batch_size, max_goals), dtype=torch.long)
    for i, sample in enumerate(valid_samples):
        goals = sample["goal_node_ids"]
        num_goals = len(goals)
        if num_goals > 0:
            goal_node_ids_padded[i, :num_goals] = goals

    ninja_states = torch.stack([s["ninja_state"] for s in valid_samples])  # [batch, 40]

    return {
        "graph_obs": graph_obs_list,
        "tile_patterns": tile_patterns,
        "entity_features": entity_features,
        "expert_waypoints_tensor": waypoints_padded,
        "waypoint_masks": waypoint_masks,
        # Graph-based fields
        "node_features": node_features,
        "edge_index": edge_index,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "expert_node_ids": expert_node_ids,
        "expert_node_mask": expert_node_mask,
        # Multimodal fusion fields
        "start_node_ids": start_node_ids,  # [batch]
        "goal_node_ids": goal_node_ids_padded,  # [batch, max_goals]
        "ninja_states": ninja_states,  # [batch, 40]
        # Metadata
        "trajectory_lengths": trajectory_lengths,
        "success": success,
        "batch_size": batch_size,
    }
