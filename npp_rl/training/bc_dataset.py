"""Behavioral Cloning Dataset for N++ Replay Data.

This module provides a PyTorch Dataset for loading compact replay files
and generating training data for behavioral cloning pretraining.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset
import multiprocessing

from nclone.replay import CompactReplay
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig

logger = logging.getLogger(__name__)


class ObservationNormalizer:
    """Computes and applies normalization statistics for observations.

    Tracks running mean and standard deviation for each observation component
    to normalize inputs to zero mean and unit variance, which improves
    neural network training stability and convergence.
    """

    def __init__(self):
        """Initialize normalizer with empty statistics."""
        self.stats = {}
        self.enabled = True

    def compute_stats(self, samples: List[Tuple[Dict, int]]) -> None:
        """Compute normalization statistics from samples.

        Args:
            samples: List of (observation, action) tuples
        """
        if not samples:
            print("No samples provided for normalization statistics")
            return

        # Visual observations (player_frame, global_view) are normalized by the feature
        # extractor (division by 255.0), so we skip them here to avoid double normalization
        visual_keys = {"player_frame", "global_view"}

        # Collect all observations by key
        obs_by_key = {}
        for obs, _ in samples:
            for key, value in obs.items():
                # Skip visual observations - they're normalized by feature extractor
                if key in visual_keys:
                    continue

                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype, np.floating
                ):
                    if key not in obs_by_key:
                        obs_by_key[key] = []
                    obs_by_key[key].append(value)

        # Compute mean and std for each key
        for key, values in obs_by_key.items():
            values_array = np.array(values)
            mean = np.mean(values_array, axis=0)
            std = np.std(values_array, axis=0)
            # Avoid division by zero: use std of 1.0 where std is very small
            std = np.where(std < 1e-6, 1.0, std)

            self.stats[key] = {"mean": mean, "std": std}

        logger.info(
            f"Computed normalization statistics for {len(self.stats)} observation keys"
        )

    def normalize(self, obs: Dict) -> Dict:
        """Normalize an observation using computed statistics.

        Args:
            obs: Raw observation dictionary

        Returns:
            Normalized observation dictionary
        """
        if not self.enabled or not self.stats:
            return obs

        normalized = {}
        for key, value in obs.items():
            if key in self.stats:
                mean = self.stats[key]["mean"]
                std = self.stats[key]["std"]
                normalized[key] = (value - mean) / std
            else:
                normalized[key] = value

        return normalized

    def save_stats(self, path: Path) -> None:
        """Save normalization statistics to file.

        Args:
            path: Path to save statistics
        """
        if not self.stats:
            return

        try:
            np.savez_compressed(
                path,
                **{
                    f"{k}_{stat}": v
                    for k, stats_dict in self.stats.items()
                    for stat, v in stats_dict.items()
                },
            )
            logger.debug(f"Saved normalization statistics to {path}")
        except Exception as e:
            print(f"Failed to save normalization statistics: {e}")

    def load_stats(self, path: Path) -> bool:
        """Load normalization statistics from file.

        Args:
            path: Path to load statistics from

        Returns:
            True if loaded successfully, False otherwise
        """
        if not path.exists():
            return False

        try:
            data = np.load(path, allow_pickle=False)
            self.stats = {}

            # Reconstruct stats dictionary
            keys = set(k.rsplit("_", 1)[0] for k in data.keys())
            for key in keys:
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"
                if mean_key in data and std_key in data:
                    self.stats[key] = {"mean": data[mean_key], "std": data[std_key]}

            logger.debug(f"Loaded normalization statistics from {path}")
            return True
        except Exception as e:
            print(f"Failed to load normalization statistics: {e}")
            return False


def _process_replay_file_worker(
    replay_path: Path,
    cache_dir: Path,
    use_cache: bool,
    filter_successful_only: bool,
    architecture_config: Optional[Any],
    frame_stack_config: Dict,
) -> Tuple[Path, List[Tuple[Dict, int]]]:
    """Worker function to process a single replay file.

    This is a standalone function designed for multiprocessing. It processes
    a single replay file and returns the results along with the replay path
    for identification.

    Args:
        replay_path: Path to replay file
        cache_dir: Directory for caching processed data
        use_cache: Whether to use cached processed data
        filter_successful_only: Only include successful replays
        architecture_config: Optional architecture config to filter observations
        frame_stack_config: Frame stacking configuration dict

    Returns:
        Tuple of (replay_path, samples) where samples is a list of (observation, action) tuples
    """
    try:
        # Check cache first
        cache_name = replay_path.stem + ".npz"
        cache_path = cache_dir / cache_name

        if use_cache and cache_path.exists():
            samples = _load_from_cache_worker(cache_path)
            return (replay_path, samples)

        # Load replay
        with open(replay_path, "rb") as f:
            replay_data = f.read()

        replay = CompactReplay.from_binary(replay_data)

        # Filter by success if requested
        if filter_successful_only and not replay.success:
            logger.debug(f"Skipping unsuccessful replay: {replay_path.name}")
            return (replay_path, [])

        # Generate observations by simulating the replay
        samples = _simulate_replay_worker(
            replay, architecture_config, frame_stack_config, filter_successful_only
        )

        # Save to cache if enabled
        if use_cache and samples:
            _save_to_cache_worker(samples, cache_path)

        return (replay_path, samples)

    except Exception as e:
        print(f"Failed to process {replay_path.name}: {e}")
        return (replay_path, [])


def _simulate_replay_worker(
    replay: CompactReplay,
    architecture_config: Optional[Any],
    frame_stack_config: Dict,
    filter_successful_only: bool,
) -> List[Tuple[Dict, int]]:
    """Simulate a replay to generate observations and actions with frame stacking.

    Worker version that doesn't rely on instance state.

    Args:
        replay: CompactReplay object
        architecture_config: Optional architecture config to filter observations
        frame_stack_config: Frame stacking configuration dict
        filter_successful_only: Whether to filter by success

    Returns:
        List of (observation, action) tuples
    """
    samples = []

    # Extract frame stacking config
    enable_visual_stacking = frame_stack_config.get(
        "enable_visual_frame_stacking", False
    )
    visual_stack_size = frame_stack_config.get("visual_stack_size", 4)
    enable_state_stacking = frame_stack_config.get("enable_state_stacking", False)
    state_stack_size = frame_stack_config.get("state_stack_size", 4)
    padding_type = frame_stack_config.get("padding_type", "zero")

    # Initialize frame buffers
    player_frame_buffer = deque(
        maxlen=visual_stack_size if enable_visual_stacking else 1
    )
    game_state_buffer = deque(maxlen=state_stack_size if enable_state_stacking else 1)

    try:
        # Use ReplayExecutor to regenerate observations deterministically
        from nclone.replay.replay_executor import ReplayExecutor

        executor = ReplayExecutor()

        # Execute replay to get observations
        observations = executor.execute_replay(replay.map_data, replay.input_sequence)

        # Validate success by checking final observation
        if filter_successful_only and observations:
            try:
                raw_obs = executor._get_raw_observation()
                if "player_won" in raw_obs:
                    if not raw_obs["player_won"]:
                        logger.debug(
                            f"Skipping replay - player_won=False at final frame. "
                            f"Episode: {replay.episode_id}"
                        )
                        executor.close()
                        return []
                elif not replay.success:
                    logger.debug(
                        f"Skipping replay - success flag is False. Episode: {replay.episode_id}"
                    )
                    executor.close()
                    return []
            except AttributeError:
                if not replay.success:
                    logger.debug(
                        f"Skipping replay - success flag is False. Episode: {replay.episode_id}"
                    )
                    executor.close()
                    return []

        executor.close()

        # Process observations with frame stacking
        for obs_data in observations:
            observation = _process_observation_worker(
                obs_data["observation"], architecture_config
            )
            action = obs_data["action"]

            # Add to frame buffers
            if enable_visual_stacking:
                player_frame = observation.get("player_frame")
                if player_frame is not None:
                    player_frame_buffer.append(player_frame.copy())
                    # Pad initial frames
                    while len(player_frame_buffer) < visual_stack_size:
                        if padding_type == "repeat" and len(player_frame_buffer) > 0:
                            player_frame_buffer.appendleft(
                                player_frame_buffer[0].copy()
                            )
                        elif player_frame is not None:
                            player_frame_buffer.appendleft(np.zeros_like(player_frame))
                        else:
                            break

            if enable_state_stacking:
                game_state = observation.get("game_state")
                if game_state is not None:
                    game_state_buffer.append(game_state.copy())
                    # Pad initial states
                    while len(game_state_buffer) < state_stack_size:
                        if padding_type == "repeat" and len(game_state_buffer) > 0:
                            game_state_buffer.appendleft(game_state_buffer[0].copy())
                        else:
                            game_state_buffer.appendleft(np.zeros_like(game_state))

            # Check if buffers are ready
            visual_ready = not enable_visual_stacking or (
                len(player_frame_buffer) >= visual_stack_size
            )
            state_ready = not enable_state_stacking or (
                len(game_state_buffer) >= state_stack_size
            )

            # Only create samples once buffers are ready
            if visual_ready and state_ready:
                stacked_obs = observation.copy()

                if enable_visual_stacking and "player_frame" in observation:
                    if len(player_frame_buffer) > 0:
                        stacked_obs["player_frame"] = np.array(
                            list(player_frame_buffer)
                        )

                if enable_state_stacking and "game_state" in observation:
                    if len(game_state_buffer) > 0:
                        stacked_obs["game_state"] = np.concatenate(
                            list(game_state_buffer)
                        )

                samples.append((stacked_obs, int(action)))

    except Exception as e:
        print(f"Failed to simulate replay (episode {replay.episode_id}): {e}")
        return []

    return samples


def _process_observation_worker(obs: Dict, architecture_config: Optional[Any]) -> Dict:
    """Process observation into format suitable for caching (worker version).

    Args:
        obs: Raw observation from environment or ReplayExecutor
        architecture_config: Optional architecture config to filter observations

    Returns:
        Processed observation dictionary
    """
    processed = {}

    # Define all possible observation keys
    all_keys = {
        "player_frame": "player_frame",
        "global_view": "global_view",
        "game_state": "game_state",
        "reachability_features": "reachability_features",
        "entity_positions": "entity_positions",
        "graph_node_feats": "graph_node_feats",
        "graph_edge_index": "graph_edge_index",
        "graph_edge_feats": "graph_edge_feats",
        "graph_node_mask": "graph_node_mask",
        "graph_edge_mask": "graph_edge_mask",
        "graph_node_types": "graph_node_types",
        "graph_edge_types": "graph_edge_types",
    }

    # Filter based on architecture config if provided
    if architecture_config is not None and hasattr(architecture_config, "modalities"):
        modalities = architecture_config.modalities

        required_keys = []
        if modalities.use_player_frame:
            required_keys.append("player_frame")
        if modalities.use_global_view:
            required_keys.append("global_view")
        if modalities.use_game_state:
            required_keys.append("game_state")
        if modalities.use_reachability:
            required_keys.append("reachability_features")
        if modalities.use_graph:
            required_keys.extend(
                [
                    "graph_node_feats",
                    "graph_edge_index",
                    "graph_edge_feats",
                    "graph_node_mask",
                    "graph_edge_mask",
                    "graph_node_types",
                    "graph_edge_types",
                ]
            )
    else:
        required_keys = list(all_keys.keys())

    # Copy only required keys that are present in observation
    for key in required_keys:
        if key in obs:
            processed[key] = obs[key].copy()

    return processed


def _save_to_cache_worker(samples: List[Tuple[Dict, int]], cache_path: Path) -> None:
    """Save processed samples to cache file (worker version).

    Args:
        samples: List of (observation, action) tuples
        cache_path: Path to cache file
    """
    if not samples:
        return

    try:
        cache_data = {"num_samples": len(samples)}

        for i, (obs, action) in enumerate(samples):
            for key, value in obs.items():
                cache_key = f"obs_{i}_{key}"
                cache_data[cache_key] = value
            cache_data[f"action_{i}"] = action

        np.savez_compressed(cache_path, **cache_data)
        logger.debug(f"Cached {len(samples)} samples to {cache_path.name}")

    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")


def _load_from_cache_worker(cache_path: Path) -> List[Tuple[Dict, int]]:
    """Load processed samples from cache file (worker version).

    Args:
        cache_path: Path to cache file

    Returns:
        List of (observation, action) tuples
    """
    try:
        data = np.load(cache_path, allow_pickle=False)
        num_samples = int(data["num_samples"])

        samples = []
        for i in range(num_samples):
            obs = {}
            for key in [
                "game_state",
                "global_view",
                "reachability_features",
                "player_frame",
                "entity_positions",
                "graph_node_feats",
                "graph_edge_index",
                "graph_edge_feats",
                "graph_node_mask",
                "graph_edge_mask",
                "graph_node_types",
                "graph_edge_types",
            ]:
                cache_key = f"obs_{i}_{key}"
                if cache_key in data:
                    obs[key] = data[cache_key]

            action = int(data[f"action_{i}"])
            samples.append((obs, action))

        logger.debug(f"Loaded {len(samples)} samples from cache {cache_path.name}")
        return samples

    except Exception as e:
        print(f"Failed to load cache from {cache_path}: {e}")
        return []


class BCReplayDataset(Dataset):
    """PyTorch Dataset for behavioral cloning from N++ replay files.

    Loads compact .replay files, generates observations using the N++ simulator,
    and provides (observation, action) pairs for BC training.

    Features:
    - Loads .replay files from directory
    - Generates observations deterministically
    - Caches processed data to NPZ files for efficiency
    - Supports filtering by success/quality
    - Handles multimodal observations (vision, graph, state)
    """

    def __init__(
        self,
        replay_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        filter_successful_only: bool = True,
        max_replays: Optional[int] = None,
        architecture_config: Optional[Any] = None,
        normalize_observations: bool = True,
        frame_stack_config: Optional[Dict] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize BC replay dataset.

        Args:
            replay_dir: Directory containing .replay files
            cache_dir: Directory for caching processed data (default: replay_dir/cache)
            use_cache: Whether to use cached processed data
            filter_successful_only: Only include successful replays
            max_replays: Maximum number of replays to load (None for all)
            architecture_config: Optional architecture config to filter observations
            normalize_observations: Whether to normalize observations (recommended)
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str ('zero' or 'repeat')
            num_workers: Number of parallel workers for processing replays.
                If None, auto-detects to min(len(replay_files), 4).
                Set to 1 for sequential processing.
        """
        self.replay_dir = Path(replay_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.replay_dir / "cache"
        self.use_cache = use_cache
        self.filter_successful_only = filter_successful_only
        self.architecture_config = architecture_config
        self.normalize_observations = normalize_observations
        self.num_workers = num_workers

        # Frame stacking configuration
        self.frame_stack_config = frame_stack_config or {}
        self.enable_visual_stacking = self.frame_stack_config.get(
            "enable_visual_frame_stacking", False
        )
        self.visual_stack_size = self.frame_stack_config.get("visual_stack_size", 4)
        self.enable_state_stacking = self.frame_stack_config.get(
            "enable_state_stacking", False
        )
        self.state_stack_size = self.frame_stack_config.get("state_stack_size", 4)
        self.padding_type = self.frame_stack_config.get("padding_type", "zero")

        # Log frame stacking configuration
        if self.enable_visual_stacking or self.enable_state_stacking:
            logger.info("Frame stacking enabled in BC dataset:")
            if self.enable_visual_stacking:
                logger.info(
                    f"  Player frame: {self.visual_stack_size} frames (padding: {self.padding_type})"
                )
                logger.info("  Global view: NOT stacked (always single frame)")
            if self.enable_state_stacking:
                logger.info(
                    f"  State: {self.state_stack_size} states (padding: {self.padding_type})"
                )

        # Initialize frame buffers (will be reset for each replay)
        # Note: Only player_frame is stacked, not global_view
        self.player_frame_buffer = deque(
            maxlen=self.visual_stack_size if self.enable_visual_stacking else 1
        )
        self.game_state_buffer = deque(
            maxlen=self.state_stack_size if self.enable_state_stacking else 1
        )

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.samples: List[Tuple[Dict, int]] = []

        # Initialize normalizer
        self.normalizer = ObservationNormalizer()
        self.normalizer.enabled = normalize_observations

        # Load replay files
        replay_files = self._load_replay_files(max_replays)
        logger.info(f"Found {len(replay_files)} replay files in {self.replay_dir}")

        # Process replays
        self._process_replays(replay_files)

        logger.info(f"Loaded {len(self.samples)} training samples")

        # Compute normalization statistics if enabled
        if self.normalize_observations and len(self.samples) > 0:
            norm_stats_path = self.cache_dir / "normalization_stats.npz"
            if self.use_cache and norm_stats_path.exists():
                logger.info("Loading cached normalization statistics")
                self.normalizer.load_stats(norm_stats_path)
            else:
                logger.info("Computing normalization statistics from data")
                self.normalizer.compute_stats(self.samples)
                if self.use_cache:
                    self.normalizer.save_stats(norm_stats_path)

        # Log dataset statistics
        if len(self.samples) > 0:
            self._log_dataset_statistics()

    def _load_replay_files(self, max_replays: Optional[int]) -> List[Path]:
        """Load list of replay files from directory.

        Args:
            max_replays: Maximum number of replays to load

        Returns:
            List of replay file paths
        """
        replay_files = sorted(self.replay_dir.glob("*.replay"))

        if not replay_files:
            print(
                f"No .replay files found in {self.replay_dir}. "
                f"Please ensure replay data is available for BC pretraining."
            )

        if max_replays is not None:
            replay_files = replay_files[:max_replays]

        return replay_files

    def _get_cache_path(self, replay_path: Path) -> Path:
        """Get cache file path for a replay file.

        Args:
            replay_path: Path to replay file

        Returns:
            Path to cache file
        """
        cache_name = replay_path.stem + ".npz"
        return self.cache_dir / cache_name

    def _process_replays(self, replay_files: List[Path]) -> None:
        """Process all replay files and build dataset.

        Args:
            replay_files: List of replay file paths
        """
        if not replay_files:
            return

        # Determine number of workers
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = min(len(replay_files), 2)

        # Use sequential processing if num_workers is 1
        if num_workers == 1:
            for replay_path in replay_files:
                try:
                    cache_path = self._get_cache_path(replay_path)

                    if self.use_cache and cache_path.exists():
                        samples = self._load_from_cache(cache_path)
                    else:
                        samples = self._process_replay_file(replay_path)

                        if self.use_cache:
                            self._save_to_cache(samples, cache_path)

                    self.samples.extend(samples)

                except Exception as e:
                    print(f"Failed to process {replay_path.name}: {e}")
                    continue
        else:
            # Parallel processing with multiprocessing
            logger.info(
                f"Processing {len(replay_files)} replay files with {num_workers} workers"
            )

            try:
                with multiprocessing.Pool(
                    processes=num_workers, maxtasksperchild=1
                ) as pool:
                    # Prepare arguments for worker function
                    worker_args = [
                        (
                            replay_path,
                            self.cache_dir,
                            self.use_cache,
                            self.filter_successful_only,
                            self.architecture_config,
                            self.frame_stack_config,
                        )
                        for replay_path in replay_files
                    ]

                    # Process replays in parallel
                    results = pool.starmap(_process_replay_file_worker, worker_args)

                    # Collect results
                    for replay_path, samples in results:
                        if samples:
                            self.samples.extend(samples)

            except Exception as e:
                print(
                    f"Multiprocessing failed: {e}. Falling back to sequential processing."
                )
                # Fall back to sequential processing
                for replay_path in replay_files:
                    try:
                        cache_path = self._get_cache_path(replay_path)

                        if self.use_cache and cache_path.exists():
                            samples = self._load_from_cache(cache_path)
                        else:
                            samples = self._process_replay_file(replay_path)

                        if self.use_cache:
                            self._save_to_cache(samples, cache_path)

                        self.samples.extend(samples)

                    except Exception as e2:
                        print(f"Failed to process {replay_path.name}: {e2}")
                        continue

    def _process_replay_file(self, replay_path: Path) -> List[Tuple[Dict, int]]:
        """Process a single replay file into training samples.

        Args:
            replay_path: Path to replay file

        Returns:
            List of (observation, action) tuples
        """
        # Load replay
        with open(replay_path, "rb") as f:
            replay_data = f.read()

        replay = CompactReplay.from_binary(replay_data)

        # Filter by success if requested
        if self.filter_successful_only and not replay.success:
            logger.debug(f"Skipping unsuccessful replay: {replay_path.name}")
            return []

        # Generate observations by simulating the replay
        samples = self._simulate_replay(replay)

        return samples

    def _reset_frame_buffers(self):
        """Reset frame buffers for a new replay.

        Only player_frame is buffered - global_view is not stacked.
        """
        self.player_frame_buffer.clear()
        self.game_state_buffer.clear()

    def _add_to_visual_buffer(self, obs: Dict):
        """Add player frame to buffer with padding if needed.

        Only player_frame is stacked. Global view is NOT buffered.
        """
        if not self.enable_visual_stacking:
            return

        player_frame = obs.get("player_frame")

        if player_frame is not None:
            self.player_frame_buffer.append(player_frame.copy())

        # Pad initial frames
        while len(self.player_frame_buffer) < self.visual_stack_size:
            if self.padding_type == "repeat" and len(self.player_frame_buffer) > 0:
                self.player_frame_buffer.appendleft(self.player_frame_buffer[0].copy())
            elif player_frame is not None:
                self.player_frame_buffer.appendleft(np.zeros_like(player_frame))
            else:
                break

    def _add_to_state_buffer(self, obs: Dict):
        """Add game state to buffer with padding if needed."""
        if not self.enable_state_stacking:
            return

        game_state = obs.get("game_state")
        if game_state is None:
            return

        self.game_state_buffer.append(game_state.copy())

        # Pad initial states
        while len(self.game_state_buffer) < self.state_stack_size:
            if self.padding_type == "repeat" and len(self.game_state_buffer) > 0:
                self.game_state_buffer.appendleft(self.game_state_buffer[0].copy())
            else:
                self.game_state_buffer.appendleft(np.zeros_like(game_state))

    def _buffers_ready(self) -> bool:
        """Check if all buffers have enough frames."""
        visual_ready = not self.enable_visual_stacking or (
            len(self.player_frame_buffer) >= self.visual_stack_size
        )
        state_ready = not self.enable_state_stacking or (
            len(self.game_state_buffer) >= self.state_stack_size
        )
        return visual_ready and state_ready

    def _stack_observations(self, current_obs: Dict) -> Dict:
        """Stack buffered observations into final format.

        Only player_frame is stacked. Global view remains as single frame.

        Args:
            current_obs: Current observation dict

        Returns:
            Observation dict with stacked player frames
        """
        stacked_obs = current_obs.copy()

        if self.enable_visual_stacking:
            # Stack player frames: (stack_size, H, W, C)
            if "player_frame" in current_obs and len(self.player_frame_buffer) > 0:
                stacked_obs["player_frame"] = np.array(list(self.player_frame_buffer))
            # Global view is NOT stacked - keep as-is from current_obs

        if self.enable_state_stacking:
            # Stack game states: concatenate along first dimension
            if "game_state" in current_obs and len(self.game_state_buffer) > 0:
                stacked_obs["game_state"] = np.concatenate(list(self.game_state_buffer))

        return stacked_obs

    def _simulate_replay(self, replay: CompactReplay) -> List[Tuple[Dict, int]]:
        """Simulate a replay to generate observations and actions with frame stacking.

        Args:
            replay: CompactReplay object

        Returns:
            List of (observation, action) tuples
        """
        samples = []

        # Reset frame buffers for new replay
        self._reset_frame_buffers()

        try:
            # Use ReplayExecutor to regenerate observations deterministically
            from nclone.replay.replay_executor import ReplayExecutor

            executor = ReplayExecutor()

            # Execute replay to get observations
            observations = executor.execute_replay(
                replay.map_data, replay.input_sequence
            )

            # Validate success by checking final observation
            if self.filter_successful_only and observations:
                # Check the raw observation from the executor after full replay
                try:
                    raw_obs = executor._get_raw_observation()
                    if "player_won" in raw_obs:
                        if not raw_obs["player_won"]:
                            logger.debug(
                                f"Skipping replay - player_won=False at final frame. "
                                f"Episode: {replay.episode_id}"
                            )
                            executor.close()
                            return []
                    elif not replay.success:
                        # Fall back to replay's success flag if player_won not available
                        logger.debug(
                            f"Skipping replay - success flag is False. Episode: {replay.episode_id}"
                        )
                        executor.close()
                        return []
                except AttributeError:
                    # If _get_raw_observation doesn't exist, rely on replay.success
                    if not replay.success:
                        logger.debug(
                            f"Skipping replay - success flag is False. Episode: {replay.episode_id}"
                        )
                        executor.close()
                        return []

            executor.close()

            # Extract observations and actions from ReplayExecutor output with frame stacking
            # ReplayExecutor returns list of dicts with keys: 'observation', 'action', 'frame'
            for obs_data in observations:
                observation = self._process_observation(obs_data["observation"])
                action = obs_data["action"]

                # Add to frame buffers
                self._add_to_visual_buffer(observation)
                self._add_to_state_buffer(observation)

                # Only create samples once buffers are ready
                if self._buffers_ready():
                    stacked_obs = self._stack_observations(observation)
                    samples.append((stacked_obs, int(action)))

        except Exception as e:
            print(f"Failed to simulate replay (episode {replay.episode_id}): {e}")
            # Try fallback method using environment directly
            try:
                samples = self._simulate_replay_with_env(replay)
            except Exception as e2:
                print(f"Fallback simulation also failed for {replay.episode_id}: {e2}")
                return []

        return samples

    def _simulate_replay_with_env(
        self, replay: CompactReplay
    ) -> List[Tuple[Dict, int]]:
        """Fallback method: simulate replay using environment directly with frame stacking.

        This is less efficient but more compatible if ReplayExecutor has issues.

        Args:
            replay: CompactReplay object

        Returns:
            List of (observation, action) tuples
        """
        samples = []

        # Reset frame buffers for new replay
        self._reset_frame_buffers()

        # Create environment
        config = EnvironmentConfig.for_training()
        env = NppEnvironment(config=config)

        try:
            # Load map data directly into NPlayHeadless
            env.nplay_headless.load_map_from_map_data(list(replay.map_data))

            # Get initial observation without calling reset()
            # (reset would reload a random map, we want the replay map)
            obs = env._get_observation()

            # Execute each input from the replay
            for input_byte in replay.input_sequence:
                # Decode input to controls (same format as ReplayExecutor)
                from nclone.replay.input_utils import decode_input_to_controls

                horizontal, jump = decode_input_to_controls(input_byte)

                # Convert to action index for environment
                # Environment action space: 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=LEFT+JUMP, 5=RIGHT+JUMP
                if horizontal < 0 and jump:
                    action = 4  # LEFT+JUMP
                elif horizontal > 0 and jump:
                    action = 5  # RIGHT+JUMP
                elif horizontal < 0:
                    action = 1  # LEFT
                elif horizontal > 0:
                    action = 2  # RIGHT
                elif jump:
                    action = 3  # JUMP
                else:
                    action = 0  # NOOP

                # Process observation and add to frame buffers
                processed_obs = self._process_observation(obs)
                self._add_to_visual_buffer(processed_obs)
                self._add_to_state_buffer(processed_obs)

                # Only create samples once buffers are ready
                if self._buffers_ready():
                    stacked_obs = self._stack_observations(processed_obs)
                    samples.append((stacked_obs, action))

                # Step environment
                obs, _, terminated, truncated, _ = env.step(action)

                # Stop if episode ends
                if terminated or truncated:
                    break

        finally:
            env.close()

        return samples

    def _process_observation(self, obs: Dict) -> Dict:
        """Process observation into format suitable for caching.

        Args:
            obs: Raw observation from environment or ReplayExecutor

        Returns:
            Processed observation dictionary
        """
        # Extract only the components needed for training
        # Handle both full environment observations and ReplayExecutor observations
        processed = {}

        # Define all possible observation keys
        all_keys = {
            "player_frame": "player_frame",
            "global_view": "global_view",
            "game_state": "game_state",
            "reachability_features": "reachability_features",
            "entity_positions": "entity_positions",
            "graph_node_feats": "graph_node_feats",
            "graph_edge_index": "graph_edge_index",
            "graph_edge_feats": "graph_edge_feats",
            "graph_node_mask": "graph_node_mask",
            "graph_edge_mask": "graph_edge_mask",
            "graph_node_types": "graph_node_types",
            "graph_edge_types": "graph_edge_types",
        }

        # Filter based on architecture config if provided
        if self.architecture_config is not None and hasattr(
            self.architecture_config, "modalities"
        ):
            modalities = self.architecture_config.modalities

            # Only include observations required by the architecture
            required_keys = []
            if modalities.use_player_frame:
                required_keys.append("player_frame")
            if modalities.use_global_view:
                required_keys.append("global_view")
            if modalities.use_game_state:
                required_keys.append("game_state")
            if modalities.use_reachability:
                required_keys.append("reachability_features")
            if modalities.use_graph:
                required_keys.extend(
                    [
                        "graph_node_feats",
                        "graph_edge_index",
                        "graph_edge_feats",
                        "graph_node_mask",
                        "graph_edge_mask",
                        "graph_node_types",
                        "graph_edge_types",
                    ]
                )
        else:
            # Include all available keys if no architecture config provided
            required_keys = list(all_keys.keys())

        # Copy only required keys that are present in observation
        for key in required_keys:
            if key in obs:
                processed[key] = obs[key].copy()

        return processed

    def _save_to_cache(self, samples: List[Tuple[Dict, int]], cache_path: Path) -> None:
        """Save processed samples to cache file.

        Args:
            samples: List of (observation, action) tuples
            cache_path: Path to cache file
        """
        if not samples:
            return

        try:
            # Prepare data for saving
            cache_data = {
                "num_samples": len(samples),
            }

            # Store each observation component and action
            for i, (obs, action) in enumerate(samples):
                for key, value in obs.items():
                    cache_key = f"obs_{i}_{key}"
                    cache_data[cache_key] = value

                cache_data[f"action_{i}"] = action

            # Save to NPZ
            np.savez_compressed(cache_path, **cache_data)
            logger.debug(f"Cached {len(samples)} samples to {cache_path.name}")

        except Exception as e:
            print(f"Failed to save cache to {cache_path}: {e}")

    def _load_from_cache(self, cache_path: Path) -> List[Tuple[Dict, int]]:
        """Load processed samples from cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            List of (observation, action) tuples
        """
        try:
            data = np.load(cache_path, allow_pickle=False)
            num_samples = int(data["num_samples"])

            samples = []
            for i in range(num_samples):
                # Reconstruct observation
                obs = {}
                for key in [
                    "game_state",
                    "global_view",
                    "reachability_features",
                    "player_frame",
                    "entity_positions",
                    "graph_node_feats",
                    "graph_edge_index",
                    "graph_edge_feats",
                    "graph_node_mask",
                    "graph_edge_mask",
                    "graph_node_types",
                    "graph_edge_types",
                ]:
                    cache_key = f"obs_{i}_{key}"
                    if cache_key in data:
                        obs[key] = data[cache_key]

                # Get action
                action = int(data[f"action_{i}"])

                samples.append((obs, action))

            logger.debug(f"Loaded {len(samples)} samples from cache {cache_path.name}")
            return samples

        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")
            return []

    def _log_dataset_statistics(self) -> None:
        """Log statistics about the dataset for debugging and validation."""
        # Count action distribution
        action_counts = {}
        for _, action in self.samples:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Action names for readability
        action_names = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT",
            3: "JUMP",
            4: "LEFT+JUMP",
            5: "RIGHT+JUMP",
        }

        logger.info("=" * 60)
        logger.info("BC Dataset Statistics:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info("  Action distribution:")
        for action_id in sorted(action_counts.keys()):
            count = action_counts[action_id]
            percentage = 100.0 * count / len(self.samples)
            action_name = action_names.get(action_id, f"UNKNOWN_{action_id}")
            logger.info(f"    {action_name:12s}: {count:6d} ({percentage:5.2f}%)")

        # Check for observation keys in first sample
        if len(self.samples) > 0:
            first_obs, _ = self.samples[0]
            logger.info(f"  Observation keys: {sorted(first_obs.keys())}")

            # Log shapes of observation components
            logger.info("  Observation shapes:")
            for key, value in first_obs.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"    {key:25s}: {value.shape}")

        logger.info("=" * 60)

    def get_action_distribution(self) -> Dict[int, int]:
        """Get distribution of actions in the dataset.

        Returns:
            Dictionary mapping action indices to counts
        """
        action_counts = {}
        for _, action in self.samples:
            action_counts[action] = action_counts.get(action, 0) + 1
        return action_counts

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (observation_dict, action_tensor)
        """
        obs, action = self.samples[idx]

        # Apply normalization if enabled
        if self.normalize_observations:
            obs = self.normalizer.normalize(obs)

        # Convert observation to tensors
        obs_tensors = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.from_numpy(value.astype(np.float32))
            else:
                obs_tensors[key] = torch.tensor(value, dtype=torch.float32)

        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long)

        return obs_tensors, action_tensor
