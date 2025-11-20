"""Shared memory approaches for parallel curriculum learning.

This module implements shared memory data structures to minimize overhead
in parallel training environments (SubprocVecEnv) by avoiding expensive
state synchronization and pickling overhead.
"""

import ctypes
import logging
import multiprocessing as mp
import threading
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from npp_rl.training.curriculum_components import CURRICULUM_ORDER

logger = logging.getLogger(__name__)


class SharedPerformanceBuffer:
    """Shared memory buffer for curriculum performance tracking across processes.

    Uses multiprocessing shared memory to avoid expensive state synchronization
    between main process and subprocess environments.
    """

    def __init__(
        self,
        performance_window: int = 50,
        max_generators: int = 10,
    ):
        """Initialize shared performance buffer.

        Args:
            performance_window: Size of rolling performance window
            max_generators: Maximum number of generator types per stage
        """
        self.n_stages = len(CURRICULUM_ORDER)
        self.performance_window = performance_window
        self.max_generators = max_generators

        # Create stage name to index mapping
        self.stage_to_idx = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}

        # Shared memory arrays
        self._create_shared_arrays()

        # Thread locks for safe concurrent access
        self.write_lock = threading.RLock()

        logger.info(f"Initialized SharedPerformanceBuffer for {self.n_stages} stages")

    def _create_shared_arrays(self):
        """Create shared memory arrays for performance data."""
        # Performance circular buffers: [stages, window_size]
        self.performance_buffers = mp.Array(
            ctypes.c_int8, self.n_stages * self.performance_window, lock=False
        )

        # Buffer positions and sizes: [stages]
        self.buffer_positions = mp.Array(ctypes.c_int32, self.n_stages, lock=False)
        self.buffer_sizes = mp.Array(ctypes.c_int32, self.n_stages, lock=False)

        # Total episode counts: [stages]
        self.episode_counts = mp.Array(ctypes.c_int64, self.n_stages, lock=False)

        # Current stage index (atomic)
        self.current_stage_idx = mp.Value(ctypes.c_int32, 0, lock=False)

        # Adaptive mixing ratios: [stages]
        self.mixing_ratios = mp.Array(ctypes.c_float, self.n_stages, lock=False)
        self.mixing_cache_valid = mp.Value(ctypes.c_bool, False, lock=False)

        # Initialize arrays (no locks since arrays created with lock=False)
        np_buffer = np.frombuffer(self.performance_buffers, dtype=np.int8)
        np_buffer.fill(-1)  # -1 = uninitialized

        np.frombuffer(self.buffer_positions, dtype=np.int32).fill(0)

        np.frombuffer(self.buffer_sizes, dtype=np.int32).fill(0)

        np.frombuffer(self.episode_counts, dtype=np.int64).fill(0)

        np.frombuffer(self.mixing_ratios, dtype=np.float32).fill(0.2)

    def record_episode(self, stage: str, success: bool) -> bool:
        """Record episode result in shared memory.

        Args:
            stage: Stage name
            success: Whether episode was successful

        Returns:
            True if recorded successfully, False if stage not found
        """
        if stage not in self.stage_to_idx:
            return False

        stage_idx = self.stage_to_idx[stage]
        success_value = 1 if success else 0

        with self.write_lock:
            # Get current position and update circular buffer
            pos_array = np.frombuffer(self.buffer_positions, dtype=np.int32)
            pos = pos_array[stage_idx]
            pos_array[stage_idx] = (pos + 1) % self.performance_window

            # Update buffer size (capped at window size)
            size_array = np.frombuffer(self.buffer_sizes, dtype=np.int32)
            size_array[stage_idx] = min(
                size_array[stage_idx] + 1, self.performance_window
            )

            # Write performance data
            buffer_array = np.frombuffer(self.performance_buffers, dtype=np.int8)
            buffer_2d = buffer_array.reshape(self.n_stages, self.performance_window)
            buffer_2d[stage_idx, pos] = success_value

            # Increment episode count
            count_array = np.frombuffer(self.episode_counts, dtype=np.int64)
            count_array[stage_idx] += 1

            # Invalidate mixing ratio cache when performance changes
            self.mixing_cache_valid.value = False

        return True

    def get_success_rate(self, stage: str) -> float:
        """Get success rate for a stage from shared memory.

        Args:
            stage: Stage name

        Returns:
            Success rate (0.0 to 1.0), or 0.0 if stage not found
        """
        if stage not in self.stage_to_idx:
            return 0.0

        stage_idx = self.stage_to_idx[stage]

        # Read buffer size
        size_array = np.frombuffer(self.buffer_sizes, dtype=np.int32)
        buffer_size = size_array[stage_idx]

        if buffer_size == 0:
            return 0.0

        # Read performance data
        buffer_array = np.frombuffer(self.performance_buffers, dtype=np.int8)
        buffer_2d = buffer_array.reshape(self.n_stages, self.performance_window)
        stage_data = buffer_2d[stage_idx, :buffer_size]

        # Calculate mean, filtering out uninitialized values
        valid_data = stage_data[stage_data >= 0]
        if len(valid_data) == 0:
            return 0.0

        return float(np.mean(valid_data))

    def get_episode_count(self, stage: str) -> int:
        """Get total episode count for a stage.

        Args:
            stage: Stage name

        Returns:
            Episode count, or 0 if stage not found
        """
        if stage not in self.stage_to_idx:
            return 0

        stage_idx = self.stage_to_idx[stage]

        count_array = np.frombuffer(self.episode_counts, dtype=np.int64)
        return int(count_array[stage_idx])

    def set_current_stage(self, stage: str) -> bool:
        """Set current curriculum stage atomically.

        Args:
            stage: Stage name

        Returns:
            True if set successfully, False if stage not found
        """
        if stage not in self.stage_to_idx:
            return False

        stage_idx = self.stage_to_idx[stage]

        self.current_stage_idx.value = stage_idx

        return True

    def get_current_stage(self) -> str:
        """Get current curriculum stage atomically.

        Returns:
            Current stage name
        """
        stage_idx = self.current_stage_idx.value

        if 0 <= stage_idx < len(CURRICULUM_ORDER):
            return CURRICULUM_ORDER[stage_idx]
        else:
            return CURRICULUM_ORDER[0]  # Fallback to first stage

    def get_current_stage_idx(self) -> int:
        """Get current curriculum stage index atomically.

        Returns:
            Current stage index
        """
        return self.current_stage_idx.value

    def set_mixing_ratio(self, stage: str, ratio: float) -> bool:
        """Set adaptive mixing ratio for a stage.

        Args:
            stage: Stage name
            ratio: Mixing ratio (0.0 to 1.0)

        Returns:
            True if set successfully, False if stage not found
        """
        if stage not in self.stage_to_idx:
            return False

        stage_idx = self.stage_to_idx[stage]

        ratio_array = np.frombuffer(self.mixing_ratios, dtype=np.float32)
        ratio_array[stage_idx] = float(ratio)

        self.mixing_cache_valid.value = True

        return True

    def get_mixing_ratio(self, stage: str, default: float = 0.2) -> float:
        """Get adaptive mixing ratio for a stage.

        Args:
            stage: Stage name
            default: Default ratio if not cached

        Returns:
            Mixing ratio or default if not found/cached
        """
        if stage not in self.stage_to_idx:
            return default

        # Check if cache is valid
        if not self.mixing_cache_valid.value:
            return default

        stage_idx = self.stage_to_idx[stage]

        ratio_array = np.frombuffer(self.mixing_ratios, dtype=np.float32)
        return float(ratio_array[stage_idx])

    def calculate_trend(self, stage: str) -> float:
        """Calculate performance trend for a stage.

        Args:
            stage: Stage name

        Returns:
            Trend value: positive = improving, negative = declining
        """
        if stage not in self.stage_to_idx:
            return 0.0

        stage_idx = self.stage_to_idx[stage]

        # Read buffer size
        size_array = np.frombuffer(self.buffer_sizes, dtype=np.int32)
        buffer_size = size_array[stage_idx]

        if buffer_size < 20:  # Need sufficient data for trend
            return 0.0

        # Read performance data
        buffer_array = np.frombuffer(self.performance_buffers, dtype=np.int8)
        buffer_2d = buffer_array.reshape(self.n_stages, self.performance_window)
        stage_data = buffer_2d[stage_idx, :buffer_size]

        # Filter out uninitialized values
        valid_data = stage_data[stage_data >= 0]
        if len(valid_data) < 20:
            return 0.0

        # Split into halves and calculate trend
        mid = len(valid_data) // 2
        first_half = valid_data[:mid]
        second_half = valid_data[mid:]

        return float(np.mean(second_half) - np.mean(first_half))

    def get_all_stage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all stages efficiently.

        Returns:
            Dictionary mapping stage names to their performance stats
        """
        stats = {}

        # Read all data in one go for efficiency
        size_array = np.frombuffer(self.buffer_sizes, dtype=np.int32).copy()
        count_array = np.frombuffer(self.episode_counts, dtype=np.int64).copy()
        buffer_array = np.frombuffer(self.performance_buffers, dtype=np.int8)
        buffer_2d = buffer_array.reshape(self.n_stages, self.performance_window).copy()

        # Calculate stats for each stage
        for stage, stage_idx in self.stage_to_idx.items():
            buffer_size = size_array[stage_idx]
            episode_count = count_array[stage_idx]

            if buffer_size > 0:
                stage_data = buffer_2d[stage_idx, :buffer_size]
                valid_data = stage_data[stage_data >= 0]
                success_rate = (
                    float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0
                )

                # Calculate trend if enough data
                trend = 0.0
                if len(valid_data) >= 20:
                    mid = len(valid_data) // 2
                    first_half = valid_data[:mid]
                    second_half = valid_data[mid:]
                    trend = float(np.mean(second_half) - np.mean(first_half))
            else:
                success_rate = 0.0
                trend = 0.0

            stats[stage] = {
                "success_rate": success_rate,
                "episodes": int(episode_count),
                "recent_episodes": buffer_size,
                "trend": trend,
            }

        return stats


class SharedMemoryCurriculumManager:
    """Curriculum manager optimized for parallel environments using shared memory.

    This manager uses shared memory to minimize overhead when used with
    SubprocVecEnv, avoiding expensive state synchronization and pickling.
    """

    def __init__(
        self,
        starting_stage: str = "simplest",
        performance_window: int = 50,
        shared_buffer: Optional[SharedPerformanceBuffer] = None,
        dataset_path: Optional[str] = None,
        lazy_cache_size: int = 500,  # Smaller cache for shared memory to reduce overhead
    ):
        """Initialize shared memory curriculum manager.

        Args:
            starting_stage: Initial curriculum stage
            performance_window: Size of rolling performance window
            shared_buffer: Existing shared buffer, or None to create new one
            dataset_path: Path to curriculum dataset (required for level sampling)
            lazy_cache_size: Size of LRU cache for level loading
        """
        self.starting_stage = starting_stage
        self.performance_window = performance_window
        self.dataset_path = dataset_path
        self.lazy_cache_size = lazy_cache_size

        # Initialize lazy level sampler if dataset path is provided
        # Note: This will be recreated in worker processes after unpickling
        self._level_sampler = None
        self._should_init_sampler = bool(dataset_path)
        # Don't initialize sampler in main process to avoid pickling issues
        # if dataset_path:
        #     self._initialize_level_sampler()

        # Validate starting stage
        if starting_stage not in CURRICULUM_ORDER:
            raise ValueError(
                f"Invalid starting stage '{starting_stage}'. Must be one of: {CURRICULUM_ORDER}"
            )

        # Create or use shared buffer
        if shared_buffer is None:
            self.shared_buffer = SharedPerformanceBuffer(performance_window)
            # Set initial stage
            self.shared_buffer.set_current_stage(starting_stage)
        else:
            self.shared_buffer = shared_buffer

        logger.info(
            f"Initialized SharedMemoryCurriculumManager with {len(CURRICULUM_ORDER)} stages"
        )
        if self._level_sampler:
            logger.info("Level sampling enabled with lazy loading")

    def __getstate__(self):
        """Custom pickle support - exclude unpickleable level sampler."""
        state = self.__dict__.copy()
        # Remove level sampler to avoid pickling issues
        state["_level_sampler"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle support - recreate level sampler if needed."""
        self.__dict__.update(state)
        # Recreate level sampler in worker process if needed
        if self._should_init_sampler and self.dataset_path:
            self._initialize_level_sampler()

    def _initialize_level_sampler(self):
        """Initialize lazy level sampler for shared memory environment."""
        try:
            from npp_rl.training.curriculum_manager import LazyLevelSampler
            from npp_rl.evaluation.test_suite_loader import TestSuiteLoader

            # Load metadata for lazy sampling
            loader = TestSuiteLoader(self.dataset_path)
            all_metadata = loader.load_all_metadata()

            # Group metadata by stage and generator
            metadata_by_stage_and_generator = {}

            for stage in CURRICULUM_ORDER:
                metadata_by_stage_and_generator[stage] = {}

                if stage not in all_metadata:
                    logger.warning(f"No metadata found for stage '{stage}' - skipping")
                    continue

                # Group metadata by generator type
                for level_metadata in all_metadata[stage]:
                    generator_type = level_metadata.get("metadata", {}).get(
                        "generator", "unknown"
                    )

                    # Load one level to get correct generator info if unknown
                    if generator_type == "unknown":
                        file_path = level_metadata.get("file_path")
                        if file_path:
                            try:
                                sample_level = loader.load_single_level(file_path)
                                if sample_level and "metadata" in sample_level:
                                    generator_type = sample_level["metadata"].get(
                                        "generator", "unknown"
                                    )
                                    level_metadata["metadata"]["generator"] = (
                                        generator_type
                                    )
                            except Exception:
                                generator_type = "unknown"

                    if generator_type not in metadata_by_stage_and_generator[stage]:
                        metadata_by_stage_and_generator[stage][generator_type] = []

                    metadata_by_stage_and_generator[stage][generator_type].append(
                        level_metadata
                    )

            # Initialize lazy sampler
            self._level_sampler = LazyLevelSampler(
                metadata_by_stage_and_generator, self.dataset_path, self.lazy_cache_size
            )

            logger.info("Lazy level sampler initialized for shared memory manager")

        except Exception as e:
            logger.warning(f"Failed to initialize level sampler: {e}")
            self._level_sampler = None

    def sample_level(self) -> Optional[Dict[str, Any]]:
        """Sample a level using the lazy sampler.

        Returns:
            Level data dictionary, or None if sampling fails
        """
        # Lazily initialize level sampler if needed
        if not self._level_sampler and self._should_init_sampler:
            self._initialize_level_sampler()

        if not self._level_sampler:
            logger.warning("Level sampler not initialized - cannot sample levels")
            return None

        try:
            # Get current stage index from shared memory
            current_stage_idx = self.shared_buffer.get_current_stage_idx()

            # Sample level using lazy sampler with minimal mixing for shared memory
            level_data, sampled_stage, sampled_generator = (
                self._level_sampler.sample_level_optimized(
                    current_stage_idx=current_stage_idx,
                    cached_mixing_ratio=0.1,  # Lower mixing ratio for shared memory
                )
            )

            return level_data

        except Exception as e:
            logger.warning(f"Failed to sample level: {e}")
            return None

    def record_episode(
        self,
        stage: str,
        success: bool,
        generator_type: Optional[str] = None,
        frames: Optional[int] = None,
    ):
        """Record episode result in shared memory.

        Args:
            stage: Stage name
            success: Whether episode was successful
            generator_type: Generator type (ignored in shared memory version)
            frames: Number of frames (ignored in shared memory version)
        """
        self.shared_buffer.record_episode(stage, success)

    def get_current_stage(self) -> str:
        """Get current curriculum stage."""
        return self.shared_buffer.get_current_stage()

    def get_current_stage_index(self) -> int:
        """Get current curriculum stage index."""
        return self.shared_buffer.get_current_stage_idx()

    def get_stage_success_rate(self, stage: str) -> float:
        """Get success rate for a stage."""
        return self.shared_buffer.get_success_rate(stage)

    def get_stage_performance(self, stage: str) -> Dict[str, any]:
        """Get performance metrics for a stage (simplified for shared memory)."""
        success_rate = self.shared_buffer.get_success_rate(stage)
        episodes = self.shared_buffer.get_episode_count(stage)
        trend = self.shared_buffer.calculate_trend(stage)

        # Simplified thresholds for shared memory version
        STAGE_THRESHOLDS = {
            "simplest": 0.70,
            "simplest_few_mines": 0.65,
            "simplest_with_mines": 0.60,
            "simpler": 0.60,
            "simple": 0.50,
            "medium": 0.45,
            "complex": 0.40,
            "exploration": 0.35,
            "mine_heavy": 0.30,
        }

        STAGE_MIN_EPISODES = {
            "simplest": 100,
            "simplest_few_mines": 100,
            "simplest_with_mines": 100,
            "simpler": 100,
            "simple": 75,
            "medium": 100,
            "complex": 150,
            "exploration": 150,
            "mine_heavy": 200,
        }

        stage_threshold = STAGE_THRESHOLDS.get(stage, 0.7)
        stage_min_episodes = STAGE_MIN_EPISODES.get(stage, 100)

        can_advance = success_rate >= stage_threshold and episodes >= stage_min_episodes

        return {
            "success_rate": success_rate,
            "episodes": episodes,
            "recent_episodes": min(episodes, self.performance_window),
            "can_advance": can_advance,
            "advancement_threshold": stage_threshold,
            "min_episodes": stage_min_episodes,
            "trend": trend,
        }

    def check_advancement(self) -> bool:
        """Check if agent should advance to next curriculum stage."""
        current_idx = self.shared_buffer.get_current_stage_idx()

        if current_idx >= len(CURRICULUM_ORDER) - 1:
            return False  # Already at final stage

        current_stage = CURRICULUM_ORDER[current_idx]
        perf = self.get_stage_performance(current_stage)

        if perf["can_advance"]:
            next_stage = CURRICULUM_ORDER[current_idx + 1]
            success = self.shared_buffer.set_current_stage(next_stage)

            if success:
                logger.info(
                    f"Shared memory curriculum advancement: {current_stage} → {next_stage}"
                )
                return True

        return False

    def get_all_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance for all stages efficiently."""
        return self.shared_buffer.get_all_stage_stats()


def create_shared_curriculum_manager(
    starting_stage: str = "simplest",
    performance_window: int = 50,
    dataset_path: Optional[str] = None,
    lazy_cache_size: int = 500,
) -> Tuple[SharedMemoryCurriculumManager, SharedPerformanceBuffer]:
    """Create shared memory curriculum manager and buffer.

    Args:
        starting_stage: Initial stage
        performance_window: Rolling window size
        dataset_path: Path to curriculum dataset (required for level sampling)
        lazy_cache_size: Size of LRU cache for level loading

    Returns:
        Tuple of (manager, shared_buffer) for use across processes
    """
    shared_buffer = SharedPerformanceBuffer(performance_window)
    manager = SharedMemoryCurriculumManager(
        starting_stage,
        performance_window,
        shared_buffer,
        dataset_path,
        lazy_cache_size,
    )

    return manager, shared_buffer
