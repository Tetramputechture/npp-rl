"""Curriculum learning manager for N++ RL training.

This module implements curriculum learning to progressively train agents
on increasingly difficult N++ levels, tracking performance and automatically
advancing through curriculum stages based on success rates.

High-performance implementation with optimized data structures:
- Fast sample_level() through precomputed data structures
- Memory-efficient tracking with consolidated data structures
- Optimized parallelization with intelligent caching
- Clean modular architecture
- Lazy loading to minimize initialization overhead
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from npp_rl.evaluation.test_suite_loader import TestSuiteLoader
from npp_rl.training.curriculum_components import CURRICULUM_ORDER

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for level data to minimize memory usage."""

    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of levels to keep in cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache, moving it to end (most recently used)."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        """Add item to cache, evicting oldest if at capacity."""
        if key in self.cache:
            # Update existing and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove oldest item (first in OrderedDict)
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __getstate__(self):
        """Custom pickle support for LRU cache."""
        return {"max_size": self.max_size, "items": list(self.cache.items())}

    def __setstate__(self, state):
        """Custom unpickle support for LRU cache."""
        self.max_size = state["max_size"]
        self.cache = OrderedDict(state["items"])


class LazyLevelSampler:
    """Lazy level sampler that loads levels on-demand with LRU caching.

    This sampler provides the same interface as FastCurriculumSampler but loads
    levels only when needed, dramatically reducing initialization time and
    memory usage for SubprocVecEnv.
    """

    def __init__(
        self,
        levels_metadata_by_stage_and_generator: Dict[
            str, Dict[str, List[Dict[str, Any]]]
        ],
        dataset_path: str,
        cache_size: int = 1000,
    ):
        """Initialize lazy sampler with level metadata only.

        Args:
            levels_metadata_by_stage_and_generator: Metadata for all levels
            dataset_path: Path to dataset for loading levels on demand
            cache_size: Maximum number of levels to keep in LRU cache
        """
        self.dataset_path = Path(dataset_path)
        self.level_cache = None  # Will be created lazily
        self.test_suite_loader = None  # Will be created lazily

        # Store original parameters for pickling and lazy initialization
        self._dataset_path_str = str(self.dataset_path)
        self._cache_size = cache_size

        # Store metadata instead of full level data
        self.levels_metadata = levels_metadata_by_stage_and_generator
        self._metadata_loaded = len(levels_metadata_by_stage_and_generator) > 0

        # Precompute sampling arrays using metadata (if available)
        if self._metadata_loaded:
            self._precompute_sampling_arrays()
            self._initialize_sample_counts()
        else:
            # Will be initialized lazily when metadata is loaded
            self.stage_generators = {}
            self.generator_metadata = {}
            self.generator_indices = {}
            self.stage_indices = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}
            self.sample_count_arrays = {}
            self.max_generators_per_stage = 0

        # Cache for adaptive mixing ratios
        self._cached_mixing_ratios = {}
        self._mixing_ratio_cache_valid = False

        logger.info(f"Initialized lazy level sampler with cache size {cache_size}")

    def __getstate__(self):
        """Custom pickle support - exclude unpickleable objects."""
        state = self.__dict__.copy()
        # Remove unpickleable TestSuiteLoader and cache
        state["test_suite_loader"] = None
        state["level_cache"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle support - recreate excluded objects."""
        self.__dict__.update(state)
        # Don't recreate TestSuiteLoader and cache here - they'll be created lazily when first needed
        self.test_suite_loader = None
        self.level_cache = None

    def _precompute_sampling_arrays(self):
        """Precompute data structures for fast sampling using metadata."""
        self.stage_generators = {}  # stage -> list of generator names
        self.generator_metadata = {}  # (stage, generator) -> list of metadata dicts
        self.generator_indices = {}  # (stage, generator) -> index in arrays
        self.stage_indices = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}

        for stage, generators in self.levels_metadata.items():
            if stage not in self.stage_indices:
                continue

            # Store generator names as list for indexing
            generator_names = list(generators.keys())
            self.stage_generators[stage] = generator_names

            # Create generator index mapping
            gen_indices = {gen: i for i, gen in enumerate(generator_names)}
            self.generator_indices[stage] = gen_indices

            # Store metadata for each generator
            for gen_name, metadata_list in generators.items():
                key = (stage, gen_name)
                self.generator_metadata[key] = metadata_list

    def _initialize_sample_counts(self):
        """Initialize sample count arrays for stratified sampling."""
        self.sample_count_arrays = {}
        self.max_generators_per_stage = 0

        for stage, generators in self.stage_generators.items():
            n_generators = len(generators)
            self.sample_count_arrays[stage] = np.zeros(n_generators, dtype=np.int32)
            self.max_generators_per_stage = max(
                self.max_generators_per_stage, n_generators
            )

    def _ensure_loader_and_cache(self):
        """Lazily initialize TestSuiteLoader and cache to avoid pickling issues."""
        if self.test_suite_loader is None:
            from npp_rl.evaluation.test_suite_loader import TestSuiteLoader

            self.test_suite_loader = TestSuiteLoader(self._dataset_path_str)

        if self.level_cache is None:
            self.level_cache = LRUCache(self._cache_size)

    def _ensure_metadata_loaded(self):
        """Lazily load metadata if not already loaded."""
        if not self._metadata_loaded:
            # Ensure we have a TestSuiteLoader
            self._ensure_loader_and_cache()

            # Load metadata
            all_metadata = self.test_suite_loader.load_all_metadata()

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
                                sample_level = self.test_suite_loader.load_single_level(
                                    file_path
                                )
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

            # Update stored metadata and mark as loaded
            self.levels_metadata = metadata_by_stage_and_generator
            self._metadata_loaded = True

            # Now precompute sampling arrays
            self._precompute_sampling_arrays()
            self._initialize_sample_counts()

            logger.info("Metadata loaded lazily for LazyLevelSampler")

    def _load_level_on_demand(
        self, level_metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Load a single level on demand with caching.

        Args:
            level_metadata: Metadata dict containing file_path

        Returns:
            Full level data, or None if loading failed
        """
        # Ensure loader and cache are initialized
        self._ensure_loader_and_cache()

        file_path = level_metadata.get("file_path")
        if not file_path:
            logger.warning(f"No file_path in level metadata: {level_metadata}")
            return None

        # Check cache first
        cached_level = self.level_cache.get(file_path)
        if cached_level is not None:
            return cached_level

        # Load from disk
        level_data = self.test_suite_loader.load_single_level(file_path)
        if level_data is not None:
            # Add to cache
            self.level_cache.put(file_path, level_data)

            # Ensure generator type is populated correctly
            if "metadata" not in level_data:
                level_data["metadata"] = {}
            if "generator" not in level_data["metadata"]:
                level_data["metadata"]["generator"] = level_metadata.get(
                    "metadata", {}
                ).get("generator", "unknown")

            # Add category field for compatibility
            level_data["category"] = level_metadata.get("category", "unknown")

        return level_data

    def invalidate_mixing_cache(self):
        """Invalidate cached mixing ratios when performance data changes."""
        self._mixing_ratio_cache_valid = False

    def set_cached_mixing_ratio(self, stage: str, ratio: float):
        """Set cached mixing ratio for a stage."""
        self._cached_mixing_ratios[stage] = ratio
        self._mixing_ratio_cache_valid = True

    def get_cached_mixing_ratio(self, stage: str, default: float = 0.2) -> float:
        """Get cached mixing ratio for a stage."""
        if not self._mixing_ratio_cache_valid:
            return default
        return self._cached_mixing_ratios.get(stage, default)

    def sample_level_optimized(
        self,
        current_stage_idx: int,
        cached_mixing_ratio: float = 0.2,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Sample level with lazy loading.

        Args:
            current_stage_idx: Current curriculum stage index
            cached_mixing_ratio: Cached mixing ratio to avoid recalculation

        Returns:
            Tuple of (level_data, sampled_stage, sampled_generator)
        """
        # Ensure metadata is loaded before sampling
        self._ensure_metadata_loaded()

        # Fast stage selection using cached mixing ratio
        if current_stage_idx > 0:
            if np.random.random() < cached_mixing_ratio:
                sample_stage_idx = current_stage_idx - 1
            else:
                sample_stage_idx = current_stage_idx
        else:
            sample_stage_idx = current_stage_idx

        sample_stage = CURRICULUM_ORDER[sample_stage_idx]

        # Check if stage has any generators
        if sample_stage not in self.stage_generators:
            raise ValueError(f"No generators found for stage '{sample_stage}'")

        generators = self.stage_generators[sample_stage]
        if not generators:
            raise ValueError(f"Stage '{sample_stage}' has empty generator list")

        # Select generator using stratified sampling
        counts = self.sample_count_arrays[sample_stage]
        generator_idx = np.argmin(counts)  # Select least sampled generator
        selected_generator = generators[generator_idx]

        # Get metadata for this stage/generator combination
        key = (sample_stage, selected_generator)
        metadata_list = self.generator_metadata[key]

        if not metadata_list:
            raise ValueError(
                f"No levels found for stage '{sample_stage}', generator '{selected_generator}'"
            )

        # Sample random level from this generator
        level_metadata = np.random.choice(metadata_list)

        # Load level on demand with caching
        level_data = self._load_level_on_demand(level_metadata)

        if level_data is None:
            # Fallback: try another level from same generator
            for fallback_metadata in metadata_list:
                if fallback_metadata != level_metadata:
                    level_data = self._load_level_on_demand(fallback_metadata)
                    if level_data is not None:
                        break

        if level_data is None:
            raise RuntimeError(
                f"Failed to load any level from stage '{sample_stage}', generator '{selected_generator}'"
            )

        # Update sample count
        counts[generator_idx] += 1

        # Add sampled metadata for curriculum tracking
        level_data["sampled_stage"] = sample_stage
        level_data["sampled_generator"] = selected_generator

        return level_data, sample_stage, selected_generator


class FastCurriculumSampler:
    """Fast level sampling with precomputed data structures.

    This class handles the hot path of curriculum sampling with optimized
    data structures to minimize runtime overhead.
    """

    def __init__(
        self, levels_by_stage_and_generator: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ):
        """Initialize fast sampler with precomputed data structures.

        Args:
            levels_by_stage_and_generator: Nested dict of level data
        """
        # Precompute sampling arrays for O(1) access
        self._precompute_sampling_arrays(levels_by_stage_and_generator)

        # Initialize sample counts as numpy arrays for fast operations
        self._initialize_sample_counts()

        # Cache for adaptive mixing ratios
        self._cached_mixing_ratios = {}
        self._mixing_ratio_cache_valid = False

    def _precompute_sampling_arrays(
        self, levels_by_stage_and_generator: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ):
        """Precompute optimized data structures for fast sampling.

        Args:
            levels_by_stage_and_generator: Raw level data
        """
        self.stage_generators = {}  # stage -> list of generator names
        self.generator_levels = {}  # (stage, generator) -> numpy array of levels
        self.generator_indices = {}  # (stage, generator) -> index in arrays
        self.stage_indices = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}

        for stage, generators in levels_by_stage_and_generator.items():
            if stage not in self.stage_indices:
                continue

            # Store generator names as list for indexing
            generator_names = list(generators.keys())
            self.stage_generators[stage] = generator_names

            # Create generator index mapping
            gen_indices = {gen: i for i, gen in enumerate(generator_names)}
            self.generator_indices[stage] = gen_indices

            # Convert level lists to numpy arrays for fast random access
            for gen_name, levels in generators.items():
                key = (stage, gen_name)
                self.generator_levels[key] = np.array(levels, dtype=object)

    def _initialize_sample_counts(self):
        """Initialize sample count arrays for fast stratified sampling."""
        self.sample_count_arrays = {}  # stage -> numpy array of counts
        self.max_generators_per_stage = 0

        for stage, generators in self.stage_generators.items():
            n_generators = len(generators)
            self.sample_count_arrays[stage] = np.zeros(n_generators, dtype=np.int32)
            self.max_generators_per_stage = max(
                self.max_generators_per_stage, n_generators
            )

    def invalidate_mixing_cache(self):
        """Invalidate cached mixing ratios when performance data changes."""
        self._mixing_ratio_cache_valid = False

    def set_cached_mixing_ratio(self, stage: str, ratio: float):
        """Set cached mixing ratio for a stage."""
        self._cached_mixing_ratios[stage] = ratio
        self._mixing_ratio_cache_valid = True

    def get_cached_mixing_ratio(self, stage: str, default: float = 0.2) -> float:
        """Get cached mixing ratio for a stage."""
        if not self._mixing_ratio_cache_valid:
            return default
        return self._cached_mixing_ratios.get(stage, default)

    def sample_level_optimized(
        self,
        current_stage_idx: int,
        cached_mixing_ratio: float = 0.2,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Fast optimized level sampling.

        Args:
            current_stage_idx: Current curriculum stage index
            cached_mixing_ratio: Cached mixing ratio to avoid recalculation

        Returns:
            Tuple of (level_data, sampled_stage, sampled_generator)
        """
        # Fast stage selection using cached mixing ratio
        if current_stage_idx > 0:
            if np.random.random() < cached_mixing_ratio:
                sample_stage_idx = current_stage_idx - 1
            else:
                sample_stage_idx = current_stage_idx
        else:
            sample_stage_idx = current_stage_idx

        sample_stage = CURRICULUM_ORDER[sample_stage_idx]

        # Fast stratified sampling using numpy arrays
        count_array = self.sample_count_arrays[sample_stage]
        min_count = np.min(count_array)

        # Find all generators with minimum count
        min_indices = np.where(count_array == min_count)[0]

        # Fast random selection from minimum indices
        selected_gen_idx = np.random.choice(min_indices)
        generator_names = self.stage_generators[sample_stage]
        selected_generator = generator_names[selected_gen_idx]

        # Fast level selection from precomputed array
        key = (sample_stage, selected_generator)
        level_array = self.generator_levels[key]
        level_idx = np.random.randint(len(level_array))
        level = level_array[level_idx]

        # Fast count increment
        count_array[selected_gen_idx] += 1

        # Return without modifying original level dict (avoid copy overhead)
        return level, sample_stage, selected_generator

    def get_sample_counts(self, stage: str) -> Dict[str, int]:
        """Get current sample counts for a stage."""
        if stage not in self.sample_count_arrays:
            return {}

        count_array = self.sample_count_arrays[stage]
        generator_names = self.stage_generators[stage]

        return {gen: int(count_array[i]) for i, gen in enumerate(generator_names)}


class CompactPerformanceTracker:
    """Memory-efficient performance tracking using numpy arrays and circular buffers.

    Consolidates performance tracking to reduce memory overhead and improve
    cache efficiency.
    """

    def __init__(self, performance_window: int = 50):
        """Initialize compact performance tracker.

        Args:
            curriculum_order: List of curriculum stages
            performance_window: Size of rolling performance window
        """
        self.performance_window = performance_window
        self.n_stages = len(CURRICULUM_ORDER)
        self.stage_indices = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}

        # Circular buffers for performance data (more memory efficient than deques)
        self.performance_buffers = np.full(
            (self.n_stages, performance_window), -1, dtype=np.int8
        )
        self.buffer_positions = np.zeros(self.n_stages, dtype=np.int32)
        self.buffer_sizes = np.zeros(self.n_stages, dtype=np.int32)

        # Episode counts per stage
        self.episode_counts = np.zeros(self.n_stages, dtype=np.int32)

        # Generator-specific performance tracking (consolidated)
        self.generator_stats = {}  # stage -> {generator: [successes, total_episodes]}

    def record_episode(
        self, stage: str, success: bool, generator_type: Optional[str] = None
    ):
        """Record an episode result efficiently.

        Args:
            stage: Stage name
            success: Whether episode was successful
            generator_type: Generator type (optional)
        """
        if stage not in self.stage_indices:
            logger.warning(f"Unknown stage '{stage}', ignoring episode")
            return

        stage_idx = self.stage_indices[stage]

        # Add to circular buffer
        pos = self.buffer_positions[stage_idx]
        self.performance_buffers[stage_idx, pos] = 1 if success else 0

        # Update position and size
        self.buffer_positions[stage_idx] = (pos + 1) % self.performance_window
        self.buffer_sizes[stage_idx] = min(
            self.buffer_sizes[stage_idx] + 1, self.performance_window
        )
        self.episode_counts[stage_idx] += 1

        # Update generator stats if provided
        if generator_type:
            if stage not in self.generator_stats:
                self.generator_stats[stage] = {}

            if generator_type not in self.generator_stats[stage]:
                self.generator_stats[stage][generator_type] = np.array(
                    [0, 0], dtype=np.int32
                )

            stats = self.generator_stats[stage][generator_type]
            if success:
                stats[0] += 1  # successes
            stats[1] += 1  # total episodes

    def get_success_rate(self, stage: str) -> float:
        """Get success rate for a stage efficiently.

        Args:
            stage: Stage name

        Returns:
            Success rate (0.0 to 1.0)
        """
        if stage not in self.stage_indices:
            return 0.0

        stage_idx = self.stage_indices[stage]
        buffer_size = self.buffer_sizes[stage_idx]

        if buffer_size == 0:
            return 0.0

        # Fast numpy sum over valid buffer entries
        buffer_data = self.performance_buffers[stage_idx, :buffer_size]
        return float(np.mean(buffer_data))

    def get_episode_count(self, stage: str) -> int:
        """Get total episode count for a stage."""
        if stage not in self.stage_indices:
            return 0
        return int(self.episode_counts[self.stage_indices[stage]])

    def calculate_trend(self, stage: str) -> float:
        """Calculate performance trend efficiently."""
        if stage not in self.stage_indices:
            return 0.0

        stage_idx = self.stage_indices[stage]
        buffer_size = self.buffer_sizes[stage_idx]

        if buffer_size < 20:
            return 0.0

        # Efficient trend calculation using numpy
        buffer_data = self.performance_buffers[stage_idx, :buffer_size]
        mid = buffer_size // 2

        first_half_mean = np.mean(buffer_data[:mid])
        second_half_mean = np.mean(buffer_data[mid:])

        return float(second_half_mean - first_half_mean)

    def get_generator_performance(self, stage: str) -> Dict[str, Dict[str, int]]:
        """Get generator-specific performance stats."""
        if stage not in self.generator_stats:
            return {}

        result = {}
        for generator, stats in self.generator_stats[stage].items():
            successes, total = stats
            result[generator] = {
                "successes": int(successes),
                "failures": int(total - successes),
                "success_rate": float(successes / total) if total > 0 else 0.0,
            }

        return result


class CurriculumManager:
    """Curriculum learning manager with high-performance data structures and algorithms.

    Manages progression through difficulty levels, tracking performance and automatically
    advancing through curriculum stages based on success rates.
    """

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

    EARLY_ADVANCEMENT_THRESHOLD = 0.90
    EARLY_ADVANCEMENT_MIN_EPISODES = 50

    REGRESSION_THRESHOLDS = {
        "simplest_few_mines": 0.30,
        "simplest_with_mines": 0.30,
        "simpler": 0.30,
        "simple": 0.30,
        "medium": 0.25,
        "complex": 0.20,
        "exploration": 0.15,
        "mine_heavy": 0.15,
    }

    REGRESSION_MIN_EPISODES = 200

    def __init__(
        self,
        dataset_path: str,
        starting_stage: str = "simplest",
        performance_window: int = 50,
        mixing_ratio: float = 0.2,
        enable_auto_adjustment: bool = False,
        auto_adjustment_freq: int = 50000,
        auto_adjustment_min_threshold: float = 0.40,
    ):
        """Initialize optimized curriculum manager.

        Args:
            dataset_path: Path to curriculum dataset
            starting_stage: Initial curriculum stage
            performance_window: Size of rolling performance window
            mixing_ratio: Base mixing ratio for stage mixing
            enable_auto_adjustment: Enable automatic threshold adjustment
            auto_adjustment_freq: Frequency for automatic adjustments
            auto_adjustment_min_threshold: Minimum threshold for auto adjustment
        """
        self.dataset_path = Path(dataset_path)
        self.performance_window = performance_window
        self.base_mixing_ratio = mixing_ratio
        self.enable_auto_adjustment = enable_auto_adjustment
        self.auto_adjustment_freq = auto_adjustment_freq
        self.auto_adjustment_min_threshold = auto_adjustment_min_threshold
        self.last_auto_adjustment_step = 0

        # Validate starting stage
        if starting_stage not in CURRICULUM_ORDER:
            raise ValueError(
                f"Invalid starting stage '{starting_stage}'. Must be one of: {CURRICULUM_ORDER}"
            )

        # Current curriculum state
        self.current_stage = starting_stage
        self.current_stage_idx = CURRICULUM_ORDER.index(starting_stage)

        # Always use lazy loading for performance - no more eager loading
        self.fast_sampler = LazyLevelSampler(
            {},  # Empty metadata - will be loaded when first needed
            str(self.dataset_path),
            1000,  # Fixed cache size for performance
        )
        self.performance_tracker = CompactPerformanceTracker(performance_window)

        # Cache for adaptive mixing ratios
        self._cached_mixing_ratios = {stage: mixing_ratio for stage in CURRICULUM_ORDER}
        self._mixing_cache_valid = False

        logger.info("=" * 60)
        logger.info("Optimized Curriculum Manager Initialized")
        logger.info("=" * 60)
        logger.info(f"Starting stage: {self.current_stage}")
        logger.info("Performance optimizations enabled:")
        logger.info("  - Lazy level loading with LRU caching (1000 level cache)")
        logger.info("  - Minimal initialization overhead")
        logger.info("  - Memory-efficient performance tracking")
        logger.info("  - Cached adaptive mixing ratios")
        logger.info("=" * 60)

    def __getstate__(self):
        """Custom pickle support for CurriculumManager."""
        state = self.__dict__.copy()
        # The fast_sampler handles its own pickling if it's a LazyLevelSampler
        return state

    def __setstate__(self, state):
        """Custom unpickle support for CurriculumManager."""
        self.__dict__.update(state)
        # LazyLevelSampler will recreate its own TestSuiteLoader

    def _load_levels(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load levels with same logic as original but optimized for speed."""
        loader = TestSuiteLoader(str(self.dataset_path))
        all_levels = loader.load_all_levels()

        levels_by_stage_and_generator = {}

        for stage in CURRICULUM_ORDER:
            levels_by_stage_and_generator[stage] = {}

            if stage not in all_levels:
                raise ValueError(f"No levels found for stage '{stage}'")

            # Group by generator type efficiently
            for level_data in all_levels[stage]:
                generator_type = level_data.get("metadata", {}).get(
                    "generator", "unknown"
                )

                if generator_type not in levels_by_stage_and_generator[stage]:
                    levels_by_stage_and_generator[stage][generator_type] = []

                # Add category field for compatibility
                level_data["category"] = stage
                levels_by_stage_and_generator[stage][generator_type].append(level_data)

            if not levels_by_stage_and_generator[stage]:
                raise ValueError(
                    f"Stage '{stage}' has no levels with generator metadata"
                )

        return levels_by_stage_and_generator

    def _load_levels_metadata(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load level metadata without full level data for lazy loading.

        This method is much faster than _load_levels() as it only reads file paths
        and basic metadata without deserializing pickle files.

        Returns:
            Dictionary mapping stage to generator to list of level metadata
        """
        loader = TestSuiteLoader(str(self.dataset_path))
        all_metadata = loader.load_all_metadata()

        metadata_by_stage_and_generator = {}

        for stage in CURRICULUM_ORDER:
            metadata_by_stage_and_generator[stage] = {}

            if stage not in all_metadata:
                raise ValueError(f"No metadata found for stage '{stage}'")

            # Group metadata by generator type
            for level_metadata in all_metadata[stage]:
                # Try to get generator type from existing metadata, or extract from actual file if needed
                generator_type = level_metadata.get("metadata", {}).get(
                    "generator", "unknown"
                )

                # If generator type is unknown, we need to load one level to get the correct generator info
                # This is a minimal penalty compared to loading all levels
                if generator_type == "unknown":
                    file_path = level_metadata.get("file_path")
                    if file_path:
                        try:
                            sample_level = loader.load_single_level(file_path)
                            if sample_level and "metadata" in sample_level:
                                generator_type = sample_level["metadata"].get(
                                    "generator", "unknown"
                                )
                                # Update the metadata for future reference
                                level_metadata["metadata"]["generator"] = generator_type
                        except Exception:
                            # If we can't load the level, keep as unknown
                            generator_type = "unknown"

                if generator_type not in metadata_by_stage_and_generator[stage]:
                    metadata_by_stage_and_generator[stage][generator_type] = []

                metadata_by_stage_and_generator[stage][generator_type].append(
                    level_metadata
                )

            if not metadata_by_stage_and_generator[stage]:
                raise ValueError(
                    f"Stage '{stage}' has no levels with generator metadata"
                )

        logger.info("Level metadata loaded for lazy sampling")
        return metadata_by_stage_and_generator

    def sample_level(self) -> Optional[Dict[str, Any]]:
        """Sample a level with optimized performance (drop-in replacement for original).

        Returns:
            Level data dictionary with generator metadata
        """
        # Get cached mixing ratio to avoid recalculation
        cached_ratio = self._get_cached_mixing_ratio(self.current_stage)

        # Fast optimized sampling
        level, sample_stage, sample_generator = (
            self.fast_sampler.sample_level_optimized(
                current_stage_idx=self.current_stage_idx,
                cached_mixing_ratio=cached_ratio,
            )
        )

        # Add tracking metadata (create a shallow copy to avoid modifying original)
        level_copy = level.copy()
        level_copy["sampled_generator"] = sample_generator
        level_copy["sampled_stage"] = sample_stage

        return level_copy

    def _get_cached_mixing_ratio(self, stage: str) -> float:
        """Get cached mixing ratio, calculating if needed."""
        # Use cached value if valid and available
        if self._mixing_cache_valid and stage in self._cached_mixing_ratios:
            return self._cached_mixing_ratios[stage]

        # Calculate fresh ratio from current performance
        success_rate = self.performance_tracker.get_success_rate(stage)

        # Same adaptive logic as original
        if success_rate < 0.50:
            adaptive_ratio = 0.40
        elif success_rate < 0.65:
            adaptive_ratio = 0.25
        elif success_rate < 0.80:
            adaptive_ratio = 0.15
        else:
            adaptive_ratio = 0.05

        # Cache result
        self._cached_mixing_ratios[stage] = adaptive_ratio
        self._mixing_cache_valid = True

        return adaptive_ratio

    def record_episode(
        self,
        stage: str,
        success: bool,
        generator_type: Optional[str] = None,
        frames: Optional[int] = None,
    ):
        """Record episode result (drop-in replacement for original)."""
        self.performance_tracker.record_episode(stage, success, generator_type)

        # Invalidate mixing ratio cache when performance changes
        self._mixing_cache_valid = False

    def get_current_stage(self) -> str:
        """Get current curriculum stage name."""
        return self.current_stage

    def get_current_stage_index(self) -> int:
        """Get current curriculum stage index."""
        return self.current_stage_idx

    def get_available_stages(self) -> List[str]:
        """Get list of available curriculum stages up to current."""
        return CURRICULUM_ORDER[: self.current_stage_idx + 1]

    def get_stage_success_rate(self, stage: str) -> float:
        """Get success rate for a stage."""
        return self.performance_tracker.get_success_rate(stage)

    def get_stage_performance(self, stage: str) -> Dict[str, Any]:
        """Get performance metrics for a stage (drop-in replacement)."""
        success_rate = self.performance_tracker.get_success_rate(stage)
        episodes = self.performance_tracker.get_episode_count(stage)

        stage_threshold = self.STAGE_THRESHOLDS.get(stage, 0.7)
        stage_min_episodes = self.STAGE_MIN_EPISODES.get(stage, 100)

        if episodes == 0:
            return {
                "success_rate": 0.0,
                "episodes": 0,
                "recent_episodes": 0,
                "can_advance": False,
                "min_episodes": stage_min_episodes,
                "advancement_threshold": stage_threshold,
                "trend": 0.0,
                "can_early_advance": False,
                "trend_bonus": False,
                "adaptive_mixing_ratio": self.base_mixing_ratio,
            }

        # Calculate trend efficiently
        trend = self.performance_tracker.calculate_trend(stage)

        # Standard advancement check
        can_advance = success_rate >= stage_threshold and episodes >= stage_min_episodes

        # Early advancement check
        can_early_advance = (
            episodes >= self.EARLY_ADVANCEMENT_MIN_EPISODES
            and success_rate >= self.EARLY_ADVANCEMENT_THRESHOLD
        )

        # Trend-based advancement
        trend_bonus = False
        HARD_MINIMUM_THRESHOLD = 0.60
        if trend > 0.15 and episodes >= (stage_min_episodes * 0.9):
            if success_rate >= max(stage_threshold - 0.02, HARD_MINIMUM_THRESHOLD):
                trend_bonus = True

        return {
            "success_rate": success_rate,
            "episodes": episodes,
            "recent_episodes": min(episodes, self.performance_window),
            "can_advance": can_advance or can_early_advance or trend_bonus,
            "advancement_threshold": stage_threshold,
            "min_episodes": stage_min_episodes,
            "trend": trend,
            "can_early_advance": can_early_advance,
            "trend_bonus": trend_bonus,
            "adaptive_mixing_ratio": self._cached_mixing_ratios.get(
                stage, self.base_mixing_ratio
            ),
        }

    def check_advancement(self) -> bool:
        """Check if agent should advance to next curriculum stage."""
        if self.current_stage_idx >= len(CURRICULUM_ORDER) - 1:
            return False

        perf = self.get_stage_performance(self.current_stage)

        if perf["can_advance"]:
            prev_stage = self.current_stage

            # Advance to next stage
            self.current_stage_idx += 1
            self.current_stage = CURRICULUM_ORDER[self.current_stage_idx]

            # Log advancement
            advancement_reasons = []
            if perf.get("can_early_advance", False):
                advancement_reasons.append("Early Advancement")
            if perf.get("trend_bonus", False):
                advancement_reasons.append(f"Trend Bonus ({perf['trend']:+.2f})")
            if not advancement_reasons:
                advancement_reasons.append("Standard Advancement")

            logger.info("=" * 70)
            logger.info("✨ OPTIMIZED CURRICULUM ADVANCEMENT! ✨")
            logger.info(f"Previous: {prev_stage} → New: {self.current_stage}")
            logger.info(f"Reason: {' + '.join(advancement_reasons)}")
            logger.info(
                f"Performance: {perf['success_rate']:.1%} success, {perf['episodes']} episodes"
            )
            logger.info("=" * 70)

            return True

        return False

    def check_regression(self) -> bool:
        """Check if agent should regress to previous curriculum stage."""
        if self.current_stage_idx == 0:
            return False

        success_rate = self.performance_tracker.get_success_rate(self.current_stage)
        episodes = self.performance_tracker.get_episode_count(self.current_stage)

        if episodes < self.REGRESSION_MIN_EPISODES:
            return False

        regression_threshold = self.REGRESSION_THRESHOLDS.get(self.current_stage, 0.2)

        if success_rate < regression_threshold:
            prev_stage_idx = self.current_stage_idx - 1
            prev_stage = CURRICULUM_ORDER[prev_stage_idx]

            logger.info(
                f"Curriculum regression: {self.current_stage} ({success_rate:.1%}) → {prev_stage}"
            )

            self.current_stage_idx = prev_stage_idx
            self.current_stage = prev_stage

            # Reset performance for regressed stage (clear circular buffer)
            stage_idx = self.performance_tracker.stage_indices[self.current_stage]
            self.performance_tracker.buffer_sizes[stage_idx] = 0
            self.performance_tracker.buffer_positions[stage_idx] = 0

            return True

        return False

    def get_progress_summary(self) -> str:
        """Get human-readable progress summary (drop-in replacement)."""
        lines = [
            "## Optimized Curriculum Learning Progress\n",
            f"**Current Stage**: {self.current_stage} ({self.current_stage_idx + 1}/{len(CURRICULUM_ORDER)})\n",
        ]

        # Performance optimizations info
        lines.append(
            "**Performance Optimizations**: Fast Sampling, Compact Tracking, Cached Ratios\n"
        )

        lines.append("\n### Performance by Stage\n")

        for i, stage in enumerate(CURRICULUM_ORDER):
            perf = self.get_stage_performance(stage)

            # Status indicator
            if i < self.current_stage_idx:
                status = "✅ Completed"
            elif i == self.current_stage_idx:
                status = "🎯 Current"
            else:
                status = "🔒 Locked"

            lines.append(f"\n**{stage.title()}** - {status}")
            lines.append(f"- Success Rate: {perf['success_rate']:.1%}")
            lines.append(f"- Episodes: {perf['episodes']}/{perf['min_episodes']}")
            lines.append(f"- Threshold: {perf['advancement_threshold']:.1%}")

            if i == self.current_stage_idx:
                if perf.get("can_early_advance", False):
                    lines.append("- ⚡ **Ready for Early Advancement!**")
                elif perf.get("trend_bonus", False):
                    lines.append(
                        f"- 📈 **Trend Bonus Active** (improvement: {perf['trend']:+.2f})"
                    )
                elif perf["can_advance"]:
                    lines.append("- ✨ **Ready to Advance!**")

        return "\n".join(lines)

    # Additional methods for backward compatibility
    def get_generator_statistics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Get generator statistics (simplified for performance)."""
        stages = [stage] if stage else CURRICULUM_ORDER[: self.current_stage_idx + 1]
        stats = {}

        for st in stages:
            gen_perf = self.performance_tracker.get_generator_performance(st)
            sample_counts = self.fast_sampler.get_sample_counts(st)

            stage_stats = {"generators": {}}
            for gen_type in sample_counts.keys():
                stage_stats["generators"][gen_type] = {
                    "sample_count": sample_counts[gen_type],
                    **gen_perf.get(
                        gen_type, {"successes": 0, "failures": 0, "success_rate": 0.0}
                    ),
                }

            stats[st] = stage_stats

        return stats


# Factory function for backward compatibility
def create_curriculum_manager(
    dataset_path: str, starting_stage: str = "simplest", **kwargs
) -> CurriculumManager:
    """Create curriculum manager with validation."""
    try:
        manager = CurriculumManager(
            dataset_path=dataset_path, starting_stage=starting_stage, **kwargs
        )
        return manager
    except Exception as e:
        logger.error(f"Failed to create curriculum manager: {e}")
        raise
