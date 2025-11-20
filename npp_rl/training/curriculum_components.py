"""Modular curriculum learning components.

This module splits the monolithic CurriculumManager into focused,
maintainable components under 500 lines each, following the project's
coding standards.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from npp_rl.evaluation.test_suite_loader import TestSuiteLoader

logger = logging.getLogger(__name__)

CURRICULUM_ORDER = [
    "simplest",
    "simplest_few_mines",
    "simplest_with_mines",
    "simpler",
    "simple",
    "medium",
    "complex",
    "exploration",
    "mine_heavy",
]


class CurriculumSampler:
    """Fast level sampling with precomputed data structures (< 300 lines).

    Handles the hot path of curriculum sampling with optimized data structures
    to minimize runtime overhead during environment resets.
    """

    def __init__(
        self, levels_by_stage_and_generator: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ):
        """Initialize fast sampler with precomputed data structures."""
        # Precompute sampling arrays for O(1) access
        self.stage_generators = {}  # stage -> list of generator names
        self.generator_levels = {}  # (stage, generator) -> numpy array of levels
        self.stage_indices = {stage: i for i, stage in enumerate(CURRICULUM_ORDER)}

        # Initialize sample counts as numpy arrays for fast operations
        self.sample_count_arrays = {}  # stage -> numpy array of counts

        self._precompute_sampling_arrays(levels_by_stage_and_generator)
        self._initialize_sample_counts()

    def _precompute_sampling_arrays(
        self, levels_by_stage_and_generator: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ):
        """Precompute optimized data structures for fast sampling."""
        for stage, generators in levels_by_stage_and_generator.items():
            if stage not in self.stage_indices:
                continue

            # Store generator names as list for indexing
            generator_names = list(generators.keys())
            self.stage_generators[stage] = generator_names

            # Convert level lists to numpy arrays for fast random access
            for gen_name, levels in generators.items():
                key = (stage, gen_name)
                self.generator_levels[key] = np.array(levels, dtype=object)

    def _initialize_sample_counts(self):
        """Initialize sample count arrays for fast stratified sampling."""
        for stage, generators in self.stage_generators.items():
            n_generators = len(generators)
            self.sample_count_arrays[stage] = np.zeros(n_generators, dtype=np.int32)

    def sample_level_optimized(
        self,
        current_stage_idx: int,
        cached_mixing_ratio: float = 0.2,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Fast optimized level sampling."""
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

        return level, sample_stage, selected_generator

    def get_sample_counts(self, stage: str) -> Dict[str, int]:
        """Get current sample counts for a stage."""
        if stage not in self.sample_count_arrays:
            return {}

        count_array = self.sample_count_arrays[stage]
        generator_names = self.stage_generators[stage]

        return {gen: int(count_array[i]) for i, gen in enumerate(generator_names)}


class CompactPerformanceTracker:
    """Memory-efficient performance tracking (< 400 lines).

    Consolidates performance tracking to reduce memory overhead and improve
    cache efficiency using numpy arrays and circular buffers.
    """

    def __init__(self, performance_window: int = 50):
        """Initialize compact performance tracker."""
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
        """Record an episode result efficiently."""
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
        """Get success rate for a stage efficiently."""
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
        """Calculate performance improvement trend for a stage."""
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


class ProgressionManager:
    """Curriculum advancement and regression logic (< 300 lines).

    Handles stage transitions, threshold checking, and progression decisions
    with support for early advancement and trend analysis.
    """

    # Stage-specific advancement thresholds
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

    EARLY_ADVANCEMENT_THRESHOLD = 0.90
    EARLY_ADVANCEMENT_MIN_EPISODES = 50
    REGRESSION_MIN_EPISODES = 200

    def get_stage_performance_metrics(
        self,
        stage: str,
        performance_tracker: CompactPerformanceTracker,
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a stage."""
        success_rate = performance_tracker.get_success_rate(stage)
        episodes = performance_tracker.get_episode_count(stage)

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
            }

        # Calculate trend efficiently
        trend = performance_tracker.calculate_trend(stage)

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
            "recent_episodes": min(episodes, performance_tracker.performance_window),
            "can_advance": can_advance or can_early_advance or trend_bonus,
            "advancement_threshold": stage_threshold,
            "min_episodes": stage_min_episodes,
            "trend": trend,
            "can_early_advance": can_early_advance,
            "trend_bonus": trend_bonus,
        }

    def check_advancement(
        self, current_stage_idx: int, performance_tracker: CompactPerformanceTracker
    ) -> Tuple[bool, Optional[str]]:
        """Check if advancement should occur.

        Returns:
            Tuple of (should_advance, advancement_reason)
        """
        if current_stage_idx >= len(self.curriculum_order) - 1:
            return False, None

        current_stage = self.curriculum_order[current_stage_idx]
        perf = self.get_stage_performance_metrics(current_stage, performance_tracker)

        if perf["can_advance"]:
            # Determine advancement reason
            reasons = []
            if perf.get("can_early_advance", False):
                reasons.append("Early Advancement")
            if perf.get("trend_bonus", False):
                reasons.append(f"Trend Bonus ({perf['trend']:+.2f})")
            if not reasons:
                reasons.append("Standard Advancement")

            return True, " + ".join(reasons)

        return False, None

    def check_regression(
        self, current_stage_idx: int, performance_tracker: CompactPerformanceTracker
    ) -> bool:
        """Check if regression should occur."""
        if current_stage_idx == 0:
            return False

        current_stage = self.curriculum_order[current_stage_idx]
        success_rate = performance_tracker.get_success_rate(current_stage)
        episodes = performance_tracker.get_episode_count(current_stage)

        if episodes < self.REGRESSION_MIN_EPISODES:
            return False

        regression_threshold = self.REGRESSION_THRESHOLDS.get(current_stage, 0.2)

        return success_rate < regression_threshold


class ModularCurriculumManager:
    """Modular curriculum manager coordinating focused components (< 500 lines).

    This is the main facade that coordinates the specialized components
    while maintaining backward compatibility with the original interface.
    """

    def __init__(
        self,
        dataset_path: str,
        starting_stage: str = "simplest",
        performance_window: int = 50,
        mixing_ratio: float = 0.2,
        **kwargs,
    ):
        """Initialize modular curriculum manager."""
        self.dataset_path = Path(dataset_path)
        self.performance_window = performance_window
        self.base_mixing_ratio = mixing_ratio

        # Validate starting stage
        if starting_stage not in CURRICULUM_ORDER:
            raise ValueError(
                f"Invalid starting stage '{starting_stage}'. Must be one of: {CURRICULUM_ORDER}"
            )

        # Current curriculum state
        self.current_stage = starting_stage
        self.current_stage_idx = CURRICULUM_ORDER.index(starting_stage)

        # Always use lazy loading for performance - no more eager loading
        from npp_rl.training.curriculum_manager import LazyLevelSampler

        self.sampler = LazyLevelSampler(
            {},  # Empty metadata - will be loaded when first needed
            str(self.dataset_path),
            1000,  # Fixed cache size for performance
        )
        self.performance_tracker = CompactPerformanceTracker(performance_window)
        self.progression_manager = ProgressionManager()

        # Cache for adaptive mixing ratios
        self._cached_mixing_ratios = {stage: mixing_ratio for stage in CURRICULUM_ORDER}
        self._mixing_cache_valid = False

        logger.info("Modular Curriculum Manager Initialized")
        logger.info(f"Starting stage: {self.current_stage}")
        logger.info("Components: LazyLevelSampler, CompactTracker, ProgressionManager")
        logger.info("Lazy loading enabled with 1000 level LRU cache")

    def _load_levels(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load levels from dataset."""
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
        """Load level metadata without full level data for lazy loading."""
        loader = TestSuiteLoader(str(self.dataset_path))
        all_metadata = loader.load_all_metadata()

        metadata_by_stage_and_generator = {}

        for stage in CURRICULUM_ORDER:
            metadata_by_stage_and_generator[stage] = {}

            if stage not in all_metadata:
                raise ValueError(f"No metadata found for stage '{stage}'")

            # Group metadata by generator type
            for level_metadata in all_metadata[stage]:
                generator_type = level_metadata.get("metadata", {}).get(
                    "generator", "unknown"
                )

                # If generator type is unknown, load one level to get correct generator info
                if generator_type == "unknown":
                    file_path = level_metadata.get("file_path")
                    if file_path:
                        try:
                            sample_level = loader.load_single_level(file_path)
                            if sample_level and "metadata" in sample_level:
                                generator_type = sample_level["metadata"].get(
                                    "generator", "unknown"
                                )
                                level_metadata["metadata"]["generator"] = generator_type
                        except Exception:
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

        logger.info("Level metadata loaded for modular lazy sampling")
        return metadata_by_stage_and_generator

    def sample_level(self) -> Optional[Dict[str, Any]]:
        """Sample a level with optimized performance."""
        # Get cached mixing ratio to avoid recalculation
        cached_ratio = self._get_cached_mixing_ratio(self.current_stage)

        # Fast optimized sampling via sampler component
        level, sample_stage, sample_generator = self.sampler.sample_level_optimized(
            current_stage_idx=self.current_stage_idx,
            cached_mixing_ratio=cached_ratio,
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

        # Adaptive mixing logic
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
        """Record episode result."""
        self.performance_tracker.record_episode(stage, success, generator_type)

        # Invalidate mixing ratio cache when performance changes
        self._mixing_cache_valid = False

    def check_advancement(self) -> bool:
        """Check if agent should advance to next curriculum stage."""
        should_advance, reason = self.progression_manager.check_advancement(
            self.current_stage_idx, self.performance_tracker
        )

        if should_advance:
            prev_stage = self.current_stage
            self.current_stage_idx += 1
            self.current_stage = CURRICULUM_ORDER[self.current_stage_idx]

            logger.info("=" * 70)
            logger.info("✨ MODULAR CURRICULUM ADVANCEMENT! ✨")
            logger.info(f"Previous: {prev_stage} → New: {self.current_stage}")
            logger.info(f"Reason: {reason}")
            logger.info("=" * 70)

            return True

        return False

    def check_regression(self) -> bool:
        """Check if agent should regress to previous curriculum stage."""
        should_regress = self.progression_manager.check_regression(
            self.current_stage_idx, self.performance_tracker
        )

        if should_regress:
            prev_stage = self.current_stage
            self.current_stage_idx -= 1
            self.current_stage = CURRICULUM_ORDER[self.current_stage_idx]

            logger.info(
                f"Modular curriculum regression: {prev_stage} → {self.current_stage}"
            )

            # Reset performance for regressed stage
            stage_idx = self.performance_tracker.stage_indices[self.current_stage]
            self.performance_tracker.buffer_sizes[stage_idx] = 0
            self.performance_tracker.buffer_positions[stage_idx] = 0

            return True

        return False

    # Backward compatibility methods
    def get_current_stage(self) -> str:
        return self.current_stage

    def get_current_stage_index(self) -> int:
        return self.current_stage_idx

    def get_available_stages(self) -> List[str]:
        return CURRICULUM_ORDER[: self.current_stage_idx + 1]

    def get_stage_success_rate(self, stage: str) -> float:
        return self.performance_tracker.get_success_rate(stage)

    def get_stage_performance(self, stage: str) -> Dict[str, Any]:
        """Get performance metrics for a stage."""
        perf = self.progression_manager.get_stage_performance_metrics(
            stage, self.performance_tracker
        )
        perf["adaptive_mixing_ratio"] = self._cached_mixing_ratios.get(
            stage, self.base_mixing_ratio
        )
        return perf

    def get_generator_statistics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Get generator statistics."""
        stages = [stage] if stage else CURRICULUM_ORDER[: self.current_stage_idx + 1]
        stats = {}

        for st in stages:
            gen_perf = self.performance_tracker.get_generator_performance(st)
            sample_counts = self.sampler.get_sample_counts(st)

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
