"""Curriculum learning manager for progressive difficulty training.

Manages progression through test suite categories from simple to complex,
enabling the agent to learn incrementally on progressively harder levels.
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from npp_rl.evaluation.test_suite_loader import TestSuiteLoader

logger = logging.getLogger(__name__)


class CurriculumManager:
    """Manages curriculum learning progression through difficulty levels.

    The curriculum follows this progression:
    1. Very Simple (minimal chambers - most basic foundational skills)
    2. Simple (single switch levels - foundational skills)
    3. Medium (intermediate difficulty)
    4. Complex (advanced level completion)
    5. Exploratory (requires exploration strategies)
    6. Mine heavy (hardest - requires precise control and mine avoidance)

    The manager tracks performance and automatically advances to the next
    difficulty level when the agent achieves sufficient mastery.

    Features granular progression with:
    - Stage-specific advancement thresholds
    - Adaptive minimum episode requirements
    - Early advancement for high performers
    - Performance trend analysis
    - Adaptive stage mixing based on current performance
    """

    # Curriculum progression order
    CURRICULUM_ORDER = [
        "simplest",
        "simplest_with_mines",
        "simpler",
        "simple",
        "medium",
        "complex",
        "exploration",
        "mine_heavy",
    ]

    # Stage-specific advancement thresholds for granular progression
    # Progressive thresholds that decrease with difficulty to ensure forward progress
    # while maintaining competence requirements
    STAGE_THRESHOLDS = {
        "simplest": 0.80,  # Reduced from 0.80
        "simplest_with_mines": 0.75,  # Slightly lower than simplest due to mines
        "simpler": 0.60,  # Reduced from 0.70
        "simple": 0.50,  # Reduced from 0.60
        "medium": 0.45,  # Reduced from 0.55
        "complex": 0.40,  # Reduced from 0.50
        "exploration": 0.35,  # Reduced from 0.45
        "mine_heavy": 0.30,  # Reduced from 0.40
    }

    # Stage-specific minimum episodes for adaptive progression
    # Harder stages require more episodes to ensure sufficient learning
    STAGE_MIN_EPISODES = {
        "simplest": 100,  # Reduced from 200 - quick validation of basics
        "simplest_with_mines": 100,  # Same as simplest - similar difficulty with mines
        "simpler": 100,  # Reduced from 200
        "simple": 75,  # Reduced from 200 - agent got stuck here
        "medium": 100,  # Reduced from 250
        "complex": 150,  # Reduced from 300
        "exploration": 150,  # Reduced from 300
        "mine_heavy": 200,  # Reduced from 300 - hardest stage needs more
    }

    # Early advancement threshold - if agent excels, can advance sooner
    EARLY_ADVANCEMENT_THRESHOLD = 0.90
    EARLY_ADVANCEMENT_MIN_EPISODES = 50

    # Regression thresholds to prevent catastrophic forgetting
    # If performance drops too low, regress to previous stage
    REGRESSION_THRESHOLDS = {
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
        allow_stage_mixing: bool = True,
        mixing_ratio: float = 0.2,
        enable_adaptive_mixing: bool = True,
        enable_early_advancement: bool = True,
        enable_trend_analysis: bool = True,
        enable_regression: bool = True,
    ):
        """Initialize curriculum manager with granular progression settings.

        Args:
            dataset_path: Path to dataset containing level categories
            starting_stage: Initial curriculum stage (default: 'simplest')
            performance_window: Window size for performance tracking (default: 50)
            allow_stage_mixing: If True, mix in previous stage levels (default: True)
            mixing_ratio: Base ratio of previous stage levels (default: 0.2, can be adapted)
            enable_adaptive_mixing: If True, adjust mixing based on performance (default: True)
            enable_early_advancement: If True, allow fast advancement for high performers (default: True)
            enable_trend_analysis: If True, consider performance trends in decisions (default: True)
            enable_regression: If True, allow regressing to easier stages on poor performance (default: True)
        """
        self.dataset_path = Path(dataset_path)

        self.performance_window = performance_window
        self.allow_stage_mixing = allow_stage_mixing
        self.base_mixing_ratio = mixing_ratio
        self.enable_adaptive_mixing = enable_adaptive_mixing
        self.enable_early_advancement = enable_early_advancement
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_regression = enable_regression  # FIXED: Store regression flag

        # Validate starting stage
        if starting_stage not in self.CURRICULUM_ORDER:
            raise ValueError(
                f"Invalid starting stage '{starting_stage}'. "
                f"Must be one of: {self.CURRICULUM_ORDER}"
            )

        # Current curriculum state
        self.current_stage = starting_stage
        self.current_stage_idx = self.CURRICULUM_ORDER.index(starting_stage)

        # Performance tracking per stage
        self.stage_performance: Dict[str, deque] = {
            stage: deque(maxlen=performance_window) for stage in self.CURRICULUM_ORDER
        }

        self.stage_episode_counts: Dict[str, int] = {
            stage: 0 for stage in self.CURRICULUM_ORDER
        }

        # Track current adaptive mixing ratio per stage
        self.stage_mixing_ratios: Dict[str, float] = {
            stage: mixing_ratio for stage in self.CURRICULUM_ORDER
        }

        # Load level data organized by stage and generator type
        self.levels_by_stage_and_generator = self._load_levels()

        # Track sampling counts per generator type for balanced sampling
        self.generator_sample_counts: Dict[str, Dict[str, int]] = {}
        for stage, generators in self.levels_by_stage_and_generator.items():
            self.generator_sample_counts[stage] = {gen: 0 for gen in generators.keys()}

        # Track performance per generator type for granular analysis
        self.generator_performance: Dict[str, Dict[str, Dict[str, int]]] = {}
        for stage, generators in self.levels_by_stage_and_generator.items():
            self.generator_performance[stage] = {
                gen: {"successes": 0, "failures": 0} for gen in generators.keys()
            }

        logger.info("=" * 60)
        logger.info("Curriculum Manager Initialized (Granular Progression)")
        logger.info("=" * 60)
        logger.info(f"Starting stage: {self.current_stage}")
        logger.info("Adaptive features:")
        logger.info(f"  - Adaptive mixing: {enable_adaptive_mixing}")
        logger.info(f"  - Early advancement: {enable_early_advancement}")
        logger.info(f"  - Trend analysis: {enable_trend_analysis}")
        logger.info(f"Stage mixing: {'enabled' if allow_stage_mixing else 'disabled'}")

        logger.info("\nStage-specific advancement thresholds:")
        for stage in self.CURRICULUM_ORDER:
            threshold = self.STAGE_THRESHOLDS.get(stage, 0.7)
            min_eps = self.STAGE_MIN_EPISODES.get(stage, 100)
            logger.info(f"  {stage}: {threshold:.0%} success, {min_eps} episodes")

        logger.info("\nLevels per stage (by generator type):")
        for stage, generators in self.levels_by_stage_and_generator.items():
            total_levels = sum(len(levels) for levels in generators.values())
            logger.info(
                f"  {stage}: {total_levels} levels across {len(generators)} generator types"
            )
            for gen_type, levels in generators.items():
                logger.info(f"    - {gen_type}: {len(levels)} levels")
        logger.info("=" * 60)

    def _load_levels(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load levels from dataset organized by stage and generator type.

        Returns:
            Nested dictionary: {stage: {generator_type: [level_data, ...]}}
        """
        loader = TestSuiteLoader(str(self.dataset_path))
        all_levels = loader.load_all_levels()

        # Organize levels by stage and generator type
        levels_by_stage_and_generator = {}

        for stage in self.CURRICULUM_ORDER:
            levels_by_stage_and_generator[stage] = {}

            if stage not in all_levels:
                logger.warning(f"No levels found for stage '{stage}'")
                continue

            # Group levels by generator type
            for level_data in all_levels[stage]:
                # Extract generator type from metadata
                generator_type = level_data.get("metadata", {}).get(
                    "generator", "unknown"
                )

                if generator_type not in levels_by_stage_and_generator[stage]:
                    levels_by_stage_and_generator[stage][generator_type] = []

                # Add explicit category field for fallback in CurriculumEnv
                # This ensures backward compatibility if sampled_stage is missing
                level_data["category"] = stage

                levels_by_stage_and_generator[stage][generator_type].append(level_data)

            # Log warning if any stage has no generator types
            if not levels_by_stage_and_generator[stage]:
                logger.warning(f"Stage '{stage}' has no levels with generator metadata")

        return levels_by_stage_and_generator

    def get_current_stage(self) -> str:
        """Get current curriculum stage name."""
        return self.current_stage

    def get_current_stage_index(self) -> int:
        """Get current curriculum stage index."""
        return self.current_stage_idx

    def get_available_stages(self) -> List[str]:
        """Get list of all available curriculum stages up to current."""
        return self.CURRICULUM_ORDER[: self.current_stage_idx + 1]

    def _get_adaptive_mixing_ratio(self, stage: str) -> float:
        """Get adaptive mixing ratio for a stage based on current performance.

        In multi-environment setups with SubprocVecEnv:
        - Main process: Calculates ratio from current performance data
        - Subprocesses: Use cached ratio synced from main process (to avoid stale data)

        Args:
            stage: Stage name

        Returns:
            Adaptive mixing ratio (0.0 to 1.0)
        """
        if not self.enable_adaptive_mixing:
            return self.base_mixing_ratio

        # If we have a cached ratio and no/minimal performance data,
        # we're likely in a subprocess - use the synced cached value
        # This prevents using stale performance data in subprocess copies
        stage_perf_count = len(self.stage_performance.get(stage, []))
        if stage in self.stage_mixing_ratios and stage_perf_count < 5:
            # Use cached ratio (synced from main process)
            return self.stage_mixing_ratios[stage]

        # Calculate fresh ratio from current performance (main process path)
        success_rate = self.get_stage_success_rate(stage)

        # Adaptive mixing based on performance:
        # - Struggling (< 50%): 40% previous stage (more support)
        # - Learning (50-65%): 25% previous stage (moderate support)
        # - Competent (65-80%): 15% previous stage (less support)
        # - Mastering (> 80%): 5% previous stage (minimal support)

        if success_rate < 0.50:
            adaptive_ratio = 0.40  # Need more support
        elif success_rate < 0.65:
            adaptive_ratio = 0.25  # Moderate support
        elif success_rate < 0.80:
            adaptive_ratio = 0.15  # Less support
        else:
            adaptive_ratio = 0.05  # Minimal support, almost ready to advance

        # Cache for future calls and subprocess syncing
        self.stage_mixing_ratios[stage] = adaptive_ratio

        return adaptive_ratio

    def sample_level(self) -> Optional[Dict[str, Any]]:
        """Sample a level from current curriculum stage with stratified generator balancing.

        Uses stratified sampling to ensure balanced exposure across all generator types
        within each stage, promoting better generalization.

        Returns:
            Level data dictionary with generator metadata, or None if no levels available
        """
        # Determine which stage to sample from
        if self.allow_stage_mixing and self.current_stage_idx > 0:
            # Get adaptive mixing ratio for current stage
            mixing_ratio = self._get_adaptive_mixing_ratio(self.current_stage)

            # Mix current stage with previous stage using adaptive ratio
            if np.random.random() < mixing_ratio:
                # Sample from previous stage
                sample_stage_idx = self.current_stage_idx - 1
            else:
                # Sample from current stage
                sample_stage_idx = self.current_stage_idx
        else:
            # Sample from current stage only
            sample_stage_idx = self.current_stage_idx

        sample_stage = self.CURRICULUM_ORDER[sample_stage_idx]
        generators = self.levels_by_stage_and_generator.get(sample_stage, {})

        if not generators:
            logger.warning(f"No generator types available for stage '{sample_stage}'")
            return None

        # Stratified sampling: find least-sampled generator type(s)
        generator_counts = self.generator_sample_counts.get(sample_stage, {})

        if not generator_counts:
            # Fallback: uniform sampling if no counts tracked
            logger.warning(
                f"No generator counts for stage '{sample_stage}', using uniform sampling"
            )
            all_levels = []
            for gen_levels in generators.values():
                all_levels.extend(gen_levels)
            if not all_levels:
                return None
            return np.random.choice(all_levels)

        # Find minimum sample count
        min_count = min(generator_counts.values())

        # Get all generator types with minimum count (ties are broken randomly)
        least_sampled_generators = [
            gen for gen, count in generator_counts.items() if count == min_count
        ]

        # Sample uniformly from least-sampled generator types
        selected_generator = np.random.choice(least_sampled_generators)

        # Get levels for selected generator
        generator_levels = generators.get(selected_generator, [])

        if not generator_levels:
            logger.warning(
                f"No levels for generator '{selected_generator}' in stage '{sample_stage}'"
            )
            return None

        # Sample random level from selected generator
        level = np.random.choice(generator_levels)

        # Increment sample count for this generator
        self.generator_sample_counts[sample_stage][selected_generator] += 1

        # Add generator type to level data for tracking (non-invasive)
        level["sampled_generator"] = selected_generator
        level["sampled_stage"] = sample_stage

        return level

    def record_episode(
        self,
        stage: str,
        success: bool,
        generator_type: Optional[str] = None,
        frames: Optional[int] = None,
    ) -> None:
        """Record episode result for a stage and optionally for a specific generator type.

        Args:
            stage: Stage name
            success: Whether episode was successful (1) or not (0)
            generator_type: Generator type that produced this level (optional)
            frames: Number of frames elapsed during the episode (optional)
        """
        if stage not in self.stage_performance:
            logger.warning(f"Unknown stage '{stage}', ignoring episode")
            return

        logger.debug(
            f"Recording episode for stage: {stage}, generator: {generator_type}, "
            f"success: {success}, frames: {frames}"
        )
        self.stage_performance[stage].append(1 if success else 0)

        # Defensive: ensure stage exists in episode counts
        if stage not in self.stage_episode_counts:
            logger.warning(f"Stage '{stage}' not in episode counts, initializing to 0")
            self.stage_episode_counts[stage] = 0

        self.stage_episode_counts[stage] += 1

        # Track generator-specific performance if provided
        if generator_type and stage in self.generator_performance:
            if generator_type in self.generator_performance[stage]:
                if success:
                    self.generator_performance[stage][generator_type]["successes"] += 1
                else:
                    self.generator_performance[stage][generator_type]["failures"] += 1
            else:
                logger.debug(
                    f"Generator type '{generator_type}' not found in stage '{stage}' performance tracking"
                )

    def _calculate_performance_trend(self, stage: str) -> float:
        """Calculate performance improvement trend for a stage.

        Compares recent performance to earlier performance to detect improvement.

        Args:
            stage: Stage name

        Returns:
            Trend value: positive = improving, negative = declining, 0 = stable
        """
        results = self.stage_performance.get(stage, deque())

        if len(results) < 20:
            return 0.0  # Not enough data

        # Split into first half and second half
        results_list = list(results)
        mid = len(results_list) // 2
        first_half = results_list[:mid]
        second_half = results_list[mid:]

        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)

        # Trend is the difference
        trend = second_avg - first_avg

        return trend

    def get_stage_performance(self, stage: str) -> Dict[str, Any]:
        """Get performance metrics for a stage with granular progression analysis.

        Args:
            stage: Stage name

        Returns:
            Dictionary with performance metrics (always includes all keys)
        """
        results = self.stage_performance.get(stage, deque())

        stage_threshold = self.STAGE_THRESHOLDS.get(stage, 0.7)
        stage_min_episodes = self.STAGE_MIN_EPISODES.get(stage, 100)

        if not results:
            return {
                "success_rate": 0.0,
                "episodes": 0,
                "recent_episodes": 0,
                "can_advance": False,
                "min_episodes": stage_min_episodes,
                "advancement_threshold": stage_threshold,
                "trend": 0.0,
                "can_early_advance": False,
            }

        success_rate = np.mean(results)
        episodes = self.stage_episode_counts.get(stage, 0)
        recent_episodes = len(results)

        # Calculate performance trend
        trend = (
            self._calculate_performance_trend(stage)
            if self.enable_trend_analysis
            else 0.0
        )

        # Standard advancement check
        can_advance = success_rate >= stage_threshold and episodes >= stage_min_episodes

        # Early advancement check (high performers can advance sooner)
        can_early_advance = False
        if (
            self.enable_early_advancement
            and episodes >= self.EARLY_ADVANCEMENT_MIN_EPISODES
        ):
            can_early_advance = success_rate >= self.EARLY_ADVANCEMENT_THRESHOLD

        # Trend-based advancement: if showing strong improvement, can advance slightly earlier
        trend_bonus = False
        # SAFETY: Add hard minimum threshold to prevent premature advancement
        HARD_MINIMUM_THRESHOLD = 0.60  # Never advance below 60% success rate
        if (
            self.enable_trend_analysis
            and trend > 0.15
            and episodes >= (stage_min_episodes * 0.9)
        ):
            # Strong positive trend + 90% of required episodes (increased from 80% for safety)
            # Reduced from 5% to 2% margin for more conservative advancement
            if success_rate >= max(stage_threshold - 0.02, HARD_MINIMUM_THRESHOLD):
                trend_bonus = True

        return {
            "success_rate": success_rate,
            "episodes": episodes,
            "recent_episodes": recent_episodes,
            "can_advance": can_advance or can_early_advance or trend_bonus,
            "advancement_threshold": stage_threshold,
            "min_episodes": stage_min_episodes,
            "trend": trend,
            "can_early_advance": can_early_advance,
            "trend_bonus": trend_bonus,
            "adaptive_mixing_ratio": self.stage_mixing_ratios.get(
                stage, self.base_mixing_ratio
            ),
        }

    def get_stage_success_rate(self, stage: str) -> float:
        """Get success rate for a stage.

        Args:
            stage: Stage name

        Returns:
            Success rate (0.0 to 1.0)
        """
        results = self.stage_performance.get(stage, deque())
        if not results:
            return 0.0
        return float(np.mean(results))

    def check_advancement(self) -> bool:
        """Check if agent should advance to next curriculum stage.

        Uses granular progression logic including:
        - Stage-specific thresholds
        - Early advancement for high performers
        - Trend-based advancement

        Returns:
            True if advanced to next stage, False otherwise
        """
        # Check if already at final stage
        if self.current_stage_idx >= len(self.CURRICULUM_ORDER) - 1:
            logger.debug("Already at final curriculum stage")
            return False

        # Check current stage performance with all adaptive features
        perf = self.get_stage_performance(self.current_stage)

        if perf["can_advance"]:
            prev_stage = self.current_stage

            # Log advancement criteria that were met (before advancing)
            if perf.get("can_early_advance", False):
                logger.info(
                    f"[Early Advancement] Stage '{prev_stage}': {perf['success_rate']:.1%} success "
                    f"after only {perf['episodes']} episodes (threshold: {self.EARLY_ADVANCEMENT_THRESHOLD:.1%})"
                )
            if perf.get("trend_bonus", False):
                logger.info(
                    f"[Trend Bonus] Stage '{prev_stage}': Strong improvement trend ({perf['trend']:+.2f}) "
                    f"with {perf['success_rate']:.1%} success, allowing advancement"
                )

            # Advance to next stage
            self.current_stage_idx += 1
            self.current_stage = self.CURRICULUM_ORDER[self.current_stage_idx]

            # Determine advancement reason for summary
            advancement_reason = []
            if perf.get("can_early_advance", False):
                advancement_reason.append("Early Advancement (High Performance)")
            if perf.get("trend_bonus", False):
                advancement_reason.append(
                    f"Trend Bonus (Improvement: {perf['trend']:+.2f})"
                )
            if not advancement_reason:
                advancement_reason.append("Standard Advancement")

            reason_str = " + ".join(advancement_reason)

            logger.info("=" * 70)
            logger.info("✨ CURRICULUM ADVANCEMENT! ✨")
            logger.info("=" * 70)
            logger.info(f"Previous stage: {prev_stage}")
            logger.info(f"New stage: {self.current_stage}")
            logger.info(f"Reason: {reason_str}")
            logger.info("")
            logger.info("Performance Summary:")
            logger.info(f"  Success rate: {perf['success_rate']:.1%}")
            logger.info(f"  Episodes completed: {perf['episodes']}")
            logger.info(f"  Threshold: {self.STAGE_THRESHOLDS[self.current_stage]:.1%}")
            logger.info(f"  Min episodes: {perf['min_episodes']}")
            if self.enable_trend_analysis:
                logger.info(f"  Performance trend: {perf['trend']:+.2f}")
            if self.enable_adaptive_mixing:
                logger.info(
                    f"  Final mixing ratio: {perf['adaptive_mixing_ratio']:.1%}"
                )
            logger.info("=" * 70)

            return True

        return False

    def check_regression(self) -> bool:
        """Check if agent should regress to previous curriculum stage.

        Prevents catastrophic forgetting by regressing when performance drops
        too low on the current stage.

        Returns:
            True if regressed to previous stage, False otherwise
        """
        if self.current_stage_idx == 0 or not self.enable_regression:
            return False

        current_stage = self.current_stage
        results = self.stage_performance.get(current_stage, deque())

        if len(results) < self.REGRESSION_MIN_EPISODES:
            return False

        success_rate = float(np.mean(results))
        regression_threshold = self.REGRESSION_THRESHOLDS.get(current_stage, 0.2)

        if success_rate < regression_threshold:
            prev_stage_idx = self.current_stage_idx - 1
            prev_stage = self.CURRICULUM_ORDER[prev_stage_idx]

            logger.warning(
                f"Curriculum regression: {current_stage} ({success_rate:.1%}) → {prev_stage} "
                f"(threshold: {regression_threshold:.1%}, episodes: {len(results)})"
            )

            self.current_stage_idx = prev_stage_idx
            self.current_stage = prev_stage
            self.stage_performance[current_stage] = deque(
                maxlen=self.performance_window
            )

            return True

        return False

    def get_progress_summary(self) -> str:
        """Get human-readable curriculum progress summary with granular metrics.

        Returns:
            Markdown-formatted progress summary
        """
        lines = [
            "## Curriculum Learning Progress (Granular Progression)\n",
            f"**Current Stage**: {self.current_stage} "
            f"({self.current_stage_idx + 1}/{len(self.CURRICULUM_ORDER)})\n",
            "**Adaptive Features**: ",
        ]

        features = []
        if self.enable_adaptive_mixing:
            features.append("Adaptive Mixing")
        if self.enable_early_advancement:
            features.append("Early Advancement")
        if self.enable_trend_analysis:
            features.append("Trend Analysis")
        lines.append(", ".join(features) if features else "None")
        lines.append("\n")

        lines.append("\n### Performance by Stage\n")

        for i, stage in enumerate(self.CURRICULUM_ORDER):
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
                else:
                    remaining = perf["min_episodes"] - perf["episodes"]
                    if remaining > 0:
                        lines.append(f"- Episodes needed: {remaining}")
                    else:
                        gap = perf["advancement_threshold"] - perf["success_rate"]
                        lines.append(f"- Success rate gap: {gap:.1%}")

                # Show adaptive metrics for current stage
                if self.enable_adaptive_mixing and i > 0:
                    mix_ratio = perf.get(
                        "adaptive_mixing_ratio", self.base_mixing_ratio
                    )
                    lines.append(
                        f"- Adaptive Mixing: {mix_ratio:.1%} from previous stage"
                    )

                if self.enable_trend_analysis and perf["episodes"] >= 20:
                    trend = perf.get("trend", 0.0)
                    trend_indicator = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    lines.append(f"- Performance Trend: {trend_indicator} {trend:+.2f}")

        return "\n".join(lines)

    def get_generator_statistics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive statistics for generator types.

        Args:
            stage: Specific stage to get stats for, or None for all stages

        Returns:
            Dictionary with generator sampling and performance statistics
        """
        stats = {}

        stages_to_report = [stage] if stage else self.CURRICULUM_ORDER

        for st in stages_to_report:
            if st not in self.generator_sample_counts:
                continue

            stage_stats = {"generators": {}}

            for gen_type in self.generator_sample_counts[st].keys():
                sample_count = self.generator_sample_counts[st].get(gen_type, 0)
                perf = self.generator_performance[st].get(gen_type, {})
                successes = perf.get("successes", 0)
                failures = perf.get("failures", 0)
                total_episodes = successes + failures

                success_rate = successes / total_episodes if total_episodes > 0 else 0.0

                stage_stats["generators"][gen_type] = {
                    "sample_count": sample_count,
                    "episodes": total_episodes,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": success_rate,
                }

            # Calculate balance variance (0 = perfect balance)
            sample_counts = list(self.generator_sample_counts[st].values())
            if sample_counts:
                stage_stats["balance_variance"] = float(np.var(sample_counts))
                stage_stats["total_samples"] = sum(sample_counts)
            else:
                stage_stats["balance_variance"] = 0.0
                stage_stats["total_samples"] = 0

            stats[st] = stage_stats

        return stats

    def log_curriculum_metrics(
        self, writer, global_step: int, current_stage_only: bool = False
    ) -> None:
        """Log comprehensive curriculum metrics to TensorBoard.

        Logs sampling distribution, performance per generator type, and aggregate metrics.

        Args:
            writer: TensorBoard SummaryWriter
            global_step: Current training step
            current_stage_only: If True, only log metrics for current stage
        """
        if writer is None:
            return

        # Determine which stages to log
        stages_to_log = (
            [self.current_stage]
            if current_stage_only
            else self.CURRICULUM_ORDER[: self.current_stage_idx + 1]
        )

        for stage in stages_to_log:
            if stage not in self.generator_sample_counts:
                continue

            # Log sample distribution metrics
            for gen_type, count in self.generator_sample_counts[stage].items():
                # Replace colons with underscores for TensorBoard compatibility
                safe_gen_name = gen_type.replace(":", "_")
                writer.add_scalar(
                    f"curriculum/sampling/{stage}/{safe_gen_name}/count",
                    count,
                    global_step,
                )

            # Removed balance_variance - not actionable, sample counts are sufficient

            # Log performance metrics per generator
            for gen_type, perf in self.generator_performance[stage].items():
                successes = perf.get("successes", 0)
                failures = perf.get("failures", 0)
                total = successes + failures

                if total > 0:
                    success_rate = successes / total
                    safe_gen_name = gen_type.replace(":", "_")

                    writer.add_scalar(
                        f"curriculum/performance/{stage}/{safe_gen_name}/success_rate",
                        success_rate,
                        global_step,
                    )
                    writer.add_scalar(
                        f"curriculum/performance/{stage}/{safe_gen_name}/episodes",
                        total,
                        global_step,
                    )

            # Log overall stage success rate
            stage_perf = self.get_stage_performance(stage)
            writer.add_scalar(
                f"curriculum/performance/{stage}/overall_success_rate",
                stage_perf["success_rate"],
                global_step,
            )

        # Log current stage indicator
        writer.add_scalar(
            "curriculum/current_stage_idx", self.current_stage_idx, global_step
        )

        # Log adaptive mixing ratio if enabled
        if self.enable_adaptive_mixing and self.current_stage_idx > 0:
            mixing_ratio = self._get_adaptive_mixing_ratio(self.current_stage)
            writer.add_scalar(
                f"curriculum/mixing/{self.current_stage}/ratio",
                mixing_ratio,
                global_step,
            )

    def save_state(self, path: Path) -> None:
        """Save curriculum state to file including adaptive features and generator tracking.

        Args:
            path: Path to save state JSON
        """
        state = {
            "current_stage": self.current_stage,
            "current_stage_idx": self.current_stage_idx,
            "stage_episode_counts": self.stage_episode_counts,
            "stage_performance": {
                stage: list(perf) for stage, perf in self.stage_performance.items()
            },
            "stage_mixing_ratios": self.stage_mixing_ratios,
            "generator_sample_counts": self.generator_sample_counts,
            "generator_performance": self.generator_performance,
            "adaptive_features": {
                "enable_adaptive_mixing": self.enable_adaptive_mixing,
                "enable_early_advancement": self.enable_early_advancement,
                "enable_trend_analysis": self.enable_trend_analysis,
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved curriculum state to {path}")

    def load_state(self, path: Path) -> None:
        """Load curriculum state from file including adaptive features and generator tracking.

        Args:
            path: Path to state JSON
        """
        with open(path, "r") as f:
            state = json.load(f)

        self.current_stage = state["current_stage"]
        self.current_stage_idx = state["current_stage_idx"]
        self.stage_episode_counts = state["stage_episode_counts"]

        # Restore performance tracking
        for stage, perf_list in state["stage_performance"].items():
            self.stage_performance[stage] = deque(
                perf_list, maxlen=self.performance_window
            )

        # Restore adaptive mixing ratios if available
        if "stage_mixing_ratios" in state:
            self.stage_mixing_ratios = state["stage_mixing_ratios"]

        # Restore generator tracking data if available
        if "generator_sample_counts" in state:
            self.generator_sample_counts = state["generator_sample_counts"]

        if "generator_performance" in state:
            self.generator_performance = state["generator_performance"]

        logger.info(f"Loaded curriculum state from {path}")
        logger.info(f"Resumed at stage: {self.current_stage}")


def create_curriculum_manager(
    dataset_path: str,
    starting_stage: str = "simplest",
    **kwargs,
) -> CurriculumManager:
    """Create curriculum manager with validation.

    Args:
        dataset_path: Path to dataset
        starting_stage: Initial stage (default: 'simplest')
        **kwargs: Additional curriculum manager arguments

    Returns:
        CurriculumManager instance
    """
    try:
        manager = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage=starting_stage,
            **kwargs,
        )
        return manager
    except Exception as e:
        logger.error(f"Failed to create curriculum manager: {e}")
        raise
