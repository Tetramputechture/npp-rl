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
    """

    # Curriculum progression order
    CURRICULUM_ORDER = [
        "very_simple",
        "simple",
        "medium",
        "complex",
        "exploration",
        "mine_heavy",
    ]

    def __init__(
        self,
        dataset_path: str,
        starting_stage: str = "very_simple",
        advancement_threshold: float = 0.7,
        min_episodes_per_stage: int = 100,
        performance_window: int = 50,
        allow_stage_mixing: bool = True,
        mixing_ratio: float = 0.2,
    ):
        """Initialize curriculum manager.

        Args:
            dataset_path: Path to dataset containing level categories
            starting_stage: Initial curriculum stage (default: 'very_simple')
            advancement_threshold: Success rate needed to advance (default: 0.7)
            min_episodes_per_stage: Minimum episodes before advancing
            performance_window: Window size for performance tracking
            allow_stage_mixing: If True, mix in previous stage levels
            mixing_ratio: Ratio of previous stage levels when mixing (0.2 = 20%)
        """
        self.dataset_path = Path(dataset_path)
        self.advancement_threshold = advancement_threshold
        self.min_episodes_per_stage = min_episodes_per_stage
        self.performance_window = performance_window
        self.allow_stage_mixing = allow_stage_mixing
        self.mixing_ratio = mixing_ratio

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

        # Load level data
        self.levels_by_stage = self._load_levels()

        logger.info("Curriculum Manager initialized")
        logger.info(f"Starting stage: {self.current_stage}")
        logger.info(f"Advancement threshold: {self.advancement_threshold:.1%}")
        logger.info(f"Min episodes per stage: {self.min_episodes_per_stage}")
        logger.info(f"Stage mixing: {'enabled' if allow_stage_mixing else 'disabled'}")

        for stage, levels in self.levels_by_stage.items():
            logger.info(f"  {stage}: {len(levels)} levels")

    def _load_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load levels from dataset organized by stage.

        Returns:
            Dictionary mapping stage name to list of level data
        """
        loader = TestSuiteLoader(str(self.dataset_path))
        all_levels = loader.load_all_levels()

        # Map to curriculum order
        levels_by_stage = {}

        for stage in self.CURRICULUM_ORDER:
            if stage in all_levels:
                levels_by_stage[stage] = all_levels[stage]
            else:
                logger.warning(f"No levels found for stage '{stage}'")
                levels_by_stage[stage] = []

        return levels_by_stage

    def get_current_stage(self) -> str:
        """Get current curriculum stage name."""
        return self.current_stage

    def get_current_stage_index(self) -> int:
        """Get current curriculum stage index."""
        return self.current_stage_idx

    def get_available_stages(self) -> List[str]:
        """Get list of all available curriculum stages up to current."""
        return self.CURRICULUM_ORDER[: self.current_stage_idx + 1]

    def sample_level(self) -> Optional[Dict[str, Any]]:
        """Sample a level from current curriculum stage with optional mixing.

        Returns:
            Level data dictionary, or None if no levels available
        """
        # Determine which stage to sample from
        if self.allow_stage_mixing and self.current_stage_idx > 0:
            # Mix current stage with previous stage
            if np.random.random() < self.mixing_ratio:
                # Sample from previous stage
                sample_stage_idx = self.current_stage_idx - 1
            else:
                # Sample from current stage
                sample_stage_idx = self.current_stage_idx
        else:
            # Sample from current stage only
            sample_stage_idx = self.current_stage_idx

        sample_stage = self.CURRICULUM_ORDER[sample_stage_idx]
        levels = self.levels_by_stage.get(sample_stage, [])

        if not levels:
            logger.warning(f"No levels available for stage '{sample_stage}'")
            return None

        # Sample random level from stage
        return np.random.choice(levels)

    def record_episode(self, stage: str, success: bool) -> None:
        """Record episode result for a stage.

        Args:
            stage: Stage name
            success: Whether episode was successful (1) or not (0)
        """
        if stage not in self.stage_performance:
            logger.warning(f"Unknown stage '{stage}', ignoring episode")
            return

        self.stage_performance[stage].append(1 if success else 0)
        self.stage_episode_counts[stage] += 1

    def get_stage_performance(self, stage: str) -> Dict[str, float]:
        """Get performance metrics for a stage.

        Args:
            stage: Stage name

        Returns:
            Dictionary with performance metrics
        """
        results = self.stage_performance.get(stage, deque())

        if not results:
            return {"success_rate": 0.0, "episodes": 0, "can_advance": False}

        success_rate = np.mean(results)
        episodes = self.stage_episode_counts[stage]

        can_advance = (
            success_rate >= self.advancement_threshold
            and episodes >= self.min_episodes_per_stage
        )

        return {
            "success_rate": success_rate,
            "episodes": episodes,
            "can_advance": can_advance,
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

        Returns:
            True if advanced to next stage, False otherwise
        """
        # Check if already at final stage
        if self.current_stage_idx >= len(self.CURRICULUM_ORDER) - 1:
            return False

        # Check current stage performance
        perf = self.get_stage_performance(self.current_stage)

        if perf["can_advance"]:
            # Advance to next stage
            self.current_stage_idx += 1
            self.current_stage = self.CURRICULUM_ORDER[self.current_stage_idx]

            logger.info("=" * 60)
            logger.info("CURRICULUM ADVANCEMENT!")
            logger.info(f"Advanced to stage: {self.current_stage}")
            logger.info(
                f"Previous stage performance: {perf['success_rate']:.1%} "
                f"over {perf['episodes']} episodes"
            )
            logger.info("=" * 60)

            return True

        return False

    def get_progress_summary(self) -> str:
        """Get human-readable curriculum progress summary.

        Returns:
            Markdown-formatted progress summary
        """
        lines = [
            "## Curriculum Learning Progress\n",
            f"**Current Stage**: {self.current_stage} "
            f"({self.current_stage_idx + 1}/{len(self.CURRICULUM_ORDER)})\n",
            f"**Stage Mixing**: {'Enabled' if self.allow_stage_mixing else 'Disabled'}\n",
            "\n### Performance by Stage\n",
        ]

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
            lines.append(f"- Episodes: {perf['episodes']}")

            if i == self.current_stage_idx and not perf["can_advance"]:
                remaining = self.min_episodes_per_stage - perf["episodes"]
                if remaining > 0:
                    lines.append(f"- Episodes to advance: {remaining}")
                else:
                    target_rate = self.advancement_threshold
                    lines.append(f"- Need {target_rate:.1%} success rate to advance")

        return "\n".join(lines)

    def save_state(self, path: Path) -> None:
        """Save curriculum state to file.

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
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved curriculum state to {path}")

    def load_state(self, path: Path) -> None:
        """Load curriculum state from file.

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

        logger.info(f"Loaded curriculum state from {path}")
        logger.info(f"Resumed at stage: {self.current_stage}")


def create_curriculum_manager(
    dataset_path: str,
    starting_stage: str = "very_simple",
    advancement_threshold: float = 0.7,
    **kwargs,
) -> CurriculumManager:
    """Create curriculum manager with validation.

    Args:
        dataset_path: Path to dataset
        starting_stage: Initial stage (default: 'very_simple')
        advancement_threshold: Success rate needed to advance
        **kwargs: Additional curriculum manager arguments

    Returns:
        CurriculumManager instance
    """
    try:
        manager = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage=starting_stage,
            advancement_threshold=advancement_threshold,
            **kwargs,
        )
        return manager
    except Exception as e:
        logger.error(f"Failed to create curriculum manager: {e}")
        raise
