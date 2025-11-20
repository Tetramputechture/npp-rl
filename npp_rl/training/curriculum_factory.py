"""Factory functions for creating curriculum learning components.

This module provides factory functions for creating curriculum managers
with different implementation strategies for various use cases.
"""

from typing import Optional, Any, Tuple

from npp_rl.training.curriculum_manager import CurriculumManager
from npp_rl.training.curriculum_components import ModularCurriculumManager
from npp_rl.training.curriculum_shared_memory import (
    SharedPerformanceBuffer,
    create_shared_curriculum_manager,
)


class DeferredCurriculumManager:
    """A placeholder curriculum manager that creates the real one only in worker processes."""

    # Class-level curriculum order constant
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

    def __init__(self, dataset_path: str, **kwargs):
        """Store parameters for later initialization."""
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self._real_manager = None

    def _ensure_real_manager(self):
        """Create the real curriculum manager on first use."""
        if self._real_manager is None:
            self._real_manager = CurriculumManager(
                dataset_path=self.dataset_path, **self.kwargs
            )

    @property
    def current_stage_idx(self):
        """Get current stage index from real manager."""
        self._ensure_real_manager()
        return self._real_manager.current_stage_idx

    @current_stage_idx.setter
    def current_stage_idx(self, value):
        """Set current stage index on real manager."""
        self._ensure_real_manager()
        self._real_manager.current_stage_idx = value

    @property
    def current_stage(self):
        """Get current stage from real manager."""
        self._ensure_real_manager()
        return self._real_manager.current_stage

    @current_stage.setter
    def current_stage(self, value):
        """Set current stage on real manager."""
        self._ensure_real_manager()
        self._real_manager.current_stage = value

    def __getattr__(self, name):
        """Delegate all other attribute and method access to the real manager."""
        self._ensure_real_manager()
        return getattr(self._real_manager, name)

    def __getstate__(self):
        """Pickle only the parameters, not the real manager."""
        return {
            "dataset_path": self.dataset_path,
            "kwargs": self.kwargs,
            "_real_manager": None,  # Don't pickle the real manager
        }

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)


class DeferredModularCurriculumManager:
    """A placeholder modular curriculum manager that creates the real one only in worker processes."""

    # Class-level curriculum order constant
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

    def __init__(self, dataset_path: str, **kwargs):
        """Store parameters for later initialization."""
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self._real_manager = None

    def _ensure_real_manager(self):
        """Create the real curriculum manager on first use."""
        if self._real_manager is None:
            self._real_manager = ModularCurriculumManager(
                dataset_path=self.dataset_path, **self.kwargs
            )

    @property
    def current_stage_idx(self):
        """Get current stage index from real manager."""
        self._ensure_real_manager()
        return self._real_manager.current_stage_idx

    @current_stage_idx.setter
    def current_stage_idx(self, value):
        """Set current stage index on real manager."""
        self._ensure_real_manager()
        self._real_manager.current_stage_idx = value

    @property
    def current_stage(self):
        """Get current stage from real manager."""
        self._ensure_real_manager()
        return self._real_manager.current_stage

    @current_stage.setter
    def current_stage(self, value):
        """Set current stage on real manager."""
        self._ensure_real_manager()
        self._real_manager.current_stage = value

    def __getattr__(self, name):
        """Delegate all other attribute and method access to the real manager."""
        self._ensure_real_manager()
        return getattr(self._real_manager, name)

    def __getstate__(self):
        """Pickle only the parameters, not the real manager."""
        return {
            "dataset_path": self.dataset_path,
            "kwargs": self.kwargs,
            "_real_manager": None,  # Don't pickle the real manager
        }

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)


class DeferredSharedMemoryCurriculumManager:
    """A placeholder shared memory curriculum manager that creates the real one only in worker processes."""

    # Class-level curriculum order constant
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

    def __init__(
        self,
        starting_stage,
        performance_window,
        dataset_path,
        **kwargs,
    ):
        """Store parameters for later initialization."""
        self.starting_stage = starting_stage
        self.performance_window = performance_window
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self._real_manager = None
        self._shared_buffer = None

    def _ensure_real_manager(self):
        """Create the real curriculum manager on first use."""
        if self._real_manager is None:
            from npp_rl.training.curriculum_shared_memory import (
                create_shared_curriculum_manager,
            )

            self._real_manager, self._shared_buffer = create_shared_curriculum_manager(
                starting_stage=self.starting_stage,
                performance_window=self.performance_window,
                dataset_path=self.dataset_path,
                **self.kwargs,
            )

    @property
    def current_stage_idx(self):
        """Get current stage index from real manager (compatibility property)."""
        self._ensure_real_manager()
        # SharedMemoryCurriculumManager uses get_current_stage_index() method
        return self._real_manager.get_current_stage_index()

    @current_stage_idx.setter
    def current_stage_idx(self, value):
        """Set current stage index on real manager (compatibility property)."""
        self._ensure_real_manager()
        # SharedMemoryCurriculumManager uses shared buffer to set stage by name
        if 0 <= value < len(self.CURRICULUM_ORDER):
            stage_name = self.CURRICULUM_ORDER[value]
            self._real_manager.shared_buffer.set_current_stage(stage_name)

    @property
    def current_stage(self):
        """Get current stage from real manager."""
        self._ensure_real_manager()
        return self._real_manager.get_current_stage()

    @current_stage.setter
    def current_stage(self, value):
        """Set current stage on real manager."""
        self._ensure_real_manager()
        self._real_manager.shared_buffer.set_current_stage(value)

    def __getattr__(self, name):
        """Delegate all other attribute and method access to the real manager."""
        self._ensure_real_manager()
        return getattr(self._real_manager, name)

    def __getstate__(self):
        """Pickle only the parameters, not the real manager."""
        return {
            "starting_stage": self.starting_stage,
            "performance_window": self.performance_window,
            "dataset_path": self.dataset_path,
            "kwargs": self.kwargs,
            "_real_manager": None,  # Don't pickle the real manager
            "_shared_buffer": None,
        }

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)


def create_curriculum_manager(
    dataset_path: str,
    starting_stage: str = "simplest",
    performance_window: int = 50,
    mixing_ratio: float = 0.2,
    enable_auto_adjustment: bool = False,
    auto_adjustment_freq: int = 50000,
    auto_adjustment_min_threshold: float = 0.40,
    implementation: str = "standard",
    **kwargs,
):
    """Create curriculum manager with specified implementation.

    This factory function creates curriculum managers with different implementation
    strategies optimized for various use cases.

    Args:
        dataset_path: Path to curriculum dataset
        starting_stage: Initial curriculum stage
        performance_window: Size of rolling performance window
        mixing_ratio: Base mixing ratio for stage mixing
        enable_auto_adjustment: Enable automatic threshold adjustment
        auto_adjustment_freq: Frequency for automatic adjustments
        auto_adjustment_min_threshold: Minimum threshold for auto adjustment
        implementation: Implementation to use ("standard", "modular", "shared_memory")
        **kwargs: Additional arguments passed to the curriculum manager

    Returns:
        Curriculum manager instance
    """

    if implementation == "standard":
        # Use deferred initialization to avoid pickling issues
        return DeferredCurriculumManager(
            dataset_path=dataset_path,
            starting_stage=starting_stage,
            performance_window=performance_window,
            mixing_ratio=mixing_ratio,
            enable_auto_adjustment=enable_auto_adjustment,
            auto_adjustment_freq=auto_adjustment_freq,
            auto_adjustment_min_threshold=auto_adjustment_min_threshold,
            **kwargs,
        )

    elif implementation == "modular":
        # Use deferred initialization to avoid pickling issues
        return DeferredModularCurriculumManager(
            dataset_path=dataset_path,
            starting_stage=starting_stage,
            performance_window=performance_window,
            mixing_ratio=mixing_ratio,
            **kwargs,
        )

    elif implementation == "shared_memory":
        manager, _ = create_shared_curriculum_manager(
            starting_stage=starting_stage,
            performance_window=performance_window,
            dataset_path=dataset_path,
        )
        return manager

    else:
        raise ValueError(
            f"Unknown implementation '{implementation}'. "
            f"Choose from: 'standard', 'modular', 'shared_memory'"
        )


def create_curriculum_for_parallel_training(
    dataset_path: str,
    starting_stage: str = "simplest",
    performance_window: int = 50,
    num_parallel_envs: int = 4,
    **kwargs,
) -> Tuple[Any, Optional[SharedPerformanceBuffer]]:
    """Create curriculum manager for parallel training environments.

    Automatically selects the best implementation based on the number of
    parallel environments and training requirements.

    Args:
        dataset_path: Path to curriculum dataset
        starting_stage: Initial curriculum stage
        performance_window: Size of rolling performance window
        num_parallel_envs: Number of parallel environments
        **kwargs: Additional arguments

    Returns:
        Tuple of (curriculum_manager, shared_buffer_if_applicable)
    """

    if num_parallel_envs >= 32:
        # Use deferred shared memory for very high parallelism (32+ environments)
        manager = DeferredSharedMemoryCurriculumManager(
            starting_stage=starting_stage,
            performance_window=performance_window,
            dataset_path=dataset_path,
        )
        return manager, None  # shared_buffer is created lazily

    else:
        # Use deferred manager for all other cases to avoid pickling issues
        manager = DeferredCurriculumManager(
            dataset_path=dataset_path,
            starting_stage=starting_stage,
            performance_window=performance_window,
            **kwargs,
        )
        return manager, None


# Convenience aliases
def create_modular_curriculum_manager(**kwargs):
    """Create modular curriculum manager."""
    return create_curriculum_manager(implementation="modular", **kwargs)


def create_shared_memory_curriculum_manager(**kwargs):
    """Create shared memory curriculum manager."""
    return create_curriculum_manager(implementation="shared_memory", **kwargs)
