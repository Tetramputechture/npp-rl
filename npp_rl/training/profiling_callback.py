"""Callback for runtime profiling during training."""

import logging

from stable_baselines3.common.callbacks import BaseCallback

from npp_rl.training.runtime_profiler import RuntimeProfiler

logger = logging.getLogger(__name__)


class RuntimeProfilingCallback(BaseCallback):
    """Callback that tracks detailed timing during training loop.

    Integrates with RuntimeProfiler to track:
    - Model forward/backward passes
    - Environment steps
    - Data collection
    - Policy updates
    """

    def __init__(
        self,
        profiler: RuntimeProfiler,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """Initialize profiling callback.

        Args:
            profiler: RuntimeProfiler instance
            log_freq: Frequency to log profiling updates
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.profiler = profiler
        self.log_freq = log_freq

        # Track step timing
        self.step_start_time = None
        self.last_log_step = 0
        self.rollout_start_time = None

    def _on_training_start(self) -> None:
        """Called when training starts."""
        logger.info("Runtime profiling callback initialized")
        if self.profiler and self.profiler.enable_pytorch_profiler:
            self.profiler.start_pytorch_profiler("training_loop")

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.profiler:
            if self.profiler.enable_pytorch_profiler:
                self.profiler.stop_pytorch_profiler()
            # Save profiling data when training ends (may be incomplete)
            try:
                self.profiler.save_summary(force=True)
            except Exception as e:
                logger.error(f"Failed to save profiling data in callback: {e}")
        logger.info("Runtime profiling callback finished")

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Step PyTorch profiler if enabled
        if self.profiler and self.profiler.enable_pytorch_profiler:
            self.profiler.step_pytorch_profiler()

        # Log periodic updates
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self.last_log_step = self.num_timesteps
            if self.verbose >= 1 and self.profiler:
                logger.debug(
                    f"Profiling: step={self.num_timesteps}, "
                    f"components_tracked={len(self.profiler.component_timings)}"
                )

        return True

    def _on_rollout_start(self) -> None:
        """Called when rollout collection starts."""
        if self.profiler:
            import time
            self.rollout_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        if self.profiler and self.rollout_start_time is not None:
            import time
            duration = time.time() - self.rollout_start_time
            # Manually add timing since we can't wrap the entire rollout in a context manager
            if "rollout_collection" not in self.profiler.component_timings:
                from npp_rl.training.runtime_profiler import ComponentTiming
                self.profiler.component_timings["rollout_collection"] = ComponentTiming(
                    name="rollout_collection"
                )
            self.profiler.component_timings["rollout_collection"].add_timing(duration)
            self.rollout_start_time = None

